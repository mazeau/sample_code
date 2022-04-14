"""
This script runs 15 simulations (each corresponding to a different starting
ratio) in Cantera.
Reactor conditions are replicated from: "Methane catalytic partial oxidation on
autothermal Rh and Pt foam catalysts: Oxidation and reforming zones, transport
effects,and approach to thermodynamic equilibrium"
Horn 2007, doi:10.1016/j.jcat.2007.05.011
Ref 17: "Syngas by catalytic partial oxidation of methane on rhodium:
Mechanistic conclusions from spatially resolved measurements and numerical
simulations"
Horn 2006, doi:10.1016/j.jcat.2006.05.008
Ref 18: "Spatial and temporal profiles in millisecond partial oxidation
processes"
Horn 2006, doi:10.1007/s10562-006-0117-8
"""
from joblib import Parallel, delayed
import multiprocessing
from functools import partial

import cantera as ct
import numpy as np
import argparse
import os

# unit conversion factors to SI
mm = 0.001
cm = 0.01
ms = mm
minute = 60.0

#######################################################################
# Input Parameters
#######################################################################
t_in = 700  # K - in the paper, it was ~698.15K at the start of the cat surface and ~373.15 for the gas inlet temp
t_cat = t_in
length = 70 * mm  # Reactor length - m
diam = 16.5 * mm  # Reactor diameter - in m, from Ref 17 & Ref 18
area = (diam / 2.0) ** 2 * np.pi  # Reactor cross section area (area of tube) in m^2
porosity = 0.81  # Monolith channel porosity, from Ref 17, sec 2.2.2
cat_area_per_vol = 1.6e4  # m2/m3, which is 160 cm2/cm3, as used in Horn 2006
flow_rate = 4.7  # slpm, as seen in as seen in Horn 2007
tot_flow = 0.208  # constant inlet flow rate in mol/min, equivalent to 4.7 slpm
flow_rate = flow_rate * 0.001 / 60  # m^3/s, as seen in as seen in Horn 2007
velocity = flow_rate / area  # m/s

# The PFR will be simulated by a chain of 'N_reactors' stirred reactors.
N_reactors = 7001

on_catalyst = 1000  # catalyst length 10mm, from Ref 17
off_catalyst = 2000
dt = 1.0

reactor_len = length / (N_reactors - 1)
rvol = area * reactor_len * porosity

# catalyst area in one reactor
cat_area = cat_area_per_vol * rvol


ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]

def monolith_simulation(path_to_cti, temp, mol_in, verbose=False, sens=False):
    """
    Set up and solve the monolith reactor simulation.
    Verbose prints out values as you go along
    Sens is for sensitivity, in the form [perturbation, reaction #]
    Args:
        path_to_cti: full path to the cti file
        temp (float): The temperature in Kelvin
        mol_in (3-tuple or iterable): the inlet molar ratios of (CH4, O2, Ar)
        verbose (Boolean): whether to print intermediate results
        sens (False or 2-tuple/list): if not False, then should be a 2-tuple or list [dk, rxn]
                in which dk = relative change (eg. 0.01) and rxn = the index of the surface reaction rate to change
    Returns:
        gas_out, # gas molar flow rate in moles/minute
        surf_out, # surface mole fractions
        gas_names, # gas species names
        surf_names, # surface species names
        dist_array, # distances (in mm)
        T_array # temperatures (in K)
    """
    gas = ct.Solution(path_to_cti, "gas")
    i_ar = gas.species_index("Ar")
    ch4, o2, ar = mol_in
    X = f"CH4:{ch4}, O2:{o2}, Ar:{ar}"
    gas.TPX = temp, ct.one_atm, X  # need to initialize mass flow rate at STP
    surf = ct.Interface(path_to_cti, "Pt_surf", [gas])
    surf.TP = t_cat, ct.one_atm
    n_surf_reactions = surf.n_reactions
    print(
        f"Running monolith simulation with CH4 and O2 concs {mol_in[0], mol_in[1]} on thread {os.getpid()}"
    )
    mass_flow_rate = flow_rate * gas.density_mass

    if verbose:
        print(
            "  distance(mm)   X_CH4        X_O2        X_H2       X_CO       X_H2O       X_CO2"
        )

    # create a new reactor
    r = ct.IdealGasReactor(gas, energy="off")
    r.volume = rvol

    # create a reservoir to represent the reactor immediately upstream. Note
    # that the gas object is set already to the state of the upstream reactor
    upstream = ct.Reservoir(gas, name="upstream")

    # create a reservoir for the reactor to exhaust into. The composition of
    # this reservoir is irrelevant.
    downstream = ct.Reservoir(gas, name="downstream")

    # Add the reacting surface to the reactor. The area is set to the desired
    # catalyst area in the reactor.
    rsurf = ct.ReactorSurface(surf, r, A=cat_area)

    # The mass flow rate into the reactor will be fixed by using a
    # MassFlowController object.
    m = ct.MassFlowController(upstream, r, mdot=mass_flow_rate)

    # We need an outlet to the downstream reservoir. This will determine the
    # pressure in the reactor. The value of K will only affect the transient
    # pressure difference.
    v = ct.PressureController(r, downstream, master=m, K=1e-5)
    sim = ct.ReactorNet([r])
    sim.max_err_test_fails = 12

    # set relative and absolute tolerances on the simulation
    sim.rtol = 1.0e-9
    sim.atol = 1.0e-21

    gas_names = gas.species_names
    surf_names = surf.species_names
    gas_out = []
    surf_out = []
    dist_array = []
    T_array = []

   # surf.set_multiplier(0.0)  # no surface reactions until the gauze
    for n in range(N_reactors):
        # Set the state of the reservoir to match that of the previous reactor
        gas.TDY = r.thermo.TDY
        upstream.syncState()
        sim.reinitialize()
        sim.advance_to_steady_state()
        dist = n * reactor_len * 1.0e3  # distance in mm
        dist_array.append(dist)
        T_array.append(surf.T)
        kmole_flow_rate = mass_flow_rate / gas.mean_molecular_weight  # kmol/s
        gas_out.append(
            1000 * 60 * kmole_flow_rate * gas.X.copy()
        )  # molar flow rate in moles/minute
        surf_out.append(surf.X.copy())

        # stop simulation when things are done changing, to avoid getting so many COVDES errors
        if n >= 1001:
            if np.max(abs(np.subtract(gas_out[-2], gas_out[-1]))) < 1e-15:
                break

        if verbose:
            if not n % 100:
                print(
                    "  {0:10f}  {1:10f}  {2:10f}  {3:10f} {4:10f} {5:10f} {6:10f}".format(
                        dist,
                        *gas["CH4(2)", "O2(3)", "H2(6)", "CO(7)", "H2O(5)", "CO2(4)"].X
                        * 1000
                        * 60
                        * kmole_flow_rate,
                    )
                )

    gas_out = np.array(gas_out)
    surf_out = np.array(surf_out)
    gas_names = np.array(gas_names)
    surf_names = np.array(surf_names)
    data_out = (
        gas_out,
        surf_out,
        gas_names,
        surf_names,
        dist_array,
        T_array,
        i_ar,
        n_surf_reactions,
    )
    print(
        f"Finished monolith simulation for CH4 and O2 concs {mol_in[0], mol_in[1]} on thread {os.getpid()}"
    )
    return data_out


def run_one_simulation(path_to_cti, ratio):
    """
    Start all of the simulations all at once using multiprocessing
    """
    fo2 = 1 / (2.0 * ratio + 1 + 79.0 / 21.0)
    fch4 = 2 * fo2 * ratio
    far = 79 * fo2 / 21
    ratio_in = [fch4, fo2, far]  # mol fractions

    sim = monolith_simulation(path_to_cti, t_in, ratio_in)
    print("Finished simulation at a C/O ratio of {:.1f}".format(ratio))
    (
        gas_out,
        surf_out,
        gas_names,
        surf_names,
        dist_array,
        T_array,
        i_ar,
        n_surf_reactions,
    ) = sim
    return [ratio, [gas_out, gas_names, dist_array, T_array, n_surf_reactions]]

def main(cantera_file):
    num_cpus = multiprocessing.cpu_count() -2
    results = Parallel(n_jobs=num_cpus, verbose=10,temp_folder="/tmp")(delayed(partial(run_one_simulation, cantera_file))(ratio) for ratio in ratios)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        dest="cantera_file",
        help="Path to the Cantera file",
    )
    args = parser.parse_args()
    cantera_file = args.cantera_file

    main(cantera_file=cantera_file)