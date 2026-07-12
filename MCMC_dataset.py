import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# comes from the simulation file
from lattice import IsingInstance, random_bimodal_instance, total_energy, simulate_monolithic, _full_local_field

#vars
#equilibrium dataset (dataclass)
@dataclass
class EquilibriumDataset:
    # states are sampled from equilibrium MCMC distribution
    # shapes: up/down are (n_samples and cols)
    #         left/right are (n_samples and rows)
    up: np.ndarray
    down: np.ndarray
    left: np.ndarray
    right: np.ndarray
        # assign the parameters a value
        # specify inverse temp

    beta: float
    burn_in: int

# calculate hamiltonian: calculates the total energy of model
# hamiltonian H = -sum of (J_ij * s_i * s_j)
def calculate_hamiltonian(state: np.ndarray, instance: IsingInstance):
    return total_energy(state, instance)

# given a spin at (i, j, k), the local field is from its 6 neighbors
def local_field(state, instance, i, j, k):
    nx, ny, nz = instance.nx, instance.ny, instance.nz

    # coords for neighbors
    right = (i+1) % nx
    left = (i-1) % nx 
    top = (j+1) % ny
    bottom = (j-1) % ny
    front = (k+1) % nz
    back = (k-1) % nz

    # spin value of its neighbors
    spin_right = state[right, j, k]
    spin_left = state[left, j, k]
    spin_top = state[i, top, k]
    spin_bottom = state[i, bottom, k]
    spin_front = state[i, j, front]
    spin_back = state[i, j, back]

    # coupling constants
    bond_right = instance.bond_x[i, j, k]
    bond_left = instance.bond_x[left, j, k]
    bond_top = instance.bond_y[i, j, k]
    bond_bottom = instance.bond_y[i, bottom, k]
    bond_front = instance.bond_z[i, j, k]
    bond_back = instance.bond_z[i, j, back]

    # field = sum of coupling constants * spin values
    field = (bond_right * spin_right + bond_left * spin_left +
             bond_top * spin_top + bond_bottom * spin_bottom +
             bond_front * spin_front + bond_back * spin_back)
    
    return field


# calculate energy change: calculates energy change if you flip 
# an arbitrary spin. u need this for accept/reject step in mh
def calculate_energy_change():


# calls calculate energy_change, if this is negative accept
# if positive, accept with probability exp(-delta_E/T)
# equation given from Boltzmann distrib.
def metropolis_step():

# the 1000s of sweeps adr said we should do
def run_mcmc():

# plot the steps vs. energy graph
    # deal w/ burn - in
def plot_energy_vs_steps():

def find_burn_in():

def collect_samples():

def generate_dataset():