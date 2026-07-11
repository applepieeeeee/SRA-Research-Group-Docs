import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# comes from the simulation file
from lattice import IsingInstance, random_bimodal_instance, total_energy

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
def calculate_hamiltonian(state: np.ndarray, instance: IsingInstance):
    return total_energy(state, instance)

#hamiltonian H = -sum of (J_ij * s_i * s_j)

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