import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from IsingOnCake.py import IsingInstance, random_bimodal_instance

#vars
#equilibrium dataset (dataclass)
@dataclass
class EquilibriumDataset:
    # states are sampled from equilibrium MCMC distribution
    # shapes: up/down are (n_samples and cols)
    #         left/right are (n_samples and rows)

# calculate hamiltonian
def calculate_hamiltonian(state):
#hamiltonian H = -sum of (J_ij * s_i * s_j)

# calculate energy change
def calculate_energy_change():

# mh

# plot the steps vs. energy graph
    # burn - in