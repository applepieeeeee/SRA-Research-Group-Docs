import numpy as np
from lattice import simulate_monolithic, simulate_partitioned,PartitionSpec, GhostBoundary, random_bimodal_instance, random_spin_state

#these are all placeholders, need to replace with Clair's configuration from dataset generation
NX = NY = NZ = 12 
BETA = 1.0 #honestly have no idea what to make this
STEPS = 1000
SEED = 42

rng = np.random.default_rng(SEED)


instance = random_bimodal_instance(NX, NY, NZ, rng)
initial_state = random_spin_state(NX, NY, NZ, rng)
#equilibrium_data = np.load("placeholder")




#this is the function that we will use in simulate_partitioned
#currently uses a frozen baseline
def equilibrium_ghost_update(step, ghost_age, ghosts, state, instance, partitions, beta, rng):
    #TODO: implement logic for the ghost update
    return ghosts


result = simulate_partitioned(instance=instance, beta=BETA, steps=STEPS, seed=SEED, partition_spec=PartitionSpec(2, 2, 2), communication_interval=10, initial_state=initial_state, ghost_update_fn=equilibrium_ghost_update)
print("Best energy:", result.best_energy)
print("Final energy:", result.energies[-1])

monolithic_result = simulate_monolithic(instance=instance, beta=BETA, steps=STEPS, seed=SEED, initial_state=initial_state)

print()
print("Best energy:", monolithic_result.best_energy)
print("Final energy:", monolithic_result.energies[-1])