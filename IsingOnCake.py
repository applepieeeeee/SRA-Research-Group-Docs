from dataclasses import dataclass

import numpy as np

# info for each different "chip" in our simulation
@dataclass(frozen=True)
class PartitionSpec:
    rows: int
    cols: int


@dataclass(frozen=True) #class for each partition's basic information
class Partition:
    partition_id: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    even_mask: np.ndarray
    odd_mask: np.ndarray

    @property
    def shape(self) -> tuple[int, int]: #returns the number of rows by number of columns in each partition 
        return (self.row_end -self.row_start, self.col_end- self.col_start)
    



#IMPORTANT: actual instance template of ising model
@dataclass(frozen=True)
class IsingInstance:
    horizontal: np.ndarray #holds all the horizontal correlations
    vertical: np.ndarray # same thing but vertical



    fields: np.ndarray  #in hindsight maybe I shouldn't have added this since our simulation should be 0 field
                        #but now finding the gridsize depends on this so probably dont need to get rid of it for the time being

    @property
    def rows(self) -> int: #returns number of rows
        return int(self.fields.shape[0])

    @property
    def cols(self) -> int: #returns number of columns 
        return int(self.fields.shape[1])

@dataclass #basically stores all resuls
class SimulationResult:
    energies: np.ndarray #energy level after simulation
    final_state: np.ndarray #final spin grid
    history: np.ndarray | None = None #history of spin grid (don't reaally need it but added just in case we want it later)
    communication_interval: int | None = None #only for partitioned system.
    @property
    def best_energy(self) -> float: #returns the best distribution found over the course of the entire eimulation
        return float(np.min(self.energies))


#stores info from adjacent "chips"from all cardinal directions
#for when communication happens in the distributed system
#each array holds a strip of boundary values
@dataclass(frozen=True)
class GhostBoundary:
    up: np.ndarray | None
    down: np.ndarray |None
    left: np.ndarray |None
    right: np.ndarray | None


def _ghost_boundary_to_float(ghost: GhostBoundary) -> GhostBoundary:
    return GhostBoundary(
        up=None if ghost.up is None else ghost.up.astype(np.float32),
        down=None if ghost.down is None else ghost.down.astype(np.float32),
        left=None if ghost.left is None else ghost.left.astype(np.float32),
        right=None if ghost.right is None else ghost.right.astype(np.float32),
    )


def _decay_ghost_boundary(ghost: GhostBoundary, decay: float) -> GhostBoundary:
    return GhostBoundary(
        up=None if ghost.up is None else (decay * ghost.up).astype(np.float32),
        down=None if ghost.down is None else (decay * ghost.down).astype(np.float32),
        left=None if ghost.left is None else (decay * ghost.left).astype(np.float32),
        right=None if ghost.right is None else (decay * ghost.right).astype(np.float32),
    )


def random_bimodal_instance(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    field_scale: float = 0.0,
):
    #creates random ising problem
    # should return a fully packed Ising instance

    horizontal = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(rows, cols - 1)) #random horizontal connections
    vertical = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(rows - 1, cols))   #random vertical connection

    if field_scale > 0.0:
        fields = rng.normal(loc=0.0, scale=field_scale, size=(rows, cols)).astype(np.float32)
    else:
        fields = np.zeros((rows, cols), dtype=np.float32) #each field is set to 0

    return IsingInstance(horizontal=horizontal.astype(np.float32), vertical=vertical.astype(np.float32), fields=fields)
    #return finished Ising model
    #in thoery now it should be ready for simulation








def random_spin_state(rows: int, cols: int, rng: np.random.Generator): #creates random grid of spins; either -1 or 1
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(rows, cols)).astype(np.int8)



#splits up the full grid into smaller partitions to simulate out networking thing
def build_partitions(rows: int, cols: int, spec: PartitionSpec):


    if rows % spec.rows != 0 or cols % spec.cols != 0:
        raise ValueError("If this error is happening. you are just really stupid")

    block_height = rows // spec.rows #floor division to convert to integer
    block_width = cols // spec.cols
    partitions: list[Partition] = [] #list of all partitions


    #just loops and calculates the row and column numbers for each partition
    partition_id = 0
    for block_row in range(spec.rows):
        for block_col in range(spec.cols):
            row_start = block_row * block_height
            row_end = row_start + block_height
            col_start = block_col * block_width
            col_end = col_start + block_width

            row_idx = np.arange(row_start, row_end)[:, None]
            col_idx = np.arange(col_start, col_end)[None, :]

            #since we are implementing a 2d Ising model to allow concurrent spins to happen we update in a checkerboard pattern
            #this allows you to update half the "chips" with their neighbors remaining unchanging so they can sample from them parralely
            #even in odd will alternate updating and remaining constant
            even_mask = ((row_idx + col_idx) % 2) == 0
            odd_mask = ~even_mask

            partitions.append(
                Partition(
                    partition_id=partition_id,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    even_mask=even_mask,
                    odd_mask=odd_mask,
                )
            )#creates list of partitions
            partition_id += 1
    return partitions



def total_energy(state: np.ndarray, instance: IsingInstance): #computes total energy of current spin grid
    #calculates contribution horizontaly and vertically
    """Compute the total Ising energy of a spin configuration.

    Sums the contribution of every horizontal bond, every vertical bond,
    and (if used) every external field, then negates the total since
    matching/aligned spins should lower the energy. Lower energy = better.
    """
    horizontal_term = np.sum(instance.horizontal * state[:, :-1] * state[:, 1:]) 
    vertical_term = np.sum(instance.vertical * state[:-1, :] * state[1:, :])
    field_term = np.sum(instance.fields * state)
    return float(-(horizontal_term + vertical_term + field_term))


def _sigmoid(x: np.ndarray) -> np.ndarray: #calculates sigmoid element by element for an array; which is why it returns an array
    return 1.0 / (1.0 + np.exp(-x))


def _full_local_field(state: np.ndarray, instance: IsingInstance) -> np.ndarray:
    local_field = instance.fields.astype(np.float32).copy()
    local_field[:, 1:] += instance.horizontal * state[:, :-1]
    local_field[:, :-1] += instance.horizontal * state[:, 1:]
    local_field[1:, :] += instance.vertical * state[:-1, :]
    local_field[:-1, :] += instance.vertical * state[1:, :]
    return local_field
#calculates the local field

#essentially does the same thing as the previous dunction but for only one partition
#for some reason this is actually a lot more code; GPT and I couldn't think of a better way to do this so had to just go through everything manually
def _partition_local_field(state: np.ndarray,instance: IsingInstance,partition: Partition,ghost: GhostBoundary):
    r0, r1 = partition.row_start, partition.row_end
    c0, c1 = partition.col_start, partition.col_end
    #row and column range for the partition


    local_state = state[r0:r1, c0:c1] #actual part of the spin grid belongign to this partition
    local_field = instance.fields[r0:r1, c0:c1].astype(np.float32).copy()

    if c1 - c0 > 1: #just a sanity check to verify that it has neighbors
        local_field[:, 1:] += instance.horizontal[r0:r1, c0 : c1 - 1] * local_state[:, :-1]
        local_field[:, :-1] += instance.horizontal[r0:r1, c0 : c1 - 1] * local_state[:, 1:]
        #if it does have neighbors adds their influence to the local field 

    if r1 - r0 > 1: #same thing as previous if statement but for vertical neighbors
        local_field[1:, :] += instance.vertical[r0 : r1 - 1, c0:c1] * local_state[:-1, :]
        local_field[:-1, :] += instance.vertical[r0 : r1 - 1, c0:c1] * local_state[1:, :]


    #accounts for neighboring partitions in all cardinal directions
    if c0 > 0 and ghost.left is not None:
        local_field[:, 0] += instance.horizontal[r0:r1, c0 - 1] * ghost.left
    if c1 < instance.cols and ghost.right is not None:
        local_field[:, -1] += instance.horizontal[r0:r1, c1 - 1] * ghost.right
    if r0 > 0 and ghost.up is not None:
        local_field[0, :] += instance.vertical[r0 - 1, c0:c1] * ghost.up
    if r1 < instance.rows and ghost.down is not None:
        local_field[-1, :] += instance.vertical[r1 - 1, c0:c1] * ghost.down

    #returns a 2d numpy array of local fields for this partition
    return local_field


def _update_sites(state_slice: np.ndarray, local_field: np.ndarray, beta: float,mask: np.ndarray, rng: np.random.Generator):
    probabilities = _sigmoid(2.0 * beta * local_field[mask])
    draws = rng.random(probabilities.shape[0])
    state_slice[mask] = np.where(draws < probabilities, 1, -1).astype(np.int8)
    #calculates probabilities for spin sites that we are updating
    #see comments in build partitions function


#basically just captures full boundary information for each partition for a given state
def _snapshot_ghosts(state: np.ndarray, partitions: list[Partition]):
    ghosts: dict[int, GhostBoundary] = {}
    rows, cols = state.shape

    for partition in partitions:
        r0, r1 = partition.row_start, partition.row_end
        c0, c1 = partition.col_start, partition.col_end
        ghosts[partition.partition_id] = GhostBoundary(
            up=state[r0 - 1, c0:c1].copy() if r0 > 0 else None,
            down=state[r1, c0:c1].copy() if r1 < rows else None,
            left=state[r0:r1, c0 - 1].copy() if c0 > 0 else None,
            right=state[r0:r1, c1].copy() if c1 < cols else None,
        )
    return ghosts





def simulate_monolithic(instance: IsingInstance, beta: float, steps: int, seed: int, initial_state: np.ndarray | None = None, record_history: bool = False):
    rng = np.random.default_rng(seed)
    if initial_state is None:
        state = random_spin_state(instance.rows, instance.cols, rng)
    else:
        state = initial_state.astype(np.int8).copy()

    row_idx = np.arange(instance.rows)[:, None]
    col_idx = np.arange(instance.cols)[None, :]
    even_mask = ((row_idx + col_idx) % 2) == 0
    odd_mask = ~even_mask

    energies = np.empty(steps, dtype=np.float32)
    history = np.empty((steps, instance.rows, instance.cols), dtype=np.int8) if record_history else None

    for step in range(steps):
        local_field = _full_local_field(state, instance)
        _update_sites(state, local_field, beta, even_mask, rng)
        local_field = _full_local_field(state, instance)
        _update_sites(state, local_field, beta, odd_mask, rng)
        energies[step] = total_energy(state, instance)
        if history is not None:
            history[step] = state

    return SimulationResult(energies=energies, final_state=state.copy(), history=history)

#WARNING: Claude
def simulate_partitioned_frozen(
    instance: IsingInstance,
    beta: float,
    steps: int,
    seed: int,
    partition_spec: PartitionSpec,
    communication_interval: int,
    initial_state: np.ndarray | None = None,
    record_history: bool = False,
) -> SimulationResult:
    """Distributed simulation where ghost boundaries are frozen between
    communication rounds (never decayed). Thin wrapper around
    simulate_partitioned_belief for backward compatibility."""
    return simulate_partitioned_belief(
        instance=instance,
        beta=beta,
        steps=steps,
        seed=seed,
        partition_spec=partition_spec,
        communication_interval=communication_interval,
        belief_mode="frozen",
        initial_state=initial_state,
        record_history=record_history,
    )

def simulate_partitioned_belief(instance: IsingInstance,
    beta: float, steps: int, seed: int, partition_spec: PartitionSpec, communication_interval: int, belief_mode: str = "decay_to_zero", belief_decay: float = 0.9,
    initial_state: np.ndarray | None = None,
    record_history: bool = False,):

    if communication_interval < 1:
        raise ValueError("communication_interval must be at least 1.")
    if belief_mode not in {"frozen", "decay_to_zero"}:
        raise ValueError("belief_mode must be one of: frozen, decay_to_zero")
    if not 0.0 <= belief_decay <= 1.0:
        raise ValueError("belief_decay must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    if initial_state is None:
        state = random_spin_state(instance.rows, instance.cols, rng)
    else:
        state = initial_state.astype(np.int8).copy()

    partitions = build_partitions(instance.rows, instance.cols, partition_spec)
    energies = np.empty(steps, dtype=np.float32)
    history = np.empty((steps, instance.rows, instance.cols), dtype=np.int8) if record_history else None
    ghosts = {partition_id: _ghost_boundary_to_float(ghost) for partition_id, ghost in _snapshot_ghosts(state, partitions).items()}

    for step in range(steps):
        if step % communication_interval == 0:
            ghosts = {
                partition_id: _ghost_boundary_to_float(ghost)
                for partition_id, ghost in _snapshot_ghosts(state, partitions).items()
            }
        elif belief_mode == "decay_to_zero":
            ghosts = {partition_id: _decay_ghost_boundary(ghost, belief_decay) for partition_id, ghost in ghosts.items()}

        for partition in partitions:
            r0, r1 = partition.row_start, partition.row_end
            c0, c1 = partition.col_start, partition.col_end
            local_state = state[r0:r1, c0:c1]

            local_field = _partition_local_field(state, instance, partition, ghosts[partition.partition_id])
            _update_sites(local_state, local_field, beta, partition.even_mask, rng)
            local_field = _partition_local_field(state, instance, partition, ghosts[partition.partition_id])
            _update_sites(local_state, local_field, beta, partition.odd_mask, rng)

        energies[step] = total_energy(state, instance)
        if history is not None:
            history[step] = state

    return SimulationResult(
        energies=energies,
        final_state=state.copy(),
        history=history,
        communication_interval=communication_interval,
    )

def check_partitioned_matches_monolithic(rows=8, cols=8, steps=200, beta=0.5, seed=0):
    rng = np.random.default_rng(seed)
    instance = random_bimodal_instance(rows, cols, rng)
    initial_state = random_spin_state(rows, cols, rng)

    mono = simulate_monolithic(
        instance, beta=beta, steps=steps, seed=seed, initial_state=initial_state
    )
    part = simulate_partitioned_belief(
        instance, beta=beta, steps=steps, seed=seed,
        partition_spec=PartitionSpec(rows=2, cols=2),
        communication_interval=1,
        belief_mode="frozen",
        initial_state=initial_state,
    )

    print("monolithic best energy:", mono.best_energy)
    print("partitioned (interval=1) best energy:", part.best_energy)
    # These won't be bit-identical (RNG draw order differs) but should be
    # in the same ballpark over many steps/seeds if the physics is consistent.
