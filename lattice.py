from dataclasses import dataclass

import numpy as np

# info for each different "chip" in our simulation
@dataclass(frozen=True)
class PartitionSpec:
    blocks_x: int
    blocks_y: int
    blocks_z: int

@dataclass(frozen=True) #class for each partition's basic information
class Partition:
    partition_id: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    z_start: int
    z_end: int
    even_mask: np.ndarray   
    odd_mask: np.ndarray    

    @property
    def shape(self) -> tuple[int, int, int]:  # return (nx, ny, nz) of this partition
        return (
            self.x_end - self.x_start,
            self.y_end - self.y_start,
            self.z_end - self.z_start,
        )


# IMPORTANT: the actual 3D Edwards-Anderson instance
@dataclass(frozen=True)
class IsingInstance:
    bond_x: np.ndarray   # couplings along x, shape (nx-1, ny, nz)
    bond_y: np.ndarray   # couplings along y, shape (nx, ny-1, nz)
    bond_z: np.ndarray   # couplings along z, shape (nx, ny, nz-1)
    fields: np.ndarray   # maybe not necessary

    @property
    def nx(self) -> int: #returns nx
        return int(self.fields.shape[0])

    @property
    def ny(self) -> int: #returns ny
        return int(self.fields.shape[1])

    @property
    def nz(self) -> int: #returns nz
        return int(self.fields.shape[2])


@dataclass  # basically stores all results
class SimulationResult:
    energies: np.ndarray                        # energy after every step
    final_state: np.ndarray                     # final spin grid
    history: np.ndarray | None = None           # history
    communication_interval: int | None = None   # only set for partitioned runs

    @property
    def best_energy(self) -> float:  # lowest energy seen over the run
        return float(np.min(self.energies))


# boundary spins seen from neighbouring chips, one face per cardinal direction in 3D
@dataclass(frozen=True)
class GhostBoundary:
    x_lo: np.ndarray | None
    x_hi: np.ndarray | None
    y_lo: np.ndarray | None
    y_hi: np.ndarray | None
    z_lo: np.ndarray | None
    z_hi: np.ndarray | None

def _ghost_boundary_to_float(ghost: GhostBoundary) -> GhostBoundary:
    def cast(face):
        return None if face is None else face.astype(np.float32)
    return GhostBoundary(
        x_lo=cast(ghost.x_lo), x_hi=cast(ghost.x_hi),
        y_lo=cast(ghost.y_lo), y_hi=cast(ghost.y_hi),
        z_lo=cast(ghost.z_lo), z_hi=cast(ghost.z_hi),
    )


def _decay_ghost_boundary(ghost: GhostBoundary, decay: float) -> GhostBoundary:
    def scale(face):
        return None if face is None else (decay * face).astype(np.float32)
    return GhostBoundary(
        x_lo=scale(ghost.x_lo), x_hi=scale(ghost.x_hi),
        y_lo=scale(ghost.y_lo), y_hi=scale(ghost.y_hi),
        z_lo=scale(ghost.z_lo), z_hi=scale(ghost.z_hi),
    )

def random_bimodal_instance(
    nx: int,
    ny: int,
    nz: int,
    rng: np.random.Generator,
    field_scale: float = 0.0,
):
    # creates random ising problem
    bond_x = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(nx - 1, ny, nz))
    bond_y = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(nx, ny - 1, nz))
    bond_z = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(nx, ny, nz - 1))

    if field_scale > 0.0:
        fields = rng.normal(loc=0.0, scale=field_scale, size=(nx, ny, nz)).astype(np.float32)
    else:
        fields = np.zeros((nx, ny, nz), dtype=np.float32)  # pure EA: zero field

    return IsingInstance(
        bond_x=bond_x.astype(np.float32),
        bond_y=bond_y.astype(np.float32),
        bond_z=bond_z.astype(np.float32),
        fields=fields,
    )


def random_spin_state(nx: int, ny: int, nz: int, rng: np.random.Generator):
    # random grid of +/-1 spins
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(nx, ny, nz)).astype(np.int8)


def build_partitions(nx: int, ny: int, nz: int, spec: PartitionSpec):
    if nx % spec.blocks_x or ny % spec.blocks_y or nz % spec.blocks_z:
        raise ValueError("how")

    bx = nx // spec.blocks_x  # block size along each axis
    by = ny // spec.blocks_y
    bz = nz // spec.blocks_z

    partitions: list[Partition] = []
    partition_id = 0
    for ix in range(spec.blocks_x):
        for iy in range(spec.blocks_y):
            for iz in range(spec.blocks_z):
                x0, x1 = ix * bx, ix * bx + bx
                y0, y1 = iy * by, iy * by + by
                z0, z1 = iz * bz, iz * bz + bz

                xi = np.arange(x0, x1)[:, None, None]
                yi = np.arange(y0, y1)[None, :, None]
                zi = np.arange(z0, z1)[None, None, :]

                # 3D checkerboard
                even_mask = ((xi + yi + zi) % 2) == 0
                odd_mask = ~even_mask

                partitions.append(
                    Partition(
                        partition_id=partition_id,
                        x_start=x0, x_end=x1,
                        y_start=y0, y_end=y1,
                        z_start=z0, z_end=z1,
                        even_mask=even_mask,
                        odd_mask=odd_mask,
                    )
                )
                partition_id += 1
    return partitions


def total_energy(state: np.ndarray, instance: IsingInstance):
    #Sums the energy
    x_term = np.sum(instance.bond_x * state[:-1, :, :] * state[1:, :, :])
    y_term = np.sum(instance.bond_y * state[:, :-1, :] * state[:, 1:, :])
    z_term = np.sum(instance.bond_z * state[:, :, :-1] * state[:, :, 1:])
    field_term = np.sum(instance.fields * state)
    return float(-(x_term + y_term + z_term + field_term))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _full_local_field(state: np.ndarray, instance: IsingInstance) -> np.ndarray:
    lf = instance.fields.astype(np.float32).copy()
    # x-axis neighbours
    lf[1:, :, :]  += instance.bond_x * state[:-1, :, :]   
    lf[:-1, :, :] += instance.bond_x * state[1:, :, :]    
    # y-axis neighbours
    lf[:, 1:, :]  += instance.bond_y * state[:, :-1, :]
    lf[:, :-1, :] += instance.bond_y * state[:, 1:, :]
    # z-axis neighbours
    lf[:, :, 1:]  += instance.bond_z * state[:, :, :-1]
    lf[:, :, :-1] += instance.bond_z * state[:, :, 1:]
    return lf


def _partition_local_field(
    state: np.ndarray,
    instance: IsingInstance,
    partition: Partition,
    ghost: GhostBoundary,
) -> np.ndarray:
    # same idea as _full_local_field but only for one block;
    x0, x1 = partition.x_start, partition.x_end
    y0, y1 = partition.y_start, partition.y_end
    z0, z1 = partition.z_start, partition.z_end

    local_state = state[x0:x1, y0:y1, z0:z1]                       # this block's spins
    local_field = instance.fields[x0:x1, y0:y1, z0:z1].astype(np.float32).copy()

    # influence of neighbors 
    if x1 - x0 > 1:
        bxs = instance.bond_x[x0:x1 - 1, y0:y1, z0:z1]
        local_field[1:, :, :]  += bxs * local_state[:-1, :, :]
        local_field[:-1, :, :] += bxs * local_state[1:, :, :]
    if y1 - y0 > 1:
        bys = instance.bond_y[x0:x1, y0:y1 - 1, z0:z1]
        local_field[:, 1:, :]  += bys * local_state[:, :-1, :]
        local_field[:, :-1, :] += bys * local_state[:, 1:, :]
    if z1 - z0 > 1:
        bzs = instance.bond_z[x0:x1, y0:y1, z0:z1 - 1]
        local_field[:, :, 1:]  += bzs * local_state[:, :, :-1]
        local_field[:, :, :-1] += bzs * local_state[:, :, 1:]

    # accounts for neighboring paritions in the x,y,z directions
    if x0 > 0 and ghost.x_lo is not None:
        local_field[0, :, :]  += instance.bond_x[x0 - 1, y0:y1, z0:z1] * ghost.x_lo
    if x1 < instance.nx and ghost.x_hi is not None:
        local_field[-1, :, :] += instance.bond_x[x1 - 1, y0:y1, z0:z1] * ghost.x_hi
    if y0 > 0 and ghost.y_lo is not None:
        local_field[:, 0, :]  += instance.bond_y[x0:x1, y0 - 1, z0:z1] * ghost.y_lo
    if y1 < instance.ny and ghost.y_hi is not None:
        local_field[:, -1, :] += instance.bond_y[x0:x1, y1 - 1, z0:z1] * ghost.y_hi
    if z0 > 0 and ghost.z_lo is not None:
        local_field[:, :, 0]  += instance.bond_z[x0:x1, y0:y1, z0 - 1] * ghost.z_lo
    if z1 < instance.nz and ghost.z_hi is not None:
        local_field[:, :, -1] += instance.bond_z[x0:x1, y0:y1, z1 - 1] * ghost.z_hi

    return local_field


def _update_sites(
    state_slice: np.ndarray,
    local_field: np.ndarray,
    beta: float,
    mask: np.ndarray,
    rng: np.random.Generator,
):
    # calculates probabilities with sigmoid
    probabilities = _sigmoid(2.0 * beta * local_field[mask])
    draws = rng.random(probabilities.shape[0])
    state_slice[mask] = np.where(draws < probabilities, 1, -1).astype(np.int8)



def _snapshot_ghosts(state: np.ndarray, partitions: list[Partition]):
    # captures full boundary information for each partition for a given state
    ghosts: dict[int, GhostBoundary] = {}
    nx, ny, nz = state.shape

    for p in partitions:
        x0, x1 = p.x_start, p.x_end
        y0, y1 = p.y_start, p.y_end
        z0, z1 = p.z_start, p.z_end
        ghosts[p.partition_id] = GhostBoundary(
            x_lo=state[x0 - 1, y0:y1, z0:z1].copy() if x0 > 0 else None,
            x_hi=state[x1,     y0:y1, z0:z1].copy() if x1 < nx else None,
            y_lo=state[x0:x1, y0 - 1, z0:z1].copy() if y0 > 0 else None,
            y_hi=state[x0:x1, y1,     z0:z1].copy() if y1 < ny else None,
            z_lo=state[x0:x1, y0:y1, z0 - 1].copy() if z0 > 0 else None,
            z_hi=state[x0:x1, y0:y1, z1    ].copy() if z1 < nz else None,
        )
    return ghosts


def simulate_monolithic(
    instance: IsingInstance,
    beta: float,
    steps: int,
    seed: int,
    initial_state: np.ndarray | None = None,
    record_history: bool = False,
):
    rng = np.random.default_rng(seed)
    if initial_state is None:
        state = random_spin_state(instance.nx, instance.ny, instance.nz, rng)
    else:
        state = initial_state.astype(np.int8).copy()

    x_idx = np.arange(instance.nx)[:, None, None]
    y_idx = np.arange(instance.ny)[None, :, None]
    z_idx = np.arange(instance.nz)[None, None, :]
    even_mask = ((x_idx + y_idx + z_idx) % 2) == 0
    odd_mask = ~even_mask

    energies = np.empty(steps, dtype=np.float32)
    history = (
        np.empty((steps, instance.nx, instance.ny, instance.nz), dtype=np.int8)
        if record_history else None
    )

    for step in range(steps):
        lf = _full_local_field(state, instance)
        _update_sites(state, lf, beta, even_mask, rng)   
        lf = _full_local_field(state, instance)
        _update_sites(state, lf, beta, odd_mask, rng)  
        energies[step] = total_energy(state, instance)
        if history is not None:
            history[step] = state

    return SimulationResult(energies=energies, final_state=state.copy(), history=history)

def _simulate_partitioned(
    instance: IsingInstance,
    beta: float,
    steps: int,
    seed: int,
    partition_spec: PartitionSpec,
    communication_interval: int,
    belief_decay: float | None = None,
    initial_state: np.ndarray | None = None,
    record_history: bool = False,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    if initial_state is None:
        state = random_spin_state(instance.nx, instance.ny, instance.nz, rng)
    else:
        state = initial_state.astype(np.int8).copy()

    partitions = build_partitions(instance.nx, instance.ny, instance.nz, partition_spec)
    energies = np.empty(steps, dtype=np.float32)
    history = (
        np.empty((steps, instance.nx, instance.ny, instance.nz), dtype=np.int8)
        if record_history else None
    )

    ghosts = {
        pid: _ghost_boundary_to_float(g)
        for pid, g in _snapshot_ghosts(state, partitions).items()
    }

    for step in range(steps):
        if step % communication_interval == 0:
            ghosts = {
                pid: _ghost_boundary_to_float(g)
                for pid, g in _snapshot_ghosts(state, partitions).items()
            }
        elif belief_decay is not None:
            ghosts = {
                pid: _decay_ghost_boundary(g, belief_decay)
                for pid, g in ghosts.items()
            }

        # even half-sweep
        read_state = state.copy()
        write_state = state.copy()

        for p in partitions:
            x0, x1 = p.x_start, p.x_end
            y0, y1 = p.y_start, p.y_end
            z0, z1 = p.z_start, p.z_end

            local_state = write_state[x0:x1, y0:y1, z0:z1]

            lf = _partition_local_field(
                read_state,
                instance,
                p,
                ghosts[p.partition_id],
            )

            _update_sites(
                local_state,
                lf,
                beta,
                p.even_mask,
                rng,
            )

        state = write_state

        # odd half-sweep
        read_state = state.copy()
        write_state = state.copy()

        for p in partitions:
            x0, x1 = p.x_start, p.x_end
            y0, y1 = p.y_start, p.y_end
            z0, z1 = p.z_start, p.z_end

            local_state = write_state[x0:x1, y0:y1, z0:z1]

            lf = _partition_local_field(
                read_state,
                instance,
                p,
                ghosts[p.partition_id],
            )

            _update_sites(
                local_state,
                lf,
                beta,
                p.odd_mask,
                rng,
            )

        state = write_state
        energies[step] = total_energy(state, instance)
        if history is not None:
            history[step] = state

    return SimulationResult(
        energies=energies,
        final_state=state.copy(),
        history=history,
        communication_interval=communication_interval,
    )

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
    # ghost values held fixed between communication 
    return _simulate_partitioned(
        instance, beta, steps, seed, partition_spec, communication_interval,
        belief_decay=None, initial_state=initial_state, record_history=record_history,
    )


def simulate_partitioned_belief(
    instance: IsingInstance,
    beta: float,
    steps: int,
    seed: int,
    partition_spec: PartitionSpec,
    communication_interval: int,
    belief_decay: float = 0.9,
    initial_state: np.ndarray | None = None,
    record_history: bool = False,
):
    # ghost values decayed toward zero between communication
    return _simulate_partitioned(
        instance, beta, steps, seed, partition_spec, communication_interval,
        belief_decay=belief_decay, initial_state=initial_state, record_history=record_history,
    )
