from dataclasses import dataclass
from collections.abc import Callable
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
    bond_x: np.ndarray   # couplings along x, shape (nx, ny, nz)
    bond_y: np.ndarray   # couplings along y, shape (nx, ny, nz)
    bond_z: np.ndarray   # couplings along z, shape (nx, ny, nz)
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

def random_bimodal_instance(
    nx: int,
    ny: int,
    nz: int,
    rng: np.random.Generator,
    field_scale: float = 0.0,
):
    # creates random ising problem
    bond_x = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(nx, ny, nz))
    bond_y = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(nx, ny, nz))
    bond_z = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(nx, ny, nz))

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
    x_term = np.sum(instance.bond_x * state * np.roll(state, -1, axis = 0))
    y_term = np.sum(instance.bond_y * state * np.roll(state, -1, axis = 1))
    z_term = np.sum(instance.bond_z * state * np.roll(state, -1, axis = 2))
    field_term = np.sum(instance.fields * state)
    return float(-(x_term + y_term + z_term + field_term))

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _full_local_field(state: np.ndarray, instance: IsingInstance) -> np.ndarray:
    lf = instance.fields.astype(np.float32).copy()
    ax = instance.bond_x * state
    lf += np.roll(ax, 1, axis=0)
    lf += instance.bond_x * np.roll(state, -1, axis=0)
    ay = instance.bond_y * state
    lf += np.roll(ay, 1, axis=1)
    lf += instance.bond_y * np.roll(state, -1, axis=1)
    az = instance.bond_z * state
    lf += np.roll(az, 1, axis=2)
    lf += instance.bond_z * np.roll(state, -1, axis=2)
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
    nx, ny, nz = instance.nx, instance.ny, instance.nz
    bx = instance.bond_x[x0:x1, y0:y1, z0:z1]
    by = instance.bond_y[x0:x1, y0:y1, z0:z1]
    bz = instance.bond_z[x0:x1, y0:y1, z0:z1]

    local_state = state[x0:x1, y0:y1, z0:z1]                       # this block's spins
    lf = instance.fields[x0:x1, y0:y1, z0:z1].astype(np.float32).copy()

    if x1 - x0 == nx:
        ax = bx * local_state
        lf += np.roll(ax, 1, axis=0)
        lf += bx * np.roll(local_state, -1, axis=0)
    else:
        lf[:-1, :, :] += bx[:-1, :, :] * local_state[1:, :, :]   # +x interior
        lf[1:, :, :]  += bx[:-1, :, :] * local_state[:-1, :, :]  # -x interior
        if ghost.x_lo is not None:
            lf[0, :, :]  += instance.bond_x[(x0 - 1) % nx, y0:y1, z0:z1] * ghost.x_lo
        if ghost.x_hi is not None:
            lf[-1, :, :] += instance.bond_x[x1 - 1, y0:y1, z0:z1] * ghost.x_hi

    if y1 - y0 == ny:
        ay = by * local_state
        lf += np.roll(ay, 1, axis=1)
        lf += by * np.roll(local_state, -1, axis=1)
    else:
        lf[:, :-1, :] += by[:, :-1, :] * local_state[:, 1:, :]   # +y interior
        lf[:, 1:, :]  += by[:, :-1, :] * local_state[:, :-1, :]  # -y interior
        if ghost.y_lo is not None:
            lf[:, 0, :]  += instance.bond_y[x0:x1, (y0 - 1) % ny, z0:z1] * ghost.y_lo
        if ghost.y_hi is not None:
            lf[:, -1, :] += instance.bond_y[x0:x1, y1 - 1, z0:z1] * ghost.y_hi

    if z1 - z0 == nz:
        az = bz * local_state
        lf += np.roll(az, 1, axis=2)
        lf += bz * np.roll(local_state, -1, axis=2)
    else:
        lf[:, :, :-1] += bz[:, :, :-1] * local_state[:, :, 1:]   # +z interior
        lf[:, :, 1:]  += bz[:, :, :-1] * local_state[:, :, :-1]  # -z interior
        if ghost.z_lo is not None:
            lf[:, :, 0]  += instance.bond_z[x0:x1, y0:y1, (z0 - 1) % nz] * ghost.z_lo
        if ghost.z_hi is not None:
            lf[:, :, -1] += instance.bond_z[x0:x1, y0:y1, z1 - 1] * ghost.z_hi
    return lf

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
    ghosts: dict[int, GhostBoundary] = {}
    nx, ny, nz = state.shape

    for p in partitions:
        x0, x1 = p.x_start, p.x_end
        y0, y1 = p.y_start, p.y_end
        z0, z1 = p.z_start, p.z_end
        x_full = (x1 - x0 == nx)
        y_full = (y1 - y0 == ny)
        z_full = (z1 - z0 == nz)
        ghosts[p.partition_id] = GhostBoundary(
            x_lo=None if x_full else state[(x0 - 1) % nx, y0:y1, z0:z1].copy(),
            x_hi=None if x_full else state[x1 % nx, y0:y1, z0:z1].copy(),
            y_lo=None if y_full else state[x0:x1, (y0 - 1) % ny, z0:z1].copy(),
            y_hi=None if y_full else state[x0:x1, y1 % ny, z0:z1].copy(),
            z_lo=None if z_full else state[x0:x1, y0:y1, (z0 - 1) % nz].copy(),
            z_hi=None if z_full else state[x0:x1, y0:y1, z1 % nz].copy(),
        )
    return ghosts


def simulate_monolithic(instance: IsingInstance,beta: float, steps: int,seed:int, initial_state: np.ndarray, record_history: bool = False, record_every: int =1, on_step: Callable[[int, np.ndarray, float], None] | None = None):
    rng = np.random.default_rng(seed)
    state = initial_state.astype(np.int8).copy()

    #x y and z
    x = np.arange(instance.nx)[:,None,None]
    y = np.arange(instance.ny)[None,:,None]
    z = np.arange(instance.nz)[None, None, :]

    #stilll have to do the even odd thing
    even_mask = ((x + y + z) % 2) == 0
    odd_mask = ~even_mask
    energies = np.empty(steps, dtype=np.float32)
    saved_states = [] if record_history else None

    for step in range(steps):
        lf = _full_local_field(state, instance)
        _update_sites(state, lf, beta, even_mask, rng)

        lf = _full_local_field(state, instance)
        _update_sites(state, lf, beta, odd_mask, rng)

        energy = total_energy(state, instance)
        energies[step] = energy

        if saved_states is not None and step % record_every == 0:
            saved_states.append(state.copy())

        #In theory should let another script look at the state each step. First time using the callable thing
        if on_step is not None:
            on_step(step, state, energy)

    history = None
    if saved_states:
        history = np.stack(saved_states)

    return SimulationResult(
        energies=energies,
        final_state=state.copy(),
        history=history,
    )

def simulate_partitioned(instance: IsingInstance, beta: float, steps: int, seed: int, partition_spec: PartitionSpec, communication_interval: int,initial_state: np.ndarray | None = None,
    ghost_update_fn: Callable[
        [
            int,
            int,
            dict[int, GhostBoundary],
            np.ndarray,
            IsingInstance,
            list[Partition],
            float,
            np.random.Generator,
        ],
        dict[int, GhostBoundary],
    ] | None = None,

    record_history: bool = False, record_every: int = 1, on_step: Callable[[int, np.ndarray, float], None] | None = None):
    rng = np.random.default_rng(seed)

    if initial_state is None:
        state = random_spin_state(
            instance.nx, instance.ny, instance.nz, rng
        )
    else:
        state = initial_state.astype(np.int8).copy()

    partitions = build_partitions(
        instance.nx,
        instance.ny,
        instance.nz,
        partition_spec,
    )

    energies = np.empty(steps, dtype=np.float32)
    saved_states = [] if record_history else None

    ghosts = {
        pid: _ghost_boundary_to_float(ghost)
        for pid, ghost in _snapshot_ghosts(state, partitions).items()
    }

    for step in range(steps):
        ghost_age = step % communication_interval

        if ghost_age == 0:
            # actual communication between partitions
            ghosts = {
                pid: _ghost_boundary_to_float(ghost)
                for pid, ghost in _snapshot_ghosts(
                    state, partitions
                ).items()
            }

        elif ghost_update_fn is not None:

            # another script can decide what to do with stale ghosts
            ghosts = ghost_update_fn(
                step,
                ghost_age,
                ghosts,
                state,
                instance,
                partitions,
                beta,
                rng,
            )

        # if no ghost function was given, the ghosts stay frozen

        # even half sweep
        read_state = state.copy()
        write_state = state.copy()

        for p in partitions:

            x0, x1 = p.x_start, p.x_end
            y0, y1 = p.y_start, p.y_end
            z0, z1 = p.z_start, p.z_end


            local_state = write_state[
                x0:x1,
                y0:y1,
                z0:z1,
            ]

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

        # odd half sweep
        read_state = state.copy()
        write_state = state.copy()

        for p in partitions:

            x0, x1 = p.x_start, p.x_end
            y0, y1 = p.y_start, p.y_end
            z0, z1 = p.z_start, p.z_end

            local_state = write_state[
                x0:x1,
                y0:y1,
                z0:z1,
            ]

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

        energy = total_energy(state, instance)
        energies[step] = energy

        if saved_states is not None and step % record_every == 0:
            saved_states.append(state.copy())

        if on_step is not None:
            on_step(step, state, energy)

    if saved_states:
        history = np.stack(saved_states)
    else:
        history = None

    return SimulationResult(
        energies=energies,
        final_state=state.copy(),
        history=history,
        communication_interval=communication_interval,
    )