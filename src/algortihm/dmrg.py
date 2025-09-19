from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse.linalg import eigsh

import time


Tensor = np.ndarray


def _svd_reshape(
    tensor: Tensor,
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Perform an SVD while respecting the original tensor shapes.

    The helper keeps all reshape operations in a single place, making the
    dimensional juggling in the algorithm easier to read and review.
    """

    reshaped = tensor.reshape(np.prod(left_shape), np.prod(right_shape))
    u, s, v_dag = np.linalg.svd(reshaped, full_matrices=False)
    return u, s, v_dag


def ZipperLeft(Tl: Tensor, Mb: Tensor, O: Tensor, Mt: Tensor) -> Tensor:
    """Contract the environment from the left."""

    temp = np.einsum("ijk,klm", Mb, Tl, optimize=True)
    temp = np.einsum("ijkl,kjmn", temp, O, optimize=True)
    return np.einsum("ijkl,jlm", temp, Mt, optimize=True)


def ZipperRight(Tr: Tensor, Mb: Tensor, O: Tensor, Mt: Tensor) -> Tensor:
    """Contract the environment from the right."""

    temp = np.einsum("ijk,klm", Mt, Tr, optimize=True)
    temp = np.einsum("ijkl,mnkj", temp, O, optimize=True)
    return np.einsum("ijkl,jlm", temp, Mb, optimize=True)


def LeftC(mps: List[Tensor]) -> List[Tensor]:
    """Left-canonicalise the MPS tensors."""

    num_sites = len(mps)

    for site in range(num_sites):
        u, s, v_dag = _svd_reshape(
            mps[site],
            left_shape=(np.shape(mps[site])[0], np.shape(mps[site])[1]),
            right_shape=(np.shape(mps[site])[2],),
        )
        mps[site] = u.reshape(
            np.shape(mps[site])[0],
            np.shape(mps[site])[1],
            np.shape(u)[1],
        )
        sv_dag = np.matmul(np.diag(s), v_dag)
        if site < num_sites - 1:
            mps[site + 1] = np.einsum("ij,jkl", sv_dag, mps[site + 1], optimize=True)
    return mps


def RightC(mps: List[Tensor]) -> List[Tensor]:
    """Right-canonicalise the MPS tensors."""

    num_sites = len(mps)

    for site in range(num_sites - 1, -1, -1):
        u, s, v_dag = _svd_reshape(
            mps[site],
            left_shape=(np.shape(mps[site])[0],),
            right_shape=(np.shape(mps[site])[1], np.shape(mps[site])[2]),
        )
        mps[site] = v_dag.reshape(
            np.shape(v_dag)[0],
            np.shape(mps[site])[1],
            np.shape(mps[site])[2],
        )
        us = np.matmul(u, np.diag(s))
        if site > 0:
            mps[site - 1] = np.einsum("ijk,kl", mps[site - 1], us, optimize=True)
    return mps


def _initial_mps(num_sites: int, mpo: Sequence[Tensor], bond_dim: int) -> List[Tensor]:
    """Create a random initial MPS with the desired bond dimension."""

    mps: List[Tensor] = [np.random.rand(1, np.shape(mpo[0])[3], bond_dim)]
    for site in range(1, num_sites - 1):
        mps.append(np.random.rand(bond_dim, np.shape(mpo[site])[3], bond_dim))
    mps.append(np.random.rand(bond_dim, np.shape(mpo[num_sites - 1])[3], 1))
    return mps


def fDMRG_conv_mem(
    H: Sequence[Tensor],
    D: int,
    Nsweeps: int,
    mps: Optional[List[Tensor]] = None,
    epsilon: float = 1e-1,
) -> Tuple[List[float], List[Tensor], List[Tuple[float, float]]]:
    """Run the one-site finite DMRG algorithm.

    Returns the full energy history, the optimised MPS tensors and the
    (time, energy) pairs collected along the optimisation.
    """

    num_sites = len(H)
    tensors = list(mps) if mps is not None else _initial_mps(num_sites, H, D)
    start_time = time.time()
    tensors = RightC(LeftC(tensors))

    env = [None] * (num_sites + 2)
    env[0] = np.ones((1, 1, 1))
    env[-1] = np.ones((1, 1, 1))
    for site in range(num_sites - 1, -1, -1):
        env[site + 1] = ZipperRight(env[site + 2], tensors[site].conj().T, H[site], tensors[site])

    time_energy: List[Tuple[float, float]] = []
    energy_history: List[float] = []

    def _store_energy(energy: float) -> None:
        energy_history.append(energy)
        time_energy.append((time.time() - start_time, energy))

    def _converged() -> bool:
        return len(energy_history) > 1 and abs(energy_history[-1] - energy_history[-2]) < epsilon

    for _ in range(Nsweeps):
        # Forward sweep
        for site in range(num_sites):
            contracted = np.einsum("ijk,jlmn", env[site], H[site], optimize=True)
            contracted = np.einsum("ijklm,nlo", contracted, env[site + 2], optimize=True)
            contracted = np.transpose(contracted, (0, 2, 5, 1, 3, 4))
            hmat = contracted.reshape(
                np.shape(contracted)[0] * np.shape(contracted)[1] * np.shape(contracted)[2],
                np.shape(contracted)[3] * np.shape(contracted)[4] * np.shape(contracted)[5],
            )

            vals, vecs = eigsh(hmat, k=1, which="SA", v0=tensors[site])
            energy = float(vals[0])
            _store_energy(energy)
            if _converged():
                break

            reshaped_vec = vecs.reshape(
                np.shape(contracted)[0] * np.shape(contracted)[1],
                np.shape(contracted)[2],
            )
            u, s, v_dag = np.linalg.svd(reshaped_vec, full_matrices=False)
            tensors[site] = u.reshape(
                np.shape(contracted)[0],
                np.shape(contracted)[1],
                np.shape(u)[1],
            )
            if site < num_sites - 1:
                tensors[site + 1] = np.einsum(
                    "ij,jkl",
                    np.matmul(np.diag(s), v_dag),
                    tensors[site + 1],
                )

            env[site + 1] = ZipperLeft(env[site], tensors[site].conj().T, H[site], tensors[site])

        if _converged():
            break

        # Backward sweep
        for site in range(num_sites - 1, -1, -1):
            contracted = np.einsum("ijk,jlmn", env[site], H[site], optimize=True)
            contracted = np.einsum("ijklm,nlo", contracted, env[site + 2], optimize=True)
            contracted = np.transpose(contracted, (0, 2, 5, 1, 3, 4))
            hmat = contracted.reshape(
                np.shape(contracted)[0] * np.shape(contracted)[1] * np.shape(contracted)[2],
                np.shape(contracted)[3] * np.shape(contracted)[4] * np.shape(contracted)[5],
            )

            vals, vecs = eigsh(hmat, k=1, which="SA", v0=tensors[site])
            energy = float(vals[0])
            _store_energy(energy)
            if _converged():
                break

            reshaped_vec = vecs.reshape(
                np.shape(contracted)[0],
                np.shape(contracted)[1] * np.shape(contracted)[2],
            )
            u, s, v_dag = np.linalg.svd(reshaped_vec, full_matrices=False)
            tensors[site] = v_dag.reshape(
                np.shape(v_dag)[0],
                np.shape(contracted)[1],
                np.shape(contracted)[2],
            )
            if site > 0:
                tensors[site - 1] = np.einsum(
                    "ijk,kl",
                    tensors[site - 1],
                    np.matmul(u, np.diag(s)),
                    optimize=True,
                )

            env[site + 1] = ZipperRight(env[site + 2], tensors[site].conj().T, H[site], tensors[site])

        if _converged():
            break

    return energy_history, tensors, time_energy




@dataclass
class DmrgOneSite:
    
    H: List[np.ndarray]                   # MPO as a list of tensors (wL,d,wR,d)
    D: int                                # desired bond dimension
    Nsweeps: int                          # number of sweeps
    mps0: Optional[List[np.ndarray]] = None  # initial MPS (optional)
    epsilon: float = 1e-3                 # stopping criterion
    seed: Optional[int] = None            # seed

    # Outputs / metrics
    energy_history: List[float] = field(default_factory=list, init=False)
    time_energy: List[Tuple[float, float]] = field(default_factory=list, init=False)
    mps_opt: Optional[List[np.ndarray]] = field(default=None, init=False)
    sweeps_run: int = field(default=0, init=False)

    def _init_mps_if_needed(self) -> Optional[List[np.ndarray]]:
        """Inicializa una MPS random si no se proporciona nada.
        Solo fijamos la semilla ; fDMRG_conv_mem hace la init.
        """
        if self.mps0 is not None:
            return self.mps0
        if self.seed is not None:
            np.random.seed(self.seed)
        return None

    def run(self) -> Tuple[List[float], List[np.ndarray], List[Tuple[float, float]]]:
        """Ejecuta tu fDMRG_conv_mem y expone (energías, mps, (t,E))."""
        mps_input = self._init_mps_if_needed()

        # Direct call to your implementation (unchanged)
        E_list, M, E_time = fDMRG_conv_mem(
            H=self.H,
            D=self.D,
            Nsweeps=self.Nsweeps,
            mps=mps_input,
            epsilon=self.epsilon,
        )

        # Store results and some metrics
        self.energy_history = list(E_list)
        self.time_energy = list(E_time)
        self.mps_opt = M
        self.sweeps_run = self.Nsweeps
        

        return self.energy_history, self.mps_opt, self.time_energy

    # Utilidades de acceso para el usuario
    @property
    def E0(self) -> Optional[float]:
        return self.energy_history[-1] if self.energy_history else None

    @property
    def MPS(self) -> Optional[List[np.ndarray]]:
        return self.mps_opt
