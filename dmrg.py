from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.sparse.linalg import eigsh
import time
import numpy as np


def ZipperLeft(Tl: np.ndarray, Mb: np.ndarray, O: np.ndarray, Mt: np.ndarray) -> np.ndarray:
    Taux = np.einsum('ijk,klm', Mb, Tl,optimize=True)
    Taux = np.einsum('ijkl,kjmn', Taux, O,optimize=True)
    Tf = np.einsum('ijkl,jlm', Taux, Mt,optimize=True)
    return Tf

def ZipperRight(Tr: np.ndarray, Mb: np.ndarray, O: np.ndarray, Mt: np.ndarray) -> np.ndarray:
    Taux = np.einsum('ijk,klm', Mt, Tr,optimize=True)
    Taux = np.einsum('ijkl,mnkj', Taux, O,optimize=True)
    Tf = np.einsum('ijkl,jlm', Taux, Mb,optimize=True)
    return Tf

def LeftC(M: list) -> list:

    N = len(M)  # numero de sitios

    for l in range(N):
        Taux = M[l]
        Taux = np.reshape(Taux, (np.shape(Taux)[0] * np.shape(Taux)[1], np.shape(Taux)[2]))
        U, S, Vdag = np.linalg.svd(Taux, full_matrices=False)
        M[l] = np.reshape(U, (np.shape(M[l])[0], np.shape(M[l])[1], np.shape(U)[1]))
        SVdag = np.matmul(np.diag(S), Vdag)
        if l < N - 1:
            M[l + 1] = np.einsum('ij,jkl', SVdag, M[l + 1],optimize=True)
    return M

def RightC(M: list) -> list:

    N = len(M)  # numero de sitios

    for l in range(N - 1, -1, -1):
        Taux = M[l]
        Taux = np.reshape(Taux, (np.shape(Taux)[0], np.shape(Taux)[1] * np.shape(Taux)[2]))
        U, S, Vdag = np.linalg.svd(Taux, full_matrices=False)
        M[l] = np.reshape(Vdag, (np.shape(Vdag)[0], np.shape(M[l])[1], np.shape(M[l])[2]))
        US = np.matmul(U, np.diag(S))
        if l > 0:
            M[l - 1] = np.einsum('ijk,kl', M[l - 1], US,optimize=True)
    return M

def fDMRG_conv_mem(H: list, D: int, Nsweeps: int, mps : list=None,epsilon: float = 1e-1 ) -> tuple:
    N = len(H)  # numero de sitios
    if mps == None:
        M = [np.random.rand(1, np.shape(H[0])[3], D)]
        for l in range(1, N - 1):
            M.append(np.random.rand(D, np.shape(H[l])[3], D))
        M.append(np.random.rand(D, np.shape(H[N - 1])[3], 1))
    else:
        M = mps
    inicio = time.time()
    M = LeftC(M)
    M = RightC(M)

    Hzip = [None] * (N + 2)
    one = np.ones((1,1,1))
    Hzip[0] = one
    Hzip[-1] = one
    
    for l in range(N-1, -1, -1):
        Hzip[l+1] = ZipperRight(Hzip[l+2], M[l].conj().T, H[l], M[l])

    E_time = []
    E_list = []
    
    for itsweeps in range(Nsweeps):

        # Bucle hacia adelante
        for l in range(N):
            Taux = np.einsum('ijk,jlmn', Hzip[l], H[l], optimize=True)
            Taux = np.einsum('ijklm,nlo', Taux, Hzip[l + 2], optimize=True)
            Taux = np.transpose(Taux, (0, 2, 5, 1, 3, 4))
            Hmat = np.reshape(Taux, (np.shape(Taux)[0] * np.shape(Taux)[1] * np.shape(Taux)[2],
                                     np.shape(Taux)[3] * np.shape(Taux)[4] * np.shape(Taux)[5]))

            val, vec = eigsh(Hmat, k=1, which='SA', v0=M[l])

            E_list.append(val[0])
            E_time.append((time.time() - inicio, val[0]))#AQUI ALMACENO TIEMPO,ENEGIA

            # Actualizacion de MPS
            Taux2 = np.reshape(vec, (np.shape(Taux)[0] * np.shape(Taux)[1], np.shape(Taux)[2]))
            U, S, Vdag = np.linalg.svd(Taux2, full_matrices=False)
            M[l] = np.reshape(U, (np.shape(Taux)[0], np.shape(Taux)[1], np.shape(U)[1]))
            if l < N - 1:
                M[l + 1] = np.einsum('ij,jkl', np.matmul(np.diag(S), Vdag), M[l + 1])

            Hzip[l + 1] = ZipperLeft(Hzip[l], M[l].conj().T, H[l], M[l])

        # Bucle hacia atras
        #inicio = time.time()
        for l in range(N - 1, -1, -1):
            Taux = np.einsum('ijk,jlmn', Hzip[l], H[l], optimize=True)
            Taux = np.einsum('ijklm,nlo', Taux, Hzip[l + 2], optimize=True)
            Taux = np.transpose(Taux, (0, 2, 5, 1, 3, 4))
            Hmat = np.reshape(Taux, (np.shape(Taux)[0] * np.shape(Taux)[1] * np.shape(Taux)[2],
                                     np.shape(Taux)[3] * np.shape(Taux)[4] * np.shape(Taux)[5]))

            val, vec = eigsh(Hmat, k=1, which='SA', v0=M[l])

            Taux2 = np.reshape(vec, (np.shape(Taux)[0], np.shape(Taux)[1] * np.shape(Taux)[2]))
            U, S, Vdag = np.linalg.svd(Taux2, full_matrices=False)
            M[l] = np.reshape(Vdag, (np.shape(Vdag)[0], np.shape(Taux)[1], np.shape(Taux)[2]))
            if l > 0:
                M[l - 1] = np.einsum('ijk,kl', M[l - 1], np.matmul(U, np.diag(S)), optimize=True)

            Hzip[l + 1] = ZipperRight(Hzip[l + 2], M[l].conj().T, H[l], M[l])

            E_list.append(val[0])
            E_time.append((time.time() - inicio, val[0]))#AQUI ALMACENO TIEMPO,ENEGIA


    return E_list, M,E_time




@dataclass
class DmrgOneSite:
    
    H: List[np.ndarray]                   # MPO como lista de tensores (wL,d,wR,d)
    D: int                                # bond dimension deseada
    Nsweeps: int                          # num de barridos
    mps0: Optional[List[np.ndarray]] = None  # MPS inicial (opcional)
    epsilon: float = 1e-3                 # criterio de parada
    seed: Optional[int] = None            # semilla 

    # Salidas / métricas
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

        # Llamada directa a tu implementación (sin cambios)
        E_list, M, E_time = fDMRG_conv_mem(
            H=self.H,
            D=self.D,
            Nsweeps=self.Nsweeps,
            mps=mps_input,
            epsilon=self.epsilon,
        )

        # Guardamos resultados y algunas métricas
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
