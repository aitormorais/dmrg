import numpy as np

from tn.wmc_tn import gen_mpo, transpose_tensors


if __name__ == "__main__":
    matriz_random = np.random.rand(10, 10)
    mpo_tensores = gen_mpo(matriz_random)
    tensores_transpuestos = transpose_tensors(mpo_tensores)

    for indice, tensor in enumerate(tensores_transpuestos, start=1):
        print(f"Tensor {indice}: shape {tensor.shape}")
