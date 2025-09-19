import numpy as np
from typing import List, Tuple

def calculate_cut_value(adj_matrix: np.ndarray, partition: list[int]) -> int:
    """
    Calcula el valor del corte para una partición dada.

    Parámetros:
    adj_matrix (np.ndarray): Matriz de adyacencia del grafo.
    partition (list[int]): Lista de 0s y 1s que representa la partición de nodos.

    Retorna:
    int: El valor total del corte.
    """
    cut_value = 0
    num_nodes = adj_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if partition[i] != partition[j]:
                cut_value += adj_matrix[i, j]
    return cut_value
def transpose_tensors(array_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Transpone cada array en una lista de arrays de NumPy segun los ejes especificados (0, 2, 3, 1).

    Parametros:
    array_list (List[np.ndarray]): Lista de arrays de entrada.

    Retorna:
    List[np.ndarray]: Lista de arrays transpuestos.
    """
    # Validar que la entrada es una lista de ndarrays
    if not all(isinstance(array, np.ndarray) for array in array_list):
        raise ValueError("Todos los elementos de array_list deben ser instancias de np.ndarray")

    # Transponer cada array en la lista segun los ejes especificados
    transposed_arrays = [np.transpose(array, (0, 3, 1, 2)) for array in array_list]

    return transposed_arrays
def generar_indices(matriz: np.ndarray, inicio: int = 4) -> List[Tuple[int, int]]:
    """
    Genera una lista de tuplas con las posiciones de la matriz a partir de un indice inicial.

    Parametros:
    matriz (np.ndarray): Matriz de referencia.
    inicio (int): indice a partir del cual se generan las posiciones (por defecto es 4).

    Retorno:
    List[Tuple[int, int]]: Lista de tuplas con las posiciones de la matriz.
    """
    filas, columnas = matriz.shape
    indices = []

    for i in range(inicio, filas):
        for j in range(inicio, columnas):
            if i < j:  # Solo tuplas donde el primer indice sea menor que el segundo
                indices.append((i, j))

    return indices

def crear_mps_sol(tensores, bitstring=None):
    """
    Creates a Matrix Product State (MPS) representation based on a bitstring.

    Parameters:
    tensores (int): Number of tensors (sites) in the MPS.
    bitstring (list or None): A binary list representing the state.
                              If None, a random bitstring is generated.

    Returns:
    list: A list of NumPy arrays representing the MPS.
    """
    if bitstring is None:
        bitstring = np.random.randint(0, 2, size=tensores).tolist()

    if len(bitstring) != tensores:
        raise ValueError("Length of bitstring must match the number of tensors.")

    mps = []

    # Create initial tensor
    inicial = np.zeros((1, 2, 2))
    inicial[0, bitstring[0], 0] = 1
    mps.append(inicial)

    # Create middle tensors
    for i in range(1, tensores - 1):
        tensor = np.zeros((2, 2, 2))
        tensor[0, bitstring[i], 0] = 1
        mps.append(tensor)

    # Create final tensor
    final = np.zeros((2, 2, 1))
    final[0, bitstring[-1], 0] = 1
    mps.append(final)

    return mps


def extract_bitstring(mps_tensors: List[np.ndarray], threshold: float = 0.1) -> List[int]:
    """
    Dado un MPS en una lista de arrays de NumPy: [A^(1), A^(2), ..., A^(N)],
    donde cada uno tiene la forma (bondDimLeft, d, bondDimRight),
    encuentra la única cadena de bits (s_1, ..., s_N) que tiene una amplitud distinta de cero.

    Parámetros:
        mps_tensors (List[np.ndarray]): Lista de tensores que representan el MPS.
        threshold (float): Umbral para considerar un valor como distinto de cero.

    Retorna:
        List[int]: Una lista de 0 y 1 que representa la cadena de bits encontrada.
    """
    alpha_left = 0  # Comienza desde el borde izquierdo (suponiendo bondDimLeft=1 en el sitio 1)
    bitstring = []

    for A in mps_tensors:
        bondL, d, bondR = A.shape
        found = False

        # Busca sobre los posibles estados locales s y el siguiente índice del enlace alpha_right
        for s in range(d):
            for alpha_right in range(bondR):
                val = A[alpha_left, s, alpha_right]

                if abs(val) > threshold:
                    bitstring.append(s)
                    alpha_left = alpha_right
                    found = True
                    break
            if found:
                break

    bitstring.insert(0, 0)
    return bitstring

def calcular_constante(matrix):
    # Sumamos todos los elementos de la matriz triangular superior, excluyendo la diagonal
    suma = np.sum(np.triu(matrix, k=1))
    return suma / 2
def crear_mpo(n: int) -> List[np.ndarray]:
    """
    Crea la MPO con todo 0s que es una lista que contiene arrays de numpy .

    Parámetros:
    - n (int): Numero de nodos/qubits o particulas del problema Max-Cut.

    Retorna:
    - List[np.ndarray]: Una lista de arreglos de numpy inicializados en ceros,representa la MPO del problema Max-cut.
    """

    #is_matrix = len(iden.shape) == 2 and iden.shape[0] == iden.shape[1] and len(z.shape) == 2 and z.shape[0] == z.shape[1]
    # Inicializar la lista con valores predefinidos en indices especificos
    lista = np.zeros(n, dtype=int)
    lista[0], lista[1], lista[2], lista[-1] = 1, 2, n, 1
    lista[3:-1] = np.arange(3,len(lista)-1)[::-1]

    # Crear MPO con dimension ajustada segun el tipo de iden y z

    mpo = [np.zeros((lista[i], lista[i + 1], 2,2)) for i in range(n - 1)]

    return mpo

def crear_mps(n: int) -> List[np.ndarray]:
    """
    Crea la "MPS" con todo 0s que es una lista que contiene arrays de numpy .


    Parametros:
    - n (int): Numero de nodos/qubits o particulas del problema Max-Cut.

    Retorna:
    - List[np.ndarray]: Una lista de arreglos de numpy inicializados en ceros,representa la MPO del problema Max-cut.
    """

    #is_matrix = len(iden.shape) == 2 and iden.shape[0] == iden.shape[1] and len(z.shape) == 2 and z.shape[0] == z.shape[1]
    # Inicializar la lista con valores predefinidos en indices especificos
    #TODO elegir si usar vecotorizado o for
    lista = np.zeros(n, dtype=int)
    lista[0], lista[1], lista[2], lista[-1] = 1, 2, n, 1
    lista[3:-1] = np.arange(3,len(lista)-1)[::-1]

    # Crear MPS con dimension ajustada segun el tipo de iden y z

    mpo = [np.zeros((lista[i], lista[i + 1], 2)) for i in range(n - 1)]

    return mpo

def _primer_tensor(lista: List[np.ndarray], iden: np.ndarray, z: np.ndarray) -> List[np.ndarray]:
    """
    Rellenamos el primer tensor de la MPO.

    Parametros:
    lista (List[np.ndarray]): Lista de tensores (arreglos multidimensionales) donde se aplicaran las modificaciones.
    iden (np.ndarray): Vector de unos utilizado para modificar el primer tensor.
    z (np.ndarray): Vector que contiene los valores 1 y -1.

    Retorno:
    List[np.ndarray]: La lista modificada con cambios en el primer tensor.
    """
    tensor = lista[0]
    tensor[0, 0] = iden
    tensor[0, 1] = z
    return lista

def _ultimo_tensor(lista: List[np.ndarray], matriz: np.ndarray, iden: np.ndarray, z: np.ndarray) -> List[np.ndarray]:
    """
    Rellenamos el ultimo tensor de la MPO.

    Parametros:
    lista (List[np.ndarray]): Lista de tensores (arreglos multidimensionales) donde se aplicaran las modificaciones.
    matriz (np.ndarray): Matriz adyacencia.
    iden (np.ndarray): Vector de unos utilizado en las modificaciones.
    z (np.ndarray): Vector que contiene los valores 1 y -1.

    Retorno:
    List[np.ndarray]: La lista modificada con cambios en el ultimo tensor.
    """
    tensor = lista[-1]
    tensor[0, 0] = z * (matriz[0, -1]) / 2
    tensor[1, 0] = z
    tensor[2, 0] = iden
    return lista



def _st_op(lista: List[np.ndarray], matriz: np.ndarray, iden: np.ndarray, z: np.ndarray) -> List[np.ndarray]:
    """
    Optimized version of _st that pre-divides z and iden by 2 to reduce execution time.

    Parametros:
    lista (List[np.ndarray]): Lista de tensores donde se aplicaran las modificaciones.
    matriz (np.ndarray): Matriz adyacencia.
    iden (np.ndarray): Vector de unos utilizado para modificar el segundo tensor.
    z (np.ndarray): Vector que contiene los valores 1 y -1.

    Retorno:
    List[np.ndarray]: La lista modificada con cambios en el segundo tensor.
    """
    limite = len(lista) - 1  # El limite excluye las dos ultimas posiciones reservadas para otros fines
    segundo = lista[1]
    segundo[0, 0] = iden

    # Dividimos para reducir el numero de operaciones ;)
    z_half = z / 2
    iden_half = iden / 2

    # Modificacion del tensor para posiciones intermedias
    for i in range(1, limite):
        indice = i + 2
        segundo[0, i] = z_half * matriz[2, indice]
        segundo[1, i] = iden_half * matriz[1, indice]

    # Ajuste de los valores en las ultimas posiciones del tensor
    segundo[1, -2] = z_half * matriz[1, 2]  # anteultimo tensor
    segundo[1, -1] = iden_half * matriz[0, 1]  # ultimo tensor
    segundo[0, -1] = z_half * matriz[0, 2]  # ultimo tensor

    return lista

def _tercer_op(lista: List[np.ndarray], matriz: np.ndarray, iden: np.ndarray, z: np.ndarray) -> List[np.ndarray]:
    """
    Rellena el tercer tensor de la MPO.

    Parametros:
    lista (List[np.ndarray]): Lista de tensores donde se aplicaran las modificaciones.
    matriz (np.ndarray): Matriz de referencia cuyos valores se utilizan en las modificaciones.
    iden (np.ndarray): Vector de unos utilizado para modificar el tercer tensor.
    z (np.ndarray): Vector que contiene los valores 1 y -1.

    Retorno:
    List[np.ndarray]: La lista modificada con cambios en el tercer tensor.
    """

    tercer = lista[2]
    tercer[0, 0] = iden
    z_medio = z/2
    tercer[0, -1] = z * (matriz[0, 3]) / 2
    rango = tercer.shape[1] - 1  # Excluye la ultima posicion reservada para finalizar

    # Aplicacion de valores de z en posiciones intermedias del tensor
    for i in range(1, rango):
        tercer[0, i] = z_medio * (matriz[3, i + 3])
        tercer[i + 1, i] = iden # Aplica identidades para el resto de posiciones

    # Asigna el valor de z para finalizar la configuracion del tensor
    tercer[1, -1] = z

    tercer[-2, -1] = iden
    tercer[-1, -1] = iden

    return lista



def _rt_opb(lista: List[np.ndarray], matriz: np.ndarray, iden: np.ndarray, z: np.ndarray) -> List[np.ndarray]:
    """
    Rellena el resto de tensores de la MPO hasta el anteultimo.

    Parametros:
    lista (List[np.ndarray]): Lista de tensores donde se aplicaran las modificaciones.
    matriz (np.ndarray): Matriz de referencia cuyos valores se utilizan en las modificaciones.
    iden (np.ndarray): Vector de unos utilizado para modificar cada tensor.
    z (np.ndarray): Vector que contiene los valores 1 y -1.

    Retorno:
    List[np.ndarray]: La lista modificada con cambios en los tensores especificados.
    """
    # Generar las posiciones para la matriz
    posiciones = generar_indices(matriz)
    # Optimizadas operaciones z = z/2 evitamos dividir todo el rato...
    z_medio = z/2
    for x in range(3, len(lista) - 1):
        #print(f"Modificando tensor en la posicion {x}")
        tensor = lista[x]

        # Asignaciones iniciales
        tensor[0, 0] = iden
        tensor[0, -1] = z_medio * (matriz[0, x + 1])
        rango = tensor.shape[1] - 1
        # Modificacion de valores intermedios
        for j in range(1, rango):
            fila , col = posiciones.pop(0)
            tensor[0, j] = z_medio * (matriz[fila, col])
            indice = -(j + 1)
            tensor[indice, indice] = iden

        # Asignaciones finales
        tensor[-1, -1] = iden
        tensor[1, -1] = z  # Propagar solo el valor de z

    return lista



def gen_mps(matriz: np.ndarray) -> List[np.ndarray]:
    """
    Procesa todos los tensores en el MPO si `n` es 9 o mayor, y valida las condiciones de entrada.

    Parametros:
    - n (int): Numero de nodos/qubits o particulas.
    - matriz (np.ndarray): Matriz de referencia para las modificaciones.
    - iden (np.ndarray): Vector de unos utilizado para modificar cada tensor.
    - z (np.ndarray): Vector que contiene los valores 1 y -1.

    Retorna:
    - List[np.ndarray]: Lista actualizada despues de procesar todos los tensores.

    Lanza:
    - ValueError: Si `n` es menor que 9, si `matriz` no es cuadrada, o si las dimensiones de `matriz` no coinciden con `n`.
    """
    n = matriz.shape[0]
    iden = np.ones(2)
    z = np.array([1, -1])
    #iden = np.eye(2)
    #z = np.array([[1, 0],[0,-1]])
    # Validacion de las condiciones de entrada
    """if n < 9:
        raise ValueError("Minimo necesitaras usar 9 nodos para que esto sea efectivo.")"""
    if matriz.shape[0] != matriz.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")
    if matriz.shape[0] != n:
        raise ValueError("La matriz no coincide con el numero de nodos.")

    # Crear la lista MPS y procesar los tensores
    lista = crear_mps(n)
    lista = _primer_tensor(lista, iden, z)
    lista = _st_op(lista, matriz, iden, z)
    lista = _tercer_op(lista, matriz, iden, z)
    lista = _rt_opb(lista, matriz, iden, z)
    lista = _ultimo_tensor(lista, matriz, iden, z)

    return lista


def gen_mpo(matriz: np.ndarray) -> List[np.ndarray]:
    """
    Procesa todos los tensores en el MPO si `n` es 9 o mayor, y valida las condiciones de entrada.

    Parametros:
    - n (int): Numero de nodos/qubits o particulas.
    - matriz (np.ndarray): Matriz de referencia para las modificaciones.
    - iden (np.ndarray): Vector de unos utilizado para modificar cada tensor.
    - z (np.ndarray): Vector que contiene los valores 1 y -1.

    Retorna:
    - List[np.ndarray]: Lista actualizada despues de procesar todos los tensores.

    Lanza:
    - ValueError: Si `n` es menor que 9, si `matriz` no es cuadrada, o si las dimensiones de `matriz` no coinciden con `n`.
    """
    n = matriz.shape[0]
    #iden = np.ones(2)
    #z = np.array([1, -1])
    iden = np.eye(2)
    z = np.array([[1, 0],[0,-1]])
    # Validacion de las condiciones de entrada
    """if n < 9:
        raise ValueError("Minimo necesitaras usar 9 nodos para que esto sea efectivo.")"""
    if matriz.shape[0] != matriz.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")
    if matriz.shape[0] != n:
        raise ValueError("La matriz no coincide con el numero de nodos.")

    # Crear la lista MPO y procesar los tensores
    lista = crear_mpo(n)
    lista = _primer_tensor(lista, iden, z)
    lista = _st_op(lista, matriz, iden, z)
    lista = _tercer_op(lista, matriz, iden, z)
    lista = _rt_opb(lista, matriz, iden, z)
    lista = _ultimo_tensor(lista, matriz, iden, z)

    return lista