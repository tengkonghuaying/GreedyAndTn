import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU

import tensorcircuit as tc
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import numpy as np
from jax.example_libraries import optimizers
import tensorflow as tf
import math

tc.set_backend('jax')
K = tc.backend

# user specified

import sys

# 计算l列表中第一个元素的长度，即nqubits
nqubits = 8
pauli_matrices = {
    'I': np.array([[1, 0], [0, 1]]),
    'X': np.array([[0, 1], [1, 0]]),
    'Y': np.array([[0, -1j], [1j, 0]]),
    'Z': np.array([[1, 0], [0, -1]])
}

def tensor_to_2d(tensor):
    # 获取张量的维度
    shape = tensor.shape
    num_dims = len(shape)

    # 分割前后维度
    half = num_dims // 2

    # 合并前半部分和后半部分的维度
    new_shape = (np.prod(shape[:half]), np.prod(shape[half:]))

    # 重塑张量为二维矩阵
    reshaped_tensor = tensor.reshape(new_shape)

    return reshaped_tensor
# 使用NumPy加载.npy文件
init_obs = np.load(f"{nqubits}-K-Local-H.npy")
init_obs = tensor_to_2d(init_obs)

num_steps_range = range(5)
num_layers_range = [2]


def put_vqc(c, param, nlayers):
    for j in range(nlayers):
        if j != 0:
            for i in range(j % 2, nqubits, 2):
                c.unitary(i, (i + 1) % nqubits,
                          unitary=np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]), name="iswap")
        for i in range(nqubits):
            c.rx(i, theta=param[j, i, 0])
        for i in range(nqubits):
            c.ry(i, theta=param[j, i, 1])
        for i in range(nqubits):
            c.rx(i, theta=param[j, i, 2])
    return c


def put_vqc_inv(c, param, nlayers):
    for j in reversed(range(nlayers)):
        for i in reversed(range(nqubits)):
            c.rx(i, theta=-param[j, i, 2])
        for i in reversed(range(nqubits)):
            c.ry(i, theta=-param[j, i, 1])
        for i in reversed(range(nqubits)):
            c.rx(i, theta=-param[j, i, 0])
        if j != 0:
            for i in reversed(range(j % 2, nqubits, 2)):
                c.unitary(i, (i + 1) % nqubits,
                          unitary=np.array([[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]]),
                          name="iswap-inv")
    return c


def transfer(param, obs, nlayers):
    input_state = obs
    dmc = tc.DMCircuit(nqubits, dminputs=input_state)
    put_vqc(dmc, param, nlayers)
    return dmc.state()


def transfer_inv(param, obs, nlayers):
    input_state = obs
    dmc = tc.DMCircuit(nqubits, dminputs=input_state)
    put_vqc_inv(dmc, param, nlayers)
    return dmc.state()


def off_diag(obs):
    dim = obs.shape[0]
    return obs * (1 - jnp.eye(dim))


def loss(param, obs, nlayers):
    off_diag_mat = off_diag(transfer(param, obs, nlayers))
    return jnp.real(jnp.trace(off_diag_mat @ off_diag_mat))


init_scale = 0.1
#maxiter = 5000

loss_vag = K.jit(
    K.value_and_grad(loss), static_argnums=(2,)
)

rho = np.load(f"sparse_rho_{nqubits}.npy")
diags = np.zeros((2, 2 ** nqubits), dtype=complex)
total_trace = 0


def reorder_qubits(x, nqubits):
    return np.transpose(x.reshape([2] * nqubits)).reshape(-1)


obs = init_obs
print(f"cal:{np.real(np.trace(rho @ init_obs))}")

num_steps = 5
nlayers = 2

def getParam():
    param_k = []
    diags = np.zeros((num_steps, 2 ** nqubits), dtype=complex)
    for i in num_steps_range:
        diags[i] = np.load(f'saved_models/{num_steps}iter_{nlayers}layers_diag_{nqubits}_{i}.npy')
        param = np.load(f'saved_models/{num_steps}iter_{nlayers}layers_param_{nqubits}_{i}.npy')
        param_k.append(param)
    return param_k, diags


def reorder_qubits(x, nqubits):
    return np.transpose(x.reshape([2] * nqubits)).reshape(-1)


nruns = 1


def estimate_expectation_value(rho, H, T, L):
    # 贪婪投影分解
    # U_k, Lambda_k = greedy_projected_decomposition(H, num_qubits, L)
    param_k, Lambda_k = getParam()

    pk = calculate_pk(Lambda_k)
    # 按照采样概率分配测量次数
    nshots_iter = [math.floor(T * p) for p in pk]
    print("nshots_iter:", nshots_iter)
    # 存储每次实验的期望值贡献
    estval_list_run = np.zeros(nruns, dtype=complex)

    # 遍历所有电路
    for i in range(len(param_k)):
        # 如果该电路的测量次数大于 0
        if nshots_iter[i] > 0:
            for j in range(nruns):
                # 在选定的电路上进行量子态转换和测量
                bt = apply_Uk_and_measure(param_k[i], rho, nqubits, nshots_iter[i])
                # 根据对角矩阵 Lambda_k 计算期望值贡献
                diags_flip = reorder_qubits(Lambda_k[i], nqubits)
                contribution = estimate_diag(diags_flip, bt, nshots_iter[i])
                estval_list_run[j] += contribution

    return np.median(estval_list_run)


def estimate_diag(diag_obs, bit_string_data, nshots):
    x = np.array([[n if (j == int(bs, 2)) else 0 for j in range(len(diag_obs))] for bs, n in (bit_string_data.items())])
    x = np.sum(x, axis=0)
    return np.sum(x * diag_obs) / nshots


def calculate_pk(Lambda_k):
    importance = np.max(np.abs(Lambda_k), axis=1)
    importance /= np.sum(importance)
    return importance


def apply_Uk_and_measure(param_k, rho, num_qubits, shots):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import DensityMatrix, Operator
    dm = DensityMatrix(rho)
    iswap_matrix = np.array([[1, 0, 0, 0],
                             [0, 0, 1j, 0],
                             [0, 1j, 0, 0],
                             [0, 0, 0, 1]])
    iswap_gate = UnitaryGate(iswap_matrix, label="iswap")
    qc = QuantumCircuit(num_qubits)
    for j in reversed(range(nlayers)):
        for i in reversed(range(num_qubits)):
            qc.rx(param_k[j, i, 2], i)

        for i in reversed(range(num_qubits)):
            qc.ry(param_k[j, i, 1], i)

        for i in reversed(range(num_qubits)):
            qc.rx(param_k[j, i, 0], i)

        if j != 0:
            for i in reversed(range(j % 2, num_qubits, 2)):
                qc.append(iswap_gate, [(i + 1) % num_qubits,i])
    dm2 = dm.evolve(qc)
    counts = dm2.sample_counts(shots)
    return counts


T = 54000
result = []
for i in range(50):
    estimated_expectation_value = estimate_expectation_value(rho, init_obs, T, num_layers_range[0])
    result.append(estimated_expectation_value)
# 计算偏差
cal = np.real(np.trace(rho @ init_obs))
squared_deviations = [(x - cal) ** 2 for x in result]
# 计算标准差
std_dev = np.sqrt(np.mean(squared_deviations))
print("std:", std_dev)
print(f"cal:{np.real(np.trace(rho @ init_obs))}")
percent = (np.real(np.trace(rho @ init_obs)) - tf.math.real(estimated_expectation_value)) / np.real(np.trace(rho @ init_obs))
