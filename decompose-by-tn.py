import argparse
import copy
import time

from scipy.linalg import solve
import numpy as np
import tensornetwork as tn
from scipy.optimize import minimize
import math


# 创建解析器
parser = argparse.ArgumentParser(
    description="Process data files",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--nqubits", type=int, default=8)
parser.add_argument("--num_steps", type=int, default=6)
parser.add_argument("--num_layers", type=int, default=3)

# 解析参数
args = parser.parse_args()


nqubits = 8
num_steps = 3
num_layers = 2


# 使用NumPy加载.npy文件
pauli_matrices = {
    'I': np.array([[1, 0], [0, 1]]),
    'X': np.array([[0, 1], [1, 0]]),
    'Y': np.array([[0, -1j], [1j, 0]]),
    'Z': np.array([[1, 0], [0, -1]])
}
# 定义 iswap 门矩阵
iswap_matrix = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
]).reshape(2, 2, 2, 2)

iswap_inv_matrix = np.array([
    [1, 0, 0, 0],
    [0, 0, -1j, 0],
    [0, -1j, 0, 0],
    [0, 0, 0, 1]
]).reshape(2, 2, 2, 2)

def rx_matrix(theta):
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ])

def ry_matrix(theta):
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
    ])


# 转化为张量网络
def vqc_to_tn(nqubits, param, nlayers):
    """
    构建参数化电路的张量网络表示，第一层量子门的输入边悬空。

    参数:
        nqubits : int - 量子比特数量
        param : ndarray - 参数数组，形状为 (nlayers, nqubits, 3)
        nlayers : int - 电路的层数

    返回:
        nodes : list - 张量网络的所有节点
        final_edges : list - 最后一层输出边
    """
    nodes = []  # 保存所有的张量节点
    layer_edges = [None] * nqubits  # 当前层的输出边
    initial_edges = []

    for j in range(nlayers):
        # 如果不是第一层，添加 iswap 门
        if j != 0:
            for i in range(j % 2, nqubits, 2):
                iswap_node = tn.Node(iswap_matrix)

                up,down,_ = tn.split_node(iswap_node, [iswap_node[0], iswap_node[2]], [iswap_node[1], iswap_node[3]],max_singular_values=4)
                up.name = f"{j}_iswap-up"
                down.name = f"{j}_iswap-down"
                up[0].name = f"{j}_iswap-up_0"
                up[1].name = f"{j}_iswap-up_1"
                down[1].name = f"{j}_iswap-down_0"
                down[2].name = f"{j}_iswap-down_1"

                # 连接上一层的边（如果是第一层，则无需连接输入边）
                if layer_edges[i] is not None:
                    layer_edges[i] ^ up[0]
                if layer_edges[(i + 1) % nqubits] is not None:
                    layer_edges[(i + 1) % nqubits] ^ down[1]

                # 更新输出边
                layer_edges[i] = up[1]
                layer_edges[(i + 1) % nqubits] = down[2]
                down.reorder_edges([down[1], down[2], down[0]])
                nodes.append(up)
                nodes.append(down)

        # 添加单比特 Rx 门
        for i in range(nqubits):
            rx_node = tn.Node(rx_matrix(param[j, i, 0]))
            rx_node.name = f"{j}_{i}_rx1"
            rx_node[0].name = f"{j}_{i}_rx1_0"
            rx_node[1].name = f"{j}_{i}_rx1_1"
            if layer_edges[i] is not None:
                layer_edges[i] ^ rx_node[0]  # 连接上一层输出边
            layer_edges[i] = rx_node[1]  # 更新当前层输出边
            if j == 0:
                initial_edges.append(rx_node[0])
            nodes.append(rx_node)

        # 添加单比特 Ry 门
        for i in range(nqubits):
            ry_node = tn.Node(ry_matrix(param[j, i, 1]))
            ry_node.name = f"{j}_{i}_ry1"
            ry_node[0].name = f"{j}_{i}_ry1_0"
            ry_node[1].name = f"{j}_{i}_ry1_1"
            layer_edges[i] ^ ry_node[0]  # 连接 Rx 的输出边
            layer_edges[i] = ry_node[1]  # 更新当前层输出边
            nodes.append(ry_node)

        # 添加单比特 Rx 门
        for i in range(nqubits):
            rx_node = tn.Node(rx_matrix(param[j, i, 2]))
            rx_node.name = f"{j}_{i}_rx2"
            rx_node[0].name = f"{j}_{i}_rx2_0"
            rx_node[1].name = f"{j}_{i}_rx2_1"
            layer_edges[i] ^ rx_node[0]  # 连接 Ry 的输出边
            layer_edges[i] = rx_node[1]  # 更新当前层输出边
            nodes.append(rx_node)

    return nodes


# 构建反向张量网络
def vqc_to_tn_inv(nqubits, param, nlayers):
    """
    构建反向参数化电路的张量网络表示，第一层量子门的输入边悬空。

    参数:
        nqubits : int - 量子比特数量
        param : ndarray - 参数数组，形状为 (nlayers, nqubits, 3)
        nlayers : int - 电路的层数

    返回:
        nodes : list - 张量网络的所有节点
        initial_edges : list - 第一层输入悬空边集合
    """
    nodes = []  # 保存所有的张量节点
    layer_edges = [None] * nqubits  # 当前层的输出边
    initial_edges = []

    for j in reversed(range(nlayers)):
        # 添加单比特 Rx 门（逆序应用）
        for i in range(nqubits):
            rx_node = tn.Node(rx_matrix(-param[j, i, 2]))
            rx_node.name = f"{j}_{i}_rx2"
            rx_node[0].name = f"{j}_{i}_rx2_0"
            rx_node[1].name = f"{j}_{i}_rx2_1"
            if layer_edges[i] is not None:
                layer_edges[i] ^ rx_node[0]  # 连接上一层的输出边
            layer_edges[i] = rx_node[1]  # 更新当前层输出边
            if j == nlayers - 1:
                initial_edges.append(rx_node[0])
            nodes.append(rx_node)

        # 添加单比特 Ry 门（逆序应用）
        for i in range(nqubits):
            ry_node = tn.Node(ry_matrix(-param[j, i, 1]))
            ry_node.name = f"{j}_{i}_ry1"
            ry_node[0].name = f"{j}_{i}_ry1_0"
            ry_node[1].name = f"{j}_{i}_ry1_1"
            layer_edges[i] ^ ry_node[0]  # 连接上一层的输出边
            layer_edges[i] = ry_node[1]  # 更新当前层输出边
            nodes.append(ry_node)

        # 添加单比特 Rx 门（逆序应用）
        for i in range(nqubits):
            rx_node = tn.Node(rx_matrix(-param[j, i, 0]))
            rx_node.name = f"{j}_{i}_rx1"
            rx_node[0].name = f"{j}_{i}_rx1_0"
            rx_node[1].name = f"{j}_{i}_rx1_1"
            layer_edges[i] ^ rx_node[0]  # 连接上一层的输出边
            layer_edges[i] = rx_node[1]  # 更新当前层输出边
            nodes.append(rx_node)

        # 添加 iswap-inv 门
        if j != 0:
            for i in reversed(range(j % 2, nqubits, 2)):
                iswap_node = tn.Node(iswap_inv_matrix)

                up,down,_ = tn.split_node(iswap_node, [iswap_node[0], iswap_node[2]], [iswap_node[1], iswap_node[3]],max_singular_values=4)
                up.name = f"{j}_iswap-up"
                down.name = f"{j}_iswap-down"
                up[0].name = f"{j}_iswap-up_0"
                up[1].name = f"{j}_iswap-up_1"
                down[1].name = f"{j}_iswap-down_0"
                down[2].name = f"{j}_iswap-down_1"

                # 连接上一层的边（如果是第一层，则无需连接输入边）
                if layer_edges[i] is not None:
                    layer_edges[i] ^ up[0]
                if layer_edges[(i + 1) % nqubits] is not None:
                    layer_edges[(i + 1) % nqubits] ^ down[1]

                # 更新输出边
                layer_edges[i] = up[1]
                layer_edges[(i + 1) % nqubits] = down[2]
                down.reorder_edges([down[1], down[2], down[0]])

                nodes.append(up)
                nodes.append(down)

    return nodes

def build_parametric_diag_tn(nqubits, params):
    """
    构建一个 n 个节点的张量网络表示的对角矩阵。
    每个节点是一个 2x2 的对角矩阵，其对角线上有两个参数。

    参数:
        nqubits : int - 量子比特数量（节点数量）。
        params : ndarray - 参数数组，形状为 (nqubits, 2)，每个节点的两个参数。

    返回:
        nodes : list of tn.Node - 张量网络节点列表。
    """

    nodes = []  # 保存所有的张量节点

    for i in range(nqubits):
        # 构造每个节点的张量
        diag_tensor = np.diag(params[i])
        if i == 0 or i == nqubits - 1:
            diag_tensor = diag_tensor.reshape(2, 2, 1)
        else:
            diag_tensor = diag_tensor.reshape(2, 2, 1, 1)

        # 创建张量节点
        node = tn.Node(diag_tensor)
        node.name = f"{i}_diag"
        node[0].name = f"{i}_diag_0"
        node[1].name = f"{i}_diag_1"
        nodes.append(node)

    for i in range(nqubits - 1):
        if i == nqubits - 2:
            nodes[i][2] ^ nodes[i + 1][2]
        else:
            nodes[i][2] ^ nodes[i + 1][3]

    return nodes


def pauli_hamiltonian_to_tn(pauli_terms):
    """
    将任意泡利字符串哈密顿量表示为张量网络形式。

    参数:
        pauli_terms : list of (float, list of str) - 泡利字符串及其系数组成的列表。
                      例如 [(1.5, ['X', 'Y', 'Z']), (2.0, ['I', 'X', 'Z'])]。

    返回:
        nodes_list : list of list of tn.Node - 每个泡利字符串对应的张量网络节点列表。
        edges_list : list of list of tn.Edge - 每个泡利字符串对应的张量网络虚拟边列表。
    """
    nodes_list = []  # 保存所有泡利字符串的节点列表

    for coeff, pauli_string in pauli_terms:
        nodes = []  # 当前泡利字符串的节点

        for i, pauli_label in enumerate(pauli_string):
            # 获取对应的泡利矩阵
            pauli_matrix = pauli_matrices[pauli_label]

            # 第一个节点的张量乘上系数
            if i == 0:
                tensor = coeff * pauli_matrix
            else:
                tensor = pauli_matrix

            tensor = tensor.astype(np.float64)

            # 创建节点
            if i == 0 or i == len(pauli_string) - 1:
                node = tn.Node(tensor.reshape(2, 2, 1))
            else:
                node = tn.Node(tensor.reshape(2, 2, 1, 1))
            nodes.append(node)

        for i in range(len(pauli_string) - 1):
            if i == len(pauli_string) - 2:
                nodes[i][2] ^ nodes[i + 1][2]
            else:
                nodes[i][2] ^ nodes[i + 1][3]

        # 保存当前泡利字符串的节点和边
        nodes_list.append(nodes)

    return nodes_list[0]


def hamiltonian_to_tn(hamiltonian):
    """
    将哈密顿量表示为张量网络形式。
    """
    nodes = []
    node = tn.Node(hamiltonian.reshape(2,2,2,2))
    up,down,_ = tn.split_node(node, [node[0],node[2]], [node[1], node[3]])
    down.reorder_edges([down[1],down[2],down[0]])
    nodes.append(up)
    nodes.append(down)
    return nodes



def build_theta(obs,diag,u_l,u_l_dagger,index):
    V = 0
    for i in range(num_steps):
        for j in range(num_steps):
            if i == index and j != index:
                u_l_copy = copy.deepcopy(u_l)
                diag_copy = copy.deepcopy(diag)
                u_l_dagger_copy = copy.deepcopy(u_l_dagger)
                u_l_copy2 = copy.deepcopy(u_l)
                diag_copy2 = copy.deepcopy(diag)
                u_l_dagger_copy2 = copy.deepcopy(u_l_dagger)
                for k in range(nqubits):
                    u_l_dagger_copy[i][-k-1][1] ^ diag_copy[i][-k-1][0]
                    diag_copy[i][-k-1][1] ^ u_l_copy[i][nqubits-k-1][0]
                    u_l_copy[i][-k-1][1] ^ u_l_dagger_copy2[j][nqubits-k-1][0]
                    u_l_dagger_copy2[j][-k-1][1] ^ diag_copy2[j][-k-1][0]
                    diag_copy2[j][-k-1][1] ^ u_l_copy2[j][nqubits-k-1][0]
                    u_l_copy2[j][-k-1][1] ^ u_l_dagger_copy[i][nqubits-k-1][0]
                node = tn.contractors.greedy(u_l_copy[i]+u_l_dagger_copy[i]+diag_copy[i]+u_l_copy2[j]+u_l_dagger_copy2[j]+diag_copy2[j])
                # print(node.tensor.shape)
                V += node.tensor
            elif j == index and i != index:
                u_l_copy = copy.deepcopy(u_l)
                diag_copy = copy.deepcopy(diag)
                u_l_dagger_copy = copy.deepcopy(u_l_dagger)
                u_l_copy2 = copy.deepcopy(u_l)
                diag_copy2 = copy.deepcopy(diag)
                u_l_dagger_copy2 = copy.deepcopy(u_l_dagger)
                for k in range(nqubits):
                    u_l_dagger_copy[i][-k-1][1] ^ diag_copy[i][-k-1][0]
                    diag_copy[i][-k-1][1] ^ u_l_copy[i][nqubits-k-1][0]
                    u_l_copy[i][-k-1][1] ^ u_l_dagger_copy2[j][nqubits-k-1][0]
                    u_l_dagger_copy2[j][-k-1][1] ^ diag_copy2[j][-k-1][0]
                    diag_copy2[j][-k-1][1] ^ u_l_copy2[j][nqubits-k-1][0]
                    u_l_copy2[j][-k-1][1] ^ u_l_dagger_copy[i][nqubits-k-1][0]
                node = tn.contractors.greedy(u_l_copy[i]+u_l_dagger_copy[i]+diag_copy[i]+u_l_copy2[j]+u_l_dagger_copy2[j]+diag_copy2[j])
                V += node.tensor
    for i in range(num_steps):
        if i == index:
            obs_copy = copy.deepcopy(obs)
            u_l_copy3 = copy.deepcopy(u_l)
            diag_copy3 = copy.deepcopy(diag)
            u_l_dagger_copy3 = copy.deepcopy(u_l_dagger)
            for k in range(nqubits):
                u_l_dagger_copy3[i][-k-1][1] ^ diag_copy3[i][-k-1][0]
                diag_copy3[i][-k-1][1] ^ u_l_copy3[i][nqubits-k-1][0]
                u_l_copy3[i][-k-1][1] ^ obs_copy[-k-1][0]
                obs_copy[-k-1][1] ^ u_l_dagger_copy3[i][nqubits-k-1][0]
            node = tn.contractors.greedy(u_l_copy3[i]+u_l_dagger_copy3[i]+diag_copy3[i]+obs_copy)
            V -= 2 * node.tensor

    return V


def build_lambda(obs,diag,u_l,u_l_dagger,index):
    V = 0
    for i in range(num_steps):
        for j in range(num_steps):
            if i == index and j == index:
                diag_copy = copy.deepcopy(diag)
                diag_copy2 = copy.deepcopy(diag)
                for k in range(nqubits):
                    diag_copy[i][-k-1][1] ^ diag_copy2[j][-k-1][0]
                    diag_copy2[j][-k-1][1] ^ diag_copy[i][-k-1][0]
                node = tn.contractors.greedy(diag_copy[i]+diag_copy2[j])
                V += node.tensor
            elif i == index:
                u_l_copy = copy.deepcopy(u_l)
                diag_copy = copy.deepcopy(diag)
                u_l_dagger_copy = copy.deepcopy(u_l_dagger)
                u_l_copy2 = copy.deepcopy(u_l)
                diag_copy2 = copy.deepcopy(diag)
                u_l_dagger_copy2 = copy.deepcopy(u_l_dagger)
                for k in range(nqubits):
                    u_l_dagger_copy[i][-k-1][1] ^ diag_copy[i][-k-1][0]
                    diag_copy[i][-k-1][1] ^ u_l_copy[i][nqubits-k-1][0]
                    u_l_copy[i][-k-1][1] ^ u_l_dagger_copy2[j][nqubits-k-1][0]
                    u_l_dagger_copy2[j][-k-1][1] ^ diag_copy2[j][-k-1][0]
                    diag_copy2[j][-k-1][1] ^ u_l_copy2[j][nqubits-k-1][0]
                    u_l_copy2[j][-k-1][1] ^ u_l_dagger_copy[i][nqubits-k-1][0]
                node = tn.contractors.greedy(u_l_copy[i]+u_l_dagger_copy[i]+diag_copy[i]+u_l_copy2[j]+u_l_dagger_copy2[j]+diag_copy2[j])
                V += node.tensor
            elif j == index:
                u_l_copy = copy.deepcopy(u_l)
                diag_copy = copy.deepcopy(diag)
                u_l_dagger_copy = copy.deepcopy(u_l_dagger)
                u_l_copy2 = copy.deepcopy(u_l)
                diag_copy2 = copy.deepcopy(diag)
                u_l_dagger_copy2 = copy.deepcopy(u_l_dagger)
                for k in range(nqubits):
                    u_l_dagger_copy[i][-k-1][1] ^ diag_copy[i][-k-1][0]
                    diag_copy[i][-k-1][1] ^ u_l_copy[i][nqubits-k-1][0]
                    u_l_copy[i][-k-1][1] ^ u_l_dagger_copy2[j][nqubits-k-1][0]
                    u_l_dagger_copy2[j][-k-1][1] ^ diag_copy2[j][-k-1][0]
                    diag_copy2[j][-k-1][1] ^ u_l_copy2[j][nqubits-k-1][0]
                    u_l_copy2[j][-k-1][1] ^ u_l_dagger_copy[i][nqubits-k-1][0]
                node = tn.contractors.greedy(u_l_copy[i]+u_l_dagger_copy[i]+diag_copy[i]+u_l_copy2[j]+u_l_dagger_copy2[j]+diag_copy2[j])
                V += node.tensor

    obs_copy = copy.deepcopy(obs)
    u_l_copy3 = copy.deepcopy(u_l)
    diag_copy3 = copy.deepcopy(diag)
    u_l_dagger_copy3 = copy.deepcopy(u_l_dagger)
    for i in range(num_steps):
        if i == index:
            for k in range(nqubits):
                u_l_dagger_copy3[i][-k-1][1] ^ diag_copy3[i][-k-1][0]
                diag_copy3[i][-k-1][1] ^ u_l_copy3[i][nqubits-k-1][0]
                u_l_copy3[i][-k-1][1] ^ obs_copy[-k-1][0]
                obs_copy[-k-1][1] ^ u_l_dagger_copy3[i][nqubits-k-1][0]
            node = tn.contractors.greedy(u_l_copy3[i]+u_l_dagger_copy3[i]+diag_copy3[i]+obs_copy)
            V -= 2 * node.tensor

    return V


def objective_theta(theta,obs,diag,u_l,u_l_dagger,i):
    theta = theta.reshape(num_layers,nqubits,3)
    u_l[i] = vqc_to_tn(nqubits,theta,num_layers)
    u_l_dagger[i] = vqc_to_tn_inv(nqubits,theta,num_layers)
    V = build_theta(obs,diag,u_l,u_l_dagger,i)
    return np.real(V)


def objective_lambda(lambda_param,obs,diag,u_l,u_l_dagger,i):
    lambda_param = lambda_param.reshape(nqubits,2)
    diag[i] = build_parametric_diag_tn(nqubits,lambda_param)
    V = build_lambda(obs,diag,u_l,u_l_dagger,i)
    return np.real(V)


def optimize_theta(obs,diag,u_l,u_l_dagger,i,theta):
    # 优化 theta 参数
    theta = minimize(objective_theta, theta.reshape(-1), args=(obs,diag,u_l,u_l_dagger,i), method='BFGS',tol=1e-12)
    return theta.x


def optimize_lambda(obs,diag,u_l,u_l_dagger,i,lambda_param):
    # 优化 lambda 参数
    lambda_param = minimize(objective_lambda, lambda_param.reshape(-1), args=(obs,diag,u_l,u_l_dagger,i), method='BFGS',tol=1e-12)
    return lambda_param.x


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


def sweep(obs,diag,u_l,u_l_dagger,theta_param,lambda_param,max_sweeps=10):
    # 迭代执行前向和后向 sweep
    for sweep in range(max_sweeps):
        for i in range(num_steps):
            theta_opt = optimize_theta(obs,diag,u_l,u_l_dagger,i,theta_param[i])
            theta_param[i] = theta_opt.reshape(num_layers,nqubits,3)

            lambda_opt = optimize_lambda(obs,diag,u_l,u_l_dagger,i,lambda_param[i])
            lambda_param[i] = lambda_opt.reshape(nqubits,2)

            # verify(obs,diag,u_l,u_l_dagger)


            diag_copy = copy.deepcopy(diag)
            output_edges = []
            for k in range(nqubits):
                output_edges.append(diag_copy[i][k][0])
            for k in range(nqubits):
                output_edges.append(diag_copy[i][k][1])
            node = tn.contractors.greedy(diag_copy[i],output_edge_order=output_edges)
            lambda_param_save = tensor_to_2d(node.tensor)
            np.save(f'saved_models_old/{num_steps}iter_{num_layers}layers_param_{nqubits}_{i}.npy', np.array(theta_param[i]))
            np.save(
                f'saved_models_old/{num_steps}iter_{num_layers}layers_diag_{nqubits}_{i}.npy',
                np.real(np.array(np.diag(lambda_param_save))))
            print("diag[i]:", np.real(np.array(np.diag(lambda_param_save))))


    return


def verify(obs,diag,u_l,u_l_dagger):
    es = 0
    for i in range(num_steps):
        u_l_copy = copy.deepcopy(u_l)
        diag_copy = copy.deepcopy(diag)
        u_l_dagger_copy = copy.deepcopy(u_l_dagger)
        for k in range(nqubits):
            u_l_dagger_copy[i][-k-1][1] ^ diag_copy[i][-k-1][0]
            diag_copy[i][-k-1][1] ^ u_l_copy[i][nqubits-k-1][0]
        output_edges = []
        for k in range(nqubits):
            output_edges.append(u_l_dagger_copy[i][k][0])
        for k in reversed(range(nqubits)):
            output_edges.append(u_l_copy[i][-k-1][1])
        node =  tn.contractors.greedy(u_l_copy[i]+u_l_dagger_copy[i]+diag_copy[i],output_edge_order=output_edges)
        es += tensor_to_2d(np.squeeze(node.tensor))

    obs_cal = np.load(f"{nqubits}-K-Local-H.npy")
    obs_cal = tensor_to_2d(obs_cal)
    # obs_cal = np.kron(np.kron(np.kron(pauli_matrices['X'],pauli_matrices['Z']),pauli_matrices['X']),pauli_matrices['Z'])
    print("obs:",obs_cal)
    print("es:",es)

    print("norm:",np.linalg.norm(obs_cal-es,ord=2))


init_scale = 0.1

I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)  # 不一定需要
Z = np.array([[1,0],[0,-1]], dtype=complex)

J = 1.0
g = 1.0


W1 = np.zeros((3, 2, 2), dtype=complex)
W1[0,:,:] = X    # I_1
W1[1,:,:] = -Z   # -Z_1
W1[2,:,:] = g*X  # g X_1

Wn = np.zeros((3, 2, 2), dtype=complex)
Wn[0,:,:] = g*X  # g X_n
Wn[1,:,:] = -Z   # -Z_n
Wn[2,:,:] = X    # I_n

# 3x3 大小, 每个单元是一个 2x2 矩阵
Wi = np.zeros((3,3,2,2), dtype=complex)

# 第一行
Wi[0,0,:,:] = X
Wi[0,1,:,:] = -Z
Wi[0,2,:,:] = g*X
# 第二行
Wi[1,0,:,:] = 0
Wi[1,1,:,:] = 0
Wi[1,2,:,:] = Z
# 第三行
Wi[2,0,:,:] = 0
Wi[2,1,:,:] = 0
Wi[2,2,:,:] = X

obs = []

for i in range(nqubits):
    if i == 0:
        W = W1
        node = tn.Node(W)
        node.reorder_edges([node[1],node[2],node[0]])
    elif i == nqubits - 1:
        W = Wn
        node = tn.Node(W)
        node.reorder_edges([node[1],node[2],node[0]])
    else:
        W = Wi
        node = tn.Node(W)
        node.reorder_edges([node[2],node[3],node[0],node[1]])

    obs.append(node)

for i in range(nqubits - 1):
    if i == nqubits - 2:
        obs[i][2] ^ obs[i + 1][2]
    else:
        obs[i][2] ^ obs[i + 1][3]

# obs = pauli_hamiltonian_to_tn([(1, ['X', 'Z','X','Z'])])
diag_param = np.random.uniform(-np.pi * init_scale, np.pi * init_scale, size=(num_steps, nqubits, 2))
diag = []
for i in range(num_steps):
    nodes = build_parametric_diag_tn(nqubits,diag_param[i])
    diag.append(nodes)

u_l_param = np.random.uniform(-np.pi * init_scale, np.pi * init_scale, size=(num_steps, num_layers, nqubits, 3))
u_l = []
for i in range(num_steps):
    nodes = vqc_to_tn(nqubits,u_l_param[i],num_layers)
    u_l.append(nodes)

u_l_dagger = []
for i in range(num_steps):
    nodes = vqc_to_tn_inv(nqubits,u_l_param[i],num_layers)
    u_l_dagger.append(nodes)

start = time.time()
sweep(obs,diag,u_l,u_l_dagger,u_l_param,diag_param,2)
end = time.time()
print("complete:", num_steps, num_layers)
print("Time used:",end-start)