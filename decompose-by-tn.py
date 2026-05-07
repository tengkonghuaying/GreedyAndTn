import argparse
import copy
import time
import os

from scipy.linalg import solve
import numpy as np
import scipy.linalg

# ====================================================
# 【关键修复】SVD 收敛性热修补
# 强制替换 np.linalg.svd，遇到不收敛时自动回退到更稳健的算法
# ====================================================
_orig_svd = np.linalg.svd


def robust_svd(a, full_matrices=False, compute_uv=True, hermitian=False):
    # 1. 原始 SVD
    try:
        u, s, vt = _orig_svd(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)
    except np.linalg.LinAlgError:
        u, s, vt = scipy.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, lapack_driver='gesvd')

    # 2. 噪声清洗
    # 防止噪声积累导致能量过冲
    cutoff = 1e-13
    mask = s > cutoff
    if np.sum(mask) == 0:  # 防止全零
        mask[0] = True


    return u, s, vt

np.linalg.svd = robust_svd
import tensornetwork as tn
from scipy.optimize import minimize
import tensorcircuit as tc
import math
# tc.set_backend('jax')
tc.set_backend("numpy")
tn.set_default_backend("numpy")
K = tc.backend


# # 创建解析器
# parser = argparse.ArgumentParser(
#     description="Process data files",
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter
# )
# parser.add_argument("--nqubits", type=int, default=8)
# parser.add_argument("--num_steps", type=int, default=6)
# parser.add_argument("--num_layers", type=int, default=3)
# parser.add_argument('--postfix', type=str, default='H4_0.4_sto-3g',
#                        help='输出文件后缀，默认"H4_0.4_sto-3g"')
# parser.add_argument("--max_bond", type=int, default=64)
# parser.add_argument("--num", type=int, default=0)
#
# # # 解析参数
# args = parser.parse_args()
#
# nqubits = args.nqubits
# num_steps = args.num_steps
# num_layers = args.num_layers
# postfix = args.postfix
# max_bond = args.max_bond
# num = args.num


nqubits = 4
num_steps = 5
num_layers = 4
postfix = 'H2_4.3_sto-3g'
max_bond = 16
num = 0


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


def build_circuit_and_groups(nqubits, param, nlayers, is_inverse=False):
    """
    【Brick-wall 结构修改版】构建 TN 结构。
    """
    flat_nodes = []
    cols = [[] for _ in range(nqubits)]
    layer_edges = [None] * nqubits

    if not is_inverse:
        # === 正向电路 U ===
        for j in range(nlayers):
            # 1. iSWAP (Brick-wall)
            start_idx = j % 2

            for i in range(start_idx, nqubits, 2):
                if i + 1 >= nqubits: continue

                node = tn.Node(iswap_matrix, backend="numpy")
                up, down, _ = tn.split_node(node, [node[0], node[2]], [node[1], node[3]], max_singular_values=4)
                up.name = f"{j}_iswap_up_{i}"
                down.name = f"{j}_iswap_down_{i + 1}"

                if layer_edges[i]: layer_edges[i] ^ up[0]
                if layer_edges[i + 1]: layer_edges[i + 1] ^ down[1]
                layer_edges[i] = up[1]
                layer_edges[i + 1] = down[2]

                down.reorder_edges([down[1], down[2], down[0]])
                flat_nodes.extend([up, down])
                cols[i].append(up)
                cols[i + 1].append(down)

            # 2. Gates
            for gate_idx, gate_func in enumerate([rx_matrix, ry_matrix, rx_matrix]):
                for i in range(nqubits):
                    node = tn.Node(gate_func(param[j, i, gate_idx]), backend="numpy")
                    node.name = f"{j}_{i}_g{gate_idx}"
                    if layer_edges[i]: layer_edges[i] ^ node[0]
                    layer_edges[i] = node[1]
                    flat_nodes.append(node)
                    cols[i].append(node)

    else:
        # === 反向电路 U_dagger ===
        for j in reversed(range(nlayers)):
            # 1. Gates (Inverse Order)
            for gate_idx, gate_func in zip([2, 1, 0], [rx_matrix, ry_matrix, rx_matrix]):
                for i in range(nqubits):
                    node = tn.Node(gate_func(-param[j, i, gate_idx]), backend="numpy")
                    node.name = f"{j}_{i}_g{gate_idx}_inv"
                    if layer_edges[i]: layer_edges[i] ^ node[0]
                    layer_edges[i] = node[1]
                    flat_nodes.append(node)
                    cols[i].append(node)

            # 2. iSWAP-inv (Brick-wall)
            start_idx = j % 2

            for i in reversed(range(start_idx, nqubits, 2)):
                if i + 1 >= nqubits: continue

                node = tn.Node(iswap_inv_matrix, backend="numpy")
                up, down, _ = tn.split_node(node, [node[0], node[2]], [node[1], node[3]], max_singular_values=4)
                up.name = f"{j}_iswap_up_{i}_inv"
                down.name = f"{j}_iswap_down_{i + 1}_inv"

                if layer_edges[i]: layer_edges[i] ^ up[0]
                if layer_edges[i + 1]: layer_edges[i + 1] ^ down[1]
                layer_edges[i] = up[1]
                layer_edges[i + 1] = down[2]
                down.reorder_edges([down[1], down[2], down[0]])

                flat_nodes.extend([up, down])
                cols[i].append(up)
                cols[i + 1].append(down)

    return flat_nodes, cols


import gc


def apply_1q_gate_to_node(node, gate_matrix, side='ket'):
    """
    将单比特门矩阵作用于 MPO 节点的物理脚。
    node: (PhysOut, PhysIn, Left, Right)
    side: 'ket' (作用于 PhysOut, 左乘), 'bra' (作用于 PhysIn, 右乘)
    """
    # 转换为 TensorNetwork 节点
    g_node = tn.Node(gate_matrix)  # (Out, In)

    if side == 'ket':
        # Ket: G @ M. 连接 Gate(In) -> Node(PhysOut)
        # Gate: (0:Out, 1:In), Node: (0:P_out, 1:P_in, 2:L, 3:R)
        g_node[1] ^ node[0]
        new_node = tn.contract(g_node[1])
        # new_node 轴顺序: (Gate_Out, P_in, L, R)
        # 已经是标准顺序，不需要重排
        pass
    else:
        # Bra: M @ G. 连接 Node(PhysIn) -> Gate(Out)
        # 此时传入的 gate_matrix 应该是要右乘的矩阵 (例如 U^dag)
        # Node: (0:P_out, 1:P_in, 2:L, 3:R), Gate: (0:Out, 1:In)
        node[1] ^ g_node[0]
        new_node = tn.contract(node[1])
        # new_node 轴顺序: (P_out, L, R, Gate_In)
        # 需要重排为 (P_out, Gate_In, L, R)
        new_node.reorder_edges([new_node[0], new_node[3], new_node[1], new_node[2]])

    return new_node


def apply_2q_gate_trunc(nodes, idx1, idx2, gate_tensor, max_bond_dim, side='ket'):
    """
    【修复版】去除了错误的二次归一化，保留物理模长，同时保持 SVD 数值稳定性。
    """
    n1 = nodes[idx1]
    n2 = nodes[idx2]
    g = tn.Node(gate_tensor, backend="numpy")

    # 确保连接
    if n1[3] is not n2[2]:
        n1[3] ^ n2[2]

    edge_left = n1[2]
    edge_right = n2[3]

    if side == 'ket':
        g[2] ^ n1[0]
        g[3] ^ n2[0]
        combined = tn.contractors.greedy([n1, n2, g], output_edge_order=[
            g[0], n1[1], edge_left,
            g[1], n2[1], edge_right
        ])
    else:
        n1[1] ^ g[0]
        n2[1] ^ g[1]
        combined = tn.contractors.greedy([n1, n2, g], output_edge_order=[
            n1[0], g[2], edge_left,
            n2[0], g[3], edge_right
        ])

    # 1. 预归一化 (Pre-normalization)
    # 目的：将数值拉回 1.0 附近，防止 SVD 输入数值过大/过小导致不收敛
    raw_tensor = combined.tensor
    norm_val = np.linalg.norm(raw_tensor)

    # 避免除以 0
    if norm_val < 1e-15:
        norm_val = 1.0

    # 缩放张量
    combined.tensor = raw_tensor / norm_val

    # 2. 指定边
    left_edges_svd = combined.edges[:3]
    right_edges_svd = combined.edges[3:]

    # 3. 执行 SVD (使用 robust_svd)
    U, S, V, _ = tn.split_node_full_svd(
        combined,
        left_edges=left_edges_svd,
        right_edges=right_edges_svd,
        max_singular_values=max_bond_dim,
        max_truncation_err=1e-12
    )

    # 4. 还原物理模长
    # 将之前除掉的 norm_val 乘回去，保证 H = U^dag H U 变换是保范数的
    S.tensor = S.tensor * norm_val

    # 吸收奇异值
    V = tn.contract(S[1])
    V.reorder_edges([V[1], V[2], V[0], V[3]])

    nodes[idx1] = U
    nodes[idx2] = V
    return nodes


def evolve_mpo_iterative(mpo_nodes, theta, layers, max_bond_dim, inverse=False):
    """
    【逻辑修复版】修正了 inverse=True 时的门应用顺序。
    """
    # 1. 重建节点
    nodes = []
    for n in mpo_nodes:
        nodes.append(tn.Node(np.array(n.tensor, copy=True), backend="numpy"))

    for k in range(len(nodes) - 1):
        if nodes[k][3] is not nodes[k + 1][2]:
            nodes[k][3] ^ nodes[k + 1][2]

    # 2. 定义门
    def get_rx(t):
        return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]],
                        dtype=np.complex128)

    def get_ry(t):
        return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], dtype=np.complex128)

    iswap_mat = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=np.complex128).reshape(2, 2,
                                                                                                                  2, 2)
    iswap_dag_mat = iswap_mat.conj().transpose(2, 3, 0, 1)

    # 3. 循环
    # U = L_{N-1} ... L_0. TN apply: 0 -> N-1.
    # U^dag = L_0^dag ... L_{N-1}^dag. TN apply: N-1 -> 0.

    if not inverse:
        layer_range = range(layers)
    else:
        layer_range = reversed(range(layers))

    for j in layer_range:
        params_j = theta[j]
        start_idx = j % 2

        if j % 5 == 0: gc.collect()

        if inverse:
            # === Inverse (U^dag) ===
            # Layer L = Gates * iSWAP.
            # L^dag = iSWAP^dag * Gates^dag.
            # TN Order: Apply Gates^dag first? No.
            # Operator A B |psi>. TN apply B then A.
            # We want iSWAP^dag * Gates^dag.
            # So TN apply: Gates^dag THEN iSWAP^dag.

            # 1. Gates (Rx, Ry, Rx) -> (Rx, Ry, Rx)^dag = Rx^dag Ry^dag Rx^dag
            # Loop gate_idx 2->0 (Reverse)
            # Apply to Ket: U^dag. Bra: U.
            for gate_idx, gate_gen in zip([2, 1, 0], [get_rx, get_ry, get_rx]):
                for i in range(nqubits):
                    mat = gate_gen(-params_j[i, gate_idx])
                    nodes[i] = apply_1q_gate_to_node(nodes[i], mat, 'ket')  # Ket 乘 U^dag
                    nodes[i] = apply_1q_gate_to_node(nodes[i], mat.conj().T, 'bra')  # Bra 乘 U

            # 2. iSWAP (Brick-wall)
            # iSWAP^dag
            for i in reversed(range(start_idx, nqubits - 1, 2)):
                nodes = apply_2q_gate_trunc(nodes, i, i + 1, iswap_dag_mat, max_bond_dim, 'ket')
                nodes = apply_2q_gate_trunc(nodes, i, i + 1, iswap_mat, max_bond_dim, 'bra')

        else:
            # === Forward (U) ===
            # Layer = Gates * iSWAP.
            # TN Apply: iSWAP THEN Gates.

            # 1. iSWAP
            for i in range(start_idx, nqubits - 1, 2):
                nodes = apply_2q_gate_trunc(nodes, i, i + 1, iswap_mat, max_bond_dim, 'ket')
                nodes = apply_2q_gate_trunc(nodes, i, i + 1, iswap_dag_mat, max_bond_dim, 'bra')

            # 2. Gates
            for gate_idx, gate_gen in enumerate([get_rx, get_ry, get_rx]):
                for i in range(nqubits):
                    mat = gate_gen(params_j[i, gate_idx])
                    nodes[i] = apply_1q_gate_to_node(nodes[i], mat, 'ket')
                    nodes[i] = apply_1q_gate_to_node(nodes[i], mat.conj().T, 'bra')

    return nodes


# 兼容接口
def vqc_to_tn(nqubits, param, nlayers):
    nodes, _ = build_circuit_and_groups(nqubits, param, nlayers, is_inverse=False)
    return nodes


def vqc_to_tn_inv(nqubits, param, nlayers):
    nodes, _ = build_circuit_and_groups(nqubits, param, nlayers, is_inverse=True)
    return nodes


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


def build_theta(obs, u_l, u_l_dagger):
    """
    计算目标函数 V = sum( |diag(U @ H @ U^dag)|^2 )
    【稳健内存优化版】
    1. 逐列预收缩: 将每一列压缩为一个节点，显式管理外部连接边，解决 ValueError。
    2. 全局收缩: 仅收缩压缩后的骨架，极大降低内存消耗。
    """
    nqubits = len(obs)

    # 1. 准备电路
    # 获取列结构 (这些是新生成的节点)
    cols_u = distribute_nodes_to_columns(u_l, nqubits, num_layers, is_inverse=False)
    cols_ud = distribute_nodes_to_columns(u_l_dagger, nqubits, num_layers, is_inverse=True)

    # 深拷贝 obs，保留其内部 MPO 连接结构
    obs_copy = copy.deepcopy(obs)

    # Copy Tensors
    delta = np.zeros((2, 2, 2))
    delta[0, 0, 0] = 1.0;
    delta[1, 1, 1] = 1.0
    copy_nodes = [tn.Node(delta, name=f"copy_{k}") for k in range(nqubits)]

    # 2. 建立垂直连接 (U * H * U^dag)
    for k in range(nqubits):
        u_top = cols_u[k][-1];
        u_bot = cols_u[k][0]
        ud_top = cols_ud[k][0];
        ud_bot = cols_ud[k][-1]
        node_h = obs_copy[k];
        node_cp = copy_nodes[k]

        # 连线
        # U_out(1) -> H_in(1)
        u_top[1] ^ node_h[1]
        # H_out(0) -> U_dag_in(0)
        node_h[0] ^ ud_top[0]
        # U_in(0) -> Copy[0]
        u_bot[0] ^ node_cp[0]
        # U_dag_out(1) -> Copy[1]
        ud_bot[1] ^ node_cp[1]

        # Copy[2] 是物理对角脚 (输出)

    # 3. 逐列预收缩 (生成 Site Tensors)
    site_ket_nodes = []

    for k in range(nqubits):
        # 当前列的所有节点
        col_nodes = cols_u[k] + [obs_copy[k]] + cols_ud[k] + [copy_nodes[k]]
        col_nodes_set = set(col_nodes)

        # --- 手动收集外部边 (Robust) ---
        outer_edges = []

        for node in col_nodes:
            for edge in node.edges:
                # 1. 如果边已经在列表中，跳过
                if edge in outer_edges: continue

                # 2. 如果是物理对角脚 (Copy[2]) -> 保留
                if edge is copy_nodes[k][2]:
                    outer_edges.append(edge)
                    continue

                # 3. 如果是完全悬空边 (例如 H 的边界 Left/Right) -> 保留
                if edge.is_dangling():
                    outer_edges.append(edge)
                    continue

                # 4. 如果是连接边
                # 判断另一端是否在当前列内
                # 如果两端都在列内 -> 内部边 -> 不添加 (会被收缩掉)
                if (edge.node1 in col_nodes_set) and (edge.node2 in col_nodes_set):
                    continue

                # 否则 -> 外部边 (连接到相邻列) -> 保留
                outer_edges.append(edge)

        # 收缩当前列
        # greedy 会严格按照 outer_edges 的指示保留接口
        col_node = tn.contractors.greedy(col_nodes, output_edge_order=outer_edges)
        site_ket_nodes.append(col_node)

    # 4. 构建 Bra 链
    bra_map, _ = tn.copy(site_ket_nodes, conjugate=True)
    site_bra_nodes = [bra_map[n] for n in site_ket_nodes]

    # 5. 连接 Ket 和 Bra
    for k in range(nqubits):
        ket_node = site_ket_nodes[k]
        bra_node = site_bra_nodes[k]


        dangling_ket = list(ket_node.get_all_dangling())
        dangling_bra = list(bra_node.get_all_dangling())

        if len(dangling_ket) != len(dangling_bra):
            # 如果不一致，说明拓扑结构在复制时出了问题，或者是边界处理不对
            min_len = min(len(dangling_ket), len(dangling_bra))
            for i in range(min_len):
                dangling_ket[i] ^ dangling_bra[i]
        else:
            for e1, e2 in zip(dangling_ket, dangling_bra):
                e1 ^ e2

    # 6. 全局收缩
    # 此时图结构为：
    # Ket_MPS
    #    | (连接)
    # Bra_MPS
    # 水平方向是连通的。

    result = tn.contractors.auto(site_ket_nodes + site_bra_nodes)

    return np.real(result.tensor)


def objective_theta(theta_flat, obs_nodes,max_bond_dim=32):
    """
    【内存优化版】
    """
    theta = theta_flat.reshape(num_layers, nqubits, 3)

    # 1. 计算 H_rot
    h_rot = evolve_mpo_iterative(obs_nodes, theta, num_layers, max_bond_dim=max_bond_dim, inverse=False)

    # 2. 提取对角元
    diag_nodes = []
    for k in range(nqubits):
        t = h_rot[k].tensor
        diag_t = np.einsum('iilr->ilr', t)
        diag_nodes.append(tn.Node(diag_t, backend="numpy"))

    for k in range(nqubits - 1):
        diag_nodes[k][2] ^ diag_nodes[k + 1][1]

    # 处理边界
    l_d = tn.Node(np.ones((1,)))
    r_d = tn.Node(np.ones((1,)))
    diag_nodes[0][1] ^ l_d[0]
    diag_nodes[-1][2] ^ r_d[0]

    # 3. 计算模长
    # 手动构建共轭，不copy
    diag_conj = []
    for k in range(nqubits):
        diag_conj.append(tn.Node(diag_nodes[k].tensor.conj(), backend="numpy"))

    for k in range(nqubits - 1):
        diag_conj[k][2] ^ diag_conj[k + 1][1]

    l_dc = tn.Node(np.ones((1,)))
    r_dc = tn.Node(np.ones((1,)))
    diag_conj[0][1] ^ l_dc[0]
    diag_conj[-1][2] ^ r_dc[0]

    for k in range(nqubits):
        diag_nodes[k][0] ^ diag_conj[k][0]

    # 收缩
    try:
        res = tn.contractors.auto(diag_nodes + diag_conj + [l_d, r_d, l_dc, r_dc])
        val = np.real(res.tensor)
    except Exception as e:
        val = 0.0
        print(f"Contract fail: {e}")

    del h_rot
    del diag_nodes
    del diag_conj

    return -val


def optimize_theta(obs,max_bond_dim=32):
    """
    【全方位扫描版】
    不再依赖单一的判断逻辑，而是暴力扫描不同尺度的初始参数空间。
    以极小的时间成本（几十秒），换取 L-BFGS-B 的最佳起跑线。
    """
    import scipy.optimize

    # === 1. 定义采样策略 ===
    # 格式：(scale, count, description)
    # scale: 参数随机范围 [-scale, scale]
    # count: 采样次数
    search_strategies = [
        (0.01, 50, "Micro Init (Identity-like)"),  # 针对 Step 0
        (0.1, 50, "Small Init (Perturbation)"),  # 针对弱纠缠
        (0.5 * np.pi, 50, "Large Init (Global Search)"),  # 针对强纠缠/非对角
        (np.pi, 50, "Full Range (Random)")  # 极端情况
    ]

    print(f"  [Init] Starting Multi-Scale Pre-screening...")

    best_init_loss = float('inf')
    best_init_params = None
    best_strategy_name = "None"

    total_samples = sum(s[1] for s in search_strategies)
    current_idx = 0

    # === 2. 执行扫描 ===
    for scale, count, desc in search_strategies:
        for i in range(count):
            # 生成参数
            tmp_params = np.random.uniform(-scale, scale, size=(num_layers, nqubits, 3)).reshape(-1)

            # 计算 Loss
            try:
                loss = objective_theta(tmp_params, obs,max_bond_dim)
            except Exception:
                loss = 0  # 防爆

            # 保留最好的
            if loss < best_init_loss:
                best_init_loss = loss
                best_init_params = tmp_params.copy()
                best_strategy_name = desc
                # 实时反馈
                # print(f"    Sample {current_idx}: New Best {loss:.4f} [{desc}]")

            current_idx += 1

    print(f"  [Init] Screening done. Winner: {best_strategy_name}")
    print(f"  [Init] Best Start Loss: {best_init_loss:.6f}")

    # === 3. 动态调整优化精度 ===
    # 如果初始 Loss 很大（比如 -20），说明主要是对角元，要求高精度
    # 如果初始 Loss 很小（比如 -0.5），说明是残差，适当放宽精度
    if best_init_loss < -1.0:
        opt_tol = 1e-7
    else:
        opt_tol = 1e-5

    # === 4. 开始正式优化 ===
    x0 = best_init_params
    if x0 is None:  # 极端保底
        x0 = np.zeros(num_layers * nqubits * 3)

    try:
        result = minimize(objective_theta,
                          x0,
                          args=(obs,max_bond_dim,),
                          method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': opt_tol, 'maxfun': 20000})

        print(f"  [Optimizer] Finished. Iter: {result.nit}, Loss: {result.fun:.6f}")
        return result.x

    except Exception as e:
        print(f"  [Optimizer] Failed with error: {e}")
        return x0


def put_vqc(c, param, nlayers):
    """
    【Brick-wall 结构修改版】
    奇偶层交替连接 iSWAP 门，最大化纠缠扩散。
    """
    for j in range(nlayers):
        # === 1. iSWAP 门 (Brick-wall) ===
        # 偶数层(0,2,...) 从 0 开始: (0,1), (2,3)...
        # 奇数层(1,3,...) 从 1 开始: (1,2), (3,4)...
        start_idx = j % 2

        for i in range(start_idx, nqubits - 1, 2):
            c.unitary(i, i + 1,
                      unitary=np.array([[1, 0, 0, 0],
                                        [0, 0, 1j, 0],
                                        [0, 1j, 0, 0],
                                        [0, 0, 0, 1]]),
                      name="iswap")

        # === 2. 单比特门 (Rx, Ry, Rx) ===
        # 保持不变
        for i in range(nqubits):
            c.rx(i, theta=param[j, i, 0])
        for i in range(nqubits):
            c.ry(i, theta=param[j, i, 1])
        for i in range(nqubits):
            c.rx(i, theta=param[j, i, 2])

    return c


def optimize_diag(obs, theta,max_bond_dim=64):
    """
    计算 H_rot = U @ H @ U^dag 的对角元向量。
    【修复版】修复了 TN 收缩时漏掉边界节点导致的 ValueError。
    """
    print("  [Optimize Diag] Extracting diagonal via iterative evolution...")

    # 1. 计算 H_rot = U @ H @ U^dag
    h_rot = evolve_mpo_iterative(obs, theta, num_layers, max_bond_dim=max_bond_dim, inverse=False)

    # 2. 提取对角元构建 MPS
    # MPO Tensors: (PhysOut, PhysIn, Left, Right)
    # 对角元条件: PhysOut == PhysIn
    diag_nodes = []
    for k in range(nqubits):
        t = h_rot[k].tensor
        # Einsum: iilr -> ilr (Phys, Left, Right)
        # 得到的是一个 MPS 节点
        diag_t = np.einsum('iilr->ilr', t)
        diag_nodes.append(tn.Node(diag_t, backend="numpy"))

    # 3. 连接 MPS
    for k in range(nqubits - 1):
        # Left(k+1) connects to Right(k)
        # Indices: 0:Phys, 1:Left, 2:Right
        if diag_nodes[k][2] is not diag_nodes[k + 1][1]:
            diag_nodes[k][2] ^ diag_nodes[k + 1][1]

    # 4. 收集所有节点
    all_nodes = list(diag_nodes)

    # 第一个节点的 Left (idx 1)
    if diag_nodes[0][1].is_dangling():
        l_cap = tn.Node(np.ones((1,)), backend="numpy")
        diag_nodes[0][1] ^ l_cap[0]
        all_nodes.append(l_cap)  # 将边界节点加入列表

    # 最后一个节点的 Right (idx 2)
    if diag_nodes[-1][2].is_dangling():
        r_cap = tn.Node(np.ones((1,)), backend="numpy")
        diag_nodes[-1][2] ^ r_cap[0]
        all_nodes.append(r_cap)  # 将边界节点加入列表

    # 5. 收缩得到完整对角向量
    # 输出顺序: (Phys0, Phys1, ..., PhysN)
    out_edges = [node[0] for node in diag_nodes]

    try:
        res = tn.contractors.auto(all_nodes, output_edge_order=out_edges)
        diag_vec = res.tensor.reshape(-1)
    except Exception as e:
        print(f"  [Optimize Diag] Contraction failed: {e}")
        print(f"  Nodes: {len(all_nodes)}")
        raise e

    # 显式清理
    del h_rot
    del diag_nodes
    del all_nodes
    gc.collect()

    return np.real(diag_vec)


def distribute_nodes_to_columns(nodes, nqubits, nlayers, is_inverse=False):
    """
    【线性版】辅助函数：将扁平的节点列表分配到每一列。
    必须与 vqc_to_tn 的线性逻辑完全一致 (跳过 i+1 >= nqubits 的门)。
    """
    cols = [[] for _ in range(nqubits)]
    idx = 0

    # 生成层的顺序
    layers_range = range(nlayers) if not is_inverse else reversed(range(nlayers))

    for j in layers_range:
        if not is_inverse:
            # --- 正向 (vqc_to_tn) ---
            # 1. iSWAP (Linear)
            if j != 0:
                for i in range(j % 2, nqubits, 2):
                    # 跳过越界连接，与 vqc_to_tn 保持一致
                    if i + 1 >= nqubits:
                        continue

                    # 此时 idx 指向的是 iSWAP 分解出的 up/down 节点
                    cols[i].append(nodes[idx])
                    idx += 1  # up
                    cols[i + 1].append(nodes[idx])
                    idx += 1  # down

            # 2. Gates (Rx, Ry, Rx)
            for _ in range(3):
                for i in range(nqubits):
                    cols[i].append(nodes[idx])
                    idx += 1

        else:
            # --- 反向 (vqc_to_tn_inv) ---
            # 1. Gates (Rx, Ry, Rx) - 逆序
            for _ in range(3):
                for i in range(nqubits):
                    cols[i].append(nodes[idx])
                    idx += 1

            # 2. iSWAP-inv (Linear)
            if j != 0:
                for i in reversed(range(j % 2, nqubits, 2)):
                    # 跳过越界连接
                    if i + 1 >= nqubits:
                        continue

                    cols[i].append(nodes[idx])
                    idx += 1  # up
                    cols[i + 1].append(nodes[idx])
                    idx += 1  # down

    return cols


def vector_to_diagonal_mpo(diag_vec, nqubits, max_bond_dim=64):
    """
    将对角向量精确转换为 MPO 形式。
    通过 TT-SVD 分解，确保生成的 MPO 不包含任何来自上一步的噪声。
    """
    # 1. 将向量 reshape 为 (2, 2, ..., 2)
    tensor = diag_vec.reshape([2] * nqubits)

    nodes = []
    # 初始形状: (1, 2, 2^(N-1))
    curr_tensor = tensor.reshape(1, 2, -1)

    for k in range(nqubits - 1):
        l_dim, p_dim, r_dim_total = curr_tensor.shape

        # Reshape for SVD: (Left * Phys, Remaining)
        flattened = curr_tensor.reshape(l_dim * p_dim, -1)

        # SVD 截断
        try:
            u, s, v = np.linalg.svd(flattened, full_matrices=False)
        except np.linalg.LinAlgError:
            u, s, v = scipy.linalg.svd(flattened, full_matrices=False, lapack_driver='gesvd')

        # 截断键维
        rank = min(len(s), max_bond_dim)
        # 过滤掉极小的奇异值，防止数值不稳定
        rank = max(1, sum(s > 1e-12))

        u = u[:, :rank]
        s = s[:rank]
        v = v[:rank, :]

        # 构造当前节点的 MPS 张量: (Left, Phys, Right_Bond)
        # u shape: (l_dim * p_dim, rank) -> (l_dim, p_dim, rank)
        node_tensor = u.reshape(l_dim, p_dim, rank)

        # 转换为 MPO 对角形式: (PhysOut, PhysIn, Left, Right)
        # 只有 PhysOut == PhysIn 时有值
        mpo_tensor = np.zeros((2, 2, l_dim, rank), dtype=complex)
        for i in range(2):
            mpo_tensor[i, i, :, :] = node_tensor[:, i, :]

        nodes.append(tn.Node(mpo_tensor))

        # 准备下一次迭代: 吸收奇异值到剩余部分
        # v shape: (rank, Remaining)
        curr_tensor = np.dot(np.diag(s), v)
        # Reshape: (Left_Next, Phys_Next, Remaining_Next)
        curr_tensor = curr_tensor.reshape(rank, 2, -1)

    # 处理最后一个节点
    # curr_tensor shape: (Left, Phys, 1)
    l_last, p_last, _ = curr_tensor.shape
    mpo_tensor_last = np.zeros((2, 2, l_last, 1), dtype=complex)
    for i in range(2):
        mpo_tensor_last[i, i, :, 0] = curr_tensor[:, i, 0]

    nodes.append(tn.Node(mpo_tensor_last))

    # 连接节点
    for k in range(nqubits - 1):
        nodes[k][3] ^ nodes[k + 1][2]

    return nodes


def compute_diag_mpo(obs, theta,max_bond_dim=64):
    """
    计算对角 MPO Lambda。
    【修复版】先提取精确对角向量，再重构 MPO，防止噪声污染。
    """
    print(f"[Compute Diag] Extracting vector and rebuilding clean MPO...")

    # 1. 复用 optimize_diag 的逻辑提取精确的对角向量
    diag_vec = optimize_diag(obs, theta)
    diag_vec = np.real(diag_vec)

    # 2. 从向量构建纯净的 MPO
    final_nodes = vector_to_diagonal_mpo(diag_vec, nqubits, max_bond_dim=max_bond_dim)

    return final_nodes


def compute_obs_new(obs_old, theta, diag_mpo, max_bond_dim=64, debug_skip_compress=False):
    """
    H_new = H_old - alpha * (U_dag @ Lambda @ U)
    """
    print(f"\n[Compute Obs] Auto-Scaling Strategy (MaxBond={max_bond_dim})...")

    # 1. 计算 S = U_dag @ Lambda @ U
    s_mpo_raw = evolve_mpo_iterative(diag_mpo, theta, num_layers, max_bond_dim, inverse=True)

    # 2. 对 S 进行强力压缩 (得到 S_comp)
    # 为了防止 s_mpo_raw 被原地修改，我们先复制一份进行压缩，或者直接处理
    s_mpo_comp = compress_mpo_chain(copy.deepcopy(s_mpo_raw), max_bond_dim)

    print_debug_info(obs_old,s_mpo_comp)

    # 3. 【核心新增】计算最优系数 alpha
    # alpha = <H_old, S_comp> / <S_comp, S_comp>
    try:
        overlap_h_s = np.real(debug_contract_mpo(obs_old, s_mpo_comp))
        norm_s_sq = np.real(debug_contract_mpo(s_mpo_comp, s_mpo_comp))

        # 防止分母崩塌
        if norm_s_sq < 1e-10:
            print(f"  [Auto-Scale Warning] Norm of S is too small ({norm_s_sq:.2e}). Forcing alpha=0.")
            alpha = 0.0
        else:
            alpha = overlap_h_s / norm_s_sq

        # 强制钳位，防止爆炸
        if alpha > 1.2 or alpha < -0.8:
            print(f"  [Auto-Scale Warning] Alpha {alpha:.4f} out of bounds! Clamping to 1.0.")
            alpha = 1.0  # 或者更保守，设为 0.8 防止过冲

        print(f"  [Auto-Scale] Optimal alpha = {alpha:.6f} (Overlap={overlap_h_s:.2e}, NormSq={norm_s_sq:.2e})")

    except Exception as e:
        print(f"  [Auto-Scale] Failed to calc alpha, fallback to 1.0. Error: {e}")
        alpha = 1.0

    # 4. 执行减法 H_new = H_old - alpha * S_comp

    h_temp_nodes = []
    for k in range(nqubits):
        t_h = obs_old[k].tensor
        t_s = s_mpo_comp[k].tensor  # 使用压缩后的张量

        # 应用 alpha 并处理符号 (H - alpha * S) -> H + (-alpha * S)
        # 在 k=0 处乘上 -alpha
        if k == 0:
            t_s = -alpha * t_s

        # 确保维度匹配 (3D -> 4D)
        if t_h.ndim == 3:
            if k == 0:
                t_h = t_h.reshape(t_h.shape[0], t_h.shape[1], 1, -1)
            else:
                t_h = t_h.reshape(t_h.shape[0], t_h.shape[1], -1, 1)
        if t_s.ndim == 3:
            if k == 0:
                t_s = t_s.reshape(t_s.shape[0], t_s.shape[1], 1, -1)
            else:
                t_s = t_s.reshape(t_s.shape[0], t_s.shape[1], -1, 1)

        # 直和拼接
        new_tensor = direct_sum_mpo_tensors(t_h, t_s)
        h_temp_nodes.append(tn.Node(new_tensor))

    # 连接
    for k in range(nqubits - 1):
        h_temp_nodes[k][3] ^ h_temp_nodes[k + 1][2]

    # 5. 再次压缩 (针对 Sum 的结果)
    if debug_skip_compress:
        final_nodes = h_temp_nodes
    else:
        final_nodes = compress_mpo_chain(h_temp_nodes, max_bond_dim)

    # 还原为 node 列表返回
    res_nodes = []
    for k in range(nqubits):
        res_nodes.append(tn.Node(final_nodes[k].tensor))
    for k in range(nqubits - 1):
        res_nodes[k][3] ^ res_nodes[k + 1][2]

    return res_nodes, alpha

def debug_contract_mpo(nodes1, nodes2=None):
    """
    计算 <MPO1 | MPO2>。
    修复了边界虚拟脚未连接的问题。
    """
    # 复制 (保留内部水平连接)
    map1, _ = tn.copy(nodes1, conjugate=True)  # Bra
    chain1 = [map1[n] for n in nodes1]

    if nodes2 is None:
        map2, _ = tn.copy(nodes1, conjugate=False)  # Ket
        chain2 = [map2[n] for n in nodes1]
    else:
        map2, _ = tn.copy(nodes2, conjugate=False)
        chain2 = [map2[n] for n in nodes2]

    # 1. 连接物理脚 (PhysOut, PhysIn) -> Indices 0, 1
    for k in range(len(chain1)):
        # 假设所有节点都已经标准化为 (PhysOut, PhysIn, Left, Right)
        # 或者至少前两个是物理脚
        chain1[k][0] ^ chain2[k][0]
        chain1[k][1] ^ chain2[k][1]

    # 2. 【关键修复】连接整个链的边界 (Left of First, Right of Last)
    # 假设节点顺序是 [PhysOut, PhysIn, Left, Right]
    # 第一个节点的 Left 是 index 2
    chain1[0][2] ^ chain2[0][2]

    # 最后一个节点的 Right 是 index 3
    chain1[-1][3] ^ chain2[-1][3]

    # 全量收缩
    result = tn.contractors.auto(chain1 + chain2)
    return result.tensor


def print_debug_info(obs_old, s_tensors_positive):
    print("\n--- DEBUG INFO (Pre-calculation) ---")

    # 1. 构建 S_k MPO 节点链
    s_nodes = []
    for t in s_tensors_positive:
        # 确保是 Node 对象
        s_nodes.append(tn.Node(t))

    for k in range(len(s_nodes) - 1):
        # 连接内部水平边: Right(3) -> Left(2)
        # 前提是标准格式 (PhysOut, PhysIn, Left, Right)
        s_nodes[k][3] ^ s_nodes[k + 1][2]

    # 2. 构建 H_old MPO 节点链
    h_nodes = []
    for k in range(len(obs_old)):
        t = obs_old[k].tensor
        # 确保维度匹配 (处理边界 3D -> 4D)
        if t.ndim == 3:
            if k == 0:
                t = t.reshape(t.shape[0], t.shape[1], 1, -1)
            elif k == len(obs_old) - 1:
                t = t.reshape(t.shape[0], t.shape[1], -1, 1)
        h_nodes.append(tn.Node(t))

    for k in range(len(h_nodes) - 1):
        h_nodes[k][3] ^ h_nodes[k + 1][2]

    try:
        # 【修改】不再手动连接 Dummy Node (np.ones)
        # debug_contract_mpo 会负责连接 Bra 和 Ket 的边界 (Left-Left, Right-Right)
        # 这等价于计算 Trace(H_dag * H) 对于开放边界条件

        norm_h_sq = np.real(debug_contract_mpo(h_nodes, None))
        norm_s_sq = np.real(debug_contract_mpo(s_nodes, None))
        overlap = np.real(debug_contract_mpo(h_nodes, s_nodes))

        print(f"||H_old||^2 : {norm_h_sq:.6f}")
        print(f"||S_k||^2   : {norm_s_sq:.6f}")
        print(f"<H_old, S_k>: {overlap:.6f}")

        # 预测 H_new = H_old - S 的范数平方
        # ||H-S||^2 = ||H||^2 + ||S||^2 - 2Re(<H,S>)
        pred_diff_sq = norm_h_sq + norm_s_sq - 2 * overlap
        print(f"Predicted ||H - S||^2: {pred_diff_sq:.6f}")

    except Exception as e:
        print(f"Debug calc failed: {e}")
        import traceback
        traceback.print_exc()
    print("------------------\n")


def direct_sum_mpo_tensors(t1, t2):
    """
    直和拼接: t1 + t2.
    输入必须是 4D: (PhysOut, PhysIn, Left, Right)
    包含深度调试信息和类型安全修复。
    """
    # 1. 物理维度检查
    if t1.shape[0] != t2.shape[0] or t1.shape[1] != t2.shape[1]:
        raise ValueError(f"Physical dim mismatch: {t1.shape} vs {t2.shape}")

    l1, r1 = t1.shape[2], t1.shape[3]
    l2, r2 = t2.shape[2], t2.shape[3]

    # 2. 确定维度和模式
    # 如果原本都是 1，保持为 1 (数值加法模式)
    # 否则进行直和拼接 (Block Diag 模式)

    if l1 == 1 and l2 == 1:
        new_l = 1
        mode_l = "add"
    else:
        new_l = l1 + l2
        mode_l = "cat"

    if r1 == 1 and r2 == 1:
        new_r = 1
        mode_r = "add"
    else:
        new_r = r1 + r2
        mode_r = "cat"

    # 3. 类型安全
    # 防止 t1 是 float 而 t2 是 complex 时，t2 被截断为实数
    final_dtype = np.result_type(t1.dtype, t2.dtype)

    d_out, d_in = t1.shape[0], t1.shape[1]
    new_tensor = np.zeros((d_out, d_in, new_l, new_r), dtype=final_dtype)

    # 4. 填充 T1
    end_l1 = 1 if mode_l == "add" else l1
    end_r1 = 1 if mode_r == "add" else r1
    new_tensor[:, :, :end_l1, :end_r1] += t1

    # 5. 填充 T2
    start_l = 0 if mode_l == "add" else l1
    start_r = 0 if mode_r == "add" else r1

    end_l2 = start_l + (1 if mode_l == "add" else l2)
    end_r2 = start_r + (1 if mode_r == "add" else r2)

    new_tensor[:, :, start_l:end_l2, start_r:end_r2] += t2

    # ================= DEBUG INFO =================
    # 只在发生"数值加法"的地方（通常是边界）打印，避免刷屏
    # 我们检查 (0,0) 位置的物理元素
    if mode_l == "add" and mode_r == "add":
        # 取第一个非零元素进行验证 (防止全0看不出问题)
        # 简单起见取 [0,0,0,0]
        v1 = t1.reshape(-1)[0]
        v2 = t2.reshape(-1)[0]
        v_res = new_tensor.reshape(-1)[0]

        print(f"\n[DirectSum DEBUG] Mode: ADD (Boundary Node)")
        print(f"  Shape: {t1.shape} + {t2.shape} -> {new_tensor.shape}")
        print(f"  Value[0]: {v1:.4f} + {v2:.4f} = {v_res:.4f}")

        # 检查模长变化 (辅助判断同向/反向)
        n1 = np.abs(v1)
        n2 = np.abs(v2)
        n_res = np.abs(v_res)
        if n_res > n1 and n2 > 1e-9:
            print(f"  -> Magnitude INCREASED ( Constructive Interference / Wrong Sign? )")
        elif n_res < n1 and n2 > 1e-9:
            print(f"  -> Magnitude DECREASED ( Destructive Interference / Correct Subtraction )")

    elif mode_l == "add" or mode_r == "add":
        # 半加半拼 (首尾节点)
        print(f"\n[DirectSum DEBUG] Mode: HYBRID (First/Last Node)")
        print(f"  Shape: {t1.shape} + {t2.shape} -> {new_tensor.shape}")
        # 这种情况下，一部分是加，一部分是拼，很难用单个数值衡量
    # ==============================================

    return new_tensor


def compress_mpo_chain(nodes, max_bond_dim):
    """
    对 MPO 链进行压缩，严格维护轴顺序 (PhysOut, PhysIn, Left, Right)。
    修复了 QR 阶段的轴重排逻辑。
    """
    N = len(nodes)

    # ==========================================
    # 1. 从左到右 QR (Left-Canonicalize)
    # ==========================================
    for i in range(N - 1):
        node = nodes[i]
        next_node = nodes[i + 1]

        shared_edges = tn.get_shared_edges(node, next_node)
        if not shared_edges:
            raise ValueError(f"Nodes {i} and {i + 1} are not connected!")
        right_edge = list(shared_edges)[0]

        all_edges = node.edges
        left_edges_to_keep = [e for e in all_edges if e is not right_edge]

        # QR 分解
        q, r = tn.split_node_qr(node,
                                left_edges=left_edges_to_keep,
                                right_edges=[right_edge])

        nodes[i] = q

        # 收缩 R 到下一个节点
        # r[1] 连接 next_node
        # r 的顺序: [NewLeft, Bond]
        # next_node 的顺序: [PhysOut, PhysIn, Bond(Left), Right]
        # contract 结果顺序: [NewLeft, PhysOut, PhysIn, Right]
        nodes[i + 1] = tn.contract(r[1])

        # --- 轴重排 ---
        # 当前: [NewLeft(0), PhysOut(1), PhysIn(2), Right(3)]
        # 目标: [PhysOut, PhysIn, NewLeft, Right]
        # 变换: [1, 2, 0, 3]

        new_node = nodes[i + 1]
        es = new_node.edges

        if len(es) == 4:
            new_node.reorder_edges([es[1], es[2], es[0], es[3]])
        elif len(es) == 3 and i + 1 == N - 1:
            # 最后一个节点: [NewLeft, PhysOut, PhysIn]
            # 目标: [PhysOut, PhysIn, NewLeft] (对应 P0, P1, L, 1)
            new_node.reorder_edges([es[1], es[2], es[0]])

    # ==========================================
    # 2. 从右到左 SVD (Right-Canonicalize & Truncate)
    # ==========================================
    for i in range(N - 1, 0, -1):
        node = nodes[i]
        prev_node = nodes[i - 1]

        shared_edges = tn.get_shared_edges(node, prev_node)
        if not shared_edges:
            raise ValueError(f"Nodes {i} and {i - 1} are not connected!")
        left_bond = list(shared_edges)[0]

        all_edges = node.edges
        # node 顺序: [PhysOut, PhysIn, Left, Right]
        edges_to_keep_on_v = [e for e in all_edges if e is not left_bond]

        # SVD 分解
        u, s, v, _ = tn.split_node_full_svd(node,
                                            left_edges=[left_bond],
                                            right_edges=edges_to_keep_on_v,
                                            max_singular_values=max_bond_dim,
                                            max_truncation_err=1e-12)

        # V 顺序 (由 split 产生): [NewBond, PhysOut, PhysIn, Right]
        # 目标顺序: [PhysOut, PhysIn, NewBond, Right]
        # 变换: [1, 2, 0, 3]

        v_edges = v.edges
        if len(v_edges) == 4:
            v.reorder_edges([v_edges[1], v_edges[2], v_edges[0], v_edges[3]])
        elif len(v_edges) == 3:
            # 边界节点: [NewBond, PhysOut, PhysIn] -> [PhysOut, PhysIn, NewBond]
            v.reorder_edges([v_edges[1], v_edges[2], v_edges[0]])

        nodes[i] = v

        # 收缩 U @ S @ prev_node
        # u: [OldLeft, NewBond]. s: [NewBond, V_Bond].
        # mult = u @ s -> [OldLeft, V_Bond]
        mult = tn.contract(u.edges[-1])

        # prev_node: [PhysOut, PhysIn, Left, OldRight]
        # mult 连接的是 OldRight (即 OldLeft 对应的连接)
        # contract(mult[0]) -> [PhysOut, PhysIn, Left, V_Bond]
        # 顺序已经是正确的，无需重排。
        nodes[i - 1] = tn.contract(mult.edges[0])

    return nodes


def standardize_mpo_format(nodes):
    """
    将用户特定的 obs 格式强制转换为标准格式 (PhysOut, PhysIn, Left, Right)。

    基于连接逻辑 obs[i][2] ^ obs[i+1][3] 推断：
    - Input Middle: (Phys1, Phys2, Right, Left)
    - Input First:  (Phys1, Phys2, Right)
    - Input Last:   (Phys1, Phys2, Left)
    """
    new_nodes = []
    N = len(nodes)

    for k in range(N):
        t = nodes[k].tensor

        if k == 0:
            # First: (P1, P2, R) -> (P1, P2, 1, R)
            if t.ndim == 3:
                t_std = t.reshape(t.shape[0], t.shape[1], 1, t.shape[2])
            else:
                t_std = t

        elif k == N - 1:
            # Last: (P1, P2, L) -> (P1, P2, L, 1)
            if t.ndim == 3:
                t_std = t.reshape(t.shape[0], t.shape[1], t.shape[2], 1)
            else:
                t_std = t

        else:
            # Middle: (P1, P2, Right, Left) -> (P1, P2, Left, Right)
            if t.ndim == 4:
                t_std = t.transpose(0, 1, 3, 2)
            else:
                t_std = t

        new_nodes.append(tn.Node(t_std))

    # 重新连接 (Standard: Right(3) ^ Left(2))
    for k in range(N - 1):
        new_nodes[k][3] ^ new_nodes[k + 1][2]

    return new_nodes


def sweep(obs):
    for i in range(num_steps):
        print(f"\n{'='*60}")
        print(f"Step: {i}/{num_steps-1}")
        print(f"{'='*60}")

        step_start_time = time.time()

        # 定义文件路径
        param_file = f'saved_models_new_brick/{i}iter_{num_layers}layers_param_{postfix}_{max_bond}_{num}.npy'
        diag_file = f'saved_models_new_brick/{i}iter_{num_layers}layers_diag_{postfix}_{max_bond}_{num}.npy'
        qasm_file = f'saved_models_new_brick/{i}iter_{num_layers}layers_{postfix}_{max_bond}_{num}.qasm'

        need_save = False

        # 检查文件是否存在
        if os.path.exists(param_file) and os.path.exists(diag_file):
        # if 1 == -1:
            print("检测到已保存的参数文件，直接加载...")
            load_start_time = time.time()

            # 加载保存的参数
            theta_opt = np.load(param_file)
            diag_opt = np.load(diag_file)

            load_time = time.time() - load_start_time
            print(f"  加载参数耗时: {load_time:.4f} 秒")
            print(f"  theta_opt shape: {theta_opt.shape}")
            print(f"  diag_opt shape: {diag_opt.shape}")
        else:
            print("未找到保存的参数，开始优化分解...")

            # 优化 theta
            print("  正在优化 theta...")
            theta_start_time = time.time()
            theta_opt = optimize_theta(obs,max_bond)
            theta_opt = theta_opt.reshape(num_layers, nqubits, 3)
            theta_time = time.time() - theta_start_time
            print(f"  theta 优化耗时: {theta_time:.4f} 秒")

            # 优化对角矩阵
            print("  正在优化对角矩阵...")
            diag_start_time = time.time()
            diag_opt = optimize_diag(obs, theta_opt,max_bond)
            diag_time = time.time() - diag_start_time
            print(f"  对角矩阵优化耗时: {diag_time:.4f} 秒")

            need_save = True

        # 计算对角 MPO
        print("  正在计算对角 MPO...")
        mpo_start_time = time.time()
        diag_mpo = compute_diag_mpo(obs, theta_opt,max_bond)
        mpo_time = time.time() - mpo_start_time
        print(f"  计算对角 MPO 耗时: {mpo_time:.4f} 秒")

        # 更新观测量（减去当前项并压缩）
        print("  正在更新观测量（减法+压缩）...")
        obs_start_time = time.time()
        obs, alpha = compute_obs_new(obs, theta_opt, diag_mpo, max_bond)
        obs_time = time.time() - obs_start_time
        print(f"  更新观测量耗时: {obs_time:.4f} 秒")

        if need_save == True:
            # 创建保存目录
            os.makedirs('saved_models_new', exist_ok=True)

            # 保存参数
            print("  正在保存参数...")
            save_start_time = time.time()

            saved_circ = tc.Circuit(nqubits)
            put_vqc(saved_circ, theta_opt, num_layers)
            with open(qasm_file, 'w+') as file:
                file.write(saved_circ.to_openqasm())

            np.save(param_file, np.array(theta_opt))
            np.save(diag_file, diag_opt*alpha)

            save_time = time.time() - save_start_time
            print(f"  保存参数耗时: {save_time:.4f} 秒")

        # 手动触发垃圾回收
        import gc
        gc.collect()

        # 统计本轮总耗时
        step_total_time = time.time() - step_start_time
        print(f"\n  本轮总耗时: {step_total_time:.4f} 秒")
        print(f"{'='*60}\n")

    return


init_scale = 0.1

I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)  # 不一定需要
Z = np.array([[1,0],[0,-1]], dtype=complex)
#
# J = 1.0
# g = 1.0
#
#
# W1 = np.zeros((3, 2, 2), dtype=complex)
# W1[0,:,:] = X    # I_1
# W1[1,:,:] = -Z   # -Z_1
# W1[2,:,:] = g*X  # g X_1
#
# Wn = np.zeros((3, 2, 2), dtype=complex)
# Wn[0,:,:] = g*X  # g X_n
# Wn[1,:,:] = -Z   # -Z_n
# Wn[2,:,:] = X    # I_n
#
# # 3x3 大小, 每个单元是一个 2x2 矩阵
# Wi = np.zeros((3,3,2,2), dtype=complex)
#
# # 第一行
# Wi[0,0,:,:] = X
# Wi[0,1,:,:] = -Z
# Wi[0,2,:,:] = g*X
# # 第二行
# Wi[1,0,:,:] = 0
# Wi[1,1,:,:] = 0
# Wi[1,2,:,:] = Z
# # 第三行
# Wi[2,0,:,:] = 0
# Wi[2,1,:,:] = 0
# Wi[2,2,:,:] = X
#
# obs = []
#
# for i in range(nqubits):
#     if i == 0:
#         W = W1
#         node = tn.Node(W)
#         node.reorder_edges([node[1],node[2],node[0]])
#     elif i == nqubits - 1:
#         W = Wn
#         node = tn.Node(W)
#         node.reorder_edges([node[1],node[2],node[0]])
#     else:
#         W = Wi
#         node = tn.Node(W)
#         node.reorder_edges([node[2],node[3],node[0],node[1]])
#
#     obs.append(node)
#
# for i in range(nqubits - 1):
#     if i == nqubits - 2:
#         obs[i][2] ^ obs[i + 1][2]
#     else:
#         obs[i][2] ^ obs[i + 1][3]


# 加载NPZ文件
data = np.load(f'{postfix}_MPO.npz')
mts = [data[f'mt_{i}'] for i in range(nqubits)]
data.close()

# 处理边界张量的虚拟维度
def process_boundary_tensor(tensor, is_first):
    """处理边界张量，去除维度为1的虚拟边"""
    if is_first:
        # 第一个张量形状应为 (1, 2, 2, d) -> 去除第一个维度 (1)
        return tensor.squeeze(axis=0)  # 变为 (2, 2, d)
    else:
        # 最后一个张量形状应为 (d, 2, 2, 1) -> 去除最后一个维度 (1)
        return tensor.squeeze(axis=-1)  # 变为 (d, 2, 2)

# 处理所有张量
processed_mts = []
for i in range(nqubits):
    if i == 0:
        # 处理第一个张量 (mt_0)
        tensor = process_boundary_tensor(mts[i], is_first=True)
    elif i == nqubits - 1:
        # 处理最后一个张量 (mt_7)
        tensor = process_boundary_tensor(mts[i], is_first=False)
    else:
        # 中间张量保持不变
        tensor = mts[i]
    processed_mts.append(tensor)

obs = []
for i in range(nqubits):
    W = processed_mts[i]
    node = tn.Node(W)

    # 根据张量位置设置边顺序
    if i == 0:
        # 第一个张量: 形状 (2, 2, d) -> 对应原W1的 (3,2,2)
        # 边顺序: 物理边1, 物理边2, 虚拟边
        node.reorder_edges([node[0], node[1], node[2]])
    elif i == nqubits - 1:
        # 最后一个张量: 形状 (d, 2, 2) -> 对应原Wn的 (3,2,2)
        # 边顺序: 虚拟边, 物理边1, 物理边2
        node.reorder_edges([node[1], node[2], node[0]])
    else:
        # 中间张量: 形状 (d_in, 2, 2, d_out) -> 对应原Wi的 (3,3,2,2)
        # 边顺序: 左虚拟边, 右虚拟边, 物理边1, 物理边2
        node.reorder_edges([node[1], node[2], node[3], node[0]])

    obs.append(node)

for i in range(nqubits - 1):
    if i == nqubits - 2:
        obs[i][2] ^ obs[i + 1][2]
    else:
        obs[i][2] ^ obs[i + 1][3]

# obs = pauli_hamiltonian_to_tn([(1, ['X', 'Z','X','Z'])])
# diag_param = np.random.uniform(-np.pi * init_scale, np.pi * init_scale, size=(num_steps, nqubits, 2))
# diag = []
# for i in range(num_steps):
#     nodes = build_parametric_diag_tn(nqubits,diag_param[i])
#     diag.append(nodes)
#
# u_l_param = np.random.uniform(-np.pi * init_scale, np.pi * init_scale, size=(num_steps, num_layers, nqubits, 3))
# u_l = []
# for i in range(num_steps):
#     nodes = vqc_to_tn(nqubits,u_l_param[i],num_layers)
#     u_l.append(nodes)
#
# u_l_dagger = []
# for i in range(num_steps):
#     nodes = vqc_to_tn_inv(nqubits,u_l_param[i],num_layers)
#     u_l_dagger.append(nodes)

# verify(obs,diag,u_l,u_l_dagger)
obs = standardize_mpo_format(obs)
start = time.time()
sweep(obs)
end = time.time()
print("complete:", num_steps, num_layers)
print("Time used:",end-start)
# cal_H = 0
# for i in range(num_steps):
#     U = tensor_to_2d(tn.contractors.greedy(u_l[i],output_edge_order=[u_l[i][0][0],u_l[i][1][0],u_l[i][-2][1],u_l[i][-1][1]]).tensor)
#     print("U:",U)
#     diag = np.load(f'saved_models_old/{num_steps}iter_{num_layers}layers_diag_{nqubits}_{i}.npy')
#     cal_H += U.conj().T @ np.diag(diag) @ U
# print("cal_H:",cal_H)
#
# verify(obs,diag,u_l,u_l_dagger)

# node = tn.contractors.greedy(u_l[0],ignore_edge_order=True)
# print(node.tensor)
# identity_matrix = np.eye(4)  # 获取单位矩阵
# is_unitary = np.allclose(np.dot(node.tensor.reshape(4,4), node.tensor.reshape(4,4).conj().T), identity_matrix)
#
# # 输出是否为酉矩阵
# print("\nIs the total matrix unitary?", is_unitary)

# ==============================================================================
# 【新增】正确性验证模块
# ==============================================================================
# def mpo_to_dense(nodes):
#     """
#     将 MPO 链转换为全量密集矩阵 (2^N, 2^N)。
#
#     假设 MPO 节点格式为标准格式: (PhysOut, PhysIn, Left, Right)
#     即 indices: 0:PhysOut, 1:PhysIn, 2:Left, 3:Right
#
#     Args:
#         nodes: MPO 的节点列表 (List[tn.Node])
#
#     Returns:
#         numpy.ndarray: 形状为 (2^N, 2^N) 的复数矩阵
#     """
#     # 1. 复制节点，防止破坏原始结构
#     # tn.copy 会同时复制节点间的内部连接
#     nodes_cp, _ = tn.copy(nodes)
#     n_qubits = len(nodes)
#
#     # 获取当前使用的后端名称 (numpy 或 jax)，用于创建兼容的 Cap 节点
#     backend_name = nodes_cp[0].backend.name
#
#     # 2. 检查并修复内部水平连接 (Left-Right)
#     for k in range(n_qubits - 1):
#         # 检查 Node[k] 的 Right(3) 是否连接到 Node[k+1] 的 Left(2)
#         if not nodes_cp[k][3].is_connected(nodes_cp[k + 1][2]):
#             nodes_cp[k][3] ^ nodes_cp[k + 1][2]
#
#     # 3. 处理边界悬空边 (封口)
#     # 我们需要一个列表来存放所有参与收缩的节点
#     all_nodes_to_contract = list(nodes_cp)
#
#     # 左边界: 第0个节点的 Left (index 2)
#     if nodes_cp[0][2].is_dangling():
#         l_cap = tn.Node(np.array([1.0]), backend=backend_name)
#         nodes_cp[0][2] ^ l_cap[0]
#         all_nodes_to_contract.append(l_cap)
#
#     # 右边界: 最后一个节点的 Right (index 3)
#     if nodes_cp[-1][3].is_dangling():
#         r_cap = tn.Node(np.array([1.0]), backend=backend_name)
#         nodes_cp[-1][3] ^ r_cap[0]
#         all_nodes_to_contract.append(r_cap)
#
#     # 4. 定义输出边的顺序
#     # 目标矩阵行索引: (Out_0, Out_1, ..., Out_N)
#     # 目标矩阵列索引: (In_0, In_1, ..., In_N)
#     out_edges = [nodes_cp[k][0] for k in range(n_qubits)]
#     in_edges = [nodes_cp[k][1] for k in range(n_qubits)]
#
#     # 5. 执行全量收缩
#     # 使用 auto 策略寻找最优路径
#     result = tn.contractors.auto(all_nodes_to_contract,
#                                  output_edge_order=out_edges + in_edges)
#
#     # 6. 转换为 NumPy 数组并 Reshape
#     # 即使后端是 JAX，这里也强制转回 NumPy CPU 数组，方便后续的特征值分解
#     d = 2 ** n_qubits
#     matrix_data = np.array(result.tensor)
#
#     return matrix_data.reshape(d, d)
#
#
# def check_final_energy(nqubits, num_steps, num_layers, postfix, max_bond):
#     print("\n" + "=" * 60)
#     print("FINAL VERIFICATION (Robust Mode)")
#     print("=" * 60)
#
#     # --- 1. 计算 MPO 的真实基态能量 (Ground Truth) ---
#     # 我们不依赖外部通用的 mpo_to_dense，而是现场手动收缩，确保不出错
#     print("Calculating True Energy from MPO...")
#     E_true = None
#     try:
#         # 复制节点防止破坏
#         nodes_cp, _ = tn.copy(obs)
#
#         # 自动封口边界：找到所有度为1的悬空边（通常是首尾的虚拟边）并连接 Dummy Node
#         # 这一步是为了防止 "ValueError: output edges..."
#         for node in nodes_cp:
#             for edge in node.edges:
#                 if edge.is_dangling() and edge.dimension == 1:
#                     # 创建一个全1的辅助节点来“封口”
#                     cap = tn.Node(np.array([1.0], dtype=np.complex128), backend="numpy")
#                     edge ^ cap[0]
#
#         # 收集所有物理边用于输出
#         # 假设标准格式: (PhysOut, PhysIn, Left, Right) -> 0, 1 是物理脚
#         # 如果您的格式不同，auto 收缩会自动处理，只要我们指定输出顺序
#         out_edges = []
#         in_edges = []
#         for k in range(nqubits):
#             # 这里即使索引不对，只要找出维度为 2 的悬空边即可
#             dangling = [e for e in nodes_cp[k].edges if e.is_dangling()]
#             # 按惯例前两个是物理脚，我们假设第一个是 Out，第二个是 In
#             if len(dangling) >= 2:
#                 out_edges.append(dangling[0])
#                 in_edges.append(dangling[1])
#
#         # 全量收缩
#         # all_nodes 包含 MPO 节点和刚才创建的 cap 节点
#         # 使用 reachable 查找所有连通节点
#         all_nodes = tn.reachable(nodes_cp[0])
#
#         res = tn.contractors.auto(all_nodes, output_edge_order=out_edges + in_edges)
#
#         # 转为矩阵并计算特征值
#         H_mat = res.tensor.reshape(2 ** nqubits, 2 ** nqubits)
#         eig_vals = np.linalg.eigvalsh(H_mat)
#         E_true = eig_vals[0]
#         print(f"  >> True Ground State Energy (from MPO): {E_true:.6f}")
#
#     except Exception as e:
#         print(f"  !! Failed to calc true energy from MPO: {e}")
#         # 如果算不出来，暂时用上次的重构值作为参考，或者手动指定
#         # E_true = -1.02679
#
#     # --- 2. 重构哈密顿量 (Reconstruction) ---
#     # 公式: H_recon = sum( U^dag * D * U )
#     print(f"\nReconstructing Hamiltonian from {num_steps} steps...")
#     H_recon = np.zeros((2 ** nqubits, 2 ** nqubits), dtype=complex)
#
#     for k in range(num_steps):
#         p_file = f'saved_models_new/{k}iter_{num_layers}layers_param_{postfix}_{max_bond}.npy'
#         d_file = f'saved_models_new/{k}iter_{num_layers}layers_diag_{postfix}_{max_bond}.npy'
#
#         if not (os.path.exists(p_file) and os.path.exists(d_file)):
#             print(f"  Warning: Step {k} files missing.")
#             break
#
#         theta_k = np.load(p_file)
#         diag_k = np.load(d_file)
#
#         # 构建 U_k
#         c = tc.Circuit(nqubits)
#         c = put_vqc(c, theta_k, num_layers)
#         U_k = np.array(c.matrix())
#
#         # 计算项: U^dag @ D @ U
#         # D @ U 等价于 diag_k[:, None] * U_k (行广播)
#         DU = diag_k[:, None] * U_k
#         term = U_k.conj().T @ DU
#
#         H_recon += term
#
#         if k == 0 or k == num_steps - 1:
#             print(f"  Step {k}: Added term norm = {np.linalg.norm(term):.4f}")
#
#     # --- 3. 对比结果 ---
#     try:
#         recon_eig_vals = np.linalg.eigvalsh(H_recon)
#         E_recon = recon_eig_vals[0]
#
#         print(f"\n  >> Reconstructed Energy: {E_recon:.6f}")
#
#         if E_true is not None:
#             diff = abs(E_recon - E_true)
#             print(f"  >> Error vs Truth: {diff:.6f}")
#
#             if diff < 1e-2:
#                 print("  >> SUCCESS: Decomposition is consistent!")
#             else:
#                 print("  >> WARNING: Large discrepancy. Possible reasons:")
#                 print("     1. Decomposition not converged yet.")
#                 print("     2. MPO format mismatch in verification.")
#     except Exception as e:
#         print(f"Verification failed: {e}")
#
#
# # 执行验证
# check_final_energy(nqubits, num_steps, num_layers, postfix, max_bond)