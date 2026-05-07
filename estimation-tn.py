import numpy as np                           # 导入numpy库并简写为np
from mindquantum.core.gates import X, H      # 导入量子门H, X
from mindquantum.simulator import Simulator  # 从mindquantum.simulator中导入Simulator类
from mindquantum.core.circuit import Circuit # 导入Circuit模块，用于搭建量子线路
from mindquantum.core.gates import Measure   # 引入测量门

nqubits = 8
iterm = 3.2
trace = []
if iterm < 4.4:
    iterm += 0.2
    # postfix = f'H6_{iterm:.1f}_sto-3g'
    postfix = 'TFIM_8'

    niter = 20
    num_layers = 2
    bond = 64
    # postfix = 'slater_plus_4to8layers_40iter'
    resultI = []
    for iter in range(1):
        niter = 5
        qasm_texts = []
        diags = np.zeros((niter, 2**nqubits), dtype=complex)

        for i in range(niter):
            with open(f'saved_models_new/{i}iter_{num_layers}layers_{postfix}_{bond}.qasm') as file:
                qasm_texts.append(file.read())
            diags[i] = np.load(f'saved_models_new/{i}iter_{num_layers}layers_diag_{postfix}_{bond}.npy')

        def reorder_qubits(x, nqubits):
            return np.transpose(x.reshape([2] * nqubits)).reshape(-1)

        # stvec = np.load(f"test/ground_state_H6_{iterm:.1f}_sto-3g.npy") # this input state is a density matrix
        #
        # H = np.load(f"test/H6_{iterm:.1f}_sto-3g.npy")
        stvec = np.load(f"TFIM_8_ground_state.npy")  # this input state is a density matrix

        H = np.load(f"TFIM_8_hamiltonian.npy")
        ground_trurh_val = stvec.conjugate().T @ H @ stvec
        # print(np.trace(np.load("SparseH_8.npy") @ np.load("ground_state_SparseH_8.npy")))
        print(ground_trurh_val)

        import re
        import math

        def find_eval_replace_expressions(input_string):
            pattern = r'(\d*)/\((\d*)\*pi\)|(\d*)\*pi/(\d*)'
            matches = re.finditer(pattern, input_string)
            updated_string = input_string
            for match in matches:
                num1 = match.group(1) if match.group(1) else match.group(3)
                num2 = match.group(2) if match.group(2) else match.group(4)
                result = eval(f'{num1} / {num2} * math.pi')
                expression = match.group(0)
                updated_string = updated_string.replace(expression, str(result), 1)  # 仅替换第一个匹配项

            return updated_string

        from mindquantum.io import OpenQASM
        estval = 0
        sim = Simulator('mqvector', nqubits)

        for i in range(niter):
            circuit = OpenQASM().from_string(find_eval_replace_expressions(qasm_texts[i]))
            sim.reset()
            sim.set_qs(reorder_qubits(stvec, nqubits))
            sim.apply_circuit(circuit)
            resstvec = sim.get_qs()
            diags_flip = reorder_qubits(diags[i], nqubits)
            estval += resstvec.conjugate() @ np.diag(diags_flip) @ resstvec

        estval_st = estval

        print(estval_st)
        trace.append(np.abs(estval_st-ground_trurh_val))

        importance = np.max(np.abs(diags), axis=1)

        importance /= np.sum(importance)

        import matplotlib.pyplot as plt

        plt.scatter(range(niter), importance)

        def estimate_diag(diag_obs, bit_string_data, nshots):
            x = np.array([[n if (j==int(bs, 2)) else 0 for j in range(len(diag_obs))] for bs, n in (bit_string_data.items())])
            x = np.sum(x, axis=0)
            return np.sum(x * diag_obs) / nshots

        import math

        sim = Simulator('mqvector', nqubits)
        nruns = 1

        def estimtate(nshots_total):
            estval_list_run = np.zeros(nruns, dtype=complex)
            nshots_iter = [math.floor(nshots_total * x) for x in importance]
            for i in range(niter):
                if nshots_iter[i]:
                    circuit = OpenQASM().from_string(find_eval_replace_expressions(qasm_texts[i]))
                    for j in range(nqubits):
                        circuit += Measure(f'q{j}').on(j)
                    diags_flip = reorder_qubits(diags[i], nqubits)
                    for j in range(nruns):
                        sim.reset()
                        sim.set_qs(reorder_qubits(stvec, nqubits))
                        result = sim.sampling(circuit, shots=nshots_iter[i])
                        contribution = estimate_diag(diags_flip, result.bit_string_data, nshots_iter[i])
                        estval_list_run[j] += contribution
            return estval_list_run

        T = [12, 45, 160, 572, 2038,7256,25848,92041]
        output_file = f'greedy_{postfix}.txt'
        for t in T:
            result = []
            for i in range(50):
                estimated_expectation_value = estimtate(t)
                result.append(estimated_expectation_value)
            print("result:", result)
            # 保存原始测量值到文件
            # np.savetxt(f'raw_measurements_H4_{postfix}_2000.txt', result)
            # 计算偏差
            cal = ground_trurh_val
            squared_deviations = [np.abs(x - cal) ** 2 for x in result]
            # 计算标准差
            std_dev = np.sqrt(np.mean(squared_deviations))
            resultI.append(std_dev)

    np.savetxt(f'error_{num_layers}layer_{postfix}.txt', resultI)
# np.savetxt(f'trace_11_10.txt', trace)
# Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot([12,45,160,572,2038,7259,25848], resultI, 'o-')
# plt.xlabel('samples')
# plt.ylabel('error')
# plt.title('error vs. samples')
# plt.grid(True)
# plt.savefig(f'error vs. samples.png')
# plt.show()