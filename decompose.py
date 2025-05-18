import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable GPU

import tensorcircuit as tc
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import numpy as np
from jax.example_libraries import optimizers
tc.set_backend('jax')
K = tc.backend

# user specified

postfix = 'sparse_hamiltonian_4_5'

# nqubits = len(l[0])
# l = np.array(l)
# weights = np.array(weights)

nqubits = 4 # fixed for convience now

num_steps_range = range(20)
num_layers_range = [5]

# init_obs = np.zeros((2**nqubits, 2**nqubits))
# init_obs[:,0] = np.loadtxt('slater/Slater_InputState.txt', dtype=complex)
# with open('slater/Slater_InputState.txt') as file:
#     lines = file.readlines()
#     for i, line in enumerate(lines):
#         init_obs[i,:] = np.array([complex(x) for x in line.split(',')])

init_obs = np.load("sparse_hamiltonian_4_5.npy")

def put_vqc(c, param, nlayers):
    for j in range(nlayers):
        if j!=0:
            for i in range(j%2,nqubits,2):
                c.unitary(i,(i+1)%nqubits, unitary=np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]]), name="iswap")
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
        if j!=0:
            for i in reversed(range(j%2,nqubits,2)):
                c.unitary(i,(i+1)%nqubits, unitary=np.array([[1,0,0,0],[0,0,-1j,0],[0,-1j,0,0],[0,0,0,1]]), name="iswap-inv")
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

def step(obs, nlayers, nsteps, maxiter=1000):
    # key = jax.random.PRNGKey(42)
    param = K.random_uniform((nlayers,nqubits,3), (-jnp.pi*init_scale, jnp.pi*init_scale))
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)
    opt_state = opt_init(param)

    def update(i, opt_state):
        param = get_params(opt_state)
        (value, gradient) = loss_vag(param, obs, nlayers)
        #print(gradient)
        return value, opt_update(i, gradient, opt_state)

    for i in range(maxiter):
        value, opt_state = update(i, opt_state)
        param = get_params(opt_state)
        # if i%200==0 or i==maxiter-1:
        #     print(i, value)
        if i%200==0 or i==maxiter-1:
            print(nlayers, i, value)

    # save circuits and diag parts
    saved_circ = tc.Circuit(nqubits)
    put_vqc(saved_circ, param, nlayers)
    with open(f'saved_models/{nsteps}iter_{nlayers}layers_{postfix}.qasm', 'w+') as file:
        file.write(saved_circ.to_openqasm())
    np.save(f'saved_models/{nsteps}iter_{nlayers}layers_param_{postfix}.npy', np.array(param))
    np.save(
        f'saved_models/{nsteps}iter_{nlayers}layers_diag_{postfix}.npy',
        np.real(np.array(jnp.diag(transfer(param, obs, nlayers)))))
    return value, transfer_inv(param, off_diag(transfer(param, obs, nlayers)), nlayers)


loss_results = np.zeros((len(num_layers_range), len(num_steps_range)))

# from concurrent.futures import ThreadPoolExecutor

for j, nlayers in enumerate(num_layers_range):
    obs = jnp.copy(init_obs)
    nlayers = list(num_layers_range)[j]
    print(f'nlayers={nlayers}')
    for i, nsteps in enumerate(num_steps_range):
        value, obs = step(obs, nlayers, nsteps, maxiter=1000)
        print(f'nlayers={nlayers} i={i} nsteps={nsteps} value={value}') # trouble maker
        # print(f'value={value}')
        loss_results[j, i] = value
        # if value < 1e-4:
        #     break

# with ThreadPoolExecutor() as executor:
#     executor.map(process_nlayers, range(len(num_layers_range)))
# import time
#
# for j, nlayers in enumerate(num_layers_range):
#     obs = jnp.copy(init_obs)
#     print(f'nlayers={nlayers}')
#     for i, nsteps in enumerate(num_steps_range):
#         start_time = time.time()
#         print(f'i={i}')
#         value, obs = step(obs, nlayers, nsteps, maxiter=600)
#         print(f'value={value}')
#         print(f'step time cost: {time.time() - start_time}')
#         #print(f'obs={obs}')
#         loss_results[j,i] = value
#         if value<1e-4:
#             break
#
# np.savetxt(f'loss_results_{postfix}.txt', loss_results)
#
# # uncomment to load the saved results
# # import numpy as np
# # loss_results = np.loadtxt(f'loss_results_{postfix}.txt')
#
# import matplotlib.pyplot as plt
#
# for j,nlayers in enumerate(num_layers_range):
#     plt.plot(list(num_steps_range), loss_results[j], label=f'{nlayers} layers')
# plt.xlabel('Number of steps')
# plt.ylabel('Loss')
# plt.title(postfix)
# plt.legend()
# plt.savefig(f'loss_results_{postfix}.png')
