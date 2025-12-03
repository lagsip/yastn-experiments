import numpy as np
import yastn
import assets.tn.mps._generator_class as gen_mps
import yastn.tn.mps as mps_fun
import yastn.operators._spin12 as spin_ops
import assets.tn.mps._env as env

config_kwargs = {"backend": "np"}

ltx_str = r"-i (\sum_{j=0}^{N-1} ([\sigma_j^z, \rho])) + \sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j}^{z} \rho \sigma_{k}^{z} - \frac{1}{2} \{ \sigma_{k}^{z} \sigma_{j}^{z}, \rho \} )"

def lindblad_mpo_latex(config_kwargs, sym='dense', N=4, gamma=np.ones([4,4])):
    """
    Creates an Mpo object for the lindbladian with the density matrix extracted
    """
    ops = spin_ops.Spin12(sym=sym, **config_kwargs)

    ltx_str_full = r"-i (\sum_{j=0}^{N-1} ([\sigma_j^z, \rho])) + \sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j}^{z} \rho \sigma_{k}^{z} - \frac{1}{2} \{ \sigma_{k}^{z} \sigma_{j}^{z}, \rho \} )"
    ltx_str_alt = r"-i {\sum_{j=0}^{N-1} (\sigma_{j,ket}^{z} - \sigma_{j,bra}^{z})} + {\sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j,ket}^{z} \sigma_{k,bra}^{z} - \frac{1}{2} ( \sigma_{k,ket}^{z} \sigma_{j,ket}^{z} + \sigma_{k,bra}^{z} \sigma_{j,bra}^{z} ) )}"
    ltx_str_simple = r"-imun (\sum_{j,jk,jb \in Nx} (z_{jk} - zcc_{jb})) + \sum_{j,k,jk,jb,kk,kb \in NxN} gamma_{j,k} (z_{j} zcc_{k} - 1.0div2.0 ( z_{k} z_{j} + zcc_{k} zcc_{j} ) )"
    parameters = {"\\gamma": gamma,
                  "N": N,
                  # "imun": 1j,
                  # "1.0div2.0": 1/2,
                  # "Nx": [(i,2*i,2*i+1) for i in range(N)],
                  # "NxN": [(i,j,2*i,2*i+1,2*j,2*j+1) for i in range(N) for j in range(N)]
                  }

    # print(ltx_str_full)
    # print(parameters)

    generate = gen_mps.GenericGenerator(2*N, ops, debug=False)
    mpo = generate.mpo_from_latex(ltx_str_full, parameters=parameters, ignore_i=False, rho2ketbra=True)
    return mpo


def test_env(config_kwargs, sym='Z2', N=4):
    
    ops = spin_ops.Spin12(sym=sym, **config_kwargs)
    ltx_str = r"-i (\sum_{j=0}^{N-1} ([\sigma_j^z, \rho])) + \sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j}^{z} \rho \sigma_{k}^{z} - \frac{1}{2} \{ \sigma_{k}^{z} \sigma_{j}^{z}, \rho \} )"
    parameters = {"gamma": np.ones([N,N]),
                  "N": N
                  }
    
    I = mps_fun.product_mpo(ops.I(), N)
    A = mps_fun.random_mpo(I)
    
    I = mps_fun.product_mpo(ops.I(), 2*N)
    L = mps_fun.random_mpo(I)
    #generate = gen_mps.GenericGenerator(2*N, ops)
    #L = generate.mpo_from_latex(ltx_str, parameters=parameters, ignore_i=False, rho2ketbra=True)


    my_env = env.Env_double_lindblad(A,L,A)
    print(my_env.F[-1, 0])
    print(my_env.F[-1, 0].to_numpy())
    assert my_env.F[-1, 0].to_numpy().shape == (1,1,1,1,1)
    assert my_env.F[-1, 0].to_numpy().size == 1

    print(my_env.F[N, N-1])
    print(my_env.F[N, N-1].to_numpy())
    assert my_env.F[N, N-1].to_numpy().shape == (1,1,1,1,1)
    assert my_env.F[N, N-1].to_numpy().size == 1


    #my_env.update_env_(0)
    #my_env.update_env_(1)


def test_env_update(config_kwargs, sym='dense', N=4):

    ops = spin_ops.Spin12(sym=sym, **config_kwargs)
    ltx_str = r"-i (\sum_{j=0}^{N-1} ([\sigma_j^z, \rho])) + \sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j}^{z} \rho \sigma_{k}^{z} - \frac{1}{2} \{ \sigma_{k}^{z} \sigma_{j}^{z}, \rho \} )"
    parameters = {"gamma": np.ones([N,N]),
                  "N": N
                  }
    
    I = mps_fun.product_mpo(ops.I(), N)
    print(I.get_virtual_legs())
    A = mps_fun.random_mpo(I)
    
    generate = gen_mps.GenericGenerator(2*N, ops)
    L = generate.mpo_from_latex(ltx_str, parameters=parameters, ignore_i=False, rho2ketbra=True)
    for n in range(1,2*N,2):
        L[n] = L[n].flip_charges(axes=(1,3))
    my_env = env.Env_double_lindblad(A,L,A)
    # check if edge envs make sense
    assert my_env.F[-1, 0].to_numpy().shape == (1,1,1,1,1)
    assert my_env.F[-1, 0].to_numpy().size == 1
    assert my_env.F[N, N-1].to_numpy().shape == (1,1,1,1,1)
    assert my_env.F[N, N-1].to_numpy().size == 1
    # calculate left-envs step by step
    for n in range(N):
        my_env.update_env_(n, to='last')
    # calculate all left-envs
    my_env.setup_(to='last')
    # TODO: the same for right envs. 
    #for n in range(N-1,-1,-1):
    #    my_env.update_env_(n, to='first')

# lindblad_mpo_latex(config_kwargs=config_kwargs)

# test_env(config_kwargs=config_kwargs)
    
test_env_update(config_kwargs=config_kwargs)
