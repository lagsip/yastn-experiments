import numpy as np
import yastn.tn.mps as mps_fun
import yastn_lenv_ext.tn.mps._generator_class as gen_mps
import yastn.operators._spin12 as spin_ops
import yastn_lenv_ext.tn.mps._env as env

def lindblad_mpo_latex(ops, ltx_str, parameters, config_kwargs = {"backend": "np"}, 
                       sym='dense'):
    """
    Creates an Mpo object for the lindbladian with the density matrix extracted
    """
    N = parameters["N"]
    assert len(parameters["gamma"]) == N
    # input operators
    generate = gen_mps.GenericGenerator(2*N, ops, debug=False)
    return generate.mpo_from_latex(ltx_str, parameters=parameters, ignore_i=False, rho2ketbra=True)

if __name__ == '__main__':
    # tensor settings
    config_kwargs = {"backend": "np"}
    sym = 'dense'
    # model settings
    N = 2
    h = np.random.rand(N)
    gamma = np.random.rand(N,N)#np.ones([N, N]) * 1

    # generate lindbladian
    ops = spin_ops.Spin12(sym=sym, **config_kwargs)
    ltx_str = r"-i (\sum_{j=0}^{N-1} h_{j} ([\sigma_j^z, \rho])) + \sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j}^{z} \rho \sigma_{k}^{z} - \frac{1}{2} \{ \sigma_{k}^{z} \sigma_{j}^{z}, \rho \} )"
    parameters = {"gamma": gamma,
                    "h": h,
                    "N": N}
    L = lindblad_mpo_latex(ops, ltx_str, parameters)

    # square the operator
    Ldag = L.conjugate_transpose()
    LL = Ldag @ L
    LL.canonize_(to="first", normalize=False)
    LL.truncate_(to="last", opts_svd={"tol": 1e-9, "D_total": 256}, normalize=False)
    print("LL after trunc:", LL.get_bond_dimensions())
    
    # initial guess of the solution
    I = mps_fun.product_mpo(ops.I(), N)
    A = mps_fun.random_mpo(I)
    
    # initiate environment
    my_env = env.Env_double_lindblad(A,LL,A)
    my_env.setup_(to='first')
    my_env.setup_(to='last')

    for n in range(A.N):
        initA = A[n] # guess tensor of the initial guess
        # calculate AdagA
        AdagA = initA.tensordot(initA.conj(), axes=((3,),(3,)))
        AdagA.transpose(axes=(3,0,1,5,2,4,))
        # test calculate Heff1
        my_env.Heff1(AdagA, n)
        