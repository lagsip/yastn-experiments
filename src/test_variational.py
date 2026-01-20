import numpy as np
import yastn.tn.mps as mps
from yastn import Tensor, ncon
from yastn.operators._spin12 import Spin12
import yastn_lenv_ext.tn.mps._generator_class as gen_mps
import yastn_lenv_ext.tn.mps._env as env
import yastn_lenv_ext.tn.mps._dmrg as dmrg

def rho_test(ops, N, dir='x'):
    """
    Generate initial state ``rho" as eigenstate of direction ``dir" 
    and get identity matrix ``id" for calculating overlaps. 

    Returns:
        rho, id:  mps.Mps
    """
    
    # get direction of the initial state
    if dir == 'x':
        vec = ops.vec_x(1)
    elif dir == 'z':
        vec = ops.vec_z(1)

    # generate rho
    vec_list = []
    for _ in range(N):
        vec_list.append(vec)
        vec_list.append(vec.conj())
    rho = mps.product_mps(vec_list)

    # construct identity matrix
    t1 = Tensor(config=vec.config, s=rho[0].s)
    t1.set_block(Ds=(1,2,2), val=[[1,0],[0,1]])
    t2 = Tensor(config=vec.config, s=rho[1].s)
    t2.set_block(Ds=(2,2,1), val=[[1,0],[0,1]])
    id = mps.Mps(2*N)
    for n in range(N):
        id[2*n] = t1
        id[2*n+1] = t2
    
    return rho, id

def time_evolve(rho, L, tmax, dt):
    """
    Time evolution of the ``rho" under lindbladian ``L". 
    Evolution should be understood as: ``rho(t) = expm(time * L) @ rho(0)"
    """
    times = np.arange(0, tmax+dt/2, dt)
    method = '2site'
    opts_svd = {"tol": 1e-10, "D_total": 32}
    opts_expmv = {"tol": 1e-12}
    for step in mps.tdvp_(rho, L, times=times, method=method, u=-1, dt=dt,
                            opts_svd=opts_svd, opts_expmv=opts_expmv):
        yield step

def lindblad_mpo_latex(ops, ltx_str, parameters, config_kwargs = {"backend": "np"}, 
                       sym='dense'):
    """
    Creates an Mpo object for the lindbladian with the density matrix extracted
    """
    N = parameters["N"]
    # input operators
    generate = gen_mps.GenericGenerator(2*N, ops, debug=False)
    return generate.mpo_from_latex(ltx_str, parameters=parameters, ignore_i=False, rho2ketbra=True)

def technical_test():
    # tensor settings
    config_kwargs = {"backend": "np"}
    sym = 'dense'
    # model settings
    N = 8
    h = np.ones(N) * 0 #np.random.rand(N)
    gamma = np.diag(np.ones(N) * 10)#np.random.rand(N,N)#np.ones([N, N]) * 1
    
    # generate lindbladian
    ops = Spin12(sym=sym, **config_kwargs)
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
    I = mps.product_mpo(ops.I(), N)
    A = mps.random_mpo(I, D_total=1)
    tmp = A[0].remove_leg(axis=0).remove_leg(axis=1).to_numpy()
    print(tmp @ tmp.conj().T)
    
    # initiate environment
    my_env = env.Env_double_lindblad(A,LL,A)
    my_env.setup_(to='first')
    my_env.setup_(to='last')
    
    E_old = my_env.measure().item().real
    print(E_old)
    
    for n in range(A.N):
        initA = A[n] # guess tensor of the initial guess

        # case A: calculate AdagA
        AdagA = initA.tensordot(initA.conj(), axes=((3,),(3,)))
        AdagA = AdagA.transpose(axes=(3,0,1,5,2,4,))
        # test calculate Heff1
        my_env.Heff1(AdagA, n)
        assert (AdagA.s == my_env.Heff1(AdagA, n).s)
        
        # case B: calculate product
        AdagA = ncon([initA, initA.conj()], [(-1,-2,-3,-4), (-5,-8,-7,-6)])
        # test calculate Heff1
        my_env.Heff1(AdagA, n)

        assert (AdagA.s == my_env.Heff1(AdagA, n).s)

    dmrg._dmrg_sweep_1site_(my_env, opts_eigs=None, Schmidt=None, precompute=False, case="B")
    
    for step in dmrg.dmrg_(A, LL, max_sweeps=10, iterator_step=1):
        print(step.sweeps, step.energy, np.array(A.get_entropy()).max() ,"\n")
    tmp = A[0].remove_leg(axis=0).remove_leg(axis=1).to_numpy()
    print(tmp @ tmp.conj().T)

def physical_test_dmrg(ltx_str):
    # tensor settings
    config_kwargs = {"backend": "np"}
    sym = 'dense'
    # model settings
    N = 8
    h = np.ones(N) * 1 #np.random.rand(N)
    gamma = np.diag(np.ones(N) * 1)#np.random.rand(N,N)#np.ones([N, N]) * 1
    
    # generate lindbladian
    ops = Spin12(sym=sym, **config_kwargs)
    # TODO: is is possible to define sigma^*/- in the generator for the lindbladian.
    # TODO what about the operator order???
    parameters = {"gamma": gamma,
                "h": h,
                "N": N,
                "imun": 1j,
                "1.0div2.0": 1/2,
                "Nx": [(i, 2*i, 2*i+1) for i in range(N)],
                "NxN": [(i, j, 2*i, 2*i+1, 2*j, 2*j+1) for i in range(N) for j in range(N)]
                }
    L = lindblad_mpo_latex(ops, ltx_str, parameters)

    # square the operator
    Ldag = L.conjugate_transpose()
    LL = Ldag @ L
    LL.canonize_(to="first", normalize=False)
    LL.truncate_(to="last", opts_svd={"tol": 1e-9, "D_total": 32}, normalize=False)
    print("LL after trunc:", LL.get_bond_dimensions())
    
    # initial guess of the solution
    I = mps.product_mpo(ops.I(), N)
    A = I#mps.random_mpo(I, D_total=1)
    
    # measure
    ltx_str = r"\sum_{j=0}^{N-1} I_j \rho"
    id = lindblad_mpo_latex(ops, ltx_str, {"N": N}) / N
    ltx_str = r"\sum_{j=0}^{N-1} \sigma_j^z \rho"
    Mz = lindblad_mpo_latex(ops, ltx_str, {"N": N}) / N
    ltx_str = r"\sum_{j=0}^{N-1} \sigma_j^x \rho"
    Mx = lindblad_mpo_latex(ops, ltx_str, {"N": N}) / N

    # dmrg
    for step in dmrg.dmrg_(A, LL, max_sweeps=10, iterator_step=1):
        norm = env.Env_double_lindblad(A,id,I).setup_(to='first').measure().item()
        mz = env.Env_double_lindblad(A,Mz,I).setup_(to='first').measure().item() / norm
        mx = env.Env_double_lindblad(A,Mx,I).setup_(to='first').measure().item() / norm
        print(step.sweeps, step.energy, 'EE', np.array(A.get_entropy()).max(), "norm: ", norm, "Mz/Mx: ", mz, "/", mx)

def physical_test_tdvp(ltx_str):
    # tensor settings
    config_kwargs = {"backend": "np"}
    sym = 'dense'
    # model settings
    N = 8
    h = np.ones(N) * 1 #np.random.rand(N)
    gamma = np.diag(np.ones(N) * 1)#np.random.rand(N,N)#np.ones([N, N]) * 1
    
    # generate lindbladian
    ops = Spin12(sym=sym, **config_kwargs)
    # TODO: is is possible to define sigma^*/- in the generator for the lindbladian.
    # TODO what about the operator order???
    parameters = {"gamma": gamma,
                "h": h,
                "N": N,
                "imun": 1j,
                "1.0div2.0": 1/2,
                "Nx": [(i, 2*i, 2*i+1) for i in range(N)],
                "NxN": [(i, j, 2*i, 2*i+1, 2*j, 2*j+1) for i in range(N) for j in range(N)]
                }
    L = lindblad_mpo_latex(ops, ltx_str, parameters)

    # measure
    ltx_str = r"\sum_{j=0}^{N-1} \sigma_j^z \rho"
    Mz = lindblad_mpo_latex(ops, ltx_str, {"N": N}) / N
    ltx_str = r"\sum_{j=0}^{N-1} \sigma_j^x \rho"
    Mx = lindblad_mpo_latex(ops, ltx_str, {"N": N}) / N

    lind, dir = L, 'z'
    print("Initial state is aligned with the polatization", dir)
    rho, id = rho_test(ops, N, dir)
    tmax, dt = 3, 0.1
    rho = id.copy()

    tol= 1e-12
    for step in time_evolve(rho, lind, tmax, dt):
        norm = mps.measure_overlap(rho, id)
        
        drho = lind @ rho
        tmp = mps.measure_overlap(drho, id) / norm
        assert abs(tmp) < tol # tr(drho) == 0 always
        
        tmp = mps.measure_overlap(drho, drho) # tr(rho Ldag L rho) should go to zero with time
        conv = abs(tmp)

        mz = mps.measure_mpo(rho, Mz, id) / norm # average magnetization
        mx = mps.measure_mpo(rho, Mx, id) / norm # average magnetization
        
        tmp = mz if dir == 'x' else mx

        print("time: ", round(step.ti, 2), round(step.tf, 2), 
              'EE', np.array(rho.get_entropy()).max(), "norm: ", norm, "Mz/Mx: ", mz, "/", mx, 
              "conv: ", conv)

if __name__ == '__main__':
    technical_test()
    ltx_str = r"\sum_{j,k,jk,jb,kk,kb \in NxN} gamma_{j,k} (sp_{jk} smcc_{kb} - 1.0div2.0 ( sm_{kk} sp_{jk} + spcc_{kb} smcc_{jb} ) )" 
    physical_test_tdvp(ltx_str)
    physical_test_dmrg(ltx_str)
