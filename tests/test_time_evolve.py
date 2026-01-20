import numpy as np
from yastn import Tensor
from yastn.tn import mps
from yastn.operators._spin12 import Spin12
import varpur.tn.mps._generator_class as gen_mps

def rho_test(ops, N, dir='x'):
    """
    Generate initial state ``rho" as eigenstate of direction ``dir" 
    and get identity matrix ``id" for calculating overlaps. 

    Returns:
        rho, id:  yastn.tn.mps.Mps
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

def get_primitive(ops, parameters):
    """
    Write the Lindladian by hand using a series of Hterms. 
    """
    N = parameters["N"]
    gamma = parameters["gamma"]
    assert len(gamma) == N
    # input operators
    z = ops.z()
    zcc = ops.z().transpose()
    z = ops.z()
    zcc = ops.z().transpose()
    id = ops.I()
    idcc = ops.I().transpose()

    # assemble full lindbladian using primitive Hterm
    Hterms = []
    # unitary term
    for n in range(N):
        hterm = mps.Hterm( -1j * parameters["h"][n], positions=[2*n], operators=[z])
        Hterms.append(hterm)
        hterm = mps.Hterm( 1j * parameters["h"][n], positions=[2*n+1], operators=[zcc])
        Hterms.append(hterm)
    # dissipation
    for i, j in zip(*np.nonzero(gamma)):
        hterm = mps.Hterm( gamma[i,j], positions=[2*i, 2*j+1], operators=[z, zcc])
        Hterms.append(hterm)
        hterm = mps.Hterm( -1/2 * gamma[i,j], positions=[2*i, 2*j], operators=[z, z])
        Hterms.append(hterm)
        hterm = mps.Hterm( -1/2 * gamma[i,j], positions=[2*i+1, 2*j+1], operators=[zcc, zcc])
        Hterms.append(hterm)
    Idlist = [id if n % 2 == 0 else idcc for n in range(2*N)]
    Id = mps.product_mpo(Idlist)
    return mps.generate_mpo(Id, Hterms)

def lindblad_mpo_latex(ops, ltx_str, parameters):
    """
    Creates an Mpo object for the lindbladian with the density matrix extracted
    """
    N = parameters["N"]
    # input operators
    generate = gen_mps.GenericGenerator(2*N, ops, debug=False)
    return generate.mpo_from_latex(ltx_str, parameters=parameters, ignore_i=False, rho2ketbra=True)

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

def test_time_evolution():
    # tensor settings
    config_kwargs = {"backend": "np"}
    sym = 'dense'
    ops = Spin12(sym=sym, **config_kwargs)
    # model settings
    N = 4
    h = np.ones(N) * 1
    gamma = np.diag(np.ones(N)) * 1

    # get from primitive approach
    parameters = {"gamma": gamma,
                    "h": h,
                    "N": N}
    ref = get_primitive(ops, parameters)

    # test
    ltx_str = r"-i (\sum_{j=0}^{N-1} h_{j} ([\sigma_j^x, \rho])) + \sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j}^{z} \rho \sigma_{k}^{z} - \frac{1}{2} \{ \sigma_{k}^{z} \sigma_{j}^{z}, \rho \} )"
    parameters = {"gamma": gamma,
                    "h": h,
                    "N": N}
    mpo1 = lindblad_mpo_latex(ops, ltx_str, parameters)

    # measure
    ltx_str = r"\sum_{j=0}^{N-1} \sigma_j^z \rho"
    Mz = lindblad_mpo_latex(ops, ltx_str, {"N": N}) / N
    ltx_str = r"\sum_{j=0}^{N-1} \sigma_j^x \rho"
    Mx = lindblad_mpo_latex(ops, ltx_str, {"N": N}) / N

    lind, dir = mpo1, 'x'
    print("Initial state is aligned with the polatization", dir)
    rho, id = rho_test(ops, N, dir)
    tmax, dt = 3, 0.1

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
        assert abs(tmp) < tol # orthogonal magnetization always zero

        if dir == 'z':
            assert abs(tmp -  1) < tol # if aligned with 'z', the magnetization stays the same

        if dir == 'z':
            assert abs(conv) < tol # if aligned with 'z', is steady state

        print("time: ", round(step.ti, 2), round(step.tf, 2), 
              "M: ", round(np.real(mz), 4), round(np.real(mx), 4),
              "conv: ", conv)


if __name__ == '__main__':
    test_time_evolution()