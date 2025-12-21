import numpy as np
from yastn.tn import mps
import yastn
import assets.tn.mps._generator_class as gen_mps
import yastn.operators._spin12 as spin_ops


def rho_test(N, dir='x'):
    ops = spin_ops.Spin12(sym=sym, **config_kwargs)
    
    if dir == 'x':
        vec = ops.vec_x(1)
    elif dir == 'z':
        vec = ops.vec_z(1)

    vec_list = []
    for _ in range(N):
        vec_list.append(vec)
        vec_list.append(vec.conj())
    rho = mps.product_mps(vec_list)

    t1 = yastn.Tensor(config=vec.config, s=rho[0].s)
    t1.set_block(Ds=(1,2,2), val=[[1,0],[0,1]])
    t2 = yastn.Tensor(config=vec.config, s=rho[1].s)
    t2.set_block(Ds=(2,2,1), val=[[1,0],[0,1]])
    id = mps.Mps(2*N)
    for n in range(N):
        id[2*n] = t1
        id[2*n+1] = t2
    return rho, id

def get_primitive(parameters):
    N = parameters["N"]
    gamma = parameters["gamma"]
    assert len(gamma) == N
    # input operators
    ops = spin_ops.Spin12(sym=sym, **config_kwargs)
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

def lindblad_mpo_latex(ltx_str, parameters, config_kwargs = {"backend": "np"}, 
                       sym='dense'):
    """
    Creates an Mpo object for the lindbladian with the density matrix extracted
    """
    N = parameters["N"]
    # input operators
    ops = spin_ops.Spin12(sym=sym, **config_kwargs)
    generate = gen_mps.GenericGenerator(2*N, ops, debug=False)
    return generate.mpo_from_latex(ltx_str, parameters=parameters, ignore_i=False, rho2ketbra=True)

def time_evolve(rho, L, tmax, dt):
    times = np.arange(0, tmax+dt/2, dt)
    method = '2site'
    opts_svd = {"tol": 1e-10, "D_total": 32}
    opts_expmv = {"tol": 1e-12}
    for step in mps.tdvp_(rho, L, times=times, method=method, u=-1, dt=dt,
                            opts_svd=opts_svd, opts_expmv=opts_expmv):
        yield step

if __name__ == '__main__':
    # tensor settings
    config_kwargs = {"backend": "np"}
    sym = 'dense'
    # model settings
    N, temp = 2, 1
    h = np.ones(N) * 0
    gamma = np.diag(np.ones(N)) * 1

    # get from primitive approach
    parameters = {"gamma": gamma,
                    "h": h,
                    "N": N}
    ref = get_primitive(parameters)

    # test
    ltx_str = r"-i (\sum_{j=0}^{N-1} h_{j} ([\sigma_j^x, \rho])) + \sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j}^{z} \rho \sigma_{k}^{z} - \frac{1}{2} \{ \sigma_{k}^{z} \sigma_{j}^{z}, \rho \} )"
    parameters = {"gamma": gamma,
                    "h": h,
                    "N": N}
    mpo1 = lindblad_mpo_latex(ltx_str, parameters)

    # measure
    ltx_str = r"\sum_{j=0}^{N-1} \sigma_j^z \rho"
    Mz = lindblad_mpo_latex(ltx_str, {"N": N}) / N
    ltx_str = r"\sum_{j=0}^{N-1} \sigma_j^x \rho"
    Mx = lindblad_mpo_latex(ltx_str, {"N": N}) / N

    lind = mpo1
    rho, id = rho_test(N, dir = 'x')
    tmax, dt = 1, 0.1

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
        
        print("time: ", round(step.ti, 2), round(step.tf, 2), 
              "M: ", round(np.real(mz), 4), round(np.real(mx), 4),
              "conv: ", conv)
