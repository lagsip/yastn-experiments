import numpy as np
from yastn.tn import mps
import assets.tn.mps._generator_class as gen_mps
import yastn.operators._spin12 as spin_ops

def get_primitive(parameters):
    N = parameters["N"]
    gamma = parameters["gamma"]
    assert len(gamma) == N
    # input operators
    ops = spin_ops.Spin12(sym=sym, **config_kwargs)
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
    #for c in Hterms:
    #    print("\n", c.amplitude * c.operators[0].to_numpy())
    return mps.generate_mpo(Id, Hterms)

def lindblad_mpo_latex(ltx_str, parameters, config_kwargs = {"backend": "np"}, 
                       sym='dense'):
    """
    Creates an Mpo object for the lindbladian with the density matrix extracted
    """
    N = parameters["N"]
    assert len(parameters["gamma"]) == N
    # input operators
    ops = spin_ops.Spin12(sym=sym, **config_kwargs)
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

    # get from primitive approach
    parameters = {"gamma": gamma,
                    "h": h,
                    "N": N}
    ref = get_primitive(parameters)
    # test
    ltx_str = r"-i (\sum_{j=0}^{N-1} h_{j} ([\sigma_j^z, \rho])) + \sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j}^{z} \rho \sigma_{k}^{z} - \frac{1}{2} \{ \sigma_{k}^{z} \sigma_{j}^{z}, \rho \} )"
    parameters = {"gamma": gamma,
                    "h": h,
                    "N": N}
    mpo1 = lindblad_mpo_latex(ltx_str, parameters)
    # test TODO: identify and fix: Recurssion error. 
    #ltx_str = r"-i {\sum_{j=0}^{N-1} h_{j} (\sigma_{j,ket}^{z} - \sigma_{j,bra}^{z})} + {\sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j,ket}^{z} \sigma_{k,bra}^{z} - \frac{1}{2} ( \sigma_{k,ket}^{z} \sigma_{j,ket}^{z} + \sigma_{k,bra}^{z} \sigma_{j,bra}^{z} ) )}"
    #parameters = {"gamma": gamma,
    #              "h": h,
    #                "N": N}
    #mpo2 = lindblad_mpo_latex(ltx_str, parameters)
    # test
    ltx_str = r"-imun (\sum_{j,jk,jb \in Nx} h_{j} (z_{jk} - zcc_{jb})) + \sum_{j,k,jk,jb,kk,kb \in NxN} gamma_{j,k} (z_{jk} zcc_{kb} - 1.0div2.0 ( z_{kk} z_{jk} + zcc_{kb} zcc_{jb} ) )"
    parameters = {"gamma": gamma,
                "h": h,
                "N": N,
                "imun": 1j,
                "1.0div2.0": 1/2,
                "Nx": [(i, 2*i, 2*i+1) for i in range(N)],
                "NxN": [(i, j, 2*i, 2*i+1, 2*j, 2*j+1) for i in range(N) for j in range(N)]
                }
    mpo3 = lindblad_mpo_latex(ltx_str, parameters)
    
    # crosscheck between test cases
    tol = 1e-12
    # check by ||Ldag L||^2 norm, should be the same
    assert abs(mpo1.norm() - ref.norm()) < tol
    assert abs(mpo1.norm() - ref.norm()) < tol
    # check by ||Ldag(a) L(b)||^2, should be the same if L(a) == L(b)
    tmp = mps.measure_overlap(mpo1, ref)
    assert abs(tmp / ref.norm()**2 - 1) < tol
    tmp = mps.measure_overlap(mpo3, ref)
    assert abs(tmp / ref.norm()**2 - 1) < tol
