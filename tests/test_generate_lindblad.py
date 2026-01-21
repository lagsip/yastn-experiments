import numpy as np
from yastn.tn import mps
from yastn.operators._spin12 import Spin12
import varpur.tn.mps._generator_class as gen_mps


def get_primitive(ops, parameters):
    """
    Write the Lindladian by hand using a series of Hterms. 
    """
    N = parameters["N"]
    gamma = parameters["gamma"]
    # input operators
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

def get_primitive_heisenberg(ops, parameters):
    """
    Write the Lindladian by hand using a series of Hterms. 
    """
    N = parameters["N"]
    h = parameters["h"]
    Jxx = parameters["Jxx"]
    Jyy = parameters["Jyy"]
    Jzz = parameters["Jzz"]
    gxx = parameters["gxx"]
    gyy = parameters["gyy"]
    gzz = parameters["gzz"]
    # input operators
    x, y, z, id = ops.x(), ops.y(), ops.z(), ops.I()
    xcc, ycc, zcc, idcc = x.T, y.T, z.T, id.T
    # assemble full lindbladian using primitive Hterm
    Hterms = []
    # UNITARY
    for n in range(N):
        hterm = mps.Hterm( -1j * parameters["h"][n], positions=[2*n], operators=[z])
        Hterms.append(hterm)
        hterm = mps.Hterm( 1j * parameters["h"][n], positions=[2*n+1], operators=[zcc])
        Hterms.append(hterm)
    for i, k in zip(*np.nonzero(Jxx)):
        hterm = mps.Hterm( -1j * Jxx[i,k], positions=[2*i,2*k], operators=[x, x])
        Hterms.append(hterm)
        hterm = mps.Hterm( 1j * Jxx[i,k], positions=[2*i+1,2*k+1], operators=[xcc, xcc])
        Hterms.append(hterm)
    for i, k in zip(*np.nonzero(Jyy)):
        hterm = mps.Hterm( -1j * Jyy[i,k], positions=[2*i,2*k], operators=[y, y])
        Hterms.append(hterm)
        hterm = mps.Hterm( 1j * Jyy[i,k], positions=[2*i+1,2*k+1], operators=[ycc, ycc])
        Hterms.append(hterm)
    for i, k in zip(*np.nonzero(Jzz)):
        hterm = mps.Hterm( -1j * Jzz[i,k], positions=[2*i,2*k], operators=[z, z])
        Hterms.append(hterm)
        hterm = mps.Hterm( 1j * Jzz[i,k], positions=[2*i+1,2*k+1], operators=[zcc, zcc])
        Hterms.append(hterm)
    # DISSIPATION
    for i, j in zip(*np.nonzero(gxx)):
        hterm = mps.Hterm( gxx[i,j], positions=[2*i, 2*j+1], operators=[x, xcc])
        Hterms.append(hterm)
        hterm = mps.Hterm( -1/2 * gxx[i,j], positions=[2*i, 2*j], operators=[x, x])
        Hterms.append(hterm)
        hterm = mps.Hterm( -1/2 * gxx[i,j], positions=[2*i+1, 2*j+1], operators=[xcc, xcc])
        Hterms.append(hterm)
    for i, j in zip(*np.nonzero(gyy)):
        hterm = mps.Hterm( gyy[i,j], positions=[2*i, 2*j+1], operators=[y, ycc])
        Hterms.append(hterm)
        hterm = mps.Hterm( -1/2 * gyy[i,j], positions=[2*i, 2*j], operators=[y, y])
        Hterms.append(hterm)
        hterm = mps.Hterm( -1/2 * gyy[i,j], positions=[2*i+1, 2*j+1], operators=[ycc, ycc])
        Hterms.append(hterm)
    for i, j in zip(*np.nonzero(gzz)):
        hterm = mps.Hterm( gzz[i,j], positions=[2*i, 2*j+1], operators=[z, zcc])
        Hterms.append(hterm)
        hterm = mps.Hterm( -1/2 * gzz[i,j], positions=[2*i, 2*j], operators=[z, z])
        Hterms.append(hterm)
        hterm = mps.Hterm( -1/2 * gzz[i,j], positions=[2*i+1, 2*j+1], operators=[zcc, zcc])
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
    generate = gen_mps.GenericGenerator(2*N, ops, debug=True)
    return generate.mpo_from_latex(ltx_str, parameters=parameters, ignore_i=False, rho2ketbra=True)

def test_transcriptions():
    # tensor settings
    config_kwargs = {"backend": "np"}
    sym = 'dense'
    ops = Spin12(sym=sym, **config_kwargs)
    # model settings
    N = 2
    h = np.random.rand(N)
    gamma = np.random.rand(N,N)

    # get from primitive approach
    parameters = {"gamma": gamma,
                    "h": h,
                    "N": N}
    ref = get_primitive(ops, parameters)
    # test
    ltx_str = r"-i (\sum_{j=0}^{N-1} h_{j} ([\sigma_j^z, \rho])) + \sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j}^{z} \rho \sigma_{k}^{z} - \frac{1}{2} \{ \sigma_{k}^{z} \sigma_{j}^{z}, \rho \} )"
    parameters = {"gamma": gamma,
                    "h": h,
                    "N": N}
    mpo1 = lindblad_mpo_latex(ops, ltx_str, parameters)
    ltx_str = r"-i (\sum_{j=0}^{N-1} h_{j} (\sigma_{j,ket}^z - \sigma_{j,bra}^z)) + (\sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j,ket}^{z} \sigma_{k,bra}^{z} - \frac{1}{2} ( \sigma_{k,ket}^{z} \sigma_{j,ket}^{z} + \sigma_{k,bra}^{z} \sigma_{j,bra}^{z} )))"
    parameters = {"gamma": gamma,
                  "h": h,
                    "N": N}
    mpo2 = lindblad_mpo_latex(ops, ltx_str, parameters)
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
    mpo3 = lindblad_mpo_latex(ops, ltx_str, parameters)
    
    # crosscheck between test cases
    tol = 1e-12
    # check by ||Ldag L||^2 norm, should be the same
    assert abs(mpo1.norm() - ref.norm()) < tol
    assert abs(mpo2.norm() - ref.norm()) < tol
    assert abs(mpo3.norm() - ref.norm()) < tol
    # check by ||Ldag(a) L(b)||^2, should be the same if L(a) == L(b)
    tmp = mps.measure_overlap(mpo1, ref)
    assert abs(tmp / ref.norm()**2 - 1) < tol
    tmp = mps.measure_overlap(mpo2, ref)
    assert abs(tmp / ref.norm()**2 - 1) < tol
    tmp = mps.measure_overlap(mpo3, ref)
    assert abs(tmp / ref.norm()**2 - 1) < tol

def test_models():
    # tensor settings
    config_kwargs = {"backend": "np"}
    sym = 'dense'
    ops = Spin12(sym=sym, **config_kwargs)
    # model settings
    N = 2
    h = np.random.rand(N)
    gamma = np.random.rand(N,N)
    parameters = {"gamma": gamma,
                    "h": h,
                    "N": N}
    # get from primitive approach
    ref = get_primitive(ops, parameters)
    # get from generator
    #ltx_str = r"-i (\sum_{j=0}^{N-1} h_{j} ([\sigma_j^z, \rho])) "
    ltx_str = r"-i (\sum_{j = 0}^{N-1} h_{j} [z_j, \rho]) "
    ltx_str += r"+ (\sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_j^z \rho \sigma_k^z - \frac{1}{2} \{ \sigma_k^z \sigma_j^z, \rho \} ))"
    ltx_str += r"+ (\sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_j^z \rho \sigma_k^z - \frac{1}{2} \{ \sigma_k^z \sigma_j^z, \rho \} ))"
    ltx_str += r"+ (\sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_j^z \rho \sigma_k^z - \frac{-1-1j}{-3+1j} \sigma_k^z \sigma_j^z \rho - \frac{1}{2} \rho \sigma_k^z \sigma_j^z ))"
    mpo1 = lindblad_mpo_latex(ops, ltx_str, parameters)
    exit()

    # crosscheck between test cases
    tol = 1e-12
    # check by ||Ldag L||^2 norm, should be the same
    assert abs(mpo1.norm() - ref.norm()) < tol
    # check by ||Ldag(a) L(b)||^2, should be the same if L(a) == L(b)
    tmp = mps.measure_overlap(mpo1, ref)
    assert abs(tmp / ref.norm()**2 - 1) < tol

def test_heisenberg():
    # tensor settings
    config_kwargs = {"backend": "np"}
    sym = 'dense'
    ops = Spin12(sym=sym, **config_kwargs)
    # model settings
    N = 2
    h = np.random.rand(N)
    J = np.random.rand(N,N)
    J = (J + J.T.conj()) / 2
    Jxx = Jyy = Jzz = J
    gamma = np.random.rand(N,N)
    gamma = gamma @ gamma.T.conj() # has real and positive eigenvalues
    gxx = gyy = gzz = gamma
    parameters = {"N": N,
                  "h": h,
                  "Jxx": Jxx,
                  "Jyy": Jyy,
                  "Jzz": Jzz,
                  "gxx": gxx,
                  "gyy": gyy,
                  "gzz": gzz,
                    }
    # get from primitive approach
    ref = get_primitive_heisenberg(ops, parameters)
    # get from generator
    ltx_str = r"-i (\sum_{j = 0}^{N-1} h_{j} ([\sigma_j^z, \rho])) "
    ltx_str += r"-i (\sum_{j,k = 0}^{N-1} Jxx_{j,k} ([\sigma_j^x \sigma_k^x, \rho])) "
    ltx_str += r"-i (\sum_{j,k = 0}^{N-1} Jyy_{j,k} ([\sigma_j^y \sigma_k^y, \rho])) "
    ltx_str += r"-i (\sum_{j,k = 0}^{N-1} Jzz_{j,k} ([\sigma_j^z \sigma_k^z, \rho])) "
    # TODO the case below also fails, fixed, upload
    #ltx_str += r"+ \sum_{j,k = 0}^{N-1} gxx_{j,k} (\sigma_j^x \rho \sigma_k^x - \frac{1}{2} \sigma_k^x \sigma_j^x \rho  - \frac{1}{2} \rho \sigma_k^x \sigma_j^x )"
    # TODO: multiple dissipators not possible, fixed, upload
    #ltx_str += r"+ \sum_{j,k = 0}^{N-1} gxx_{j,k} (\sigma_j^x \rho \sigma_k^x - \frac{1}{2} \{ \sigma_k^x \sigma_j^x, \rho \} )"
    #ltx_str += r"+ \sum_{j,k = 0}^{N-1} gyy_{j,k} (\sigma_j^y \rho \sigma_k^y - \frac{1}{2} \{ \sigma_k^y \sigma_j^y, \rho \} )"
    ltx_str += r"+ \sum_{j,k = 0}^{N-1} gzz_{j,k} (\sigma_j^z \rho \sigma_k^z - \frac{1}{2} \{ \sigma_k^z \sigma_j^z, \rho \} )"
    mpo1 = lindblad_mpo_latex(ops, ltx_str, parameters)
    print(mpo1)

    # crosscheck between test cases
    #tol = 1e-12
    # check by ||Ldag L||^2 norm, should be the same
    #assert abs(mpo1.norm() - ref.norm()) < tol
    # check by ||Ldag(a) L(b)||^2, should be the same if L(a) == L(b)
    #tmp = mps.measure_overlap(mpo1, ref)
    #assert abs(tmp / ref.norm()**2 - 1) < tol

if __name__ == '__main__':
    #test_transcriptions()
    test_models()
    #test_heisenberg() TODO in prepartion, include Jxy terms to check order on the conjugated space


# This is c2: 
# [single_term(op=(('minus',), ('imun',), ('h', 0), ('z', 'jk'), ('e',), ('n',), ('d',), ('s',), ('u',), ('m',))), single_term(op=(('minus',), ('imun',), ('h', 0), ('minus',), ('zcc', 'jb'), ('e',), ('n',), ('d',), ('s',), ('u',), ('m',))), single_term(op=(('minus',), ('imun',), ('h', 1), ('z', 'jk'), ('e',), ('n',), ('d',), ('s',), ('u',), ('m',))), single_term(op=(('minus',), ('imun',), ('h', 1), ('minus',), ('zcc', 'jb'), ('e',), ('n',), ('d',), ('s',), ('u',), ('m',)))]


# This is c2: 
# [single_term(op=(('minus',), ('1',), ('h', 0), ('z', 0))), single_term(op=(('minus',), ('1',), ('h', 0), ('minus',), ('zcc', 1))), single_term(op=(('minus',), ('1',), ('h', 1), ('z', 2))), single_term(op=(('minus',), ('1',), ('h', 1), ('minus',), ('zcc', 3)))]