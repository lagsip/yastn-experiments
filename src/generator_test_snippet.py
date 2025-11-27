import numpy as np
import yastn
import assets.tn.mps._generator_class as gen_mps
import yastn.operators._spin12 as spin_ops

config_kwargs = {"backend": "np"}

def lindblad_mpo_latex(config_kwargs, sym='dense', N=4, gamma=np.ones([4,4])):
    """
    Creates an Mpo object for the lindbladian with the density matrix extracted
    """
    ops = spin_ops.Spin12(sym=sym, **config_kwargs)

    ltx_str_full = r"-i (\sum_{j=0}^{N-1} ([\sigma_j^z, \rho])) + \sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j}^{z} \rho \sigma_{k}^{z} - \frac{1}{2} \{ \sigma_{k}^{z} \sigma_{j}^{z}, \rho \} )"
    ltx_str_alt = r"-i {\sum_{j=0}^{N-1} (\sigma_{j,ket}^{z} - \sigma_{j,bra}^{z})} + {\sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j,ket}^{z} \sigma_{k,bra}^{z} - \frac{1}{2} ( \sigma_{k,ket}^{z} \sigma_{j,ket}^{z} + \sigma_{k,bra}^{z} \sigma_{j,bra}^{z} ) )}"
    ltx_str_simple = r"-imun (\sum_{j,jk,jb \in Nx} (z_{jk} - zcc_{jb})) + \sum_{j,k,jk,jb,kk,kb \in NxN} gamma_{j,k} (z_{j} zcc_{k} - 1.0div2.0 ( z_{k} z_{j} + zcc_{k} zcc_{j} ) )"
    parameters = {"imun": 1j,
                  "gamma": gamma,
                  "1.0div2.0": 1/2,
                  "N": N,
                #   "Nx": [(i,2*i,2*i+1) for i in range(N)],
                #   "NxN": [(i,j,2*i,2*i+1,2*j,2*j+1) for i in range(N) for j in range(N)]
                  }

    print(ltx_str_full)
    print(parameters)

    generate = gen_mps.GenericGenerator(2*N, ops)
    mpo = generate.mps_from_latex(ltx_str_full, parameters=parameters, ignore_i=False)
    return mpo

print("Hej")
print(lindblad_mpo_latex(config_kwargs=config_kwargs))
