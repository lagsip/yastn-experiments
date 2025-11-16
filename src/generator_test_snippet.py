import numpy as np
import yastn
import assets.tn.mps._generator_class as gen_mps
import assets.operators._spin12 as spin_ops

config_kwargs = {"backend": "np"}


def mpo_nn_hopping_latex(config_kwargs, sym='U1', N=9, t=1, mu=1):
    """
    Nearest-neighbor hopping Hamiltonian on N sites
    with hopping amplitude t and chemical potential mu.
    """
    ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs)

    Hstr = r"\sum_{j,k \in NN} t (cp_{j} c_{k}+cp_{k} c_{j})"
    Hstr += r" + \sum_{i \in sites} mu cp_{i} c_{i}"
    parameters = {"t": t,
                  "mu": mu,
                  "sites": list(range(N)),
                  "NN": [(i, i+1) for i in range(N-1)]}

    print(Hstr)

    generate = gen_mps.GenericGenerator(N, ops)
    mps = generate.mps_from_latex(Hstr, parameters=parameters)
    return mps

def lindblad_mpo_latex(config_kwargs, sym='dense', N=4, gamma=np.ones([4,4])):
    """
    Creates an Mpo object for the lindbladian with the density matrix extracted
    """
    ops = spin_ops.Spin12(sym=sym, **config_kwargs)

    Hstr = r"\sum_{j,k \in NN} t (cp_{j} c_{k}+cp_{k} c_{j})"
    Hstr += r" + \sum_{i \in sites} mu cp_{i} c_{i}"
    ltx_str_full = r"-i {\sum_{j=0}^{N-1} ([\sigma_{j}^{z}, \rho])} + {\sum_{i,j = 0}^{N-1} \gamma_{i,j} (\sigma_{i}^{z} \rho \sigma_{j}^{z} - \frac{1}{2} \{ \sigma_{j}^{z} \sigma_{i}^{z}, \rho \} )}"
    ltx_str_alt = r"-i {\sum_{j=0}^{N-1} (\sigma_{j,ket}^{z} - \sigma_{j,bra}^{z})} + {\sum_{i,j = 0}^{N-1} \gamma_{i,j} (\sigma_{i,ket}^{z} \sigma_{j,bra}^{z} - \frac{1}{2} ( \sigma_{j,ket}^{z} \sigma_{i,ket}^{z} + \sigma_{j,bra}^{z} \sigma_{i,bra}^{z} ) )}"
    ltx_str_simple = r"negimun (\sum_{j \in N} (z_{j} - zcc_{j})) + \sum_{i,j \in NN} gamma_{i,j} (z_{i} zcc_{j} - 1div2 ( z_{j} z_{i} + zcc_{j} zcc_{i} ) )"
    parameters = {"negimun": -1j,
                  "gamma": gamma,
                  "1div2": 1/2,
                  "N": list(range(N)),
                  "NN": [(i, j) for i in range(N) for j in range(N)]}

    print(ltx_str_simple)
    print(parameters)

    generate = gen_mps.GenericGenerator(N, ops)
    mps = generate.mps_from_latex(ltx_str_full, parameters=parameters)
    return mps

print("Hej")
print(lindblad_mpo_latex(config_kwargs=config_kwargs))
