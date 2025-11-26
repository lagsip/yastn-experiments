import numpy as np
import yastn
import assets.tn.mps._generator_class as gen_mps
import yastn.operators._spin12 as spin_ops

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
    ltx_str_full = r"-i (\sum_{j=0}^{N-1} ([\sigma_{j}^{z}, \rho])) + \sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j}^{z} \rho \sigma_{k}^{z} - \frac{1}{2} \{ \sigma_{k}^{z} \sigma_{j}^{z}, \rho \} )"
    ltx_str_alt = r"-i {\sum_{j=0}^{N-1} (\sigma_{j,ket}^{z} - \sigma_{j,bra}^{z})} + {\sum_{j,k = 0}^{N-1} \gamma_{j,k} (\sigma_{j,ket}^{z} \sigma_{k,bra}^{z} - \frac{1}{2} ( \sigma_{k,ket}^{z} \sigma_{j,ket}^{z} + \sigma_{k,bra}^{z} \sigma_{j,bra}^{z} ) )}"
    ltx_str_simple = r"-imun (\sum_{j \in N1} (z_{j} - zcc_{j})) + \sum_{j,k \in NN} gamma_{j,k} (z_{j} zcc_{k} - 1.0div2.0 ( z_{k} z_{j} + zcc_{k} zcc_{j} ) )"
    parameters = {"imun": 1j,
                  "gamma": gamma,
                  "1.0div2.0": 1/2,
                  "N": N,
                  "N1": [i for i in range(N)],
                  "NN": [(i, j) for i in range(N) for j in range(N)]}

    print(ltx_str_full)
    print(parameters)

    generate = gen_mps.GenericGenerator(N, ops)
    mps = generate.mps_from_latex(ltx_str_full, parameters=parameters, ignore_imunit=False)
    return mps

print("Hej")
print(lindblad_mpo_latex(config_kwargs=config_kwargs))
