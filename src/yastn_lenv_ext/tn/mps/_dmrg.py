
from typing import NamedTuple
import logging
from yastn import eigs, YastnError, ncon
import numpy as np
from yastn import eigh, svd_with_truncation
from yastn_lenv_ext.tn.mps._env import Env_double_lindblad

logger = logging.Logger('dmrg')


#################################
#           dmrg                #
#################################

class DMRG_out(NamedTuple):
    sweeps: int = 0
    method: str = ''
    energy: float = None
    denergy: float = None
    max_dSchmidt: float = None
    max_discarded_weight: float = None


def dmrg_(psi, H, method='1site',
        energy_tol=None, Schmidt_tol=None, max_sweeps=1, iterator_step=None,
        opts_eigs=None, opts_svd=None, precompute=False):
    tmp = _dmrg_(psi, H, method,
                energy_tol, Schmidt_tol, max_sweeps, iterator_step,
                opts_eigs, opts_svd, precompute)
    return tmp if iterator_step else next(tmp)

def _dmrg_(psi, H, method,
        energy_tol, Schmidt_tol, max_sweeps, iterator_step,
        opts_eigs, opts_svd, precompute):
    """ Generator for dmrg_(). """

    if not psi.is_canonical(to='first'):
        psi.canonize_(to='first')

    env = Env_double_lindblad(psi,H,psi)
    env.setup_(to='first')

    E_old = env.measure().item().real

    if opts_eigs is None:
        opts_eigs = {'hermitian': True, 'ncv': 3, 'which': 'SR'}

    if Schmidt_tol is not None:
        Schmidt_old = psi.get_Schmidt_values()
        Schmidt_old = {(n-1, n): sv for n, sv in enumerate(Schmidt_old)}

    max_dS = None
    Schmidt = None if Schmidt_tol is None else {}

    if energy_tol is not None and not energy_tol > 0:
        raise YastnError('DMRG: energy_tol has to be positive or None.')

    if method not in ('1site', ):
        raise YastnError('DMRG: dmrg method %s not recognized.' % method)

    if opts_svd is None and method == '2site':
        raise YastnError("DMRG: provide opts_svd for %s method." % method)

    for sweep in range(1, max_sweeps + 1):
        if method == '1site':
            _dmrg_sweep_1site_(env, opts_eigs=opts_eigs, Schmidt=Schmidt, precompute=precompute)
            max_dw = None
        else: # method == '2site':
            pass

        E = env.measure().item().real
        dE, E_old = abs(E_old - E), E
        converged = []

        if energy_tol is not None:
            converged.append(abs(dE) < energy_tol)

        if Schmidt_tol is not None:
            max_dS = max((Schmidt[k] - Schmidt_old[k]).norm().item() for k in Schmidt.keys())
            Schmidt_old = Schmidt.copy()
            converged.append(max_dS < Schmidt_tol)

        logger.info(f'Sweep = {sweep:03d}  energy = {E:0.14f}  dE = {dE:0.4f}  dSchmidt = {max_dS}')

        if len(converged) > 0 and all(converged):
            break
        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield DMRG_out(sweep, str(method), E, dE, max_dS, max_dw)
    yield DMRG_out(sweep, str(method), E, dE, max_dS, max_dw)


def _dmrg_sweep_1site_(env, opts_eigs=None, Schmidt=None, precompute=False, case="B"):
    r"""
    Perform sweep with 1-site DMRG, see :meth:`dmrg` for description.

    Returns
    -------
    env: Env3
        Environment of the <rhoA|H|rhoA> ready for the next iteration.
    """
    if opts_eigs is None:
        opts_eigs = {'hermitian': True, 'ncv': 3, 'which': 'SR'}

    rhoA = env.bra
    for to in ('last', 'first'):
        for n in rhoA.sweep(to=to):
            # calculate AdagA
            initA = rhoA[n]
            if case is "A":
                AdagA = initA.tensordot(initA.conj(), axes=((3,),(3,)))
                AdagA = AdagA.transpose(axes=(3,0,1,5,2,4,))
            if case is "B":
                AdagA = ncon([initA, initA.conj()], [(-1,-2,-3,-4), (-5,-8,-7,-6)])
            _, (BdagB,) = eigs(lambda x: env.Heff1(x, n), AdagA, k=1, **opts_eigs)
            if case is "A":
                exit()
            if case is "B":
                u, s, v = svd_with_truncation(BdagB, axes=((0,1,2,3), (4,5,6,7)), D_total=1)
                extractA = u.remove_leg()
                print(s.to_numpy().diagonal(), (extractA - initA).norm())
            rhoA[n] = extractA
            rhoA.orthogonalize_site_(n, to=to, normalize=True)
            if Schmidt is not None and to == 'first' and n != rhoA.first:
                Schmidt[rhoA.pC] = rhoA[rhoA.pC].svd(sU=1, compute_uv=False)
            rhoA.absorb_central_(to=to)
            env.clear_site_(n)
            env.update_env_(n, to=to)

