from TODO import eigs

def _dmrg_sweep_1site_(env, opts_eigs=None, Schmidt=None, precompute=False):
    r"""
    Perform sweep with 1-site DMRG, see :meth:`dmrg` for description.

    Returns
    -------
    env: Env3
        Environment of the <rhoA|H|rhoA> ready for the next iteration.
    """
    rhoA = env.bra
    for to in ('last', 'first'):
        for n in rhoA.sweep(to=to):
            AdagA = #TODO: take A=rhoA[n] and construct Adag A to math Heff1 #/rhoA.pre_1site(n, precompute=precompute)
            _, (BdagB,) = eigs(lambda x: env.Heff1(x, n), AdagA, k=1, **opts_eigs)
            # TODO: split BdagB to extract B of the purification
            # TODO: paste rhoA[n]=B, # rhoA.post_1site_(A, n)
            rhoA.orthogonalize_site_(n, to=to, normalize=True)
            #if Schmidt is not None and to == 'first' and n != rhoA.first:
            #    Schmidt[rhoA.pC] = rhoA[rhoA.pC].svd(sU=1, compute_uv=False)
            rhoA.absorb_central_(to=to)
            env.clear_site_(n)
            env.update_env_(n, to=to)

