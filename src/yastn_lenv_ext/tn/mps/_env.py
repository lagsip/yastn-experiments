# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Environments for the <mps| mpo |mps> and <mps|mps>  contractions. """
from __future__ import annotations
# import abc
# import copy
# from numbers import Number
from yastn import eye, ones, tensordot, ncon, vdot, qr, svd, Tensor, YastnError
# from yastn.tn.mps import MpsMpoOBC, MpoPBC
from yastn.tn.mps._env import EnvParent

from yastn.tn.mps._mps_obc import multiply



class EnvParent_double3(EnvParent):

    def __init__(self, bra, op, ket):
        super().__init__(bra)
        self.ket = ket
        self.op = op

        if op.N != self.N*2 or ket.N != self.N:
            raise YastnError("Env: bra and ket should have the same number of sites, while op should have 2x that number")
        if self.bra.nr_phys != self.ket.nr_phys:
            raise YastnError('Env: bra and ket should have the same number of physical legs.')
        if self.op.nr_phys != 2:
            raise YastnError('Env: MPO operator should have 2 physical legs.')

    def factor(self) -> float:
        return self.bra.factor * self.op.factor * self.ket.factor

    def charges_missing(self, n):
        op_t = self.op.A[n].get_legs(axes=1).t
        psi_t = self.bra.A[n].get_legs(axes=1).t
        return any(tt not in psi_t for tt in op_t)


class EnvParent_double3_obc(EnvParent_double3):

    def __init__(self, bra, op, ket):

        super().__init__(bra, op, ket)
        
        legs = [self.bra.virtual_leg('first'), self.ket.virtual_leg('first').conj(), 
                op.virtual_leg('first').conj(), 
                self.ket.virtual_leg('first').conj(), self.bra.virtual_leg('first')]
        self.F[-1, 0] = ones(self.config, legs=legs, isdiag=False)
        
        legs = [self.bra.virtual_leg('last'), self.ket.virtual_leg('last').conj(), 
                op.virtual_leg('last').conj(), 
                self.ket.virtual_leg('last').conj(), self.bra.virtual_leg('last')]
        self.F[self.N, self.N - 1] = ones(self.config, legs=legs, isdiag=False)

    def measure(self, bd=(-1, 0)):
        return vdot(self.F[bd], self.F[bd[::-1]], conj=(0, 0)) * self.factor()
    
    def Heff0(self, C, bd):
        # bd, ibd = (bd[::-1], bd) if bd[1] < bd[0] else (bd, bd[::-1])
        # tmp = tensordot(self.F[bd], C @ self.F[ibd], axes=((0, 1), (0, 1)))
        # return tmp * self.op.factor
        pass
    

class Env_double_lindblad(EnvParent_double3_obc):

    def update_env_(self, n, to='last'):
        if to == 'last': # update left-env
            # contract 'upper' layer A, Adag to the left env
            axes = [(-1,-2,-3,2,1),(1,-5,-7,3),(2,-4,-6,3)]
            tmp = ncon([self.F[n-1, n], self.bra[n].conj(), self.ket[n]], axes)
            # contract operator
            axes = [(-1,-2,1,2,4,-6,-7),(1,-3,3,2),(3,-4,-5,4)]
            tmp = ncon([tmp, self.op[2*n], self.op[2*n+1]], axes)
            # contract 'lower' layer A, Adag
            axes = [(1,3,2,5,-3,-4,-5),(1,2,-1,4),(3,5,-2,4)]
            self.F[n, n+1] = ncon([tmp, self.bra.A[n].conj(), self.ket.A[n]], axes)
        elif to == 'first':
            # contract 'upper' layer A, Adag to the right env
            axes = [(1,2,-5,-6,-7),(-1,-3,1,3),(-2,-4,2,3)]
            tmp = ncon([self.F[n+1, n], self.bra[n].conj(), self.ket[n]], axes)
            # contract operator
            axes = [(-1,-2,2,4,1,-6,-7),(-3,-4,3,4),(3,-5,1,2)]
            tmp = ncon([tmp, self.op[2*n], self.op[2*n+1]], axes)
            # contract 'lower' layer A, Adag
            axes = [(-1,-2,-3,2,5,3,1),(-5,2,1,4),(-4,5,3,4)]
            self.F[n, n-1] = ncon([tmp, self.bra.A[n].conj(), self.ket.A[n]], axes)

    def Heff1(self, AdagA, n):
        FL = self.F[n - 1, n]
        FR = self.F[n + 1, n]
        op1, op2 = self.op[2*n], self.op[2*n+1]
        if AdagA.ndim == 6:
            tmp = FL.tensordot(AdagA, axes=((3,4,),(1,0,)))
            tmp = tmp.tensordot(FR, axes=((4,5,),(0,1,)))
            tmp = tmp.tensordot(op1, axes=((2,3,),(0,3,)))
            tmp = tmp.tensordot(op2, axes=((2,3,7,),(3,2,0,)))
            tmp = tmp.transpose(axes=(1,0,4,2,3,5))
        if AdagA.ndim == 8:
            tmp = AdagA.trace(axes=(3,5))
            tmp = FL.tensordot(tmp, axes=((3,4,),(0,3,)))
            tmp = tmp.tensordot(op1, axes=((2,3,),(0,3,)))
            tmp = tmp.tensordot(op2, axes=((6,4,),(0,3,)))
            tmp = tmp.tensordot(FR, axes=((3,2,6,),(0,1,2,)))
            legs = AdagA.get_legs(axes=(3,5))
            trace_dummy = eye(tmp.config, legs = [l for l in legs], isdiag=False)
            tmp = tmp.transpose(axes=(0,2,5,1,4,3))
            tmp = ncon([tmp, trace_dummy], [(-1,-2,-3, -5,-7,-8),(-4,-6,)])
        return tmp * self.op.factor

    def Heff2(self, AA, bd):
        # n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        # # version with no fusion in AA
        # # tmp = AA @ self.F[n2 + 1, n2]
        # # tmp = tensordot(self.op.A[n2], tmp, axes=((2, 3), (5, 3)))
        # # tmp = tensordot(self.op.A[n1], tmp, axes=((2, 3), (0, 3)))
        # # tmp = tmp.transpose(axes=(3, 0, 1, 4, 2, 5, 6))
        # # tmp = tensordot(self.F[n1 - 1, n1], tmp, axes=((0, 1), (0, 1)))
        # #
        # # fuse AA as [l, (b,b), (t,t), r]
        # tmp = AA @ self.F[n2 + 1, n2]
        # tmp = tmp.unfuse_legs(axes=1)
        # tmp = tensordot(self.op.A[n2], tmp, axes=((2, 3), (4, 2)))
        # tmp = tensordot(self.op.A[n1], tmp, axes=((2, 3), (0, 3)))
        # tmp = tmp.fuse_legs(axes=(3, 0, (1, 2), 4, 5))
        # tmp = tensordot(self.F[n1 - 1, n1], tmp, axes=((0, 1), (0, 1)))
        # return tmp * self.op.factor
        pass



# class Env_double_mps_mpo_mps(EnvParent_double3_obc):

#     def update_env_(self, n, to='last'):
#         if to == 'last':
#             tmp = self.F[n - 1, n] @ self.bra.A[n].conj()
#             tmp = tensordot(self.op.A[n], tmp, axes=((0, 1), (1, 2)))
#             self.F[n, n + 1] = tensordot(self.ket.A[n], tmp, axes=((0, 1), (2, 1)))
#         elif to == 'first':
#             tmp = self.ket.A[n] @ self.F[n + 1, n]
#             tmp = tensordot(tmp, self.op.A[n], axes=((2, 1), (2, 3)))
#             self.F[n, n - 1] = tensordot(tmp, self.bra.A[n].conj(), axes=((3, 1), (1, 2)))

#     def Heff1(self, A, n):
#         tmp = A @ self.F[n + 1, n]
#         tmp = tensordot(self.op.A[n], tmp, axes=((2, 3), (2, 1)))
#         tmp = tensordot(self.F[n - 1, n], tmp, axes=((0, 1), (2, 0)))
#         return tmp * self.op.factor

#     def Heff2(self, AA, bd):
#         n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
#         tmp = AA @ self.F[n2 + 1, n2]
#         tmp = tensordot(self.op.A[n2], tmp, axes=((2, 3), (3, 2)))
#         tmp = tensordot(self.op.A[n1], tmp, axes=((2, 3), (0, 3)))
#         tmp = tensordot(self.F[n1 - 1, n1], tmp, axes=((0, 1), (3, 0)))
#         return tmp * self.op.factor

#     def hole(self, n):
#         """ Hole for peps tensor at site n. """
#         tmp = self.F[n - 1, n] @ self.bra.A[n].conj()
#         tmp = tensordot(tmp, self.F[n + 1, n], axes=(3, 2))
#         tmp = tensordot(tmp, self.ket.A[n], axes=((3, 0), (2, 0)))
#         return tmp


# class Env_double_mps_mpo_mps_precompute(EnvParent_double3_obc):

#     def clear_site_(self, *args):
#         r"""
#         Clear environments pointing from sites whose indices are provided in args.
#         """
#         for n in args:
#             self.F.pop((n, n - 1), None)
#             self.F.pop((n, n + 1), None)
#             self.F.pop((n, n - 1, n - 1), None)
#             self.F.pop((n, n + 1, n + 1), None)

#     def update_env_(self, n, to='last'):
#         if to == 'last':
#             if (n - 1, n, n) in self.F:
#                 Aket = self.ket.A[n].fuse_legs(axes=((0, 1), 2))
#                 Abra = self.bra.A[n].fuse_legs(axes=((0, 1), 2))
#                 tmp = tensordot(Aket, self.F[n - 1, n, n], axes=(0, 0))
#                 tmp = tmp @ Abra.conj()
#             else:
#                 tmp = self.F[n - 1, n] @ self.bra.A[n].conj()
#                 tmp = tensordot(self.op.A[n], tmp, axes=((0, 1), (1, 2)))
#                 tmp = tensordot(self.ket.A[n], tmp, axes=((0, 1), (2, 1)))
#             self.F[n, n + 1] = tmp
#         elif to == 'first':
#             if (n + 1, n, n) in self.F:
#                 Aket = self.ket.A[n].fuse_legs(axes=(0, (1, 2)))
#                 Abra = self.bra.A[n].fuse_legs(axes=(0, (1, 2)))
#                 tmp = Aket @ self.F[n + 1, n, n]
#                 tmp = tensordot(tmp, Abra.conj(), axes=(2, 1))
#             else:
#                 tmp = self.ket.A[n] @ self.F[n + 1, n]
#                 tmp = tensordot(tmp, self.op.A[n], axes=((2, 1), (2, 3)))
#                 tmp = tensordot(tmp, self.bra.A[n].conj(), axes=((3, 1), (1, 2)))
#             self.F[n, n - 1] = tmp

#     def get_FL(self, n):
#         if (n - 1, n, n) not in self.F:
#             tmp = tensordot(self.F[n - 1, n], self.op.A[n], axes=(1, 0))
#             self.F[n - 1, n, n] = tmp.fuse_legs(axes=((0, 4), 3, (1, 2)))
#         return self.F[n - 1, n, n]

#     def get_FR(self, n):
#         if (n + 1, n, n) not in self.F:
#             tmp = tensordot(self.op.A[n], self.F[n + 1, n], axes=(2, 1))
#             self.F[n + 1, n, n] = tmp.fuse_legs(axes=((2, 3), 0, (1, 4)))
#         return self.F[n + 1, n, n]

#     def Heff1(self, A, n):
#         FR = self.get_FR(n)
#         tmp = tensordot(self.F[n - 1, n], A @ FR, axes=((0, 1), (0, 1)))
#         return tmp * self.op.factor

#     def Heff2(self, AA, bd):
#         n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
#         FL = self.get_FL(n1)
#         FR = self.get_FR(n2)
#         tmp = tensordot(FL, AA @ FR, axes=((0, 1), (0, 1)))
#         return tmp * self.op.factor


# class Env_double_mpo_mpo_mpo(EnvParent_double3_obc):

#     def update_env_(self, n, to='last'):
#         if to == 'last':
#             tmp = self.F[n - 1, n] @ self.bra.A[n].conj()
#             tmp = tensordot(self.op.A[n], tmp, axes=((0, 1), (1, 2)))
#             self.F[n, n + 1] = tensordot(self.ket.A[n], tmp, axes=((1, 0, 3), (1, 2, 4)))
#         elif to == 'first':
#             tmp = tensordot(self.ket.A[n], self.F[n + 1, n], axes=(2, 0))
#             tmp = tensordot(tmp, self.op.A[n], axes=((3, 1), (2, 3)))
#             self.F[n, n - 1] = tensordot(tmp, self.bra.A[n].conj(), axes=((4, 2, 1), (1, 2, 3)))

#     def Heff1(self, A, n):
#         tmp = A @ self.F[n + 1, n]
#         tmp = tensordot(self.op.A[n], tmp, axes=((2, 3), (3, 1)))
#         tmp = tensordot(self.F[n - 1, n], tmp, axes=((0, 1), (2, 0)))
#         return tmp * self.op.factor

#     def Heff2(self, AA, bd):
#         n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
#         # version with no fusion in AA
#         # tmp = AA @ self.F[n2 + 1, n2]
#         # tmp = tensordot(self.op.A[n2], tmp, axes=((2, 3), (5, 3)))
#         # tmp = tensordot(self.op.A[n1], tmp, axes=((2, 3), (0, 3)))
#         # tmp = tmp.transpose(axes=(3, 0, 1, 4, 2, 5, 6))
#         # tmp = tensordot(self.F[n1 - 1, n1], tmp, axes=((0, 1), (0, 1)))
#         #
#         # fuse AA as [l, (b,b), (t,t), r]
#         tmp = AA @ self.F[n2 + 1, n2]
#         tmp = tmp.unfuse_legs(axes=1)
#         tmp = tensordot(self.op.A[n2], tmp, axes=((2, 3), (4, 2)))
#         tmp = tensordot(self.op.A[n1], tmp, axes=((2, 3), (0, 3)))
#         tmp = tmp.fuse_legs(axes=(3, 0, (1, 2), 4, 5))
#         tmp = tensordot(self.F[n1 - 1, n1], tmp, axes=((0, 1), (0, 1)))
#         return tmp * self.op.factor


# class Env_double_mpo_mpobra_mpo(EnvParent_double3_obc):

#     def update_env_(self, n, to='last'):
#         if to == 'last':
#             tmp = tensordot(self.F[n - 1, n], self.ket.A[n], axes=(0, 0))
#             tmp = tensordot(tmp, self.op.A[n], axes=((0, 4), (0, 1)))
#             self.F[n, n + 1] = tensordot(tmp, self.bra.A[n].conj(), axes=((0, 1, 4), (0, 1, 3)))
#         elif to == 'first':
#             tmp = tensordot(self.F[n + 1, n], self.bra.A[n].conj(), axes=(2, 2))
#             tmp = tensordot(self.op.A[n], tmp, axes=((2, 3), (1, 4)))
#             self.F[n, n - 1] = tensordot(self.ket.A[n], tmp, axes=((1, 2, 3), (4, 2, 1)))

#     def Heff1(self, A, n):
#         tmp = tensordot(self.F[n - 1, n], A, axes=(0, 0))
#         tmp = tensordot(tmp, self.op.A[n], axes=((0, 3), (0, 1)))
#         tmp = tensordot(tmp, self.F[n + 1, n], axes=((2, 3), (0, 1)))
#         return tmp * self.op.factor

#     def Heff2(self, AA, bd):
#         n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
#         # version with no fusion in AA
#         # tmp = tensordot(self.F[n1 - 1, n1], AA, axes=(0, 0))
#         # tmp = tensordot(tmp, self.op.A[n1], axes=((0, 3), (0, 1)))
#         # tmp = tensordot(tmp, self.op.A[n2], axes=((5, 3), (0, 1)))
#         # tmp = tmp.transpose(axes=(0, 1, 4, 2, 6, 3, 5))
#         # tmp = tensordot(tmp, self.F[n2 + 1, n2], axes=((5, 6), (0, 1)))
#         #
#         # fuse AA as [l, (t,t), (b,b), r]
#         tmp = tensordot(self.F[n1 - 1, n1], AA, axes=(0, 0))
#         tmp = tmp.unfuse_legs(axes=3)
#         tmp = tensordot(tmp, self.op.A[n1], axes=((0, 3), (0, 1)))
#         tmp = tensordot(tmp, self.op.A[n2], axes=((4, 2), (0, 1)))
#         tmp = tmp.fuse_legs(axes=(0, 1, (3, 5), 2, 4))
#         tmp = tensordot(tmp, self.F[n2 + 1, n2], axes=((3, 4), (0, 1)))
#         return tmp * self.op.factor

#     def charges_missing(self, n):
#         op_t = self.op.A[n].get_legs(axes=3).t
#         psi_t = self.bra.A[n].get_legs(axes=3).t
#         return any(tt not in psi_t for tt in op_t)


# class EnvParent_double3_pbc(EnvParent_double3):

#     def __init__(self, bra, op, ket):
#         super().__init__(bra, op, ket)

#         lfk = self.ket.virtual_leg('first')
#         lfo = self.op.virtual_leg('first')
#         lfb = self.bra.virtual_leg('first')
#         tmp_oo = eye(self.config, legs=[lfo.conj(), lfo], isdiag=False)
#         tmp_bk = eye(self.config, legs=[lfk.conj(), lfb], isdiag=False)
#         self.F[-1, 0] = ncon([tmp_bk, tmp_oo], ((-0, -3), (-1, -2)))

#         llk = self.ket.virtual_leg('last')
#         llo = self.op.virtual_leg('last')
#         llb = self.bra.virtual_leg('last')
#         tmp_oo = eye(self.config, legs=[llo.conj(), llo],isdiag=False)
#         tmp_bk = eye(self.config, legs=[llk.conj(), llb], isdiag=False)
#         self.F[self.N, self.N - 1] = ncon([tmp_bk, tmp_oo], ((-0, -3), (-1, -2)))

#     def measure(self, bd=(-1, 0)):
#         return vdot(self.F[bd], self.F[bd[::-1]], conj=(0, 0)) * self.factor()

#     def Heff0(self, C, bd):
#         bd, ibd = (bd[::-1], bd) if bd[1] < bd[0] else (bd, bd[::-1])
#         tmp = tensordot(self.F[bd], C @ self.F[ibd], axes=((0, 1, 2), (0, 1, 2)))
#         return tmp * self.op.factor


# class Env_double_mps_mpopbc_mps(EnvParent_double3_pbc):

#     def update_env_(self, n, to='last'):
#         if to == 'last':
#             tmp = self.F[n - 1, n] @ self.bra.A[n].conj()
#             tmp = tensordot(self.op.A[n], tmp, axes=((0, 1), (1, 3)))
#             self.F[n, n + 1] = tensordot(self.ket.A[n], tmp, axes=((0, 1), (2, 1)))
#         elif to == 'first':
#             tmp = tensordot(self.F[n + 1, n], self.bra.A[n].conj(), axes=(3, 2))
#             tmp = tensordot(self.op.A[n], tmp, axes=((1, 2), (4, 1)))
#             self.F[n, n - 1] = tensordot(self.ket.A[n], tmp, axes=((1, 2), (1, 2)))

#     def Heff1(self, A, n):
#         precompute = (A.ndim == 2)
#         if precompute:  # Env_mps_mpopbc_mps does not have a separate precompute version
#             A = A.unfuse_legs(axes=1)

#         tmp = A @ self.F[n + 1, n]
#         tmp = tensordot(self.op.A[n], tmp, axes=((2, 3), (2, 1)))
#         tmp = tensordot(self.F[n - 1, n], tmp, axes=((0, 1, 2), (2, 0, 3)))

#         if precompute:
#             tmp = tmp.fuse_legs(axes=(0, (1, 2)))
#         return tmp * self.op.factor

#     def Heff2(self, AA, bd):
#         precompute = (AA.ndim == 2)
#         if precompute:
#             AA = AA.unfuse_legs(axes=(0, 1))

#         n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
#         tmp = AA @ self.F[n2 + 1, n2]
#         tmp = tensordot(self.op.A[n2], tmp, axes=((2, 3), (3, 2)))
#         tmp = tensordot(self.op.A[n1], tmp, axes=((2, 3), (0, 3)))
#         tmp = tensordot(self.F[n1 - 1, n1], tmp, axes=((0, 1, 2), (3, 0, 4)))

#         if precompute:
#             tmp = tmp.fuse_legs(axes=((0, 1), (2, 3)))
#         return tmp * self.op.factor

#     def hole(self, n):
#         """ Hole for peps tensor at site n. """
#         tmp = self.F[n - 1, n] @ self.bra.A[n].conj()
#         tmp = tensordot(tmp, self.F[n + 1, n], axes=((2, 4), (2, 3)))
#         tmp = tensordot(tmp, self.ket.A[n], axes=((3, 0), (2, 0)))
#         return tmp

