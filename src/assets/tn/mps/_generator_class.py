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
from __future__ import annotations
from yastn import YastnError
from yastn.tn.mps._mps_obc import Mps, Mpo, MpsMpoOBC
from yastn.tn.mps._latex2term import latex2term
from yastn.tn.mps._initialize import random_mpo, random_mps
from yastn.tn.mps._generate_mpo import Hterm, generate_mpo

import numpy as np
import assets.math.operators as opmath


class GenericGenerator:

    def __init__(self, N, operators, map=None, Is=None, parameters=None, opts={"tol": 1e-13}):
        r"""
        Generator is a convenience class building MPSs from a set of local operators.

        Parameters
        ----------
        N: int
            number of sites of MPS.
        operators: object or dict[str, yastn.Tensor]
            a set of local operators, e.g., an instance of :class:`yastn.operators.Spin12`.
            Or a dictionary with string-labeled local operators, including at least ``{'I': <identity operator>,...}``.
        map: dict[int, int]
            custom labels of N sites indexed from 0 to N-1 , e.g., ``{3: 0, 21: 1, ...}``.
            If ``None``, the sites are labled as :code:`{site: site for site in range(N)}`.
        Is: dict[int, str]
            For each site (using default or custom label), specify identity operator by providing
            its string key as defined in ``operators``.
            If ``None``, assumes ``{i: 'I' for i in range(N)}``, which is compatible with all predefined
            ``operators``.
        parameters: dict
            Default parameters used by the interpreters :meth:`Generator.mps` and :meth:`Generator.mps`.
            If None, uses default ``{'sites': [*map.keys()]}``.
        opts: dict
            used if compression is needed. Options passed to :meth:`yastn.linalg.svd_with_truncation`.
        """
        # Notes
        # ------
        # * Names `minus` and `1j` are reserved paramters in self.parameters
        # * Write operator `a` on site `3` as `a_{3}`.
        # * Write element if matrix `A` with indicies `(1,3)` as `A_{1,2}`.
        # * Write sumation over one index `j` taking values from 1D-array `listA` as `\sum_{j \in listA}`.
        # * Write sumation over indicies `j0,j1` taking values from 2D-array `listA` as `\sum_{j0,j1 \in listA}`.
        # * In an expression only round brackets, i.e., ().
        self.N = N
        self._ops = operators
        self._map = {i: i for i in range(N)} if map is None else map
        if len(self._map) != N or sorted(self._map.values()) != list(range(N)):
            raise YastnError("MPS: Map is inconsistent with mps of N sites.")
        self._Is = {k: 'I' for k in self._map.keys()} if Is is None else Is
        if self._Is.keys() != self._map.keys():
            raise YastnError("MPS: Is is inconsistent with map.")
        if not all(hasattr(self._ops, v) and callable(getattr(self._ops, v)) for v in self._Is.values()):
            raise YastnError("MPS: operators do not contain identity specified in Is.")

        self._I = Mps(self.N)
        for label, site in self._map.items():
            local_I = getattr(self._ops, self._Is[label])
            self._I.A[site] = local_I().add_leg(axis=0, s=-1).add_leg(axis=2, s=1)

        self.config = self._I.A[0].config
        self.parameters = {} if parameters is None else parameters
        self.parameters["minus"] = -float(1.0)
        self.parameters["1j"] = 1j

        self.opts = opts

    def random_seed(self, seed):
        r"""
        Set seed for random number generator used in backend (of self.config).

        Parameters
        ----------
        seed: int
            Seed number for random number generator.
        """
        self.config.backend.random_seed(seed)

    def I(self) -> MpsMpoOBC:
        """ Identity MPS derived from identity in local operators class. """
        return self._I.shallow_copy()

    def random_mps(self, n=None, D_total=8, sigma=1, dtype='float64', **kwargs) -> MpsMpoOBC:
        r"""
        Generate a random MPS of total charge ``n`` and bond dimension ``D_total``.

        Equivalent to :meth:`mps.random_mps<yastn.tn.mps.random_mps>`.
        """
        return random_mps(self._I, n=n, D_total=D_total, sigma=sigma, dtype=dtype, **kwargs)

    def random_mpo(self, D_total=8, sigma=1, dtype='float64', **kwargs) -> MpsMpoOBC:
        r"""
        Generate a random MPO with bond dimension ``D_total``.

        Equivalent to :meth:`mps.random_mps<yastn.tn.mps.random_mpo>`.
        """
        return random_mpo(self._I, D_total=D_total, sigma=sigma, dtype=dtype, **kwargs)

    def mps_from_latex(self, ltx_str, parameters=None, opts=None, ignore_i=True) -> MpsMpoOBC:
        r"""
        Convert latex-like string to yastn.tn.mps MPS.

        Parameters
        -----------
        ltx_str: str
            The definition of the MPS given as latex expression. The definition uses string names of the operators given in. The assignment of the location is
            given e.g. for ``cp`` operator as ``cp_{j}`` (always with ``{}``-brackets!) for ``cp`` operator acting on site ``j``.
            The space and * are interpreted as multiplication by a number of by an operator. E.g., to multiply by a number use ``g * cp_j c_{j+1}`` where ``g`` has to be defines in ``parameters`` or writen directly as a number,
            You can define automatic summation with expression ``\sum_{j \in A}``, where ``A`` has to be iterable, one-dimensional object with specified values of ``j``.

        parameters: dict
            Keys for the dict define the expressions that occur in ``ltx_str``.

        opts: dict
            Options passed to :meth:`yastn.linalg.truncation_mask`.
            It includes information on how to truncate the Schmidt values.
        """

        c1, generated_params = self.any2simple_latex(ltx_str, parameters, ignore_i=ignore_i)
        print("This is c1: \n", c1, "\n")
        generated_params = {**self.parameters, **generated_params}
        c2 = latex2term(c1, generated_params)
        print("This is c2: \n", c2, "\n")
        cc_operators = opmath.compl_conjugate(self._ops)
        expanded_ops = dict(self._ops.to_dict(), **cc_operators)
        print("This is the full list of operators: \n", expanded_ops, "\n")
        c3 = self.term2generic_term(c2, expanded_ops, generated_params)
        print("This is c3: \n", c3, "\n")
        if opts is None:
            opts={'tol': 5e-15}
        return generate_mpo(self._I, c3, opts)



        # parameters = {**self.parameters, **parameters}
        # c2 = latex2term(ltx_str, parameters)
        # print("This is c2: \n", c2, "\n")
        # cc_operators = opmath.compl_conjugate(self._ops)
        # expanded_ops = dict(self._ops.to_dict(), **cc_operators)
        # print("This is the full list of operators: \n", expanded_ops, "\n")
        # c3 = self.term2generic_term(c2, expanded_ops, parameters)
        # print("This is c3: \n", c3, "\n")
        # if opts is None:
        #     opts={'tol': 5e-15}
        # return generate_mpo(self._I, c3, opts)

    def mps_from_templete(self, templete, parameters=None):   # remove from docs (DELETE)
        r"""
        Convert instruction in a form of single_term-s to yastn.tn.mps MPS.

        single_term is a templete which which take named from operators and templetes.

        Parameters
        -----------
        templete: list
            List of single_term objects. The object is defined in ._latex2term
        parameters: dict
            Keys for the dict define the expressions that occur in ltx_str

        Returns
        -------
        yastn.tn.mps.MpsMpoOBC
        """
        parameters = {**self.parameters, **parameters}
        c3 = self.term2generic_term(templete, self._ops.to_dict(), parameters)
        return generate_mpo(self._I, c3)

    def any2simple_latex(self, ltx_str, parameters, ignore_i=True):
        r"""
        Helper function to rewrite the instruction given as a latex string
        to a simpler string that can be decoded further (see latex2term).

        Parameters
        ----------
        ltx_str: string
            full latex string
        parameters: dict
            dictionary with parameters for the generator
        """

        new_expr = ltx_str
        new_params = parameters.copy()

        # resolve commutators
        new_expr = self.resolve_commutators(new_expr)
        print("\nResolved Commutators:\n", new_expr)

        # extract rho
        new_expr = self.extract_rho(new_expr)
        print("\nExtraced Rho:\n", new_expr)

        # rephrase summation
        new_expr, new_params = self.rewrite_sum(new_expr, parameters)
        print("\nRephrased Sum:\n", new_expr)

        # rename greek symbols
        new_expr, new_params = self.rename_symbols(new_expr, new_params)
        print("\nRenamed Symbols:\n", new_expr)

        # extract constants
        new_expr, new_params = self.extract_constants(new_expr, new_params, ignore_i=ignore_i)
        print("\nExtraced constants:\n", new_expr)

        # resolve ket/bra-marked operators
        new_expr = self.resolve_ketbra(new_expr)
        print("\nResolved Ket-Bra-notation:\n", new_expr)

        print("\nThe extended Parameters:\n", new_params)

        
        return new_expr, new_params

    def resolve_commutators(self, expression):
        r"""
        Helper function to rewrite the instruction given as a string

        Parameters
        ----------
        expression: string
            expression as string to resolve commutators in
        """
        # Resolve commutators
        new_expr = expression

        open_ind = np.array([i for i, ltr in enumerate(new_expr) if ltr == '[' or (ltr == '{' and new_expr[i-1] == '\\') or ltr == '('])
        close_ind = np.array([i for i, ltr in enumerate(new_expr) if ltr == ']' or (ltr == '}' and new_expr[i-1] == '\\') or ltr == ')'])

        if(len(open_ind) != len(close_ind)): raise YastnError("Brackets should close and open in equal counts")
        while(len(open_ind) > 0):
            #print(new_expr)
            #print(open_ind)
            #print(close_ind)
            # find innermost pair
            ind_num_open = 0
            while(ind_num_open + 1 < len(open_ind) and open_ind[ind_num_open + 1] < close_ind[0]):
                ind_num_open += 1 # ensures closest bracket pair

            bracket = new_expr[open_ind[ind_num_open]]
            #print(bracket)
            #print(new_expr[close_ind[0]])
            if(bracket == '[' and ']' != new_expr[close_ind[0]] or
               bracket == '{' and '}' != new_expr[close_ind[0]] or
               bracket == '(' and ')' != new_expr[close_ind[0]]): raise YastnError("Inconsistent brackets")

            # check whether it's a commutator and which
            if (bracket != '('):
                operator = ''
                length = 0
                if (bracket == '['):
                    operator = '-'
                    length = 2
                elif (bracket == '{'):
                    operator = '+'
                    length = 6

                    # due to the backslash required for { brackets, remove them for smooth replacement later
                    new_expr = new_expr[0:(open_ind[ind_num_open]-1)] \
                        + new_expr[open_ind[ind_num_open]:(close_ind[0]-1)] \
                        + new_expr[close_ind[0]:]
                    #print("Modified expression: \n", new_expr, "\n")

                    open_ind[ind_num_open] -= 1
                    close_ind[0] -=2

                # extract content within brackets
                content = new_expr[open_ind[ind_num_open]+1:close_ind[0]]
                comma_ind = [i for i, ltr in enumerate(content) if ltr == ',']
                if len(comma_ind) == 1: 
                    # resolve commutator
                    terms = content.split(',')
                    new_term = '(' + terms[0] + terms[1] + operator + terms[1] + terms[0] + ')'

                    # insert new term
                    new_expr = new_expr[0:open_ind[ind_num_open]] + new_term + new_expr[close_ind[0]+1:]

                    #update other indices
                    len_diff = len(new_term) - len(content) - length
                    open_ind[ind_num_open + 1:] += len_diff
                    close_ind[1:] += len_diff

                #else: 
                #    raise YastnError("Commutators should only have one comma")


            # remove from list
            open_ind = np.delete(open_ind, ind_num_open)
            close_ind = np.delete(close_ind, 0)

        return new_expr

    def extract_rho(self, expression):
        r"""
        Helper function to rewrite the instruction given as a string

        Parameters
        ----------
        expression: string
            expression as string to extract rho from
        """

        new_expr = expression
        gen_terminator = ['+','-',',',';']
        right_terminator = [')',']','}']
        left_terminator = ['(','[','{']

        incl_separator = ['/', ' ']
        excl_separator = []

        rho_loc = new_expr.find('\\rho')
        while(rho_loc != -1):
            # get right side
            end_ind = rho_loc + 4
            separator_ind = [end_ind]
            layer = 0
            while (end_ind < len(new_expr)):
                symbol = new_expr[end_ind]
                if layer == 0:
                    if symbol in gen_terminator or symbol in right_terminator:
                        separator_ind += [end_ind]
                        break
                
                    if symbol in incl_separator or symbol in excl_separator:
                        separator_ind += [end_ind]
                else:
                    # if we are in a bracket right now ignore it and wait for it to close again
                    if symbol in right_terminator:
                        layer -= 1

                if symbol in left_terminator:
                    layer += 1

                end_ind += 1
            
            # collect various objects that need be conjugated
            objects = []
            for i in range(len(separator_ind)-1):
                start_ind = separator_ind[i]
                end_ind = separator_ind[i+1]
                # if just empty space don't consider
                if end_ind - start_ind < 2:
                    continue

                # if separator was '\' keep it, ' ' is not needed however
                if new_expr[start_ind] in excl_separator:
                    start_ind += 1
                
                # add to objects
                objects += [new_expr[start_ind:end_ind]]

            # for each object add the bra subscript
            new_objects = []
            for obj in objects:
                subscript_ind = obj.find('_{') + 2
                if(subscript_ind == 1):
                    subscript_ind = obj.find('_') + 1
                    if(subscript_ind == 0):
                        subscript_ind = obj.find('^')
                        if(subscript_ind == -1):
                            new_objects += (obj + '_{bra}')
                        else:
                            new_objects += obj[:subscript_ind] + "_{bra}" + obj[subscript_ind:]
                    else:
                        new_objects += obj[:subscript_ind] + '{' + obj[subscript_ind] + ',bra}' + obj[(subscript_ind+1):]
                else:
                    layer = 0
                    while subscript_ind < len(obj):
                        if obj[subscript_ind] in left_terminator:
                            layer += 1
                        if obj[subscript_ind] in right_terminator:
                            layer -=1
                        # assuming brackets are set correctly, then layer -1 indicates the closing curly bracket
                        if layer == -1:
                            new_objects += obj[:subscript_ind] + ",bra" + obj[subscript_ind:]
                            break
                        subscript_ind += 1
            #print(separator_ind)
            #print(objects)
            #print(new_objects)
            
            # replace section
            object_str = ''.join(new_objects)
            new_expr = new_expr[:separator_ind[0]] + object_str + new_expr[separator_ind[-1]:]

            # get left side
            start_ind = rho_loc -1
            separator_ind = [start_ind]
            layer = 0
            while (start_ind >= 0):
                symbol = new_expr[start_ind]
                if layer == 0:
                    if symbol in gen_terminator or symbol in left_terminator:
                        separator_ind = [start_ind] + separator_ind
                        break
                
                    if symbol in incl_separator or symbol in excl_separator:
                        separator_ind = [start_ind] + separator_ind
                else:
                    # if we are in a bracket right now ignore it and wait for it to close again
                    if symbol in left_terminator:
                        layer -= 1

                if symbol in right_terminator:
                    layer += 1

                start_ind -= 1
            
            # collect various objects to add the ket subscript to
            objects = []
            for i in range(len(separator_ind)-1):
                start_ind = separator_ind[i]+1
                end_ind = separator_ind[i+1]+1
                # if just empty space don't consider
                if end_ind - start_ind < 2:
                    continue

                # if separator was '\' keep it, ' ' is not needed however
                if new_expr[start_ind] in excl_separator:
                    start_ind += 1
                
                # add to objects
                objects += [new_expr[start_ind:end_ind]]

            # for each object add the ket subscript
            new_objects = []
            for obj in objects:
                subscript_ind = obj.find('_{') + 2
                if(subscript_ind == 1):
                    subscript_ind = obj.find('_') + 1
                    if(subscript_ind == 0):
                        subscript_ind = obj.find('^')
                        if(subscript_ind == -1):
                            new_objects += (obj + '_{ket}')
                        else:
                            new_objects += obj[:subscript_ind] + "_{ket}" + obj[subscript_ind:]
                    else:
                        new_objects += obj[:subscript_ind] + '{' + obj[subscript_ind] + ',ket}' + obj[(subscript_ind+1):]
                else:
                    layer = 0
                    while subscript_ind < len(obj):
                        if obj[subscript_ind] in left_terminator:
                            layer += 1
                        if obj[subscript_ind] in right_terminator:
                            layer -=1
                        # assuming brackets are set correctly, then layer -1 indicates the closing curly bracket
                        if layer == -1:
                            new_objects += obj[:subscript_ind] + ",ket" + obj[subscript_ind:]
                            break
                        subscript_ind += 1
            #print(separator_ind)
            #print(objects)
            #print(new_objects)
            # replace section
            object_str = ''.join(new_objects)
            new_expr = new_expr[:(separator_ind[0]+1)] + object_str + new_expr[separator_ind[-1]:]
            new_expr = new_expr.replace('\\rho', '', 1)
            
            rho_loc = new_expr.find("\\rho")

            #print(new_expr)

        return new_expr

    # def rewrite_sum_old(self, expression):
    #     r"""
    #     Helper function to rewrite the instruction given as a string

    #     Parameters
    #     ----------
    #     expression: string
    #         expression as string to rewrite sum in
    #     """

    #     new_expr = expression
    #     sum_loc = new_expr.find("\\sum")

    #     right_terminator = [')',']','}']
    #     left_terminator = ['(','[','{']

    #     used_iterators = []
    #     while (sum_loc != -1):

    #         if(new_expr[sum_loc+4] == '_'):
    #             # search for a '^'
    #             curr_ind = sum_loc + 5
    #             layer = 0
    #             left = True # if it remains true at the end -> sum does not conform to _{k=0}_^{N} schema
    #             iterators = [""]
    #             start_val = ""

    #             # iterate over all symbols to find full subscript term
    #             while (curr_ind < len(new_expr)):
    #                 symbol = new_expr[curr_ind]
    #                 if(symbol in left_terminator):
    #                     layer += 1
    #                 elif(symbol in right_terminator):
    #                     layer += 1
                    
    #                 # record iterators
    #                 if not left:
    #                     start_val += symbol
    #                 elif symbol == "," and layer == 1: 
    #                     # must only differentiate terms in the first {}-layer, others might be nested inside but should be ignored
    #                     iterators += [""]
    #                 elif symbol == "=" and layer == 1:
    #                     left = False
    #                 else:
    #                     iterators[-1] += symbol
                    
    #                 curr_ind += 1
    #                 if layer == 0:
    #                     break
                
    #             # if sum does not conform to _{k=0}^{N} then ignore it
    #             if left:
    #                 continue

    #             # if there is a superscript, it is on this next index
    #             if(new_expr[curr_ind] == '^'):
    #                 if(new_expr[curr_ind+1] == '{'):
    #                     new_expr = new_expr[:sum_loc+4] + '_{' + start_val + "\\leq" + ",".join(map(str, iterators)) + "\\leq" + new_expr[curr_ind+2:]
    #                 else:
    #                     new_expr = new_expr[:sum_loc+4] + '_{' + start_val + "\\leq" + ",".join(map(str, iterators)) + "\\leq" + new_expr[curr_ind+2] + '}' + new_expr[curr_ind+3:]
                
    #             used_iterators += iterators

    #         elif(new_expr[sum_loc+4] == '^'):
    #             # simply replace as next alphabet character \in whatever is in the superscript
    #             new_iterator = 'k' if len(used_iterators) == 0 else \
    #                 chr(used_iterators[-1] + 1)
    #             if(new_expr[sum_loc+5] == '{'):
    #                 new_expr = new_expr[:sum_loc+4] + '_{' + new_iterator + "\\in" + new_expr[sum_loc+6:]
    #             else:
    #                 new_expr = new_expr[:sum_loc+4] + '_{' + new_iterator + "\\in" + new_expr[sum_loc+6] + '}' + new_expr[sum_loc+7:]
    #             used_iterators += [new_iterator]

    #         sum_loc = new_expr.find("\\sum")

    #     return expression

    def rewrite_sum(self, expression, parameters):
        r"""
        Helper function to rewrite the instruction given as a string

        Parameters
        ----------
        expression: string
            expression as string to rewrite sum in
        """

        new_expr = expression
        new_params = parameters.copy()
        global_iterators = []

        sum_loc = new_expr.find("\\sum")
        while(sum_loc != -1):
            # only need up to 5 args for sum - _ - subscript - ^ - superscript
            sum_terms, sum_end = self.split_ltx2terms(ltx_str=new_expr, index=sum_loc, start_layer=1, max_terms=5, return_length=True)
            #print(sum_terms)
            if(sum_terms[1] == "_"):
                sum_type = ""
                # test for e.g. "i = 0" in subscript with "N" in superscript
                sub_terms = self.split_ltx2terms(ltx_str=sum_terms[2])
                left = True
                iterators = [""]
                subvalues = [""]
                bound_supvalues = [""]
                sum_conditions = []

                #print(sub_terms)

                for i in range(1, len(sub_terms)-1):
                    term = sub_terms[i]
                    #print(term)
                    if(term == ";"):
                        # a ; indicates a new set of conditions so we save the current set of conditions
                        sum_conditions += [{"type":sum_type if sum_type else "over",
                                            "iterators":iterators.copy(),
                                            "subvalues":subvalues.copy(),
                                            "supvalues":[]}]
                        sum_type = ""
                        iterators.clear()
                        subvalues.clear()
                        left = True
                    elif(term == "="):
                        if(not sum_type):
                            # if an = sign is found we seemingly have a range pattern where the subscript assigns a starting value and the superscript the max value
                            sum_type = "range"
                        left = False
                    elif(term in ["<",">","\\lt","\\gt","\\leq","\\leqslant","\\geq","\\geqslant"]):
                        if(not sum_type):
                            # if any inequality operator is found we seemingly have an inequality pattern where the iterator is bound through a max/min value
                            sum_type = "inequality"
                        left = False
                    elif(term in ["\\in"]):
                        if(not sum_type):
                            # if the element operator is found, then we seemingly have a set pattern where the subscript states the iterators to be elements of a set
                            sum_type = "in"
                        left = False
                    elif(left):
                        # if left is true we haven't found an = sign (this could of course also entail a < or >= or similar but )
                        if(term != ","):
                            iterators[-1] += term
                        else:
                            iterators += [""]
                    else:
                        # if right an = sign has been found and we should get the bound value
                        if(term != ","):
                            subvalues[-1] += term
                        else:
                            subvalues += []
                sum_conditions += [{"type":sum_type if sum_type else "over",
                                    "iterators":iterators.copy(),
                                    "subvalues":subvalues.copy(),
                                    "supvalues":[]}]
                if(sum_terms[3] == "^"):
                    # only needed to continue if we actually have a iterator - = - value pattern
                    sup_terms = self.split_ltx2terms(sum_terms[4])
                    condition_num = 0

                    #print(sup_terms)

                    for i in range(1,len(sup_terms)-1):
                        term = sup_terms[i]
                        #print(term)

                        if(term == ";"):
                            while(True):
                                if(sum_conditions[condition_num]["type"] in ["range","over"]):
                                    # if the nth sum_condition is of range or set type (which require a superscript) then assign to it the latest superscript term
                                    sum_conditions[condition_num]["supvalues"] = bound_supvalues.copy()
                                    condition_num += 1
                                    break
                                condition_num += 1
                                if(condition_num >= len(sum_conditions)):
                                    raise SyntaxError("Sum should have as many superscript terms as range or set terms in subscript")
                        elif(term != ","):
                            bound_supvalues[-1] += term
                        else:
                            bound_supvalues += []
                    
                    while(True):
                        if(sum_conditions[condition_num]["type"] in ["range","over"]):
                            # if the nth sum_condition is of range or set type (which require a superscript) then assign to it the latest superscript term
                            sum_conditions[condition_num]["supvalues"] = bound_supvalues.copy()
                            condition_num += 1
                            break
                        condition_num += 1
                        if(condition_num >= len(sum_conditions)):
                            raise SyntaxError("Sum should have as many superscript terms as range or set terms in subscript")
                else:
                    _, sum_end = self.split_ltx2terms(new_expr, sum_loc, start_layer=1, max_terms=3, return_length=True)

                new_sums = ""
                for cond in sum_conditions:
                    iterator_str = ""
                    set_str = ""
                    #print(cond)
                    iterator_str = ",".join(cond["iterators"])
                    if(cond["type"] == "range"):
                        lower = self.resolve_term(self.split_ltx2terms(cond["subvalues"][0], start_layer=1), new_params)
                        higher = self.resolve_term(self.split_ltx2terms(cond["supvalues"][0], start_layer=1), new_params)
                        l = int(lower.real)
                        h = int(higher.real)
                        
                        iterator_count = len(cond["iterators"])
                        set_str = str(iterator_count) + "range" + str(l) + "to" + str(h)
                        if(iterator_count > 1):
                            new_params[set_str] = self.gen_tuple_list(list(range(l,h+1)), iterator_count)
                        else:
                            new_params[set_str] = list(range(l,h+1))

                    elif(cond["type"] == "over"):
                        higher = self.resolve_term(self.split_ltx2terms(cond["supvalues"][0], start_layer=1), new_params)
                        
                        iterator_count = len(cond["iterators"])
                        set_str = str(iterator_count) + "rangeto" + str(h)

                        if isinstance(higher, list):
                            new_params[set_str] = higher
                        else:
                            h = int(higher.real)
                            if(iterator_count > 1):
                                new_params[set_str] = self.gen_tuple_list(list(range(0,h+1)), iterator_count)
                            else:
                                new_params[set_str] = list(range(0, h+1))

                    elif(cond["type"] == "inequality"):
                        # assume smaller sign
                        ## TODO: implement differentiation of inequalities
                        higher = self.resolve_term(self.split_ltx2terms(cond["supvalues"][0], start_layer=1), new_params)
                        h = int(higher.real)
                            
                        set_str = str(len(cond["iterators"])) + "rangeto" + str(h)

                        new_params[set_str] = list(range(0, h+1))
                    
                    elif(cond["type"] == "in"):
                        set_str = cond["subvalues"][0]
                    
                    new_sums += "\\sum_{" + iterator_str + " \\in " + set_str + "} "

                #print("loc",sum_loc)
                #print("end",sum_end)
                new_expr = new_expr[:sum_loc] + new_sums + new_expr[sum_end:]

            sum_loc = new_expr.find("\\sum", sum_loc+4)

            #print(new_expr)

        return new_expr, new_params
    
    def rename_symbols(self, expression, parameters):
        r"""
        Helper function to rewrite the instruction given as a string

        Parameters
        ----------
        expression: string
            expression as string to rename symbols in
        """

        new_expr = expression
        new_params = parameters.copy()

        # first consider sigma as the standard operator
        sigma_loc = new_expr.find("\\sigma")
        while(sigma_loc != -1):
            sigma_terms, sigma_end = self.split_ltx2terms(new_expr, sigma_loc, start_layer=1, max_terms=5, return_length=True)
            sigma_mid = sigma_loc + len(sigma_terms[0]) + len(sigma_terms[1]) + len(sigma_terms[2])
            op = "sigma"
            if(sigma_terms[1] == "^"):
                # potentially the sigma doesn't have a subscript and only superscript
                sup_terms = self.split_ltx2terms(sigma_terms[2])
                if (len(sup_terms) == 1):
                    op = sup_terms[0]
                else:
                    op = sup_terms[1]
                
                # erase superscript
                new_expr = new_expr[:(sigma_loc+6)] + new_expr[sigma_mid:]
                
            elif(sigma_terms[3] == "^"):
                # this should be the common case in which we find a subscript and a superscript
                sup_terms = self.split_ltx2terms(sigma_terms[4])
                if (len(sup_terms) == 1):
                    op = sup_terms[0]
                else:
                    op = "".join(sup_terms[1:-1])
                    
                # erase superscript
                new_expr = new_expr[:sigma_mid] + new_expr[sigma_end:]
            
            new_expr = new_expr[:sigma_loc] + op + new_expr[(sigma_loc+6):]
            sigma_loc = new_expr.find("\\sigma", sigma_loc+1)

        
        # now \gamma is another common symbol
        gamma_loc = new_expr.find("\\gamma")
        while(gamma_loc != -1):
            new_expr = new_expr[:gamma_loc] + new_expr[(gamma_loc+1):]

            gamma_loc = new_expr.find("\\gamma", gamma_loc+6)
        if("\\gamma" in new_params):
            new_params["gamma"] = new_params["\\gamma"]



        return new_expr, new_params

    def extract_constants(self, expression, parameters, ignore_i=False):
        r"""
        Helper function to rewrite the instruction given as a string

        Parameters
        ----------
        expression: string
            expression as string to extract constants from
        """

        new_expr = expression
        new_params = parameters.copy()

        terminators = [","," ","{","}","[","]","(",")","\\","+","-","*","/"]
        digits = ["0","1","2","3","4","5","6","7","8","9"]

        if(not ignore_i):
            new_params["imun"] = 1j


            # look for imaginary unit
            i_loc = new_expr.find("i")
            while(i_loc != -1):
                # test whether used as imaginary unit and not part of a name
                if(new_expr[i_loc+1] in terminators):
                    # only try if for sure not the start of a word
                    curr_pos = i_loc - 1
                    while(new_expr[curr_pos] in digits and 0 < curr_pos):
                        curr_pos -= 1
                    if(new_expr[curr_pos] in terminators):
                        # this means that it really is an i
                        new_expr = new_expr[:i_loc] + "imun" + new_expr[(i_loc+1):]

                i_loc = new_expr.find("i", i_loc+1)

        # look for fractions
        f_loc = new_expr.find("\\frac{")
        while(f_loc != -1):
            top_terms, mid_loc = self.split_ltx2terms(new_expr, f_loc + 5, return_length=True)
            bot_terms, end_loc = self.split_ltx2terms(new_expr, mid_loc, return_length=True)
            try:
                top_val = self.resolve_term(top_terms[1:-1], parameters=parameters)
                bot_val = self.resolve_term(bot_terms[1:-1], parameters=parameters)
                str_val = str(top_val.real) + "pls" + str(top_val.imag) + "jfrac" + str(bot_val.real) + "pls" + str(bot_val.imag) + "j"
                val = top_val/bot_val

                new_expr = new_expr[:f_loc] + str_val + new_expr[end_loc:]
                new_params[str_val] = val
            except:
                # In this case the fraction probably contained unresolvable values such as operators. Ignore for now
                pass

            f_loc = new_expr.find("\\frac{", f_loc+1)



        return new_expr, new_params

    def resolve_ketbra(self, expression):
        r"""
        Helper function to rewrite the instruction given as a string

        Parameters
        ----------
        expression: string
            expression as string to resolve ket/bra notation
        """

        new_expr = expression

        ket_loc = new_expr.find("ket")
        while(ket_loc != -1):

            # so on the same layer as the ket notation, we gather all the terms until the end of the subscript bracket
            end_subscr, end_ind = self.split_ltx2terms(new_expr, ket_loc, start_layer=1, return_length=True)
            # now, knowing the end index, we start splitting from here backwards to the beginning
            subscript, start_ind = self.split_ltx2terms(new_expr, end_ind-1, return_length=True)

            # if ket actually isn't in subscript skip
            if(new_expr[start_ind] != "_"):
                break

            ket_arg_ind = len(subscript) - len(end_subscr)
            # remove ket from subscript (since we know the amount of elements from the ket to end iteration)
            subscript = subscript[:-len(end_subscr)] + end_subscr[1:]
            # if multiple objects were in the subscript separated by a comma, remove the comma on 1 side
            if(subscript[ket_arg_ind] == ","):
                subscript = subscript[:ket_arg_ind] + subscript[(ket_arg_ind+1):]
            elif(subscript[ket_arg_ind-1] == ","):
                subscript = subscript[:(ket_arg_ind-1)] + subscript[ket_arg_ind:]
            # if only ket was in subscript, remove the leftover brackets
            if(len(subscript) < 3):
                subscript = []

            new_expr = new_expr[:(start_ind+1)] + "".join(subscript) + new_expr[end_ind:]


            ket_loc = new_expr.find("ket", ket_loc+1)

            
        bra_loc = new_expr.find("bra")
        while(bra_loc != -1):

            # so on the same layer as the ket notation, we gather all the terms until the end of the subscript bracket
            end_subscr, end_ind = self.split_ltx2terms(new_expr, bra_loc, start_layer=1, return_length=True)
            # now, knowing the end index, we start splitting from here backwards to the beginning
            subscript, start_ind = self.split_ltx2terms(new_expr, end_ind-1, return_length=True)

            # if ket actually isn't in subscript skip
            if(new_expr[start_ind] != "_"):
                break

            bra_arg_ind = len(subscript) - len(end_subscr)
            # remove ket from subscript (since we know the amount of elements from the ket to end iteration)
            subscript = subscript[:-len(end_subscr)] + end_subscr[1:]
            # if multiple objects were in the subscript separated by a comma, remove the comma on 1 side
            if(subscript[bra_arg_ind] == ","):
                subscript = subscript[:bra_arg_ind] + subscript[(bra_arg_ind+1):]
            elif(subscript[bra_arg_ind-1] == ","):
                subscript = subscript[:(bra_arg_ind-1)] + subscript[bra_arg_ind:]
            # if only ket was in subscript, remove the leftover brackets
            if(len(subscript) < 3):
                subscript = []

            new_expr = new_expr[:(start_ind)] + "cc_" + "".join(subscript) + new_expr[end_ind:]


            bra_loc = new_expr.find("bra", bra_loc+1)


        return new_expr


    def term2generic_term(self, c2, obj_yast, obj_number):
        r"""
        Helper function to rewrite the instruction given as a list of single_term-s (see _latex2term)
        to a list of GenericTerm-s (see here).

        Differentiates operators from numberical values.

        Parameters
        ----------
        c2: list
            list of single_term-s
        obj_yastn: dict
            dictionary with operators for the generator
        obj_number: dict
            dictionary with parameters for the generator
        """
        # can be used with latex-form interpreter or alone.
        Hterm_list = []
        for ic in c2:
            # create a single Hterm using single_term
            amplitude, positions, operators = float(1), [], []
            for iop in ic.op:
                element, *indicies = iop
                #print("\nindices\n", indicies)
                if element in obj_number:
                    # can have many indicies for cross terms
                    mapindex = tuple([self._map[ind] for ind in indicies]) if indicies else None
                    amplitude *= obj_number[element] if mapindex is None else obj_number[element][mapindex]
                elif element in obj_yast:
                    # is always a single index for each site
                    mapindex = self._map[indicies[0]] if len(indicies) == 1 else YastnError("Operator has to have single index as defined by self._map")
                    positions.append(mapindex)
                    operators.append(obj_yast[element](mapindex))
                else:
                    # the only other option is that is a number, imaginary number is in self.obj_number
                    amplitude *= float(element)
            Hterm_list.append(Hterm(amplitude, positions, operators))
        return Hterm_list

    def split_ltx2terms(self, ltx_str, index=0, start_layer=0, max_terms=-1,
                       left_terminators=['(','[','{'],
                       right_terminators=[')',']','}'],
                       symbols=['+','-',"*","/",',',';',':','=','<','>','|','_','^'],
                       spacers=[' '],
                       escape_chars=['\\'],
                       return_length=False):
        r"""
        Helper function to split a latex term into its constituent terms as seen from the level of the starting point.

        Parameters
        ----------
        ltx_str: string
            the latex term as a string
        index: int
            the index from where to start
        start_layer: int
            the layer where to start (0 is the termination layer and the default)
        left_terminators: [char]
            collection of characters that increase the depth of the term (opening brackets)
        right_terminators: [char]
            collection of characters that decrease the depth of the term (closing brackets)
        symbols: [char]
            collection of characters that are standalone symbols to differentiate from other variables and terms
        spacers: [char]
            collection of characters that indicate the end of the previous term without beginning the next
        escape_chars: [char]
            collection of characters that are used as escape keys for codes and are grouped with the following character
        """

        layer = start_layer
        step = -1 if ltx_str[index] in right_terminators else 1 # if starting on a closing bracket backtrack and negate the step
        curr_ind = index
        terms = [""]
        escape = ''
        #print(ltx_str)
        while(True):
            symbol = ltx_str[curr_ind]

            if(symbol in left_terminators):
                if(layer == 1):
                    terms += [symbol] # no matter if stepping out from 1 to 0 or in from 1 to 2, we want a separate term to start
                else:
                    terms[-1] += symbol
                layer += step
                if(layer == 1):
                    terms += [""]
            elif(symbol in right_terminators):
                if(layer == 1):
                    terms += [symbol] # no matter if stepping out from 1 to 0 or in from 1 to 2, we want a separate term to start
                else:
                    terms[-1] += symbol
                layer -= step
                if(layer == 1):
                    terms += [""]
            elif(layer == 1):
                if(symbol in symbols):
                    terms += [symbol,""]
                elif(symbol in spacers or symbol in escape_chars):
                    terms += [""]
                else:
                    terms[-1] += symbol
            elif(symbol not in escape_chars):
                terms[-1] += symbol

            # if the escape character (usually \) has been found (saved as escape) then insert it back before the following character
            if escape:
                if(terms[-1]):
                    terms[-1] = terms[-1][:-1] + escape + terms[-1][-1]
                else:
                    terms[-2] = terms[-2][:-1] + escape + terms[-2][-1]
                escape = ''

            if(symbol in escape_chars and not escape):
                escape = symbol

            curr_ind += step

            if(max_terms != -1):
                if(len([x for x in terms if x]) > max_terms):
                    terms = [x for x in terms if x]
                    curr_ind -= len(terms[-1])
                    while(ltx_str[curr_ind-1] == " "): curr_ind -= 1
                    terms = terms[:-1]
                    break
            if(layer == 0 and not escape): break
            if(not 0 <= curr_ind < len(ltx_str)): 
                if(start_layer == 0): raise SyntaxError("Latex Term should contain all closing brackets")
                else: break

        # after the loop clean up the resulting terms
        # remove empty string in the array
        terms = [x for x in terms if x]

        # reverse the array and the strings if we were stepping through backwards
        if(step < 0):
            terms = terms[::-1]
            terms = [x[::-1] for x in terms]
        
        if(return_length):
            return terms, curr_ind
        else:
            return terms
    

    def resolve_term(self, term, parameters,
                         operations={
                                     "*": lambda a,b: a*b,
                                     "/": lambda a,b: a/b,
                                     "+": lambda a,b: a+b,
                                     "-": lambda a,b: a-b},
                             brackets=["{","[","("]):
        
        new_term = term.copy()

        for i in range(len(new_term)):
            t = new_term[i]
            if t[0] in brackets or (t[0] == "\\" and t[1] in brackets):
                new_term[i] = self.resolve_term(self.split_ltx2terms(t),parameters=parameters,operations=operations)

        for op in list(operations):
            while(op in new_term):
                ind = new_term.index(op)
                if(ind == 0):
                    # if the found index is 0 consider that it might be a sign, otherwise ignore it
                    if(op == "-"):
                        new_term = [-1*new_term[1]] + new_term[2:]
                    else:
                        new_term = new_term[1:]
                elif(0 < ind < (len(new_term)-1)):
                    # if the ind is inbetween two other entries, perform the operation on those two entries
                    be4 = []
                    after = []
                    if(1 < ind):
                        be4 = new_term[:(ind-2)]
                    if(ind < len(new_term)-2):
                        after = new_term[(ind+2):]

                    args = [self.get_value(new_term[ind-1], parameters), self.get_value(new_term[ind+1], parameters)]
                    new_term = be4 + [operations[op](args[0], args[1])] + after
                else:
                    new_term = new_term[:-1]


        return complex(new_term[0])

        
    def get_value(self, variable, parameters):
        out = -1
        try:
            out = parameters[variable]
        except:
            try:
                out = complex(variable)
            except:
                raise TypeError(f"{variable} couldn't be resolved to a parameter or numerical value")
            
        return out
    

    def gen_tuple_list(self, el_list, dims):

        tuples = []

        for i in range(dims):
            if tuples:
                tuples_temp = tuples.copy()
                tuples = []
                for el in el_list:
                    for tup in tuples_temp:
                        tuples += [tup + (el,)]
                tuples_temp = []

            else:
                for el in el_list:
                    tuples += [(el,)]

        return tuples




