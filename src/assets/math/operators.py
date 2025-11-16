import numpy as np

def compl_conjugate(meta_op) -> dict:
    r"""
    Helper function to complex conjugate an operator.

    Parameters
    ----------
    meta_op: meta_operators
        The operator to complex conjugate
    """

    new_operators = {}
    for opcode in meta_op.operators:
        new_code = opcode + "cc"
        tensor = getattr(meta_op, opcode)()
        #print("Old tensor: \n", tensor)
        new_tensor = tensor.clone().conj_blocks()
        #print("New tensor: \n", new_tensor)
        new_operators[new_code] = lambda j: new_tensor
    return new_operators



