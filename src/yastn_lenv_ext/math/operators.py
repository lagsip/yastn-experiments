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
        #print("Code: \n", opcode)
        new_code = opcode + "cc"
        #print("Ne code: \n", new_code)
        tensor = getattr(meta_op, opcode)()
        #print("Old tensor: \n", tensor.to_numpy())
        new_operators[new_code] = lambda j, new_tensor=tensor.transpose(): new_tensor
        #print("New tensor: \n", new_operators[new_code].to_numpy())
    return new_operators



