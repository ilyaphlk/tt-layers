from t3nsor import TensorTrainBatch
from t3nsor import TensorTrain
import torch
from typing import List
import string


def gather_rows(tt_mat, inds):
    """
    inds -- list of indices of shape batch_size x d
    d = len(tt_mat.raw_shape[1])
    """
    cores = tt_mat.cores
    slices = []
    batch_size = int(inds.shape[0])

    ranks = [int(cores.shape[0]) for cores in tt_mat.cores] + [1, ]

    for k, core in enumerate(cores):
        i = inds[:, k]
        cur_slice = torch.index_select(core, 1, i)
        # r x B x M x r

        if k == 0:
            res = cur_slice.transpose(0, 1)
            # B x r x M x r

        else:
            res = res.contiguous().view(batch_size, -1, ranks[k])
            # B x rM x r
            curr_core = cur_slice.view(ranks[k], batch_size, -1)
            # r x B x Mr
            res = torch.einsum('oqb,bow->oqw', (res, curr_core))
    res = torch.einsum('i...i->...', res.view(batch_size, ranks[0], res.shape[1] // ranks[0], -1, ranks[0]).transpose(0, 1))

    return res

def transpose(tt_matrix):
    cores = []
    for core in tt_matrix.tt_cores:
        cores.append(core.transpose(1, 2))
    return TensorTrain(cores)


def tt_dense_matmul(tt_matrix_a, matrix_b):
    """Multiplies a TT-matrix by a regular matrix, returns a regular matrix.
    Args:
    tt_matrix_a: `TensorTrain` object containing a TT-matrix of size M x N
    matrix_b: torch.Tensor of size N x P
    Returns
    torch.Tensor of size M x P
    """

    ndims = tt_matrix_a.ndims
    a_columns = tt_matrix_a.shape[1]
    b_rows = matrix_b.shape[0]
    if a_columns is not None and b_rows is not None:
        if a_columns != b_rows:
            raise ValueError('Arguments shapes should align got %d and %d instead.' %
                       (tt_matrix_a.shape, matrix_b.shape))

    a_shape = tt_matrix_a.shape
    a_raw_shape = tt_matrix_a.raw_shape
    b_shape = matrix_b.shape
    a_ranks = tt_matrix_a.ranks

    # If A is (i0, ..., id-1) x (j0, ..., jd-1) and B is (j0, ..., jd-1) x K,
    # data is (K, j0, ..., jd-2) x jd-1 x 1
    data = matrix_b.transpose(0, 1)
    #data = data.view(-1, a_raw_shape[1][-1], 1)
    data = data.reshape(-1, a_raw_shape[1][-1], 1)

    for core_idx in reversed(range(ndims)):
        curr_core = tt_matrix_a.tt_cores[core_idx]
        # On the k = core_idx iteration, after applying einsum the shape of data
        # becomes ik x (ik-1..., id-1, K, j0, ..., jk-1) x rank_k
        data = torch.einsum('aijb,rjb->ira', curr_core, data)
        if core_idx > 0:
          # After reshape the shape of data becomes
          # (ik, ..., id-1, K, j0, ..., jk-2) x jk-1 x rank_k
            new_data_shape = (-1, a_raw_shape[1][core_idx - 1], a_ranks[core_idx])
            data = data.contiguous().view(new_data_shape)

    # At the end the shape of the data is (i0, ..., id-1) x K
    return data.view(a_shape[0], b_shape[1])

@torch.jit.script
def mul_cores_fast(data, tt_cores: List[torch.Tensor]):
    for tt_core in tt_cores:
        sh = data.shape
        res_shape = sh[0:1] + sh[2:-1] + tt_core.shape[2:]
        idx = list(range(len(sh)))
        idx = idx[0:1] + idx[2:] + idx[1:2]
        data = torch.matmul(data.permute(idx).reshape(-1, sh[-1]*sh[1]), tt_core.reshape(sh[-1]*sh[1], -1)).reshape(res_shape)
    return data


def dense_tt_matmul(matrix_a, tt_matrix_b, use_scripted_mul=False, cores_nonlinearity=None):
    ndims = tt_matrix_b.ndims
    a_columns = matrix_a.shape[-1]
    b_rows = tt_matrix_b.shape[0]

    if a_columns is not None and b_rows is not None:
        if a_columns != b_rows:
            raise ValueError(f'Arguments shapes should align got {matrix_a.shape} and {tt_matrix_b.shape} instead.')

    a_shape = matrix_a.shape
    b_shape = tt_matrix_b.shape
    b_raw_shape = tt_matrix_b.raw_shape
    # print("a_shape", a_shape)
    # print("b_shape", b_shape)
    # print("b_raw_shape", b_raw_shape)

    data = matrix_a

    new_shape = [-1, ] + b_raw_shape[0] + [1, ]
    # print("new_shape", new_shape)

    data = data.view(*new_shape)
    # print("data.shape", data.shape)

    if use_scripted_mul:
        data = mul_cores_fast(data, tt_matrix_b.tt_cores)
    else:
        # correct but slow for large ndims
        for core_idx in range(ndims):
            curr_core = tt_matrix_b.tt_cores[core_idx]
            data = torch.tensordot(data, curr_core, dims=[[1, -1], [1, 0]])
            if cores_nonlinearity is not None:
                data = cores_nonlinearity(data)

    # print("data.shape after tdot", data.shape)

    if len(a_shape) == 2:
        return data.view(a_shape[0], b_shape[1])
    elif len(a_shape) == 3:
        return data.view(a_shape[0], a_shape[1], b_shape[1])
  

def naive_dense_tt_matmul(matrix_a, tt_matrix_b):
    ndims = tt_matrix_b.ndims
    a_columns = matrix_a.shape[-1]
    b_rows = tt_matrix_b.shape[0]
    if a_columns is not None and b_rows is not None:
        if a_columns != b_rows:
            raise ValueError('Arguments shapes should align got {} and {} instead.'.format(matrix_a.shape, tt_matrix_b.shape))

    a_shape = matrix_a.shape
    b_shape = tt_matrix_b.shape

    # assert ndims == 3

    # core0 = tt_matrix_b.tt_cores[0]  # 1 x n x m x r
    # core1 = tt_matrix_b.tt_cores[1]  # r x n x m x r
    # core2 = tt_matrix_b.tt_cores[2]  # r x n x m x 1

    # input = matrix_a.view(-1, core0.shape[1], core1.shape[1], core2.shape[1])
    # B = input.shape[0]
    a_view = matrix_a.view(-1, *list(map(lambda x: x.shape[1], tt_matrix_b.tt_cores)))
    B = a_view.shape[0]

    # full = torch.einsum('abcd,defg,ghij->bcefhi', core0, core1, core2)
    def core_mul_str(n_cores):
        alpha_str = string.ascii_lowercase + string.ascii_uppercase
        inputs = [alpha_str[i:i+4] for i in range(0, 3*n_cores, 3)]
        output = "".join([elem[1:3] for elem in inputs])
        return ",".join(inputs)+"->"+output
    full = torch.einsum(core_mul_str(len(tt_matrix_b.tt_cores)), *tt_matrix_b.tt_cores)

    # res = torch.einsum('abcd,bqcsdx->aqsx', input, full)
    def res_mul_str(n_cores):
        alpha_str = string.ascii_lowercase + string.ascii_uppercase
        in_str = alpha_str[:n_cores+1]
        full_str = "".join([alpha_str[i+1]+alpha_str[n_cores+1+i] for i in range(n_cores)])
        res_str = alpha_str[0]+"".join([alpha_str[n_cores+1+i] for i in range(n_cores)])
        return in_str+","+full_str+"->"+res_str
    res = torch.einsum(res_mul_str(len(tt_matrix_b.tt_cores)), a_view, full)


    if len(a_shape) == 2:
        return res.contiguous().view(a_shape[0], b_shape[1])
    elif len(a_shape) == 3:
        return res.contiguous().view(a_shape[0], a_shape[1], b_shape[1])


    #return res.contiguous().view(B, -1)


def naive_full(tt_a):
    ndims = tt_a.ndims
    assert ndims == 3
    try:
        # TT-Embedding
        core0, core1, core2 = tt_a.tt_cores
    except:
        # TR-Embedding
        core0, core1, core2 = tt_a.tr_cores

    full = torch.einsum('abcd,defg,ghia->bcefhi', core0, core1, core2)
    full = full.reshape(tt_a.shape[0], tt_a.shape[1])
    return full

def naive_dense_tr_matmul(matrix_a, tr_matrix_b):
    ndims = tr_matrix_b.ndims
    a_columns = matrix_a.shape[1]
    b_rows = tr_matrix_b.shape[0]
    if a_columns is not None and b_rows is not None:
        if a_columns != b_rows:
            raise ValueError('Arguments shapes should align got %d and %d instead.' %
                             (matrix_a.shape, tr_matrix_b.shape))

    assert ndims == 3

    core0 = tr_matrix_b.tr_cores[0]  # 1 x n x m x r
    core1 = tr_matrix_b.tr_cores[1]  # r x n x m x r
    core2 = tr_matrix_b.tr_cores[2]  # r x n x m x 1

    input = matrix_a.view(-1, core0.shape[1], core1.shape[1], core2.shape[1])
    B = input.shape[0]

    full = torch.einsum('abcd,defg,ghia->bcefhi', core0, core1, core2)
    res = torch.einsum('abcd,bqcsdx->aqsx', input, full)
    return res.contiguous().view(B, -1)

