import torch
import numpy as np
import torch.nn as nn
import t3nsor as t3


class TTEmbedding(nn.Module):
    def __init__(self,
                 init=None,
                 shape=None,
                 voc_size=None,
                 emb_size=None,
                 auto_shapes=None,
                 auto_shape_mode='ascending',
                 auto_shape_criterion='entropy',
                 d=3,
                 tt_rank=8,
                 batch_dim_last=None,
                 padding_idx=None,
                 naive=False):

        super(TTEmbedding, self).__init__()

        if auto_shapes:
            voc_quantization = t3.utils.suggest_shape(
                voc_size, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            emb_quantization = t3.utils.auto_shape(
                emb_size, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [voc_quantization, emb_quantization]
            self.shape = shape

        else:
            self.shape = shape

        if init is None:
            if shape is None:
                raise ValueError('if init is not provided,'
                                 ' please specify shape')
        else:
            self.shape = init.raw_shape


        if init is None:
            init = t3.glorot_initializer(self.shape, tt_rank=tt_rank)

        self.tt_matrix = init.to_parameter()
        self.parameters = self.tt_matrix.parameter

        # for p in self.parameters():
        #    p.name = 'tt_core'

        self.batch_dim_last = batch_dim_last
        self.voc_size = int(np.prod(self.shape[0]))
        self.emb_size = int(np.prod(self.shape[1]))

        self.voc_quant = self.shape[0]
        self.emb_quant = self.shape[1]

        self.padding_idx = padding_idx
        self.naive = naive

    def forward(self, x):

        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size, ]
        x = x.view(-1)

        # x_ind = t3.ind2sub(self.voc_quant, x)
        # rows = t3.gather_rows(self.tt_matrix, x_ind)
        #
        # rows = rows.view(x.shape[0], -1)
        if self.naive:
            full = t3.naive_full(self.tt_matrix)
        else:
            full = self.tt_matrix.full()
        rows = full[x]

        if self.padding_idx is not None:
            rows = torch.where(x.view(-1, 1) != self.padding_idx, rows, torch.zeros_like(rows))

        rows = rows.view(*xshape_new)

        return rows.to(x.device)

class TREmbedding(nn.Module):
    def __init__(self,
                 init=None,
                 shape=None,
                 voc_size=None,
                 emb_size=None,
                 auto_shapes=None,
                 auto_shape_mode='ascending',
                 auto_shape_criterion='entropy',
                 d=3,
                 tt_rank=8,
                 batch_dim_last=None,
                 padding_idx=None,
                 naive=False):

        super(TREmbedding, self).__init__()

        if auto_shapes:
            voc_quantization = t3.utils.suggest_shape(
                voc_size, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            emb_quantization = t3.utils.auto_shape(
                emb_size, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [voc_quantization, emb_quantization]
            self.shape = shape

        else:
            self.shape = shape

        if init is None:
            if shape is None:
                raise ValueError('if init is not provided,'
                                 ' please specify shape')
        else:
            self.shape = init.raw_shape


        if init is None:
            init = t3.glorot_initializer_tr(self.shape, tr_rank=tt_rank)

        self.tr_matrix = init.to_parameter()
        self.parameters = self.tr_matrix.parameter

        # for p in self.parameters():
        #    p.name = 'tt_core'

        self.batch_dim_last = batch_dim_last
        self.voc_size = int(np.prod(self.shape[0]))
        self.emb_size = int(np.prod(self.shape[1]))

        self.voc_quant = self.shape[0]
        self.emb_quant = self.shape[1]

        self.padding_idx = padding_idx
        self.naive = naive

    def forward(self, x):

        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size, ]
        x = x.view(-1)

        # x_ind = t3.ind2sub(self.voc_quant, x)
        # rows = t3.gather_rows(self.tr_matrix, x_ind)
        #
        # rows = rows.view(x.shape[0], -1)
        if self.naive:
            full = t3.naive_full(self.tr_matrix)
        else:
            full = self.tr_matrix.full()
        rows = full[x]

        if self.padding_idx is not None:
            rows = torch.where(x.view(-1, 1) != self.padding_idx, rows, torch.zeros_like(rows))

        rows = rows.view(*xshape_new)

        return rows.to(x.device)


class TTLinear(nn.Module):
    def __init__(self, in_features=None, out_features=None, bias=True, init=None, shape=None,
                 auto_shapes=True, d=3, tt_rank=8, auto_shape_mode='ascending',
                 auto_shape_criterion='entropy', naive=False,
                 reverse_out_shape=False, factorize_smaller_dim=True, use_scripted_mul=False,
                 cores_nonlinearity=None
                 ):
        super(TTLinear, self).__init__()

        if auto_shapes:
            if in_features is None or out_features is None:
                raise ValueError("Shape is not specified")

            in_quantization = t3.utils.auto_shape(
                in_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            out_quantization = t3.utils.auto_shape(
                out_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            if not factorize_smaller_dim and out_features < in_features:
                out_quantization = [1] * (d-1) + [out_features]
                if auto_shape_mode != 'ascending':
                    out_quantization.reverse()

            if not factorize_smaller_dim and in_features < out_features:
                in_quantization = [1] * (d-1) + [in_features]
                if auto_shape_mode != 'ascending':
                    in_quantization.reverse()

            if reverse_out_shape:
                out_quantization.reverse()

            shape = [in_quantization, out_quantization]

        if init is None:
            if shape is None:
                raise ValueError(
                    "if init is not provided, please specify shape, or set auto_shapes=True")
        else:
            shape = init.raw_shape

        self.tt_rank = tt_rank
        if init is None:
            init = t3.glorot_initializer(shape, tt_rank=tt_rank)

        self.shape = shape
        self.weight = init.to_parameter()
        self.parameters = self.weight.parameter
        if naive:
            self.mm_op = t3.naive_dense_tt_matmul
        else:
            self.mm_op = lambda matrix_a, tt_matrix_b: t3.dense_tt_matmul(
                matrix_a, tt_matrix_b,
                use_scripted_mul=use_scripted_mul,
                cores_nonlinearity=cores_nonlinearity
            )
        if bias:
            self.bias = torch.nn.Parameter(1e-3 * torch.ones(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weight = self.weight
        if self.bias is None:
            return self.mm_op(x, weight)
        else:
            return self.mm_op(x, weight) + self.bias

    def reset_parameters(self):
        self.weight = t3.glorot_initializer(self.shape, tt_rank=self.tt_rank).to_parameter()
        self.parameters = self.weight.parameter


class TTBias(nn.Module):
    def __init__(self, in_features=None, out_features=None, c=1e-3,
                 init=None, shape=None,
                 auto_shapes=True, d=3, tt_rank=2, auto_shape_mode='ascending',
                 auto_shape_criterion='entropy', naive=False,
                 ):
        super(TTBias, self).__init__()

        if auto_shapes:
            if in_features is None or out_features is None:
                raise ValueError("Shape is not specified")

            in_quantization = t3.utils.auto_shape(
                in_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            out_quantization = t3.utils.auto_shape(
                out_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [in_quantization, out_quantization]

        if init is None:
            if shape is None:
                raise ValueError(
                    "if init is not provided, please specify shape, or set auto_shapes=True")
        else:
            shape = init.raw_shape

        if init is None:
            init = t3.const_initializer(shape, tt_rank=tt_rank, scale_const=c)

        self.shape = shape
        self.weight = init.to_parameter()
        self.parameters = self.weight.parameter

        # unpacked = self.weight.tt_cores[0]
        # for tt_core in self.weight.tt_cores[1:]:
        #     unpacked = torch.tensordot(unpacked, tt_core, dims=[[-1],[0]])
        # print("init bias", unpacked)

    def forward(self, x):
        unpacked = self.weight.tt_cores[0]
        for tt_core in self.weight.tt_cores[1:]:
            unpacked = torch.tensordot(unpacked, tt_core, dims=[[-1],[0]])

        # TODO: if normalized weight is not a vector, specify correct shape
        return unpacked.reshape(-1) + x



class TTLayerNorm(nn.Module):
    def __init__(self, in_features=None, out_features=None, eps=1e-6,
                 bias=False, init=None, shape=None,
                 auto_shapes=True, d=3, tt_rank=2, auto_shape_mode='ascending',
                 auto_shape_criterion='entropy', naive=False,
                 scale_const=1.0
                 ):
        super(TTLayerNorm, self).__init__()

        if auto_shapes:
            if in_features is None or out_features is None:
                raise ValueError("Shape is not specified")

            in_quantization = t3.utils.auto_shape(
                in_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            out_quantization = t3.utils.auto_shape(
                out_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [in_quantization, out_quantization]

        if init is None:
            if shape is None:
                raise ValueError(
                    "if init is not provided, please specify shape, or set auto_shapes=True")
        else:
            shape = init.raw_shape

        if init is None:
            init = t3.const_initializer(shape, tt_rank=tt_rank, scale_const=scale_const)


        self.shape = shape
        self.weight = init.to_parameter()
        self.parameters = self.weight.parameter
        self.variance_epsilon = eps

        # unpacked = self.weight.tt_cores[0]
        # for tt_core in self.weight.tt_cores[1:]:
        #     unpacked = torch.tensordot(unpacked, tt_core, dims=[[-1],[0]])
        # print("init layernorm", unpacked)



    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        unpacked = self.weight.tt_cores[0]
        for tt_core in self.weight.tt_cores[1:]:
            unpacked = torch.tensordot(unpacked, tt_core, dims=[[-1],[0]])

        # TODO: if normalized weight is not a vector, specify correct shape
        return unpacked.reshape(-1) * hidden_states


class TRLinear(nn.Module):
    def __init__(self, in_features=None, out_features=None, bias=True, init=None, shape=None,
                 auto_shapes=True, d=3, tt_rank=8, auto_shape_mode='ascending',
                 auto_shape_criterion='entropy', naive=False
                 ):
        super(TRLinear, self).__init__()

        if auto_shapes:
            if in_features is None or out_features is None:
                raise ValueError("Shape is not specified")

            in_quantization = t3.utils.auto_shape(
                in_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            out_quantization = t3.utils.auto_shape(
                out_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [in_quantization, out_quantization]

        if init is None:
            if shape is None:
                raise ValueError(
                    "if init is not provided, please specify shape, or set auto_shapes=True")
        else:
            shape = init.raw_shape

        if init is None:
            init = t3.glorot_initializer_tr(shape, tr_rank=tt_rank)

        self.shape = shape
        self.weight = init.to_parameter()
        self.parameters = self.weight.parameter
        if naive:
            self.mm_op = t3.naive_dense_tr_matmul
        else:
            raise ValueError('Not implemented, use naive option.')
        if bias:
            self.bias = torch.nn.Parameter(1e-3 * torch.ones(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weight = self.weight
        if self.bias is None:
            return self.mm_op(x, weight)
        else:
            return self.mm_op(x, weight) + self.bias