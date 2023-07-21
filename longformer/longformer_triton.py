from typing import List
import math
import torch
from torch import nn
import torch.nn.functional as F
from longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from longformer.sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv
from longformer.sliding_chunks import sliding_chunks_no_overlap_matmul_qk, sliding_chunks_no_overlap_matmul_pv
from transformers.modeling_roberta import RobertaConfig, RobertaModel, RobertaForMaskedLM
from sparta.opset import *
from sparta.opset.triton_dynamic_sparse_attention import TritonDynamicAttention
import time

class Longformer(RobertaModel):
    def __init__(self, config):
        super(Longformer, self).__init__(config)
        self.spa = TritonDynamicAttention(32, 32, config.num_attention_heads, global_model=True)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i, spa=self.spa)


    def forward(self, attention_mask=None, **kwargs):
        # print(self.config.num_attention_heads)
        attention_mask = attention_mask.repeat(self.config.num_attention_heads, 1, 1).to(torch.int32)
        # print(attention_mask.shape)
        self.spa.set_global_mask(attention_mask, True, 32, 32, self.config.num_attention_heads)
        # self.spa.set_global_static_attention(attention_mask[0])
        # self.spa.set_global_dynamic_attention(dynamic)
        return super(Longformer, self).forward(attention_mask=attention_mask, **kwargs)


class LongformerForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super(LongformerForMaskedLM, self).__init__(config)
        self.spa = TritonDynamicAttention(32, 32, config.num_attention_heads)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.roberta.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i, spa=self.spa)

    def forward(self, **kwargs):
        # DynamicSparseAttention.set_global_sparse_pattern(attention_mask[0])
        return super(Longformer, self).forward(**kwargs)


class LongformerConfig(RobertaConfig):
    def __init__(self, attention_window: List[int] = None, attention_dilation: List[int] = None,
                 autoregressive: bool = False, attention_mode: str = 'sliding_chunks', **kwargs):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Longformer
                selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'n2', 'sliding_chunks_no_overlap']


class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id, spa=None):
        super(LongformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        self.attention_window = config.attention_window[self.layer_id]
        self.attention_dilation = config.attention_dilation[self.layer_id]
        self.attention_mode = config.attention_mode
        self.autoregressive = config.autoregressive
        assert self.attention_window > 0
        assert self.attention_dilation > 0
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'sliding_chunks_no_overlap']
        if self.attention_mode in ['sliding_chunks', 'sliding_chunks_no_overlap']:
            assert not self.autoregressive  # not supported
            assert self.attention_dilation == 1  # dilation is not supported
        self.spa = spa
        self.time = 0

    def predict(self, q, k, v, mask):
        # mask  = (mask == 0).to(torch.int32)
        # mask = mask.squeeze(1)
        attn = self.spa(q, k, v)
        return attn

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        '''
        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention
        '''
        torch.cuda.synchronize()
        st = time.time()
        assert encoder_hidden_states is None, "`encoder_hidden_states` is not supported and should be None"
        assert encoder_attention_mask is None, "`encoder_attention_mask` is not supported and shiould be None"

        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.reshape(-1, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        # import joblib
        # joblib.dump([ii.contiguous() for ii in [q, k, v]], "sparse_msa_inputs.pkl")
        # assert False
        context_layer = self.predict(q, k, v, attention_mask)
        context_layer = context_layer.transpose(1, 2).reshape(bsz, seq_len, embed_dim).contiguous()
        attn_weights = None
        outputs = (context_layer, attn_weights) if output_attentions else (context_layer,)
        torch.cuda.synchronize()
        end = time.time()
        self.time += (end - st)
        return outputs
