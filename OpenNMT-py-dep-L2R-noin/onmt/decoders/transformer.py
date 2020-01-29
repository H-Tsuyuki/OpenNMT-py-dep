"""
Implementation of "Attention is All You Need"
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, opt, d_model, heads, d_ff, dropout, attention_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 aan_useffn=False, dict_size=None, label_emb=None):
        super(TransformerDecoderLayer, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=dropout,
                max_relative_positions=max_relative_positions, dict_size=dict_size, label_emb=label_emb, opt=opt)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model,
                                              dropout=attention_dropout,
                                              aan_useffn=aan_useffn)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, step=None, gold_par_attn=None, gold_ch_attn=None):
        
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``

        """
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
                #future_mask = future_mask.triu_(0).view(1, tgt_len, tgt_len)
                #future_mask[0,0,0]=0
            # BoolTensor was introduced in pytorch 1.2
            try:
                future_mask = future_mask.bool()
            except AttributeError:
                pass
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
        
#        elif step!=0 and synsa:
#            self_mask = torch.zeros(
#                [1,1, step+1],
#                device=tgt_pad_mask.device,
#                dtype=torch.uint8)
#            self_mask[:,:,-1]=1
#            try:
#                self_mask = self_mask.bool()
#            except AttributeError:
#                pass
#            dec_mask = torch.gt(self_mask, 0)
        
        
        input_norm = self.layer_norm_1(inputs)
        if isinstance(self.self_attn, MultiHeadedAttention):
            query, tgt_attn, second_attn, ch_labels, par_labels = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         attn_type="self", gold_par_attn=gold_par_attn, gold_ch_attn=gold_ch_attn)
        elif isinstance(self.self_attn, AverageAttention):
            query, attn = self.self_attn(input_norm, mask=dec_mask,
                                         layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, src_attn, _, _, _ = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      attn_type="context")
        output = self.feed_forward(self.drop(mid) + query)
        return output, src_attn, tgt_attn, second_attn, dec_mask, ch_labels, par_labels 

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.context_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout


class TransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, opt, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, label_emb, max_relative_positions, aan_useffn, dict_size):
        super(TransformerDecoder, self).__init__()
        self.embeddings = embeddings
        self.label_embeddings = label_emb
        self.dict_size = dict_size

        # Decoder State
        self.state = {}

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(opt, d_model, heads, d_ff, dropout,
             attention_dropout, self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions,
             aan_useffn=aan_useffn)
             for i in range(num_layers-1)])
        
        self.syntactic_attention = \
            TransformerDecoderLayer(opt, d_model, heads, d_ff, dropout,
             attention_dropout, self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions,
             aan_useffn=aan_useffn, dict_size=dict_size, label_emb=label_emb)
            
        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings, label_emb, dict_size):
        """Alternate constructor."""
        return cls(
            opt,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.dropout,
            embeddings,
            label_emb,
            opt.max_relative_positions,
            opt.aan_useffn,
            dict_size)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank, gold_par_attn=None, gold_ch_attn=None, step=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)
       
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        
        emb = F.relu(self.embeddings(tgt, step=step))
        assert emb.dim() == 3  # len x batch x embedding_dim
        output = emb.transpose(0,1)
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()
        
        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]
        
        for i, layer in enumerate(self.transformer_layers[:3]):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, src_attn, tgt_attn, _, dec_mask, _, _ = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step)

        layer_cache = self.state["cache"]["synsa"] \
            if step is not None else None
        output, src_attn, tgt_par_attn, tgt_ch_attn, dec_mask, ch_labels, par_labels = self.syntactic_attention( 
                output,
                src_memory_bank,
               src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
                gold_par_attn=gold_par_attn,
                gold_ch_attn=gold_ch_attn)
 
        for i, layer in enumerate(self.transformer_layers[3:]): 
            layer_cache = self.state["cache"]["layer_{}".format(i+3)] \
                if step is not None else None
            output, src_attn, tgt_attn, _, dec_mask, _, _ = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        src_attn = src_attn.transpose(0, 1).contiguous()

        attns = {
                "src": src_attn,
                "tgt_par": tgt_par_attn,
                "tgt_ch": tgt_ch_attn
                }

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns, dec_mask, ch_labels, par_labels

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth),
                                                    device=memory_bank.device)
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache
        layer_cache = {"memory_keys": None, "memory_values": None}
        layer_cache["self_keys"] = None
        layer_cache["self_values"] = None
        self.state["cache"]["synsa"] = layer_cache
        self.state["cache"]["word_emb"] = None

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
