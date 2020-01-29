""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.utils.misc import generate_relative_positions_matrix,\
                            relative_matmul
# from onmt.utils.misc import aeq


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1,
                 max_relative_positions=0, dict_size=None, label_emb=None, opt=None):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        self.label_emb = label_emb

        if dict_size and label_emb:
            self.linear_keys = nn.Linear(model_dim,
                                     (head_count-1) * self.dim_per_head)
            self.linear_query = nn.Linear(model_dim,
                                      (head_count-1) * self.dim_per_head)
            self.linear_par = nn.Linear(model_dim,
                                     1 * self.dim_per_head)
            self.linear_ch = nn.Linear(model_dim,
                                      1 * self.dim_per_head)
        else:
            self.linear_keys = nn.Linear(model_dim,
                                    head_count * self.dim_per_head)
            self.linear_query = nn.Linear(model_dim,
                                    head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)
        self.opt=opt
        if dict_size and label_emb:
            self.p_ch_label = nn.Linear(model_dim*2, dict_size)
            self.p_par_label = nn.Linear(model_dim*2, dict_size)
            if opt.biaffine:
                self.Warc_par_linear = nn.Linear(self.dim_per_head, self.dim_per_head, bias=False)
                self.barc_par_linear = nn.Linear(self.dim_per_head, 1, bias=False)
                self.Warc_ch_linear = nn.Linear(self.dim_per_head, self.dim_per_head, bias=False)
                self.barc_ch_linear = nn.Linear(self.dim_per_head, 1, bias=False)
                self.Wlabel_par_linear = nn.Bilinear(self.dim_per_head, self.dim_per_head, dict_size, bias=False)
                self.blabel_par_linear = nn.Linear(self.dim_per_head*2, dict_size)
                self.Wlabel_ch_linear = nn.Bilinear(self.dim_per_head, self.dim_per_head, dict_size, bias=False)
                self.blabel_ch_linear = nn.Linear(self.dim_per_head*2, dict_size)

        
        self.max_relative_positions = max_relative_positions
        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, attn_type=None, gold_par_attn=None, gold_ch_attn=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        
        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        def predict_ch_label(attn, value):
            
            if not self.opt.biaffine:
                value = unshape(value)
            ch_attn_repeat = torch.repeat_interleave(attn, value.size(2), dim=2) \
                .view(value.size(0), value.size(1), value.size(1), value.size(2))
            value_repeat = torch.repeat_interleave(value, value.size(1), dim=1) \
                .view(value.size(0), value.size(1), value.size(1), value.size(2)).transpose(1,2).contiguous()
            chs = ch_attn_repeat*value_repeat
            ch_label_h = torch.cat([chs, value_repeat],3)
            if self.opt.biaffine:
                w_ch = self.Wlabel_ch_linear(chs, value_repeat) 
                b_ch = self.blabel_ch_linear(ch_label_h)    
                ch_labels = w_ch + b_ch
            else:
                ch_labels = self.p_ch_label(ch_label_h)
            return ch_labels

        def predict_par_label(attn, value):
            if not self.opt.biaffine:
                value = unshape(value)
            par = torch.matmul(attn, value)
            #par_label_h = torch.cat([par, value],2)
            par_label_h = torch.cat([value, par],2)
            if self.opt.biaffine:
                w_par = self.Wlabel_par_linear(par, value) 
                b_par = self.blabel_par_linear(par_label_h)
                par_labels = w_par + b_par
            else:
                par_labels = self.p_ch_label(par_label_h)
                #par_labels = self.p_par_label(par_label_h)
            return par_labels

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            if self.label_emb is not None:
                key_par = self.linear_par(key).unsqueeze(2)
                query_par = self.linear_par(query).unsqueeze(2)
                key_ch = self.linear_ch(key).unsqueeze(2)
                query_ch = self.linear_ch(query).unsqueeze(2)
                query_syn = torch.cat([query_ch, query_par], 2).transpose(1, 2)
                key_syn = torch.cat([key_par, key_ch], 2).transpose(1, 2)

            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            value = shape(value)
            if self.label_emb is None:
                key = shape(key)
            else:
                key = key.view(batch_size, -1, head_count-1, dim_per_head) \
                    .transpose(1, 2)
                value_syn = value[:,:2]
                value = value[:,2:]


        if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))

        if self.label_emb is None:
            query = shape(query)
        else:
            query = query.view(batch_size, -1, head_count-1, dim_per_head) \
                .transpose(1, 2)
            query_room = query[:,0].unsqueeze(1)
            query = query[:,1:]
            key_room = key[:,0].unsqueeze(1)
            key = key[:,1:]
            # 2) Calculate and scale scores.
            query_room = query_room / math.sqrt(dim_per_head)
            query_syn = query_syn / math.sqrt(dim_per_head)
            scores_room = torch.matmul(query_room, key_room.transpose(2, 3)).float()
            scores_syn = torch.matmul(query_syn, key_syn.transpose(2, 3)).float()
            

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
       
        if self.label_emb is not None and self.opt.biaffine:
            query_key = torch.matmul(query[:,2:], key.transpose(2, 3)[:,2:])
            w_par = torch.matmul( self.Warc_par_linear(query[:,0]), key.transpose(2,3)[:,0] )
            w_ch = torch.matmul( self.Warc_ch_linear(query[:,1]), key.transpose(2,3)[:,1] )
            b_par = self.barc_par_linear(query[:,0]).repeat_interleave(query.size(2),dim=2)
            b_ch = self.barc_ch_linear(query[:,1]).repeat_interleave(query.size(2),dim=2)    
            arc_par = w_par + b_par
            arc_ch = w_ch + b_ch
            query_key = torch.cat([arc_par.unsqueeze(1), arc_ch.unsqueeze(1), query_key], dim=1)
        else:
            # batch x num_heads x query_len x key_len
            query_key = torch.matmul(query, key.transpose(2, 3))
        
        
        if self.max_relative_positions > 0 and attn_type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)
            if self.label_emb is not None:
                scores_syn = scores_syn.masked_fill(mask, -1e18)
                scores_room = scores_room.masked_fill(mask, -1e18)
        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)


        drop_attn = self.dropout(attn)
        
        context_original = torch.matmul(drop_attn, value)
        if self.label_emb is not None:
            attn_syn = self.softmax(scores_syn).to(query.dtype)
            drop_attn_syn = self.dropout(attn_syn)
            attn_room = self.softmax(scores_room).to(query.dtype)
            drop_attn_room = self.dropout(attn_room)

            context_original_syn = torch.matmul(drop_attn_syn, value_syn) 
            value_room = torch.cat([value_syn[:,0], value_syn[:,1]], 2).unsqueeze(1)
            context_original_room = torch.matmul(drop_attn_room, value_room).squeeze() 
            context_syn = context_original_room.view(batch_size, -1, 2, dim_per_head).transpose(1, 2) + context_original_syn

            context_original = torch.cat([context_syn , context_original], 1)  
            attn = torch.cat([attn_syn,attn],1)

        

        if self.max_relative_positions > 0 and attn_type == "self":
            context = unshape(context_original
                              + relative_matmul(drop_attn,
                                                relations_values,
                                                False))
        else:
            context = unshape(context_original)
        
        ch_labels = None
        par_labels = None
        
        if gold_ch_attn is not None:
            if self.opt.biaffine:
                par_labels = predict_par_label(gold_par_attn, value[:,0]) 
                ch_labels = predict_ch_label(gold_ch_attn, value[:,1]) 
            else:
                value = torch.cat([value_syn, value], 1)
                par_labels = predict_par_label(gold_par_attn, value) 
                ch_labels = predict_ch_label(gold_ch_attn, value) 

        output = self.final_linear(context)
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()
        second_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 1, :, :] \
            .contiguous()


        return output, top_attn, second_attn, ch_labels, par_labels

    def update_dropout(self, dropout):
        self.dropout.p = dropout
