"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax
import onmt.Constants as Constants

def build_loss_compute(model, data, opt, train=True):
    """
    Returns a LossCompute subclass which wraps around an nn.Module subclass
    (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
    object allows this loss to be computed in shards and passes the relevant
    data to a Statistics object which handles training/validation logging.
    Currently, the NMTLossCompute class handles all loss computation except
    for when using a copy mechanism.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    padding_idx = Constants.PAD
    unk_idx = Constants.UNK

    criterion = {} 
    #if opt.label_smoothing > 0 and train:
    if True:
        criterion["word"] = LabelSmoothingLoss(
            opt.label_smoothing, len(data["dict"]["tgt"]), ignore_index=padding_idx
        )
        criterion["par_order"] = LabelSmoothingLoss(
            opt.label_smoothing, 2 
        )
        criterion["label"] = LabelSmoothingLoss(
            opt.label_smoothing, len(data["dict"]["tgt_label"]), ignore_index=padding_idx, label_dict=data["dict"]["tgt_label"]
        )
    else:
        criterion["word"] = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
        criterion["par_order"] = nn.NLLLoss(ignore_index=5, reduction='sum')
        criterion["label"] = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
    
    compute = NMTLossCompute(
            criterion, model.generator, opt, lambda_coverage=opt.lambda_coverage)
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion, generator):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator

    @property
    def padding_idx(self):
        return self.criterion["word"].ignore_index

    def _make_shard_state(self, batch, output, range_, attns=None, ch_labels=None, par_labels=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def __call__(self,
                 batch,
                 output,
                 attns,
                 ch_labels,
                 par_labels,
                 dec_mask,
                 normalization=1.0,
                 shard_size=0,
                 trunc_start=0,
                 trunc_size=None):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
        if trunc_size is None:
            trunc_size = batch["tgt"].size(0) - trunc_start
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(batch, output, trunc_range, attns, ch_labels, par_labels)
        shard_size = 0
        if shard_size == 0:
            loss, stats = self._compute_loss(batch, dec_mask, **shard_state)
            return loss / float(normalization), stats
        batch_stats = onmt.utils.Statistics()
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return None, batch_stats

    def _stats(self, word_loss, label_loss, par_order_loss, ch_label_loss, ch_order_loss, \
            word_prob, g_word, label_prob, g_label, par_prob, g_par, ch_label_prob, g_ch_label):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = word_prob.max(2)[1]
        non_padding = g_word.ne(self.padding_idx)
        num_correct_word = pred.eq(g_word).masked_select(non_padding).sum().item()
        num_non_padding_word = non_padding.sum().item()

        pred = label_prob.max(2)[1]
        non_padding = g_label.ne(self.padding_idx)
        num_correct_label = pred.eq(g_label).masked_select(non_padding).sum().item()
        num_non_padding_label = non_padding.sum().item()
        
        pred = ch_label_prob.max(3)[1]
        non_padding_ch_label = g_ch_label.ne(self.padding_idx)
        num_correct_ch_label = pred.eq(g_ch_label).masked_select(non_padding_ch_label).sum().item()
        num_non_padding_ch_label = non_padding_ch_label.sum().item()
        
        pred = par_prob.max(2)[1]
        num_correct_par = pred.eq(g_par).masked_select(non_padding).sum().item()
        num_non_padding_par = non_padding.sum().item()

        if not num_non_padding_par==num_non_padding_word==num_non_padding_label:
            from IPython.core.debugger import Pdb; Pdb().set_trace()
 
        return onmt.utils.Statistics(
            word_loss.item(), label_loss.item(), par_order_loss.item(), ch_label_loss.item(), ch_order_loss.item(), \
            num_non_padding_word, num_correct_word, num_correct_label, num_correct_par, num_correct_ch_label)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100, label_dict=None):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()
        self.label_dict = label_dict
        if ignore_index!=-100:
            self.smoothing_value = label_smoothing / (tgt_vocab_size - 2)
            one_hot = torch.full((tgt_vocab_size,), self.smoothing_value).cuda()
            one_hot[self.ignore_index] = 0
            self.register_buffer('one_hot', one_hot.unsqueeze(0))
            self.confidence = 1.0 - label_smoothing
        else:
            self.label_smoothing = label_smoothing


    def forward(self, output, target, dec_mask=None, opt=None):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        
        if self.ignore_index==-100:
            if len(target.shape)==3:
                model_prob = target
            else:
                one_hot = torch.full((output.size(2),), 0).cuda()
                self.register_buffer('one_hot', one_hot.unsqueeze(0))

                model_prob = self.one_hot.repeat(target.size(0), target.size(1), 1)
                model_prob.scatter_(2, target.unsqueeze(2), 1)
        else:
            if len(output.shape)==4:
                model_prob = self.one_hot.repeat(target.size(0), target.size(1), target.size(2), 1)
                model_prob.scatter_(3, target.unsqueeze(3), self.confidence)
                model_prob.masked_fill_((target == self.ignore_index).unsqueeze(3), 0)
                #if opt.par_a:    
                #    model_prob.masked_fill_((target == self.label_dict["no_child"]).unsqueeze(3), 0)
            else:
                model_prob = self.one_hot.repeat(target.size(0), target.size(1), 1)
                model_prob.scatter_(2, target.unsqueeze(2), self.confidence)
                model_prob.masked_fill_((target == self.ignore_index).unsqueeze(2), 0)
                #if self.label_dict is not None and opt.par_a :
                #    model_prob.masked_fill_((target == self.label_dict["no_parent"]).unsqueeze(2), 0)
        
        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterion, generator, opt, normalization="sents",
                 lambda_coverage=0.0):
        super(NMTLossCompute, self).__init__(criterion, generator)
        self.logsm = nn.LogSoftmax(dim=-1)
        self.opt = opt

    def _make_shard_state(self, batch, output, range_, attns, ch_labels, par_labels):
        
        shard_state = {
            "output": output,
            "word": batch["tgt"][range_[0] + 1: range_[1], :, 0],
            "label": batch["tgt_par_label"][range_[0] + 1: range_[1], :, 0],
            "par_order": batch["tgt_par_pos"][ :, range_[0] + 1: range_[1], range_[0] + 1: range_[1] ],
            "par_order_idx": batch["tgt_par_pos_idx"][ :, range_[0] + 1: range_[1]]-1,
            "g_ch_label": batch["tgt_ch_label"][: ,range_[0] + 1: range_[1], range_[0] + 1: range_[1]],
            "ch_order": batch["tgt_ch_pos"][ :, range_[0] + 1: range_[1], range_[0] + 1: range_[1] ],
            "src_attn": attns.get("src"),
            "tgt_par_attn": attns.get("tgt_par"),
            "tgt_ch_attn": attns.get("tgt_ch"),
            "ch_label_prob": ch_labels,
            "par_label_prob": par_labels,
        }
        return shard_state

    def _compute_loss(self, batch, dec_mask, output, word, label, par_order, par_order_idx, g_ch_label, ch_order,\
                    src_attn, tgt_par_attn, tgt_ch_attn, ch_label_prob, par_label_prob):
        word_prob = self.generator(output)
        ch_label_prob = self.logsm(ch_label_prob)
        label_prob = self.logsm(par_label_prob)
        loss_word = self.criterion["word"](word_prob, word.transpose(0,1))
        loss_par_label = self.criterion["label"](label_prob, label.transpose(0,1), opt=self.opt)
        loss_ch_label = self.criterion["label"](ch_label_prob, g_ch_label, opt=self.opt)

        
        par_attn = torch.clamp(tgt_par_attn, min=1e-5, max=1-(1e-5))
        par_attn = torch.log(par_attn)
        
        ch_attn = torch.clamp(tgt_ch_attn, min=1e-5, max=1-(1e-5))
        ch_attn = torch.log(ch_attn)
        
        loss_par_order = self.criterion["par_order"](par_attn, par_order_idx, dec_mask)
        loss_ch_order = self.criterion["par_order"](ch_attn, ch_order, dec_mask)


        #loss = loss_word  + 0.1*loss_par_label + 0.1*loss_ch_label + 0.2*loss_par_order + 0.01*loss_ch_order
        if self.opt.loss_type=='base' or self.opt.loss_type=='test':
            loss = loss_word
        elif self.opt.loss_type=='p':
            loss = loss_word  + 0.1*loss_par_order
        elif self.opt.loss_type=='pl':
            loss = loss_word  + 0.2*loss_par_label + 0.1*loss_par_order
        elif self.opt.loss_type=='c':
            loss = loss_word + 0.01*loss_ch_order
        elif self.opt.loss_type=='lc':
            loss = loss_word  + 0.2*loss_par_label + 0.01*loss_ch_order
        elif self.opt.loss_type=='cl':
            loss = loss_word  + 0.01*loss_ch_order + 0.1*loss_ch_label
        elif self.opt.loss_type=='pc':
            loss = loss_word  + 0.1*loss_par_order + 0.01*loss_ch_order
        elif self.opt.loss_type=='plc':
            loss = loss_word  + 0.2*loss_par_label + 0.1*loss_par_order + 0.01*loss_ch_order
        elif self.opt.loss_type=='plcl':
            if self.opt.par_a:    
                loss = loss_word  + 0.2*loss_par_label + 0.1*loss_par_order + 0.1*loss_ch_order + 0.2*loss_ch_label
            else:
                loss = loss_word  + 0.3*loss_par_label + 0.2*loss_par_order + 0.2*loss_ch_order + 0.3*loss_ch_label

        stats = self._stats(loss_word.clone(), loss_par_label.clone(), loss_par_order.clone(), loss_ch_label.clone(), loss_ch_order.clone(), \
            word_prob, word.transpose(0,1), label_prob, label.transpose(0,1), par_attn, par_order_idx, ch_label_prob, g_ch_label)
        return loss, stats

def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
