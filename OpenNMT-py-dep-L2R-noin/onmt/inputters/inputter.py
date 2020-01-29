# -*- coding: utf-8 -*-
import glob
import os
import codecs
import math
import random
import numpy as np

from collections import Counter, defaultdict
from itertools import chain, cycle

import torch

from onmt.utils.logging import logger
import onmt.Constants as Constants 

import gc


class DatasetLazyIterTest(object):
    """Yield data from sharded dataset files.

    Args:
        dataset_paths: a list containing the locations of dataset files.
        fields (dict[str, Field]): fields dict for the
            datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: See :class:`OrderedIterator` ``device``.
        is_train (bool): train or valid?
    """

    def __init__(self, dataset,  batch_size, device, is_train, repeat=True):
        self.dataset = list(zip(*[dataset['src'], dataset['tgt'], dataset['tgt_par'], dataset['tgt_ch'], dataset["tgt_ch_label"], dataset['src_raw'],dataset['tgt_raw'],dataset['indices']])) #[[[src1],[tgt1]],[[src2],[tgt2]]...] datasize * 5 * n_token
        logger.info('number of examples: %d' % len(self.dataset))
        self.dataset = [self.dataset[i:i+batch_size] for i in range(0,len(self.dataset),batch_size)] #n_iteration * batch_size * 2 *n_token
        self.batch_size = batch_size
        self.device = device
        self.is_train = is_train
        self.repeat = repeat
        if self.is_train and self.repeat:
            random.shuffle(self.dataset)
            self.dataset = cycle(self.dataset)


    def __iter__(self):
        num_batches = 0
        for idx, batch in enumerate(self.dataset):
            batch_dict={}
            batch = sorted(batch, key=lambda x : (len(x[0]),len(x[1])))[::-1]
            L_src = [len(i[0]) for i in batch]
            L_tgt = [len(i[1]) for i in batch]
            batch_dict["src"] = torch.LongTensor(list(zip(*[ i[0] + [Constants.PAD]*(max(L_src)-len(i[0])) for i in batch]))).view(-1,len(batch),1).to(self.device)
            batch_dict["tgt"] = torch.LongTensor(list(zip(*[ i[1] + [Constants.PAD]*(max(L_tgt)-len(i[1])) for i in batch]))).view(-1,len(batch),1).to(self.device) 
            batch_dict["src_L"] = torch.Tensor(L_src).long().to(self.device) 
            batch_dict["tgt_L"] = torch.Tensor(L_tgt).long().to(self.device) 
            
            batch_dict["tgt_par_set"] = torch.LongTensor([ i[2] + [[Constants.PAD, Constants.PAD, 1]]*(max(L_tgt)-len(i[1])) for i in batch]).to(self.device)
            pars = [[ w[0] for w in sent[2]] for sent in batch] 
            batch_dict["tgt_par"] = torch.LongTensor(list(zip(*[i + [Constants.PAD]*(max(L_tgt)-len(i)) for i in pars]))).view(-1,len(batch),1).to(self.device) 
            par_labels = [[ w[1] for w in sent[2]] for sent in batch] 
            batch_dict["tgt_par_label"] = torch.LongTensor(list(zip(*[i + [Constants.PAD]*(max(L_tgt)-len(i)) for i in par_labels]))).view(-1,len(batch),1).to(self.device) 
            par_pos_idx = [[ w[2] for w in sent[2]] for sent in batch] 
            tmp = torch.LongTensor([ i + [1]*(max(L_tgt)-len(i)) for i in par_pos_idx])
            tgt_par_pos = torch.full((len(tmp),len(tmp[0]),len(tmp[0])), 0).scatter(2, tmp.unsqueeze(2), 1 )   
            batch_dict["tgt_par_pos_idx"] = tmp.to(self.device)
            batch_dict["tgt_par_pos"] = tgt_par_pos.to(self.device)
            
            n_ch = [[ w[0] for w in sent[3]] for sent in batch] 
            ch_set = [[ w[1:] for w in sent[3]] for sent in batch]
            ch_pos_idx = [[[ch[2] for ch in  w] for w in sent] for sent in ch_set]
            ch_pos_idx = [ i + [[1]]*(max(L_tgt)-len(i)) for i in ch_pos_idx]
            
            tmp_ch = torch.full((len(batch),max(L_tgt),max(L_tgt)), 0)
            for i, sent in enumerate(ch_pos_idx):
                for j, w in enumerate(sent):
                    for ch_pos in w:
                        tmp_ch[i][j][ch_pos] += 1.0/len(w)

            batch_dict["tgt_ch_pos"] = tmp_ch.to(self.device)
            ch_label_pad = [Constants.PAD]*max(L_tgt)
            ch_labels = [sent[4] + [ch_label_pad]*(max(L_tgt)-len(sent[4])) for sent in batch]
            ch_labels_padded = [[w + [Constants.PAD]*(max(L_tgt)-len(w)) for w in sent] for sent in ch_labels] 
            batch_dict["tgt_ch_label"] = torch.LongTensor(ch_labels_padded).to(self.device)
 
            batch_dict["src_raw"] = [i[5] for i in batch]
            batch_dict["tgt_raw"] = [i[6] for i in batch] 
            batch_dict["indices"] = [i[7] for i in batch] 
            yield batch_dict
            num_batches += 1
#
class DatasetLazyIter(object):
    """Yield data from sharded dataset files.

    Args:
        dataset_paths: a list containing the locations of dataset files.
        fields (dict[str, Field]): fields dict for the
            datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: See :class:`OrderedIterator` ``device``.
        is_train (bool): train or valid?
    """

    def __init__(self, dataset,  batch_size, device, is_train, repeat=True):
#        self.dataset = dataset
        self.dataset_orig = list(zip(*[dataset['src'], dataset['tgt'], dataset['tgt_par'], dataset['tgt_ch'], dataset["tgt_ch_label"]]))
        self.n_data = len(self.dataset_orig)
        #logger.info('number of examples: %d' % self.n_data)
        self.n_step_per_epoch = self.n_data // batch_size
 
        self.dataset = [self.dataset_orig[i:i+batch_size] for i in range(0,len(self.dataset_orig),batch_size)] #n_iteration * batch_size * 2 *n_token
        self.batch_size = batch_size
        self.device = device
        self.is_train = is_train
        self.repeat = repeat
        if self.is_train and self.repeat:
            random.shuffle(self.dataset)
#            self.dataset = cycle(self.dataset)

    def shuffle(self):
        random.shuffle(self.dataset_orig)
        self.dataset = [self.dataset_orig[i:i+self.batch_size] for i in range(0,len(self.dataset_orig),self.batch_size)] #n_iteration * batch_size * 2 *n_token
        random.shuffle(self.dataset)

    def __iter__(self):
        num_batches = 0
#        from IPython.core.debugger import Pdb; Pdb().set_trace()
        #random.shuffle(self.dataset)
        for idx, batch in enumerate(self.dataset):
            batch_dict={}
            batch = sorted(batch, key=lambda x : (len(x[0]),len(x[1])))[::-1]
            L_src = [len(i[0]) for i in batch]
            L_tgt = [len(i[1]) for i in batch]
            batch_dict["src"] = torch.LongTensor(list(zip(*[ i[0] + [Constants.PAD]*(max(L_src)-len(i[0])) for i in batch]))).view(-1,len(batch),1).to(self.device)
            batch_dict["tgt"] = torch.LongTensor(list(zip(*[ i[1] + [Constants.PAD]*(max(L_tgt)-len(i[1])) for i in batch]))).view(-1,len(batch),1).to(self.device) 
            batch_dict["src_L"] = torch.LongTensor(L_src).to(self.device) 
            batch_dict["tgt_L"] = torch.LongTensor(L_tgt).to(self.device) 
            
            batch_dict["tgt_par_set"] = torch.LongTensor([ i[2] + [[Constants.PAD, Constants.PAD, 1]]*(max(L_tgt)-len(i[1])) for i in batch]).to(self.device)
            pars = [[ w[0] for w in sent[2]] for sent in batch] 
            batch_dict["tgt_par"] = torch.LongTensor(list(zip(*[i + [Constants.PAD]*(max(L_tgt)-len(i)) for i in pars]))).view(-1,len(batch),1).to(self.device) 
            par_labels = [[ w[1] for w in sent[2]] for sent in batch] 
            batch_dict["tgt_par_label"] = torch.LongTensor(list(zip(*[i + [Constants.PAD]*(max(L_tgt)-len(i)) for i in par_labels]))).view(-1,len(batch),1).to(self.device) 
            par_pos_idx = [[ w[2] for w in sent[2]] for sent in batch] 
            tmp = torch.LongTensor([ i + [1]*(max(L_tgt)-len(i)) for i in par_pos_idx])
            tgt_par_pos = torch.full((len(tmp),len(tmp[0]),len(tmp[0])), 0).scatter(2, tmp.unsqueeze(2), 1 )   
            batch_dict["tgt_par_pos_idx"] = tmp.to(self.device)
            batch_dict["tgt_par_pos"] = tgt_par_pos.to(self.device)
            
            n_ch = [[ w[0] for w in sent[3]] for sent in batch] 
            ch_set = [[ w[1:] for w in sent[3]] for sent in batch]
            ch_pos_idx = [[[ch[2] for ch in  w] for w in sent] for sent in ch_set]
            ch_pos_idx = [ i + [[1]]*(max(L_tgt)-len(i)) for i in ch_pos_idx]
            
            tmp_ch = torch.full((len(batch),max(L_tgt),max(L_tgt)), 0)
            for i, sent in enumerate(ch_pos_idx):
                for j, w in enumerate(sent):
                    for ch_pos in w:
                        tmp_ch[i][j][ch_pos] += 1.0/len(w)
            
            batch_dict["tgt_ch_pos"] = tmp_ch.to(self.device)
            ch_label_pad = [Constants.PAD]*max(L_tgt)
            ch_labels = [sent[4] + [ch_label_pad]*(max(L_tgt)-len(sent[4])) for sent in batch]
            ch_labels_padded = [[w + [Constants.PAD]*(max(L_tgt)-len(w)) for w in sent] for sent in ch_labels] 
            batch_dict["tgt_ch_label"] = torch.LongTensor(ch_labels_padded).to(self.device)
            yield batch_dict


def build_dataset_iter(corpus_type, data, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    device = opt.gpu 

    return DatasetLazyIter(
        data[corpus_type],
        batch_size,
        device,
        is_train,
        repeat=not opt.single_pass)
