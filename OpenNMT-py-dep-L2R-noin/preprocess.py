''' Handling the data io '''

import json
import argparse
import progressbar
import numpy as np

import torch

import depparser2
import stfd_depparser2
import onmt.Constants as Constants

def read_instances_from_srcfile(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    words = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            #word_list = word_tokenize(sent)
            word_list = sent.split()
            if len(word_list) > max_sent_len:
                trimmed_sent_count += 1
            word_list = word_list[:max_sent_len]
            if words is not None:
                words += [[Constants.BOS_WORD] + word_list + [Constants.EOS_WORD]]
            else:
                from IPython.core.debugger import Pdb; Pdb().set_trace()

    print('[Info] Get {} instances from {}'.format(len(words), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return words


def read_instances_from_tgtfile(inst_file, max_sent_len, keep_case, depparser_type):
    ''' Convert file into word seq lists and vocab '''

    sents_words = []
    sents_labels = []
    sents_pars = []
    sents_chs = []
    sents_chs_labels = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.strip().lower()
            #word_list = word_tokenize(sent)
#            word_list = sent.split()
            if depparser_type=='spacy':
                sent = depparser2.top_downize(sent)
            elif depparser_type=='stanford':
                sent = stfd_depparser2.top_downize(sent)
            if len(sent) > max_sent_len:
                trimmed_sent_count += 1
            sent = sent[:max_sent_len]
            words = [w["word"] for w in sent]
            pars = [w["parent"] for w in sent]
            chs = [w["children"] for w in sent]
            labels_for_make_dict = [l[1] for l in pars]
            sent_chs_labels = []
            for j, word_chs in enumerate(chs):
                labels_for_make_dict += [l[1] for l in word_chs[1:]]
                ch_labels=[Constants.PAD_WORD]*len(chs)
                for i in range(len(word_chs[1:])):
                   # ch_labels[:j] = [Constants.BOS_WORD] * j
                    ch_labels[word_chs[1:][i][2]] = word_chs[1:][i][1]
                sent_chs_labels += [ch_labels]

            sents_words += [words]
            sents_labels += [labels_for_make_dict]
            sents_pars += [pars]
            sents_chs += [chs]
            sents_chs_labels += [sent_chs_labels]

    print('[Info] Get {} instances from {}'.format(len(sents_words), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return sents_words, sents_labels, sents_pars, sents_chs, sents_chs_labels


def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK,
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS
        }

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def convert_pars_to_idx_seq(word_insts, word2idx, label2idx):
    a =  [[[word2idx.get(w[0], Constants.UNK), label2idx.get(w[1], Constants.UNK), w[2]] for w in s] for s in word_insts]
    return a

def convert_chs_to_idx_seq(word_insts, word2idx, label2idx):
    a = [[[[word2idx.get(ch[0], Constants.UNK), label2idx.get(ch[1], Constants.UNK), ch[2]] if i>0 else ch for i, ch in enumerate(w)] for w in s] for s in word_insts]
    return a

def convert_chs_label_to_idx_seq(word_insts, label2idx):
    a = [[[label2idx.get(ch, Constants.UNK)  for ch in w] for w in s] for s in word_insts]
    return a

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-test_src', required=True)
    parser.add_argument('-test_tgt', required=True)
    parser.add_argument('-save_run_data', required=True)
    parser.add_argument('-save_val_data', required=True)
    parser.add_argument('-save_test_data', required=True)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)
    parser.add_argument('-depparser_type', type=str, default='spacy',
                        choices=['spacy', 'stanford'])

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>
    print(json.dumps(opt.__dict__, indent=2))


    # Training set
    train_src_word_insts = read_instances_from_srcfile(
        opt.train_src, opt.max_word_seq_len, opt.keep_case)
    train_tgt_word_insts, train_tgt_labels, train_tgt_pars, train_tgt_chs, train_tgt_chs_labels = read_instances_from_tgtfile(
        opt.train_tgt, opt.max_word_seq_len, opt.keep_case, opt.depparser_type)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]
   
    #- Remove empty instances
#    train_src_word_insts, train_tgt_word_insts = list(zip(*[
#        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    #- Sort src&tgt
#    tmp = list(zip(*[train_src_word_insts, train_tgt_word_insts, train_tgt_labels, train_tgt_pars, train_tgt_chs, train_tgt_chs_labels]))
#    tmp = sorted(tmp, key=lambda x : (len(x[0]),len(x[1])))
#    train_src_word_insts, train_tgt_word_insts, train_tgt_labels, train_tgt_pars, train_tgt_chs, train_tgt_chs_labels = list(zip(*tmp))
 
    # Validation set
    valid_src_word_insts = read_instances_from_srcfile(
        opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    valid_tgt_word_insts, valid_tgt_labels, valid_tgt_pars, valid_tgt_chs, valid_tgt_chs_labels = read_instances_from_tgtfile(
        opt.valid_tgt, opt.max_word_seq_len, opt.keep_case, opt.depparser_type)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
#    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
#        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    #- Sort src&tgt
#    tmp = list(zip(*[valid_src_word_insts, valid_tgt_word_insts, valid_tgt_labels, valid_tgt_pars, valid_tgt_chs, valid_tgt_chs_labels]))
#    tmp = sorted(tmp, key=lambda x : (len(x[0]),len(x[1])))
#    valid_src_word_insts, valid_tgt_word_insts, valid_tgt_labels, valid_tgt_pars, valid_tgt_chs, valid_tgt_chs_labels = list(zip(*tmp))
 

    # Test set
    test_src_word_insts = read_instances_from_srcfile(
        opt.test_src, opt.max_word_seq_len, opt.keep_case)
    test_tgt_word_insts, test_tgt_labels, test_tgt_pars, test_tgt_chs, test_tgt_chs_labels = read_instances_from_tgtfile(
        opt.test_tgt, opt.max_word_seq_len, opt.keep_case, opt.depparser_type)

    if len(test_src_word_insts) != len(test_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(test_src_word_insts), len(test_tgt_word_insts))
        test_src_word_insts = test_src_word_insts[:min_inst_count]
        test_tgt_word_insts = test_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
#    test_src_word_insts, test_tgt_word_insts = list(zip(*[
#        (s, t) for s, t in zip(test_src_word_insts, test_tgt_word_insts) if s and t]))


    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target word.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target Dep label.')
            tgt_label2idx = build_vocab_idx(train_tgt_labels, 0)

    def get_swap_dict(d):
        return {v: k for k, v in d.items()} 
    # Build idndex to word Dictionary
    src_idx2word = get_swap_dict(src_word2idx)
    tgt_idx2word = get_swap_dict(tgt_word2idx)
    tgt_idx2label = get_swap_dict(tgt_label2idx)


    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)
    test_src_insts = convert_instance_to_idx_seq(test_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    train_tgt_pars = convert_pars_to_idx_seq(train_tgt_pars, tgt_word2idx, tgt_label2idx)
    train_tgt_chs = convert_chs_to_idx_seq(train_tgt_chs, tgt_word2idx, tgt_label2idx)
    train_tgt_chs_labels = convert_chs_label_to_idx_seq(train_tgt_chs_labels, tgt_label2idx)
    
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)
    valid_tgt_pars = convert_pars_to_idx_seq(valid_tgt_pars, tgt_word2idx, tgt_label2idx)
    valid_tgt_chs = convert_chs_to_idx_seq(valid_tgt_chs, tgt_word2idx, tgt_label2idx)
    valid_tgt_chs_labels = convert_chs_label_to_idx_seq(valid_tgt_chs_labels, tgt_label2idx)

    test_tgt_insts = convert_instance_to_idx_seq(test_tgt_word_insts, tgt_word2idx)
    test_tgt_pars = convert_pars_to_idx_seq(test_tgt_pars, tgt_word2idx, tgt_label2idx)
    test_tgt_chs = convert_chs_to_idx_seq(test_tgt_chs, tgt_word2idx, tgt_label2idx)
    test_tgt_chs_labels = convert_chs_label_to_idx_seq(test_tgt_chs_labels, tgt_label2idx)
    run_data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx,
            'tgt_label': tgt_label2idx,
            },
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts,
            'tgt_par': train_tgt_pars,
            'tgt_ch': train_tgt_chs,
            'tgt_ch_label': train_tgt_chs_labels,
            },
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts,
            'tgt_par': valid_tgt_pars,
            'tgt_ch': valid_tgt_chs,
            'tgt_ch_label': valid_tgt_chs_labels,
            }
        }
 
    val_data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx,
            'tgt_label': tgt_label2idx,
            },
        '_dict': {
            'src': src_idx2word,
            'tgt': tgt_idx2word,
            'tgt_label': tgt_idx2label,
            },
        'test': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts,
            'tgt_label': valid_tgt_labels,
            'tgt_par': valid_tgt_pars,
            'tgt_ch': valid_tgt_chs,
            'tgt_ch_label': valid_tgt_chs_labels,
            'src_raw': valid_src_word_insts,
            'tgt_raw': valid_tgt_word_insts,
            'indices': np.arange(len(valid_src_insts))
            }
        }


   
    test_data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx,
            'tgt_label': tgt_label2idx,
            },
        '_dict': {
            'src': src_idx2word,
            'tgt': tgt_idx2word,
            'tgt_label': tgt_idx2label,
            },
        'test': {
            'src': test_src_insts,
            'tgt': test_tgt_insts,
            'tgt_label': test_tgt_labels,
            'tgt_par': test_tgt_pars,
            'tgt_ch': test_tgt_chs,
            'tgt_ch_label': test_tgt_chs_labels,
            'src_raw': test_src_word_insts,
            'tgt_raw': test_tgt_word_insts,
            'indices': np.arange(len(test_src_insts))
            }
        }


    print('[Info] Dumping the processed data to pickle file', opt.save_run_data, opt.save_val_data, opt.save_test_data)
    torch.save(run_data, opt.save_run_data)
    torch.save(val_data, opt.save_val_data)
    torch.save(test_data, opt.save_test_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
