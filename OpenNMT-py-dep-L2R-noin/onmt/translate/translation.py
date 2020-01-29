""" Translation main class """
from __future__ import unicode_literals, print_function

import torch
import onmt.Constants as Constants

class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (onmt.inputters.Dataset): Data.
       fields (List[Tuple[str, torchtext.data.Field]]): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, idx2word, n_best=1, replace_unk=False,
                 has_tgt=True, phrase_table="", opt=None):
        self.idx2word = idx2word
        self._has_text_src = True
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.phrase_table = phrase_table
        self.has_tgt = has_tgt
        self.opt = opt

    def _build_target_tokens(self, src_raw, pred, attn=None):
        tokens = []
        for tok in pred.tolist():
            if tok < len(self.idx2word["tgt"]):
                tokens.append(self.idx2word["tgt"][tok])
            if tokens[-1] == Constants.EOS_WORD:
                tokens = tokens[:-1]
                break
        if self.replace_unk and attn is not None:
            for i in range(len(tokens)):
                if tokens[i] == Constants.UNK_WORD:
                    _, max_index = attn[i][:len(src_raw)].max(0)
                    tokens[i] = src_raw[max_index.item()]
                    if self.phrase_table != "":
                        with open(self.phrase_table, "r") as f:
                            for line in f:
                                if line.startswith(src_raw[max_index.item()]):
                                    tokens[i] = line.split('|||')[1].strip()
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = len(batch["src"][1])
        preds = translation_batch["predictions"]
        pred_score = translation_batch["scores"]
        attn = translation_batch["attention"]
        gold_score = translation_batch["gold_score"]

        # Sorting
        sort = list(zip(*[preds, pred_score, attn, gold_score, \
            batch["src_raw"],batch["tgt_raw"],batch["indices"]]))
        sort = sorted(sort, key=lambda x : x[6])

        preds, pred_score, attn, gold_score, \
        src_raw, tgt_raw, indices = list(zip(*sort))
        translations = []

        for b in range(batch_size):
            pred_sents = [self._build_target_tokens( 
                src_raw[b], preds[b][n], attn[b][n])
                for n in range(self.n_best)]
  
            gold_sent = tgt_raw[b][1:-1]
            translation = Translation(
                src_raw[b], pred_sents, attn[b], pred_score[b],
                gold_sent, gold_score[b]
            )
            translations.append(translation)

        return translations


class Translation(object):
    """Container for a translated sentence.

    Attributes:
        src (LongTensor): Source word IDs.
        src_raw (List[str]): Raw source words.
        pred_sents (List[List[str]]): Words from the n-best translations.
        pred_scores (List[List[float]]): Log-probs of n-best translations.
        attns (List[FloatTensor]) : Attention distribution for each
            translation.
        gold_sent (List[str]): Words from gold translation.
        gold_score (List[float]): Log-prob of gold translation.
    """

    __slots__ = ["src_raw", "pred_sents", "attns", "pred_scores",
                 "gold_sent", "gold_score"]

    def __init__(self, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        msg = ['\nSENT {}: {}\n'.format(sent_number, self.src_raw)]

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        msg.append('PRED {}: {}\n'.format(sent_number, pred_sent))
        msg.append("PRED SCORE: {:.4f}\n".format(best_score))

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            msg.append('GOLD {}: {}\n'.format(sent_number, tgt_sent))
            msg.append(("GOLD SCORE: {:.4f}\n".format(self.gold_score)))
        if len(self.pred_sents) > 1:
            msg.append('\nBEST HYP:\n')
            for score, sent in zip(self.pred_scores, self.pred_sents):
                msg.append("[{:.4f}] {}\n".format(score, sent))

        return "".join(msg)
