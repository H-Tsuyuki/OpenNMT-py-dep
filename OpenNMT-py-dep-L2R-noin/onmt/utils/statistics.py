""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys

from onmt.utils.logging import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, word_loss=0, label_loss=0, par_order_loss=0, ch_label_loss=0, ch_order_loss=0, \
                n_words=0, n_correct_word=0, n_correct_label=0, n_correct_par=0, n_correct_ch_label=0):
        self.word_loss = word_loss
        self.label_loss = label_loss
        self.par_order_loss = par_order_loss
        self.ch_label_loss = ch_label_loss
        self.ch_order_loss = ch_order_loss
        self.n_words = n_words
        self.n_correct_word = n_correct_word
        self.n_correct_label = n_correct_label
        self.n_correct_par = n_correct_par
        self.n_correct_ch_label = n_correct_ch_label
        self.n_src_words = 0
        self.start_time = time.time()
        self.bleu = 0

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        from torch.distributed import get_rank
        from onmt.utils.distributed import all_gather_list

        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.word_loss += stat.word_loss
        self.label_loss += stat.label_loss
        self.par_order_loss += stat.par_order_loss
        self.ch_label_loss += stat.ch_label_loss
        self.ch_order_loss += stat.ch_order_loss
        self.n_words += stat.n_words
        self.n_correct_word += stat.n_correct_word
        self.n_correct_label += stat.n_correct_label
        self.n_correct_par += stat.n_correct_par
        self.n_correct_ch_label += stat.n_correct_ch_label

        if update_n_src_words:
            self.n_src_words += stat.n_src_words
    
    def update_bleu(self, bleu):
        self.bleu = bleu

    def word_accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct_word / self.n_words)

    def label_accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct_label / self.n_words)

    def par_accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct_par / self.n_words)

    def ch_label_accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct_ch_label / self.n_words)

    def word_xent(self):
        """ compute cross entropy """
        return self.word_loss / self.n_words
    
    def label_xent(self):
        """ compute cross entropy """
        return self.label_loss / self.n_words
 
    def ch_label_xent(self):
        """ compute cross entropy """
        return self.ch_label_loss / self.n_words
 
    def par_order_xent(self):
        """ compute cross entropy """
        return self.par_order_loss / self.n_words

    def ch_order_xent(self):
        """ compute cross entropy """
        return self.ch_order_loss / self.n_words


    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.word_loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
            ("Step %s; acc: %4.2f; xent: %4.2f;  par_label_acc: %4.2f; " +
              "ch_label_acc: %4.2f; par_acc: %4.2f; par_xent: %4.2f; " +
             "ch_xent: %4.2f;  %6.0f sec")
            % (step_fmt,
               self.word_accuracy(),
               self.word_xent(),
               self.label_accuracy(),
               self.ch_label_accuracy(),
               self.par_accuracy(),
               self.par_order_xent(),
               self.ch_order_xent(),
               time.time() - start))
        sys.stdout.flush()
#               learning_rate,

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
