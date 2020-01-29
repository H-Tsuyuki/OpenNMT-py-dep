"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

import torch
import traceback
import subprocess
from copy import deepcopy

import onmt.utils
from onmt.utils.logging import logger


def build_trainer(opt, device_id, model, data, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    train_loss = onmt.utils.loss.build_loss_compute(model, data, opt)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, data, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    
    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None
    report_manager = onmt.utils.build_report_manager(opt)
    trainer = onmt.Trainer(opt, model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           report_manager,
                           model_saver=model_saver,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           dropout=dropout,
                           dropout_steps=dropout_steps)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, opt, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 report_manager=None, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0]):
        # Basic attributes.
        self.opt = opt 
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d"
                            % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch["tgt"][1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += len(batch["src"][0])
                if len(batches) == self.accum_count:
                    yield batches, normalization
                    self.accum_count = self._accum_count(self.optim.training_step)
                    batches = []
                    normalization = 0
#        if batches:
#            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1)/(step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        valid_steps = train_iter.n_step_per_epoch/self.accum_count
        save_checkpoint_steps = train_iter.n_step_per_epoch /self.accum_count
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        for j in range(self.opt.n_epochs):
            if self.opt.loss_type=="plcl" or self.opt.loss_type=="test": 
                if j==0: 
                    train_iter.shuffle()
            else:
                train_iter.shuffle()
            for i, (batches, normalization) in enumerate(
                    self._accum_batches(train_iter)):
                step = self.optim.training_step
                # UPDATE DROPOUT
                self._maybe_update_dropout(step)


                self._gradient_accumulation(
                    batches, normalization, total_stats,
                    report_stats)

                if self.average_decay > 0 and i % self.average_every == 0:
                    self._update_average(step)
                report_stats = self._maybe_report_training(
                    step, valid_steps*self.opt.n_epochs,#train_steps,
                    self.optim.learning_rate(),
                    report_stats)
                
            #if (self.model_saver is not None
            #    and (save_checkpoint_steps != 0
            #         and step % save_checkpoint_steps == 0)):
                #if self.earlystopper.is_improving():
            if j >= self.opt.from_n_epoch:
                chkpt, chkpt_path = self.model_saver.save(step, moving_average=self.moving_average)
                dir_path = chkpt_path.split("model")[0]
                val_path = chkpt["opt"].val_path
                #out_path = dir_path + "pred_" + dir_path.split("/")[-2] + '_step_'+str(step)+'.txt'
                typestep_path = chkpt_path.split("model")[1].split("_step_")[0]
                out_path = dir_path + "pred_" + typestep_path + '.txt'
                gpu_id = chkpt["opt"].gpu
                subprocess.check_call(["python", "../translate.py", "-model", chkpt_path, \
                    "-data", val_path, "-output", out_path, "-replace_unk", "-gpu_id", str(1)]) 
                gold_path = chkpt["opt"].gold_path
                bleu = subprocess.check_output(["perl", "../tools/multi-bleu.perl", gold_path], stdin=open(out_path)).decode('utf-8') 
                bleu = float(bleu.split("= ")[1].split(",")[0])
                     
                #if valid_iter is not None and step % valid_steps == 0:
                valid_stats = self.validate(
                    valid_iter, moving_average=self.moving_average)
                valid_stats = self._maybe_gather_stats(valid_stats)
                valid_stats.update_bleu(bleu)
                
                self._report_step(self.optim.learning_rate(),
                                  step, valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step, chkpt_path)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

                #if train_steps > 0 and step >= train_steps:
                #    break
        if self.earlystopper is not None:
            if not self.earlystopper.has_stopped():
                self.earlystopper._log_best_step()
        elif self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model

        # Set model in validating mode.
        valid_model.eval()
        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch["src"], batch["src_L"]
                tgt = batch["tgt"]

                # F-prop through the model.
                outputs, attns, dec_mask, ch_labels, par_labels = valid_model(src, tgt, batch, src_lengths)

                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns, ch_labels, par_labels, dec_mask)

                # Update statistics.
                stats.update(batch_stats)

        # Set model back to training mode.
        valid_model.train()

        return stats

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            target_size = len(batch["tgt"])
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src, src_lengths = batch["src"], batch["src_L"]  
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch["tgt"]

            bptt = False
            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()
                outputs, attns, dec_mask, ch_labels, par_labels = \
                    self.model(src, tgt, batch, src_lengths, bptt=bptt)
                bptt = True

                # 3. Compute loss.
                #try:
                loss, batch_stats = self.train_loss(
                    batch,
                    outputs,
                    attns,
                    ch_labels,
                    par_labels,
                    dec_mask,
                    normalization=normalization,
                    shard_size=self.shard_size,
                    trunc_start=j,
                    trunc_size=trunc_size)

                if loss is not None:
                    self.optim.backward(loss)

                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                #except Exception:
                #    traceback.print_exc()
                #    logger.info("At step %d, we removed a batch - accum %d",
                #                self.optim.training_step, k)

                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    self.optim.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            self.optim.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=False)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)
