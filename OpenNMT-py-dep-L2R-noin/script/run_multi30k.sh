python ../train.py -data ../data/multi30k/run.pt -save_model ../result/multi30k/model_test \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 1000 -max_generator_batches 2 -dropout 0.1 \
    -batch_size 64 -batch_type sents -normalization sents  -accum_count 2 \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot \
    -label_smoothing 0.1 -valid_steps 450 -save_checkpoint_steps 450 \
    -gpu 0 -report_every 100 -type none
