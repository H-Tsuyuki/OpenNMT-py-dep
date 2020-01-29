set -e
trap 'python ~/ErrorNote_slack/bad.py -m plc' ERR

TYPE="plc"
for i in {0..29}; do
    python ../train.py -data ../data/multi30k/run_stanford.pt -save_model "../result/multi30k_stfd/$TYPE/model$TYPE$i" \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 5000 -max_generator_batches 2 -dropout 0.1 \
        -batch_size 64 -batch_type sents -normalization sents  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 450 -save_checkpoint_steps 450000 \
        -gpu 0 -report_every 100 -type $TYPE 

    python ../translate.py \
        -model "../result/multi30k_stfd/$TYPE/model$TYPE$i""_step_5000.pt" \
        -data ../data/multi30k/test_stanford.pt \
        -output "../result/multi30k_stfd/$TYPE/pred_$TYPE$i.txt" \
        -report_time \
        -replace_unk \
        -gpu 0 \
    #    -verbose \
done
python ~/ErrorNote_slack/good.py -m plc
