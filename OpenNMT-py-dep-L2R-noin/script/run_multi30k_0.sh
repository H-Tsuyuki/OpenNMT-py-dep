TYPE="plcl"
set -e
trap 'python ~/ErrorNote_slack/bad.py -m'$TYPE ERR

for i in {53..59}; do
    python ../train.py -data ../data/multi30k/run_spacy.pt -save_model "../result/multi30k/$TYPE/model$TYPE$i" \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -n_epochs 50 -max_generator_batches 2 -dropout 0.1 \
        -batch_size 50 -batch_type sents -normalization sents  -accum_count 2 \
        -optim adam -adam_beta2 0.98 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot -early_stopping 3 \
        -label_smoothing 0.1 -valid_steps 450 -save_checkpoint_steps 450 \
        -gpu 1 -report_every 100 -type $TYPE -par_a

    python ../translate.py \
        -model "../result/multi30k/$TYPE/model$TYPE$i""_best.pt" \
        -data ../data/multi30k/test_spacy.pt \
        -output "../result/multi30k/$TYPE/pred_$TYPE$i.txt" \
        -report_time \
        -replace_unk \
        -gpu 1 \
    #    -verbose \
done
python ~/ErrorNote_slack/good.py -m $TYPE
