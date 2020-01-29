TYPE="test"
DATA="ASPEC"
set -e
trap 'python ~/ErrorNote_slack/bad.py -m'$TYPE ERR

for i in {11..29}; do
    python ../train.py -data ../data/$DATA/run_stanford.pt -save_model "../result/$DATA/$TYPE/model$TYPE$i" \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -n_epochs 50 -max_generator_batches 2 -dropout 0.1 \
        -batch_size 10 -batch_type sents -normalization sents  -accum_count 10 -valid_batch_size 8 \
        -optim adam -adam_beta2 0.98 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot -early_stopping 5 \
        -label_smoothing 0.1 -valid_steps 450 -save_checkpoint_steps 450 --from_n_epoch 8 \
        -gpu 1 -report_every 500 -type $TYPE -val_path ../data/$DATA/val_stanford.pt -gold_path ~/$DATA/ASPEC-JE/dev/dev.en.atok.lower

    python ../translate.py \
        -model "../result/$DATA/$TYPE/model$TYPE$i""_best.pt" \
        -data ../data/$DATA/test_stanford.pt \
        -output "../result/$DATA/$TYPE/pred_$TYPE$i.txt" \
        -report_time \
        -replace_unk \
        -gpu 1 \
    #    -verbose \
done
python ~/ErrorNote_slack/good.py -m $TYPE
