DEPTYPE="stanford"

python ../preprocess.py \
    -train_src ~/ASPEC/ASPEC-JE/train/train.ja.atok \
    -train_tgt ~/ASPEC/ASPEC-JE/train/train.en \
    -valid_src ~/ASPEC/ASPEC-JE/dev/dev.ja.atok \
    -valid_tgt ~/ASPEC/ASPEC-JE/dev/dev.en \
    -test_src ~/ASPEC/ASPEC-JE/test/test.ja.atok \
    -test_tgt ~/ASPEC/ASPEC-JE/test/test.en \
    -save_run_data ../data/ASPEC/run_$DEPTYPE.pt \
    -save_val_data ../data/ASPEC/val_$DEPTYPE.pt \
    -save_test_data ../data/ASPEC/test_$DEPTYPE.pt \
    -max_len 55 \
    -min_word_count 1 \
    -depparser_type $DEPTYPE \
