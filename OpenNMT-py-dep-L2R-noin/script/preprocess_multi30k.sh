DEPTYPE="spacy"

python ../preprocess.py \
    -train_src ../../trans/data/multi30k/train.de.atok \
    -train_tgt ../../trans/data/multi30k/train.en \
    -valid_src ../../trans/data/multi30k/val.de.atok \
    -valid_tgt ../../trans/data/multi30k/val.en \
    -test_src ../../trans/data/multi30k/test.de.atok \
    -test_tgt ../../trans/data/multi30k/test.en \
    -save_run_data ../data/multi30k/run_$DEPTYPE.pt \
    -save_val_data ../data/multi30k/val_$DEPTYPE.pt \
    -save_test_data ../data/multi30k/test_$DEPTYPE.pt \
    -max_len 50 \
    -min_word_count 0 \
    -depparser_type $DEPTYPE \
