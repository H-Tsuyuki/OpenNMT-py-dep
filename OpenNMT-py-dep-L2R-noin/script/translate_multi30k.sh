python ../translate.py \
    -model ../result/multi30k/plcl/modelplcl72_best.pt \
    -data ../data/multi30k/test_spacy.pt \
    -output ../result/multi30k/plcl/pred_plcl72.txt \
    -report_time \
    -replace_unk \
    -gpu 1 \
#    -verbose \
    #-output ../result/multi30k/plc/pred_plc0.txt \
