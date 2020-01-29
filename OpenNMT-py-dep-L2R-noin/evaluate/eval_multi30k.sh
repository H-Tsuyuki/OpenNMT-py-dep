#REF=../result/multi30k/test/pred_test_step_450.txt
#HYP=../../trans/data/multi30k/val.en.spacy_atok.lower
REF=../result/ASPEC/base/pred_base3.txt
HYP=~/ASPEC/ASPEC-JE/test/test.en.atok.lower

perl ../tools/multi-bleu.perl $HYP < $REF

