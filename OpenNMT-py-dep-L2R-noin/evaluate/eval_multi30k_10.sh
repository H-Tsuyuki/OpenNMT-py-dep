#REF=../result/multi30k/base/pred_1.txt
TYPE=plcl
REF=../result/multi30k/$TYPE/pred
HYP=../../trans/data/multi30k/test.en.spacy_atok.lower

for i in {70..99}; do
    perl ../tools/multi-bleu.perl $HYP < $REF"_"$TYPE$i".txt" | sed -e 's/BLEU = //' -e 's/,.*//g'  | tr '\n' '\t' >>out
    echo $i
    perl ../tools/multi-bleu.perl $HYP < $REF"_$TYPE$i.txt"
done 

