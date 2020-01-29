#REF=../result/multi30k/base/pred_1.txt
TYPE="plcl"
REF="../result/ASPEC/$TYPE/pred"
HYP="/home/tsuyuki/ASPEC/ASPEC-JE/test/test.en.atok.lower"

for i in {30..70}; do
    perl ../tools/multi-bleu.perl $HYP < $REF"_"$TYPE$i".txt" | sed -e 's/BLEU = //' -e 's/,.*//g'  | tr '\n' '\t' >>out
    echo $i
    perl ../tools/multi-bleu.perl $HYP < $REF"_$TYPE$i.txt"
done 

