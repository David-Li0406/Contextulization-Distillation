array=(FB15k-237N WN18RR)
for DATASET in ${array[@]}
    do
        python distill_palm.py --dataset $DATASET
done