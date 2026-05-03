# Create necessary directories
path="generated_labels"

# Given labels classification
#for dataset in arxiv_fine go_emotion massive_intent massive_scenario mtop_intent

for dataset in massive_intent
do
    nohup python given_label_classification.py \
        --data $dataset \
    > ./logs/${path}/${dataset}_small_given_label_generation.log 2>&1 &
done
