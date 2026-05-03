# Create necessary directories
mkdir -p ./logs/generated_labels

path="generated_labels"
# Label generation
# for dataset in arxiv_fine go_emotion massive_intent massive_scenario mtop_intent
for dataset in go_emotion
do
    nohup python label_generation.py \
        --data $dataset \
    > ./logs/${path}/${dataset}_small_label_generation.log 2>&1 &
done
