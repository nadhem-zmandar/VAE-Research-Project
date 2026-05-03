path="generated_labels"
# Label generation

# Evaluation
for dataset in arxiv_fine go_emotion massive_intent massive_scenario mtop_intent
do
    nohup python evaluate.py \
    --data $dataset \
    --predict_file_path ./${path}/ \
    --predict_file ${dataset}_small_find_labels.json \
    > ./logs/${path}/evaluate_${dataset}.log 2>&1 &
done