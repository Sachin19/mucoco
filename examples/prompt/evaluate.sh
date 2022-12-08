task=$1
output=$2
results_path=$3
category=$4 #optional
reference=$5 #optionally required for some metrics

if [[ "$task" == "sentiment" ]]
then
    python ./evaluation/prompted_sampling/evaluate.py --generations_file $output --metrics ppl-big,dist-n,sentiment,allsat --output_file $results_path
elif [[ "$task" == "nontoxic" ]]
then
    echo "nontoxic"
    python ./evaluation/prompted_sampling/evaluate.py --generations_file $output --metrics toxicity,ppl-big,dist-n --output_file $results_path
elif [[ "$task" == "commongen" || "$task" == "roc" ]]
then
    echo $task
    python ./evaluation/prompted_sampling/evaluate.py --generations_file $output --metrics ppl-big,dist-n,allsat,repetition,keywordcount --output_file $results_path --extra $category
else
    echo "wrong task: $task"
fi
