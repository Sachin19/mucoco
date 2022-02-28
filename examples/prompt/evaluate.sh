task=$1
output=$2
results_path=$3
reference=$4 #optionally required for some metrics

## QUALITY EVALUATION
if [[ "$task" == "degen" ]]
then
    python evaluation/unconstrained_sampling/evaluate_degen.py --generations_file $output --metrics ppl-own --output_file $results_path
elif [[ "$task" == "unconditional" || "$task" == "conditional" || "$task" == "conditional-nucleus" ]]
then
    # python evaluation/unconstrained_sampling/evaluate.py --generations_file $output --metrics self-bleu --output_file $results_path
    python evaluation/unconstrained_sampling/evaluate.py --generations_file $output --metrics ppl-own,ppl-big,cola,self-bleu,zipf,repetition,dist-n --output_file $results_path
elif [[ "$task" == "positive-neutral" || "$task" == "positive-neutral-p" || "$task" == "negative-neutral-p" || "$task" == "positive-adv" || "$task" == "negative-neutral" || "$task" == "negative-adv" || "$task" == "unconstrained-neutral" ]]
then
    python ./evaluation/unconstrained_sampling/evaluate.py --generations_file $output --metrics ppl-own,ppl-big,dist-n,sentiment,cola --output_file $results_path
elif [[ "$task" == "debug-sentiment" ]]
then
    python ./evaluation/unconstrained_sampling/evaluate.py --generations_file $output --metrics cola --output_file $results_path
elif [[ "$task" == "nontoxic" || "$task" == "nontoxic-p" || "$task" == "nontoxic-linear-scale" || "$task" == "nontoxic-0.8-small-scale" ]]
then
    echo "nontoxic"
    python ./evaluation/unconstrained_sampling/evaluate.py --generations_file $output --metrics toxicity,ppl-own,ppl-big,dist-n,cola --output_file $results_path
elif [[ "$task" == "commongen" || "$task" == "keyword5" ]]
then
    echo $task
    python ./evaluation/unconstrained_sampling/evaluate.py --generations_file $output --metrics ppl-own,ppl-big,dist-n,cola --output_file $results_path
elif [[ "$task" == "topic" ]]
then
    echo "topic evaluation"
else
    echo "wrong task $task"
fi