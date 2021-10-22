source activate langvar
option=$1
domain=$2
outdir=${domain}_$3

mkdir -p ../../data/predictions/formality/$outdir
OUTDIR=../../data/predictions/formality/$outdir
DATA_DIR=../../data/GYAFC_Corpus/$2/test/

PRIMARYMODEL=$4
STSMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-nli-stsb-mean-tokens/0_Transformer
CLASSIFIERMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-formality-classifier/checkpoint_best

mal=5
if [[ "$option" == "plain" ]]
then 
    echo "plain"
    model=$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL
    model_types=AutoModelForCausalLM
    betas=1.0
    loss=gpt2conditional
    lossabbr="logpyx"
    epsilons=none
    min_epsilons=none
    epsilon_warmup_steps=none
    epsilon_cooldown_steps=none
    epsilon_decay_functions=none
elif [[ "$option" == "sts" ]]
then
    echo "sts"
    model=$PRIMARYMODEL:$STSMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL
    model_types=AutoModelForCausalLM:AutoModel
    betas=0.8:0.2
    loss=gpt2conditional:sts
    lossabbr="logpyx:sts"
    epsilons=0.4
    min_epsilons=0.25
    epsilon_warmup_steps=20
    epsilon_cooldown_steps=40
    epsilon_decay_functions=step
elif [[ "$option" == "wmd" ]]
then
    echo "wmd"
    model=$PRIMARYMODEL:$STSMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL
    model_types=AutoModelForCausalLM:AutoModel
    betas=0.8:0.2
    loss=gpt2conditional:wmd
    lossabbr="logpyx:wmd"
    epsilons=0.8
    min_epsilons=0.4
    epsilon_warmup_steps=20
    epsilon_cooldown_steps=40
    epsilon_decay_functions=step
elif [[ "$option" == "sts:wmd" ]]
then
    echo "sts:wmd"
    model=$PRIMARYMODEL:$STSMODEL:$STSMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL:$STSMODEL
    model_types=AutoModelForCausalLM:AutoModel:AutoModel
    betas=0.8:0.1:0.1
    loss=gpt2conditionalloss:semantic_loss:wmd
    lossabbr="logp(y|x):sts:wmd"
    epsilons=0.4:0.8
    min_epsilons=0.25:0.4
    epsilon_warmup_steps=20:20
    epsilon_cooldown_steps=20:20
    epsilon_decay_functions=step:step
elif [[ "$option" == "wmd" ]]
then
    echo 2
    model=$PRIMARYMODEL:$STSMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL
    model_types=AutoModelForCausalLM:AutoModel
    betas=1.0:0.0
    loss=gpt2conditionalloss:wmd
    lossabbr="logp(y|x):wmd"
    epsilons=0.8
    min_epsilons=0.4
    epsilon_warmup_steps=20
    epsilon_cooldown_steps=0:0
    epsilon_decay_functions=step:step
elif [[ "$option" == "binary_ce" ]]
then
    echo "else"
    model=$PRIMARYMODEL:$CLASSIFIERMODEL
    tokenizer=$PRIMARYMODEL:$CLASSIFIERMODEL
    model_types=AutoModelForCausalLM:AutoModelForSequenceClassification
    betas=0.8:0.2
    loss=gpt2conditional:binary_classification
    lossabbr="pyx:binary"
    epsilons=10.0
    min_epsilons=0.69
    epsilon_warmup_steps=5
    epsilon_cooldown_steps=40
    epsilon_decay_functions=step
else
    echo "else"
    model=$PRIMARYMODEL:$STSMODEL:$CLASSIFIERMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL:$CLASSIFIERMODEL
    model_types=AutoModelForCausalLM:AutoModel:AutoModelForSequenceClassification
    betas=0.8:0.1:0.1
    loss=gpt2conditionalloss:semantic_loss:binary_classification_loss
    lossabbr="pyx:sts:binary_ce"
    epsilons=1.0:10.0
    min_epsilons=0.:$7
    epsilon_warmup_steps=20:20
    epsilon_cooldown_steps=40:40
    epsilon_decay_functions=step:step
    mal=10
fi

debug=$5
suffix=$6
selection_criterion="primary_allsat"

if [[ "$debug" == "debug" ]]
then
    python -W ignore -u decode.py\
        --data $DATA_DIR/informal${suffix}:$DATA_DIR/formal.ref0${suffix}\
        --additional-data $DATA_DIR/informal${suffix}.paraphrase\
        --model $model\
        --tokenizer $tokenizer\
        --model_types $model_types\
        --betas $betas\
        --loss $loss\
        --lossabbr "$lossabbr"\
        --suffix-length -2\
        --model_dtype fp32\
        --fp16_source pytorch\
        --target-type probs\
        --max-grad-norm 0.05\
        --optim-steps 100\
        --scale_loss none\
        --log-interval 10\
        --decode-temperature 0.1\
        --bos\
        --eos\
        --st\
        --selection_criterion $selection_criterion\
        --length_diff 2\
        --optim expgd\
        --expgd_mw 2\
        --lr 50\
        --length-normalize\
        --dampness 1.0\
        --epsilons $epsilons\
        --min_epsilons $min_epsilons\
        --epsilon_warmup_steps $epsilon_warmup_steps\
        --epsilon_cooldown_steps $epsilon_cooldown_steps\
        --epsilon_decay_functions $epsilon_decay_functions\
        --lambda-lr 2.5\
        --num-examples 10\
        --debug
else
    python evaluation/formality/all_evaluation_metrics.py --data $PRIMARYMODEL/eval_nucleus_paraphrase_0.0/transfer_formal_test.txt,$DATA_DIR/informal${suffix} --model sentence-transformers/roberta-base-nli-stsb-mean-tokens --tokenizer sentence-transformers/roberta-base-nli-stsb-mean-tokens --evaluation_metrics sts_sim,cls_sim,wieting_sim,transfer,fluency,bleu,ppl --outfile $PRIMARYMODEL/eval_nucleus_paraphrase_0.0/performance_wrt_source.json --match_with source

    python evaluation/formality/all_evaluation_metrics.py --data $PRIMARYMODEL/eval_nucleus_paraphrase_0.0/transfer_formal_test.txt,$DATA_DIR/formal.ref0${suffix},$DATA_DIR/formal.ref1${suffix},$DATA_DIR/formal.ref2${suffix},$DATA_DIR/formal.ref3${suffix} --model sentence-transformers/roberta-base-nli-stsb-mean-tokens --tokenizer sentence-transformers/roberta-base-nli-stsb-mean-tokens --evaluation_metrics sts_sim,cls_sim,wieting_sim,transfer,fluency,bleu,ppl --outfile $PRIMARYMODEL/eval_nucleus_paraphrase_0.0/performance_wrt_source.json --match_with reference

    for length_diff in 0 -1 1
    do
        python -W ignore -u multiple_constrained_optimize.py\
            --data $DATA_DIR/informal${suffix}:$PRIMARYMODEL/eval_nucleus_paraphrase_0.0/transfer_formal_test.txt\
            --additional-data $DATA_DIR/informal${suffix}.paraphrase\
            --model $model\
            --tokenizer $tokenizer\
            --model_types $model_types\
            --betas $betas\
            --loss $loss\
            --lossabbr "$lossabbr"\
            --suffix-length -2\
            --model_dtype fp32\
            --fp16_source pytorch\
            --target-type probs\
            --max-grad-norm 0.1\
            --optim-steps 100\
            --scale_loss none\
            --log-interval 50\
            --decode-temperature 0.1\
            --cache_dir cache\
            --bos\
            --eos\
            --st\
            --length_diff $length_diff\
            --stopping_criterion $stopping_criterion\
            --optim expgd\
            --expgd_mw 2\
            --lr 50\
            --length-normalize\
            --dampness 1.0\
            --epsilons $epsilons\
            --min_epsilons $min_epsilons\
            --epsilon_warmup_steps $epsilon_warmup_steps\
            --epsilon_cooldown_steps $epsilon_cooldown_steps\
            --epsilon_decay_functions $epsilon_decay_functions\
            --lambda-lr 2.5\
            --num-examples 0\
            --max-allowed-length $mal\
            --outfile $OUTDIR/transfer_formal_${option}_${length_diff}.txt
        numlines=$(wc -l $OUTDIR/transfer_formal_${option}_${length_diff}.txt | awk '{ print $1 }')
        numoriglines=$(wc -l $DATA_DIR/informal${suffix} | awk '{ print $1 }')
        echo $numlines,$numoriglines
        best=$PRIMARYMODEL/eval_nucleus_paraphrase_0.0/transfer_formal_test.txt
        bash evaluation/formality/evaluate_single.sh $OUTDIR/transfer_formal_${option}_${length_diff}.txt $best $OUTDIR/eval_${option}_${length_diff} $numlines $domain ${suffix}
    done
fi
