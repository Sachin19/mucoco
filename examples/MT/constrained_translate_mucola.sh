source activate 2022
graddistance=${7}
OUTDIR=$2
mkdir -p $OUTDIR

OUTFILE=$OUTDIR/outputs.txt
EVALFILE=$OUTDIR/results${15}.txt

DATA=$1 #need to change

PRIMARYMODEL=$3

gold_loss_epsilons="none"
mal=5

DATASTYLE="text"
ADDITIONALDATAFILE="none"
JSON_PKEY="none"
JSON_SKEY="none"
JSON_TKEY="none"
TARGETTOPIC="none"
KEYWORDS="none"
KEYWORDTOPK=1
JSONTOK=false
WORDLIST="none"
OUTPUTLEN=20
MAXLEN=20
LAMBDALR=2.0
selection_criterion="mrr_allsat"
NUM_SAMPLES=1
length_diff=0
linear_scale="false"
OPTIMSTEPS=500
OUTPUTSTYLE="text"
embedgd_do_sample="true"
embedgd_top_p=1.0
noise_variance=0.0
debug_gradients=${14}
begin_temperature=${16}
final_temperature=${17}
temp_red_steps=${18}
init_style=${19}
LABELID=${22} #label ids for the classifier(s) to predict
coeff_pattern=
LAMBDA_UPDATES=${24}
CLASSLOGPROB=${25}
always_mucoco=${26}
only_mucoco=${27}
MAXLR=${28}
LRUPDATESIZE=${29}
RESTARTS=${30}
OUTPUTLEN=${31}
RANDOMEXAMPLE=${32}
STARTIDX=${33}
ENDIDX=${34}
SENTIMENTMODELID=${35}
USECONTEXT="false"
DAMPNESS=0.1
if [[ -z "$OUTPUTLEN" ]]
then
    OUTPUTLEN=20
fi
if [[ -z "$RANDOMEXAMPLE" ]]
then
    RANDOMEXAMPLE="true"
fi
if [[ -z "$STARTIDX" ]]
then
    STARTIDX=0
fi
if [[ -z "$ENDIDX" ]]
then
    ENDIDX=-1
fi
USECONTEXT="false"
if [[ -z "$RESTARTS" ]]
then
    RESTARTS=0
    # p > 0.5 default
fi
if [[ -z "$MAXLR" ]]
then
    MAXLR=0.45
    # p > 0.5 default
fi
if [[ -z "$LRUPDATESIZE" ]]
then
    LRUPDATESIZE=0.01
    # p > 0.5 default
fi
if [[ -z "$always_mucoco" ]]
then
    always_mucoco="false"
    # p > 0.5 default
fi
if [[ -z "$only_mucoco" ]]
then
    only_mucoco="false"
    # p > 0.5 default
fi
if [[ -z "$CLASSLOGPROB" ]]
then
    CLASSLOGPROB=0.693
    # p > 0.5 default
fi
if [[ -z "$LAMBDA_UPDATES" ]]
then
    LAMBDA_UPDATES=50
fi
if [[ -z "$begin_temperature" ]]
then
    begin_temperature=1.0
fi
if [[ -z "$final_temperature" ]]
then
    final_temperature=0.01
fi
if [[ -z "$temp_red_steps" ]]
then
    temp_red_steps=20
fi
if [[ -z "$debug_gradients" ]]
then
    debug_gradients="false"
fi
repetition_penalty=${13}
if [[ -z "$repetition_penalty" ]]
then
    repetition_penalty=0
fi
if [[ -z "$init_style" ]]
then
    init_style=zeros
fi
if [[ -z "$coeff_pattern" ]]
then
    coeff_pattern="constant"
fi

NUMEXAMPLES=${20}
if [[ -z "$NUMEXAMPLES" ]]
then
    NUMEXAMPLES=0
fi

option=$1
if [[ "$option" == "marian-translate" ]]  ## no prompt just generate directly
then 
    echo "marian-translate"
    OPTIMSTEPS=200
    DATASTYLE="text"
    DATAFILE=data/MT/terminology_dataset/newstest2017-iate.414.terminology.tsv.en
    OUTPUTLEN=200
    MAXLEN=200
    NUM_SAMPLES=1
    OUTPUTSTYLE="text"
    model=$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL
    model_types=AutoModelForSeq2SeqLM
    betas=1.0
    loss=marianmt
    lossabbr="logpyx"
    epsilons=none
    custom_epsilons=none
    label_id=none
    min_epsilons=none
    epsilon_warmup_steps=none
    epsilon_cooldown_steps=none
    epsilon_decay_functions=none
    noise_variance=1.0
    embedgd_do_sample="false"
    LAMBDALR=1.0
elif [[ "$option" == "translate-keyword-1" ]]  ## no prompt just generate directly
then 
    echo "translate-keyword-1"
    OPTIMSTEPS=250
    DATASTYLE="text"
    DATAFILE=data/MT/terminology_dataset/newstest2017-iate.414.1.terminology.tsv.en
    ADDITIONALDATAFILE=data/MT/terminology_dataset/iate/iate.414.1.terminology.tsv
    KEYWORDS="_iate_"
    OUTPUTLEN=150
    MAXLEN=150
    NUM_SAMPLES=1
    model=$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForSeq2SeqLM:AutoModelForSeq2SeqLM
    betas=1.0:0.0
    loss=marianmt:ngramsl22
    lossabbr="logpyx:ngramsl22"
    epsilons=$CLASSLOGPROB
    label_id=0:0
    custom_epsilons=true
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=step
    noise_variance=1.0
    embedgd_do_sample="false"
    LAMBDALR=1.0
elif [[ "$option" == "translate-keyword-2" ]]  ## no prompt just generate directly
then 
    echo "translate-keyword-1"
    OPTIMSTEPS=250
    DATASTYLE="text"
    DATAFILE=data/MT/terminology_dataset/newstest2017-iate.414.2.terminology.tsv.en
    ADDITIONALDATAFILE=data/MT/terminology_dataset/iate/iate.414.2.terminology.tsv
    KEYWORDS="_iate_"
    OUTPUTLEN=150
    MAXLEN=150
    NUM_SAMPLES=1
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForSeq2SeqLM:AutoModelForSeq2SeqLM:AutoModelForSeq2SeqLM
    betas=1.0:0.0:0.0
    loss=marianmt:ngramsl22:ngramsl22
    lossabbr="logpyx:ngramsl22:ngramsl22"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB
    label_id=0:0:0
    custom_epsilons=true:true
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB
    epsilon_warmup_steps=1:1
    epsilon_cooldown_steps=2:2
    epsilon_decay_functions=step:step
    noise_variance=1.0
    embedgd_do_sample="false"
    LAMBDALR=1.0
elif [[ "$option" == "translate-keyword-3" ]]  ## no prompt just generate directly
then 
    echo "translate-keyword-1"
    OPTIMSTEPS=250
    DATASTYLE="text"
    DATAFILE=data/MT/terminology_dataset/newstest2017-iate.414.3.terminology.tsv.en
    ADDITIONALDATAFILE=data/MT/terminology_dataset/iate/iate.414.3.terminology.tsv
    KEYWORDS="_iate_"
    OUTPUTLEN=150
    MAXLEN=150
    NUM_SAMPLES=1
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForSeq2SeqLM:AutoModelForSeq2SeqLM:AutoModelForSeq2SeqLM:AutoModelForSeq2SeqLM
    betas=1.0:0.0:0.0:0.0
    loss=marianmt:ngramsl22:ngramsl22:ngramsl22
    lossabbr="logpyx:ngramsl22:ngramsl22:ngramsl22"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB
    label_id=0:0:0:0
    custom_epsilons=true:true:true
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB
    epsilon_warmup_steps=1:1:1
    epsilon_cooldown_steps=2:2:2
    epsilon_decay_functions=step:step:step
    noise_variance=1.0
    embedgd_do_sample="false"
    LAMBDALR=1.0
elif [[ "$option" == "translate-keyword-3-padless" ]]  ## no prompt just generate directly
then 
    echo "translate-keyword-1"
    OPTIMSTEPS=250
    DATASTYLE="text"
    DATAFILE=data/MT/terminology_dataset/newstest2017-iate.414.3.terminology.tsv.en
    ADDITIONALDATAFILE=data/MT/terminology_dataset/iate/iate.414.3.terminology.tsv
    KEYWORDS="_iate_pad"
    OUTPUTLEN=150
    MAXLEN=150
    NUM_SAMPLES=1
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForSeq2SeqLM:AutoModelForSeq2SeqLM:AutoModelForSeq2SeqLM:AutoModelForSeq2SeqLM:AutoModelForSeq2SeqLM
    betas=1.0:0.0:0.0:0.0:0.0
    loss=marianmt:ngramsl22:ngramsl22:ngramsl22:ngramsl22
    lossabbr="logpyx:ngramsl22:ngramsl22:ngramsl22:blacklist22"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB
    label_id=0:0:0:0:0
    custom_epsilons=true:true:true:true
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB
    epsilon_warmup_steps=0:0:0:0
    epsilon_cooldown_steps=2:2:2:2
    epsilon_decay_functions=step:step:step:step
    noise_variance=1.0
    embedgd_do_sample="false"
    LAMBDALR=1.0
fi



debug=$4
lr=$5
if [ -z "$lr" ]
then
    lr=0.1
fi

losstype=${6}
if [ -z "${6}" ]
then 
    losstype=dotplusplus
fi


if [[ "$debug" == "debug" ]]
then
    echo $epsilons
    python -W ignore -u decode_new.py\
        --datastyle $DATASTYLE\
        --data $DATAFILE\
        --additional-data $ADDITIONALDATAFILE\
        --jsonl-primary-key $JSON_PKEY\
        --jsonl-secondary-key $JSON_SKEY\
        --jsonl-tertiary-key $JSON_TKEY\
        --jsonl-tokenized $JSONTOK\
        --model $model\
        --tokenizer $tokenizer\
        --model_types $model_types\
        --betas $betas\
        --loss $loss\
        --lossabbr "$lossabbr"\
        --topic-target $TARGETTOPIC\
        --topic-word-lists $WORDLIST\
        --keywords $KEYWORDS\
        --keyword_topk $KEYWORDTOPK\
        --prefix-length 0\
        --model_dtype fp32\
        --fp16_source pytorch\
        --target-type embeds\
        --loss-type $losstype\
        --same-embeds\
        --beam-size 6\
        --always-mucoco $always_mucoco\
        --only-mucoco $only_mucoco\
        --metric l2\
        --max-grad-norm 0\
        --optim-steps $OPTIMSTEPS\
        --max-length $OUTPUTLEN\
        --max-output-length $OUTPUTLEN\
        --max-allowed-length 200\
        --AR-top-k 0\
        --AR-temperature 1.0\
        --AR-top-p 0.95\
        --embedgd-grad-distance $7\
        --embedgd-momentum $8\
        --scale_loss none\
        --log-interval 1\
        --decode-temperature 0.1\
        --label-id $label_id\
        --bos\
        --eos\
        --seed 42\
        --selection_criterion $selection_criterion\
        --linear-scale $linear_scale\
        --debug-gradients $debug_gradients\
        --length-diff=0:2:4:6:8:10:15:20:30\
        --num-samples $NUM_SAMPLES\
        --optim embedgd\
        --init $init_style\
        --coeff-pattern $coeff_pattern\
        --embedgd-gumbel-noise-max 0.0\
        --embedgd-lr-pattern $9\
        --embedgd-do-sample $embedgd_do_sample\
        --embedgd-top-k ${11}\
        --embedgd-top-p ${embedgd_top_p}\
        --embedgd-begin-temperature $begin_temperature\
        --embedgd-final-temperature $final_temperature\
        --embedgd-temperature-reduction-steps $temp_red_steps\
        --embedgd-noise-variance $noise_variance\
        --repetition-penalty $repetition_penalty\
        --gold-loss-epsilons $gold_loss_epsilons\
        --target-tokenize-different\
        --kweight 5.0\
        --lr $lr\
        --dampness $DAMPNESS\
        --epsilons=$epsilons\
        --min_epsilons=$min_epsilons\
        --custom-epsilons $custom_epsilons\
        --epsilon_warmup_steps $epsilon_warmup_steps\
        --epsilon_cooldown_steps $epsilon_cooldown_steps\
        --epsilon_decay_functions $epsilon_decay_functions\
        --lambda-lr $LAMBDALR\
        --lambda-update $LAMBDA_UPDATES\
        --dynamic-lambda-update\
        --dynamic-lr-update\
        --max-lr $MAXLR\
        --lr-update-size $LRUPDATESIZE\
        --num-examples $NUMEXAMPLES\
        --early-stop-steps 20\
        --restarts $RESTARTS\
        --use_context $USECONTEXT\
        --debug
    echo "--epsilons $epsilons"
elif [[ "$debug" == "interactive" ]]
then
    echo "interactive (in debug mode)"
    python -W ignore -u decode_new.py\
        --jsonl-primary-key $JSON_PKEY\
        --jsonl-secondary-key $JSON_SKEY\
        --jsonl-tokenized $JSONTOK\
        --model $model\
        --tokenizer $tokenizer\
        --model_types $model_types\
        --betas $betas\
        --loss $loss\
        --lossabbr "$lossabbr"\
        --topic-target $TARGETTOPIC\
        --topic-word-lists $WORDLIST\
        --keywords $KEYWORDS\
        --keyword_topk $KEYWORDTOPK\
        --prefix-length 0\
        --model_dtype fp32\
        --fp16_source pytorch\
        --target-type embeds\
        --loss-type $losstype\
        --same-embeds\
        --beam-size 6\
        --always-mucoco $always_mucoco\
        --only-mucoco $only_mucoco\
        --metric l2\
        --max-grad-norm 0\
        --optim-steps $OPTIMSTEPS\
        --max-length $OUTPUTLEN\
        --max-output-length $OUTPUTLEN\
        --max-allowed-length 200\
        --AR-top-k 0\
        --AR-temperature 1.0\
        --AR-top-p 0.95\
        --embedgd-grad-distance $7\
        --embedgd-momentum $8\
        --scale_loss none\
        --log-interval 5\
        --decode-temperature 0.1\
        --label-id $label_id\
        --bos\
        --eos\
        --seed 0\
        --selection_criterion $selection_criterion\
        --linear-scale $linear_scale\
        --debug-gradients $debug_gradients\
        --length-diff 5\
        --num-samples $NUM_SAMPLES\
        --optim embedgd\
        --init $init_style\
        --coeff-pattern $coeff_pattern\
        --embedgd-gumbel-noise-max 0.0\
        --embedgd-lr-pattern $9\
        --embedgd-do-sample $embedgd_do_sample\
        --embedgd-top-k ${11}\
        --embedgd-top-p ${embedgd_top_p}\
        --embedgd-begin-temperature $begin_temperature\
        --embedgd-final-temperature $final_temperature\
        --embedgd-temperature-reduction-steps $temp_red_steps\
        --embedgd-noise-variance $noise_variance\
        --repetition-penalty $repetition_penalty\
        --gold-loss-epsilons $gold_loss_epsilons\
        --kweight 5.0\
        --lr $lr\
        --dampness $DAMPNESS\
        --epsilons=$epsilons\
        --min_epsilons=$min_epsilons\
        --epsilon_warmup_steps $epsilon_warmup_steps\
        --epsilon_cooldown_steps $epsilon_cooldown_steps\
        --epsilon_decay_functions $epsilon_decay_functions\
        --lambda-lr $LAMBDALR\
        --lambda-update $LAMBDA_UPDATES\
        --dynamic-lambda-update\
        --dynamic-lr-update\
        --max-lr 0.45\
        --lr-update-size 0.01\
        --num-examples $NUMEXAMPLES\
        --early-stop-steps 50\
        --restarts $RESTARTS\
        --use_context $USECONTEXT\
        --debug
elif [[ "$debug" == "run_and_evaluate" ]]
then
    python -W ignore -u decode_new.py\
        --datastyle $DATASTYLE\
        --data $DATAFILE\
        --additional-data $ADDITIONALDATAFILE\
        --jsonl-primary-key $JSON_PKEY\
        --jsonl-secondary-key $JSON_SKEY\
        --jsonl-tokenized $JSONTOK\
        --model $model\
        --tokenizer $tokenizer\
        --model_types $model_types\
        --betas $betas\
        --loss $loss\
        --lossabbr "$lossabbr"\
        --topic-target $TARGETTOPIC\
        --topic-word-lists $WORDLIST\
        --keywords $KEYWORDS\
        --keyword_topk $KEYWORDTOPK\
        --prefix-length 0\
        --model_dtype fp32\
        --fp16_source pytorch\
        --target-type embeds\
        --loss-type $losstype\
        --same-embeds\
        --beam-size 6\
        --always-mucoco $always_mucoco\
        --only-mucoco $only_mucoco\
        --metric l2\
        --max-grad-norm 0\
        --optim-steps $OPTIMSTEPS\
        --max-length $OUTPUTLEN\
        --max-output-length $OUTPUTLEN\
        --max-allowed-length 200\
        --AR-top-k 0\
        --AR-temperature 1.0\
        --AR-top-p 0.95\
        --embedgd-grad-distance $7\
        --embedgd-momentum $8\
        --scale_loss none\
        --log-interval 25\
        --decode-temperature 0.1\
        --label-id $label_id\
        --bos\
        --eos\
        --seed 42\
        --selection_criterion $selection_criterion\
        --linear-scale $linear_scale\
        --debug-gradients $debug_gradients\
        --length-diff=0:2:4:6:8:10:15:20:30\
        --num-samples $NUM_SAMPLES\
        --optim embedgd\
        --init $init_style\
        --coeff-pattern $coeff_pattern\
        --embedgd-gumbel-noise-max 0.0\
        --embedgd-lr-pattern $9\
        --embedgd-do-sample $embedgd_do_sample\
        --embedgd-top-k ${11}\
        --embedgd-top-p ${embedgd_top_p}\
        --embedgd-begin-temperature $begin_temperature\
        --embedgd-final-temperature $final_temperature\
        --embedgd-temperature-reduction-steps $temp_red_steps\
        --embedgd-noise-variance $noise_variance\
        --repetition-penalty $repetition_penalty\
        --gold-loss-epsilons $gold_loss_epsilons\
        --target-tokenize-different\
        --custom-epsilons $custom_epsilons\
        --kweight 5.0\
        --lr $lr\
        --dampness $DAMPNESS\
        --epsilons=$epsilons\
        --min_epsilons=$min_epsilons\
        --epsilon_warmup_steps $epsilon_warmup_steps\
        --epsilon_cooldown_steps $epsilon_cooldown_steps\
        --epsilon_decay_functions $epsilon_decay_functions\
        --lambda-lr $LAMBDALR\
        --lambda-update $LAMBDA_UPDATES\
        --dynamic-lambda-update\
        --dynamic-lr-update\
        --max-lr $MAXLR\
        --lr-update-size $LRUPDATESIZE\
        --num-examples $NUMEXAMPLES\
        --random-example $RANDOMEXAMPLE\
        --start-idx $STARTIDX\
        --end-idx $ENDIDX\
        --early-stop-steps 10\
        --restarts $RESTARTS\
        --outfile $OUTFILE\
        --output-style $OUTPUTSTYLE
    bash examples/prompt/evaluate.sh $option $OUTFILE $EVALFILE $DATAFILE $TARGETTOPIC
else
    bash examples/prompt/evaluate.sh $option $OUTFILE $EVALFILE $DATAFILE $TARGETTOPIC
fi
