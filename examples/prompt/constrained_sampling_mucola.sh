graddistance=${7}
option=$1
OUTDIR=$2

mkdir -p $OUTDIR

OUTFILE=$OUTDIR/outputs.txt
EVALFILE=$OUTDIR/results${15}.txt

DATA_DIR=data/

PRIMARYMODEL=$3

#for mucola-disc and mucola-two-disc setups
SENTIMENTMODELSST2UNCASED=/path/to/sst2-sentimentclassifier-disc
SENTIMENTMODELYELPUNCASED=/path/to/yelp-sentimentclassifier-disc

#for mucola-gedi setup
SENTIMENTGENERATIVEMODELSST2=/path/to/sst2-sentimentclassifier-gedi
SENTIMENTGENERATIVEMODELYELP=/path/to/sst2-sentimentclassifier-gedi

#for mucola-dexperts setup
SENTIMENTGENERATIVEMODEL2LMSST2=/path/to/finetuned-lm-negative-sst2#/path/to/finetuned-lm-positive-sst2
SENTIMENTGENERATIVEMODEL2LMYELP=/path/to/finetuned-lm-negative-yelp#/path/to/finetuned-lm-positive-yelp

TOXICITYCLASSIFIER=/path/to/toxicity-classifier

#many of these hyperparams were used while experimentation and debugging and do not need to be changed
gold_loss_epsilons="none"
DATASTYLE="text"
ADDITIONALDATAFILE="none"
JSON_PKEY="none"
JSON_SKEY="none"
TARGETTOPIC="none"
KEYWORDS="none"
KEYWORDTOPK=1
KEYWORDTAU=0.01
JSONTOK=false
WORDLIST="none"
OUTPUTLEN=20
MAXLEN=20
LAMBDALR=2.0
selection_criterion="primary_allsat" #primary allsat: select the sample satisfying all constraints with lowest primary loss; mrr_allsat: select the most recent sample satisfying all constraints which is repeating (mrr)
NUM_SAMPLES=1
length_diff=0
linear_scale="false"
OPTIMSTEPS=500
OUTPUTSTYLE="jsonl"
embedgd_do_sample="true"
embedgd_top_p=1.0
noise_variance=0.0
debug_gradients=${14}
begin_temperature=${16}
final_temperature=${17}
temp_red_steps=${18}
init_style=${19}
LABELID=${22} #label ids for the classifier(s) to predict
KEYWORDTAU=${23}
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
DAMPNESS=${36}
custom_epsilons="none"
if [[ -z "$KEYWORDTAU" ]]
then
    KEYWORDTAU=0.01
fi 
if [[ -z "$DAMPNESS" ]]
then
    DAMPNESS=0.1
fi
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

if [[ "$option" == "nontoxic" ]]
then
    echo "nontoxicity"
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/nontoxic_prompts-10k.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25
    OUTPUTLEN=20
    MAXLEN=20
    OPTIMSTEPS=200
    model=$PRIMARYMODEL:$TOXICITYCLASSIFIER
    tokenizer=$PRIMARYMODEL:$TOXICITYTOKENIZER
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification_no_prefix
    label_id=0:0
    lossabbr="pyx:toxicity"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    noise_variance=1.0
    embedgd_do_sample="false"
    LAMBDALR=1.0
    selection_criterion="mrr_allsat"
elif [[ "$option" == "sentiment-disc" ]]
then
    echo "sentiment-disc (label ${LABELID})"
    OPTIMSTEPS=300
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/control-prompts/pplm-discrim-prompts/prompts.txt
    NUM_SAMPLES=20  
    OUTPUTLEN=$OUTPUTLEN
    MAXLEN=$OUTPUTLEN
    sentmodel=SENTIMENTMODEL${SENTIMENTMODELID}UNCASED
    echo "model",${!sentmodel}
    model=$PRIMARYMODEL:${!sentmodel}
    tokenizer=$PRIMARYMODEL:${!sentmodel}
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classificationlogits
    label_id=0:$LABELID
    lossabbr="pyx:binary"
    selection_criterion="mrr_allsat"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
    option="sentiment"
elif [[ "$option" == "sentiment-two-disc" ]]
then
    echo "sentiment-two-disc (label ${LABELID})"
    OPTIMSTEPS=250
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/control-prompts/pplm-discrim-prompts/prompts.txt
    NUM_SAMPLES=20  
    MAXLEN=$OUTPUTLEN
    model=$PRIMARYMODEL:$SENTIMENTMODELSST2UNCASED:$SENTIMENTMODELYELPUNCASED
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODELSST2UNCASED:$SENTIMENTMODELYELPUNCASED
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification:RobertaCustomForSequenceClassification
    betas=0.8:0.2:0.0
    loss=gpt2:classificationlogits:classificationlogits
    label_id=0:$LABELID:$LABELID
    lossabbr="pyx:binary:binary2"
    selection_criterion="mrr_allsat"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB
    epsilon_warmup_steps=0:0
    epsilon_cooldown_steps=1:1
    epsilon_decay_functions=linear:linear
    LAMBDALR=1.0
    noise_variance=0.0
    embedgd_do_sample="false"
    option="sentiment"
elif [[ "$option" == "sentiment-gedi" ]]
then
    echo "sentiment-gedi (label ${LABELID})"
    OPTIMSTEPS=250
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/control-prompts/pplm-discrim-prompts/prompts.txt
    NUM_SAMPLES=20  
    OUTPUTLEN=$OUTPUTLEN
    MAXLEN=$OUTPUTLEN
    sentmodel=SENTIMENTGENERATIVEMODEL${SENTIMENTMODELID}
    #echo ${!sentmodel}
    model=$PRIMARYMODEL:${!sentmodel}
    tokenizer=$PRIMARYMODEL:${!sentmodel}
    model_types=AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.8:0.2
    loss=gpt2:generativeclassificationgpt2
    label_id=0:$LABELID
    lossabbr="pyx:binary"
    selection_criterion="mrr_allsat"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
    option="sentiment"
elif [[ "$option" == "sentiment-dexperts" ]]
then
    echo "sentiment-dexperts (label ${LABELID})"
    OPTIMSTEPS=250
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/control-prompts/pplm-discrim-prompts/prompts.txt
    NUM_SAMPLES=20  
    OUTPUTLEN=$OUTPUTLEN
    MAXLEN=$OUTPUTLEN
    sentmodel=SENTIMENTGENERATIVEMODEL2LM${SENTIMENTMODELID}
    echo ${!sentmodel}
    model=$PRIMARYMODEL:${!sentmodel}
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.8:0.2
    loss=gpt2:generativeclassification2lm
    label_id=0:$LABELID
    lossabbr="pyx:binary"
    selection_criterion="mrr_allsat"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
    option="positive-tiny"
elif [[ "$option" == "sentiment-prompt" ]]
then
    echo "sentiment prompt (label $LABELID)"
    OPTIMSTEPS=500
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/control-prompts/pplm-discrim-prompts/prompts.txt
    NUM_SAMPLES=20  
    OUTPUTLEN=$OUTPUTLEN
    MAXLEN=$OUTPUTLEN
    model=$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.8:0.2
    loss=gpt2:generativeclassificationgpt22
    label_id=0:$LABELID
    lossabbr="pyx:binary"
    selection_criterion="mrr_allsat"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=0.5
    noise_variance=1.0
    embedgd_do_sample="false"
    option="positive-tiny"
elif [[ "$option" == "unique-l2-commongen-l2-4" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/commongen/$SENTIMENTMODELID.txt
    ADDITIONALDATAFILE=$DATA_DIR/commongen/commongen.test_noref.jsonl_4
    KEYWORDS="_commongenunique_"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=1
    MAXLEN=$OUTPUTLEN
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4:0.0:0.0:0.0
    loss=gpt2:ngramsl2:ngramsl2:ngramsl2:ngramsl2:uniquel2
    label_id=0:0:0:0:0:0
    lossabbr="pyx:dist1:dist2:dist3:dist4:unlikelihood"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1
    custom_epsilons=true:true:true:true:false
    epsilon_warmup_steps=50:50:50:50:50
    epsilon_cooldown_steps=50:50:50:50:50
    epsilon_decay_functions=step:step:step:step:step
    LAMBDALR=1.0
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
    option="commongen"
    EXTRAS=$ADDITIONALDATAFILE
elif [[ "$option" == "unique-l2-commongen-l2-n" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/commongen/allblankprompts.txt
    ADDITIONALDATAFILE=$DATA_DIR/commongen/commongen.test_noref.jsonl_$SENTIMENTMODELID
    KEYWORDS="_commongenunique_n"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=1
    MAXLEN=$OUTPUTLEN
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4:0.0:0.0:0.0:0.0
    loss=gpt2:ngramsl2:ngramsl2:ngramsl2:ngramsl2:uniquel2:blacklistl2
    label_id=0:0:0:0:0:0:0
    lossabbr="pyx:dist1:dist2:dist3:dist4:unlikelihood:blacklist-dist"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1:$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1:$CLASSLOGPROB
    custom_epsilons=true:true:true:true:false:true
    epsilon_warmup_steps=1:1:1:1:1:1
    epsilon_cooldown_steps=50:50:50:50:50:50
    epsilon_decay_functions=step:step:step:step:step:step
    LAMBDALR=1.0
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
    option="commongen"
    EXTRAS=$ADDITIONALDATAFILE
elif [[ "$option" == "unique-l2-commongen-l2-4-nn" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/commongen/allblankprompts.txt
    ADDITIONALDATAFILE=$DATA_DIR/commongen/commongen.test_noref.jsonl_4
    KEYWORDS="_commongenunique_nn"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=1
    MAXLEN=$OUTPUTLEN
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4:0.0:0.0:0.0:0.0:0.0
    loss=gpt2:ngramsl22:ngramsl22:ngramsl22:ngramsl22:uniquel2:blacklistl22:blacklistl22
    label_id=0:0:0:0:0:0:0:0
    lossabbr="pyx:d1:d2:d3:d4:ul:bd1:bd2"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1:$CLASSLOGPROB:$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1:$CLASSLOGPROB:$CLASSLOGPROB
    custom_epsilons=true:true:true:true:false:true:true
    epsilon_warmup_steps=1:1:1:1:1:1:1
    epsilon_cooldown_steps=50:50:50:50:50:50:50
    epsilon_decay_functions=step:step:step:step:step:step:step
    LAMBDALR=1.0
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
    option="commongen"
    EXTRAS=$ADDITIONALDATAFILE
elif [[ "$option" == "unique-l2-commongenmorpho-l2-4-nn" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/commongen/allblankprompts.txt
    ADDITIONALDATAFILE=$DATA_DIR/commongen/test.constraint.json_4
    KEYWORDS="_commongenmorphounique_nn"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=1
    MAXLEN=$OUTPUTLEN
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4:0.0:0.0:0.0:0.0:0.0
    loss=gpt2:ngramsl22:ngramsl22:ngramsl22:ngramsl22:uniquel2:blacklistl22:blacklistl22
    label_id=0:0:0:0:0:0:0:0
    lossabbr="pyx:d1:d2:d3:d4:ul:bd1:bd2"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1:$CLASSLOGPROB:$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1:$CLASSLOGPROB:$CLASSLOGPROB
    custom_epsilons=true:true:true:true:false:true:true
    epsilon_warmup_steps=1:1:1:1:1:1:1
    epsilon_cooldown_steps=50:50:50:50:50:50:50
    epsilon_decay_functions=step:step:step:step:step:step:step
    LAMBDALR=1.0
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
    option="commongen"
    EXTRAS=$ADDITIONALDATAFILE
elif [[ "$option" == "unique-l2-commongen-l2-5-nn" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/commongen/allblankprompts.txt
    ADDITIONALDATAFILE=$DATA_DIR/commongen/commongen.test_noref.jsonl_5
    KEYWORDS="_commongenunique_nn"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=1
    MAXLEN=$OUTPUTLEN
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4:0.0:0.0:0.0:0.0:0.0:0.0
    loss=gpt2:ngramsl22:ngramsl22:ngramsl22:ngramsl22:ngramsl22:uniquel2:blacklistl22:blacklistl22
    label_id=0:0:0:0:0:0:0:0:0
    lossabbr="pyx:d1:d2:d3:d4:d5:ul:bd1:bd2"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1:$CLASSLOGPROB:$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1:$CLASSLOGPROB:$CLASSLOGPROB
    custom_epsilons=true:true:true:true:true:false:true:true
    epsilon_warmup_steps=1:1:1:1:1:1:1:1
    epsilon_cooldown_steps=50:50:50:50:50:50:50:50
    epsilon_decay_functions=step:step:step:step:step:step:step:step
    LAMBDALR=1.0
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
    option="commongen"
    EXTRAS=$ADDITIONALDATAFILE
elif [[ "$option" == "unique-l2-commongenmorpho-l2-5-nn" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/commongen/allblankprompts.txt
    # ADDITIONALDATAFILE=$DATA_DIR/commongen/commongen.test_noref.jsonl_5
    ADDITIONALDATAFILE=$DATA_DIR/commongen/test.constraint.json_5
    KEYWORDS="_commongenmorphounique_nn"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=1
    MAXLEN=$OUTPUTLEN
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4:0.0:0.0:0.0:0.0:0.0:0.0
    loss=gpt2:ngramsl22:ngramsl22:ngramsl22:ngramsl22:ngramsl22:uniquel2:blacklistl22:blacklistl22
    label_id=0:0:0:0:0:0:0:0:0
    lossabbr="pyx:d1:d2:d3:d4:d5:ul:bd1:bd2"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1:$CLASSLOGPROB:$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1:$CLASSLOGPROB:$CLASSLOGPROB
    custom_epsilons=true:true:true:true:true:false:true:true
    epsilon_warmup_steps=1:1:1:1:1:1:1:1
    epsilon_cooldown_steps=50:50:50:50:50:50:50:50
    epsilon_decay_functions=step:step:step:step:step:step:step:step
    LAMBDALR=1.0
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
    option="commongen"
    EXTRAS=$ADDITIONALDATAFILE
elif [[ "$option" == "unique-l2-commongen-l2-5" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/commongen/$SENTIMENTMODELID.txt
    ADDITIONALDATAFILE=$DATA_DIR/commongen/commongen.test_noref.jsonl_5
    KEYWORDS="_commongenunique_"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=1
    MAXLEN=$OUTPUTLEN
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4:0.0:0.0:0.0:0.0
    loss=gpt2:ngramsl2:ngramsl2:ngramsl2:ngramsl2:ngramsl2:uniquel2
    label_id=0:0:0:0:0:0:0
    lossabbr="pyx:dist1:dist2:dist3:dist4:dist5:unlikelihood"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1
    custom_epsilons=true:true:true:true:true:false
    epsilon_warmup_steps=1:1:1:1:1:1
    epsilon_cooldown_steps=50:50:50:50:50:50
    epsilon_decay_functions=step:step:step:step:step:step
    LAMBDALR=1.0
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
    option="commongen"
    EXTRAS=$ADDITIONALDATAFILE
elif [[ "$option" == "unique-roc-l2" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/keywords-k2t/ROC/prompts.txt
    ADDITIONALDATAFILE=$DATA_DIR/keywords-k2t/ROC/ROCStories_20_storylines_500_0.txt
    KEYWORDS="_rocunique_"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=1
    MAXLEN=$OUTPUTLEN
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4:0.0:0.0:0.0
    loss=gpt2:ngramsl2:ngramsl2:ngramsl2:ngramsl2:unique
    label_id=0:0:0:0:0:0
    lossabbr="pyx:dist1:dist2:dist3:dist4:unlikelihood"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-6.5
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-6.5
    custom_epsilons=true:true:true:true:false
    epsilon_warmup_steps=50:50:50:50:50
    epsilon_cooldown_steps=50:50:50:50:50
    epsilon_decay_functions=step:step:step:step:step
    LAMBDALR=1.0
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
    option="roc"
    EXTRAS=$ADDITIONALDATAFILE
elif [[ "$option" == "unique-l2-roc-l2" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/keywords-k2t/ROC/prompts.txt
    ADDITIONALDATAFILE=$DATA_DIR/keywords-k2t/ROC/ROCStories_20_storylines_500_0.txt
    KEYWORDS="_rocunique_"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=1
    MAXLEN=$OUTPUTLEN
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4:0.0:0.0:0.0:0.0
    loss=gpt2:ngramsl2multiples:ngramsl2multiple:ngramsl2multiple:ngramsl2multiple:ngramsl2multiple:uniquel2
    label_id=0:0:0:0:0:0:0
    lossabbr="pyx:dist1:dist2:dist3:dist4:dist5:unlikelihood"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1
    custom_epsilons=true:true:true:true:true:false
    # epsilon_warmup_steps=50:50:50:50:50
    epsilon_warmup_steps=0:0:0:0:0:0
    epsilon_cooldown_steps=50:50:50:50:50:50
    epsilon_decay_functions=step:step:step:step:step:step
    LAMBDALR=1.0
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
    option="roc"
    EXTRAS=$ADDITIONALDATAFILE
elif [[ "$option" == "unique-l2-roc-l2-nn" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/keywords-k2t/ROC/prompts.txt
    ADDITIONALDATAFILE=$DATA_DIR/keywords-k2t/ROC/ROCStories_20_storylines_500_0.txt
    KEYWORDS="_rocunique_nn"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=1
    MAXLEN=$OUTPUTLEN  
    NUM_SAMPLES=1
    MAXLEN=$OUTPUTLEN
    OPTIMSTEPS=1000
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4:0.0:0.0:0.0:0.0:0.0:0.0
    loss=gpt2:ngramsl22:ngramsl22:ngramsl22:ngramsl22:ngramsl22:uniquel2:blacklistl22:blacklistl22
    label_id=0:0:0:0:0:0:0:0:0
    lossabbr="pyx:d1:d2:d3:d4:d5:ul:bd1:bd2"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1:$CLASSLOGPROB:$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-1:$CLASSLOGPROB:$CLASSLOGPROB
    custom_epsilons=true:true:true:true:true:false:true:true
    # custom_epsilons=false:false:false:false:false:false:true:true
    epsilon_warmup_steps=1:1:1:1:1:1:1:1
    epsilon_cooldown_steps=50:50:50:50:50:50:50:50
    epsilon_decay_functions=step:step:step:step:step:step:step:step
    LAMBDALR=1.0
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
    option="roc"
    EXTRAS=$ADDITIONALDATAFILE
elif [[ "$option" == "unique-commongen-l2-2" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/commongen/allprompts.txt
    # DATASTYLE="jsonl"
    # JSON_PKEY="concept_set"
    # DATAFILE=$DATA_DIR/commongen/commongen.test_noref.jsonl
    ADDITIONALDATAFILE=$DATA_DIR/commongen/commongen.test_noref.jsonl
    KEYWORDS="_commongenunique_"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=1
    MAXLEN=$OUTPUTLEN
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4:0.0:0.0:0.0
    loss=gpt2:ngramsl2:ngramsl2:ngramsl2:ngramsl2:unique
    label_id=0:0:0:0:0:0
    lossabbr="pyx:dist1:dist2:dist3:dist4:unlikelihood"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-6
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:-6
    custom_epsilons=true:true:true:true:false
    epsilon_warmup_steps=0:0:0:0:0
    epsilon_cooldown_steps=1:1:1:1:1
    epsilon_decay_functions=linear:linear:linear:linear:linear
    LAMBDALR=1.0
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
    option="commongen"
    EXTRAS=$ADDITIONALDATAFILE
elif [[ "$option" == "bart-summarize" ]]  ## no prompt just generate directly
then 
    echo "bart-summarize"
    OPTIMSTEPS=200
    DATASTYLE="jsonl"
    JSON_PKEY="article"
    DATAFILE=$DATA_DIR/../cnn-dailymail/test.jsonl
    OUTPUTLEN=50
    MAXLEN=50
    NUM_SAMPLES=1
    model=$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL
    model_types=AutoModelForSeq2SeqLM
    betas=1.0
    loss=bart
    lossabbr="logpyx"
    epsilons=none
    label_id=none
    min_epsilons=none
    epsilon_warmup_steps=none
    epsilon_cooldown_steps=none
    epsilon_decay_functions=none
    noise_variance=1.0
    embedgd_do_sample="false"
    LAMBDALR=1.0
elif [[ "$option" == "bart-summarize-keyword" ]]  ## no prompt just generate directly
then 
    echo "bart-summarize-keyword"
    OPTIMSTEPS=200
    DATASTYLE="jsonl"
    JSON_PKEY="article"
    KEYWORDS="none:${10}"
    DATAFILE=$DATA_DIR/../cnn-dailymail/test.jsonl
    OUTPUTLEN=50
    MAXLEN=50
    NUM_SAMPLES=1
    model=$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForSeq2SeqLM:AutoModelForSeq2SeqLM
    betas=1.0:0.0
    loss=bart:ngrams
    lossabbr="logpyx:ngrams"
    epsilons=$CLASSLOGPROB
    label_id=0:0
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=step
    noise_variance=1.0
    embedgd_do_sample="false"
    LAMBDALR=1.0
else
    echo "wrong task"
    exit 1
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

# echo $TARGETTOPIC
# echo "noise-variance=${noise_variance}"

if [[ "$debug" == "debug" ]]
then
    echo $epsilons
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
        --keyword_tau $KEYWORDTAU\
        --prefix-length 0\
        --model_dtype fp32\
        --fp16_source pytorch\
        --target-type embeds\
        --loss-type $losstype\
        --same-embeds\
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
        --AR-top-p 0.96\
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
        --length-diff 0\
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
        --early-stop-steps 40\
        --restarts $RESTARTS\
        --use_context $USECONTEXT\
        --show-all-outputs\
        --allow-diff-vocab\
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
        --keyword_tau $KEYWORDTAU\
        --prefix-length 0\
        --model_dtype fp32\
        --fp16_source pytorch\
        --target-type embeds\
        --loss-type $losstype\
        --same-embeds\
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
        --AR-top-p 0.96\
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
        --length-diff 0\
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
        --keyword_tau $KEYWORDTAU\
        --prefix-length 0\
        --model_dtype fp32\
        --fp16_source pytorch\
        --target-type embeds\
        --loss-type $losstype\
        --same-embeds\
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
        --AR-top-p 0.96\
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
        --length-diff 0\
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
        --early-stop-steps 40\
        --restarts $RESTARTS\
        --outfile $OUTFILE\
        --output-style $OUTPUTSTYLE\
        --allow-diff-vocab
    bash examples/prompt/evaluate.sh $option $OUTFILE $EVALFILE $EXTRAS $DATAFILE 
    done="true"
else
    if [[ "$done" != "true" ]]
    then
        echo $EXTRAS
        bash examples/prompt/evaluate.sh $option $OUTFILE $EVALFILE $EXTRAS $DATAFILE 
    fi
fi
