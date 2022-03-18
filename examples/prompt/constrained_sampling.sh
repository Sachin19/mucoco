# if [ -z "${7}" ]
# then
#     source activate langvar
# else
#     source activate ${7}
# fi 
source activate 2022
graddistance=${7}
option=$1
outdir=$2

OUTDIR=/projects/tir5/users/sachink/embed-style-transfer/data/predictions/prompttest/$outdir
mkdir -p $OUTDIR

OUTFILE=$OUTDIR/outputs.txt
EVALFILE=$OUTDIR/results${15}.txt

DATA_DIR=/projects/tir5/users/sachink/embed-style-transfer/data/prompts/

PRIMARYMODEL=$3
FORMALITYMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-frozen-embeds-formality-classifier/checkpoint_best/
BARTFORMALITYMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-frozen-embeds-bart-base-formality-classifier/checkpoint_best
# SENTIMENTMODEL2=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-sst-sentiment-classifier/checkpoint_best/
# SENTIMENTMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-sentiment-binary-classifier/checkpoint_best/
# SENTIMENTMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-sst-2-with-gpt2-large-embeds/checkpoint_best
# SENTIMENTMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-medium-sst-2-with-gpt2-large-embeds/checkpoint_best
# SENTIMENTMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-medium-sentiment-binary-classifier-with-gp2-large-embeds/results/checkpoint-12000/
# SENTIMENTMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-sentiment-5-classifier/checkpoint_best/
# SENTIMENTMODEL1=/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-sst-2-with-gpt2-large-embeds/checkpoint_best
# SENTIMENTMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-sst-2-with-gpt2-large-embeds${23}/checkpoint_best
# SENTIMENTMODEL4=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-medium-jmamou-sst-2-with-gpt2-large-embeds/checkpoint_best
# SENTIMENTMODEL3=/projects/tir5/users/sachink/embed-style-transfer/models/roberta-large-sst-2-with-gpt2-large-embeds/checkpoint_best
# SENTIMENTMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-textattack-sst-2-with-gpt2-large-embeds/checkpoint_best
SENTIMENTMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-textattack-very-sst-2-with-gpt2-large-embeds/checkpoint_best/
SENTIMENTMODEL3=/projects/tir5/users/sachink/embed-style-transfer/code/models/roberta-base-textattack-very-yelp-with-gpt2-large-embeds/checkpoint_best
SENTIMENTMODEL2=/projects/tir5/users/sachink/embed-style-transfer/code/models/roberta-base-victorsanh-yelp-with-gpt2-large-embeds/checkpoint_best
# SENTIMENTMODEL2=/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-sst-2-with-gpt2-large-embeds/checkpoint_best
SHAKESPEAREMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-Shakespeare-frozen-embeds/checkpoint_best/
ENGLISHESMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-englishes-frozen-embeds/results/checkpoint-1000
# TOXICITYCLASSIFIER=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-jigsaw-freeze-embeds-toxicity-classifier/checkpoint_best
# TOXICITYTOKENIZER=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-jigsaw-freeze-embeds-toxicity-classifier/checkpoint_best
TOXICITYCLASSIFIER=/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/checkpoint_best
TOXICITYTOKENIZER=/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds/checkpoint_best

STSMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-nli-stsb-mean-tokens/0_Transformer
STYLEFORMALITYMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-formality-classifier/checkpoint_best/

gold_loss_epsilons="none"
mal=5

echo "SENTIMENTMODEL=$SENTIMENTMODEL"
DATASTYLE="text"
ADDITIONALDATAFILE="none"
JSON_PKEY="none"
JSON_SKEY="none"
TARGETTOPIC="none"
KEYWORDS="the"
KEYWORDTOPK=1
JSONTOK=false
WORDLIST="none"
OUTPUTLEN=20
MAXLEN=20
LAMBDALR=2.0
selection_criterion="primary_allsat"
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
coeff_pattern=${22}
LAMBDA_UPDATES=${24}
CLASSLOGPROB=${25}
always_mucoco=${26}
only_mucoco=${27}
MAXLR=${28}
LRUPDATESIZE=${29}
RESTARTS=${30}
RANDOMEXAMPLE=${31}
STARTIDX=${32}
ENDIDX=${33}
USECONTEXT="false"
if [[ -z "$RANDOMEXAMPLE" ]]
then
    RANDOMEXAMPLE="true"
    # p > 0.5 default
fi
if [[ -z "$STARTIDX" ]]
then
    STARTIDX=0
fi
if [[ -z "$ENDIDX" ]]
then
    ENDIDX=-1
fi
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

if [[ "$option" == "unconditional" ]]  ## no prompt just generate directly
then 
    echo "unconditional"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/blanks.txt
    OUTPUTLEN=200
    MAXLEN=200
    NUM_SAMPLES=1
    model=$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL
    model_types=AutoModelForCausalLM
    betas=1.0
    loss=gpt2
    lossabbr="logpyx"
    embedgd_do_sample="false"
    epsilons=none
    label_id=none
    min_epsilons=none
    epsilon_warmup_steps=none
    epsilon_cooldown_steps=none
    epsilon_decay_functions=none
    noise_variance=1.0
elif [[ "$option" == "unconditional-wiki" ]]  ## no prompt just generate directly
then 
    echo "unconditional"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/blanks.txt
    OUTPUTLEN=50
    MAXLEN=50
    NUM_SAMPLES=1
    model=$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL
    model_types=AutoModelForCausalLM
    betas=1.0
    loss=gpt2
    lossabbr="logpyx"
    epsilons=none
    label_id=none
    min_epsilons=none
    epsilon_warmup_steps=none
    epsilon_cooldown_steps=none
    epsilon_decay_functions=none
elif [[ "$option" == "unique" ]]  ## no prompt just generate directly with uniqueness constraint to reduce repetitions
then 
    echo "unique"
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/unconstrained-prompts-degen/data/conditional/conditional_gold.jsonl
    JSON_PKEY="context"
    JSON_SKEY="none"
    JSONTOK=true
    NUM_SAMPLES=1 #CHECK
    OUTPUTLEN=100
    MAXLEN=100
    OUTFILE=$OUTDIR/outputs.jsonl
    model=$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.8:0.2
    noise_variance=1.0
    embedgd_do_sample="false"
    loss=gpt2:unique
    lossabbr="logpyx:unlikelihood"
    linear_scale="false"
    selection_criterion="mrr_allsat"
    epsilons=$CLASSLOGPROB
    label_id=none
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=1.0
elif [[ "$option" == "debug" ]]  ## no prompt just generate directly with uniqueness constraint to reduce repetitions
then 
    echo "debugging gradient mode"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/blanks.txt
    OUTPUTLEN=20
    MAXLEN=20
    model=$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM
    betas=1.0:0.0
    loss=gpt2:gpt2other
    lossabbr="logpyx:logpyx-other"
    selection_criterion="weighted_sum"
    linear_scale="true"
    embedgd_do_sample="false"
    epsilons=0.0
    label_id=none
    min_epsilons=0.0
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
elif [[ "$option" == "conditional" ]]  ## Just a prompt with no constraint
then 
    echo "plain"
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/unconstrained-prompts-degen/data/conditional/conditional_gold.jsonl
    JSON_PKEY="context"
    JSON_SKEY="none"
    JSONTOK=true
    NUM_SAMPLES=1 #CHECK
    OUTPUTLEN=50
    MAXLEN=50
    OUTFILE=$OUTDIR/outputs.jsonl
    model=$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL
    embedgd_top_p=${21}
    model_types=AutoModelForCausalLM
    betas=1.0
    loss=gpt2
    lossabbr="logpyx"
    embedgd_do_sample="false"
    noise_variance=1.0
    epsilons=none
    label_id=none
    min_epsilons=none
    epsilon_warmup_steps=none
    epsilon_cooldown_steps=none
    epsilon_decay_functions=none
elif [[ "$option" == "conditional-nucleus" ]]  ## Just a prompt with no constraint
then 
    echo "plain"
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/unconstrained-prompts-degen/data/conditional/conditional_gold.jsonl
    JSON_PKEY="context"
    JSON_SKEY="none"
    JSONTOK=true
    NUM_SAMPLES=1 #CHECK
    OUTPUTLEN=200
    MAXLEN=200
    OUTFILE=$OUTDIR/outputs.jsonl
    model=$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL
    model_types=AutoModelForCausalLM
    betas=1.0
    loss=gpt2
    lossabbr="logpyx"
    epsilons=none
    label_id=none
    min_epsilons=none
    epsilon_warmup_steps=none
    epsilon_cooldown_steps=none
    epsilon_decay_functions=none
elif [[ "$option" == "positive-tiny" ]]
then
    echo "positive-tiny"
    OPTIMSTEPS=250
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/control-prompts/pplm-discrim-prompts/prompts.txt
    NUM_SAMPLES=100  
    OUTPUTLEN=20
    MAXLEN=20
    model=$PRIMARYMODEL:$SENTIMENTMODEL
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification
    label_id=0:1
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
elif [[ "$option" == "negative-tiny" ]]
then
    echo "negative-tiny"
    OPTIMSTEPS=250
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/control-prompts/pplm-discrim-prompts/prompts.txt
    NUM_SAMPLES=100  
    OUTPUTLEN=40
    MAXLEN=40
    model=$PRIMARYMODEL:$SENTIMENTMODEL
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification
    label_id=0:0
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
elif [[ "$option" == "positive-adv" ]]
then
    echo "positive-adv"
    OPTIMSTEPS=200
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/negative_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL:$SENTIMENTMODEL
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification
    label_id=0:1
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
elif [[ "$option" == "negative-adv" ]]
then
    echo "negative-adv"
    OPTIMSTEPS=200
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/positive_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL:$SENTIMENTMODEL
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification
    label_id=0:0
    lossabbr="pyx:binary"
    selection_criterion="mrr_allsat"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=2.0
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "positive-neutral" ]]
then
    echo "positive-neutral"
    OPTIMSTEPS=250
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/neutral_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL:$SENTIMENTMODEL
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification
    label_id=0:1
    lossabbr="pyx:binary"
    selection_criterion="mrr_allsat"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=step
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "positive-neutral-pa" ]]
then
    echo "positive-neutral"
    OPTIMSTEPS=250
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/neutral_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL:$SENTIMENTMODEL
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification
    label_id=0:1
    lossabbr="pyx:binary"
    selection_criterion="primary_allsat"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "negative-neutral-pa" ]]
then
    echo "positive-neutral"
    OPTIMSTEPS=250
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/neutral_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL:$SENTIMENTMODEL
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification
    label_id=0:0
    lossabbr="pyx:binary"
    selection_criterion="primary_allsat"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "positive-neutral-yelp" ]]
then
    echo "positive-neutral"
    OPTIMSTEPS=250
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/neutral_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL:$SENTIMENTMODEL2
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL2
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification
    label_id=0:1
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
elif [[ "$option" == "positive-neutral-yelp2" ]]
then
    echo "positive-neutral"
    OPTIMSTEPS=250
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/neutral_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL:$SENTIMENTMODEL3
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL3
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification
    label_id=0:1
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
elif [[ "$option" == "positive-neutral2" ]]
then
    echo "positive-neutral"
    OPTIMSTEPS=250
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/neutral_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL:$SENTIMENTMODEL:$SENTIMENTMODEL2
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL:$SENTIMENTMODEL
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification:RobertaCustomForSequenceClassification
    betas=0.8:0.2:0.0
    loss=gpt2:classification:classification
    label_id=0:1:1
    lossabbr="pyx:binary:binary2"
    selection_criterion="mrr_allsat"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB
    epsilon_warmup_steps=0:0
    epsilon_cooldown_steps=1:1
    epsilon_decay_functions=linear:linear
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "unconstrained-neutral" ]]
then
    echo "unconstrained-neutral"
    OPTIMSTEPS=200
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/neutral_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL
    model_types=AutoModelForCausalLM
    betas=1.0
    loss=gpt2
    label_id=0
    lossabbr="pyx"
    epsilons=none
    label_id=none
    min_epsilons=none
    epsilon_warmup_steps=none
    epsilon_cooldown_steps=none
    epsilon_decay_functions=none
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "negative-neutral" ]]
then
    echo "negative-neutral"
    OPTIMSTEPS=200
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/neutral_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    selection_criterion="mrr_allsat"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL:$SENTIMENTMODEL
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification
    label_id=0:0
    lossabbr="pyx:binary"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "negative-neutral2" ]]
then
    echo "negative-neutral2"
    OPTIMSTEPS=200
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/neutral_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    selection_criterion="mrr_allsat"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL:$SENTIMENTMODEL:$SENTIMENTMODEL2
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL:$SENTIMENTMODEL
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification:RobertaCustomForSequenceClassification
    betas=0.8:0.2:0.0
    loss=gpt2:classification:classification
    label_id=0:0:0
    lossabbr="pyx:binary:binary2"
    selection_criterion="mrr_allsat"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB
    epsilon_warmup_steps=0:0
    epsilon_cooldown_steps=1:1
    epsilon_decay_functions=linear:linear
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "negative-neutral2" ]]
then
    echo "negative-neutral2"
    OPTIMSTEPS=200
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/neutral_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL:$SENTIMENTMODEL:$SENTIMENTMODEL1
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL:$SENTIMENTMODEL1
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification:RobertaCustomForSequenceClassification
    betas=0.8:0.2:0.0
    loss=gpt2:classification:classification
    label_id=0:0:0
    lossabbr="pyx:binary:binary2"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB
    epsilon_warmup_steps=0:0
    epsilon_cooldown_steps=1:1
    epsilon_decay_functions=linear:linear
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "positive-neutral-p" ]]
then
    echo "positive-neutral"
    OPTIMSTEPS=200
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/neutral_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL:$SENTIMENTMODEL
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification_no_prefix
    label_id=0:1
    lossabbr="pyx:binary"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "negative-neutral-p" ]]
then
    echo "negative-neutral"
    OPTIMSTEPS=200
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/sentiment_prompts-10k/neutral_prompts.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25  
    model=$PRIMARYMODEL:$SENTIMENTMODEL
    tokenizer=$PRIMARYMODEL:$SENTIMENTMODEL
    model_types=AutoModelForCausalLM:RobertaCustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification_no_prefix
    label_id=0:0
    lossabbr="pyx:binary"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "abductive" ]]
then
    echo "abductive"
    model=$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL
    NUM_SAMPLES=1
    OPTIMSTEPS=200
    OUTPUTLEN=20
    MAXLEN=20
    DATASTYLE="single-jsonl"
    JSON_PKEY="obs1"
    JSON_SKEY="obs2"
    USECONTEXT="true"
    DATAFILE=$DATA_DIR/alpha-nlg/anlg/test-w-comet-preds.jsonl
    model_types=AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.8:0.2
    selection_criterion="mrr_allsat"
    loss=gpt2:gpt2context
    label_id=0:0
    lossabbr="pyx:pzyx"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=20
    epsilon_decay_functions=linear
    gold_loss_epsilons="true"
    LAMBDALR=1.0
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "topic" ]]
then
    echo "topic"
    DATASTYLE="text"
    OPTIMSTEPS=500
    DATAFILE=$DATA_DIR/control-prompts/topic-prompts.txt
    TARGETTOPIC="${10}"
    # selection_criterion="weighted_sum"
    linear_scale="false"
    WORDLIST=/projects/tir5/users/sachink/embed-style-transfer/related-work/naacl-2021-fudge-controlled-generation/topic_data/wordlists/
    NUM_SAMPLES=1
    OUTPUTLEN=40
    MAXLEN=40
    KEYWORDTOPK=${12}
    model=$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.5:0.5
    loss=gpt2:keywordclassification
    label_id=0:0
    lossabbr="pyx:dist"
    selection_criterion="mrr_allsat"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=10.0
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "keyword" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/control-prompts/topic-prompts.txt
    KEYWORDS="${10}"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=1
    MAXLEN=10
    OUTPUTLEN=10
    model=$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.5:0.5
    noise_variance=1.0
    embedgd_do_sample="false"
    loss=gpt2:ngrams
    label_id=0:0
    lossabbr="pyx:dist"
    epsilons=$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=0.5
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
elif [[ "$option" == "keyword2" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/blanks.txt
    KEYWORDS="${10}"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=1
    MAXLEN=20
    OUTPUTLEN=20
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4
    loss=gpt2:ngrams:ngrams
    label_id=0:0:0
    lossabbr="pyx:dist1:dist2"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB
    epsilon_warmup_steps=0:0
    epsilon_cooldown_steps=1:1
    epsilon_decay_functions=linear:linear
    LAMBDALR=2.0
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
elif [[ "$option" == "keyword5" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/keywords-k2t/ROC/prompts.txt
    ADDITIONALDATAFILE=$DATA_DIR/keywords-k2t/ROC/ROCStories_20_storylines_500_0.txt
    KEYWORDS="_roc_"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=5
    MAXLEN=90
    OUTPUTLEN=90
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4:0.0:0.0:0.0
    loss=gpt2:ngrams:ngrams:ngrams:ngrams:ngrams
    label_id=0:0:0:0:0:0
    lossabbr="pyx:dist1:dist2:dist3:dist4:dist5"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB
    epsilon_warmup_steps=0:0:0:0:0
    epsilon_cooldown_steps=1:1:1:1:1
    epsilon_decay_functions=linear:linear:linear:linear:linear
    LAMBDALR=0.5
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
elif [[ "$option" == "commongen" ]]
then
    echo "keyword"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/commongen/prompts.txt
    ADDITIONALDATAFILE=$DATA_DIR/commongen/commongen.test_noref.jsonl
    KEYWORDS="_commongen_"
    # selection_criterion="weighted_sum"
    NUM_SAMPLES=3
    MAXLEN=40
    OUTPUTLEN=40
    model=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL:$PRIMARYMODEL
    model_types=AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM:AutoModelForCausalLM
    betas=0.2:0.4:0.4:0.0:0.0
    loss=gpt2:ngrams:ngrams:ngrams:ngrams
    label_id=0:0:0:0:0
    lossabbr="pyx:dist1:dist2:dist3:dist4"
    epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB
    min_epsilons=$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB:$CLASSLOGPROB
    epsilon_warmup_steps=0:0:0:0
    epsilon_cooldown_steps=1:1:1:1
    epsilon_decay_functions=linear:linear:linear:linear
    LAMBDALR=10.0
    selection_criterion="mrr_allsat"
    noise_variance=1.0
    embedgd_do_sample="false"
    KEYWORDTOPK=${12}
elif [[ "$option" == "nontoxic" ]]
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
elif [[ "$option" == "nontoxic-p" ]]
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
    loss=gpt2:classification_p_no_prefix
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
elif [[ "$option" == "nontoxic-0.8" ]]
then
    echo "nontoxicity > 0.8"
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/nontoxic_prompts-10k.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25
    OUTPUTLEN=20
    MAXLEN=20
    OPTIMSTEPS=150
    model=$PRIMARYMODEL:$TOXICITYCLASSIFIER
    tokenizer=$PRIMARYMODEL:$TOXICITYTOKENIZER
    model_types=AutoModelForCausalLM:GPT2CustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification
    label_id=0:0
    lossabbr="pyx:toxicity"
    epsilons=0.23
    min_epsilons=0.23
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    noise_variance=1.0
    embedgd_do_sample="false"
elif [[ "$option" == "nontoxic-linear-scale" ]]
then
    echo "nontoxicity-linear-scale"
    DATASTYLE="jsonl"
    DATAFILE=$DATA_DIR/control-prompts/nontoxic_prompts-10k.jsonl
    JSON_PKEY="prompt"
    JSON_SKEY="text"
    NUM_SAMPLES=25
    OUTPUTLEN=20
    MAXLEN=20
    OPTIMSTEPS=250
    linear_scale="true"
    selection_criterion="weighted_sum"
    model=$PRIMARYMODEL:$TOXICITYCLASSIFIER
    tokenizer=$PRIMARYMODEL:$TOXICITYTOKENIZER
    model_types=AutoModelForCausalLM:GPT2CustomForSequenceClassification
    betas=0.6:0.4
    loss=gpt2:classification
    label_id=0:0
    lossabbr="pyx:toxicity"
    epsilons=0.23
    min_epsilons=0.23
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
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
    loss=gpt2conditional:usim:wmd
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
    loss=gpt2conditional:wmd
    lossabbr="logp(y|x):wmd"
    epsilons=0.8
    min_epsilons=0.4
    epsilon_warmup_steps=20
    epsilon_cooldown_steps=0:0
    epsilon_decay_functions=step:step
elif [[ "$option" == "formality" ]]
then
    echo "formality"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/control-prompts/topic-prompts.txt
    NUM_SAMPLES=5
    model=$PRIMARYMODEL:$FORMALITYMODEL
    tokenizer=$PRIMARYMODEL:$FORMALITYMODEL
    model_types=AutoModelForCausalLM:GPT2CustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification
    label_id=0:1
    lossabbr="pyx:binary"
    epsilons=0.69
    min_epsilons=0.69
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=2.0
elif [[ "$option" == "informality" ]]
then
    echo "informality"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/control-prompts/topic-prompts.txt
    NUM_SAMPLES=5
    model=$PRIMARYMODEL:$FORMALITYMODEL
    tokenizer=$PRIMARYMODEL:$FORMALITYMODEL
    model_types=AutoModelForCausalLM:GPT2CustomForSequenceClassification
    betas=0.8:0.2
    loss=gpt2:classification
    label_id=0:0
    lossabbr="pyx:binary"
    epsilons=0.69
    min_epsilons=0.69
    epsilon_warmup_steps=0
    epsilon_cooldown_steps=1
    epsilon_decay_functions=linear
    LAMBDALR=2.0
elif [[ "$option" == "shakespeare" ]]
then
    echo "shakespeare"
    model=$PRIMARYMODEL:$SHAKESPEAREMODEL
    tokenizer=$PRIMARYMODEL:$SHAKESPEAREMODEL
    model_types=AutoModelForCausalLM:AutoModelForSequenceClassification
    betas=0.8:0.2
    loss=gpt2conditional:classification
    label_id=1
    lossabbr="pyx:binary"
    epsilons=10.0
    min_epsilons=0.69
    epsilon_warmup_steps=40
    epsilon_cooldown_steps=80
    epsilon_decay_functions=step
elif [[ "$option" == "englishes" ]]
then
    echo "englishes"
    model=$PRIMARYMODEL:$STSMODEL:$ENGLISHESMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL:$ENGLISHESMODEL
    model_types=AutoModelForCausalLM:AutoModel:AutoModelForSequenceClassification
    betas=0.8:0.1:0.1
    loss=gpt2conditional:usim:classification
    lossabbr="pyx:sts:binary_ce"
    epsilons=10.0:50.0
    label_id=1:1:1
    # epsilons=0.2:0.69
    min_epsilons=0.25:0.69
    epsilon_warmup_steps=1:1
    epsilon_cooldown_steps=200:200
    epsilon_decay_functions=step:step
    mal=10
elif [[ "$option" == "american" ]]
then
    echo "american"
    model=$PRIMARYMODEL:$STSMODEL:$ENGLISHESMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL:$ENGLISHESMODEL
    model_types=AutoModelForCausalLM:AutoModel:AutoModelForSequenceClassification
    betas=0.8:0.1:0.1
    loss=gpt2conditional:usim:classification
    lossabbr="pyx:sts:binary_ce"
    epsilons=10.0:50.0
    label_id=1:1:0
    # epsilons=0.2:0.69
    min_epsilons=0.4:0.69
    epsilon_warmup_steps=1:1
    epsilon_cooldown_steps=200:200
    epsilon_decay_functions=step:step
    mal=10
elif [[ "$option" == "usim_classify" ]]
then
    echo "usim,classify"
    model=$PRIMARYMODEL:$STSMODEL:$FORMALITYMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL:$FORMALITYMODEL
    model_types=AutoModelForCausalLM:AutoModel:AutoModelForSequenceClassification
    betas=0.8:0.1:0.1
    loss=gpt2conditional:usim:classification
    lossabbr="pyx:sts:binary_ce"
    epsilons=1.0:10.0
    min_epsilons=0.4:0.69
    epsilon_warmup_steps=1:1
    epsilon_cooldown_steps=40:40
    epsilon_decay_functions=step:step
    mal=10
elif [[ "$option" == "bart" ]]  ## no prompt just generate directly
then 
    echo "bart"
    DATASTYLE="text"
    DATAFILE=$DATA_DIR/control-prompts/topic-prompts.txt #CHANGE THIS
    OUTPUTLEN=-1
    MAXLEN=-1
    NUM_SAMPLES=1
    model=$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL
    model_types=AutoModelForSeq2SeqLM
    betas=1.0
    loss=bart
    lossabbr="logpyx"
    embedgd_do_sample="false"
    noise_variance=0.0
    epsilons=none
    label_id=none
    min_epsilons=none
    epsilon_warmup_steps=none
    epsilon_cooldown_steps=none
    epsilon_decay_functions=none
elif [[ "$option" == "bart-formal" ]]  
then 
    echo "bart-formal"
    DATASTYLE="text"
    DATAFILE="/projects/tir5/users/sachink/embed-style-transfer/data/GYAFC_Corpus/Entertainment_Music/test/informal:/projects/tir5/users/sachink/embed-style-transfer/data/GYAFC_Corpus/Entertainment_Music/test/formal.ref0"
    ADDITIONALDATAFILE="/projects/tir5/users/sachink/embed-style-transfer/data/GYAFC_Corpus/Entertainment_Music/test/informal.paraphrase"
    embedgd_do_sample="false"
    OUTPUTLEN=50
    MAXLEN=25
    NUM_SAMPLES=1
    model=$PRIMARYMODEL:$BARTFORMALITYMODEL:$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL:$BARTFORMALITYMODEL:$PRIMARYMODEL
    model_types=AutoModelForSeq2SeqLM:GPT2CustomForSequenceClassification:AutoModelForSeq2SeqLM
    betas=1.0:0.0:0.0
    loss=bart:conditional_classification:wmd
    lossabbr="logpyx:classify:wmd"
    label_id=0:1:0
    epsilons=10.0:1.0
    min_epsilons=0.22:0.4
    epsilon_warmup_steps=30:30
    epsilon_cooldown_steps=70:70
    epsilon_decay_functions=linear:linear
else
    echo "usim,wmd,classify"
    DATASTYLE="text"
    DATAFILE="/projects/tir5/users/sachink/embed-style-transfer/data/GYAFC_Corpus/Entertainment_Music/test/informal:/projects/tir5/users/sachink/embed-style-transfer/data/GYAFC_Corpus/Entertainment_Music/test/formal.ref0"
    ADDITIONALDATAFILE="/projects/tir5/users/sachink/embed-style-transfer/data/GYAFC_Corpus/Entertainment_Music/test/informal.paraphrase"
    embedgd_do_sample="false"
    OUTPUTLEN=50
    MAXLEN=-1
    model=$PRIMARYMODEL:$STSMODEL:$STSMODEL:$STYLEFORMALITYMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL:$STSMODEL:$STYLEFORMALITYMODEL
    model_types=AutoModelForCausalLM:AutoModel:AutoModel:AutoModelForSequenceClassification
    betas=0.7:0.1:0.1:0.1
    loss=gpt2conditional:usim:wmd:conditional_classification
    label_id=0:0:0:1
    lossabbr="pyx:sts:wmd:binary"
    epsilons=1.0:1.0:10.0
    min_epsilons=0.5:0.55:0.69
    epsilon_warmup_steps=30:30:30
    epsilon_cooldown_steps=70:70:70
    epsilon_decay_functions=linear:linear:linear
    mal=10
    length_diff=0:1
    OPTIMSTEPS=150
    OUTPUTSTYLE="text"
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

echo $TARGETTOPIC
echo "noise-variance=${noise_variance}"

if [[ "$debug" == "debug" ]]
then
    python -W ignore -u decode.py\
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
        --kweight 5.0\
        --lr $lr\
        --dampness 0.1\
        --epsilons $epsilons\
        --min_epsilons $min_epsilons\
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
        --random-example $RANDOMEXAMPLE\
        --early-stop-steps 10\
        --restarts $RESTARTS\
        --use_context $USECONTEXT\
        --debug
elif [[ "$debug" == "interactive" ]]
then
    echo "interactive (in debug mode)"
    python -W ignore -u decode.py\
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
        --kweight 5.0\
        --lr $lr\
        --dampness 0.1\
        --epsilons $epsilons\
        --min_epsilons $min_epsilons\
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
        --random-example $RANDOMEXAMPLE\
        --early-stop-steps 10\
        --restarts $RESTARTS\
        --use_context $USECONTEXT\
        --debug
elif [[ "$debug" == "run_and_evaluate" ]]
then
    python -W ignore -u decode.py\
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
        --kweight 5.0\
        --lr $lr\
        --dampness 0.1\
        --epsilons $epsilons\
        --min_epsilons $min_epsilons\
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
    bash examples/prompt/evaluate.sh $option $OUTFILE $EVALFILE $DATAFILE 
else
    bash examples/prompt/evaluate.sh $option $OUTFILE $EVALFILE $DATAFILE 
fi
