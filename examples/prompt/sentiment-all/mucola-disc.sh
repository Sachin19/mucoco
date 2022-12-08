# LENGTH=$1
# OUTPUTDIRSUFFIX=$2
LENGTH=20
# datevar=$(date +%d-%m-%Y)
MODELNAME=gpt2-large
OUTPUTDIR=outputs/sentiment/mucola-disc
mkdir -p $OUTPUTDIR

BEGINNOISE=5.0
ENDNOISE=0.05
NOISESTEPS=150

POSITIVELABEL=1
NEGATIVELABEL=0

# EPSILONS=(-5 -4 -3 -2 -1 0)
EPSILONS=(-2) # the constraint is  log(negative_prob) - log (positive_prob) < epsilon, where positive_prob is the desired label

for EPSILON in "${EPSILONS[@]}"
do
    bash examples/prompt/constrained_sampling_mucola.sh sentiment-disc $OUTPUTDIR $MODELNAME run_and_evaluate 0.25 dotplusplus l2 0.0 constant legal 0 1 0.0 false 1 0.5 0.05 50 target 100 1.0 $POSITIVELABEL 2 50 $EPSILON false false 0.45 0.01 0 true
    #note: most arguments passed to constrained_sampling_mucola.sh are not used anymore
done











# LENGTH=$1
# OUTPUTDIRSUFFIX=$3
# datevar=$(date +%d-%m-%Y)
# BEGINNOISE=5.0
# ENDNOISE=0.05
# NOISESTEPS=100

# kind=$4

# POSITIVELABEL=1
# NEGATIVELABEL=0

# EPSILON=$2
# sbatch sbatch.sh examples/prompt/constrained_sampling_new.sh sentiment-tiny sentiment-tiny/$datevar-positive-tiny-disc-sst2-$LENGTH-$OUTPUTDIRSUFFIX gpt2-large run_and_evaluate 0.25 $kind l2 0.0 constant plain 0 2 0.0 false 1 $BEGINNOISE $ENDNOISE $NOISESTEPS random_vocab 20 1.0 $POSITIVELABEL 2 20 $EPSILON true false 0.45 0.01 0 $LENGTH false 0 -1 SST2
# sbatch sbatch.sh examples/prompt/constrained_sampling_new.sh sentiment-tiny sentiment-tiny/$datevar-negative-tiny-disc-sst2-$LENGTH-$OUTPUTDIRSUFFIX gpt2-large run_and_evaluate 0.25 $kind l2 0.0 constant plain 0 2 0.0 false 1 $BEGINNOISE $ENDNOISE $NOISESTEPS random_vocab 20 1.0 $NEGATIVELABEL 2 20 $EPSILON true false 0.45 0.01 0 $LENGTH false 0 -1 SST2

##############
# sbatch sbatch.sh examples/prompt/constrained_sampling_new.sh sentiment-tiny sentiment-tiny/$datevar-positive-tiny-disc-damp1.0-sst2-$LENGTH-$OUTPUTDIRSUFFIX gpt2-large run_and_evaluate 0.25 dotplusplus l2 0.0 constant plain 0 2 0.0 false 1 $BEGINNOISE $ENDNOISE $NOISESTEPS target 20 1.0 $POSITIVELABEL 2 50 $EPSILON true false 0.45 0.01 0 $LENGTH false 0 -1 SST2

# EPSILON=0.105
# sbatch sbatch.sh examples/prompt/constrained_sampling_new.sh sentiment-tiny-logp sentiment-tiny/$datevar-positive-tiny-disc-logp-sst2-$LENGTH-$OUTPUTDIRSUFFIX gpt2-large run_and_evaluate 0.25 dotplusplus l2 0.0 constant plain 0 2 0.0 false 1 $BEGINNOISE $ENDNOISE $NOISESTEPS target 20 1.0 $POSITIVELABEL 2 50 $EPSILON true false 0.45 0.01 0 $LENGTH false 0 -1 SST2
# sbatch sbatch.sh examples/prompt/constrained_sampling_new.sh sentiment-tiny-logp sentiment-tiny/$datevar-negative-tiny-disc-logp-damp1.0-sst2-$LENGTH-$OUTPUTDIRSUFFIX gpt2-large run_and_evaluate 0.25 dotplusplus l2 0.0 constant plain 0 2 0.0 false 1 $BEGINNOISE $ENDNOISE $NOISESTEPS target 20 1.0 $NEGATIVELABEL 2 50 $EPSILON true false 0.45 0.01 0 $LENGTH false 0 -1 SST2 1.0
# sbatch sbatch.sh examples/prompt/constrained_sampling_new.sh sentiment-tiny-logp sentiment-tiny/$datevar-positive-tiny-disc-logp-damp1.0-sst2-$LENGTH-$OUTPUTDIRSUFFIX gpt2-large run_and_evaluate 0.25 dotplusplus l2 0.0 constant plain 0 2 0.0 false 1 $BEGINNOISE $ENDNOISE $NOISESTEPS target 20 1.0 $POSITIVELABEL 2 50 $EPSILON true false 0.45 0.01 0 $LENGTH false 0 -1 SST2 1.0
##############

# sbatch sbatch.sh examples/prompt/constrained_sampling_new.sh unique-sentiment-tiny sentiment-tiny/$datevar-positive-tiny-disc-yelp-$LENGTH-$OUTPUTDIRSUFFIX gpt2-large run_and_evaluate 0.25 dotplusplus l2 0.0 constant plain 0 2 0.0 false 1 $BEGINNOISE $ENDNOISE $NOISESTEPS target 20 1.0 $POSITIVELABEL 2 50 $EPSILON false false 0.45 0.01 0 $LENGTH false 0 -1 YELP 
# sbatch sbatch.sh examples/prompt/constrained_sampling_new.sh unique-sentiment-tiny sentiment-tiny/$datevar-negative-tiny-disc-yelp-$LENGTH-$OUTPUTDIRSUFFIX gpt2-large run_and_evaluate 0.25 dotplusplus l2 0.0 constant plain 0 2 0.0 false 1 $BEGINNOISE $ENDNOISE $NOISESTEPS target 20 1.0 $NEGATIVELABEL 2 50 $EPSILON false false 0.45 0.01 0 $LENGTH false 0 -1 YELP


# bash examples/prompt/constrained_sampling_new.sh conditional test gpt2-large debug 0.25 dotplusplus l2 0.0 constant plain 0 2 0.0 false 1 0.5 0.05 100 random_vocab 20 1.0 1 2 20 -5 true false 0.45 0.01 0 20 false 0 -1 SST2