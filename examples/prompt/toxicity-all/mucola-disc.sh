
# LENGTH=$1
# OUTPUTDIRSUFFIX=$2
LENGTH=20
# datevar=$(date +%d-%m-%Y)
MODELNAME=gpt2-large
OUTPUTDIR=/dir/to/save/outputs

BEGINNOISE=2.0
ENDNOISE=0.02
NOISESTEPS=150

POSITIVELABEL=1
NEGATIVELABEL=0

# EPSILONS=(-5 -4 -3 -2 -1 0)
EPSILONS=(-5)

for EPSILON in "${EPSILONS[@]}"
do
    # sbatch sbatch.sh examples/prompt/constrained_sampling_new.sh nontoxic nontoxic/$datevar-nontoxic-$EPSILON $MODELNAME run_and_evaluate 0.25 dotplusplus l2 0.0 constant legal 0 1 0.0 false 1 0.5 0.05 50 target 100 1.0 constant 2 50 $EPSILON false false 0.45 0.01 0 true
    bash examples/prompt/constrained_sampling_mucola.sh nontoxic $OUTPUTDIR $MODELNAME run_and_evaluate 0.25 dotplusplus l2 0.0 constant legal 0 1 0.0 false 1 0.5 0.05 50 target 100 1.0 constant 2 50 $EPSILON false false 0.45 0.01 0 true
    #note: most arguments passed to constrained_sampling_mucola.sh are not used anymore
done

