
MODELNAME=Helsinki-NLP/opus-mt-en-de
OUTPUTDIR=outputs/keyphrases/MT
mkdir -p $OUTPUTDIR

EPSILON=none #the epsilon is not a hyperparameter but computed automatically in this case

# note: most arguments passed to constrained_sampling_mucola.sh are not used 

# Number of constraints have to be prespeficied. The MT test set has 1, 2 or 3 keyphrase requirements. So I divide the test file into three parts and decode them separately. 
# This code first decodes a string with beam-search (say of length L). If the constraints are satisfied, it stops. If not, it samples outputs of length L, L+2, L+4 .... L+20, and selects the one which satisfies constraints and has the lowest length averaged NLL. We do this because we do not know output length in advance.
# The final output is also post processed to remove any trailing <pad> tokens.

# 1-keyword
bash examples/MT/constrained_translate_mucola.sh translate-keyword-1 $OUTPUTDIR $MODELNAME run_and_evaluate 0.1 dotplusplus l2 0.0 constant plain 0 2 0.0 false 1 0.02 0.02 50 zeros 0 1.0 0 0.2 20 0 false false 0.4 0.01 0 20 false 0 -1 SST2

# 2-keyword
bash examples/MT/constrained_translate_mucola.sh translate-keyword-2 $OUTPUTDIR $MODELNAME run_and_evaluate 0.1 dotplusplus l2 0.0 constant plain 0 2 0.0 false 1 0.02 0.02 50 zeros 0 1.0 0 0.2 20 0 false false 0.4 0.01 0 20 false 0 -1 SST2

# 3-keyword
bash examples/MT/constrained_translate_mucola.sh translate-keyword-3 $OUTPUTDIR $MODELNAME run_and_evaluate 0.1 dotplusplus l2 0.0 constant plain 0 2 0.0 false 1 0.02 0.02 50 zeros 0 1.0 0 0.2 20 0 false false 0.4 0.01 0 20 false 0 -1 SST2
