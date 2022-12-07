#download the data from https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data (it requires creating an account on Kaggle)
echo "make sure you have downloaded the data from https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data (all_data.csv) and placed it in data/toxicity/jigsaw-unintended-bias-in-toxicity-classification"

DATADIR=data/toxicity/jigsaw-unintended-bias-in-toxicity-classification
#processing the data
echo "preprocessing the data"
python -u data/toxicity/create_jigsaw_toxicity_data.py 

N=$(wc -l ${DATADIR}/toxicity_eq0.jsonl | cut -d ' ' -f1)
n=$(wc -l ${DATADIR}/toxicity_gte0.5.jsonl | cut -d ' ' -f1)
python -u data/toxicity/random_sample.py ${DATADIR}/toxicity_eq0.jsonl ${DATADIR}/data/toxicity/toxicity_eq0_subsample.jsonl $N $n


TEST=2000
DEV=2000
TRAIN=$(($n-$DEV-$TEST))
python -u split_train_dev_test.py ${DATADIR}/data/toxicity/toxicity_eq0_subsample.jsonl $TRAIN $DEV $TEST

#train the classifier
python -u examples/training_constraint_models/train_classifier.py\
    data/toxicity/jigsaw-unintended-bias-in-toxicity-classification\
    0,1\
    train\
    dev\
    test\
    roberta-base\
    models/roberta-base-jigsaw-toxicity-classifier-with-gpt2-large-embeds\
    gpt2-roberta full gpt2-large freeze-vecmap dontbinarize jsonl