DATADIR=data/sentiment/$1
mkdir -p $DATADIR

#processing the data
if [[ "$1" == "sst2" ]]
then 
    echo "download and preprocessing sst data"
    python -u data/sentiment/create_sst_sentiment_data.py $DATADIR
    dev=dev
elif [[ "$1" == "yelp" ]]
then 
    echo "download and preprocess yelp data"
    python -u data/sentiment/create_yelp_sentiment_data.py $DATADIR
    dev=test ### this dataset doesn't provide a dev dataset so I used the test for development, since the classifier is only used as a constraint not for evaluation.
else
    echo "wrong data provided. Acceptable options are sst or yelp"
    exit 1
fi 

#train the classifier
echo "training $1 classifier"
python -u examples/training_constraint_models/train_classifier.py\
    $DATADIR\
    0,1\
    train\
    $dev\
    test\
    roberta-base\
    models/$1-classifier-disc\
    gpt2-roberta full gpt2-large freeze-vecmap dontbinarize jsonl

# ## uncomment this to train gedi classifiers
# python -u examples/training_constraint_models/train_generative_classifier.py\
#     $DATADIR\
#     0,1\
#     train\
#     $dev\
#     test\
#     roberta-base\
#     models/$1-classifier-gedi\
#     gpt2-roberta full gpt2-large freeze-vecmap dontbinarize jsonl