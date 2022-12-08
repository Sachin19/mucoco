This repository contains the code for the decoding algorithm MuCoLa described in the EMNLP 2022 paper [Gradient Based Constrained Sampling from Language Models](https://arxiv.org/abs/2205.12558). If you have any questions, please feel free to reach out sachink@cs.cmu.edu

To start, create a conda environment called `mucola` and activate it.

```
conda env create -f environment.yml
```

<!-- # Dependencies

* [pytorch](#) >= 1.6
* [transformers](https://huggingface.co/transformers/) >= 4.5.1
* (optional for some constraints) [sentence-transformers](https://github.com/UKPLab/sentence-transformers) 
* (optional for some constraints) [POT](https://pythonot.github.io/) -->

# Experiments

The main file to run this decoding algorithm is `decode_new.py`. All models used in this code are based on huggingface transformers. 


## Toxicity Avoidance

1. Download the [test set](https://drive.google.com/uc?id=1bI49aJvmEoLdqSNb30JkORdsNJmv7Aep) provided by the authors of DExperts containing the prompts and place it under `data/control-prompts/nontoxic_prompts-10k.jsonl`.
2. Download the classifier training data from [here](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data) (`all_data.csv`) and place it in `data/toxicity/jigsaw-unintended-bias-in-toxicity-classification`. 
3. Preprocess and train the classifier using `bash examples/training_constraint_models/train_toxicity_classifier.sh`
4. Generate and evaluate the samples using `bash examples/prompt/toxicity-all/mucola-disc.sh <output_dir>`. This model generates 25 samples per prompt from `GPT2-Large`. To modify any hyperparameters check out `examples/prompt/toxicity-all/mucola-disc.sh` and `examples/prompt/constrained_sampling_mucola.sh`


## Sentiment 

1. Download and process SST-2 data from as `bash examples/training_constraint_models/train_sentiment_classifier.sh sst2`
2. (Optional if you want to use two discriminators) Download and process Yelp training data from as `bash examples/training_constraint_models/train_sentiment_classifier.sh yelp`. You can train GeDi like classifiers by switching out `train_classifier.py` inside `examples/training_constraint_models/train_sentiment_classifier.sh` for `train_generative_classifier.py`.
3. Generate and evaluate the samples using `bash examples/prompt/sentiment-all/mucola-disc.sh <output_dir>`. This model generates 20 samples per prompt from `GPT2-Large`. Check out other files under `examples/prompt/sentiment-all` for different experiments. To modify any hyperparameters check out `examples/prompt/constrained_sampling_mucola.sh`.


## Keywords

Generate translations with hard constraints that specific named entities should appear in the output using `bash examples/MT/mucola-keyphrases.sh`.


See `examples/prompt/constrained_sampling_mucola.sh`, `examples/summarization/constrained_summarize_mucola.sh` for additional experiments. 


# License 

The source code is licensed under the MIT license, which you can find in the LICENSE.md file

