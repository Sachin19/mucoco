This repository contains the code for the NeurIPS 2021 paper: [Controlled Text Generation as Continuous Optimization with Multiple Constraints](https://arxiv.org/abs/2108.01850)

# Dependencies

* [pytorch](#) >= 1.6
* [transformers](https://huggingface.co/transformers/) >= 4.5.1
* (optional for some constraints) [sentence-transformers](https://github.com/UKPLab/sentence-transformers) 
* (optional for some constraints) [POT](https://pythonot.github.io/)

# Quick Start

The main file to run this decoding algorithm is `decode.py`. All models used in this code are based on huggingface transformers. 

## Machine Translation experiments

see examples

## Style Transfer experiments

see examples

# Adding new constraints

This code currently supports the following losses:

* Sentence Classification (Cross Entropy)
* Semantic Similarity (Cosine Similarity, WMD between representations)
* Conditional generation losses (MarianMT, GPT2)

To add more losses/constraints, follow examples from 'mucoco/losses/'
