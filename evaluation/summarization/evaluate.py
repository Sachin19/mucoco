import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import click
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, TextClassificationPipeline

import argparse
import json
import os
import operator

from functools import partial
from collections import Counter
from scipy import stats
from multiprocessing.pool import Pool

import random

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sacrerouge.metrics import Rouge

import torch.nn as nn
import torch

import logging

logger = logging.getLogger(__name__)



def bleu_i(weights, all_sentences, smoothing_function, i):
    # noinspection PyTypeChecker
    return sentence_bleu(
        references=all_sentences[:i] + all_sentences[i + 1:],
        hypothesis=all_sentences[i],
        weights=weights,
        smoothing_function=smoothing_function)

def fluency_classify(generations_df, batch_size=32):
    from fairseq.models.roberta import RobertaModel
    from fairseq.data.data_utils import collate_tokens

    model = RobertaModel.from_pretrained(
            '/projects/tir5/users/sachink/embed-style-transfer/evaluation_models/cola_classifier_fluency/',
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path='./cola-bin'
        )
    model.cuda()

    def label_fn(label):
        return model.task.label_dictionary.string(
            [label + model.task.target_dictionary.nspecial]
        )
    
    def predict_batch(batch):
        batch = collate_tokens([model.task.source_dictionary.encode_line("<s> " + sd + " </s>", append_eos=False) for sd in batch], pad_idx=1)
        batch = batch[:, :512]

        with torch.no_grad():
            predictions = model.predict('sentence_classification_head', batch.long())
            # prediction_probs = [torch.exp(x).max(axis=0)[0].item() for x in predictions]
            prediction_labels = [label_fn(x.argmax(axis=0).item()) for x in predictions]
        
        return prediction_labels
            
    batch = []
    all_prediction_labels = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating CoLA fluency'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        for j, gen in enumerate(generations):
            batch.append(model.bpe.encode(f'{prompt}{gen}'))
            if len(batch) == batch_size:
                prediction_labels = predict_batch(batch)
                all_prediction_labels += prediction_labels
                batch = []
        
        if len(batch) != 0:
            prediction_labels = predict_batch(batch)
            all_prediction_labels += prediction_labels
            batch = []

    accuracy = np.array(all_prediction_labels) == "acceptable"
    accuracy = np.nanmean(accuracy.astype("float32"))
    return accuracy

def allsat_accuracy(generations_df):
    accuracies = []
    sat_once = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation allsats'):
        allsats = [float(gen['allsat']) for gen in row['generations']]
        
        sat_proportion = sum(allsats)/len(allsats)

        accuracies.append(sat_proportion)
        sat_once.append(float(sat_proportion > 0))

    print(np.nanmean(sat_once))
    return np.nanmean(accuracies), np.std(accuracies), np.mean(sat_once)


def distinctness(generations_df):
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating dist-n'):
        generations = [gen['text'] for gen in row['generations']]
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(' ')
            # o = [str(tok) for tok in gen]
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i+1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)
    
    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)

def distinctness2(generations_df): #not over samples but averaged over individual outputs
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating dist-n'):
        generations = [gen['text'] for gen in row['generations']]
        for gen in generations:
            unigrams, bigrams, trigrams = set(), set(), set()
            total_words = 0
            o = gen.split(' ')
            # o = [str(tok) for tok in gen]
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i+1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
            dist1.append(len(unigrams) / total_words)
            dist2.append(len(bigrams) / total_words)
            dist3.append(len(trigrams) / total_words)
    
    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)

def self_bleu(generations_df, n_sample=1000):

    # import spacy
    random.seed(0)
    # nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))

    smoothing_function = SmoothingFunction().method1
    all_sentences = []
    for i, row in generations_df.iterrows():
        gens = [gen['tokens'] for gen in row['generations']]
        all_sentences += gens
    
    pool = Pool(processes=os.cpu_count())
    bleu_scores = []
    for n_gram in range(1, 6):

        if n_gram == 1:
            weights = (1.0, 0, 0, 0)
        elif n_gram == 2:
            weights = (0.5, 0.5, 0, 0)
        elif n_gram == 3:
            weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
        elif n_gram == 4:
            weights = (0.25, 0.25, 0.25, 0.25)
        elif n_gram == 5:
            weights = (0.2, 0.2, 0.2, 0.2, 0.2)
        else:
            raise ValueError
        bleu_scores.append(
            list(tqdm(
                pool.imap_unordered(
                    partial(bleu_i, weights, all_sentences, smoothing_function),
                    random.sample(range(len(all_sentences)), min(n_sample, len(all_sentences)))),
                total=min(n_sample, len(all_sentences)),
                smoothing=0.0,
                desc=f"bleu-{n_gram}")))
        # print(f"\n\nbleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / n_sample}")
    
    pool.close()
    pool.join()

    bleus = []
    for n_gram in range(5):
        bleus.append(sum(bleu_scores[n_gram]) / n_sample)
        # print(f"bleu-{n_gram + 1} = {sum(bleu_scores[n_gram]) / n_sample}")
    
    return bleus

    # if args.logto:
    #     with open(args.logto, 'a') as fout:
    #         print(f"{os.path.basename(args.file)}", end='\t', file=fout)
    #         for n_gram in range(5):
    #             print(f"{sum(bleu_scores[n_gram]) / n_sample}", end='\t', file=fout)
    #         print(file=fout)

def self_bleu2(generations_df, n_sample=100):

    # import spacy
    random.seed(0)
    smoothing_function = SmoothingFunction().method1
    # nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))
    all_bleus = [[] for _ in range(3)]
    for i, row in generations_df.iterrows():
        # all_sentences = []
        all_sentences = [gen['tokens'] for gen in row['generations']]
        # all_sentences += gens
        
        pool = Pool(processes=os.cpu_count())
        bleu_scores = []
        for i in range(3):
            n_gram = i+3
            if n_gram == 1:
                weights = (1.0, 0, 0, 0)
            elif n_gram == 2:
                weights = (0.5, 0.5, 0, 0)
            elif n_gram == 3:
                weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
            elif n_gram == 4:
                weights = (0.25, 0.25, 0.25, 0.25)
            elif n_gram == 5:
                weights = (0.2, 0.2, 0.2, 0.2, 0.2)
            else:
                raise ValueError
            bleu_scores.append(
                list(tqdm(
                    pool.imap_unordered(
                        partial(bleu_i, weights, all_sentences, smoothing_function),
                        random.sample(range(len(all_sentences)), min(n_sample, len(all_sentences)))),
                    total=min(n_sample, len(all_sentences)),
                    smoothing=0.0,
                    desc=f"bleu-{n_gram}")))
            # print(f"\n\nbleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / n_sample}")
        
        pool.close()
        pool.join()

        for i in range(3):
            all_bleus[i].append(sum(bleu_scores[i]) / n_sample)
            # print(f"bleu-{n_gram + 1} = {sum(bleu_scores[n_gram]) / n_sample}")
    all_bleus = [np.nanmean(bleu) for bleu in all_bleus]
    return all_bleus

    # if args.logto:
    #     with open(args.logto, 'a') as fout:
    #         print(f"{os.path.basename(args.file)}", end='\t', file=fout)
    #         for n_gram in range(5):
    #             print(f"{sum(bleu_scores[n_gram]) / n_sample}", end='\t', file=fout)
    #         print(file=fout)

def repetition(generations_df, tokenizer, numbers_only=True, rep_file=None):
    SEP = tokenizer.encode(tokenizer.bos_token)[0]

    objs = []
    max_n = 90

    n_repeated_examples = 0
    total_examples = 0

    if rep_file is not None:
        fout = open(rep_file, "w")
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating repetitions'):
        generations = [gen['tokens'] for gen in row['generations']]
        for gen in generations:
            total_examples += 1
            if gen[-1] == SEP:
                gen.pop(-1)
            rev_gen = list(reversed(gen))
            last_n_repeats = [0] * max_n

            for n in range(1, max_n + 1):
                n_repeat = 1
                while len(rev_gen[n*n_repeat:n*(n_repeat+1)]) == n and \
                        rev_gen[n*n_repeat:n*(n_repeat+1)] == rev_gen[:n]:
                    n_repeat += 1
                last_n_repeats[n - 1] = n_repeat
            max_repeated_n = max(range(max_n), key=lambda x: last_n_repeats[x])

            if last_n_repeats[max_repeated_n] > 1 and (max_repeated_n+1 >= 3 or last_n_repeats[max_repeated_n] > 50):
                repetition = {
                    'repeated_phrase': list(reversed(rev_gen[:max_repeated_n + 1])),
                    'repeated_times': last_n_repeats[max_repeated_n],
                    'repeated_phrase_length': max_repeated_n + 1,
                }
                n_repeated_examples += 1
            else:
                repetition = {}
            
            if rep_file is not None:
                json.dump(repetition, fout)
                fout.write("\n")
    
    if rep_file is not None:
        fout.close()

    return n_repeated_examples*1.0/total_examples

    # if not numbers_only:
    #     print("filename\tnumber of repeating examples")
    #     print(f"{os.path.basename(args.file)}\t{n_repeated_examples}")
    # if args.output:
    #     output_filename = os.path.join(os.path.dirname(args.file), "repetition_" + os.path.basename(args.file))
    #     with open(output_filename, 'w+') as fout:
    #         for obj in objs:
    #             print(json.dumps(obj), file=fout)

def HUSE(generations_df):
    pass
    ##need human evaluation for this

@click.command()
@click.option('--generations_file', required=True, type=str, help='a jsonl file with generations and attribute scores')
@click.option('--output_file', required=True, type=str, help='filename to write outputs')
@click.option('--metrics', required=True, type=str, help='which metrics to compute, write comma separeted, ppl-own,ppl-big,cola,self-bleu,zipf,repetition,dist-n,sentiment')
def main(generations_file, output_file, metrics, topic):
    assert os.path.exists(generations_file)
    output_dir = Path(os.path.dirname(generations_file))
    generations_df = pd.read_json(generations_file, lines=True)
    
    metrics = set(metrics.strip().lower().split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### calculate quality metrics
    # Fluency
    fo = open(output_dir / output_file, 'w') #just creating the file

    if "rouge" in metrics:
        rouge = compute_rouge(generations_file):
    
    # self-bleu
    if "self-bleu2" in metrics:
        bleus = self_bleu2(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            for i, bleu in enumerate(bleus):
                fo.write(f'bleu-{i+3} = {bleu}\n')
                print(f'bleu-{i+3} = {bleu}')

    # repetition
    if "repetition" in metrics:
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
        rep_rate = repetition(generations_df, eval_tokenizer, rep_file=output_dir / (output_file+".repetitions"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'repetition_rate: {rep_rate}')
            print(f'repetition_rate: {rep_rate}')

    if "allsat" in metrics:
        print("allsat")
        sat_accuracy, sat_std, sat_once = allsat_accuracy(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean allsat accuracy, std = {sat_accuracy}--{sat_std}, {sat_once}\n')
            print(f'mean allsat accuracy, std = {sat_accuracy}--{sat_std}, {sat_once}')

    fo.close()
    # HUSE: TODO


if __name__ == '__main__':
    main()