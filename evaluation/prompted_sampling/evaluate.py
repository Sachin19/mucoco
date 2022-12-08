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

from perspective_api import PerspectiveWorker, unpack_scores

from transformers import GPT2PreTrainedModel, GPT2Model
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, SequenceClassifierOutput

from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

import torch.nn as nn
import torch

import logging

logger = logging.getLogger(__name__)

class GPT2CustomForSequenceClassification(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        print(config.vocab_size)
        self.transformer = GPT2Model(config)

        embeds = self.transformer.get_input_embeddings()
        old_dim = getattr(config,'n_embd', embeds.embedding_dim)
        new_dim = getattr(config,'new_n_embd', None)
        if new_dim is not None:
            new_embeds = nn.Sequential(nn.Embedding(embeds.num_embeddings, new_dim), nn.Linear(new_dim, old_dim))
            self.transformer.set_input_embeddings(new_embeds)

        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[range(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
        
class RobertaCustomForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        print(config.vocab_size)


        self.roberta = RobertaModel(config, add_pooling_layer=False)
        embeds = self.roberta.get_input_embeddings()
        old_dim = getattr(config,'n_embd', embeds.embedding_dim)
        new_dim = getattr(config,'new_n_embd', None)
        new_vocab_size = getattr(config,'new_vocab_size', config.vocab_size)
        if new_dim is not None:
            new_embeds = nn.Sequential(nn.Embedding(new_vocab_size, new_dim), nn.Linear(new_dim, old_dim, bias=False))
            self.roberta.set_input_embeddings(new_embeds)

        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def bleu_i(weights, all_sentences, smoothing_function, i):
    # noinspection PyTypeChecker
    return sentence_bleu(
        references=all_sentences[:i] + all_sentences[i + 1:],
        hypothesis=all_sentences[i],
        weights=weights,
        smoothing_function=smoothing_function)

def conditional_perplexity(generations_df, model, tokenizer, device='cuda', write_file=None):
    perplexities = []
    goodperplexities = []
    total_nll = 0
    total_tokens = 0
    g = 0
    ct = 0
    if write_file is not None:
        fout = open(write_file, "w")

    # for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating PPL'):
        # prompt_input_ids = torch.LongTensor([row.prompt['tokens']]).to(device)
        prompt = row.prompt['text']
        prompt_input_ids = tokenizer.encode(row.prompt['text'], return_tensors='pt').to(device)
        if not (prompt_input_ids.shape[1] == 1 and prompt_input_ids[0].tolist()[0] == tokenizer.bos_token_id): # this means unconditional, prompt is BOS token (verify)
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
            # print("in")
        else:
            prompt_loss = 0
            # print("out")
        # for every generation conditioned on the prompt
        generations = [gen['text'] for gen in row['generations']]
        # for gen_ids in generations:
        for gen in generations:

            # full_input_ids = torch.LongTensor([row.prompt['tokens'] + gen_ids]).to(device)
            full_input_ids = tokenizer.encode(f'{prompt}{gen}', return_tensors='pt').to(device)
            # print(f'{prompt}{gen}')
            # print(full_input_ids)
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
            loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])

            ppl = np.exp(loss.item())
            # print(ppl)
            # input()
            if ppl < 100:   # for sanity
                goodperplexities.append(ppl)
                # perplexities.append(ppl)
                g += 1

            # if ppl < 1e4:
            perplexities.append(ppl)
            # else:
                # print("ppl values are weirldly large. Check for errors")

            total_nll += (full_loss - prompt_loss).item()
            total_tokens += (full_input_ids.shape[1] - prompt_input_ids.shape[1])
            # print(full_input_ids[0], prompt_input_ids[0])
            # print(full_loss, prompt_loss)
            # input()
            if write_file is not None:
                fout.write(f"{ppl}, {(full_loss - prompt_loss).item()}, {(full_input_ids.shape[1] - prompt_input_ids.shape[1])}\n")
        
        # input("ok")
    
    print(np.nanmean(goodperplexities), len(goodperplexities), len(perplexities), g)
    return np.nanmean(perplexities), np.exp(total_nll/total_tokens)

def perplexity(generations_df, model, tokenizer, device='cuda', write_file=None):
    #TODO spearman correlation between human ppl and model ppl, not needed anymore, check degen ppl calculation, it's different from this.
    total_nll = 0
    total_tokens = 0
    g = 0
    ct = 0
    if write_file is not None:
        fout = open(write_file, "w")

    # for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating total PPL'):
        prompt = row.prompt['text']
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        if not (prompt_input_ids.shape[1] == 1 and prompt_input_ids[0].tolist()[0] == tokenizer.bos_token_id): # this means unconditional, prompt is BOS token (verify)
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
        else:
            prompt_loss = 0
        # for every generation conditioned on the prompt
        generations = [gen['text'] for gen in row['generations']]
        for gen in generations:
            full_input_ids = tokenizer.encode(f'{prompt}{gen}', return_tensors='pt').to(device)
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
            total_nll += (full_loss - prompt_loss).item()
            total_tokens += (full_input_ids.shape[1] - prompt_input_ids.shape[1])
            
            if write_file is not None:
                fout.write(f"{total_nll} {total_tokens}\n")
        
    return np.exp(total_nll/total_tokens)

def fluency_classify(generations_df, output_file, batch_size=32):
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
    
    with open(output_file, "w") as fout:
        fout.write("\n".join(all_prediction_labels))

    accuracy = np.array(all_prediction_labels) == "acceptable"
    accuracy = np.nanmean(accuracy.astype("float32"))
    return accuracy


def morpho_syntactic(generations_df, morpho_file=None):
    pass

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

def keyword_count_coverage(generations_df):
    counts = []
    bestcounts = []
    l = 0
    s = 0
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation allsats'):
        # print(input_file.readline())
        # keywords = json.loads(input_file.readline())['concept_set'].split("#")
        keywords = json.loads(row['keywords'])['concept_set'].split("#")
        print(keywords)
        l += 1
        maxcount = 0
        onesat = False
        print(row['generations'])
        for gen in row['generations']:
            s += 1
            count = 0
            # print(keywords)
            for keyword in keywords:
                if keyword in gen['text']:
                    count += 1
            counts.append(count)
            maxcount = max(maxcount, count)
            onesat = onesat or gen['allsat']
            if gen['allsat'] and count < 4:
                print(keywords)
                print(maxcount, gen['allsat'])
                print(gen['text'])
                # input()
        # print(maxcount, onesat)
        # input()
        bestcounts.append(maxcount)

    print(np.sum(bestcounts)/l, np.sum(counts)/s, l, s)
    return np.sum(bestcounts)/l, np.sum(counts)/s

def sentiment_classify(generations_df, sentiment_file=None):

    # score generations and write to sentiment.jsonl
    classifier = pipeline('sentiment-analysis', device=0)
    # classifier = pipeline(model='siebert/sentiment-roberta-large-english')
    print("writing outputs to ", str(sentiment_file))
    if sentiment_file is not None:
        fo = open(sentiment_file, 'w')
    from pprint import pprint
    accuracies = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation sentiments'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        sentences_for_prompt= []
        for gen in generations:
            sentences_for_prompt.append(f'{prompt}{gen}')
        
        positive_proportion = 0
        try:
            predictions_for_prompt = classifier(sentences_for_prompt)
        except IndexError: # sometimes the generation is too long?
            print("exception occured, please check")
            predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
        # print(predictions_for_prompt)
        for prediction in predictions_for_prompt:
            positive_proportion += float(prediction["label"] == "POSITIVE")
        positive_proportion = positive_proportion / len(predictions_for_prompt)
        # print(positive_proportion)
        accuracies.append(positive_proportion)
        # input()
        # print(predictions_for_prompt)
        if sentiment_file is not None:
            for res in predictions_for_prompt:  
                fo.write(json.dumps(res) + '\n')
        
    return np.nanmean(accuracies), np.std(accuracies)

def sentiment_classify_promptonly(generations_df, sentiment_file=None):

    # score generations and write to sentiment.jsonl
    classifier = pipeline('sentiment-analysis', device=0)
    # classifier = pipeline(model='siebert/sentiment-roberta-large-english')
    print("writing outputs to ", str(sentiment_file))
    if sentiment_file is not None:
        fo = open(sentiment_file, 'w')
    from pprint import pprint
    accuracies = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation sentiments'):
        prompt = row.prompt['text']
        # generations = [gen['text'] for gen in row['generations']]
        # sentences_for_prompt= []
        num_neutral = 0
        sentences_for_prompt = [f'{prompt}']
        # for gen in generations:
        #     sentences_for_prompt.append(f'{prompt}')
        
        positive_proportion = 0
        try:
            predictions_for_prompt = classifier(sentences_for_prompt)
        except IndexError: # sometimes the generation is too long?
            print("exception occured, please check")
            predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
        # print(predictions_for_prompt)
        for prediction in predictions_for_prompt:
            positive_proportion += int(prediction["label"] == "POSITIVE")
        if positive_proportion in [12,13]:
            num_neutral += 1
        positive_proportion = (1.0*positive_proportion) / len(predictions_for_prompt)
        # print(positive_proportion)
        accuracies.append(positive_proportion)
        # input()
        # print(predictions_for_prompt)
        if sentiment_file is not None:
            for res in predictions_for_prompt:  
                fo.write(json.dumps(res) + '\n')
        
    return np.nanmean(accuracies), np.std(accuracies), num_neutral

def sentiment_classify_big(generations_df, sentiment_file=None):

    # score generations and write to sentiment.jsonl
    print("lalala")
    classifier = pipeline(model='siebert/sentiment-roberta-large-english', device=0)
    # classifier.cuda()
    print("lalala2")
    # classifier = pipeline(model='siebert/sentiment-roberta-large-english')
    print("writing outputs to ", str(sentiment_file))
    if sentiment_file is not None:
        fo = open(sentiment_file, 'w')
    
    accuracies = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation sentiments'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        sentences_for_prompt= []
        for gen in generations:
            sentences_for_prompt.append(f'{prompt}{gen}')
            
        # print(sentences_for_prompt)
        

        positive_proportion = 0
        try:
            predictions_for_prompt = classifier(sentences_for_prompt)
        except IndexError: # sometimes the generation is too long?
            print("exception occured, please check")
            predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
        # print(predictions_for_prompt)
        for prediction in predictions_for_prompt:
            positive_proportion += float(prediction["label"] == "POSITIVE")
        positive_proportion = positive_proportion / len(predictions_for_prompt)
        # print(positive_proportion)
        accuracies.append(positive_proportion)
        # input()
        # print(predictions_for_prompt)
        if sentiment_file is not None:
            for res in predictions_for_prompt:  
                fo.write(json.dumps(res) + '\n')
        
    return np.nanmean(accuracies), np.std(accuracies)

def sentiment_classify_yelp(generations_df, sentiment_file=None):

    # score generations and write to sentiment.jsonl
    print("jajaja")
    classifier = pipeline(model='textattack/bert-base-uncased-yelp-polarity', device=0)
    # classifier.cuda()
    print("jajaja2")
    # classifier = pipeline(model='siebert/sentiment-roberta-large-english')
    print("writing outputs to ", str(sentiment_file))
    if sentiment_file is not None:
        fo = open(sentiment_file, 'w')
    
    accuracies = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation sentiments'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        sentences_for_prompt= []
        for gen in generations:
            sentences_for_prompt.append(f'{prompt}{gen}')
            
        # print(sentences_for_prompt)
        

        positive_proportion = 0
        try:
            predictions_for_prompt = classifier(sentences_for_prompt)
        except IndexError: # sometimes the generation is too long?
            print("exception occured, please check")
            predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
        # print(predictions_for_prompt)
        for prediction in predictions_for_prompt:
            positive_proportion += float(prediction["label"] == "LABEL_1")
        positive_proportion = positive_proportion / len(predictions_for_prompt)
        # print(positive_proportion)
        accuracies.append(positive_proportion)
        # input()
        # print(predictions_for_prompt)
        if sentiment_file is not None:
            for res in predictions_for_prompt:  
                fo.write(json.dumps(res) + '\n')
        
    return np.nanmean(accuracies), np.std(accuracies)

def sentiment_classify_own(generations_df, sentiment_file=None):

    # score generations and write to sentiment.jsonl
    # classifier = pipeline('sentiment-analysis')
    # model_path="/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-sentiment-binary-classifier/checkpoint_best"
    model_path = "/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-textattack-sst-2-with-gpt2-large-embeds/checkpoint_best"
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    classifier_model = GPT2CustomForSequenceClassification.from_pretrained(model_path, config=config)
    classifier = TextClassificationPipeline(task="text-classification", model=classifier_model, tokenizer=tokenizer, device=0)
    print("writing outputs to ", str(sentiment_file))
    if sentiment_file is not None:
        fo = open(sentiment_file, 'w')
    
    accuracies = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation sentiments'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        sentences_for_prompt= []
        for gen in generations:
            sentences_for_prompt.append(f'{prompt}{gen}')
        # print(sentences_for_prompt)
        

        positive_proportion = 0
        try:
            predictions_for_prompt = classifier(sentences_for_prompt)
        except IndexError: # sometimes the generation is too long?
            print("exception occured, please check")
            predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
        # print(predictions_for_prompt)
        for prediction in predictions_for_prompt:
            positive_proportion += float(prediction["label"] == "LABEL_1")
        positive_proportion = positive_proportion / len(predictions_for_prompt)
        # print(positive_proportion)
        accuracies.append(positive_proportion)
        # input()
        # print(predictions_for_prompt)
        if sentiment_file is not None:
            for res in predictions_for_prompt: 

                fo.write(json.dumps(res) + '\n')
        
    return np.nanmean(accuracies), np.std(accuracies)

def sentiment_classify_own2(generations_df, sentiment_file=None):

    # score generations and write to sentiment.jsonl
    # classifier = pipeline('sentiment-analysis')
    # model_path="/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-sst-2-with-gpt2-large-embeds/checkpoint_best"
    # model_path="/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-textattack-sst-2-with-gpt2-large-embeds-proper/checkpoint_best"
    model_path="/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-textattack-uncased-sst-2-with-gpt2-large-embeds-proper/checkpoint_best"
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    classifier_model = RobertaCustomForSequenceClassification.from_pretrained(model_path, config=config)
    classifier = TextClassificationPipeline(task="text-classification", model=classifier_model, tokenizer=tokenizer)
    print("writing outputs to ", str(sentiment_file))
    if sentiment_file is not None:
        fo = open(sentiment_file, 'w')
    
    accuracies = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation sentiments'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        sentences_for_prompt= []
        for gen in generations:
            sentences_for_prompt.append(f'{prompt}{gen}')
        # print(sentences_for_prompt)
        

        positive_proportion = 0
        try:
            predictions_for_prompt = classifier(sentences_for_prompt)
        except IndexError: # sometimes the generation is too long?
            print("exception occured, please check")
            predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
        # print(predictions_for_prompt)
        for prediction in predictions_for_prompt:
            positive_proportion += float(prediction["label"] == "LABEL_1")
        positive_proportion = positive_proportion / len(predictions_for_prompt)
        # print(positive_proportion)
        accuracies.append(positive_proportion)
        # input()
        # print(predictions_for_prompt)
        if sentiment_file is not None:
            for res in predictions_for_prompt: 

                fo.write(json.dumps(res) + '\n')
        
    return np.nanmean(accuracies), np.std(accuracies)

def politeness_classify(generations_df, politeness_file=None):
    model_path="/projects/tir5/users/sachink/embed-style-transfer/models/roberta-base-3class-politeness-with-gpt2-large-embeds/checkpoint_best/"
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    classifier_model = RobertaCustomForSequenceClassification.from_pretrained(model_path, config=config)
    classifier = TextClassificationPipeline(task="text-classification", model=classifier_model, tokenizer=tokenizer)
    print("writing outputs to ", str(politeness_file))
    if politeness_file is not None:
        fo = open(politeness_file, 'w')
    
    accuracies = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation politeness'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        sentences_for_prompt= []
        for gen in generations:
            sentences_for_prompt.append(f'{prompt}{gen}')
        # print(sentences_for_prompt)
        

        positive_proportion = 0
        try:
            predictions_for_prompt = classifier(sentences_for_prompt)
        except IndexError: # sometimes the generation is too long?
            print("exception occured, please check")
            predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
        # print(predictions_for_prompt)
        for prediction in predictions_for_prompt:
            positive_proportion += float(prediction["label"] == "LABEL_2")
        positive_proportion = positive_proportion / len(predictions_for_prompt)
        # print(positive_proportion)
        accuracies.append(positive_proportion)
        # input()
        # print(predictions_for_prompt)
        if politeness_file is not None:
            for res in predictions_for_prompt: 

                fo.write(json.dumps(res) + '\n')
        
    return np.nanmean(accuracies), np.std(accuracies), 0
        
def toxicity_score(generations_df, perspective_file, perspective_rate_limit=5):
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
    import time

    API_KEY="none"
    assert API_KEY != "none", "Please set the API_KEY before proceeding"

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    total_failed = 0
    avg_max = []
    toxic_probability = []

    # if not os.path.exists(perspective_file):
    fout = open(perspective_file, "w")
    # else:
        # raise ValueError("perspective file path already exists")

    last_request_time = -1
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating toxicity via perspective API'):
        prompt = row.prompt['text']
        # generations = [gen['text'] if len(gen['text']) <= 3000 else gen['text'][:3000] for gen in row['generations'] ] 
        allresponses = []
        # for genid in range(len(row['generations'])):
        for genid in range(1):
            # generations = [gen['text'] for gen in row['generations'][genid:genid+1]]  
            generations = [gen['text'] for gen in row['generations']]  
            # print(sum([len(gen) for gen in generations]))
            generations = [gen if gen != "" else "\n" for gen in generations]

            # for j, gen in enumerate(generations):
            #     not_done = True
            #     while not_done:
            #         try:
            #             analyze_request= {
            #                 'comment': {'text': gen},
            #                 'requestedAttributes': {"TOXICITY":{}},
            #                 'spanAnnotations': True,
            #                 "languages": ["en"],
            #             }
            #             response = client.comments().analyze(body=analyze_request).execute()
            #             print(response)
            #             time.sleep(1.0)
            #             not_done=False
            #         except Exception as e:
            #             print(j, str(e))
                        # time.sleep(1.0)


            responses = {f'gen-{i}-{genid}-{j}': None for j in range(len(generations))}
            
            not_done = np.array([1.0/len(generations) for gen in generations])
            def response_callback(request_id, response, exception):
                responses[request_id] = (response, exception)
                if exception is None:
                    not_done[int(request_id.split("-")[-1])] = 0
                if exception is not None:
                    not_done[int(request_id.split("-")[-1])] = 1
                    print(request_id, exception)
            
            while not_done.sum() > 0:
                try:
                    time_since_last_request = time.time() - last_request_time

                    if time_since_last_request < 1:
                        time.sleep(1-time_since_last_request)

                    if not_done.sum() > 1:
                        print(i, "extra_sleep", not_done.sum(), flush=True)
                        time.sleep(1.0)

                    batch_request = client.new_batch_http_request()
                    for j, text in enumerate(generations):
                        analyze_request= {
                            'comment': {'text': text},
                            'requestedAttributes': {"TOXICITY":{}},
                            'spanAnnotations': True,
                            "languages": ["en"],
                        }
                        batch_request.add(client.comments().analyze(body=analyze_request), callback=response_callback, request_id=f"gen-{i}-{genid}-{j}")
                    # print(batch_request)
                    # input()
                    batch_request.execute()
                    last_request_time = time.time()
                    # print(responses)
                    # time.sleep(1.0)
                except Exception as e:
                    print(e)
                    print("sleeping for 60 sec and retrying")
                    time.sleep(60.0)
            allresponses.append(responses)

        json.dump({"allresponses": responses}, fout)
        fout.write("\n")
        max_prob = 0.0
        toxicity_proportion = 0
        for responses in allresponses: 
            # print(responses)
            for req_id, (response, exception) in responses.items():    
                prob = response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']
                max_prob = max(max_prob, prob)
                toxicity_proportion += int(prob > 0.5)
                # response[]
                # x = input("break")
        
        avg_max.append(max_prob)
        toxic_probability.append(int(toxicity_proportion >= 1))

        # if x == "y":
        #     break

    print(avg_max, toxic_probability)
    print(np.nanmean(avg_max), sum(toxic_probability)/len(toxic_probability))

    return (np.nanmean(avg_max), sum(toxic_probability)/len(toxic_probability))

def toxicity_score_old(generations_df, perspective_file, perspective_rate_limit=5):
    all_generations = []
    prompts = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Scoring generation sentiments'):
        prompts.append(row.prompt['text'])
        all_generations += [gen['text'] for gen in row['generations']]

    assert len(all_generations) % len(prompts) == 0
    n = len(all_generations) / len(prompts)
        
    perspective = PerspectiveWorker(
        out_file=perspective_file,
        total=len(generations_df.index) * len(generations_df.iloc[0]['generations']),
        rate_limit=perspective_rate_limit
    )

    # Generate and collate perspective scores
    generations = []
    for i, gen in enumerate(all_generations):
        perspective(f'generation-{i}', gen)

    perspective.stop()
    print('Finished generation and perspective scoring! check', perspective_file)


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

def zipf_coefficient(generations_df, N=5000):
    cnt = Counter()
    
    for i, row in generations_df.iterrows():
        generations = [gen['tokens'] for gen in row['generations']]
        for gen in generations:
            cnt.update(gen)

    xs = np.arange(1, min(len(cnt), N)+1)
    ys = np.array(sorted(cnt.values(), key=operator.neg)[:N])
    s, b, r, p, std = stats.linregress(np.log(xs), np.log(ys))
    return -s, -r, p

def topic_eval(generations_df, category, cap=None):
    # num matches of distinct words
    tw_dir="/projects/tir5/users/sachink/embed-style-transfer/related-work/naacl-2021-fudge-controlled-generation/topic_data/test_wordlists"
    import string
    words = []
    with open(os.path.join(tw_dir, category), 'r') as rf:
        for line in rf:
            words.append(line.strip().lower())
    num_match = 0
    num_unit_match = 0
    c = 0
    for i, row in generations_df.iterrows():
        generations = [gen['text'] for gen in row['generations']]
        for sent in generations:
            c += 1
            sent_match = 0
            sent = sent.strip().lower().split()
            sent = [tok.strip(string.punctuation) for tok in sent]
            for word in words:
                if word in sent:
                    sent_match += 1
            if cap is None:
                num_match += sent_match
            else:
                num_match += min(cap, sent_match)
            num_unit_match += min(1, sent_match)
    return num_match, num_unit_match, c

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
@click.option('--extra', required=False, type=str, help='extra params like which topic category or keyword file')
def main(generations_file, output_file, metrics, extra):
    assert os.path.exists(generations_file)
    output_dir = Path(os.path.dirname(generations_file))
    if generations_file.endswith(".jsonl"):
        generations_df = pd.read_json(generations_file, lines=True)
    else:
        with open(generations_file) as fin:
            generations_df = [{'prompt':{'text':''}, 'generations':[{'text':l.strip()}]} for l in fin.readlines()]
            generations_df = pd.DataFrame(generations_df)
    
    metricset = set(metrics.strip().lower().split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### calculate quality metrics
    # Fluency
    fo = open(output_dir / output_file, 'w') #just creating the file
    fo.close()
    if "pplcustom" in metrics:
        print("ppl")
        allmetrics = metrics.split(",")
        for metric in allmetrics:
            if "pplcustom" in metric:
                eval_modelname = metric.split("#")[1]
                print(eval_modelname)
                eval_model = AutoModelForCausalLM.from_pretrained(eval_modelname).to(device)
                eval_tokenizer = AutoTokenizer.from_pretrained(eval_modelname)
                torch.cuda.empty_cache()
                with torch.no_grad():
                    ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-"+eval_modelname.replace("/", "-")))

                # write output results
                with open(output_dir / output_file, 'a') as fo:
                    fo.write(f'{eval_modelname} perplexity, {eval_modelname} total perplexity = {ppl}, {total_ppl}\n')
                    print(f'{eval_modelname} perplexity, {eval_modelname} total perplexity = {ppl}, {total_ppl}\n')

    if "ppl-big" in metricset: #GPT2-XL
        print("big")
        
        eval_model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-big"))

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'gpt2-xl perplexity, gpt2-xl total perplexity = {ppl}, {total_ppl}\n')
            print(f'gpt2-xl perplexity, gpt2-xl total perplexity = {ppl}, {total_ppl}\n')

    
    if "ppl-own" in metricset: #GPT2-Large
        print("own")
        eval_model = AutoModelForCausalLM.from_pretrained('gpt2-large').to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-own"))

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'gpt2-large perplexity, gpt2-large total perplexity = {ppl}, {total_ppl}\n')
            print(f'gpt2-large perplexity, gpt2-large total perplexity = {ppl}, {total_ppl}\n')
        
    if "ppl-small" in metricset: #GPT2
        print("small")
        eval_model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl, total_ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device, write_file=output_dir / (output_file+".ppl-own"))

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')
            print(f'gpt2 perplexity, gpt2 total perplexity = {ppl}, {total_ppl}\n')
    
    if 'sentiment' in metricset:
        print("sentiment-big") #c3
        sentiment_accuracy, sentiment_std = sentiment_classify_big(generations_df, sentiment_file=output_dir / (output_file+".sentiment-big"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean sentiment-big accuracy = {sentiment_accuracy}, {sentiment_std}\n')
            print(f'mean sentiment-big accuracy = {sentiment_accuracy}, {sentiment_std}')

        print("sentiment") #c1
        sentiment_accuracy, sentiment_std = sentiment_classify(generations_df, sentiment_file=output_dir / (output_file+".sentiment"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}\n')
            print(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}')

        print("sentiment-yelp") #c2
        sentiment_accuracy, sentiment_std = sentiment_classify_yelp(generations_df, sentiment_file=output_dir / (output_file+".sentiment-yelp"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean sentiment-yelp accuracy = {sentiment_accuracy}, {sentiment_std}\n')
            print(f'mean sentiment-yelp accuracy = {sentiment_accuracy}, {sentiment_std}')

        print("sentiment-own") #internal classifier
        sentiment_accuracy, sentiment_std = sentiment_classify_own2(generations_df, sentiment_file=output_dir / (output_file+".sentiment-own"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean sentiment-own accuracy = {sentiment_accuracy}, {sentiment_std}\n')
            print(f'mean sentiment-own accuracy = {sentiment_accuracy}, {sentiment_std}')
    
    if 'sentiment_promptonly' in metricset:
        print("sentiment")
        sentiment_accuracy, sentiment_std, num_neutral = sentiment_classify_promptonly(generations_df, sentiment_file=output_dir / (output_file+".sentiment"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}, {num_neutral}\n')
            print(f'mean sentiment accuracy = {sentiment_accuracy}, {sentiment_std}, {num_neutral}')
    
    if 'toxicity' in metricset:
        print("toxicity")
        (avg_max, toxic_probability) = toxicity_score(generations_df, perspective_file=output_dir / (output_file+".toxicity"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'avg_max = {avg_max}, toxicity prob={toxic_probability}\n')
            print(f'avg_max = {avg_max}, toxicity prob={toxic_probability}\n')
    
    #cola
    if "cola" in metricset:
        cola_accuracy = fluency_classify(generations_df, output_file=output_dir / (output_file+".cola"))
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'cola acceptability accuracy = {cola_accuracy}\n')
            print(cola_accuracy)

    ### calculate diversity
    # dist-n
    if "dist-n" in metricset:
        dist1, dist2, dist3 = distinctness(generations_df)
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            for i, dist_n in enumerate([dist1, dist2, dist3]):
                fo.write(f'dist-{i+1} = {dist_n}\n')
                print(f'dist-{i+1} = {dist_n}')

    # self-bleu
    if "self-bleu" in metricset:
        bleus = self_bleu(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            for i, bleu in enumerate(bleus):
                fo.write(f'bleu-{i+1} = {bleu}\n')
                print(f'bleu-{i+1} = {bleu}')
    
    if "dist-n2" in metricset:
        dist1, dist2, dist3 = distinctness2(generations_df)
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            for i, dist_n in enumerate([dist1, dist2, dist3]):
                fo.write(f'dist-{i+1} = {dist_n}\n')
                print(f'dist-{i+1} = {dist_n}')

    # self-bleu
    if "self-bleu2" in metricset:
        bleus = self_bleu2(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            for i, bleu in enumerate(bleus):
                fo.write(f'bleu-{i+3} = {bleu}\n')
                print(f'bleu-{i+3} = {bleu}')

    # zipf-coefficient
    if "zipf" in metricset:
        s, r, p = zipf_coefficient(generations_df)
        
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'zipf: s={s}, r={r}, p={p}\n')
            print(f'zipf: s={s}, r={r}, p={p}')

    # repetition
    if "repetition" in metricset:
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
        rep_rate = repetition(generations_df, eval_tokenizer, rep_file=output_dir / (output_file+".repetitions"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'repetition_rate: {rep_rate}')
            print(f'repetition_rate: {rep_rate}')

    if "allsat" in metricset:
        print("allsat")
        sat_accuracy, sat_std, sat_once = allsat_accuracy(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean allsat accuracy, std = {sat_accuracy}--{sat_std}, {sat_once}\n')
            print(f'mean allsat accuracy, std = {sat_accuracy}--{sat_std}, {sat_once}')
    
    if "keywordcount" in metricset:
        print("keywordcount")
        print(extra)
        bestcount, allcount = keyword_count_coverage(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean allsat accuracy, std = {bestcount}--{allcount}\n')
            print(f'mean allsat accuracy, std = {bestcount}--{allcount}')
    
    if "politeness" in metricset:
        print("politeness")
        polit_accuracy, politeness_std, politeness_once = politeness_classify(generations_df, politeness_file=output_dir / (output_file+".politeness"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'mean politeness accuracy, std = {polit_accuracy}--{politeness_std}, {politeness_once}\n')
            print(f'mean politeness accuracy, std = {polit_accuracy}--{politeness_std}, {politeness_once}')
    
    if "topic" in metricset:
        print(f"topic -- {extra}")
        num_match, num_unit_match, total_sent_count = topic_eval(generations_df, extra, None)

        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'num match, num unit match, total sent count: {num_match}--{num_unit_match}, {total_sent_count}\n')
            print(f'num match, num unit match, total sent count: {num_match}--{num_unit_match}, {total_sent_count}')

    # HUSE: TODO

if __name__ == '__main__':
    main()