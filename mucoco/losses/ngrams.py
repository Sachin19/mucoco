from mucoco.losses import BaseLoss, register_loss

import torch 
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import logging
import os

try:
    import ot
except:
    ot = None

@register_loss("ngrams")
class KeywordLoss(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 
        
        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device
        self.topk = args.keyword_topk 

        self.eos_token_id = self.tokenizer.eos_token_id    
    
    def compute_loss(self, batch, preds, **kwargs):
        '''
        batch: a tuple (source, prefix). If giving a prompt to the decoder, it can be specified using "prefix"
        preds: a tuple containing (predicted tokens, predicted embeddings, predicted probabilities), this is obtained through a forward pass on the optimizable target parameters (See utils/target.py)
        '''
        if len(batch) == 2:
            source, prefix = batch
        else:
            prefix = batch
            
        pred_tokens, pred_embeds, pred_probs = preds
        embed_lut = self.model.get_input_embeddings()

        step = kwargs.get("step", -1)
        
        keyword_logits = -torch.square(torch.cdist(pred_embeds, embed_lut.weight.unsqueeze(0)))
        keyword_probs_all = F.softmax(keyword_logits, dim=-1)
        output_length = keyword_probs_all.size(1)
        losses = []
        for keyword_tokenized in self.keywords_tokenized:
            keyword_probs = keyword_probs_all.index_select(dim=-1, index=keyword_tokenized)
            ngram_length = keyword_tokenized.size(0)
            ngram_probs = [keyword_probs[:, :output_length-ngram_length+1, 0]] #first token of the ngram
            for i in range(1, ngram_length):
                ngram_probs.append(keyword_probs[:, i:output_length-ngram_length + i + 1, i])

            ngram_probs = torch.stack(ngram_probs, dim=2)
            ngram_nll = -torch.log(ngram_probs).mean(dim=-1)
            
            tau=0.1
            ngram_nll_q = F.gumbel_softmax(-ngram_nll/tau)
            loss = ngram_nll_q * ngram_nll
            loss = loss.sum(dim=-1)

            losses.append(loss)
        
        loss = torch.stack(losses, dim=0).min(dim=0)[0]
        logging_output = {
            "loss": loss.data.cpu(),
        }

        return loss, logging_output

    def compute_gold_loss(self, batch, **kwargs):
        '''
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        _, target = batch
        embed_lut = self.model.get_input_embeddings()
        pred_embeds = embed_lut(target)

        self.keyword = kwargs.get("keyword").strip()
        # all_combos = set([self.keyword, self.keyword.capitalize(), " "+self.keyword, " "+self.keyword.capitalize()])
        all_combos = [" "+self.keyword]
        # all_combos = [self.keyword]
        self.keywords_tokenized = []
        for word in all_combos:
            keyword_tokenized = self.tokenizer.encode(word)
            self.keywords_tokenized.append(torch.LongTensor(keyword_tokenized).to(self.device))
            # print(torch.log(F.softmax(-torch.square(torch.cdist(embed_lut(self.keywords_tokenized[-1]), embed_lut.weight)), dim=-1)).topk(k=2, dim=-1))
            # print(self.keyword, keyword_tokenized)
        # input()

        keyword_logits = -torch.square(torch.cdist(pred_embeds, embed_lut.weight.unsqueeze(0)))

        losses = []
        for keyword_tokenized in self.keywords_tokenized:
            keyword_probs = F.softmax(keyword_logits, dim=-1).index_select(dim=-1, index=keyword_tokenized)
            ngram_length = keyword_tokenized.size(0)
            output_length = keyword_probs.size(1)

            ngram_probs = [keyword_probs[:, :output_length-ngram_length+1, 0]] #first token of the ngram
            for i in range(1, ngram_length):
                ngram_probs.append(keyword_probs[:, i:output_length-ngram_length + i + 1, i])

            ngram_probs = torch.stack(ngram_probs, dim=2)
            # print(-torch.log(ngram_probs))
            ngram_nll = -torch.log(ngram_probs).mean(dim=-1)
            # ngram_logprobs = torch.log(ngram_probs).mean(dim=-1)
            
            # g = (ngram_logprobs).topk(dim=-1, k=self.topk)
            # loss = -g[0].mean(dim=-1)
            # print(loss.size())
            tau = 0.1
            # tau=1.0
            # print(ngram_nll.size())
            # ngram_nll_q = F.gumbel_softmax(-ngram_nll, tau=tau)
            ngram_nll_q = F.gumbel_softmax(-ngram_nll/tau)
            # print(ngram_nll_q)
            
            # print(ngram_nll_q)
            loss = ngram_nll_q * ngram_nll
            # print(ngram_nll)
            # print(loss)
            loss = loss.sum(dim=-1)

            losses.append(loss)
        
        loss = torch.stack(losses, dim=0).min(dim=0)[0]
        #     topic_probs = F.softmax(topic_logits, dim=-1).index_select(dim=-1, index=keyword_tokenized)
        #     topic_probs = topic_probs.sum(dim=-1)

        #     g = topic_probs.topk(dim=-1, k=self.topk)
        #     loss = -torch.log(g[0])
        #     losses.append(loss.mean(dim=-1))
        
        # losses = torch.stack(losses, dim=0)
        # print(losses.size())

        # loss = losses.min(dim=0)[0]

        # print(losses, loss)
        # input()
        logging_output = {
            "loss": loss.data.cpu(),
        }
        return loss, logging_output   
