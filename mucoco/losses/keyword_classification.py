from mucoco.losses import BaseLoss, register_loss

import torch 
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

import logging
import os


@register_loss("keywordclassification")
class KeyWordClassification(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 
        
        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device
        self.topic_target = args.topic_target
        self.topk = args.keyword_topk

        self.topic2words = {}
        if args.topic_word_lists is not None:   
            vocabsize = self.model.get_input_embeddings().num_embeddings
            for filename in os.listdir(args.topic_word_lists):
                topicname = filename.split("/")[-1].split(".")[0]
                # wordlist = torch.zeros((vocabsize,)).to(self.device)
                allwords = []
                for word in open(args.topic_word_lists+"/"+filename):
                    wordtok = self.tokenizer.encode(" "+word.strip())
                    allwords += wordtok
                    # for tokid in wordtok:
                        # wordlist[tokid] = 1           
                allwords = list(set(allwords))
                self.topic2words[topicname] = torch.LongTensor(allwords).to(self.device) #wordlist#/wordlist.sum(dim=-1)
                print(topicname, len(allwords))# wordlist.sum(dim=-1))
        
        self.epsilon_additive = 0#-np.log(self.topic2words[self.topic_target].sum().item())
        print(topicname, self.epsilon_additive)

    
    def compute_loss(self, batch, preds, **kwargs):
        '''
        batch: a tuple (source, prefix). If giving a prompt to the decoder, it can be specified using "prefix"
        preds: a tuple containing (predicted tokens, predicted embeddings, predicted probabilities), this is obtained through a forward pass on the optimizable target parameters (See utils/target.py)
        '''
        topic = self.topic_target
        topk = self.topk
        step = kwargs.get("step", -1)
        if len(batch) == 2:
            source, prefix = batch
        else:
            prefix = batch

        pred_tokens, pred_embeds, pred_probs = preds
        batch_size = pred_embeds.size(0)
        embed_lut = self.model.get_input_embeddings()

        # predlen = pred_embeds.size(1)
        # if step == 0:
        #     block_size = 10
        #     pyramid = [1 for i in range(min(block_size, predlen))] + [0 for i in range(max(0, predlen-block_size))]
        #     # reverse_pyramid = [1 for i in range(min(block_size, predlen))] + [0 for i in range(max(0, predlen-block_size))]
        #     # reverse_pyramid.reverse()
        #     reverse_pyramid = [1 for i in range(predlen)]
        #     self.masks = torch.Tensor(reverse_pyramid + pyramid).unsqueeze(0).to(self.device)

        #     coeff_steps = 200
        #     self.freq = max(1, coeff_steps // predlen)
        
        # idx = max(0, predlen - 1 - step // self.freq)
        # masks = self.masks[:, idx:idx + predlen]

        # print(idx, masks)
        # input_embeds = torch.cat([embed_lut(source), embed_lut(prefix), pred_embeds], dim=1)
        # hidden_states = self.model(inputs_embeds=input_embeds, step=step, go_inside="transformer")[0]
        
        topicwords = self.topic2words[topic]
        # topic_logits = pred_embeds.matmul(embed_lut.weight.t())
        # topic_logits = torch.square(pred_embeds.unsqueeze(2) - embed_lut.weight.unsqueeze(0).unsqueeze(1)).sum(dim=-1)
        logits = -torch.square(torch.cdist(pred_embeds, embed_lut.weight.unsqueeze(0), p=2))
        topic_logits = logits.index_select(dim=-1, index=topicwords)
        # topic_probs = (F.softmax(topic_logits, dim=-1) * topicwords)
        topic_probs = F.softmax(topic_logits, dim=-1)
        # print(topic_probs.size())
        
        topic_probs = topic_probs.sum(dim=-1)# * masks
        # print(topic_probs)
        # print(topic_probs.topk(dim=-1, k=self.topk))
        # loss = -torch.log(topic_probs.topk(dim=-1, k=self.topk)[0]) 
        # loss = -torch.log(topic_probs)
        # loss = loss.mean(dim=-1)

        topic_nll = -torch.log(topic_probs)
        # ngram_logprobs = torch.log(ngram_probs).mean(dim=-1)
        
        # g = (ngram_logprobs).topk(dim=-1, k=self.topk)
        # loss = -g[0].mean(dim=-1)
        # print(loss.size())
        # tau = 0.00001
        tau=1.0
        topic_nll_q = F.gumbel_softmax(-topic_nll/tau)
        
        # print(ngram_nll_q)
        loss = topic_nll_q * topic_nll
        loss = loss.sum(dim=-1)
        
        logging_output = {
            "loss": loss.data.cpu(),
        }

        return loss, logging_output

    def compute_gold_loss(self, batch, **kwargs):
        '''
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        source, target = batch

        batch_size=target.size(0)
    
        topic = self.topic_target
        #input_tokens = torch.cat([bos, prefix, pred_tokens, eos], dim=1)
        embed_lut = self.model.get_input_embeddings()
        
        # print(self.topic2words.keys())
        topicwords = self.topic2words[topic]
        return torch.Tensor([0.])
        # print(topic)

        # topic_logits = embed_lut(target).matmul(embed_lut.weight.t())
        # # print(topic_logits.size())
        # topic_probs = (F.softmax(topic_logits, dim=-1) * topicwords).sum(dim=-1)

        # # print(topic_probs)
        # print(topic_probs)
        # print(topic_probs.topk(dim=-1, k=self.topk))
        # loss = -torch.log(topic_probs.topk(dim=-1, k=self.topk)[0]) 
        # loss = loss.mean(dim=-1)
        # loss = -torch.log(topic_probs.max(dim=-1)[0]) #max pooling, try others
        # loss = -torch.log(topic_probs).sum(dim=-1) #max pooling, try others
        # print(loss)
        # loss = loss.sum()
        # print(loss)
    
        logging_output = {
            "loss": loss.data.cpu(),
        }
        return loss, logging_output   
