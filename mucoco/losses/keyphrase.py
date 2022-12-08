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

def squared_cdist(x, E):
    # |x_i - y_j|_2^2 = <x_i - y_j, x_i - y_j> = <x_i, x_i> + <y_j, y_j> - 2*<x_i, y_j>
    x_sq_norm = x.square().sum(dim=-1, keepdim=True)
    y_sq_norm = E.t().square().sum(dim=0, keepdim=True)
    x_dot_y = x.matmul(E.t())
    sq_dist = (x_sq_norm + y_sq_norm - 2*x_dot_y).clamp_(min=0.0)
    # print(sq_dist.size())
    return sq_dist
    # For numerical issues
    # sq_dist.clamp_(min=0.0)
    # return torch.sqrt(sq_dist)

@register_loss("ngramsl22")
class KeywordL2Loss(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 
        
        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device
        self.topk = args.keyword_topk 
        self.tau = args.keyword_tau

        self.keyword = "------------------------"
        self.keywords_embeds = []
        self.eos_token_id = self.tokenizer.eos_token_id 
        self.gumbel_update_step = 1
        self.eye_filter = torch.eye(10, 10).view(1, 1, 10, 10).to(self.device).detach()   
    
    def compute_loss(self, batch, preds, **kwargs):
        '''
        batch: a tuple (source, prefix). If giving a prompt to the decoder, it can be specified using "prefix"
        preds: a tuple containing (predicted tokens, predicted embeddings, predicted probabilities), this is obtained through a forward pass on the optimizable target parameters (See utils/target.py)
        '''
        if len(batch) == 2:
            source, prefix = batch
        else:
            prefix = batch
        step = kwargs.get("step")
            
        pred_tokens, pred_embeds, pred_probs = preds
        embed_lut = self.model.get_input_embeddings()
        # output_length = pred_embeds.size(1)       

        ndist = -torch.square(torch.cdist(embed_lut.weight.unsqueeze(0), pred_embeds))
        alllogprobs = F.log_softmax(ndist, dim=1)
        losses = []
        for kid, keyword_id in enumerate(self.keywords_ids):
            logprobs = alllogprobs.index_select(dim=1, index=keyword_id)
            ngram_length = keyword_id.size(0)
            # print(logprobs[:, :, 5:5+ngram_length])
            # print(logprobs[:, 5:5 +ngram_length].sum()/ngram_length)
            try:
                logprobs = F.conv2d(logprobs.unsqueeze(1), self.eye_filter[:, :, :ngram_length, :ngram_length])/ngram_length
                logprobs = logprobs.squeeze(2).squeeze(1)

                #print(pred_tokens)
                if step % self.gumbel_update_step == 0:
                    self.logprobs_q = F.gumbel_softmax(logprobs/self.tau, hard=True, dim=-1)#.detach()
                #print(self.logprobs_q)
                # print(logprobs)
                loss = -(self.logprobs_q * logprobs).sum(dim=-1)
                # print(loss)
                # input()
                losses.append(loss)
            except:
                print(ngram_length, logprobs.size())
                raise ValueError("something happened with conv")
        # print(losses)
        losses = torch.stack(losses, dim=1)
        losses_q = F.gumbel_softmax(-losses/self.tau, hard=True, dim=-1)#.detach()

        loss = (losses_q * losses).sum(dim=-1)

        
        logging_output = {
            "loss": loss.data.cpu(),
        }

        return loss, logging_output
    
    def process_keyword(self, keyword):
        if self.keyword == keyword:
            return
        
        else:
            self.keyword = keyword
            for kt in self.keywords_embeds:
                del kt # releasing old keywords from GPU?

            all_combos = [self.keyword]
            self.keywords_embeds = []
            self.keywords_ids = []

            delta = 0.1
            embed_lut = self.model.get_input_embeddings()
            
            all_combos = [" "+k.strip() for k in self.keyword.split("#")]
            all_othercombos = []# ["("+k.strip() for k in self.keyword.split("#")]
            # all_combos = [k.strip() if k[0].isupper() else " "+k.strip() for k in self.keyword.split("#")]# [" " + self.keyword, self.keyword]
            # all_combos = sum(all_combos, [])fs
            # print(all_combos)

            self.epsilon = 0
            for word in all_combos: #all combos is meaningless at the moment, there is only one combo
                if self.args.target_tokenize_different:
                    with self.tokenizer.as_target_tokenizer():
                        self.keywords_ids.append(self.tokenizer.encode(word, add_special_tokens=False, return_tensors="pt").to(self.device)[0])
                else:
                    self.keywords_ids.append(self.tokenizer.encode(word, add_special_tokens=False, return_tensors="pt").to(self.device)[0])
                # print(word, self.keywords_ids[-1])
                keyword_embeds = embed_lut(self.keywords_ids[-1]).detach()
               
                    # print(self.tokenizer.encode(word, add_special_tokens=False, return_tensors="pt"))
                    # keyword_ndist = -squared_cdist(keyword_embeds, self.model.get_input_embeddings().weight)
                    # ngram_length = keyword_embeds.size(0)
                    # self.keywords_embeds.append(keyword_embeds)

                ndist = -torch.square(torch.cdist(keyword_embeds, embed_lut.weight))
                    # print(self.keywords_ids)
                    # print(F.log_softmax(ndist, dim=-1).topk(k=10, dim=-1))
                ndist = F.log_softmax(ndist, dim=-1)#.index_select(dim=-1, index=self.keywords_ids[-1])
                    # print(ndist.size())
                    # self.epsilon = (-ndist.mean(dim=0).topk(k=2, dim=-1)[0][..., 1]).item() - delta #delta added to ensure strict inequality not less than or equal to
                    # print(ndist.size())
                # print(ndist.max(dim=-1)[0])
                self.epsilon = max(self.epsilon, -ndist.max(dim=-1)[0].mean(dim=0).item() + delta)
                # print(word, self.keywords_ids[-1], self.epsilon)
                # input()
                
                    # print(ndist.max(dim=-1), self.keywords_ids)
                    # print(ndist.mean(dim=-1).topk(k=10, dim=0)[0] - delta)
                    # self.epsilon = -self.epsilon
                    # print(self.epsilon)
            # input()

            for word in all_othercombos: #all combos is meaningless at the moment, there is only one combo
                if self.args.target_tokenize_different:
                    with self.tokenizer.as_target_tokenizer():
                        self.keywords_ids.append(self.tokenizer.encode(word, add_special_tokens=False, return_tensors="pt").to(self.device)[0][1:])
                else:
                    self.keywords_ids.append(self.tokenizer.encode(word, add_special_tokens=False, return_tensors="pt").to(self.device)[0][1:])
                # print(word, self.keywords_ids[-1])
                keyword_embeds = embed_lut(self.keywords_ids[-1]).detach()
               
                    # print(self.tokenizer.encode(word, add_special_tokens=False, return_tensors="pt"))
                    # keyword_ndist = -squared_cdist(keyword_embeds, self.model.get_input_embeddings().weight)
                    # ngram_length = keyword_embeds.size(0)
                    # self.keywords_embeds.append(keyword_embeds)

                ndist = -torch.square(torch.cdist(keyword_embeds, embed_lut.weight))
                    # print(self.keywords_ids)
                    # print(F.log_softmax(ndist, dim=-1).topk(k=10, dim=-1))
                ndist = F.log_softmax(ndist, dim=-1)#.index_select(dim=-1, index=self.keywords_ids[-1])
                    # print(ndist.size())
                    # self.epsilon = (-ndist.mean(dim=0).topk(k=2, dim=-1)[0][..., 1]).item() - delta #delta added to ensure strict inequality not less than or equal to
                    # print(ndist.size())
                # print(ndist.max(dim=-1)[0])
                self.epsilon = max(self.epsilon, -ndist.max(dim=-1)[0].mean(dim=0).item() + delta)
            print(self.keywords_ids)

    def compute_gold_loss(self, batch, **kwargs):
        '''
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        with torch.no_grad():
            _, target = batch
            # embed_lut = self.model.get_input_embeddings()
            pred_embeds = self.model.get_input_embeddings()(target)
            # output_length = pred_embeds.size(1)
            
            self.process_keyword(kwargs.get("keyword"))
            
            embed_lut = self.model.get_input_embeddings()
            ndist = -torch.square(torch.cdist(embed_lut.weight.unsqueeze(0), pred_embeds))

            losses = []
            for kid, keyword_id in enumerate(self.keywords_ids):
                logprobs = F.log_softmax(ndist, dim=1).index_select(dim=1, index=keyword_id)
                ngram_length = keyword_id.size(0)
                try:
                    logprobs = F.conv2d(logprobs.unsqueeze(1), self.eye_filter[:, :, :ngram_length, :ngram_length])/ngram_length
                    logprobs = logprobs.squeeze(2).squeeze(1)

                    logprobs_q = F.gumbel_softmax(logprobs/self.tau, hard=True, dim=-1)#.detach()
                    loss = -(logprobs_q * logprobs).sum(dim=-1)

                    losses.append(loss)
                except:
                    print(ngram_length, logprobs.size())
                    raise ValueError("something happened with conv")
            # print(losses)
            losses = torch.stack(losses, dim=1)
            losses_q = F.gumbel_softmax(-losses/self.tau, hard=True, dim=-1)#.detach()

            loss = (losses_q * losses).sum(dim=-1)
            
            logging_output = {
                "loss": loss.data.cpu(),
            }
            return loss, logging_output   
