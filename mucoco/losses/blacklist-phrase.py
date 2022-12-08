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

@register_loss("blacklistl22")
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

        dist = -torch.square(torch.cdist(embed_lut.weight.unsqueeze(0), pred_embeds))
        # print(dist.size())
        dist = F.log_softmax(dist, dim=1).index_select(dim=1, index=self.keywords_ids[0])
        # print(dist.size())
        # print(dist)
        # print(F.log_softmax(-torch.square(torch.cdist(pred_embeds, embed_lut.weight.unsqueeze(0))),dim=-1).index_select(dim=-1, index=self.keywords_ids[0]))
        try:
            ngram_nll = F.conv2d(dist.unsqueeze(1), self.eye_filter[:, :, :self.ngram_length, :self.ngram_length])/self.ngram_length
            ngram_nll = ngram_nll.squeeze(2).squeeze(1)
        except:
            print(self.ngram_length, dist.size())

        # print(ngram_nll.size())
        

        # tau=0.01
        # if step % self.gumbel_update_step == 0:
        #     self.ngram_nll_q = F.gumbel_softmax(-ngram_nll/self.tau, dim=-1).detach()
        # ngram_nll_q = self.ngram_nll_q
        

        # loss = ngram_nll_q * ngram_nll
        loss = ngram_nll.max(dim=-1)[0]
        # print(ngram_nll)
        # print(pred_tokens)
        # print("ok")
        # loss = -loss

        logging_output = {
            "loss": loss.data.cpu(),
        }

        return loss, logging_output
    
    def process_keyword(self, keyword):
        # self.keyword = kwargs.get("keyword").strip()
        # print("ok", self.keyword)
        # print("process", self.keyword, keyword, "ok")
        if self.keyword == keyword:
            # print("hein")
            return
        
        else:
            self.keyword = keyword
            # print("ok2",self.keyword)
            for kt in self.keywords_embeds:
                del kt # releasing old keywords from GPU?

            all_combos = [self.keyword]
            self.keywords_embeds = []
            self.keywords_ids = []
            # print(all_combos)
            delta = 1e-6
            embed_lut = self.model.get_input_embeddings()
        # output_length = pred_embeds.size(1)       

            for word in all_combos: #all combos is meaningless at the moment, there is only one combo
                if self.args.target_tokenize_different:
                    with self.tokenizer.as_target_tokenizer():
                        self.keywords_ids.append(self.tokenizer.encode(word, add_special_tokens=False, return_tensors="pt").to(self.device)[0])
                else:
                    self.keywords_ids.append(self.tokenizer.encode(word, add_special_tokens=False, return_tensors="pt").to(self.device)[0])
                keyword_embeds = self.model.get_input_embeddings()(self.keywords_ids[-1]).detach()
                # print(self.tokenizer.encode(word, add_special_tokens=False, return_tensors="pt"))
                # keyword_ndist = -squared_cdist(keyword_embeds, self.model.get_input_embeddings().weight)
                self.ngram_length = keyword_embeds.size(0)
                self.keywords_embeds.append(keyword_embeds)

                ndist = -torch.square(torch.cdist(keyword_embeds, embed_lut.weight))
                # print(self.keywords_ids)
                # print(F.log_softmax(ndist, dim=-1).topk(k=10, dim=-1))
                ndist = F.log_softmax(ndist, dim=-1)#.index_select(dim=-1, index=self.keywords_ids[-1])
                # print(ndist.size())
                # self.epsilon = (-ndist.mean(dim=0).topk(k=2, dim=-1)[0][..., 1]).item() - delta #delta added to ensure strict inequality not less than or equal to
                self.epsilon = ndist.mean(dim=0).topk(k=2, dim=0)[0][..., 1].item() - delta
                # print(ndist.mean(dim=-1).topk(k=10, dim=0)[0] - delta)
                # self.epsilon = -self.epsilon
                # input()

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
            # print(kwargs.get("keyword"))

            # losses = []
            # keyword_embeds = self.keywords_embeds[0]
            #for keyword_embeds in self.keywords_embeds: #self.keywords_embeds just has one element
            
            # print(keyword_embeds.size(), pred_embeds.size())
            dist = torch.cdist(self.keywords_embeds[0], pred_embeds).square()
            # print(dist.size())
            try:
                ngram_nll = F.conv2d(dist.unsqueeze(1), self.eye_filter[:, :, :self.ngram_length, :self.ngram_length])/self.ngram_length
                ngram_nll = ngram_nll.squeeze(2).squeeze(1)
            except:
                print(kwargs.get("keyword").strip())
                print(self.ngram_length, dist.size(), self.keyword_embeds)

            # ngram_nlls = []
            # for i in range(ngram_length):
            #     ngram_nlls.append(dist[:, i, i:i+output_length-ngram_length+1])
            
            # ngram_nll2 = torch.stack(ngram_nlls, dim=1)
            # # print(ngram_nll.size())
            # ngram_nll2 = ngram_nll2.mean(dim=1) #mean across the length of ngram

            # assert torch.allclose(ngram_nll, ngram_nll2)

            # tau=0.01
            ngram_nll_q = F.gumbel_softmax(-ngram_nll/self.tau, dim=-1).detach()
            
            loss = ngram_nll_q * ngram_nll
            loss = loss.sum(dim=-1)
            # losses.append(loss)
            loss = -loss
            
            # loss = torch.stack(losses, dim=0).min(dim=0)[0]
            # print(loss)
            
            logging_output = {
                "loss": loss.data.cpu(),
            }
            return loss, logging_output   
