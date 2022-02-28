import torch.nn as nn
import torch

import torch.nn.functional as F

import numpy as np

def _get_scores(predict_emb, target_embedding):
    return predict_emb.matmul(target_embedding.weight.t())

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf'), filter_indices=[]):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            filter_indices: do not predict the given set of indices.
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # print(sorted_indices)
        
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # print(sorted_indices_to_remove, sorted_indices)
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        # print(filter_value)
        # print(indices_to_remove)
        # input("topp")
        logits[indices_to_remove] = filter_value

    elif top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if len(filter_indices) > 0:
        pass

    return logits

class TargetSimplex(nn.Module):
    def __init__(
        self,
        vocabsize,
        sent_length,
        batch_size,
        device,
        temperature=1.0,
        st=False,
        init_value=None,
        random_init=False,
        embed_scales=None,
        **kwargs,
    ):
        super(TargetSimplex, self).__init__()
        # special = torch.Tensor(batch_size, sent_length, 3).fill_(-1000)
        # special.requires_grad=False
        # self._pred_logiprobnn.Parameter(
            # torch.cat([special, torch.Tensor(batch_size, sent_length, vocabsize-3)], dim=-1).to(device)
        # )
        self._pred_logits = nn.Parameter(torch.Tensor(batch_size, sent_length, vocabsize).to(device))
        self.special_mask = torch.ones_like(self._pred_logits)
        self.temperature = temperature
        self.initialize(random_init=random_init, init_value=init_value)
        self.device = device
        self.st = st
        # self.sampling_strategy = sampling_strategy
        # self.sampling_strategy_k = sampling_strategy_k
        self.embed_scales = embed_scales
    
    # def sanitize(self, tokenizer):
    #     # this function reduces the probability of illegal tokens like <s> and other stuff to  not have a repeated sequence of </s>
    #     self._pred_logits[:,:, tokenizer.bos_token_id] = -1000.0
    #     self._pred_logits[:,:, tokenizer.eos_token_id] = -1000.0

        

    def forward(self, content_embed_lut, style_embed_lut=None, tokenizer=None, debug=False):
        
        # self.sanitize(tokenizer)
        # self.special_mask[:, :, :3] = 0.
        _, index = (self._pred_logits * self.special_mask).max(-1, keepdim=True)
        # print(index.size())
        predictions = index.squeeze(-1)
        # print(predictions.size())
        # input()

        if self.temperature == 0: # no softmax, special case, doesn't work don't use
            pred_probs = self._pred_logits
        else:    
            pred_probs = F.softmax(self._pred_logits / self.temperature, dim=-1)
        
        softmax_pred_probs = pred_probs
        if self.st:
            y_hard = torch.zeros_like(pred_probs).scatter_(-1, index, 1.0)
            pred_probs = y_hard - pred_probs.detach() + pred_probs

        if debug:
            print(softmax_pred_probs.max(dim=-1))
            print(softmax_pred_probs.gather(-1, index))
            print(self._pred_logits.gather(-1, index))
            print(torch.exp(self._pred_logits).eq(1.).all())
            print(index)
            input()

        source_pred_emb = (pred_probs.unsqueeze(-1) * content_embed_lut.weight).sum(dim=-2)
        target_pred_emb = None
        if style_embed_lut is not None:
            target_pred_emb = (pred_probs.unsqueeze(-1) * style_embed_lut.weight).sum(dim=-2)
        return (source_pred_emb, target_pred_emb, softmax_pred_probs), predictions, pred_probs
    

    def forward_multiple(self, embed_luts, **kwargs):
        if self.temperature == 0: # no softmax, special case, doesn't work don't use
            pred_probs = self._pred_logits
        else:    
            pred_probs = F.softmax(self._pred_logits / self.temperature, dim=-1)

        logits = self._pred_logits
        if self.sampling_strategy == "greedy":

            _, index = (logits * self.special_mask).max(-1, keepdim=True)
            predictions = index.squeeze(-1)
        elif self.sampling_strategy.startswith("topk"):
            top_k = int(self.sampling_strategy_k)
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=0)
            probabilities = F.softmax(filtered_logits, dim=-1)
            index = torch.multinomial(probabilities.view(-1, filtered_logits.size(-1)), 1)
            index = index.view(filtered_logits.size(0), filtered_logits.size(1), -1)
            predictions = index.squeeze(-1)
        elif self.sampling_strategy.startswith("topp"):
            top_p = float(self.sampling_strategy_k)
            filtered_logits = top_k_top_p_filtering(logits, top_k=0, top_p=top_p)
            probabilities = F.softmax(filtered_logits, dim=-1)
            index = torch.multinomial(probabilities.view(-1, filtered_logits.size(-1)), 1)
            index = index.view(filtered_logits.size(0), filtered_logits.size(1), -1)
            predictions = index.squeeze(-1)
        else:
            raise ValueError("wrong decode method. If you want to do beam search, change the objective function")
        
        softmax_pred_probs = pred_probs
        if self.st:
            y_hard = torch.zeros_like(pred_probs).scatter_(-1, index, 1.0)
            pred_probs = y_hard - pred_probs.detach() + pred_probs
        
        pred_embs = []
        for embed_lut, embed_scale in zip(embed_luts, self.embed_scales):
            pred_embs.append((pred_probs.unsqueeze(-1) * embed_lut.weight).sum(dim=-2))
        
        return (pred_embs, ), predictions, (pred_probs, softmax_pred_probs) #pred_probs is actually just logits


    def initialize(self, random_init=False, init_value=None):
        if init_value is not None:
            # print(init_value.size())
            eps = 0.9999
            V = self._pred_logits.size(2)
            print(V)
            print(1-eps-eps/(V-1))
            print(eps/(V-1))
            init_value = torch.zeros_like(self._pred_logits).scatter_(-1, init_value.unsqueeze(2), 1.0-eps-eps/(V-1))
            # init_value = torch.log(init_value + eps/(V-1))
            self._pred_logits.data.copy_(init_value.data)
        elif random_init:
            torch.nn.init.uniform_(self._pred_logits, 1e-6, 1e-6)
        else:
            torch.nn.init.zeros_(self._pred_logits)
            # init_value = torch.empty_like(self._pred_logits).fill_(0.)
            # self._pred_logits.data.copy_(init_value.data)

    @classmethod
    def decode_beam(cls, pred_probs, model, embed_lut, prefix, device, beam_size=1):
        answers = []
        # pred_probs = F.softmax(self._pred_logits / self.temperature, dim=-1)
        print(pred_probs.unsqueeze(-1).size())
        pred_emb = (pred_probs.unsqueeze(-1) * embed_lut.weight).sum(dim=-2)
        print(pred_emb.size())
        prefix_emb = embed_lut(prefix)

        _, topk_words = pred_probs.topk(2 * beam_size, dim=-1)
        print(topk_words.size())
        for b in range(pred_probs.size(0)):
            beam = torch.empty((0, beam_size)).long().to(device)
            beam_emb = torch.empty((0, beam_size, embed_lut.weight.size(1))).to(device)

            for i in range(pred_probs.size(1)):
                cand = embed_lut(topk_words[b, i : i + 1])
                print(cand.size())
                input_emb = torch.cat(
                    [
                        prefix_emb.repeat(2 * beam_size, 1, 1),
                        beam_emb.repeat(2, 1, 1),
                        cand,
                        pred_emb[b, i:-1, :].repeat(2 * beam_size, 1, 1),
                    ],
                    dim=1,
                )
                print(input_emb)

                # feed into model and get scores
                model_output = model(inputs_embeds=input_emb)
                lm_logits = model_output[0]
                lm_logprobs = F.log_softmax(lm_logits, dim=-1)

                print(lm_logprobs.size())
                print(beam.size())
                print(pred_probs.size())
                input()
                # compute nll loss
                # prefix

                # might have to transpose
                loss = F.nll_loss(
                    lm_logprobs[:, : prefix.size(1) - 1, :].squeeze(0),
                    prefix[:, 1 : prefix.size(1)].squeeze(0),
                    reduction="none",
                )

                loss += F.nll_loss(
                    lm_logprobs[:, prefix.size(1) - 1 : prefix.size() + b, :].squeeze(
                        0
                    ),
                    torch.cat(
                        [
                            prefix[:, prefix.size() - 1 :],
                            beam.repeat(2, 1),
                            topk_words[b : i : i + 1],
                        ],
                        dim=1,
                    ),
                )

                # suffix
                suffix_loss = (
                    pred_probs[b, i + 1 :, :].unsqueeze() * lm_logprobs[i:-1, :]
                ).sum(dim=-1)
                print(suffix_loss.size())

                # sort beam
                _, beam = torch.topk(-loss, dim=-1)

            answer.append(beam[0])

        return answer


class TargetProbability(nn.Module): #this is the class used in the final results
    def __init__(
        self,
        vocabsize,
        sent_length,
        batch_size,
        device,
        st=False,
        init_value=None,
        random_init=False,
        do_sample=False,
        top_p=1.0,
        top_k=0.0,
        embed_scales=None,
        max_steps=1,
    ):
        super(TargetProbability, self).__init__()
        self._pred_probs = nn.Parameter(torch.Tensor(batch_size, sent_length, vocabsize).to(device))
        self.initialize(random_init=random_init, init_value=init_value)
        self.device = device
        self.st = st #straight-through or not
        self.do_sample = do_sample
        self.top_p = top_p
        self.top_k = top_k
        self.embed_scales = embed_scales   
        self.max_steps = max_steps

        self.begintemp = 10.0
        self.finaltemp = 0.1
        self.step = 0
        self.tempschedule="geometric"
        self.r = np.power(self.finaltemp/self.begintemp, 1/(self.max_steps-1))
        

    def forward_multiple(self, embed_luts, **kwargs):
        
        if self.embed_scales is None:
            embed_scales = [1.0 for i in embed_luts]

        pred_probs = self._pred_probs
        if not self.do_sample:
            _, index = pred_probs.max(-1, keepdim=True)
            predictions = index.squeeze(-1)
        # elif self.sampling_strategy == "notpad": #top-1 might be a <pad> token, this ensure pad is never sampled
        #     _, index = pred_probs.topk(-1, k=2, keepdim=True)
        #     predictions = index.squeeze(-1)
        else:
            temperature = self.begintemp * pow(self.r, self.step)
            pred_logits = torch.log(pred_probs)
            next_token = torch.empty((pred_logits.size(0), pred_logits.size(1))).long().to(pred_logits.device)
            for i in range(pred_logits.size(1)):
                filtered_logits = top_k_top_p_filtering(pred_logits[:, i, :]/temperature, self.top_k, self.top_p)
                next_token[:, i] = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            self.step += 1
            predictions = next_token
            index = next_token.unsqueeze(2)
            
        
        softmax_pred_probs = pred_probs
        if self.st:
            y_hard = torch.zeros_like(pred_probs).scatter_(-1, index, 1.0)
            pred_probs = y_hard - pred_probs.detach() + pred_probs
        
        pred_embs = []
        for embed_lut, embed_scale in zip(embed_luts, self.embed_scales):
            if embed_lut.weight.size(0) > pred_probs.size(2):
                pred_embs.append((pred_probs.unsqueeze(-1) * embed_lut.weight[:pred_probs.size(2), :]).sum(dim=-2))
            elif embed_lut.weight.size(0) < pred_probs.size(2):
                pred_embs.append((pred_probs[:, :, :embed_lut.weight.size(0)].unsqueeze(-1) * embed_lut.weight).sum(dim=-2))
            else:
                pred_embs.append((pred_probs.unsqueeze(-1) * embed_lut.weight).sum(dim=-2))
        
        return (pred_embs, ), predictions, (pred_probs, softmax_pred_probs) #pred_probs is actually just logits


    def initialize(self, random_init=False, init_value=None):
        if init_value is not None:
            eps = 0.999
            V = self._pred_probs.size(2)
            init_value_ = torch.zeros_like(self._pred_probs).fill_(eps/(V-1))
            init_value_ = init_value_.scatter_(-1, init_value.unsqueeze(2), 1.0-eps)
            self._pred_probs.data.copy_(init_value_.data)
        elif random_init: #sample a simplex from a dirichlet distribution for each token probability
            p = torch.distributions.dirichlet.Dirichlet(10000 * torch.ones(self._pred_probs.size(-1)))
            init_value = torch.empty_like(self._pred_probs)
            for i in range(self._pred_probs.size(0)):
                for j in range(self._pred_probs.size(1)):
                    init_value[i, j] = p.sample()
            self._pred_probs.data.copy_(init_value.data) 
        else: # uniform
            torch.nn.init.ones_(self._pred_probs)
            self._pred_probs.data.div_(self._pred_probs.data.sum(dim=-1, keepdims=True))

class TargetEmbeddings(nn.Module): 
    def __init__(
        self,
        embed_dim,
        embed_lut,
        sent_length,
        batch_size,
        device,
        st=False,
        init_value=None,
        random_init=False,
        sampling_strategy="argmax",
        sampling_strategy_k = 0,
        embed_scales=None,
        metric="dot",
        same_embed=True,
        final_bias=None,
        eos_token_id=None
    ):
        super(TargetEmbeddings, self).__init__()
        self._pred_embeds = nn.Parameter(torch.Tensor(batch_size, sent_length, embed_dim).to(device))
        self.device = device
        self.st = st #straight-through or not
        self.sampling_strategy = sampling_strategy
        self.sampling_strategy_k = sampling_strategy_k   
        self.embed_scales = embed_scales   
        self.metric=metric   
        self.same_embed=same_embed
        self.temperature=0.1
        self.eos_token_id = eos_token_id
        
        if self.metric == "cosine":
            self.tgt_emb = torch.nn.functional.normalize(embed_lut.weight.data, p=2, dim=-1)
        else:
            self.tgt_emb = embed_lut.weight.data
        
        if final_bias is not None:
            self.final_bias = final_bias.data
        else:
            self.final_bias = None
        
        self.initialize(random_init=random_init, init_value=init_value)
        print(self.eos_token_id)

    def forward_multiple(self, embed_luts, new_predictions=None, **kwargs):
        if self.same_embed:
            embed_luts = [embed_luts[0] for _ in embed_luts] #need to verify
        
        if self.embed_scales is None:
            embed_scales = [1.0 for i in embed_luts]
  
        predictions = new_predictions        
        if predictions is None:
            pred_logits = _emb_to_scores(self.metric, self._pred_embeds, self.tgt_emb, self.final_bias)
            pred_probs = pred_logits#F.softmax(pred_logits / self.temperature, dim=-1)
            _, predictions = pred_logits.max(-1) 
            softmax_pred_probs = pred_probs     # not used

            # eos_true = predictions.eq(self.eos_token_id)
            # if eos_true.any():
            #     eos_mask = eos_true.float().cumsum(dim=-1).eq(0).long()
            #     print(eos_mask)
            #     input()
            # else:
            #     eos_mask = torch.ones_like(eos_true).long().to(self.device)
            # all_eos = torch.empty_like(predictions).fill_(self.eos_token_id)
            # predictions = eos_mask * predictions + (1 - eos_mask) * all_eos
        else:
            pred_probs, softmax_pred_probs = None, None

        pred_embs = []
        if self.st:
            # y_hard = torch.zeros((self._pred_embeds.size(0), self._pred_embeds.size(1), self.tgt_out_emb.size(0))).scatter_(-1, predictions.unsqueeze(-1), 1.0)
            # pred_probs = y_hard - pred_probs.detach() + pred_probs
            for embed_lut, embed_scale in zip(embed_luts, self.embed_scales):
                n_pred_embs = embed_lut(predictions)
                replace_mask = predictions.unsqueeze(2).ne(self._pred_ids).float()
                # print(replace_mask)
                self._pred_embeds.data.copy_((replace_mask * n_pred_embs +  (1. - replace_mask) * self._pred_embeds).data)
                pred_embs.append(self._pred_embeds + (n_pred_embs-self._pred_embeds).detach())
                # input()
        else:
            for embed_lut, embed_scale in zip(embed_luts, self.embed_scales):
                pred_embs.append(self._pred_embeds)
        
        return (pred_embs, self._pred_embeds), predictions, (pred_probs, softmax_pred_probs)


    def initialize(self, random_init=False, init_value=None, ):
        if init_value is not None:
            self._pred_embeds.data.copy_(init_value.data)
        elif random_init:
           torch.nn.init.normal_(self._pred_embeds, 0.0, 1.0)
        else: # uniform
            # vocabsize = self.tgt_emb.size(0)
            # uniform = torch.ones((self._pred_embeds.size(0), self._pred_embeds.size(1), vocabsize)).to(self.device)/vocabsize
            # vec = (uniform.unsqueeze(-1) * self.tgt_emb).sum(dim=-2)
            # self._pred_embeds.data.copy_(vec.data)
            # init_value = torch.ones_like(self._pred_embeds).to(self.device)
            # init_value = torch.nn.functional.normalize(init_value, p=2, dim=-1)
            # self._pred_embeds.data.copy_(init_value.data)
            # print(torch.linalg.norm(self.tgt_emb, dim=-1).max(-1))
            # print(torch.linalg.norm(self.tgt_emb, dim=-1).min(-1))
            # input()
            torch.nn.init.zeros_(self._pred_embeds)
        
        _, self._pred_ids = _emb_to_scores(self.metric, self._pred_embeds, self.tgt_emb, self.final_bias).max(dim=-1, keepdim=True)



            # proj = self._pred_embeds.matmul(self.tgt_emb.t())
            # projembds = proj.max(dim=-1)[0]
            # self._pred_embeds.data.copy_(projembds.data)
            
        
    def printparams(self):
        print(self._pred_embeds)


def _emb_to_scores(metric, pred_emb, tgt_out_emb, bias=None, norm=None):
    if metric == "l2": 
        return -torch.cdist(pred_emb, tgt_out_emb.unsqueeze(0))
        # scores = (pred_emb.unsqueeze(2) - tgt_out_emb.unsqueeze(0))
        # scores = -(scores*scores).sum(dim=-1)

    elif metric == "cosine": # cosine and vmf work more or less the same for decoding
        pred_emb_unitnorm = torch.nn.functional.normalize(pred_emb, p=2, dim=-1)
        target_unitnorm = torch.nn.functional.normalize(tgt_out_emb, p=2, dim=-1)
        scores = pred_emb_unitnorm.matmul(target_unitnorm.t())
    
    elif metric == "dotbias":
        return pred_emb.matmul(tgt_out_emb.t()) + bias

    else: # dot product
        return pred_emb.matmul(tgt_out_emb.t())
    
    return scores