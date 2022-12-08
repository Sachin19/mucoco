from mucoco.losses import BaseLoss, register_loss

import torch 
import torch.nn.functional as F
import torch.nn as nn

import logging

@register_loss("classification")
class ClassificationLoss(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 
        
        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id  

        self.eos = torch.empty((1, 1)).long().to(self.device).fill_(self.eos_token_id)  
        # print(self.eos_token_id)
    
    def compute_loss(self, batch, preds, **kwargs):
        '''
        batch: a tuple (source, prefix). If giving a prompt to the decoder, it can be specified using "prefix"
        preds: a tuple containing (predicted tokens, predicted embeddings, predicted probabilities), this is obtained through a forward pass on the optimizable target parameters (See utils/target.py)
        '''
        logging.getLogger(self.model.__class__.__name__).disabled=True
        if len(batch) == 2:
            source, prefix = batch
        else:
            prefix = batch
        
        pred_tokens, pred_embeds, pred_probs = preds
        batch_size = pred_embeds.size(0)

        # bos = torch.empty((source.size(0), 1)).long().to(self.device).fill_(self.bos_token_id)
        # eos = torch.empty((source.size(0), 1)).long().to(self.device).fill_(self.eos_token_id)
        eos = self.eos
        #input_tokens = torch.cat([bos, prefix, pred_tokens, eos], dim=1)

        embed_lut = self.model.get_input_embeddings()
        if isinstance(embed_lut, nn.Sequential):
            input_embeds = torch.cat([embed_lut(source), embed_lut(prefix), embed_lut[1](pred_embeds)], dim=1) * kwargs["embed_scale"]
        else:
            input_embeds = torch.cat([embed_lut(source), embed_lut(prefix), pred_embeds], dim=1) * kwargs["embed_scale"]
        # if isinstance(embed_lut, nn.Sequential):
        #     input_embeds = torch.cat([embed_lut[1](pred_embeds), embed_lut(eos), embed_lut(eos)], dim=1) * kwargs["embed_scale"]
        # else:
        #     input_embeds = torch.cat([pred_embeds, embed_lut(eos), embed_lut(eos)], dim=1) * kwargs["embed_scale"]
        

        model_output = self.model(inputs_embeds=input_embeds)
        lm_logits = model_output[0]
        lm_logprobs = F.log_softmax(lm_logits, dim=-1)
        probs = F.softmax(lm_logits, dim=-1)
        label_id = kwargs.get("label_id", 1)
        loss = -lm_logprobs[:, label_id]

        label_prediction = lm_logits.argmax(dim=-1).item()

        logging_output = {
            "loss": loss.data.cpu(),
            "max_length": prefix.size(1) + pred_tokens.size(1),
            "nsentences": batch_size,
            "lm_logprobs": lm_logprobs.data.cpu(),
        }

        return loss, logging_output

    def compute_gold_loss(self, batch, **kwargs):
        '''
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        with torch.no_grad():
            source, target = batch

            batch_size=target.size(0)
            target = torch.cat([source, target], dim=1)
            model_output = self.model(target)

            lm_logits = model_output[0]
            lm_logprobs = F.log_softmax(lm_logits, dim=-1)
            label_id=kwargs.get("label_id", 1)
            loss = -lm_logprobs[:, label_id] #label_id = 1
            # loss = lm_logits[:, 1-label_id] - lm_logits[:, label_id]
            label_prediction = lm_logprobs.argmax(dim=-1).item()

            logging_output = {
                "loss": loss.data.cpu(),
                "nsentences": batch_size,
                "label_prediction": label_prediction
            }
            return loss, logging_output   
