from mucoco.losses import BaseLoss, register_loss


import torch 
import torch.nn.functional as F

@register_loss("usim")
class USimLoss(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 
        
        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device

        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id    
    
    def compute_loss(self, batch, preds, **kwargs):
        '''
        batch: a tuple (source, prefix). If giving a prompt to the decoder, it can be specified using "prefix"
        preds: a tuple containing (predicted tokens, predicted embeddings, predicted probabilities), this is obtained through a forward pass on the optimizable target parameters (See utils/target.py)
        '''
        real_source, prefix = batch
        source = kwargs.get("additional_batch") #in STRAP model, real_source x is paraphrased to source y which is then transformed into target y. The language model sees only source and target, not the real source

        pred_tokens, pred_embeds, pred_probs = preds
        pred_probs = pred_probs[0]

        batch_size = source.size(0)

        bos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.bos_token_id)
        eos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.eos_token_id) 
        # input_tokens = torch.cat([source, bos, prefix, pred_tokens, eos, eos], dim=1)

        embed_lut = self.model.get_input_embeddings()
        input_embeds = torch.cat([embed_lut(source), embed_lut(bos), embed_lut(prefix), pred_embeds, embed_lut(eos), embed_lut(eos)], dim=1)

        source_segment_id = torch.empty((batch_size, source.size(1))).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[1])
        target_segment_id = torch.empty((batch_size, prefix.size(1) + pred_tokens.size(1) + 3)).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[2])
        segment = torch.cat([source_segment_id, target_segment_id], dim=1)

        model_output = model(inputs_embeds=input_embeds, token_type_ids=segment)

        lm_logits = model_output[0][:, source.size(1):]
        lm_logprobs = F.log_softmax(lm_logits, dim=-1)

        if prefix.size(1) > 0:
            xentropy_prefix = F.nll_loss(lm_logprobs[:,:prefix.size(1),:].squeeze(0), prefix.squeeze(0), reduction="none").sum(dim=-1)
        else:
            xentropy_prefix = 0.0
        
        xentropy_pred = (-lm_logprobs[:, prefix.size(1): -3, :] * pred_probs).sum(dim=-1).sum(dim=-1) # / (pred_probs.size(-1))
        xentropy_pred = xentropy_pred - lm_logprobs[:, -3, self.eos_token_id] - lm_logprobs[:, -2, self.eos_token_id] 

        #entropy_pred = -(pred_probs * torch.log(pred_probs)).sum(dim=-1).sum(dim=-1)
        _, mm = lm_logprobs.max(dim=-1)

        xentropy = xentropy_pred + xentropy_prefix  # - entropy_pred
        if self.args.length_normalize:
            xentropy /= lm_logprobs.size(1)

        
        loss = xentropy

        logging_output = {
            "loss": loss.data.cpu(),
            "max_length": prefix.size(1) + pred_tokens.size(1),
            "nsentences": batch_size,
            "lm_logprobs": lm_logprobs.data.cpu(),
            "mm": mm,
        }

        return loss, logging_output

    def compute_gold_loss(self, batch):
        '''
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        real_source, target = batch
        source = kwargs.get("additional_batch") #in STRAP model, real_source x is paraphrased to source y which is then transformed into target y. The language model sees only source and target, not the real source

        batch_size = source.size(0)

        bos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.bos_token_id)
        eos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.eos_token_id) 

        input_tokens = torch.cat([source, bos, target, eos, eos], dim=1)

        source_segment_id = torch.empty((batch_size, source.size(1))).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[1])
        target_segment_id = torch.empty((batch_size, target.size(1) + 3)).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[2])
        segment = torch.cat([source_segment_id, target_segment_id], dim=1)

        model_output = model(input_tokens, token_type_ids=segment)
        target = torch.cat([bos, target, eos, eos], dim=1)

        lm_logits = model_output[0][:, source.size(1):]
        lm_logprobs = F.log_softmax(lm_logits, dim=-1)

        loss = F.nll_loss(lm_logprobs[:,:target.size(1) - 1,:].squeeze(0), target[:, 1:target.size(1)].squeeze(0), reduction="none").sum(dim=-1)
        
        if self.args.length_normalize:
            loss /= lm_logprobs.size(1)

        _, mm = lm_logprobs.max(dim=-1) # used for debugging

        logging_output = {
            "loss": loss.data.cpu(),
            "max_length": target.size(1),
            "nsentences": batch_size,
            "mm": mm,
        }
        return loss, logging_output   
    
