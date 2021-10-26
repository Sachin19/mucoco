from mucoco.losses import BaseLoss, register_loss


import torch 
import torch.nn.functional as F

@register_loss("marianmt")
class MarianMTLoss(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 

        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device

        self.pad_token_id = self.tokenizer.pad_token_id
    
    def compute_loss(self, batch, preds, **kwargs):
        '''
        batch: a tuple (source, prefix). If giving a prompt to the decoder, it can be specified using "prefix"
        preds: a tuple containing (predicted tokens, predicted embeddings, predicted probabilities), this is obtained through a forward pass on the optimizable target parameters (See utils/target.py)
        '''
        source, prefix = batch

        pred_tokens, pred_embeds, pred_probs = preds
        pred_probs = pred_probs[0]

        bos = torch.empty((source.size(0), 1)).long().to(self.device).fill_(self.pad_token_id)
        target_input_tokens = torch.cat([bos, prefix, pred_tokens], dim=1)

        embed_lut = self.model.get_decoder().get_input_embeddings()
        target_input_embeds = torch.cat([embed_lut(bos), embed_lut(prefix), pred_embeds], dim=1) * kwargs["embed_scale"]
        model_output = self.model(input_ids=source, decoder_inputs_embeds=target_input_embeds)

        lm_logits = model_output.logits
        lm_logprobs = F.log_softmax(lm_logits, dim=-1)

        if prefix.size(1) > 0:
            xentropy_prefix = F.nll_loss(lm_logprobs[:,:prefix.size(1),:].squeeze(0), prefix.squeeze(0), reduction="none").sum(dim=-1)
        else:
            xentropy_prefix = 0.0
        
        xentropy_pred = (-lm_logprobs[:, prefix.size(1): -1, :] * pred_probs).sum(dim=-1) 
        xentropy_pred = xentropy_pred.sum(dim=-1)
        xentropy_pred = xentropy_pred - lm_logprobs[:, -1, tokenizer.eos_token_id]

        _, mm = lm_logprobs.max(dim=-1)

        xentropy = xentropy_pred + xentropy_prefix 
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

    def compute_gold_loss(self, batch, **kwargs):
        '''
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        source, target = batch
        batch_size = source.size(0)
        bos = torch.empty((source.size(0), 1)).long().to(self.device).fill_(self.pad_token_id)
        eos = torch.empty((source.size(0), 1)).long().to(self.device).fill_(self.eos_token_id)    
        target_input_tokens = torch.cat([bos, target, eos], dim=1)

        model_output = self.model(input_ids=source, decoder_input_ids=target_input_tokens[:, :-1], labels=target_input_tokens[:, 1:])

        lm_logits = model_output.logits
        lm_logprobs = F.log_softmax(lm_logits, dim=-1)

        loss = F.nll_loss(lm_logprobs.squeeze(0), target_input_tokens[:, 1:].squeeze(0), reduction="none")
        loss = loss.sum(dim=-1)

        if self.args.length_normalize:
            loss /= lm_logprobs.size(1)
        
        _, mm = lm_logprobs.max(dim=-1)

        logging_output = {
            "loss": loss.data.cpu(),
            "max_length": target.size(1),
            "nsentences": batch_size,
            "mm": mm,
        }
        return loss, logging_output   
    
def marianMTloss(model,
    batch,
    preds,
    tokenizer,
    args, **kwargs):
    
    source, prefix = batch

    pred_tokens, pred_embeds, pred_probs = preds
    pred_probs = pred_probs[0]

    bos = torch.empty((source.size(0), 1)).long().to(source.device).fill_(tokenizer.pad_token_id)
    target_input_tokens = torch.cat([bos, prefix, pred_tokens], dim=1)

    embed_lut = model.get_decoder().get_input_embeddings()
    target_input_embeds = torch.cat([embed_lut(bos), embed_lut(prefix), pred_embeds], dim=1) * kwargs["embed_scale"]
    model_output = model(input_ids=source, decoder_inputs_embeds=target_input_embeds)

    lm_logits = model_output.logits
    lm_logprobs = F.log_softmax(lm_logits, dim=-1)

    # print(source, target_input_tokens)
    # input()
    if prefix.size(1) > 0:
        xentropy_prefix = F.nll_loss(lm_logprobs[:,:prefix.size(1),:].squeeze(0), prefix.squeeze(0), reduction="none").sum(dim=-1)
    else:
        xentropy_prefix = 0.0
    
    xentropy_pred = (-lm_logprobs[:, prefix.size(1): -1, :] * pred_probs).sum(dim=-1) # / (pred_probs.size(-1))
    # print(xentropy_pred, -lm_logprobs[:, -1, tokenizer.eos_token_id])
    # input()
    xentropy_pred = xentropy_pred.sum(dim=-1)
    xentropy_pred = xentropy_pred - lm_logprobs[:, -1, tokenizer.eos_token_id]

    #entropy_pred = -(pred_probs * torch.log(pred_probs)).sum(dim=-1).sum(dim=-1)
    _, mm = lm_logprobs.max(dim=-1)

    # print(f"entropy {entropy_pred}")
    xentropy = xentropy_pred + xentropy_prefix  # - entropy_pred
    if args.length_normalize:
        xentropy /= lm_logprobs.size(1)

    #if expected_nll is None:
    loss = xentropy
    # print("here:", source, target_input_tokens, target_input_embeds, loss)
    #else:
    #    loss = (xentropy - expected_nll) ** 2
    sample_size = prefix.size(1)

    logging_output = {
        "loss": loss.data.cpu(),
        "ntokens": prefix.size(1) + pred_tokens.size(1),
        "nsentences": prefix.size(0),
        "sample_size": sample_size,
        "lm_logprobs": lm_logprobs.data.cpu(),
        "mm": mm,
    }
    return loss, sample_size, logging_output

def gold_marianMTloss(model, batch, tokenizer, args, **kwargs):
    source, target = batch
    bos = torch.empty((source.size(0), 1)).long().to(source.device).fill_(tokenizer.pad_token_id)
    eos = torch.empty((source.size(0), 1)).long().to(source.device).fill_(tokenizer.eos_token_id)    
    target_input_tokens = torch.cat([bos, target, eos], dim=1)

    model_output = model(input_ids=source, decoder_input_ids=target_input_tokens[:, :-1], labels=target_input_tokens[:, 1:])

    lm_logits = model_output.logits
    lm_logprobs = F.log_softmax(lm_logits, dim=-1)

    loss = F.nll_loss(lm_logprobs.squeeze(0), target_input_tokens[:, 1:].squeeze(0), reduction="none")
    # print(loss)
    loss = loss.sum(dim=-1)
    if args.length_normalize:
        loss /= lm_logprobs.size(1)
    # print(source, target_input_tokens, loss, model_output.loss/lm_logprobs.size(1))
    
    _, mm = lm_logprobs.max(dim=-1)
    # print(mm)
    # input("gold")
    sample_size = target.size(1)
    # print("gold", loss)

    logging_output = {
        "loss": loss.data,
        "ntokens": target.size(1),
        "nsentences": target.size(0),
        "sample_size": sample_size,
        "mm": mm,
    }
    return loss, sample_size, logging_output