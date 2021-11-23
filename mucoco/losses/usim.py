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

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id    
    
    def compute_loss(self, batch, preds, **kwargs):
        '''
        batch: a tuple (source, prefix). If giving a prompt to the decoder, it can be specified using "prefix"
        preds: a tuple containing (predicted tokens, predicted embeddings, predicted probabilities), this is obtained through a forward pass on the optimizable target parameters (See utils/target.py)
        '''
        source, target_prefix = batch
        pred_tokens, pred_embeds, pred_probs = preds

        batch_size = source.size(0)

        gold_features = mean_pooling(self.model(input_ids=source), attention_mask=torch.ones(source.size(0), source.size(1)).to(self.device))
        gold_features = gold_features.detach()  # don't need to pass gradients through this

        bos = torch.empty((source.size(0), 1)).long().to(source.device).fill_(self.bos_token_id)
        eos = torch.empty((source.size(0), 1)).long().to(source.device).fill_(self.eos_token_id)
        # target = torch.cat([bos, target_prefix, pred_tokens, eos], dim=1)
        #probably useless, delete if true

        embed_lut = self.model.get_input_embeddings()
        target_embeds = torch.cat([embed_lut(bos), embed_lut(target_prefix), pred_embeds, embed_lut(eos)], dim=1)
        
        target_features = mean_pooling(self.model(inputs_embeds=target_embeds), attention_mask=torch.ones(target_embeds.size(0), target_embeds.size(1)).to(self.device))
        
        loss = (1.0 - (F.normalize(gold_features, dim=-1, p=2) * F.normalize(target_features, dim=-1, p=2)).sum(dim=-1))

        logging_output = {
            "loss": loss.data.cpu(),
            "max_length": target_prefix.size(1) + pred_tokens.size(1),
            "nsentences": batch_size,
        }

        return loss, logging_output

    def compute_gold_loss(self, batch, **kwargs):
        '''
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        # change this based on bert or whatever
        source, target = batch
        batch_size = target.size(0)
        source_features = mean_pooling(self.model(input_ids=source), attention_mask=torch.ones(source.size(0), source.size(1)).to(self.device))

        bos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.bos_token_id)
        eos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.eos_token_id) 
        target = torch.cat([bos, target, eos], dim=1)
        target_features = mean_pooling(self.model(input_ids=target), attention_mask=torch.ones(target.size(0), target.size(1)).to(self.device))
    
        loss = (1.0 - (F.normalize(source_features, dim=-1, p=2) * F.normalize(target_features, dim=-1, p=2)).sum(dim=-1))

        logging_output = {
            "loss": loss.data.cpu(),
            "max_length": target.size(1),
            "nsentences": batch_size,
        }
        return loss, logging_output   
    
#Mean Pooling for content loss- Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask