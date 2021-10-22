from mucoco.losses import BaseLoss
from mucoco.losses import register_loss

import torch 
import torch.nn.functional as F

try:
    import ot
except:
    ot = None

import numpy as np

@register_loss('wmd')
class WMD(BaseLoss):
    def __init__(self, model, tokenizer, args):
        super().__init__()

        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device
        self.distance_metric = getattr(args, "wmd_metric", "cosine")

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id    
    
    def compute_loss(self, batch, preds, **kwargs):

        source, target_prefix = batch
        pred_tokens, pred_embeds, pred_probs = preds

        batch_size = source.size(0)

        bos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.bos_token_id)
        eos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.eos_token_id) 

        # target = torch.cat([bos, target_prefix, pred_tokens, eos], dim=1)

        embed_lut = self.model.get_input_embeddings()
        source_embeds = embed_lut(source)
        target_embeds = torch.cat([embed_lut(bos), embed_lut(target_prefix), pred_embeds, embed_lut(eos)], dim=1)
        
        if self.distance_metric == "cosine":
            source_embeds = F.normalize(source_embeds, p=2, dim=-1)
            target_embeds = F.normalize(target_embeds, p=2, dim=-1)
            pairwise_distance = 1. - (source_embeds.unsqueeze(2) * target_embeds.unsqueeze(1)).sum(dim=-1)
        else:
            pairwise_distance = (source_embeds.unsqueeze(2) - target_embeds.unsqueeze(1))
            pairwise_distance = torch.sqrt((pairwise_distance * pairwise_distance).sum(dim=-1))
        
        a = np.ones((source_embeds.size(1),))/source_embeds.size(1)
        b = np.ones((target_embeds.size(1),))/target_embeds.size(1)
        
        M = pairwise_distance.data.cpu().numpy()
        allT = []
        alld = []
        for i in range(source_embeds.size(0)):
            T = ot.emd(a, b, M[i])
            d = np.sum(T * M[i])
            allT.append(T)
            alld.append(d)
        
        allT = torch.from_numpy(np.concatenate(allT, axis=0)).to(pairwise_distance.device)

        loss = (allT * pairwise_distance).sum(2).sum(1)

        logging_output = {
            "loss": loss.data.cpu(),
            "max_length": target_prefix.size(1) + pred_tokens.size(1),
            "nsentences": batch_size,
        }
        return loss, logging_output

    def compute_gold_loss(self, batch, **kwargs):
        source, target = batch
        embed_lut = self.model.get_input_embeddings()
        source_embeds = embed_lut(source)
        target_embeds = embed_lut(target)
        
        if self.distance_metric == "cosine":
            source_embeds = F.normalize(source_embeds, p=2, dim=-1)
            target_embeds = F.normalize(target_embeds, p=2, dim=-1)
            pairwise_distance = 1. - (source_embeds.unsqueeze(2) * target_embeds.unsqueeze(1)).sum(dim=-1)
        else:
            pairwise_distance = (source_embeds.unsqueeze(2) - target_embeds.unsqueeze(1))
            pairwise_distance = torch.sqrt((pairwise_distance * pairwise_distance).sum(dim=-1))
        
        a = np.ones((source_embeds.size(1),))/source_embeds.size(1)
        b = np.ones((target_embeds.size(1),))/target_embeds.size(1)
        
        M = pairwise_distance.data.cpu().numpy()
        allT = []
        alld = []
        for i in range(source_embeds.size(0)):
            T = ot.emd(a, b, M[i])
            d = np.sum(T * M[i])
            allT.append(T)
            alld.append(d)
        
        allT = torch.from_numpy(np.concatenate(allT, axis=0)).to(pairwise_distance.device)
        
        # print((allT * pairwise_distance).size())
        loss = (allT * pairwise_distance).sum(2).sum(1)

        sample_size = target.size(1)
        logging_output = {
            "loss": loss.data,
            "max_length": target.size(1),
            "nsentences": target.size(0),
        }
        return loss, logging_output