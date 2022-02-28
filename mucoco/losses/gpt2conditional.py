from mucoco.losses import BaseLoss, register_loss


import torch 
import torch.nn.functional as F

@register_loss("gpt2conditional")
class GPT2ConditionalLoss(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 

        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device

        self.bos_token_id = self.tokenizer.bos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id    
    
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
        max_prefix_length = getattr(self.args, 'max_prefix_length', source.size(1) + 1)
        pad_length = max(0, max_prefix_length - source.size(1))
        bos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.bos_token_id)
        pad = torch.empty((batch_size, pad_length)).long().to(self.device).fill_(self.pad_token_id)
        eos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.eos_token_id) 
        # input_tokens = torch.cat([pad, source, bos, prefix, pred_tokens, eos], dim=1)

        embed_lut = self.model.get_input_embeddings()
        input_embeds = torch.cat([embed_lut(pad), embed_lut(source), embed_lut(bos), embed_lut(prefix), pred_embeds, embed_lut(eos)], dim=1)

        source_segment_id = torch.empty((batch_size, pad_length + source.size(1))).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[1])
        target_segment_id = torch.empty((batch_size, prefix.size(1) + pred_tokens.size(1) + 2)).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[2])
        segment = torch.cat([source_segment_id, target_segment_id], dim=1)

        losstype = getattr(self.args, "loss_type", "xentropy")
        if losstype == "xentropy":
            model_output = self.model(inputs_embeds=input_embeds, token_type_ids=segment)
            lm_logits = model_output[0][:, source.size(1) + pad_length:]
            lm_logprobs = F.log_softmax(lm_logits, dim=-1)

            if prefix.size(1) > 0:
                xentropy_prefix = F.nll_loss(lm_logprobs[:,:prefix.size(1),:].squeeze(0), prefix.squeeze(0), reduction="none").unsqueeze(0).sum(dim=-1)
            else:
                xentropy_prefix = 0.0
            
            xentropy_pred = (-lm_logprobs[:, prefix.size(1): -2, :] * pred_probs).sum(dim=-1).sum(dim=-1) # / (pred_probs.size(-1))
            xentropy_pred = xentropy_pred - lm_logprobs[:, -2, self.eos_token_id] - lm_logprobs[:, -2, self.eos_token_id] 

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
        elif losstype in ["l2", "cosine", "dot", "dotplusplus", "detachdot"]:
            model_output = self.model.transformer(inputs_embeds=input_embeds, token_type_ids=segment)
            
            hidden_states = model_output[0][:, source.size(1) + pad_length:-1, :]
            
            if losstype == "cosine":
                # print(input_embeds.size())
                # print(hidden_states.size())
                # input()
                hidden_states_unitnorm = torch.nn.functional.normalize(hidden_states, p=2, dim=-1).contiguous()
                pred_embs_unitnorm = torch.nn.functional.normalize(input_embeds[:, source.size(1)+pad_length+1:, :], p=2, dim=-1).contiguous()
                loss = (1.0 - (hidden_states_unitnorm * pred_embs_unitnorm).sum(dim=-1)).sum(dim=-1)
            
            elif losstype == "dot":
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds[:, source.size(1)+pad_length+1:, :].contiguous()
                loss = -(hidden_states * pred_embs).sum(dim=-1)
                # loss += torch.log(torch.exp(hidden_states.matmul(embed_lut.weight.t())).sum(dim=-1))
                loss = loss.sum(dim=-1)
                # loss = (-hidden_states * pred_embs).sum(dim=-1).sum(dim=-1)
            
            elif losstype == "detachdot":
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds[:, source.size(1)+pad_length+1:, :].contiguous()
                loss = -(hidden_states.detach() * pred_embs).sum(dim=-1)
                loss += torch.logsumexp(hidden_states.matmul(embed_lut.weight.t()), dim=-1).detach()
                # loss += torch.log(torch.exp(hidden_states.matmul(embed_lut.weight.t())).sum(dim=-1))
                loss = loss.sum(dim=-1)
                # loss = (-hidden_states * pred_embs).sum(dim=-1).sum(dim=-1)

            elif losstype == "dotplusplus":
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds[:, source.size(1)+pad_length+1:, :].contiguous()
                loss = -(hidden_states.detach() * pred_embs).sum(dim=-1)
                loss += torch.logsumexp(hidden_states.matmul(embed_lut.weight.t()), dim=-1).detach()
                # loss += torch.log(torch.exp(hidden_states.matmul(embed_lut.weight.t())).sum(dim=-1))
                loss = loss.sum(dim=-1)
                # loss = (-hidden_states * pred_embs).sum(dim=-1).sum(dim=-1)

            else:
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds[:, source.size(1)+pad_length+1:, :].contiguous()
                loss = (hidden_states - pred_embs)
                loss = (loss*loss).sum(dim=-1).sum(dim=-1)
            
            if self.args.length_normalize:
                loss = loss/hidden_states.size(1)    

            logging_output = {
                "loss": loss.data.cpu(),
                "max_length": prefix.size(1) + pred_tokens.size(1),
                "nsentences": batch_size,
                "lm_logprobs": hidden_states.data.cpu()
            }
        else:
            raise ValueError(f"wrong losstype provided: {losstype}")

        

        return loss, logging_output

    def compute_gold_loss(self, batch, **kwargs):
        '''
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        real_source, target = batch
        source = kwargs.get("additional_batch") #in STRAP model, real_source x is paraphrased to source y which is then transformed into target y. The language model sees only source and target, not the real source

        batch_size = source.size(0)

        max_prefix_length = getattr(self.args, 'max_prefix_length', source.size(1) + 1)
        pad_length = max(0, max_prefix_length - source.size(1))
        bos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.bos_token_id)
        pad = torch.empty((batch_size, pad_length)).long().to(self.device).fill_(self.pad_token_id)
        eos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.eos_token_id) 

        input_tokens = torch.cat([pad, source, bos, target, eos], dim=1)

        source_segment_id = torch.empty((batch_size, pad_length + source.size(1))).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[1])
        target_segment_id = torch.empty((batch_size, target.size(1) + 2)).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[2])
        segment = torch.cat([source_segment_id, target_segment_id], dim=1)

        losstype = getattr(self.args, "loss_type", "xentropy") 
        if losstype == "xentropy":
            model_output = self.model(input_tokens, token_type_ids=segment)
            target = torch.cat([bos, target, eos], dim=1)

            lm_logits = model_output[0][:, source.size(1)+pad_length:]
            lm_logprobs = F.log_softmax(lm_logits, dim=-1)

            loss = F.nll_loss(lm_logprobs[:,:target.size(1) - 1,:].squeeze(0), target[:, 1:target.size(1)].squeeze(0), reduction="none").unsqueeze(0).sum(dim=-1)

            if self.args.length_normalize:
                loss /= lm_logprobs.size(1)

            _, mm = lm_logprobs.max(dim=-1) # used for debugging

            logging_output = {
                "loss": loss.data.cpu(),
                "max_length": target.size(1),
                "nsentences": batch_size,
                "mm": mm,
            }
        elif losstype in ["l2", "cosine", "dot", "dotplusplus", "detachdot"]:
            model_output = self.model.transformer(input_tokens, token_type_ids=segment)
            hidden_states = model_output[0][:, source.size(1)+pad_length:]
            input_embeds = self.model.get_input_embeddings()(input_tokens)

            if losstype == "cosine":
                # print(input_embeds.size())
                # print(hidden_states.size())
                # input()
                
                hidden_states_unitnorm = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)[:, :-1, :].contiguous()
                pred_embs_unitnorm = torch.nn.functional.normalize(input_embeds[:, source.size(1)+pad_length:, :], p=2, dim=-1)[:, 1:, :].contiguous()
                loss = (1.0 - (hidden_states_unitnorm * pred_embs_unitnorm).sum(dim=-1)).sum(dim=-1)
            
            elif losstype == "dot":
                hidden_states = hidden_states[:, :-1, :].contiguous()
                pred_embs = input_embeds[:, source.size(1)+pad_length+1:, :].contiguous()

                loss = -(hidden_states * pred_embs).sum(dim=-1)
                # loss += torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t())).sum(dim=-1))
                loss = loss.sum(dim=-1)
            
            elif losstype == "dotplusplus" or losstype == "detachdot":
                hidden_states = hidden_states[:, :-1, :].contiguous()
                pred_embs = input_embeds[:, source.size(1)+pad_length+1:, :].contiguous()

                loss = -(hidden_states * pred_embs).sum(dim=-1)
                loss += torch.logsumexp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()), dim=-1)
                # loss += torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()).sum(dim=-1)))
                loss = loss.sum(dim=-1)
            

            else:
                hidden_states = hidden_states[:, :-1, :].contiguous()
                pred_embs = input_embeds[:, source.size(1)+pad_length+1:, :].contiguous()
                loss = (hidden_states - pred_embs)
                loss = (loss*loss).sum(dim=-1).sum(dim=-1)
            
            if self.args.length_normalize:
                loss = loss/(hidden_states.size(1)-1)

            logging_output = {
                "loss": loss.data.cpu(),
                "max_length": target.size(1),
                "nsentences": batch_size,
                "lm_logprobs": hidden_states.data.cpu()
            }
        else:
            raise ValueError(f"wrong losstype provided: {losstype}")

        return loss, logging_output   
    
    def generate(self, input_ids, **kwargs):
        prepared_input = self._prepare_input_for_generation(input_ids, **kwargs)
        output = self.model.generate(**prepared_input)
        # print(self.model.get_input_embeddings().weight)
        # print(str(**prepared_input))
        # print("gen", output)
        
        return self._postprocess_output(prepared_input, output)

    def _prepare_input_for_generation(self, input_ids, **kwargs):
        
        input_ids = kwargs.get('additional_ids')
        max_prefix_length = getattr(self.args, 'max_prefix_length', input_ids.size(1) + 1)
        pad_length = max(0, max_prefix_length - input_ids.size(1))
        max_output_length = kwargs.get('max_output_length', 50)
        batch_size = input_ids.size(0)
        #batch size is 1, padding and stuff needs to be modified for this to work for larger batches

        bos = torch.empty((input_ids.size(0), 1)).long().to(self.device).fill_(self.bos_token_id)
        pad = torch.empty((input_ids.size(0), pad_length)).long().to(self.device).fill_(self.tokenizer.pad_token_id)

        source_segment_length = pad_length + input_ids.size(1)
        source_segment_id = torch.empty((batch_size, source_segment_length)).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[1])
        target_segment_id = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[2])
        segment = torch.cat([source_segment_id, target_segment_id], dim=1)

        input_ids = torch.cat([pad, input_ids, bos], dim=1)
        # print("prep", input_ids)



        return_object = {'input_ids': input_ids,
                'token_type_ids': segment,
                'max_length': source_segment_length + 1 + max_output_length,
                'num_beams': self.args.beam_size,
                'source_segment_length': source_segment_length}
                # 'pad_token_id':self.eos_token_id} 
        # print(return_object)

        return return_object
    
    def _postprocess_output(self, prepared_input, output_ids):
        return output_ids[:, prepared_input['source_segment_length']+1:]
