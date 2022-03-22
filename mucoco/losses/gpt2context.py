from mucoco.losses import BaseLoss, register_loss


import torch 
import torch.nn.functional as F

@register_loss("gpt2context")
class GPT2Loss(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 

        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device
        
        self.eos_token_id = self.tokenizer.eos_token_id    
        self.model.config.pad_token_id = self.model.config.eos_token_id # to remove the warning

        self.epsilon_additive = 0.0
    
    def compute_loss(self, batch, preds, **kwargs):
        '''
        batch: a tuple (source, prefix). If giving a prompt to the decoder, it can be specified using "prefix"
        preds: a tuple containing (predicted tokens, predicted embeddings, predicted probabilities), this is obtained through a forward pass on the optimizable target parameters (See utils/target.py)
        '''
        prompt, prefix = batch #prompt is the real deal, prefix can be provided as an extended prompt (generated by the model autoregressively)
        pred_tokens, pred_embeds, pred_probs = preds
        pred_probs = pred_probs[0]
        batch_size = prompt.size(0)
        step = kwargs.get("step")
        # print(kwargs.get("use_context"))
        if kwargs.get("use_context", False):
            # print("context")
            context = kwargs.get('context_batch').squeeze(1) #only one context for now
        else:
            context = torch.empty((batch_size, 0)).long().to(self.device)
        # print("context", context)
        eos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.eos_token_id) 
        # input_tokens = torch.cat([pad, source, bos, prefix, pred_tokens, eos], dim=1)

        embed_lut = self.model.get_input_embeddings()
        input_embeds = torch.cat([embed_lut(prompt), embed_lut(prefix), pred_embeds, embed_lut(context), embed_lut(eos)], dim=1)
        preflen = prompt.size(1) + prefix.size(1) 
        predlen = pred_embeds.size(1)
        suflen = context.size(1) + 1

        # source_segment_id = torch.empty((batch_size, pad_length + source.size(1))).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[1])
        # target_segment_id = torch.empty((batch_size, prefix.size(1) + pred_tokens.size(1) + 2)).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[2])
        # segment = torch.cat([source_segment_id, target_segment_id], dim=1)

        losstype = getattr(self.args, "loss_type", "xentropy")
        if losstype == "xentropy": #TODO
            model_output = self.model(inputs_embeds=input_embeds)
            lm_logits = model_output[0][:, source.size(1) + pad_length:]
            lm_logprobs = F.log_softmax(lm_logits, dim=-1)

            if prefix.size(1) > 0:
                xentropy_prefix = F.nll_loss(lm_logprobs[:,:prefix.size(1),:].squeeze(0), prefix.squeeze(0), reduction="none").sum(dim=-1)
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
            model_output = self.model.transformer(inputs_embeds=input_embeds)
            
            hidden_states = model_output[0]
            
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
            
            elif losstype == "dotplusplus" or losstype == "detachdot":
                k = kwargs.get("kweight")
                step = kwargs.get("step")           
                temperature = 1.0
                hidden_states = hidden_states[:, preflen+predlen-1:-1, :].contiguous()
                pred_embs = input_embeds[:, preflen+predlen:, :].contiguous()
                
                loss = -(hidden_states * pred_embs).sum(dim=-1) / temperature

                logits = hidden_states.matmul(embed_lut.weight.t()) / temperature
                maxlogit = logits.max(dim=-1, keepdim=True)[0]
                logits = logits - maxlogit
                additive = torch.exp((hidden_states * pred_embs).sum(dim=-1) / temperature - maxlogit.squeeze(-1)) - torch.exp((hidden_states * pred_embs.detach()).sum(dim=-1) / temperature - maxlogit.squeeze(-1))
                lognorm = (logits.exp().sum(dim=-1) + additive).log()
                lognorm = maxlogit.squeeze(-1) + lognorm 
                loss += lognorm
                
                # coeff = self.get_coeff(step, hidden_states.size(1), sched=self.coeff_schedule)
                # loss = coeff * loss - (coeff * loss).detach() + loss.detach()
                # print(coeff)
                loss = loss.sum(dim=-1)

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
        prompt, target = batch
        batch_size = prompt.size(0)
        
        if kwargs.get("use_context",False):
            context = kwargs.get('context_batch').squeeze(1) #only one context for now
        else:
            context = torch.empty((batch_size, 0)).long().to(self.device)
        
        # eos = torch.empty((batch_size, context.size(1), 1)).long().to(self.device).fill_(self.eos_token_id) 
        # input_tokens = torch.cat([prompt.unsqueeze(1).expand(-1, context.size(1), -1), target.unsqueeze(1).expand(-1, context.size(1), -1), context, eos], dim=1)

        eos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.eos_token_id) 
        input_tokens = torch.cat([prompt, target, context, eos], dim=1)
        preflen = prompt.size(1)
        predlen = target.size(1)
        suflen = context.size(1) + 1

        losstype = getattr(self.args, "loss_type", "xentropy") 
        if losstype == "xentropy":
            model_output = self.model(input_tokens)
            # target = input_tokens[:, prompt.size(1):,]
            torch.cat([target, eos], dim=1)

            lm_logits = model_output[0][:, prompt.size(1)-1:-1, :]
            lm_logprobs = F.log_softmax(lm_logits, dim=-1)

            loss = F.nll_loss(lm_logprobs.squeeze(0), target.squeeze(0), reduction="none").sum(dim=-1)
            
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
            model_output = self.model.transformer(input_tokens)
            hidden_states = model_output[0]
            input_embeds = self.model.get_input_embeddings()(input_tokens)

            if losstype == "cosine":
                # print(input_embeds.size())
                # print(hidden_states.size())
                # input()
                
                hidden_states_unitnorm = torch.nn.functional.normalize(hidden_states, p=2, dim=-1).contiguous()
                pred_embs_unitnorm = torch.nn.functional.normalize(input_embeds[:, source.size(1)+pad_length:, :], p=2, dim=-1)[:, 1:, :].contiguous()
                loss = (1.0 - (hidden_states_unitnorm * pred_embs_unitnorm).sum(dim=-1)).sum(dim=-1)
            
            elif losstype == "dot":
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds[:, source.size(1)+pad_length+1:, :].contiguous()

                loss = -(hidden_states * pred_embs).sum(dim=-1)
                # loss += torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t())).sum(dim=-1))
                loss = loss.sum(dim=-1)
            
            elif losstype == "dotplusplus" or losstype == "detachdot":
                hidden_states = hidden_states[:, preflen+predlen-1:-1].contiguous()
                pred_embs = input_embeds[:, preflen+predlen:, :].contiguous()

                loss = -(hidden_states * pred_embs).sum(dim=-1)

                logits = hidden_states.matmul(self.model.get_input_embeddings().weight.t())
                maxlogit = logits.max(dim=-1, keepdim=True)[0]
                logits = logits - maxlogit
                additive = torch.exp(hidden_states * pred_embs).sum(dim=-1) - torch.exp(hidden_states * pred_embs).sum(dim=-1).detach()
                lognorm = (logits.exp().sum(dim=-1) + additive).log()
                lognorm = maxlogit.squeeze(-1) + lognorm 

                loss += lognorm
                loss = loss.sum(dim=-1)
                # print(loss)
                # print(hidden_states.matmul(self.model.get_input_embeddings().weight.t()))
                # print(torch.logsumexp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()), dim=-1))
                # print(torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t())).sum(dim=-1)))
                # input("gold")
                # loss += torch.logsumexp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()), dim=-1)
                # # print()
                # # loss += torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()).sum(dim=-1)))
                # loss = loss.sum(dim=-1)
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
        max_output_length = getattr(self.args, "max_output_length", 10)
        batch_size = input_ids.size(0)
        #batch size is 1, padding and stuff needs to be modified for this to work for larger batches

        return_object = {'input_ids': input_ids,
                'max_length': input_ids.size(1) + max_output_length,
                'do_sample': True,
                'temperature': self.args.AR_temperature,
                'top_k': self.args.AR_top_k,
                'top_p': self.args.AR_top_p}
                # 'pad_token_id':self.eos_token_id} 
        # print(return_object)

        return return_object
    
    def _postprocess_output(self, prepared_input, output_ids):
        return output_ids[:, prepared_input['input_ids'].size(1):, ]