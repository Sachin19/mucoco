from mucoco.losses import BaseLoss, register_loss


import torch 
import torch.nn.functional as F

@register_loss("bart")
class BARTLoss(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 

        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device
        
        self.eos_token_id = self.tokenizer.eos_token_id    
        self.bos_token_id = self.tokenizer.bos_token_id    
    
    def compute_loss(self, batch, preds, **kwargs):
        '''
        batch: a tuple (source, prefix). If giving a prompt to the decoder, it can be specified using "prefix"
        preds: a tuple containing (predicted tokens, predicted embeddings, predicted probabilities), this is obtained through a forward pass on the optimizable target parameters (See utils/target.py)
        '''
        source, prefix = batch
        batch_size = source.size(0)

        pred_tokens, pred_embeds, pred_probs = preds
        pred_probs = pred_probs[0]

        bos = torch.empty((source.size(0), 1)).long().to(self.device).fill_(self.bos_token_id)
        eos = torch.empty((source.size(0), 1)).long().to(self.device).fill_(self.eos_token_id)
        target_input_tokens = torch.cat([eos, bos, prefix, pred_tokens, eos], dim=1)

        embed_lut = self.model.get_decoder().get_input_embeddings()
        target_input_embeds = torch.cat([embed_lut(eos), embed_lut(bos), embed_lut(prefix), pred_embeds, embed_lut(eos)], dim=1)
        scaled_target_input_embeds = target_input_embeds * kwargs["embed_scale"]

        losstype = getattr(self.args, "loss_type", "xentropy")
        if losstype == "xentropy":
            model_output = self.model(input_ids=source, decoder_inputs_embeds=scaled_target_input_embeds[:, :-1])

            lm_logits = model_output.logits
            lm_logprobs = F.log_softmax(lm_logits, dim=-1)

            if prefix.size(1) > 0:
                xentropy_prefix = F.nll_loss(lm_logprobs[:,:prefix.size(1),:].squeeze(0), prefix.squeeze(0), reduction="none").sum(dim=-1)
            else:
                xentropy_prefix = 0.0
            
            xentropy_pred = (-lm_logprobs[:, prefix.size(1) + 1: -1, :] * pred_probs).sum(dim=-1) # + 1 because of extra eos in the front
            xentropy_pred = xentropy_pred.sum(dim=-1)
            xentropy_pred = xentropy_pred - lm_logprobs[:, -1, self.eos_token_id]

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

        elif losstype in ["dot", "dotplusplus", "detachdot"]:
            model_output = self.model(input_ids=source, decoder_inputs_embeds=scaled_target_input_embeds[:, :-1], go_inside="model")
            hidden_states = model_output[0]
            input_embeds = target_input_embeds[:, 1:]
            final_logits_biases = self.model.final_logits_bias[0, target_input_tokens[:, 1:]] #needs to be a part of the trainable parameters maybe?
            
            if losstype == "dot":
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds.contiguous()

                loss = -(hidden_states * pred_embs).sum(dim=-1) - final_logits_biases
                # print(loss.size())
                # loss += torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t())).sum(dim=-1))
                loss = loss.sum(dim=-1)
            
            elif losstype == "detachdot":
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds.contiguous()

                loss = -(hidden_states.detach() * pred_embs).sum(dim=-1) - final_logits_biases

                logits = hidden_states.matmul(self.model.get_decoder().get_input_embeddings().weight.t())
                logits = logits + self.model.final_logits_bias
                lognorm = torch.logsumexp(logits, dim=-1).detach()
                loss += lognorm

                loss = loss.sum(dim=-1)
            
            elif losstype == "dotplusplus":
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds.contiguous()

                loss = -(hidden_states * pred_embs).sum(dim=-1) - final_logits_biases

                # logits = hidden_states.matmul(self.model.get_decoder().get_input_embeddings().weight.t())
                # logits = logits + self.model.final_logits_bias                
                # lognorm = torch.logsumexp(logits, dim=-1)
                # loss += lognorm

                # loss = loss.sum(dim=-1)

                # k = kwargs.get("kweight")
                # step = kwargs.get("step")
                
                # self.begintemp = 1.0
                # self.finaltemp = 0.9
                # self.r = pow(self.finaltemp/self.begintemp, 1/19)

                # temperature = max(self.finaltemp, self.begintemp * pow(self.r, step))
                temperature = 1.0
                loss = -(hidden_states * pred_embs).sum(dim=-1) / temperature

                logits = (hidden_states.matmul(embed_lut.weight.t()) + self.model.final_logits_bias) / temperature
                maxlogit = logits.max(dim=-1, keepdim=True)[0]
                logits = logits - maxlogit
                additive = torch.exp(((hidden_states * pred_embs).sum(dim=-1) + final_logits_biases) / temperature - maxlogit.squeeze(-1)) - torch.exp(((hidden_states * pred_embs.detach()).sum(dim=-1) + final_logits_biases) / temperature - maxlogit.squeeze(-1))
                lognorm = (logits.exp().sum(dim=-1) + additive).log()
                lognorm = maxlogit.squeeze(-1) + lognorm 
                
                # coeff = min(1.0, (1.0*step)/predlen)
                # loss += coeff * hidden_contribution #+ (1 - coeff) * hidden_contribution.detach()
                loss += lognorm
                loss = loss.sum(dim=-1)

            
            if self.args.length_normalize:
                loss = loss/hidden_states.size(1)
            
            # print(loss)
            # input()

            logging_output = {
                "loss": loss.data.cpu(),
                "max_length": target_input_embeds.size(1),
                "nsentences": batch_size,
                "lm_logprobs": hidden_states.data.cpu()
            }

        return loss, logging_output

    def compute_gold_loss(self, batch, **kwargs):
        '''
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        print("ok2")
        source, target = batch
        batch_size = source.size(0)
        
        # if kwargs.get("use_context",False):
        #     context = kwargs.get('context_batch').squeeze(1) #only one context for now
        # else:
        # context = torch.empty((batch_size, 0)).long().to(self.device)
        
        eos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.eos_token_id) 
        bos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.bos_token_id) 
        target_input_tokens = torch.cat([eos, bos, target, eos], dim=1)

        losstype = getattr(self.args, "loss_type", "xentropy") 
        if losstype == "xentropy":
            model_output = self.model(input_ids=source, decoder_input_ids=target_input_tokens[:, :-1], labels=target_input_tokens[:, 1:])

            lm_logits = model_output.logits
            lm_logprobs = F.log_softmax(lm_logits, dim=-1)

            loss = F.nll_loss(lm_logprobs.squeeze(0), target_input_tokens[:, 1:].squeeze(0), reduction="none")

            print(loss)
            loss = loss.sum(dim=-1)

            if self.args.length_normalize:
                loss /= lm_logprobs.size(1)
            
            _, mm = lm_logprobs.max(dim=-1)

            logging_output = {
                "loss": loss.data.cpu(),
                "max_length": target.size(1),
                "nsentences": batch_size,
            }
        elif losstype in ["l2", "cosine", "dot", "dotplusplus", "detachdot"]:
            # model_output = self.model.model(input_ids=source, decoder_input_ids=target_input_tokens[:, :-1])
            model_output = self.model(input_ids=source, decoder_input_ids=target_input_tokens[:, :-1], go_inside="model")
            hidden_states = model_output[0]
            
            input_embeds = self.model.get_decoder().get_input_embeddings()(target_input_tokens[:, 1:])

            final_logits_biases = self.model.final_logits_bias[0, target_input_tokens[:, 1:]]
            # print(final_logits_biases.size())
            
            if losstype == "dot":
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds.contiguous()

                loss = -(hidden_states * pred_embs).sum(dim=-1) - final_logits_biases
                # print(loss.size())
                # loss += torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t())).sum(dim=-1))
                loss = loss.sum(dim=-1)
            
            else:
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds.contiguous()
                # print(hidden_states.size())
                # print(pred_embs.size())
                # print(final_logits_biases.size())
                # input()
                loss = -(hidden_states * pred_embs).sum(dim=-1) - final_logits_biases
                logits = hidden_states.matmul(self.model.get_decoder().get_input_embeddings().weight.t())
                # print(logits.size())
                logits = logits + self.model.final_logits_bias
                loss += torch.log(torch.exp(logits).sum(dim=-1))

                # print(loss)
                loss = loss.sum(dim=-1)
            
            if self.args.length_normalize:
                loss = loss/hidden_states.size(1)

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
        print("ok")
        prepared_input = self._prepare_input_for_generation(input_ids, **kwargs)
        output = self.model.generate(**prepared_input)
        # print(self.model.get_input_embeddings().weight)
        # print(str(**prepared_input))
        # print("gen", output)
        
        return self._postprocess_output(prepared_input, output)

    def _prepare_input_for_generation(self, input_ids, **kwargs):
        # max_output_length = getattr(self.args, "max_output_length", 10)
        batch_size = input_ids.size(0)
        #batch size is 1, padding and stuff needs to be modified for this to work for larger batches

        return_object = {'input_ids': input_ids, 'num_return_sequences': 1}#kwargs.get('num_return_sequences', 1)}

        return return_object
    
    def _postprocess_output(self, prepared_input, output_ids):
        return output_ids
