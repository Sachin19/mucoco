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

        with tokenizer.as_target_tokenizer():
            print(self.tokenizer.eos_token)
            self.pad_token_id = self.tokenizer.pad_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
    
    def compute_loss(self, batch, preds, **kwargs):
        '''
        batch: a tuple (source, prefix). If giving a prompt to the decoder, it can be specified using "prefix"
        preds: a tuple containing (predicted tokens, predicted embeddings, predicted probabilities), this is obtained through a forward pass on the optimizable target parameters (See utils/target.py)
        '''
        source, prefix = batch
        batch_size = source.size(0)

        pred_tokens, pred_embeds, pred_probs = preds
        pred_probs = pred_probs[0]
        # print(pred_tokens)

        bos = torch.empty((source.size(0), 1)).long().to(self.device).fill_(self.pad_token_id)
        eos = torch.empty((source.size(0), 1)).long().to(self.device).fill_(self.eos_token_id)
        # print(bos)
        # print(eos)
        # input()

        target_input_tokens = torch.cat([bos, prefix, pred_tokens, eos], dim=1)
        # print(target_input_tokens)

        embed_lut = self.model.get_decoder().get_input_embeddings()
        target_input_embeds = torch.cat([embed_lut(bos), embed_lut(prefix), pred_embeds, embed_lut(eos)], dim=1)
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
            
            xentropy_pred = (-lm_logprobs[:, prefix.size(1): -1, :] * pred_probs).sum(dim=-1) 
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

        elif losstype in ["dot", "dotplusplus"]:
            # model_output = self.model(input_ids=source, decoder_inputs_embeds=scaled_target_input_embeds[:, :-1])

            # lm_logits = model_output.logits
            # print((-lm_logits[:, :-1] * pred_probs).sum(dim=-1))
            # lm_logprobs = F.log_softmax(lm_logits, dim=-1)

            # if prefix.size(1) > 0:
            #     xentropy_prefix = F.nll_loss(lm_logprobs[:,:prefix.size(1),:].squeeze(0), prefix.squeeze(0), reduction="none").sum(dim=-1)
            # else:
            #     xentropy_prefix = 0.0
            
            # xentropy_pred = (-lm_logprobs[:, prefix.size(1):-1, :] * pred_probs).sum(dim=-1) 
            # print(xentropy_pred)
            # xentropy_pred = xentropy_pred.sum(dim=-1)
            # xentropy_pred = xentropy_pred - lm_logprobs[:, -1, self.eos_token_id]
            # print(- lm_logprobs[:, -1, self.eos_token_id])
            # _, mm = lm_logprobs.max(dim=-1)

            # xentropy = xentropy_pred + xentropy_prefix 
            # if self.args.length_normalize:
            #     xentropy /= lm_logprobs.size(1)
            
            # print(xentropy)
            # print("xentropy ends")


            model_output = self.model(input_ids=source, decoder_inputs_embeds=scaled_target_input_embeds[:, :-1], go_inside="model")
            hidden_states = model_output[0]
            input_embeds = target_input_embeds[:, 1:]
            final_logits_biases = self.model.final_logits_bias[0, target_input_tokens[:, 1:]]
            # print(final_logits_biases)
            # print(self.model.final_logits_bias[0] * pred_probs[0])
            # print("after")
            
            if losstype == "dot":
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds.contiguous()

                loss = -(hidden_states * pred_embs).sum(dim=-1) - final_logits_biases
                # print(loss.size())
                # loss += torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t())).sum(dim=-1))
                loss = loss.sum(dim=-1)
            
            else:
                hidden_states = hidden_states.contiguous().detach()
                pred_embs = input_embeds.contiguous()

                # output_embs = (pred_probs.unsqueeze(-1) * self.model.lm_head.weight).sum(dim=-2)
                # print("embeds")
                # print(pred_embs[:, :-1]
                # print(output_embs)

                loss = -(hidden_states * pred_embs).sum(dim=-1) - final_logits_biases
                # print((hidden_states * pred_embs).sum(dim=-1))
                # print((hidden_states[:, :-1] * output_embs).sum(dim=-1))
                # print(-loss)
                # print(((self.model.lm_head(hidden_states)) [:, :-1]* pred_probs).sum(dim=-1))
                # print(((self.model.lm_head(hidden_states)+self.model.final_logits_bias) [:, :-1]* pred_probs).sum(dim=-1))
                # print("here")
                logits = hidden_states.matmul(self.model.get_decoder().get_input_embeddings().weight.t())
                logits = logits + self.model.final_logits_bias
                # print(logits)
                # print(-logits[:, :-1] * pred_probs)
                # print((-logits[:, :-1] * pred_probs).sum(dim=-1))
                # deno = torch.log(torch.exp(logits).sum(dim=-1))
                deno = torch.logsumexp(logits, dim=-1)
                loss += deno
                # print(loss)
                loss = loss.sum(dim=-1)
            
            # if self.args.length_normalize:
            loss_ = loss/hidden_states.size(1)

            loss = loss - loss.detach() + loss_.detach()
            
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
        source, target = batch
        batch_size = source.size(0)
        bos = torch.empty((source.size(0), 1)).long().to(self.device).fill_(self.pad_token_id)
        eos = torch.empty((source.size(0), 1)).long().to(self.device).fill_(self.eos_token_id)    
        target_input_tokens = torch.cat([bos, target, eos], dim=1)

        losstype = getattr(self.args, "loss_type", "xentropy") 
        if losstype == "xentropy":

            model_output = self.model(input_ids=source, decoder_input_ids=target_input_tokens[:, :-1], labels=target_input_tokens[:, 1:])

            lm_logits = model_output.logits
            lm_logprobs = F.log_softmax(lm_logits, dim=-1)

            loss = F.nll_loss(lm_logprobs.squeeze(0), target_input_tokens[:, 1:].squeeze(0), reduction="none")

            # print(loss)
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
            
        elif losstype in ["dot", "dotplusplus"]:
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

                loss = -(hidden_states * pred_embs).sum(dim=-1) - final_logits_biases
                # logits = hidden_states.matmul(self.model.get_decoder().get_input_embeddings().weight.t())
                # print(logits.size())
                # logits = logits + self.model.final_logits_bias
                # loss += torch.log(torch.exp(logits).sum(dim=-1))

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
        prepared_input = self._prepare_input_for_generation(input_ids, **kwargs)
        output = self.model.generate(**prepared_input)
        # print(self.model.get_input_embeddings().weight)
        # print(str(prepared_input))
        # print("gen", output)
        
        return self._postprocess_output(prepared_input, output)

    def _prepare_input_for_generation(self, source, **kwargs):
        
        # source = kwargs.get('additional_ids')
        # max_prefix_length = getattr(self.args, 'max_prefix_length', source.size(1) + 1)
        # pad_length = max(0, max_prefix_length - source.size(1))
        max_output_length = kwargs.get('max_output_length', 50)
        batch_size = source.size(0)
        #batch size is 1, padding and stuff needs to be modified for this to work for larger batches

        # bos = torch.empty((source.size(0), 1)).long().to(self.device).fill_(self.pad_token_id)
        # pad = torch.empty((source.size(0), pad_length)).long().to(self.device).fill_(self.tokenizer.pad_token_id)

        # source_segment_length = pad_length + source.size(1)
        # source_segment_id = torch.empty((batch_size, source_segment_length)).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[1])
        # target_segment_id = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.tokenizer.additional_special_tokens_ids[2])
        # segment = torch.cat([source_segment_id, target_segment_id], dim=1)

        # source = torch.cat([source, bos], dim=1)
        # print("prep", input_ids)



        return_object = {'input_ids': source,
                # 'token_type_ids': segment,
                'max_length': max_output_length,
                'num_beams': kwargs.get('num_beams', 1),
                'num_return_sequences': kwargs.get('num_return_sequences', 1)
                # 'source_segment_length': source_segment_length}
                # 'pad_token_id':self.eos_token_id
                } 
        # print(return_object)

        return return_object
    
    def _postprocess_output(self, prepared_input, output_ids):
        return output_ids