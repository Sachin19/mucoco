import sys
import torch
import os
import json

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AddedToken, DataCollatorWithPadding
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

import torch.nn.functional as F
import numpy as np
import click

def compute_metrics(p): #redefine
    # print(p.predictions)
    # print(p.predictions.size())
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # print(type(preds))
    preds = np.argmax(preds, axis=1)
    # print(preds)
    # print(type(preds))
    # print(preds == p.label_ids)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

@click.command()
@click.option('--data-dir', required=True, type=str)
@click.option('--label-names', nargs=2, required=True, type=str)
@click.option('--label-ids', nargs=2, required=True, type=int)
@click.option('--train', required=True, type=str)
@click.option('--dev', required=True, type=str)
@click.option('--test', required=True, type=str)
@click.option('--filetype', required=False, type=str, help="how to reconcile different embeddings (project/freeze/..")
@click.option('--output-dir', required=True, type=str)
@click.option('--base-model', required=True, type=str)
@click.option('--block-size', required=False, default=128, type=int)
@click.option('--tokenizer-update-strategy', required=True, type=str)
@click.option('--secondary-model', default="none", required=False, type=str, help="model whose embeddings to use")
@click.option('--reconciliation-strategy', required=False, type=str, help="how to reconcile different embeddings (project/freeze/..")
def main(data_dir, label_names, label_ids, train, dev, test, filetype, output_dir, base_model, block_size, tokenizer_update_strategy, secondary_model, reconciliation_strategy):
    print("hello there")
    # label_names = label_names.split(";")
    # label_ids = [int(label_id) for label_id in label_ids]
    assert(len(label_ids) == len(label_names))
    # labelid2tok = {label_id:name for zip(label_ids, label_names)}
    train_paths = []
    valid_paths = []
    test_paths = []
    for label in label_ids:
        train_paths.append(open(f"{data_dir}/{train}_{label}.{filetype}"))
        valid_paths.append(open(f"{data_dir}/{dev}_{label}.{filetype}"))
        test_paths.append(open(f"{data_dir}/{test}_{label}.{filetype}"))

    def create_dataset(paths, labelses):
        texts, labels = [], []
        # print(paths)
        for i, path in enumerate(paths):
            for l in path:
                if filetype == "jsonl":
                    text = json.loads(l)["text"]
                else:
                    text = l.strip()
                labels.append(labelses[i])
                texts.append(text)
                
        print("create_dataset", len(texts), len(labels), set(labels))
        return texts, labels
    
    train_texts, train_labels = create_dataset(train_paths, label_ids)
    traindataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_texts, val_labels = create_dataset(valid_paths, label_ids)
    valdataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_texts, test_labels = create_dataset(test_paths, label_ids)
    testdataset = Dataset.from_dict({"text": test_texts, "label": test_labels})


    tokenizer_ = AutoTokenizer.from_pretrained(base_model, cache_dir="hf_cache")
    if secondary_model != "none":
        tokenizer = AutoTokenizer.from_pretrained(secondary_model, cache_dir="hf_cache")
        tokenizer.model_max_length = min(tokenizer_.model_max_length, tokenizer.model_max_length)
    else:
        tokenizer = tokenizer_
        # tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir="hf_cache")

    config = AutoConfig.from_pretrained(base_model, cache_dur="hf_cache")
    config2 = None
    if secondary_model != "none":  
        config2 = AutoConfig.from_pretrained(secondary_model, cache_dur="hf_cache")
        print(config2.pad_token_id)
        config2.pad_token_id = tokenizer.pad_token_id
        print(config2.pad_token_id)
        print("look above for padding")

        tokenizer_ = AutoTokenizer.from_pretrained(base_model, config=config)
        tokenizer.model_max_length = min(tokenizer_.model_max_length, tokenizer.model_max_length)

    # config.n_positions = max_length
    # config.max_position_embeddings = max_length
    
    elif tokenizer_update_strategy == "roberta" :
        SPECIAL_TOKENS = {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"}
        print("Adding special tokens")
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        config.pad_token_id = tokenizer.pad_token_id

    elif tokenizer_update_strategy == "dialogpt": 
        SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
        print("Adding special tokens")
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        config.pad_token_id = tokenizer.pad_token_id

    elif tokenizer_update_strategy == "gpt2":
        SPECIAL_TOKENS = {"pad_token": "[PAD]"}
        config.pad_token_id = tokenizer.eos_token_id
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Adding special tokens")
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        print(tokenizer)

    elif tokenizer_update_strategy == "bert":
        # SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
        # config.pad_token_id = tokenizer.eos_token_id
        # print("Adding special tokens")
        # tokenizer.add_special_tokens(SPECIAL_TOKENS)
        pass

    elif tokenizer_update_strategy == "gpt2-roberta":
        SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
        # config.pad_token_id = tokenizer.eos_token_id
        print("Adding special tokens")
        tokenizer.add_special_tokens(SPECIAL_TOKENS)


    tokenizer.save_pretrained(f"{output_dir}/checkpoint_best")

    labelid2tok = {label_id:tokenizer.encode(label_name, add_special_tokens=False)[0] for (label_id, label_name) in zip(label_ids, label_names)} 
    def preprocess_function(examples):
        return tokenizer(examples["text"], max_length=block_size, truncation=True)    

    traindataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    valdataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    testdataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    raw_dataset = DatasetDict({"train": traindataset, "validation": valdataset, "test": testdataset})
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")
    # print(tokenized_dataset['train'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    print("datasets and tokenizer loaded")

    if secondary_model != "none":
        model = AutoModelForCausalLM.from_pretrained(secondary_model, config=config2)
        # model.resize_token_embeddings(len(tokenizer))
        if sys.argv[11] == "random":  
            embeds = model.get_input_embeddings()
            embeds.weight.data.normal_(mean=0.0, std=0.02)
            if embeds.padding_idx is not None:
                embeds.weight.data[module.padding_idx].zero_()
            model.resize_token_embeddings(len(tokenizer))

        elif sys.argv[11] == "freeze":
            model = AutoModelForCausalLM.from_pretrained(base_model, config=config)
            embeds = model.get_input_embeddings()
            for p in embeds.parameters():
                p.requires_grad = False
            # embeds.requires_grad=False
            model.resize_token_embeddings(len(tokenizer))
    else:
        model =  AutoModelForCausalLM.from_pretrained(base_model, config=config)
        model.resize_token_embeddings(len(tokenizer))
    # model =  AutoModelForCausalLM.from_pretrained(base_model, config=config)
    print(model.device)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()

    def process_batch(batch):
        batch = {k: v.to(device) for k, v in batch.items()}
        # print(batch.keys())
        # print(batch.values())
        input_ids = batch['input_ids']
        seq_a = (torch.ones(input_ids.shape[0])*labelid2tok[0]).type_as(input_ids).view(-1,1)
        seq_b = (torch.ones(input_ids.shape[0])*labelid2tok[1]).type_as(input_ids).view(-1,1)
        seq_a = torch.cat((seq_a, input_ids), dim=1)[:,:-1]
        seq_b = torch.cat((seq_b, input_ids), dim=1)[:,:-1]
        input_ids = torch.cat((seq_a,seq_b),dim=0)
        # print(input_ids)

        attention_mask = batch['attention_mask']
        seq_a = (torch.ones(attention_mask.shape[0])).type_as(attention_mask).view(-1,1)
        seq_b = (torch.ones(attention_mask.shape[0])).type_as(attention_mask).view(-1,1)
        seq_a = torch.cat((seq_a, attention_mask), dim=1)[:,:-1]
        seq_b = torch.cat((seq_b, attention_mask), dim=1)[:,:-1]
        attention_mask = torch.cat((seq_a, seq_b),dim=0)
        # print(attention_mask)

        labels = batch['labels']

        batch['input_ids'] = input_ids
        batch['attention_mask'] = attention_mask
        bsz = input_ids.size(0)
        del batch['labels']

        return batch, labels, bsz

    class GeDiTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):
            batch, labels, bsz = process_batch(inputs)
            # implement custom logic here
            outputs = model(**batch)
            
            logits = outputs.logits
            shift_logprobs = F.log_softmax(logits[..., :-1, :], dim=-1).contiguous()
            shift_target = batch['input_ids'][..., 1:].contiguous()
            nll = F.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(bsz, -1)
            # print(shift_target.ne(tokenizer.pad_token_id).float().sum(dim=-1))
            # input()
            nll = nll.sum(dim=-1) / shift_target.ne(tokenizer.pad_token_id).float().sum(dim=-1)         
            outputs = nll.view(2, -1)

            # print(outputs.size())
            num = outputs.gather(0, labels.view(1, -1))
            # print(num.size())
            # input()
            deno = torch.logsumexp(-outputs, dim=0)

            loss = num + deno
            loss = loss.sum()
            return (loss, outputs) if return_outputs else loss
        
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
            inputs = self._prepare_input(inputs)
            with torch.no_grad():
                batch, labels, bsz = process_batch(inputs)
                # implement custom logic here
                outputs = model(**batch)

                logits = outputs.logits
                # print(logits)
                shift_logprobs = F.log_softmax(logits[..., :-1, :], dim=-1).contiguous()
                shift_target = batch['input_ids'][..., 1:].contiguous()
                nll = F.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(bsz, -1)

                # print(nll)
                nll = nll.sum(dim=-1)         
                nlogits = nll.view(2, -1)

                # print("nl", nlogits)
                num = nlogits.gather(0, labels.view(1, -1))
                deno = torch.logsumexp(-nlogits, dim=0)

                loss = num + deno
                loss = loss.sum()

            if prediction_loss_only:
                return (loss, None, None)
            
            return (loss, -nlogits.transpose(1, 0), labels)

    training_args = TrainingArguments(
        output_dir=f'{sys.argv[7]}/results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=4,   # batch size for evaluation
        logging_dir=f'{sys.argv[7]}/logs',            # directory for storing logs
        logging_steps=100,
        evaluation_strategy="steps",
        save_total_limit=1,
        eval_steps=50,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        gradient_accumulation_steps=16,
        load_best_model_at_end=True,
    )

    print(training_args.n_gpu)


    trainer = GeDiTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=tokenized_dataset['train'],         # training dataset
        eval_dataset=tokenized_dataset['validation'],            # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )



    train_result = trainer.train()

    print("training finished")

    trainer.save_model(output_dir=f"{output_dir}/checkpoint_best") 
    # trainer.save_pretrained(f"{output_dir}/checkpoint_best")
    print("model saved")

    print("running evaluation now")

    metrics = trainer.evaluate(tokenized_dataset['validation'])
    print("validation", metrics)
    metrics = trainer.evaluate(tokenized_dataset['test'])
    print("test", metrics)

if __name__ == "__main__":
    main()

