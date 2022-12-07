import sys
import torch
import os
import json

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AddedToken

import numpy as np

# os.makeirs(sys.argv[7], exist_ok=True)

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

base_path = sys.argv[1]
binarize_labels = False
if len(sys.argv) > 12:
    binarize_labels = sys.argv[12] == "binarize_labels"

filetype = "txt"
if len(sys.argv) > 13:
    filetype = sys.argv[13]

labels = [int(label) for label in sys.argv[2].split(",")]
train_paths = []
valid_paths = []
test_paths = []
for label in labels:
    train_paths.append(open(f"{base_path}/{sys.argv[3]}_{label}.{filetype}"))
    valid_paths.append(open(f"{base_path}/{sys.argv[4]}_{label}.{filetype}"))
    test_paths.append(open(f"{base_path}/{sys.argv[5]}_{label}.{filetype}"))

def create_dataset(paths, labelses):
    texts, labels = [], []
    # print(paths)
    for i, path in enumerate(paths):
        for l in path:
            if filetype == "jsonl":
                text = json.loads(l)["text"]
            else:
                text = l.strip()
            if binarize_labels:
                label = 0
                if labelses[i] <= 2:
                    label = 0
                    texts.append(text)
                    labels.append(label)
                elif labelses[i] >= 3:
                    label = 1
                    texts.append(text)
                    labels.append(label)
            else:
                labels.append(labelses[i])
                texts.append(text)
            
    print("create_dataset", len(texts), len(labels), set(labels))
    return texts, labels
    
train_texts, train_labels = create_dataset(train_paths, labels)
val_texts, val_labels = create_dataset(valid_paths, labels)
test_texts, test_labels = create_dataset(test_paths, labels)


# tokenizer = AutoTokenizer.from_pretrained(sys.argv[6], cache_dir="hf_cache")
# config = AutoConfig.from_pretrained(sys.argv[6], cache_dur="hf_cache", num_labels=len(labels))

tokenizer_ = AutoTokenizer.from_pretrained(sys.argv[6], cache_dir="hf_cache")
if sys.argv[10] != "none":
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[10], cache_dir="hf_cache")
    tokenizer.model_max_length = min(tokenizer_.model_max_length, tokenizer.model_max_length)
else:
    tokenizer = tokenizer_
    # tokenizer = AutoTokenizer.from_pretrained(sys.argv[6], cache_dir="hf_cache")

config = AutoConfig.from_pretrained(sys.argv[6], cache_dur="hf_cache", num_labels=len(labels))
config2 = None
if sys.argv[10] != "none":  
    config2 = AutoConfig.from_pretrained(sys.argv[10], cache_dur="hf_cache", num_labels=len(labels))
    print(config2.pad_token_id)
    config2.pad_token_id = tokenizer.pad_token_id
    print(config2.pad_token_id)
    print("look above for padding")

    tokenizer_ = AutoTokenizer.from_pretrained(sys.argv[6], config=config)
    tokenizer.model_max_length = min(tokenizer_.model_max_length, tokenizer.model_max_length)

# config.n_positions = max_length
# config.max_position_embeddings = max_length


if sys.argv[8] == "krishna":
    SPECIAL_TOKENS = {
        "additional_special_tokens": ["<dense-vectors>", "<tokens>", "<verb>", "<ARG0>", "<ARG1>", "<global-dense-vectors>"],
        "pad_token": "<eos>",
        "bos_token": "<bos>",
        "eos_token": "<eos>"
    }
    print("Adding special tokens")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    config.pad_token_id = tokenizer.pad_token_id

elif sys.argv[8] == "roberta" :
    SPECIAL_TOKENS = {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"}
    print("Adding special tokens")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    config.pad_token_id = tokenizer.pad_token_id

elif sys.argv[8] == "dialogpt": 
    SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
    print("Adding special tokens")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    config.pad_token_id = tokenizer.pad_token_id

elif sys.argv[8] == "gpt2":
    SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
    config.pad_token_id = tokenizer.eos_token_id
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Adding special tokens")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    print(tokenizer)

elif sys.argv[8] == "bert":
    # SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
    # config.pad_token_id = tokenizer.eos_token_id
    # print("Adding special tokens")
    # tokenizer.add_special_tokens(SPECIAL_TOKENS)
    pass

elif sys.argv[8] == "gpt2-roberta":
    SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
    # config.pad_token_id = tokenizer.eos_token_id
    print("Adding special tokens")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

elif sys.argv[8] == "gpt2-distilbert":
    SPECIAL_TOKENS = {"pad_token": tokenizer.eos_token}
    # config.pad_token_id = tokenizer.eos_token_id
    print("Adding special tokens")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)


tokenizer.save_pretrained(f"{sys.argv[7]}/checkpoint_best")

if sys.argv[9] != "only_tokenizer":

    print("tokenizer loaded")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    print(test_encodings['input_ids'][0], len(test_encodings['input_ids'][0]))
    # input()
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)
    test_dataset = Dataset(test_encodings, test_labels)

    print("datasets loaded and tokenizer")

    if sys.argv[10] != "none":
        model = AutoModelForSequenceClassification.from_pretrained(sys.argv[10], config=config2)
        # model.resize_token_embeddings(len(tokenizer))
        if sys.argv[11] == "random":  
            embeds = model.get_input_embeddings()
            embeds.weight.data.normal_(mean=0.0, std=0.02)
            if embeds.padding_idx is not None:
                embeds.weight.data[module.padding_idx].zero_()
            model.resize_token_embeddings(len(tokenizer))

        elif sys.argv[11] == "freeze":
            model = AutoModelForSequenceClassification.from_pretrained(sys.argv[6], config=config)
            embeds = model.get_input_embeddings()
            for p in embeds.parameters():
                p.requires_grad = False
            # embeds.requires_grad=False
            model.resize_token_embeddings(len(tokenizer))
        
        elif sys.argv[11] == "freeze-project":
            embeds = model.get_input_embeddings()
            new_embeds = torch.nn.Embedding(embeds.num_embeddings, embeds.embedding_dim)
            for p in new_embeds.parameters():
                p.requires_grad = False
            # new_embeds.requires_grad = False
            new_embeds.weight.data.copy_(embeds.weight)
            print(model.device)
            config.new_n_embd = new_embeds.embedding_dim
            config.new_vocab_size = new_embeds.num_embeddings
            # if sys.argv[8] == "gpt2-roberta":
            #     config.pad_token_id = tokenizer.eos_token_id
            model_ = AutoModelForSequenceClassification.from_pretrained(sys.argv[6], config=config)
            new_embeds = torch.nn.Sequential(new_embeds, torch.nn.Linear(new_embeds.embedding_dim, model_.get_input_embeddings().embedding_dim, bias=False))
            model_.set_input_embeddings(new_embeds)
            model = model_
        
        elif sys.argv[11] == "freeze-eye":
            embeds = model.get_input_embeddings()
            new_embeds = torch.nn.Embedding(embeds.num_embeddings, embeds.embedding_dim)
            for p in new_embeds.parameters():
                p.requires_grad = False
            # new_embeds.requires_grad = False
            new_embeds.weight.data.copy_(embeds.weight)
            print(model.device)
            config.new_n_embd = new_embeds.embedding_dim
            # if sys.argv[8] == "gpt2-roberta":
            #     config.pad_token_id = tokenizer.eos_token_id
            model_ = AutoModelForSequenceClassification.from_pretrained(sys.argv[6], config=config)
            eye = torch.nn.Linear(new_embeds.embedding_dim, model_.get_input_embeddings().embedding_dim, bias=False)
            eye.weight.data.copy_(torch.eye(new_embeds.embedding_dim).data)
            new_embeds = torch.nn.Sequential(new_embeds, eye)
            model_.set_input_embeddings(new_embeds)
            model = model_
        
        elif sys.argv[11] == "freeze-vecmap":
            def learn_vecmap(X, y):
                print("computing vecmap")
                w = torch.inverse(X.t().matmul(X)).matmul(X.t()).matmul(y)
                vecmap = torch.nn.Linear(w.size(0), w.size(1), bias=False)
                print(w.size(), vecmap.weight.size())
                vecmap.weight.data.copy_(w.data.t())
                return vecmap
            
            def vocab_permutation(vocab1, vocab2):
                vocab2itos = {k:v for v,k in vocab2.items()}
                vocab2list = [vocab2itos[k] for k in range(len(vocab2itos))]

                perm1 = []
                perm2 = []
                unincluded = []
                for i, word in enumerate(vocab2list):
                    if word in vocab1:
                        perm1.append(vocab1[word])
                        perm2.append(i)
                    else:
                        unincluded.append(word)
                
                print(unincluded)
                return perm1, perm2

            embeds = model.get_input_embeddings()
            new_embeds = torch.nn.Embedding(embeds.num_embeddings, embeds.embedding_dim)
            for p in new_embeds.parameters():
                p.requires_grad = False
            # new_embeds.requires_grad = False
            new_embeds.weight.data.copy_(embeds.weight)
            print(model.device)
            config.new_n_embd = new_embeds.embedding_dim
            config.new_vocab_size = new_embeds.num_embeddings
            # if sys.argv[8] == "gpt2-roberta":
            #     config.pad_token_id = tokenizer.eos_token_id
            model_ = AutoModelForSequenceClassification.from_pretrained(sys.argv[6], config=config)
            tokenizer_ = AutoTokenizer.from_pretrained(sys.argv[6], config=config)
            perm, perm_ = vocab_permutation(tokenizer.vocab, tokenizer_.vocab)
            old_embeds = model_.get_input_embeddings()
            vecmap = learn_vecmap(new_embeds.weight[perm], old_embeds.weight[perm_])
            new_embeds = torch.nn.Sequential(new_embeds, vecmap)
            model_.set_input_embeddings(new_embeds)
            model = model_

    else:
        model =  AutoModelForSequenceClassification.from_pretrained(sys.argv[6], config=config)
        model.resize_token_embeddings(len(tokenizer))
    # model =  AutoModelForSequenceClassification.from_pretrained(sys.argv[6], config=config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    print(model.device)
    os.makedirs(sys.argv[7], exist_ok=True)

    training_args = TrainingArguments(
        output_dir=f'{sys.argv[7]}/results',          # output directory
        num_train_epochs=10,              # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=4,   # batch size for evaluation
        warmup_steps=600,
        weight_decay=0.01,               # strength of weight decay
        learning_rate=1e-5,
        logging_dir=f'{sys.argv[7]}/logs',            # directory for storing logs
        logging_steps=100,
        evaluation_strategy="steps",
        save_total_limit=1,
        eval_steps=500,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
    )

    print(training_args.n_gpu)


    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
    )



    train_result = trainer.train()

    print("training finished")

    trainer.save_model(output_dir=f"{sys.argv[7]}/checkpoint_best") 
    print("model saved")

    print("running evaluation now")

    metrics = trainer.evaluate(val_dataset)
    print("validation", metrics)
    metrics = trainer.evaluate(test_dataset)
    print("test", metrics)

