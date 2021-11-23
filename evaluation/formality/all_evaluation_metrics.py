import logging
import math
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import random
import ot

from transformers import AutoTokenizer, AutoModel, AutoConfig

import mucoco.losses as lossbuilder
import mucoco.options as options

import bert_score
from evaluation.similarity.test_sim import find_similarity as weiting_similarity_fn

from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

def detokenize(x):
    x = x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )", ")").replace("( ", "(")
    return x

def plot(scores, fname="hist.png", bins=100):
    plt.clf()
    plt.hist(scores, bins=bins)
    plt.savefig(fname)

def transfer_classify(input1, model):
    def label_fn(label):
        return model.task.label_dictionary.string(
            [label + model.task.target_dictionary.nspecial]
        )
    
    input1 = [model.bpe.encode(detokenize(sd)) for sd in input1]
    batch = collate_tokens(
        [model.task.source_dictionary.encode_line("<s> " + sd + " </s>", append_eos=False) for sd in input1], pad_idx=1
    )
    batch = batch[:, :512]

    with torch.no_grad():
        predictions = model.predict('sentence_classification_head', batch.long())

    prediction_labels = [label_fn(x.argmax(axis=0).item()) for x in predictions]
    return prediction_labels

def fluency_classify(input1, model):
    def label_fn(label):
        return model.task.label_dictionary.string(
            [label + model.task.target_dictionary.nspecial]
        )

    input1s = [model.bpe.encode(detokenize(inp)) for inp in input1]
    batch = collate_tokens(
        [model.task.source_dictionary.encode_line("<s> " + sd + " </s>", append_eos=False) for sd in input1s], pad_idx=1
    )
    batch = batch[:, :512]

    with torch.no_grad():
        predictions = model.predict('sentence_classification_head', batch.long())

    # prediction_probs = [torch.exp(x).max(axis=0)[0].item() for x in predictions]
    prediction_labels = [label_fn(x.argmax(axis=0).item()) for x in predictions]
    return prediction_labels

    # ncorrect += sum([1 if l1.lower() == l2.lower() else 0 for l1, l2 in zip(prediction_labels, lds)])

    # nsamples += len(sds)

    # for sd, ld, pld, ppd in zip(sds, lds, prediction_labels, prediction_probs):
    #     sd1 = sd.strip()
    #     sd1 = sd1.replace("<unk>", unk_bpe).strip()
    #     argmax_results.append(pld.lower())
    #     prediction_data[ld.lower()].append({
    #         "sentence": model.bpe.decode(sd1),
    #         "prediction": pld.lower(),
    #         "prediction_prob": ppd,
    #         "correct": ld.lower() == pld.lower()
    #     })

def wieting_sim(input1, input2, roberta):
    # input1 = [roberta.bpe.decode(x) for x in input1]
    # input2 = [roberta.bpe.decode(x) for x in input2]
    return weiting_similarity_fn(input1, input2)
    

def wmd(input1, input2, embed_lut, dist="cosine"):
    input1_embs = embed_lut(input1)
    input2_embs = embed_lut(input2)

    if dist == "cosine":
        input1_embs = F.normalize(input1_embs, p=2, dim=-1)
        input2_embs = F.normalize(input2_embs, p=2, dim=-1)
        pairwise_distance = 1. - (input1_embs.unsqueeze(2) * input2_embs.unsqueeze(1)).sum(dim=-1)
    else:
        pairwise_distance = (input1_embs.unsqueeze(2) - input2_embs.unsqueeze(1))
        pairwise_distance = torch.sqrt((pairwise_distance * pairwise_distance).sum(dim=-1))

    a = np.ones((input1_embs.size(1),))/input1_embs.size(1)
    b = np.ones((input2_embs.size(1),))/input2_embs.size(1)
    
    M = pairwise_distance.data.cpu().numpy()
    allT = []
    alld = []
    for i in range(input1.size(0)):
        T = ot.emd(a, b, M[i])
        d = np.sum(T * M[i])
        allT.append(T)
        alld.append(d)
    
    allT = np.concatenate(allT, axis=0)

    return alld

def bertscore(input_texts1, input_texts2, scorer):
    return scorer.score(input_texts1, input_texts2)[-1].tolist()

def moverscore(input1, input2, model, dist="cosine"):
    input1_features = model(input_ids=input1)[0] # get all embeddings
    input2_features = model(input_ids=input2)[0]  # get all embeddings
    
    if dist == "cosine":
        input1_features = F.normalize(input1_features, p=2, dim=-1)
        input2_features = F.normalize(input2_features, p=2, dim=-1)
        pairwise_distance = 1. - (input1_features.unsqueeze(2) * input2_features.unsqueeze(1)).sum(dim=-1)
    else:
        pairwise_distance = (input1_features.unsqueeze(2) - input2_features.unsqueeze(1))
        pairwise_distance = torch.sqrt((pairwise_distance * pairwise_distance).sum(dim=-1))
    
    a = np.ones((input1.size(1),))/input1.size(1)
    b = np.ones((input2.size(1),))/input2.size(1)
    
    M = pairwise_distance.data.cpu().numpy()
    allT = []
    alld = []
    for i in range(input1.size(0)):
        T = ot.emd(a, b, M[i])
        d = np.sum(T * M[i])
        allT.append(T)
        alld.append(d)
    
    allT = np.concatenate(allT, axis=0)

    return alld
    #TODO: compute EMD


def cls_similarity(input1, input2, model, metric="cosine"):
    input1_features = model(input_ids=input1)[0][:, 0, :] #CLS token representation
    input2_features = model(input_ids=input2)[0][:, 0, :]  #CLS token representation
    
    if metric=="cosine":
        sim = (F.normalize(input1_features, dim=-1, p=2) * F.normalize(input2_features, dim=-1, p=2)).sum(dim=-1)
    else:
        diff = (input1_features - input2_features)
        sim = -(diff * diff).sum(dim=-1).sum(dim=-1)

    return sim.tolist()

def sts_similarity(input1, input2, model):
    input1_features = mean_pooling(model(input_ids=input1), attention_mask=torch.ones(input1.size(0), input1.size(1)).to(model.device))
    input2_features = mean_pooling(model(input_ids=input2), attention_mask=torch.ones(input2.size(0), input2.size(1)).to(model.device))

    sim = (F.normalize(input1_features, dim=-1, p=2) * F.normalize(input2_features, dim=-1, p=2)).sum(dim=-1)
    return sim.tolist()

#Mean Pooling for content loss- Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def main(args):
    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(args.results_path, f"generate-{args.gen_subset}.txt")
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def get_ppl(input, model, eos=-1):
    if eos == -1:
        model_output = model(input_ids=input[:, :-1])
        lm_logits = model_output[0]
        lm_logprobs = F.log_softmax(lm_logits, dim=-1)

        # print(lm_logprobs.size())
        # print(input[:, 1:].size())
        nll = F.nll_loss(
            lm_logprobs.squeeze(0), input[:, 1:].squeeze(0), reduction="none"
        )

        return nll.sum()
    else:
        model_output = model(input_ids=input)
        lm_logits = model_output[0]
        lm_logprobs = F.log_softmax(lm_logits, dim=-1)

        nll = F.nll_loss(
            lm_logprobs[:, :-1, :].squeeze(0), input[:, 1:].squeeze(0), reduction="none"
        )
        print(nll.size())
        nll -= lm_logprobs[:, -1, eos]
        return nll.sum()


def _main(args, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )

    logger = logging.getLogger("__main__")
    logger.info(args)
    
    use_cuda = torch.cuda.is_available() and not args.cpu

    evaluation_metrics = set(args.evaluation_metrics.split(","))

    data_paths = args.data.split(",")
    if len(data_paths) == 1:
        source_data = data_paths[0]
        target_datas = [data_paths[0]]
    else:
        source_data = data_paths[0]
        target_datas = data_paths[1:]

    source_dataset = [l.strip() for l in open(source_data)]#load_dataset("text", data_files={"test": source_data}, cache_dir="hf_cache")
    target_datasets = [[l.strip() for l in open(target_data)] for target_data in target_datas]#load_dataset("text", data_files={"test": target_data}, cache_dir="hf_cache")
    

    logger.info(f'dataset loaded with {len(source_dataset)} sentence pairs')

    all_performance_metrics = {}
    if "bleu" in evaluation_metrics:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu([detokenize(sent) for sent in source_dataset], target_datasets)
        bleuscore = bleu.score
        print(f"method=bleu, average_score={bleuscore}")
        all_performance_metrics["bleu"] = bleuscore

    if len(set(evaluation_metrics).difference(set(['bertscore', 'wieting_sim', 'transfer', 'fluency']))) > 0:
        #only load these models if the evaluation metric requires it
        model_path = args.model
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="cache")
        content_config = AutoConfig.from_pretrained(model_path, cache_dir="cache")
        content_model = AutoModel.from_pretrained(model_path, config=content_config, cache_dir="cache")
        content_model.eval()
    
    if "bertscore" in evaluation_metrics:
        scorer = bert_score.BERTScorer(lang="en", model_type=model_path, num_layers=10)
    
    if "wieting_sim" in evaluation_metrics:
        # os.environ['ROBERTA_BASE']="fairseq_cache/roberta.base"
        # os.environ['TORCH_HOME']="./fairseq_cache"
        wieting_roberta = torch.hub.load("pytorch/fairseq", "roberta.base", force_reload=True)
    
    if "transfer" in evaluation_metrics:
        transfer_model = RobertaModel.from_pretrained(
            "/path/to/evaluation/models/formality_classifier",
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path="formality-data-bin"
        )
        if use_cuda:
            transfer_model.cuda()
            transfer_model.eval()
    
    if "fluency" in evaluation_metrics:
        fluency_model = RobertaModel.from_pretrained(
            '/path/to/evaluation/models/cola_classifier_fluency/',
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path='cola-bin'
        )
        if use_cuda:
            fluency_model.cuda()
            fluency_model.eval()
    
    logger.info("model and tokenizers loaded")

    if args.model_dtype == "fp16":
        content_model.half()
    if use_cuda:
        content_model.cuda()
    
    from collections import defaultdict
    allscores = defaultdict(list)
    c=0

    for idx in range(0, len(source_dataset), args.batch_size):
        source_tokenized = [tokenizer.encode(detokenize(sent), return_tensors="pt") for sent in source_dataset[idx:idx + args.batch_size]]
        targets_tokenized = [[tokenizer.encode(detokenize(sent), return_tensors="pt") for sent in target_dataset[idx:idx + args.batch_size]] for target_dataset in target_datasets]

        if len(source_tokenized) > 0 and len(targets_tokenized[0]) > 0:
            source_tokenized = torch.cat(source_tokenized, dim=0).to(content_model.device)
            targets_tokenized = [torch.cat(target_tokenized, dim=0).to(content_model.device) for target_tokenized in targets_tokenized]
        else:
            c+=args.batch_size
            raise ValueError(f"one of source or target batch is empty")
            continue

        if "transfer" in evaluation_metrics:
            transfers = transfer_classify(source_dataset[idx:idx + args.batch_size], transfer_model)
            allscores["transfer"] += transfers
        
        if "fluency" in evaluation_metrics:
            fluencys = fluency_classify(source_dataset[idx:idx + args.batch_size], fluency_model)
            allscores["fluency"] += fluencys

        if "bertscore" in evaluation_metrics:
            bestscores = [0. for i in range(args.batch_size)]
            for target_dataset in target_datasets:
                scores = bertscore(source_dataset[idx:idx + args.batch_size], target_dataset[idx:idx + args.batch_size], scorer)
                for i in range(len(scores)):
                    scores[i] = max(bestscores[i], scores[i])
                bestscores = scores
            allscores["bertscore"] += bestscores
        
        if "wmd" in evaluation_metrics:
            bestscores = [100000. for i in range(args.batch_size)]
            for target_tokenized in targets_tokenized:
                scores = wmd(source_tokenized, target_tokenized, content_model.get_input_embeddings())
                for i in range(len(scores)):
                    scores[i] = min(bestscores[i], scores[i])
                bestscores = scores
            allscores['wmd'] += bestscores
            
        if "moverscore" in evaluation_metrics:
            bestscores = [100000. for i in range(args.batch_size)]
            for target_tokenized in targets_tokenized:
                scores = moverscore(source_tokenized, target_tokenized, content_model)
                for i in range(len(scores)):
                    scores[i] = min(bestscores[i], scores[i])
                bestscores = scores
            allscores['moverscore'] += bestscores
            
        if "cls_sim" in evaluation_metrics:  
            bestscores = [0. for i in range(args.batch_size)]
            for target_tokenized in targets_tokenized:
                scores = cls_similarity(source_tokenized, target_tokenized, content_model)
                for i in range(len(scores)):
                    scores[i] = max(bestscores[i], scores[i])
                bestscores = scores
            allscores['cls_sim'] += bestscores
        
        if "sts_sim" in evaluation_metrics:  
            bestscores = [0. for i in range(args.batch_size)]
            for target_tokenized in targets_tokenized:
                scores = sts_similarity(source_tokenized, target_tokenized, content_model)
                for i in range(len(scores)):
                    scores[i] = max(bestscores[i], scores[i])
                bestscores = scores
            allscores['sts_sim'] += bestscores
        
        if "wieting_sim" in evaluation_metrics:
            bestscores = [0. for i in range(args.batch_size)]
            for target_dataset in target_datasets:
                scores = wieting_sim(source_dataset[idx:idx + args.batch_size], target_dataset[idx:idx + args.batch_size], wieting_roberta)
                for i in range(len(scores)):
                    scores[i] = max(bestscores[i], scores[i])
                bestscores = scores

            allscores["weiting_sim"] += bestscores
        
        if idx % 100 == 0:
            print(idx, end="...", flush=True)

    for method, scores in allscores.items():
        # scores_ = scores
        if method == "transfer":
            scores_ = np.array(scores) == "formal"
        elif method == "fluency":
            scores_ = np.array(scores) == "acceptable"
        else:
            scores_ = np.array(scores)
        x= 1.0 * np.mean(scores_.astype("float32"))
        print(f"method={method}, average_score={x}")
        all_performance_metrics[method] = 1.*np.mean(scores_.astype("float32"))
        if args.match_with == "source":
            outname = f"{source_data}.source{method}"
        else:
            outname = f"{source_data}.{method}"
        with open(f"{source_data}.{method}", "w") as fout:
            fout.write("\n".join([str(score) for score in scores]))
    
    

    if args.outfile is not None:
        import json
        with open(args.outfile, "w") as fout:
            json.dump(all_performance_metrics, fout)
        print(f"dumped the performance metrics to {args.outfile}")

    print(f'ignore={c}')

def cli_main():
    parser = options.get_parser()
    parser.add_argument("--pred")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
