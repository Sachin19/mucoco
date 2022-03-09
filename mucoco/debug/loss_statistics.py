#this file takes as input different datasets and models, and gives various statistics over different losses supported by this code

import logging
import math
import os
import sys
import re
import torch
import numpy as np
import transformers
import random


from transformers import AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer, util

from mucoco.utils import TargetProbability, TargetEmbeddings, TargetSimplex, Lambda, Optimizer, get_epsilon
import mucoco.losses as lossbuilder
import mucoco.options as options

import torch.nn.functional as F

import matplotlib.pyplot as plt

# To control logging level for various modules used in the application:
# from here: https://github.com/huggingface/transformers/issues/3050
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

def plot(scores, fname="hist.png", bins=100):
    plt.clf()
    plt.hist(scores, bins=bins)
    plt.savefig(fname)

def main(args):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.ERROR,
        stream=sys.stdout,
    )
    logger = logging.getLogger("mucoco")
    logger.setLevel(logging.ERROR)
    logger.info(args)

    if args.outfile is not None: 
        targetf = open(args.outfile+".targetloss", "w")
        targetsatf = open(args.outfile+".targetsat", "w")
        shuffled_targetf = open(args.outfile+".shuffled_targetloss", "w")
        shuffled_targetsatf = open(args.outfile+".shuffled_targetsat", "w")
        predictedf = open(args.outfile+".predictedloss", "w")
        predictedsatf = open(args.outfile+".predictedsat", "w")

    # Fix seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(0)

    use_cuda = torch.cuda.is_available() and not args.cpu
    logger.info(
        "loading model(s) from {} and tokenizer(s) from {}".format(
            args.model, args.tokenizer
        )
    )

    name2tokenizer = {}
    name2model = {}
    name2config = {}
    loss2modelname = {}
    loss2tokenizer = {}
    embed_luts = []
    embed_scales = []

    betas = []
    model_paths = args.model.split(":")
    tokenizer_paths = args.tokenizer.split(":")

    if args.model_types is not None:
        model_types = args.model_types.split(":")
    else:
        model_types = [AutoModel for _ in model_paths]

    losses = args.loss.split(":")
    if args.lossabbr is not None:
        lossabbr = args.lossabbr.split(":")
    else:
        lossabbr = [x for x in losses]

    if args.label_id is None or args.label_id == "none":
        label_ids = [1 for _ in losses]
    else:
        label_ids = [int(i) for i in args.label_id.split(":")]
    
    if args.selection_criterion == "primary_allsat": 
        # with this flag, the output which minimized the primary objective while satisfying all objectives is selected. In case all constraints are not satisfied (e.g when constraints are competing or optimization fails), this will predict the default output (Using an autoregressive decoding setup: beam search in this case)
        betas = [1.0] + [0.0 for _ in range(len(losses)-1)]
    elif args.selection_criterion == "weighted_sum" and args.betas is not None:
        # this setup will select the best outputs according to the weights betas for each of the losses (even though they are not satisfied)
        betas = [float(beta) for beta in args.betas.split(":")]
    else:
        raise ValueError("correct selection_criterion or betas needs to be specified")

    assert len(betas) == len(losses) and len(losses) == len(model_paths) and len(model_paths) == len(model_types) and len(betas) == len(lossabbr)
    assert np.abs(sum(betas) - 1.0) < 1e-6, f"sum of betas is {sum(betas)} != 1.0"

    prev_vocab_size = None
    vocab_size = None
    primary_vocab_size = None

    #Load the models and tokenizers
    for i, model_path in enumerate(model_paths):
        if model_path not in name2model: #making sure we are not loading the model twice in case some constraints use the same model. 
            name2tokenizer[model_path] = AutoTokenizer.from_pretrained(tokenizer_paths[i], cache_dir=args.cache_dir,  use_fast=True)
            name2config[model_path] = AutoConfig.from_pretrained(model_path, cache_dir=args.cache_dir)

            if model_types[i] == "sentence-transformer":
                name2model[model_path] = SentenceTransformer(model_path)
            else:
                name2model[model_path] = getattr(transformers, model_types[i]).from_pretrained(model_path, config=name2config[model_path], cache_dir=args.cache_dir)
            
            if not args.show_warnings:
                # print(logging.root.manager.loggerDict)
                # input()
                set_global_logging_level(logging.ERROR, [name2model[model_path].__module__])
                # logging.getLogger(name2model[model_path].__class__.__name__).setLevel(logging.ERROR) 
            
            name2model[model_path].eval()
            new_vocab_size = name2model[model_path].get_input_embeddings().num_embeddings
            if prev_vocab_size is None:
                vocab_size=new_vocab_size
            if new_vocab_size != prev_vocab_size and prev_vocab_size is not None:
                if not args.allow_diff_vocab:
                    raise ValueError(f"all models should have the same vocabulary {prev_vocab_size} != {vocab_size}")
                else:
                    logger.warning("all models don't have the same vocabulary and we are still proceeding")
            prev_vocab_size = vocab_size
        
        if args.target_tokenize_different: # for seq2seq models where target tokenizer is different than the source tokenizer
            embed_luts.append(name2model[model_path].get_decoder().get_input_embeddings())
        else:
            embed_luts.append(name2model[model_path].get_input_embeddings())
        
        if i == 0:
            primary_vocab_size = vocab_size
            primary_embed_dim = embed_luts[-1].embedding_dim
        
        if getattr(name2model[model_path], "get_decoder", None) is None: #this is for MarianMT models which have a weird embedding_scale parameter
            embed_scales.append(1.0)
        else:
            embed_scales.append(getattr(name2model[model_path].get_decoder(), "embed_scale", 1.0))
    
    if use_cuda:
        for name, model in name2model.items():
            model.cuda()
        logger.info("model(s) moved to GPU")
      
    #first loss is the primary loss, others are constraints
    lossfns = []
    for i, loss in enumerate(losses):
        lossfns.append(lossbuilder.build_loss(loss, name2model[model_paths[i]], name2tokenizer[model_paths[i]], args))
        loss2modelname[loss] = model_paths[i]
        loss2tokenizer[loss] = name2tokenizer[model_paths[i]]
    primary_tokenizer = loss2tokenizer[losses[0]]
    
    logger.info("tokenizer(s), model(s) and loss function(s) loaded")

    if args.model_dtype == "fp16": #while this is supported it doesn't work that well yet. Not recommended
        for name, model in name2model.items():
            model.half()
        logger.info("changed everything to fp16")
    
    #constraint thresholds. In the paper, we recommend to start with a high threshold value which is usually satisfied by default or easily satisfied and then decrease it gradually, otherwise weird adversarial solutions come up. This code supports different kinds of schedules for decreasing this threshold (usually just step or linear suffices). If no schedule is specified, it just remains the same as the original. 
    if args.epsilons is not None and args.epsilons != "none":
        epsilons = [float(eps) for eps in args.epsilons.split(":")]
        if args.min_epsilons is not None:
            min_epsilons = [float(eps) for eps in args.min_epsilons.split(":")]
            epsilon_warmup_steps = [int(steps) for steps in args.epsilon_warmup_steps.split(":")]
            epsilon_cooldown_steps = [int(steps) for steps in args.epsilon_cooldown_steps.split(":")]
            epsilon_decay_functions = [f for f in args.epsilon_decay_functions.split(":")]
        else:
            min_epsilons = [float(eps) for eps in args.epsilons.split(":")]
            epsilon_warmup_steps = [1 for eps in min_epsilons]
            epsilon_cooldown_steps = [2 for eps in min_epsilons]
            epsilon_decay_functions = ["none" for eps in min_epsilons]
    else:
        epsilons = []
        min_epsilons = []
        decay_function = []
        epsilon_warmup_steps = []
        epsilon_cooldown_steps = []
    
    assert args.data is not None or args.additional_data is not None, "no data path has been provided"
    if args.data is not None:
        data_paths = args.data.split(":")
        if len(data_paths) == 1:
            source_data = data_paths[0]
            target_data = data_paths[0]
        else:
            source_data = data_paths[0]
            target_data = data_paths[1] # useful for debugging
    
        additional_data = args.additional_data
        if additional_data is None:
            additional_data = source_data # additional data was used in STRAP (Krishna et al 2020) when x is paraphrased to z, then the model is used to generate y in the target style. If there's no additional_data, it defaults to the source text
    else:
        source_data = args.additional_data
        target_data = args.additional_data
        additional_data = args.additional_data
    
    logger.info("Loading the dataset ...")
    source_dataset = [l.strip() for l in open(source_data)]
    target_dataset = [l.strip() for l in open(target_data)]
    shuffled_target_dataset = [l.strip() for l in open(target_data)]
    random.shuffle(shuffled_target_dataset)
    
    additional_dataset = [l.strip() for l in open(additional_data)]
    logger.info("Data loaded")

    source_batch, target_batch, additional_batch, for_predicted_source_batch, predicted_batch, shuffled_target_batch = [], [], [], [], [], []
    batch_size = args.batch_size # higher than 1 batch size does not work at the moment. It won't fit in a single GPU anyway 
    
    device = "cuda" if use_cuda else "cpu"
    c = 0

    targetlosslist = []
    shuffled_targetlosslist = []
    predictedlosslist = []
    all_stepcounts = []

    #data loading is very simple and probably can be sped up
    for source_text, target_text, additional_text, shuffled_target_text in zip(source_dataset, target_dataset, additional_dataset, shuffled_target_dataset):
        
        early_skip="n"
        if args.debug:
            early_skip = input(f"skip this example? {source_text} [yes(y)/no(n)]")
            if early_skip == "y":
                continue

        if args.num_examples > 0 and c > 0 and c == args.num_examples: #stop after processing num_examples if it is set 
            print(f"done {c}")
            break

        c += 1

        source_indices = primary_tokenizer.encode(source_text, return_tensors="pt").to(device)
        additional_indices = primary_tokenizer.encode(additional_text, return_tensors="pt", add_special_tokens=False).to(device)

        eos_token_id=primary_tokenizer.eos_token_id
        if args.target_tokenize_different:
            with primary_tokenizer.as_target_tokenizer():
                eos_token_id=primary_tokenizer.eos_token_id
                
        with torch.no_grad():
            predicted_indices = clean_output(lossfns[0].generate(input_ids=input_ids, additional_ids=additional_indices)[0].tolist(), eos_token_id=eos_token_id, return_tensors=True) #some bug about length
            # print(additional_indices)
            # print(predicted_indices)
            # input()

        if args.target_tokenize_different:
            with primary_tokenizer.as_target_tokenizer():
                beam_prediction = primary_tokenizer.decode(predicted_indices[0].tolist())
        else:
            beam_prediction = primary_tokenizer.decode(predicted_indices[0].tolist())

        if not args.target_tokenize_different and "Seq2SeqLM" in model_paths[0]:
            logger.warning("you are using a seq2seq model for your primary loss but not tokenizing the target sentences with a different target tokenizer.")

        #for_predicted_source_indices, are used to compute the primary loss wrt source as target. Useful for debugging style transfer models. 
        if args.target_tokenize_different:
            with primary_tokenizer.as_target_tokenizer():
                for_predicted_source_indices = primary_tokenizer.encode(source_text, return_tensors="pt").to(device)
                target_indices = primary_tokenizer.encode(target_text, return_tensors="pt", add_special_tokens=False).to(device)
                shuffled_target_indices = primary_tokenizer.encode(shuffled_target_text, return_tensors="pt", add_special_tokens=False).to(device)
        else:
            for_predicted_source_indices = source_indices
            target_indices = primary_tokenizer.encode(target_text, return_tensors="pt", add_special_tokens=False).to(device)
            shuffled_target_indices = primary_tokenizer.encode(shuffled_target_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        source_batch.append(source_indices)
        target_batch.append(target_indices)
        shuffled_target_batch.append(shuffled_target_indices)
        for_predicted_source_batch.append(for_predicted_source_indices)
        predicted_batch.append(predicted_indices)
        additional_batch.append(additional_indices)

        if len(source_batch) == batch_size: #this is just one for now, greater than 1 batch size will not work

            source_batch = torch.cat(source_batch, dim=0).to(device)
            target_batch = torch.cat(target_batch, dim=0).to(device)
            shuffled_target_batch = torch.cat(shuffled_target_batch, dim=0).to(device)
            additional_batch = torch.cat(additional_batch, dim=0).to(device)
            
            predicted_batch = torch.cat(predicted_batch, dim=0).to(device)
            # for_predicted_source_batch = torch.cat(for_predicted_source_batch, dim=0).to(device)  
            predicted_allsat=False
            # losses of the beam-search output: we should perform atleast as well as this. If we don't, we predict this output
            # Also, if the beam-search output already satisfies the constraints, we skip mucoco unless, args.always_mucoco is true
            
            predicted_labels = {}
            total_predicted_loss = 0.0
            predicted_allsat=True
            predicted_sats = []
            predictedlosses = []

            for lossid in range(len(losses)):
                lossname = losses[lossid]
                predicted_loss, predicted_lo =\
                    lossfns[lossid].compute_gold_loss(
                        (source_batch, predicted_batch), 
                        additional_batch=additional_batch, 
                        label_id=label_ids[lossid])
                
                predicted_loss = predicted_loss.sum().item()
                predictedlosses.append(predicted_loss)
                total_predicted_loss += betas[lossid] * predicted_loss

                if lossid > 0 and (predicted_loss <= min_epsilons[lossid-1]):
                    predicted_sats.append("satisfied")
                elif lossid > 0:
                    predicted_sats.append("violated")
                    predicted_allsat = False
                
                if "label_prediction" in predicted_lo:
                    predicted_labels[lossid] = predicted_lo['label_prediction']
                else:
                    predicted_labels[lossid] = "NA"
                
            predictedlosslist.append(predictedlosses)

            if args.outfile is not None:
                predictedf.write(" ".join([str(x) for x in predictedlosses])+"\n")
                predictedsatf.write(" ".join(predicted_sats)+"\n")

            target_labels = {}
            total_target_loss = 0.0
            target_allsat=True
            target_sats = []
            targetlosses = []

            for lossid in range(len(losses)):
                lossname = losses[lossid]
                target_loss, target_lo =\
                    lossfns[lossid].compute_gold_loss(
                        (source_batch, target_batch), 
                        additional_batch=additional_batch, 
                        label_id=label_ids[lossid])
                
                target_loss = target_loss.sum().item()
                targetlosses.append(target_loss)
                total_target_loss += betas[lossid] * target_loss

                if lossid > 0 and (target_loss <= min_epsilons[lossid-1]):
                    target_sats.append("satisfied")
                elif lossid > 0:
                    target_sats.append("violated")
                    target_allsat = False

                if "label_prediction" in target_lo:
                    target_labels[lossid] = target_lo['label_prediction']
                else:
                    target_labels[lossid] = "NA"
            
            targetlosslist.append(targetlosses)
            if args.outfile is not None:
                targetf.write(" ".join([str(x) for x in targetlosses])+"\n")
                targetsatf.write(" ".join(target_sats)+"\n")
            
            shuffled_target_labels = {}
            total_shuffled_target_loss = 0.0
            shuffled_target_allsat=True
            shuffled_target_sats = []
            shuffled_targetlosses = []

            for lossid in range(len(losses)):
                lossname = losses[lossid]
                shuffled_target_loss, shuffled_target_lo =\
                    lossfns[lossid].compute_gold_loss(
                        (source_batch, shuffled_target_batch), 
                        additional_batch=additional_batch, 
                        label_id=label_ids[lossid])
                
                shuffled_target_loss = shuffled_target_loss.sum().item()
                shuffled_targetlosses.append(shuffled_target_loss)
                total_shuffled_target_loss += betas[lossid] * shuffled_target_loss

                if lossid > 0 and (shuffled_target_loss <= min_epsilons[lossid-1]):
                    shuffled_target_sats.append("satisfied")
                elif lossid > 0:
                    shuffled_target_sats.append("violated")
                    shuffled_target_allsat = False

                if "label_prediction" in shuffled_target_lo:
                    shuffled_target_labels[lossid] = shuffled_target_lo['label_prediction']
                else:
                    shuffled_target_labels[lossid] = "NA"
            
            shuffled_targetlosslist.append(shuffled_targetlosses)
            if args.outfile is not None:
                shuffled_targetf.write(" ".join([str(x) for x in shuffled_targetlosses])+"\n")
                shuffled_targetsatf.write(" ".join(target_sats)+"\n")

            

            del source_batch
            del target_batch
            del additional_batch
            del for_predicted_source_batch
            del predicted_batch
            source_batch = []
            target_batch = []
            shuffled_target_batch = []
            for_predicted_source_batch = []
            additional_batch = []
            predicted_batch = []

    if args.outfile is not None:
        targetf.close()
        targetsatf.close()
        predictedf.close()
        predictedsatf.close()
    
    print('plotting')
    for lossname, lossvalues in zip(losses, zip(*predictedlosslist)):
        plot(lossvalues, f'{args.outfile}.predicted_{lossname}.png', 100)
    
    for lossname, lossvalues in zip(losses, zip(*targetlosslist)):
        plot(lossvalues, f'{args.outfile}.target_{lossname}.png', 100)
        
    for lossname, lossvalues in zip(losses, zip(*shuffled_targetlosslist)):
        plot(lossvalues, f'{args.outfile}.shuffled_target_{lossname}.png', 100)
    # print("average numbers of steps to converge =", np.mean(all_stepcounts))

def clean_output(tokens, eos_token_id, return_tensors=False):
    new_tokens = []
    for tok in tokens:
        if tok != eos_token_id:
            new_tokens.append(tok)
        else:
            break
    if return_tensors:
        return torch.LongTensor([new_tokens])
    return new_tokens
    
def cli_main():
    parser = options.get_parser()
    args = parser.parse_args()
    main(args)
