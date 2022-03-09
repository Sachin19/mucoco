#this file takes as input different datasets and models, and gives various statistics over different losses supported by this code

import logging
import math
import os
import sys
import re
import torch
import numpy as np
import transformers


from transformers import AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer, util

from mucoco.utils import TargetProbability, TargetEmbeddings, TargetSimplex, Lambda, Optimizer, get_epsilon
import mucoco.losses as lossbuilder
import mucoco.options as options

import torch.nn.functional as F

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
        targetf = open(args.outfile+"."+str(args.max_prefix_length), "w")

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
    additional_dataset = [l.strip() for l in open(additional_data)]
    logger.info("Data loaded")

    source_batch, target_batch, additional_batch, for_predicted_source_batch, predicted_batch = [], [], [], [], []
    batch_size = args.batch_size # higher than 1 batch size does not work at the moment. It won't fit in a single GPU anyway 
    
    device = "cuda" if use_cuda else "cpu"
    c = 0

    targetlosslist = []
    predictedlosslist = []
    all_stepcounts = []

    #data loading is very simple and probably can be sped up
    for source_text, target_text, additional_text in zip(source_dataset, target_dataset, additional_dataset):
        source_indices = primary_tokenizer.encode(source_text, return_tensors="pt").to(device)
        additional_indices = primary_tokenizer.encode(additional_text, return_tensors="pt", add_special_tokens=False).to(device)

        eos_token_id=primary_tokenizer.eos_token_id
        if args.target_tokenize_different:
            with primary_tokenizer.as_target_tokenizer():
                eos_token_id=primary_tokenizer.eos_token_id
                
        with torch.no_grad():
            predicted_indices = clean_output(lossfns[0].generate(input_ids=input_ids, additional_ids=additional_indices)[0].tolist(), eos_token_id=eos_token_id, return_tensors=True) #some bug about length

        if args.target_tokenize_different:
            with primary_tokenizer.as_target_tokenizer():
                beam_prediction = primary_tokenizer.decode(predicted_indices[0].tolist())
        else:
            beam_prediction = primary_tokenizer.decode(predicted_indices[0].tolist())
        
        if args.outfile is not None:
            targetf.write(beam_prediction.replace("\n", " ")+"\n")


    if args.outfile is not None:
        targetf.close()

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
