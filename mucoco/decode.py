import logging
import math
import os
import sys
import re
import torch
import numpy as np
import transformers
import gc
import time
import json


from transformers import AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer, util

from mucoco.utils import TargetProbability, TargetEmbeddings, TargetSimplex, Lambda, Optimizer, get_epsilon
import mucoco.losses as lossbuilder
import mucoco.options as options
import mucoco.utils as utils
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
        outf = open(args.outfile, "w")
        outallsatf = open(args.outfile + ".allsat", "w")

    # Fix seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

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
    cur_lr = args.lr
    args.jsonl_tokenized = args.jsonl_tokenized == "true"

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
    
    if args.keywords is None or args.keywords == "none":
        keywords = ["the" for _ in losses]
    elif args.keywords in "_roc_" or args.keywords == "_commongen_":
        keywords = ["" for _ in losses] # will be different for each input
    else:
        keywords = args.keywords.split(":")
        if len(keywords) == 1:
            keywords = [f"_topic_:{args.keywords[0]}" for _ in losses] #when keyword isn't used but topic is passed
    
    if "allsat" in args.selection_criterion: 
        # with this flag, the output which minimized the primary objective while satisfying all objectives is selected. In case all constraints are not satisfied (e.g when constraints are competing or optimization fails), this will predict the default output (Using an autoregressive decoding setup: beam search in this case)
        betas = [1.0] + [0.0 for _ in range(len(losses)-1)]
    elif (args.selection_criterion == "weighted_sum" and args.betas is not None) or args.selection_criterion == "last":
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
                name2model[model_path] = lossbuilder.ModelWrapper(SentenceTransformer(model_path))
            elif "Custom" in model_types[i]:
                name2model[model_path] = lossbuilder.ModelWrapper(getattr(utils, model_types[i]).from_pretrained(model_path, config=name2config[model_path], cache_dir=args.cache_dir))
            else:
                name2model[model_path] = lossbuilder.ModelWrapper(getattr(transformers, model_types[i]).from_pretrained(model_path, config=name2config[model_path], cache_dir=args.cache_dir))
            
            if not args.show_warnings:
                # print(logging.root.manager.loggerDict)
                # input()
                set_global_logging_level(logging.ERROR, [name2model[model_path].__module__])
                # logging.getLogger(name2model[model_path].__class__.__name__).setLevel(logging.ERROR) 
            
            name2model[model_path].eval()
            embed_lut_ = name2model[model_path].get_input_embeddings()
            if isinstance(embed_lut_, torch.nn.Sequential):
                new_vocab_size = embed_lut_[0].num_embeddings
            else:
                new_vocab_size = embed_lut_.num_embeddings
            if prev_vocab_size is None:
                vocab_size=new_vocab_size
            if new_vocab_size != prev_vocab_size and prev_vocab_size is not None:
                if not args.allow_diff_vocab:
                    raise ValueError(f"all models should have the same vocabulary {new_vocab_size} != {vocab_size}")
                else:
                    logger.warning("all models don't have the same vocabulary and we are still proceeding")
            prev_vocab_size = vocab_size
        
        if args.target_tokenize_different: # for seq2seq models where target tokenizer is different than the source tokenizer
            embed_luts.append(name2model[model_path].get_decoder().get_input_embeddings())
        else:
            input_embeds = name2model[model_path].get_input_embeddings()
            if isinstance(input_embeds, torch.nn.Sequential):
                input_embeds = input_embeds[0]
            embed_luts.append(input_embeds)
        
        if args.target_type == "embeds":
            embed_luts[-1].requires_grad=False
        
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
        min_epsilons = [eps + getattr(lossfns[i+1], "epsilon_additive", 0)  for i, eps in enumerate(min_epsilons)]
    else:
        epsilons = []
        min_epsilons = []
        decay_function = []
        epsilon_warmup_steps = []
        epsilon_cooldown_steps = []
    
    # assert args.data is not None or args.additional_data is not None, "no data path has been provided"
    source_dataset = None
    target_dataset = None
    additional_dataset = None
    print("yass queen", args.use_context)
    args.use_context = args.use_context == "true"
    print(args.use_context)
    if args.data is not None:
        data_paths = args.data.split(":")
        if len(data_paths) == 1:
            source_data = data_paths[0]
            target_data = data_paths[0] #not used
            context_data = data_paths[0] # not used
            # args.use_context = False
        elif len(data_paths) == 2:
            source_data = data_paths[0]
            target_data = data_paths[1] # useful for debugging
            context_data = data_paths[1] #not used here
            # args.use_context = False
        else:
            source_data = data_paths[0]
            target_data = data_paths[1] # useful for debugging
            context_data = data_paths[2] # tsv file
    
        additional_data = args.additional_data
        if additional_data is None or additional_data == "none":
            additional_data = source_data # additional data was used in STRAP (Krishna et al 2020) when x is paraphrased to z, then the model is used to generate y in the target style. If there's no additional_data, it defaults to the source text
    elif args.additional_data is not None and additional_data != "none":
        source_data = args.additional_data
        target_data = args.additional_data
        additional_data = args.additional_data
    else:
        source_dataset = sys.stdin
        start_idx = 0
        end_idx = 1000000 # a very high number
    
    if source_dataset is None:
        logger.info("Loading the dataset ...")
        if args.datastyle == "text":
            source_dataset = [l.strip() for l in open(source_data)]
            target_dataset = [l.strip() for l in open(target_data)]
            context_dataset = []
            import csv
            with open(context_data) as csvfile: #there can be multiple contexts, for example for paraphrasing, so we allow for a list of contexts for every input
                reader = csv.reader(csvfile, delimiter="\t")
                for row in reader:
                    context_dataset.append(row)
            additional_dataset = [l.strip() for l in open(additional_data)]
        elif args.datastyle == "jsonl": #for some prompts datasets
            source_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(source_data)]
            target_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(target_data)]
            additional_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(additional_data)]
            if args.jsonl_secondary_key is not None and args.jsonl_secondary_key != "none":
                source_dataset = [x[args.jsonl_secondary_key] for x in source_dataset]
                target_dataset = [x[args.jsonl_secondary_key] for x in target_dataset]
                additional_dataset = [x[args.jsonl_secondary_key] for x in additional_dataset]

            context_dataset = [None] * len(source_dataset)
            if args.use_context:
                context_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(context_data)]
                if args.jsonl_secondary_key is not None and args.jsonl_secondary_key != "none":
                    context_dataset = [x[args.jsonl_secondary_key] for x in context_dataset]
        elif args.datastyle == "single-jsonl": #one jsonl file has all the information
            source_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(source_data)]
            target_dataset = [json.loads(l)[args.jsonl_secondary_key] for l in open(target_data)]
            additional_dataset = [json.loads(l)[args.jsonl_secondary_key] for l in open(additional_data)]
            
            context_dataset = [None] * len(source_dataset)
            if args.use_context:
                context_dataset = [[json.loads(l)[args.jsonl_secondary_key]] for l in open(context_data)] #meaningful
        start_idx = args.start_idx
        end_idx = (len(source_dataset) + args.end_idx) % len(source_dataset) # also works with negative end_idx
            
        logger.info("Data loaded")

    source_batch, target_batch, additional_batch, for_predicted_source_batch, predicted_batch, context_batch = [], [], [], [], [], []
    batch_size = args.batch_size # higher than 1 batch size does not work at the moment. It won't fit in a single GPU anyway 
    
    device = "cuda" if use_cuda else "cpu"
    c = 0

    losslists = [[] for _ in range(len(losses))]
    predictedlosslists = [[] for _ in range(len(losses))]
    source_primarylosslist = [] 
    # allparetosets = []
    all_stepcounts = []
    avg_time = 0

    #data loading is very simple and probably can be sped up

    if args.gold_loss_epsilons is not None and args.gold_loss_epsilons != "none":
        args.gold_loss_epsilons = args.gold_loss_epsilons.lower().split(":")
        assert len(args.gold_loss_epsilons) == len(losses)-1
    else:
        args.gold_loss_epsilons = ["false" for _ in range(len(losses)-1)]

    # for source_text, target_text, additional_text in zip(source_dataset, target_dataset, additional_dataset):
    example_p = 1.0
    args.random_example = args.random_example == "true"
    if args.num_examples > 0 and target_dataset is not None:
        example_p = args.num_examples*1.0/len(source_dataset)
    print(example_p, args.random_example)
    print(start_idx, end_idx)
    for text_id, source_text in enumerate(source_dataset):
        
        if text_id < start_idx or text_id > end_idx:
            continue

        if args.num_examples > 0 and c > 0 and c == args.num_examples: #stop after processing num_examples if it is set 
            print(f"done {c}")
            break
        
        if args.random_example:
            do_this_example = np.random.rand() <= example_p
            if not do_this_example:
                continue

        c += 1

        new_kweight = args.kweight
        if target_dataset is not None:
            target_text = target_dataset[text_id]
            additional_text = additional_dataset[text_id]
            context_texts = context_dataset[text_id]
            # print(context_texts)
            # input()
        else:
            args.jsonl_tokenized = False
            items = source_text.split("::")
            source_text = items[0].rstrip()
            target_text = items[1].rstrip()
            if target_text == "-":
                args.init = "zeros"
            elif target_text == "--":
                args.init = "target"
            else:
                args.init = "targettarget"
            additional_text = items[1]

            if len(items) > 2:
                args.max_output_length = int(items[2])
                args.max_length = int(items[2])
            if len(items) > 3:
                args.use_context = True
                context_texts = [items[3].rstrip()]                                
            else:
                args.use_context = False
                context_texts = []

            if len(items) > 4:
                new_kweight = float(items[4])
            # if len(items) > 4:
            #     target_text = items[4].rstrip()
            
            print(args.use_context, context_texts)
                
        if args.keywords == "_roc_":
            keywords = ["none"] + additional_text.split(", ")
            # input(keywords)
        elif args.keywords == "_commongen_":
            keywords = ["none"] + json.loads(additional_text)['concept_set'].split("#")
            # input(keywords)

        
        early_skip="n"
        if args.debug:
            early_skip = input(f"skip this example? {source_text} [yes(y)/maybe(m)/no(n)]")
            if early_skip == "y":
                continue

        if not args.jsonl_tokenized:
            if source_text == "":
                source_text = primary_tokenizer.bos_token
            source_indices = primary_tokenizer.encode(source_text, return_tensors="pt").to(device)
            source_indices_write = source_indices[0].tolist()
            # if source_indices
            additional_indices = primary_tokenizer.encode(additional_text, return_tensors="pt", add_special_tokens=False).to(device)
            
            eos_token_id = primary_tokenizer.eos_token_id
            bos_token_id = primary_tokenizer.bos_token_id
            context_indices = None
            if args.target_tokenize_different:
                with primary_tokenizer.as_target_tokenizer():
                    eos_token_id=primary_tokenizer.eos_token_id
                    bos_token_id = primary_tokenizer.bos_token_id
                    if args.use_context:
                        context_indices = primary_tokenizer.encode(context_texts[0], return_tensors="pt", add_special_tokens=False).to(device).unsqueeze(1)
            elif args.use_context:
                context_indices = primary_tokenizer.encode(context_texts[0], return_tensors="pt", add_special_tokens=False).to(device).unsqueeze(1)

            if not args.target_tokenize_different and "Seq2SeqLM" in model_paths[0]:
                logger.warning("you are using a seq2seq model for your primary loss but not tokenizing the target sentences with a different target tokenizer.")

            #for_predicted_source_indices are used to compute the primary loss wrt source as target. Useful for debugging style transfer models. 
            if args.target_tokenize_different:
                with primary_tokenizer.as_target_tokenizer():
                    for_predicted_source_indices = primary_tokenizer.encode(source_text, return_tensors="pt").to(device)
                    target_indices = primary_tokenizer.encode(target_text, return_tensors="pt", add_special_tokens=False).to(device)
            else:
                for_predicted_source_indices = source_indices
                target_indices = primary_tokenizer.encode(target_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        else:
            source_indices_write = source_text # to write to file
            source_indices = source_text
            target_indices = target_text
            additional_indices = additional_text
            context_indices = context_texts
            if len(source_indices) == 0:
                source_indices.append(primary_tokenizer.bos_token_id)

            source_indices = torch.LongTensor([source_indices]).to(device)
            additional_indices = torch.LongTensor([additional_indices]).to(device)
                        
            #unused
            context_indices = None
            if args.use_context:
                context_indices = torch.LongTensor([context_indices]).to(device).to(device).unsqueeze(1)
            #end unused

            #for_predicted_source_indices are used to compute the primary loss wrt source as target. Useful for debugging style transfer models. 
            for_predicted_source_indices = source_indices
            target_indices = torch.LongTensor([target_indices]).to(device)

            bos_token_id = primary_tokenizer.bos_token_id
            eos_token_id = primary_tokenizer.eos_token_id
            if args.target_tokenize_different:
                with primary_tokenizer.as_target_tokenizer():
                    bos_token_id = primary_tokenizer.bos_token_id
                    eos_token_id = primary_tokenizer.eos_token_id
            
            source_text = primary_tokenizer.decode(source_indices[0].tolist())

        source_batch.append(source_indices)
        target_batch.append(target_indices)
        for_predicted_source_batch.append(for_predicted_source_indices)
        additional_batch.append(additional_indices)
        context_batch.append(context_indices)

        if len(source_batch) == batch_size: #this is just one for now, greater than 1 batch size will not work

            source_batch = torch.cat(source_batch, dim=0).to(device)
            target_batch = torch.cat(target_batch, dim=0).to(device)
            additional_batch = torch.cat(additional_batch, dim=0).to(device)
            for_predicted_source_batch = torch.cat(for_predicted_source_batch, dim=0).to(device)  
            
            # print("what", args.use_context)
            if args.use_context:
                context_batch = torch.cat(context_batch, dim=0).to(device)
                print(context_batch)

            broken_skip = False
            for sample_idx in range(args.num_samples):
                for restart_idx in range(args.restarts + 1): # restart the optimization if the constraints are not satisfied
                    skip=False
                    predicted_allsat=False
                    lengthwise_best_prediction = [None] * batch_size

                    predicted_batch = []
                    for batchidx in range(source_batch.size(0)):
                        with torch.no_grad():
                            starttime = time.time()
                            AR_predicted_indices = clean_output(lossfns[0].generate(input_ids=source_batch[batchidx].unsqueeze(0), additional_ids=additional_batch[batchidx].unsqueeze(0))[0].tolist(), eos_token_id=eos_token_id, return_tensors=True, allow_first_eos=losses[0] == "bart", skip_special_tokens=[bos_token_id, eos_token_id]) #some bug about length
                            if args.time:
                                print(time.time()-starttime)
                            if args.debug:
                                print("AR output:", source_text, additional_text, AR_predicted_indices)

                        if args.target_tokenize_different:
                            with primary_tokenizer.as_target_tokenizer():
                                AR_prediction = primary_tokenizer.decode(AR_predicted_indices[0].tolist())
                        else:
                            AR_prediction = primary_tokenizer.decode(AR_predicted_indices[0].tolist())

                        predicted_batch.append(AR_predicted_indices)
                    predicted_batch = torch.cat(predicted_batch, dim=0).to(device)

                    # losses of the beam-search output: we should perform atleast as well as this. If we don't, we predict this output
                    # Also, if the beam-search output already satisfies the constraints, we skip mucoco unless, args.always_mucoco is true
                    predicted_labels = {}
                    total_predicted_loss = 0.0
                    predicted_allsat=True
                    predictedlosses = []
                    # print("what2", args.use_context)
                    for lossid in range(len(losses)):
                        lossname = losses[lossid]
                        predicted_loss, predicted_lo =\
                            lossfns[lossid].compute_gold_loss(
                                (source_batch, predicted_batch), 
                                additional_batch=additional_batch, 
                                context_batch=context_batch,
                                use_context=args.use_context,
                                label_id=label_ids[lossid],
                                keyword=keywords[lossid],
                                kweight=new_kweight)

                        predictedlosses.append(predicted_loss.data.cpu())
                        predicted_loss = predicted_loss.sum().item()
                        total_predicted_loss += betas[lossid] * predicted_loss

                        if lossid > 0:
                            predicted_allsat = predicted_allsat and (predicted_loss <= min_epsilons[lossid-1])
                        
                        if "label_prediction" in predicted_lo:
                            predicted_labels[lossid] = predicted_lo['label_prediction']
                        else:
                            predicted_labels[lossid] = "NA"
                        
                        if lossid > 0 and args.gold_loss_epsilons[lossid-1] == "true": #use the predicted loss as the threshold, mucoco has to beat it then
                            min_epsilons[lossid - 1] = predicted_loss + getattr(lossfns[lossid], "epsilon_additive", 0)
                            epsilons[lossid - 1] = predicted_loss + getattr(lossfns[lossid], "epsilon_additive", 0) ##TODO check 
                        
                    predictedlosslists.append(predictedlosses)
                    
                    if args.only_mucoco == "false":
                        lengthwise_best_prediction = [(AR_prediction, total_predicted_loss, predicted_allsat, AR_predicted_indices[0].tolist(), -1)]
                    skip = predicted_allsat
                        
                    definite_skip = False
                    ask_skip = ""
                    if args.debug and early_skip=="m": 
                        print(f"new example: {source_text}\nautoregressive output: {AR_prediction}")
                        for lossid in range(len(losses)):
                            print(f"{lossabbr[lossid]} for desired label_id({label_ids[lossid]}): {predictedlosslists[-1][lossid]}; predicted label: {predicted_labels[lossid]}")
                        if predicted_allsat:
                            print(f"autoregressive output already satisfies the constraints")
                        ask_skip = input(f"skip this example? [y/n]")
                        definite_skip = ask_skip == "y"

                        # if definite_skip:
                        #     print("Skipping this example. the beam search output already satisfies all the constraints or there's no constraints")
                        #     prediction_ids = ", ".join([str(idx) for idx in AR_predicted_indices[0].tolist()])
                        #     prediction = AR_prediction
                        #     print(f"Prediction ids: {prediction_ids}")
                        #     print(f"Prediction: {prediction}")
                        #     print()

                    elif skip and predicted_allsat and (args.always_mucoco == "false"):
                        # print("Skipping this example. the beam search output already satisfies all the constraints or there's no constraints")
                        # if args.debug:
                        #     prediction_ids = ", ".join([str(idx) for idx in AR_predicted_indices[0].tolist()])
                        #     prediction = AR_prediction
                        #     print(f"Prediction ids: {prediction_ids}")
                        #     print(f"Prediction: {prediction}")
                        #     print()
                        definite_skip = True
                    if args.debug:
                        print('definite_skip',definite_skip, skip, predicted_allsat, args.always_mucoco)
                    
                    if not definite_skip:
                        print(args.max_length)
                        if (args.max_length is None or args.max_length == -1) and args.init not in ["source", "target"]: 
                            #since we don't know the about length, we search in a (-length_diff, length_diff) window and predict the best performing one. 

                            predicted_length = predicted_batch.size(1)
                            length_range = [predicted_length + int(diff) for diff in args.length_diff.split(":")]
                            # length_range = range(max(1, predicted_length-args.length_diff), predicted_length+args.length_diff+1)
                            length_range = [x for x in length_range if x <= args.max_allowed_length and x >= 1]
                            if len(length_range) == 0:
                                length_range = [args.max_allowed_length]
                            # print(predicted_length, length_range)
                            length_range = sorted(list(set(length_range)))

                        elif args.init == "targettarget":
                            length_range = [target_batch.size(1)]
                            
                        elif args.init == "target":
                            length_range = [predicted_batch.size(1)]
                            
                        elif args.init == "source":
                            length_range = [source.size(1)]
                            
                        else: 
                            #another way to use this approach is train models which also compute loss on <pad> token and then predict the entire sentence including pad, it has shown to work in some of our experiments
                            length_range = [args.max_length]           

                        print(length_range)
                    
                        for sent_length_ in length_range:
                            # prefix_length is used to indicate if instead of predicting the entire sentence via optimization, we want to fix a prefix (of specified length) and predict the remaining suffix. We use part of the beam search prediction as the prefix. 
                            if args.prefix_length > 0:
                                sent_length = sent_length_ - args.prefix_length
                                target_prefix = predicted_batch[:, :args.prefix_length]
                            else:
                                sent_length = sent_length_
                                target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                            
                            if sent_length <= 0:
                                continue
                            if sent_length > args.max_allowed_length:
                                #max_allowed_length is just to make sure things don't go out of memory,
                                old_l = sent_length
                                sent_length = args.max_allowed_length
                                print(f"changed output length to {sent_length} from {old_l} to avoid GPU overflow. This is a temporary solution")
                            else:
                                print("predicting a sentence length: ", sent_length)
                                
                            if args.target_type == "simplex": # use V sized real vector for each token and apply softmax before output
                                outputs = TargetSimplex(
                                    vocabsize=primary_vocab_size,
                                    sent_length=sent_length,
                                    batch_size=batch_size,
                                    device=device,
                                    temperature=args.decode_temperature,
                                    st=args.st,
                                    init_value=source_batch[:,1:-1] if args.init == "source" else None,
                                    random_init=args.init == "random",
                                    do_sample=args.expgd_do_sample,
                                    top_p=args.expgd_top_p,
                                    top_k=args.expgd_top_k,
                                    embed_scales=embed_scales
                                )
                            elif args.target_type == "probs": # use V sized vector which sums to one for each token and apply softmax before output
                                init_value = None
                                break_after=False
                                if args.init == "source": #initialize the target with the source
                                    init_value = source_batch
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after=True
                                    # print(source_batch, init_value, sent_length, init_value)
                                elif args.init == "target": #initialize the target with the autoregressive output
                                    init_value = target_batch
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after=True
                                    # print(source_batch, init_value)
                                
                                outputs = TargetProbability(
                                    vocabsize=primary_vocab_size,
                                    sent_length=sent_length,
                                    batch_size=batch_size,
                                    device=device,
                                    st=args.st,
                                    init_value=init_value,
                                    random_init=args.init == "random",
                                    do_sample=args.expgd_do_sample,
                                    top_p=args.expgd_top_p,
                                    top_k=args.expgd_top_k,
                                    embed_scales=embed_scales,
                                    max_steps=args.optim_steps
                                )
                            elif args.target_type == "embeds":
                                init_value = None
                                break_after=False
                                if args.init == "source": #initialize the target with the source
                                    init_value = embed_luts[0](source_batch)
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after=True
                                    # print(source_batch, init_value, sent_length, init_value)
                                elif args.init == "targettarget": #initialize the target with given target
                                    init_value = embed_luts[0](target_batch)
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after=True 
                                    print(predicted_batch.size())   
                                    print(sent_length)
                                elif args.init == "target": #initialize the target with the autoregressive output
                                    init_value = embed_luts[0](predicted_batch)
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after=True 
                                    print(predicted_batch.size())   
                                    print(sent_length)
                                elif args.init == "random_vocab":
                                    random_indices = torch.multinomial(torch.ones(primary_vocab_size,)/primary_vocab_size, num_samples=batch_size*sent_length, replacement=True).view(batch_size, sent_length).to(device)
                                    init_value = embed_luts[0](random_indices)
                                elif args.init == "embedgd-zeros":
                                    if args.target_tokenize_different:
                                        with primary_tokenizer.as_target_tokenizer():
                                            indices = torch.empty((batch_size, sent_length)).long().fill_(primary_tokenizer.eos_token_id).to(device)
                                    else:
                                        indices = torch.empty((batch_size, sent_length)).long().fill_(primary_tokenizer.eos_token_id).to(device)
                                    # print(primary_tokenizer.decode(indices[0]))
                                    init_value = embed_luts[0](indices)
                                elif args.init == "zeros":
                                    indices = torch.zeros((batch_size, sent_length)).long().to(device)
                                    init_value = embed_luts[0](indices)

                                
                                final_bias = None
                                if args.final_bias:
                                    final_bias = lossfns[0].model.final_logits_bias

                                outputs = TargetEmbeddings(
                                    embed_dim=primary_embed_dim,
                                    embed_lut=embed_luts[0],
                                    sent_length=sent_length,
                                    batch_size=batch_size,
                                    device=device,
                                    st=args.st,
                                    init_value=init_value,
                                    random_init=args.init == "random",
                                    sampling_strategy=args.sampling_strategy,
                                    sampling_strategy_k=args.sampling_strategy_k,
                                    embed_scales=embed_scales,
                                    metric=args.metric,
                                    same_embed=args.same_embeds,
                                    final_bias=final_bias,
                                    eos_token_id=primary_tokenizer.eos_token_id
                                )
                            else:
                                raise ValueError("Wrong target_type")

                            if len(losses) > 1:
                                lambda_ = Lambda(count=len(epsilons))
                                if use_cuda:
                                    lambda_.cuda()

                            optimizer = Optimizer.from_opt(outputs, args)
                            cur_lr = args.lr
                            # print(optimizer._optimizer.param_groups)
                            # input()
                            if len(losses) > 1:
                                old_optim = args.optim
                                args.optim = "gradascent"
                                old_lr = args.lr
                                args.lr = args.lambda_lr
                                optimizer_lambda = Optimizer.from_opt(lambda_, args)
                                args.optim = old_optim
                                args.lr = old_lr

                            best_loss = [None] * batch_size
                            best_allsat = [False] * batch_size
                            best_repeat_count = [0] * batch_size
                            best_losses = [[None] * batch_size for _ in range(len(losses))]
                            best_step = -100
                            
                            best_pred_tokens = [None] * batch_size
                            best_prediction_set = [set() for _ in range(batch_size)]
                            best_pred_probs = [None] * batch_size
                            best_index = [-1 for i in range(batch_size)]
                            
                            scaler = None
                            if args.model_dtype == "fp16" and args.fp16_source == "pytorch":
                                scaler = torch.cuda.amp.GradScaler()
                        
                            for lossid, lossname in enumerate(losses):
                                losslists[lossid].append([])

                            broken = False
                            prev_loss = None
                            dynamic_lambda_update_prev_loss = None
                            same_loss_count = 0
                            dynamic_loss_update_same_loss_count = 0
                            starttime = time.time()
                            repeat_counts = [0] * batch_size

                            for step in range(args.optim_steps):
                                try:
                                    with torch.cuda.amp.autocast():
                                        losses_for_backward = []
                                        logging_outputs = []

                                        # print(optimizer.new_predictions)
                                        pred_embeds, pred_tokens, pred_probs = outputs.forward_multiple(embed_luts, new_predictions=getattr(optimizer._optimizer, "new_predictions", None))  # forward
                                        if not args.time and args.debug:
                                            def get_sent(tokens, tokenizer):
                                                batch = []
                                                if args.target_tokenize_different:
                                                    with tokenizer.as_target_tokenizer():
                                                        for toks in tokens:
                                                            batch.append(tokenizer.decode(clean_output(toks.tolist(), -1, allow_first_eos=losses[0] == "bart")))
                                                else:
                                                    for toks in tokens:
                                                        batch.append(tokenizer.decode(clean_output(toks.tolist(), -1, allow_first_eos=losses[0] == "bart")))
                                                return batch

                                            target_sents = get_sent(torch.cat([target_prefix, pred_tokens], dim=1), primary_tokenizer)
                                            print(target_sents, end="\n")
                                        
                                        original_preds = None
                                        if len(pred_embeds) > 1:
                                            original_preds = pred_embeds[1]

                                        # print("what", args.use_context)
                                        for lossid, lossname in enumerate(losses):
                                            lossvalue, logging_output =\
                                                lossfns[lossid].compute_loss(
                                                    [source_batch, target_prefix], 
                                                    [pred_tokens, pred_embeds[0][lossid], pred_probs], 
                                                    additional_batch=additional_batch, 
                                                    context_batch=context_batch,
                                                    use_context=args.use_context,
                                                    embed_scale=embed_scales[lossid], 
                                                    label_id=label_ids[lossid],
                                                    keyword=keywords[lossid],
                                                    original_preds=original_preds,
                                                    kweight=new_kweight,
                                                    step=step
                                                )

                                            losslists[lossid][-1].append(lossvalue.sum().item())  #for logging
                                            losses_for_backward.append(lossvalue)  # for backward
                                            logging_outputs.append(logging_output)
                                        
                                        optimizer.zero_grad(set_to_none=True)
                                        outputs.zero_grad()
                                        if len(losses) > 1:
                                            optimizer_lambda.zero_grad(set_to_none=True)
                                            lambda_.zero_grad()

                                        for model in name2model.values():
                                            model.zero_grad(set_to_none=True)
                                        
                                        if args.linear_scale == "true": # no lagragian, plain old linear sum
                                            # 
                                            # grads = []
                                            # if args.debug and args.debug_gradients == "true":
                                            #     for sid in range(len(losses_for_backward)):
                                            #         optimizer.backward(losses_for_backward[sid], retain_graph=True, scaler=scaler)
                                            #         grad = []
                                            #         for p in outputs.parameters():
                                            #             grad.append(p.grad.data)
                                            #             param_norm = p.grad.data.norm(2, -1).sum(dim=0)
                                            #             print(sid, "for theta", param_norm)
                                            #         grads.append(grad[0])
                                            #         optimizer.zero_grad(set_to_none=True)
                                            #         outputs.zero_grad(set_to_none=True)
                                            #         for modelname in loss2modelname.values():
                                            #             name2model[modelname].zero_grad(set_to_none=True) 
                                            #     graddot = (grads[0] * grads[1]).sum(dim=-1)
                                            #     print(graddot)
                                            #     grads0norm = torch.nn.functional.normalize(grads[0], p=2, dim=-1)
                                            #     grads1norm = torch.nn.functional.normalize(grads[1], p=2, dim=-1)
                                            #     print((grads0norm * grads1norm).sum(dim=-1))
                                                # input()
                                            # else:
                                            # total_loss = betas[0] * losses_for_backward[0]
                                            total_loss = 0
                                            cur_epsilons = [] # just for avoiding syntax errors, epsilons are useless in this setting
                                            for sid in range(len(losses_for_backward)):
                                                total_loss = total_loss + betas[sid] * losses_for_backward[sid]
                                                cur_epsilons.append(0.0)
                                            
                                            total_batchloss = total_loss.sum()
                                            optimizer.backward(total_batchloss, retain_graph=False, scaler=scaler)
                                        else:
                                            total_loss = 0.0
                                            total_loss = losses_for_backward[0]
                                            # total_loss_for_lambda = 0.0
                                            cur_epsilons = []
                                            # print(total_loss.item(), end=", ")

                                            constraint_values = []
                                            for sid in range(1, len(losses_for_backward)): #the secondary losses or constraints
                                                cur_epsilon = get_epsilon(step, epsilons[sid-1], min_epsilons[sid-1], epsilon_warmup_steps[sid-1], epsilon_cooldown_steps[sid-1], epsilon_decay_functions[sid-1])
                                                constraint_value = (cur_epsilon - losses_for_backward[sid]).detach()
                                                damp = args.dampness * constraint_value
                                                # lambda_.set_active(sid-1, constraint_value)
                                                mask = lambda_.get_mask(sid-1, damp)
                                                # mask = 1.0

                                                closs_for_theta = lambda_.get_loss(sid - 1, damp * mask, (cur_epsilon - losses_for_backward[sid]))
                                                total_loss = total_loss - closs_for_theta
                                                
                                                cur_epsilons.append(cur_epsilon)                             
                                                constraint_values.append(constraint_value.item())
                                        
                                            total_batchloss = total_loss.sum()
                                            optimizer.backward(total_batchloss, retain_graph=False, scaler=scaler)

                                        if args.debug and args.debug_gradients == "true":
                                            total_norm = 0
                                            gi=0
                                            for p in outputs.parameters():
                                                gi+=1
                                                param_norm = p.grad.data.norm(2, -1).sum(dim=0)
                                                # print(p.dtype)
                                                print("for theta", param_norm)
                                            for p in lambda_.parameters():
                                                print("for lambda", p.grad)
                                            
                                            # input()
                                    
                                    if logging_outputs[0].get('entropy', None) is not None:
                                        optimizer.step(scaler=scaler, entropy=logging_outputs[0].get('entropy', None))
                                    else:
                                        optimizer.step(scaler=scaler)
                                    
                                    update_lr_condition = "none"
                                    if args.linear_scale != "true" and  len(losses) > 1:
                                        sats = torch.Tensor(constraint_values).ge(0.).to(device)
                                        update_lambda_condition = (step % args.lambda_update == 0)
                                        lambda_mask = float(update_lambda_condition) * torch.ones_like(sats)
                                        
                                        lambda_mask += (1-sats.float()) * (lambda_.is_zero())
                                        # if not sats.all() and (lambda_.any_zero()):
                                        #     print("funky new update")
                                        #     update_lambda_condition = True
                                        #     lambda_mask = torch.ones_like(sats)
                                        lambda_mask += sats.float()

                                        # if step > args.lambda_update:
                                    
                                    total_batchlossitem = total_batchloss.item()
                                    # if dynamic_lambda_update_prev_loss is not None:
                                        # print(abs(total_batchlossitem - dynamic_lambda_update_prev_loss))
                                    if dynamic_lambda_update_prev_loss is not None and abs(total_batchlossitem - dynamic_lambda_update_prev_loss) <= 1e-6:
                                        repeat_counts[0] += 1
                                        if args.linear_scale != "true" and  len(losses) > 1 and args.dynamic_lambda_update:
                                            lambda_mask = (1 - sats.float())
                                            # print("what now", total_batchlossitem, dynamic_lambda_update_prev_loss, constraint_values, sats.float())
                                            # if sats.all(): #constraints are satisfied
                                            #     update_lambda_condition = False
                                            #     print("constraints are satisfied and output is not changing, lambdas will not update!")
                                            # else:
                                            #     update_lambda_condition = True

                                        if args.dynamic_lr_update and best_allsat[0] is not None and best_allsat[0]:
                                            update_lr_condition = "increase"
                                    else:
                                        repeat_counts[0] = 1

                                    
                                    dynamic_lambda_update_prev_loss = total_batchlossitem

                                    if update_lr_condition == "increase":
                                        cur_lr = optimizer._optimizer.update_lr(min(cur_lr + args.lr_update_size, args.max_lr))

                                    if args.linear_scale != "true" and len(losses) > 1:
                                        # print(lambda_mask, repeat_counts)
                                        optimizer_lambda._optimizer.set_mask(lambda_mask.clamp(max=1.0, min=0.0))
                                        optimizer_lambda.step()
                                        lambda_.make_positive()
                                    
                                    

                                        # total_batchloss_for_lambda = total_loss_for_lambda.sum()
                                        # optimizer_lambda.backward(total_batchloss_for_lambda, retain_graph=True, scaler=scaler)
                                        
                                    gc.collect()

                                    
                                    # outputs.printparams()
                                    # input()
                                    
                                    
                                    # print(repeat_counts, allsat)
                                    cur_losses = []
                                    for b in range(batch_size):
                                        cur_loss = 0.0
                                        for beta, lossval in zip(betas, losses_for_backward):
                                            cur_loss = cur_loss + beta * lossval[b].item()     
                                        cur_losses.append(cur_loss)
                                        
                                        constrained = []
                                        allsat = True
                                        for i in range(1, len(losses)):
                                            if losses_for_backward[i] <= min_epsilons[i - 1]:
                                                constrained.append("sat")
                                            else:
                                                constrained.append("vio")
                                                allsat=False
                                        
                                        if args.show_all_outputs and len(losses) > 1 and allsat:
                                            best_prediction_set[b].add(target_sents[b])
                                            
                                        constrained = ",".join(constrained)

                                        modify_condition =\
                                            args.selection_criterion == "last" or\
                                            (best_loss[b] is None and args.selection_criterion == "weighted_sum") or\
                                            (best_loss[b] is not None and args.selection_criterion == "weighted_sum" and best_loss[b] > cur_loss)
                                        
                                        # print(repeat_counts, allsat, best_loss, best_allsat)
                                        if not modify_condition and args.selection_criterion == "mrr_allsat":
                                            modify_condition =\
                                                (best_loss[b] is None and allsat and repeat_counts[b] == 2) or\
                                                (best_loss[b] is not None and best_allsat[b] and allsat and repeat_counts[b] == 2)
                                            # print(modify_condition)
                                            # modify_condition = (best_loss[b] is not None and best_allsat[b] and allsat and repeat_counts[b] == 2)

                                        elif not modify_condition and args.selection_criterion == "primary_allsat":
                                            modify_condition =\
                                                (best_loss[b] is None and allsat) or\
                                                (best_loss[b] is not None and not best_allsat[b] and allsat) or\
                                                (best_allsat[b] and allsat and best_loss[b] > cur_loss)

                                        # step>20 and 

                                        if modify_condition:
                                            if args.dynamic_lr_update:
                                                print("resetting the learning rate, a constraint has been satisfied")
                                                cur_lr = optimizer._optimizer.update_lr(args.lr)
                                                # lambda_.reset(torch.Tensor(constraint_values).le(0.).to(device))
                                            if args.selection_criterion != "last":
                                                print(f"modify condition @{step}", time.time()-starttime, end="\n")
                                            best_loss[b] = cur_loss
                                            best_allsat[b] = allsat
                                            best_repeat_count[b] = repeat_counts[b]
                                            for i in range(len(losses)):
                                                best_losses[i][b] = losses_for_backward[i][b].item()
                                            
                                            best_pred_tokens[b] = pred_tokens[b]
                                            best_index[b] = step
                                            # best_pred_probs[b] = (pred_probs[b].cpu(), logging_outputs[0]["lm_logprobs"][b])
                                            best_constrained = constrained
                                            best_step = step
                                        # elif best_step < step - 1 and args.dynamic_lr_update:
                                        #     print("resetting the learning rate, the constraint just got unsatisfied")
                                            
                                    if not args.time and step > 0 and step % args.log_interval == 0:
                                        if len(losses) > 1:
                                            log = f"beam cons: {predicted_allsat}; "
                                            log = f"Step {step}: lr:{cur_lr}; total_loss:{total_batchloss:.4f}; current [loss:{sum(cur_losses):.4f}; l:{','.join([f'{x:.4f}' for x in lambda_().tolist()])}; e:{','.join([f'{x:.4f}' for x in cur_epsilons])}; cons:{constrained}; "
                                            for i in range(len(losslists)):
                                                log = log + f" {lossabbr[i]}:{losslists[i][-1][-1]:.4f}; "
                                            
                                            if best_loss[0] is not None:
                                                log = log[:-1] + f"] |||| best [cur_loss:{sum(best_loss):.4f}; cons:{best_constrained};  "
                                                for i in range(len(best_losses)):
                                                    log = log + f"{lossabbr[i]}:{sum(best_losses[i]):.4f}; "
                                                log = log[:-1] + f"@ step #{best_index[-1]}" 
                                                log = log + "]"
                                            else:
                                                log = log[:-1] + f"] |||| best [none of the generations so far satisfies constraints]"
                                            print(log)
                                        else:
                                            log = f"Step {step}: lr:{cur_lr}; loss:{total_batchloss:.4f}; current [loss:{sum(cur_losses):.4f}; "
                                            for i in range(len(losslists)):
                                                log = log + f" {lossabbr[i]}:{losslists[i][-1][-1]:.4f}; "

                                            if best_loss[0] is not None:
                                                log = log[:-1] + f"] best [loss:{sum(best_loss):.4f} "
                                                for i in range(len(best_losses)):
                                                    log = log + f"{lossabbr[i]}:{sum(best_losses[i]):.4f}; "
                                                log = log[:-1] + f" at step {best_index[-1]}" 
                                                log = log + "]"
                                            else:
                                                log = log[:-1] + f"] |||| best [none of the generations so far satisfies constraints]"
                                            print(log, end="\n")
                                    
                                    del losses_for_backward

                                    if args.early_stop_steps > 0: #[0] is batch index, batch size in our case in 1 always so it doesn't matter.
                                        # print(args.selection_criterion)
                                        # print(lengthwise_best_prediction)
                                        early_stop_condition =\
                                            ("allsat" in args.selection_criterion and best_allsat[0]) or\
                                            (args.selection_criterion == "weighted_sum") or\
                                            (args.selection_criterion == "last")

                                        # print(early_stop_condition)
                                        if prev_loss is not None and abs(cur_loss - prev_loss) <= 1e-6:
                                            same_loss_count += 1
                                        else:   
                                            same_loss_count = 0

                                        if early_stop_condition and same_loss_count >= args.early_stop_steps:
                                            print(f"Early stop at @{step} with a loss value of {cur_loss} and satisfied constraints")
                                            break
                                        elif same_loss_count >= args.early_stop_steps + 2 * args.lambda_update:
                                            print(f"Early stop at @{step} with a loss value of {cur_loss} and unsatisfied constraints")
                                            break
                                            
                                        prev_loss = cur_loss



                                except KeyboardInterrupt:
                                    print("skipping remaining optimizing steps and showing the best option so far")
                                    broken=True
                                    break

                            if args.time:
                                r = time.time()-starttime
                                print(r)
                                avg_time += r

                            predictions = []
                            prediction_idss = []
                            broken_skip = False
                            skip_printing = False
                            for b, item in enumerate(best_pred_tokens):
                                if item is None and broken:
                                    skip_printing = True
                                    if broken:
                                        broken_skip=input("Skip this input entirely? yes(y)/no(continue)/press ctrl+c to exit")
                                        broken_skip = broken_skip.lower() == "y"
                                        break
                                if (args.only_mucoco == "false" and not best_allsat[b]) or (item is None): #item is none happens when optimization fails
                                    prediction_ids = ", ".join([str(idx) for idx in AR_predicted_indices[0].tolist()])
                                    prediction_indices = AR_predicted_indices[0].tolist()
                                    prediction = AR_prediction

                                    lossvalue = 0.0
                                    for lossid in range(len(betas)):
                                        lossvalue += betas[lossid] * predictedlosslists[-1][lossid][b] # VERIFICATION NEEDED
                                    print(f"best prediction is from beam search, all constraints were not satisfied, allsat={lengthwise_best_prediction[b][2]}")
                                else:
                                    prediction_ids = ", ".join([str(x) for x in target_prefix[b].tolist()])
                                    prediction_ids +=   f'[{", ".join([str(x) for x in item.tolist()])}]'
                                    prediction_indices = target_prefix[b].tolist() + item.tolist()
                                    
                                    targets = clean_output(item.tolist(), primary_tokenizer.eos_token_id, allow_first_eos=losses[0] == "bart")
                                    if args.target_tokenize_different:
                                        with primary_tokenizer.as_target_tokenizer():
                                            prediction = primary_tokenizer.decode(target_prefix[b].tolist() + targets)
                                    else:
                                        prediction = primary_tokenizer.decode(target_prefix[b].tolist() + targets)

                                    print("best prediction at step",best_index[b])
                                    lossvalue = best_loss[b]

                                    modify_condition =\
                                        lengthwise_best_prediction[b] is None or\
                                        (args.selection_criterion == "weighted_sum" and lengthwise_best_prediction[b][1] > lossvalue)
                                    
                                    if not modify_condition and args.selection_criterion in ["primary_allsat", "mrr_allsat"]:
                                        modify_condition =\
                                            (not lengthwise_best_prediction[b][2] and best_allsat[b]) or\
                                            (lengthwise_best_prediction[b][2] and best_allsat[b] and lengthwise_best_prediction[b][1] > lossvalue)
                                    
                                    # elif not modify_condition and args.selection_criterion == "mrr_allsat":
                                    #     modify_condition =\
                                    #         (not lengthwise_best_prediction[b][2] and best_allsat[b] and best_repeat_count[b] >= 2) or\
                                    #         (lengthwise_best_prediction[b][2] and lengthwise_best_prediction[b][1] > lossvalue)
                                            # (lengthwise_best_prediction[b][2] and lengthwise_best_prediction[b][4] >= 2 and lengthwise_best_prediction[b][1] > lossvalue)
                                        
                                    
                                    if modify_condition:
                                        if args.debug:
                                            print("modify condition satisfied", end="\n")
                                        else:
                                            outallsatf.write("modify_condition satisfied ")
                                        lengthwise_best_prediction[b] = (prediction, lossvalue, best_allsat[b], prediction_indices, best_repeat_count[b])
                                
                                prediction_idss.append(prediction_ids)
                                predictions.append(prediction)

                            if args.debug and not skip_printing:                    
                                for i, item in enumerate(best_pred_tokens):
                                    print(f"predicting length: {sent_length}")
                                    print("Given source:", source_text)
                                    print("Given target: ", target_text)
                                    print("Given additional: ", additional_text)
                                    print(f"Prediction ids: {prediction_ids}")
                                    print(f"Prediction: {prediction}")
                                    print("All generations that satisfied the constraints: ", best_prediction_set[i])

                                    out = []
                                    # print(predictedlosslists)
                                    # input()
                                    # if target_batch is not None:
                                    #     for lossid in range(len(losses)):
                                    #         out.append(f"Gold {lossabbr[lossid]}: {predictedlosslists[lossid][-1]}")
                                    #out.append(f"Source {lossabbr[0]}: {source_primarylosslist[-1]}")
                                    # print("; ".join(out))

                                    out = []
                                    for lossid in range(len(losses)):
                                        out.append(f"{losses[lossid]}: {best_losses[lossid][i]}")
                                    print("; ".join(out))
                                
                                
                                if broken:
                                    broken_skip=input("Skip this input entirely? yes(y)/no(continue)/press ctrl+c to exit")
                                    broken_skip = broken_skip.lower() == "y"

                            all_stepcounts += best_index

                            optimizer.zero_grad(set_to_none=True)
                            del outputs
                            del optimizer
                            if len(losses) > 1:
                                optimizer_lambda.zero_grad()
                                del optimizer_lambda
                                del lambda_
                            for modelname in loss2modelname.values():
                                name2model[modelname].zero_grad(set_to_none=True) 
                            torch.cuda.empty_cache()

                            if args.debug and broken_skip: 
                                break
                            
                            if break_after:
                                break
                        
                        ### RESTART HERE
                        b=0
                        if lengthwise_best_prediction[b] is None or not lengthwise_best_prediction[b][2]: #constraints are not satisfied
                            if restart_idx < args.restarts: #atleast one more restart is left
                                continue #skip printing and loop over
                            elif lengthwise_best_prediction[b] is None:
                                lengthwise_best_prediction = [("", -1, False, [], -1)] #just blank which didn't satisfy the constraints
  
                        if args.debug:
                            if not skip_printing:
                                for b in range(batch_size):
                                    print("sample #"+str(sample_idx), f"repeat count: {lengthwise_best_prediction[b][4]}" , "best prediction for all lengths: ", lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n")
                        else:   
                            if args.output_style == "text":
                                for b in range(batch_size):
                                    outf.write(lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n")
                                    outf.flush()
                                    outallsatf.write(str(lengthwise_best_prediction[b][2]) + "\n")
                                    outallsatf.flush()
                            else:
                                if sample_idx == 0:
                                    output = {
                                        "prompt":{
                                            "text":source_text,
                                            "tokens":source_indices_write}, 
                                        "generations":[{
                                            "text": lengthwise_best_prediction[b][0],
                                            "tokens": lengthwise_best_prediction[b][3],
                                            "allsat": lengthwise_best_prediction[b][2],
                                            "repeat_count": lengthwise_best_prediction[b][4],
                                            "mucoco": True
                                            }]
                                    }
                                else:
                                    output['generations'].append(
                                        {
                                            "text": lengthwise_best_prediction[b][0],
                                            "tokens": lengthwise_best_prediction[b][3],
                                            "allsat": lengthwise_best_prediction[b][2],
                                            "repeat_count": lengthwise_best_prediction[b][4],
                                            "mucoco": True
                                        }
                                    )
                                
                                if sample_idx + 1 == args.num_samples:
                                    json.dump(output, outf)
                                    outf.write("\n")
                                    outf.flush()

                                    outallsatf.write(str(lengthwise_best_prediction[b][2]) + "\n")
                                    outallsatf.flush()
                                    #VERIFY
                        print(f"required output achieved or number of restarts ran out at attempt #{restart_idx+1}")
                        break # don't restart if already reached here

                    else: # skipping mucoco and writing beam search output 
                        if ask_skip != "y":
                            if args.debug:
                                print("Skipping this example. the beam search output already satisfies all the constraints or there's no constraints")
                                for b in range(batch_size):
                                    print("best prediction for all lengths: ", lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n")
                            else:
                                print("Skipping this example. the beam search output already satisfies all the constraints or there's no constraints")
                                if args.output_style == "text":
                                    for b in range(batch_size):
                                        outf.write(lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n")
                                        outf.flush()
                                        outallsatf.write(str(lengthwise_best_prediction[b][2]) + "\n")
                                        outallsatf.flush()
                                else:
                                    for b in range(batch_size):
                                        if sample_idx == 0:
                                            output = {
                                                "prompt":{
                                                    "text":source_text,
                                                    "tokens":source_indices_write}, 
                                                "generations":[{
                                                    "text": lengthwise_best_prediction[b][0],
                                                    "tokens": lengthwise_best_prediction[b][3],
                                                    "allsat": lengthwise_best_prediction[b][2],
                                                    "mucoco": False
                                                    }]
                                            }
                                            # print(output)
                                        else:
                                            output['generations'].append(
                                                {
                                                    "text": lengthwise_best_prediction[b][0],
                                                    "tokens": lengthwise_best_prediction[b][3],
                                                    "allsat": lengthwise_best_prediction[b][2],
                                                    "mucoco": False
                                                }
                                            )
                                    
                                    if sample_idx + 1 == args.num_samples:
                                        json.dump(output, outf)
                                        outf.write("\n")
                                        outf.flush()
                                        #VERIFY
                        break # don't restart
                
                    if args.debug and broken_skip:
                        break

                if args.debug and broken_skip: 
                    break

            del source_batch
            del target_batch
            del additional_batch
            del for_predicted_source_batch
            del predicted_batch
            source_batch = []
            target_batch = []
            for_predicted_source_batch = []
            additional_batch = []
            predicted_batch = []
            context_batch = []

    if args.outfile is not None:
        outf.close()
        outallsatf.close()
    print("average numbers of steps to converge =", np.mean(all_stepcounts))
    print("average time = ", avg_time/c)

def sentence_completion():
    pass

def clean_output(tokens, eos_token_id, return_tensors=False, allow_first_eos=False, skip_special_tokens=[], sentence_complete=False):
    # print(tokens)
    new_tokens = []
    for i, tok in enumerate(tokens):
        if tok == eos_token_id and (not allow_first_eos or i > 0):
            break
        
        if (tok not in skip_special_tokens):
            new_tokens.append(tok)
        
    if return_tensors:
        return torch.LongTensor([new_tokens])
    return new_tokens
    
def cli_main():
    parser = options.get_parser()
    args = parser.parse_args()
    main(args)
