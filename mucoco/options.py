import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data", default=None, type=str, help="path of source/target sentences"
    )
    parser.add_argument(
        "--start-idx", default=0, type=int, help="can be used to specify a subset of input file to decode (start index)"
    )
    parser.add_argument(
        "--end-idx", default=-1, type=int, help="can be used to specify a subset of input file to decode (last index)"
    )
    parser.add_argument(
        "--datastyle", default="text", type=str, choices=["jsonl", "single-jsonl", "text"], help="jsonl or plaintext (text)"
    )
    parser.add_argument(
        "--jsonl-primary-key", default="context", type=str, help="primary key to access text in the jsonl file"
    )
    parser.add_argument(
        "--jsonl-secondary-key", default=None, type=str, help="secondary key to access text in the jsonl file"
    )
    parser.add_argument(
        "--jsonl-tertiary-key", default=None, type=str, help="tertiary key to access text in the jsonl file"
    )
    parser.add_argument(
        "--jsonl-tokenized", default="false", type=str, choices=["true", "false"],  help="the text in jsonl files is already tokenized if true"
    )
    parser.add_argument(
        "--additional-data", default=None, type=str, help="path of additional data used in some losses"
    )
    parser.add_argument("--max-length", default=None, type=int, help="L: the sentence length you want to predict at every step. Use this for models which have no padding/trained while also predicting the padding. Not used in experiments reported in the paper")
    parser.add_argument("--max-output-length", default=None, type=int, help="L: the sentence length you want to predict at every step. Use this for models which have no padding/trained while also predicting the padding. Not used in experiments reported in the paper")
    parser.add_argument("--max-prefix-length", default=50, type=int, help="L: the sentence length you want to predict at every step. Use this for models which have no padding/trained while also predicting the padding. Not used in experiments reported in the paper")
    parser.add_argument("--max-allowed-length", default=1000, type=int, help="This is the max length that will fit into the GPU, max_length <= max_allowed_length")
    parser.add_argument("--length-diff", default="0", type=str, help="change the length of the target by adding these values")
    parser.add_argument("--restarts", default=0, type=int, help="If the constraints are not satisfied in a run, restart the optimization")
    parser.add_argument("--num_samples", default=1, type=int, help="number of times to run the decoding process for each input")
    parser.add_argument(
        "--model", default=None, type=str, help="path to the trained lm"
    )
    parser.add_argument(
        "--model_types", default=None, type=str, help="data types of models"
    )
    parser.add_argument(
        "--tokenizer", default=None, type=str, help="path to the trained lm"
    )
    parser.add_argument("--target-tokenize-different", action="store_true", help="use target specific tokenizer")
    parser.add_argument("--use_context", type=str, default="false", help="use context to compute loss (for some of them atleast)")
    parser.add_argument("--random-example", type=str, default="true", help="use context to compute loss (for some of them atleast)")
    parser.add_argument("--final-bias", action="store_true", help="use target specific tokenizer")
    parser.add_argument("--show-all-outputs", action="store_true", help="show all valid outputs")
    parser.add_argument("--debug-gradients", default="false", choices=["true", "false"], help="")
    parser.add_argument("--allow-diff-vocab", action="store_true", help="show all valid outputs")
    parser.add_argument("--linear-scale", default="false", choices=["true", "false"], help="use a linear scale instead of lagrange multipliers")

    parser.add_argument(
        "--results-path", default=None, type=str, help="where to write results"
    )
    parser.add_argument(
        "--outfile", default=None, type=str, help="where to write results"
    )
    parser.add_argument(
        "--output-style", default="text", type=str, help="write output in jsonl or text format"
    )
    parser.add_argument("--cpu", action="store_true", help="use cpu instead of gpu")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--time", action="store_true", help="time mode")
    parser.add_argument("--beam", action="store_true", help="do beam search ")
    parser.add_argument("--st", action="store_true", help="do straight through")
    parser.add_argument("--gold-loss-epsilons", type=str, default="none", help="use gold loss as epsilons")
    parser.add_argument("--custom-epsilons", type=str, default="none", help="use gold loss as epsilons")
    parser.add_argument("--seed", default=10, type=int, help="random seed")
    parser.add_argument(
        "--log-interval", default=10, type=int, help="interval for logging"
    )

    parser.add_argument(
        "--target-type", default="simplex", type=str, choices=["embeds", "probs", "simplex"]
    )
    parser.add_argument(
        "--init",
        default="zeros",
        type=str,
        choices=["zeros", "random", "source", "target", "targettarget", "random_vocab", "embedgd-zeros"],
    )
    parser.add_argument(
        "--sampling-strategy",
        default="greedy",
        type=str,
        choices=["greedy", "beam", "topk", "topp"],
    )
    parser.add_argument(
        "--sampling-strategy-k",
        default="none",
        type=str,
        help="if topk or topk, what is the value of k or p"
    )
    parser.add_argument(
        "--AR-top-k",
        default=0,
        type=int,
        help="top k in autoregressive decoding setup (for baseline)"
    )
    parser.add_argument(
        "--AR-top-p",
        default=1.0,
        type=float,
        help="top p (nucleus) in autoregressive decoding setup (for baseline)"
    )
    parser.add_argument(
        "--AR-temperature",
        default=1.0,
        type=float,
        help="top p (nucleus) in autoregressive decoding setup (for baseline)"
    )
    parser.add_argument(
        "--topic-target",
        default="none",
        type=str,
        # choices=['computers', 'legal', 'military', 'politics', 'religion', 'science', 'space', 'pizza', 'none', "extreme", "medium", "mild"],
        help="different LR patterns for different indices, default is constant for all. It gets updated as the training progresses too."
    )
    parser.add_argument(
        "--keywords",
        default="the",
        type=str,
        help="keywords you want appearing in the output text"
    )
    parser.add_argument(
        "--keyword_topk",
        default=1,
        type=int,
        help="keywords you want appearing in the output text"
    )
    parser.add_argument(
        "--keyword_tau",
        default=0.01,
        type=float,
        help="keywords. Temperature for Gumbel Softmax"
    )
    parser.add_argument(
        "--topic-word-lists",
        default=None,
        type=str,
        help="path to folder containing wordlists"
    )
    parser.add_argument(
        "--embedgd-lr-pattern",
        default="constant",
        type=str,
        choices=['constant', 'linear', 'flat-linear', 'geometric', 'harmonic', 'rsqrt', 'block'],
        help="different LR patterns for different indices, default is constant for all. It gets updated as the training progresses too."
    )
    parser.add_argument(
        "--coeff-pattern",
        default="constant",
        type=str,
        choices=['constant', 'rsqrt', 'geometric', 'exp', 'rsqrt-freq'],
        help="different LR patterns for different indices, default is constant for all. It gets updated as the training progresses too."
    )
    parser.add_argument(
        "--embedgd-grad-distance",
        default="dot",
        type=str,
        choices=['l1', 'l2', 'dot', "cosine"],
        help="different LR patterns for different indices, default is constant for all. It gets updated as the training progresses too."
    )

    parser.add_argument("--num-samples", default=1, type=int, help="number of times to run the decoding process for each input")

    parser.add_argument("--embedgd-do-sample", default="false", choices=["true", "false"], help="sample when doing embedgd updates")
    parser.add_argument(
        "--embedgd-top-k",
        default=0,
        type=int,
        help="top k in autoregressive decoding setup (for baseline)"
    )
    
    parser.add_argument(
        "--embedgd-top-p",
        default=1.0,
        type=float,
        help="top k in autoregressive decoding setup (for baseline)"
    )

    parser.add_argument(
        "--embedgd-begin-temperature",
        default=1.0,
        type=float,
        help="top k in autoregressive decoding setup (for baseline)"
    )
    parser.add_argument(
        "--embedgd-final-temperature",
        default=0.01,
        type=float,
        help="top k in autoregressive decoding setup (for baseline)"
    )
    parser.add_argument(
        "--embedgd-temperature-reduction-steps",
        default=20,
        type=int,
        help="top k in autoregressive decoding setup (for baseline)"
    )
    parser.add_argument("--expgd-do-sample", action="store_true", help="sample when doing expgd updates")
    parser.add_argument(
        "--expgd-top-k",
        default=0,
        type=int,
        help="top k in autoregressive decoding setup (for baseline)"
    )

    parser.add_argument(
        "--repetition-penalty",
        default=0.0,
        type=float,
        help=""
    )
    
    parser.add_argument(
        "--expgd-top-p",
        default=1.0,
        type=float,
        help="top k in autoregressive decoding setup (for baseline)"
    )

    parser.add_argument(
        "--kweight",
        default=1.0,
        type=float,
        help="kweight. check gpt2.py"
    )
    parser.add_argument(
        "--embedgd-noise-variance",
        default=0.0,
        type=float,
        help="Use Langevin dynamics with gaussian noise for sampling"
    )
    parser.add_argument(
        "--embedgd-gumbel-noise-max",
        default=0.0,
        type=float,
        help="Use Langevin dynamics with gumbelnoise for sampling"
    )
    parser.add_argument(
        "--expgd-gumbel-noise-max",
        default=0.0,
        type=float,
        help="Use Langevin dynamics with gumbelnoise for sampling"
    )
    
    parser.add_argument(
        "--eos", action="store_true", help="end the sequence with eos tag"
    )
    parser.add_argument(
        "--show_warnings", action="store_true", help="show huggingface weird warnings"
    )
    parser.add_argument(
        "--damp", action="store_true", help="use Platt and Barr damp algorithm 1987"
    )
    parser.add_argument("--num-examples", default=50, type=int)
    parser.add_argument("--dampness", default=10.0, type=float)
    parser.add_argument("--epsilon", default=0.2, type=float)
    parser.add_argument("--epsilons", default=None, type=str)
    parser.add_argument("--min_epsilons", default=None, type=str)
    parser.add_argument("--epsilon_warmup_steps", default=None, type=str)
    parser.add_argument("--epsilon_cooldown_steps", default=None, type=str)
    parser.add_argument("--epsilon_decay_functions", default=None, type=str)
    parser.add_argument("--cache_dir", default="hf_cache", type=str)
    parser.add_argument(
        "--selection_criterion", default="primary_allsat", help="", choices=['weighted_sum', 'primary_allsat', "last", "mrr_allsat"] 
    ) #mrr = most recent repetition
    parser.add_argument("--early-stop-steps", default=-1, type=int, help="stop if the output hasn't changed in this many steps and the constraints are satisfied")
    parser.add_argument(
        "--bos", action="store_true", help="add bos tag to the sequence"
    )
    parser.add_argument(
        "--length-normalize",
        action="store_true",
        help="normalize loss by length of sequence (for token level losses only)",
    )
    parser.add_argument(
        "--scale_loss",
        default="none",
        choices=["none", "sigmoid", "linear", "gradnorm"],
        help="",
    )
    parser.add_argument("--beam-size", default=1, type=int)
    parser.add_argument("--suffix-length", default=0, type=int)
    parser.add_argument("--prefix-length", default=0, type=int)
    parser.add_argument("--label-id", default="none", type=str, help="for classification losses, which is the label of interest")
    parser.add_argument("--suffix-source", default=None, type=str)
    parser.add_argument(
        "--decode-temperature", default=0.1, type=float, help="softmax temperature"
    )
    parser.add_argument(
        "--betas", default="1.0", type=str, help="weights for different loss components"
    )
    parser.add_argument(
        "--loss",
        default="perplexity_loss",
        type=str,
        help="different loss components to combine",
    )
    parser.add_argument(
        "--lossabbr",
        default=None,
        type=str,
        help="abbreviations for loss names displayed during logging",
    )
    parser.add_argument(
        "--semantic_methods",
        default="bertscore",
        type=str,
        help="only for compute_semantic.py",
    )
    parser.add_argument(
        "--evaluation_metrics",
        default="fluency",
        type=str,
        help="only for all_evaluation_metrics.py",
    )
    parser.add_argument(
        "--match_with",
        default="reference",
        type=str,
        help="",
    )

    parser.add_argument("--optim", default="sgd", help="which optimizer")
    parser.add_argument("--optim-steps", default=10, type=int)
    parser.add_argument("--coeff-steps", default=200, type=int)
    parser.add_argument("--warmup-steps", default=1, type=int)
    parser.add_argument("--warmup-init-lr", default=None, type=float)
    parser.add_argument("--warmup-end-lr", default=None, type=float)
    parser.add_argument("--min-lr", default=None, type=float)
    parser.add_argument("--lambda-lr", default=1.0, type=float)
    parser.add_argument("--lambda-update", default=1, type=int)
    parser.add_argument("--dynamic-lambda-update", action="store_true")
    parser.add_argument("--dynamic-lr-update", action="store_true")
    parser.add_argument("--max-lr", default=1.0, type=float)
    parser.add_argument("--lr-update-size", default=0.05, type=float)

    parser.add_argument("--lr-decay", default=1.0, type=float)
    parser.add_argument("--start-decay-steps", default=1, type=int)
    parser.add_argument("--decay-steps", default=1, type=int)

    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--model_dtype", default="fp32", help="fp32 or fp16")
    parser.add_argument("--fp16_source", default="pytorch", help="apex or pytorch", choices=["apex", "pytorch"])
    parser.add_argument(
        "--decay-method", default=None, help="how to decay the learning rate"
    )
    parser.add_argument(
        "--embedgd-decay-method", default=None, help="how to decay the learning rate"
    )
    parser.add_argument("--half-lr", action="store_true", help="half the learning rate if the loss doesn't decrease for 20 steps")
    parser.add_argument("--always-mucoco", type=str, default="false", choices=["true", "false"], help="always use mucoco generated outputs (that is never pick beam search outputs even if the optimization fails)")
    parser.add_argument("--only-mucoco", type=str, default="false", choices=["true", "false"], help="always use mucoco generated outputs (that is never pick beam search outputs even if the optimization fails)")
    parser.add_argument("--same-embeds", action="store_true", help="use same embeddings for all models (used with target_type embeddings")
    parser.add_argument("--metric", default="dot", type=str, help="metric to compute NN for target_type embeddings")
    parser.add_argument("--loss-type", default="xentropy", type=str, help="")
    parser.add_argument(
        "--max-grad-norm", default=0.0, type=float, help="clip threshold of gradients"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1.0,
        type=float,
        metavar="LR_1,LR_2,...,LR_N",
        help="learning rate for the first N epochs; all epochs >N using LR_N"
        " (note: this may be interpreted differently depending on --lr-scheduler)",
    )
    parser.add_argument(
        "--adam-betas",
        default="(0.9, 0.999)",
        metavar="B",
        help="betas for Adam optimizer",
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
        metavar="D",
        help="epsilon for Adam optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=0.0,
        type=float,
        metavar="WD",
        help="weight decay",
    )
    parser.add_argument(
        "--sgd-momentum",
        "--sm",
        default=0.0,
        type=float,
        metavar="SM",
        help="Momentum in SGD",
    )
    parser.add_argument(
        "--embedgd-momentum",
        "--em",
        default=0.0,
        type=float,
        metavar="EM",
        help="Momentum in EmbedGD",
    )
    parser.add_argument(
        "--expgd-momentum",
        "--exm",
        default=0.0,
        type=float,
        metavar="EM",
        help="Momentum in EmbedGD",
    )
    parser.add_argument(
        "--sgd-nesterov",
        "--sn",
        action="store_true",
        help="Nesterov Momentum",
    )
    parser.add_argument(
        "--expgd_mw",
        default=1,
        type=int,
        help="MW=1|2 for exponentiated GD, mw=1 means exponential update, 2 means 1-eta * grad",
    )

    return parser
