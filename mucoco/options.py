import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data", default=None, type=str, help="path of source/target sentences"
    )
    parser.add_argument(
        "--additional-data", default=None, type=str, help="path of additional data used in some losses"
    )
    parser.add_argument("--max-length", default=None, type=int, help="L: the sentence length you want to predict at every step. Use this for models which have no padding/trained while also predicting the padding. Not used in experiments reported in the paper")
    parser.add_argument("--max-allowed-length", default=20, type=int, help="This is the max length that will fit into the GPU, max_length <= max_allowed_length")
    parser.add_argument("--length_diff", default=0, type=int, help="change the length of the target by adding this value")
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
    parser.add_argument("--show-all-outputs", action="store_true", help="show all valid outputs")
    parser.add_argument("--allow-diff-vocab", action="store_true", help="show all valid outputs")
    parser.add_argument("--linear_scale", action="store_true", help="show all valid outputs")

    parser.add_argument(
        "--results-path", default=None, type=str, help="where to write results"
    )
    parser.add_argument(
        "--outfile", default=None, type=str, help="where to write results"
    )
    parser.add_argument("--cpu", action="store_true", help="use cpu instead of gpu")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--beam", action="store_true", help="do beam search ")
    parser.add_argument("--st", action="store_true", help="do straight through")
    parser.add_argument("--gold_loss_epsilons", action="store_true", help="use gold loss as epsilons")
    parser.add_argument("--seed", default=None, type=int, help="random seed")
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
        choices=["zeros", "random", "source", "target"],
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
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument(
        "--selection_criterion", default="weight_sum", help="", choices=['weighted_sum', 'primary_allsat']
    )
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
    parser.add_argument("--label-id", default="none", type=str, help="for binary classification losses, which is the label of interest")
    parser.add_argument("--suffix-source", default=None, type=str)
    parser.add_argument(
        "--decode-temperature", default=1.0, type=float, help="softmax temperature"
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
    parser.add_argument("--warmup-steps", default=1, type=int)
    parser.add_argument("--warmup-init-lr", default=None, type=float)
    parser.add_argument("--warmup-end-lr", default=None, type=float)
    parser.add_argument("--min-lr", default=None, type=float)
    parser.add_argument("--lambda-lr", default=1.0, type=float)

    parser.add_argument("--lr-decay", default=1.0, type=float)
    parser.add_argument("--start-decay-steps", default=1, type=int)
    parser.add_argument("--decay-steps", default=1, type=int)

    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--model_dtype", default="fp32", help="fp32 or fp16")
    parser.add_argument("--fp16_source", default="apex", help="apex or pytorch", choices=["apex", "pytorch"])
    parser.add_argument(
        "--decay-method", default=None, help="how to decay the learning rate"
    )
    parser.add_argument("--half-lr", action="store_true", help="half the learning rate if the loss doesn't decrease for 20 steps")
    parser.add_argument("--always_mucoco", action="store_true", help="half the learning rate if the loss doesn't decrease for 20 steps")
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
