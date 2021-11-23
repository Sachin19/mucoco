DATA_DIR=$1
hyp=$2
suffix=$3

refsource=$DATA_DIR/informal${suffix}
refstarget=$DATA_DIR/formal.ref0${suffix},$DATA_DIR/formal.ref1${suffix},$DATA_DIR/formal.ref2${suffix},$DATA_DIR/formal.ref3${suffix}

#deduplicate
python evaluation/postprocess.py $hyp $hyp.dedup

#evaluation wrt source
python evaluate.py --data $hyp,$refsource --model sentence-transformers/roberta-base-nli-stsb-mean-tokens --tokenizer sentence-transformers/roberta-base-nli-stsb-mean-tokens --evaluation_metrics sts_sim,cls_sim,wieting_sim,transfer,fluency,bleu,ppl --outfile ${hyp}_source_results.json --match_with source

#evaluation wrt reference
python evaluate.py --data $hyp,$refstarget --model sentence-transformers/roberta-base-nli-stsb-mean-tokens --tokenizer sentence-transformers/roberta-base-nli-stsb-mean-tokens --evaluation_metrics sts_sim,cls_sim,wieting_sim,transfer,fluency,bleu,ppl --outfile ${hyp}_reference_results.json --match_with reference