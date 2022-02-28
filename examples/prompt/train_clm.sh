source activate 2022
nvidia-smi | cat
TRANFORMERS=/projects/tir5/users/sachink/embed-style-transfer/related-work/transformers
echo "what"
python $TRANFORMERS/examples/pytorch/language-modeling/run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --do_train \
    --do_eval \
    --num_train_epochs 10\
    --fp16 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=16 \
    --output_dir /projects/tir5/users/sachink/embed-style-transfer/models/gpt2_wikitext_103 \
    --save_total_limit 1 \
    --overwrite_output_dir

# echo "what what hush hush"
# python -u $TRANFORMERS/examples/language-modeling/run_clm.old.py\
#     --model_type gpt2 \
#     --tokenizer_name gpt2 \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-103-raw-v1 \
#     --do_train \
#     --do_eval \
#     --fp16\
#     --line_by_line\
#     --num_train_epochs 10\
#     --per_device_train_batch_size=4 \
#     --per_device_eval_batch_size=4 \
#     --gradient_accumulation_steps=16 \
#     --save_total_limit 1\
#     --output_dir /projects/tir5/users/sachink/embed-style-transfer/models/gpt2_wikitext_103-pad\
#     --overwrite_output_dir

