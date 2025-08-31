#!/bin/bash

# cd ..

# llm_id microsoft/phi-2
# llm_id tiiuae/falcon-7b-instruct
# llm_id mosaicml/mpt-7b-instruct
# llm_id meta-llama/Llama-2-7b-chat-hf
# llm_id=meta-llama/Llama-3.1-8B-Instruct
llm_id=meta-llama/Llama-3.2-3B-Instruct

# CUDA_VISIBLE_DEVICES=0 python src/generate_answers_llm.py \
#     --output_dir data/gen_res \
#     --llm_id $llm_id \
#     --model_max_length 4096 \
#     --load_full_corpus False \
#     --use_random False \
#     --use_adore False \
#     --gold_position 0 \
#     --num_documents_in_context 2 \
#     --get_documents_without_answer True \
#     --batch_size 320 \
#     --save_every 250

CUDA_VISIBLE_DEVICES=0 python src/generate_answers_llm.py \
    llm.llm_id=$llm_id \
    