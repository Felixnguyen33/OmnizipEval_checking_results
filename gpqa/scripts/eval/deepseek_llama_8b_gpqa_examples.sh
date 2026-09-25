CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 lm_eval \
    --model hf \
    --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,dtype=float16,parallelize=True,trust_remote_code=True \
    --tasks gpqa \
    --num_fewshot 5 \
    --gen_kwargs max_gen_toks=32768,temperature=0.6,top_p=0.95,do_sample=True \
    --batch_size auto \
    --apply_chat_template \
    --output_path results/gpqa_5shot_chat_deepseek_r1_distill_llama_8b