export CUDA_VISIBLE_DEVICES=0 

python ctm_inference.py \
    --text_encoder_name "google/flan-t5-large" \
    --ctm_unet_model_config "configs/diffusion_model_config.json" \
    --training_args '/path/to/checkpoints/soundctm_ckpt/030000/summary.jsonl' \
    --model "/path/to/checkpoints/soundctm_ckpt/030000/pytorch_model.bin" \
    --ema_model "/path/to/checkpoints/soundctm_ckpt/030000/ema_0.999_030000.pt" \
    --test_file "/path/to/audiocaps/test.csv" \
    --sampler 'determinisitc' --sampling_gamma 0. --omega 3.5 \
    --num_steps 1 --nu 1. --num_samples 1 --batch_size 1 --test_references '/path/to/audiocaps/test' \
    --output_dir "/path/to/output_dir/"

# python ctm_inference.py \
#     --text_encoder_name "google/flan-t5-large" \
#     --ctm_unet_model_config "configs/diffusion_model_config.json" \
#     --training_args '/path/to/checkpoints/soundctm_ckpt/030000/summary.jsonl' \
#     --model "/path/to/checkpoints/soundctm_ckpt/030000/pytorch_model.bin" \
#     --ema_model "/path/to/checkpoints/soundctm_ckpt/030000/ema_0.999_030000.pt" \
#     --test_file "/path/to/audiocaps/test.csv" \
#     --sampler 'determinisitc' --sampling_gamma 0. --omega 3.5 \
#     --num_steps 2 --nu 1. --num_samples 1 --batch_size 1 --test_references '/path/to/audiocaps/test' \
#     --output_dir "/path/to/output_dir/"

# python ctm_inference.py \
#     --text_encoder_name "google/flan-t5-large" \
#     --ctm_unet_model_config "configs/diffusion_model_config.json" \
#     --training_args '/path/to/checkpoints/soundctm_ckpt/030000/summary.jsonl' \
#     --model "/path/to/checkpoints/soundctm_ckpt/030000/pytorch_model.bin" \
#     --ema_model "/path/to/checkpoints/soundctm_ckpt/030000/ema_0.999_030000.pt" \
#     --test_file "/path/to/audiocaps/test.csv" \
#     --sampler 'determinisitc' --sampling_gamma 0. --omega 3.5 \
#     --num_steps 4 --nu 1. --num_samples 1 --batch_size 1 --test_references '/path/to/audiocaps/test' \
#     --output_dir "/path/to/output_dir/"

# python ctm_inference.py \
#     --text_encoder_name "google/flan-t5-large" \
#     --ctm_unet_model_config "configs/diffusion_model_config.json" \
#     --training_args '/path/to/checkpoints/soundctm_ckpt/030000/summary.jsonl' \
#     --model "/path/to/checkpoints/soundctm_ckpt/030000/pytorch_model.bin" \
#     --ema_model "/path/to/checkpoints/soundctm_ckpt/030000/ema_0.999_030000.pt" \
#     --test_file "/path/to/audiocaps/test.csv" \
#     --sampler 'determinisitc' --sampling_gamma 0. --omega 3.5 \
#     --num_steps 8 --nu 1.5 --num_samples 1 --batch_size 1 --test_references '/path/to/audiocaps/test' \
#     --output_dir "/path/to/output_dir/"

# python ctm_inference.py \
#     --text_encoder_name "google/flan-t5-large" \
#     --ctm_unet_model_config "configs/diffusion_model_config.json" \
#     --training_args '/path/to/checkpoints/soundctm_ckpt/030000/summary.jsonl' \
#     --model "/path/to/checkpoints/soundctm_ckpt/030000/pytorch_model.bin" \
#     --ema_model "/path/to/checkpoints/soundctm_ckpt/030000/ema_0.999_030000.pt" \
#     --test_file "/path/to/audiocaps/test.csv" \
#     --sampler 'determinisitc' --sampling_gamma 0. --omega 3.5 \
#     --num_steps 16 --nu 2.0 --num_samples 1 --batch_size 1 --test_references '/path/to/audiocaps/test' \
#     --output_dir "/path/to/output_dir/"