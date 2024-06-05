export CUDA_VISIBLE_DEVICES=0
python teacher_eval.py \
    --text_encoder_name "google/flan-t5-large" \
    --unet_model_config "configs/diffusion_model_config.json" \
    --model="ckpt/teacher/pytorch_model_2_sigma_025.bin" \
    --test_file "/path/to/test.csv" \
    --num_steps 40 --guidance 3.5 --num_samples 1 --batch_size 1  \
    -output_dir "/path/to/output_dir/"
