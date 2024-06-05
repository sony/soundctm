#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7




accelerate launch --main_process_port 29504 ctm_train.py \
    --train_file "path/to/train.csv" \
    --text_encoder_name "google/flan-t5-large" --unet_model_config "configs/diffusion_model_config.json" \
    --freeze_text_encoder --ctm_unet_model_config "configs/diffusion_model_config.json" \
    --gradient_accumulation_steps 1 --per_device_train_batch_size 6 --tango_data_augment --augment_num 2 --num_train_epochs 40 --lr 0.00008 \
    --loss_type 'feature_space' --match_point 'zs' --unet_mode 'full' --w_min 2.0 --w_max 5.0 --unform_sampled_cfg_distill True \
    --num_heun_step 39 --start_scales 40 --end_scales 40 --mixed_precision 'bf16' \
    --diffusion_training True --denoising_weight 1.0 \
    --text_column caption --audio_column file_name --checkpointing_steps "best" --output_dir "/path/to/output_dir" \
    --with_tracking \

# MPIOPTS="-np 8 -x MASTER_ADDR=${HOSTNAME}"
# mpirun ${MPIOPTS} \
#     singularity exec --bind /data:/data --nv soundctm.sif /bin/bash -c "
#     ./python_accelerate.sh ctm_train.py \
#     --train_file "/path/to/train.csv" \
#     --text_encoder_name "google/flan-t5-large" --unet_model_config "configs/diffusion_model_config.json" \
#     --freeze_text_encoder --ctm_unet_model_config "configs/diffusion_model_config.json" \
#     --gradient_accumulation_steps 1 --per_device_train_batch_size 6 --tango_data_augment --augment_num 2 --num_train_epochs 40 --lr 0.00008 \
#     --loss_type 'feature_space' --match_point 'zs' --unet_mode 'full' --w_min 2.0 --w_max 5.0 --unform_sampled_cfg_distill True \
#     --num_heun_step 39 --start_scales 40 --end_scales 40 --mixed_precision 'bf16' \
#     --diffusion_training True --denoising_weight 5.0 \
#     --text_column caption --audio_column file_name --checkpointing_steps "best" --output_dir "/path/to/output_dir" \
#     --with_tracking \
#     "
