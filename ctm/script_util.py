import argparse

import numpy as np
from tango_edm.models_edm import AudioDiffusionEDM

from ctm.resample import create_named_schedule_sampler

from .karras_diffusion import KarrasDenoiser


def ctm_train_defaults():
    return dict(
        # CTM hyperparams
        consistency_weight=1.0,
        loss_type='feature_space',
        unet_mode = 'full',
        schedule_sampler="uniform",
        weight_schedule="uniform",
        parametrization='euler',
        inner_parametrization='edm',
        num_heun_step=39,
        num_heun_step_random=True,
        training_mode="ctm",
        match_point='zs', #
        target_ema_mode="fixed",
        scale_mode="fixed",
        start_ema=0.999, 
        start_scales=40,
        end_scales=40,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7,
        latent_channels=8,
        latent_f_size=16,
        latent_t_size=256,
        lr=0.00008,
        weight_decay=0.0,
        lr_anneal_steps=0,
        ema_rate="0.999", 
        total_training_steps=600000,
        save_interval=3000,
        cfg_single_distill=False,
        single_target_cfg=3.5,
        unform_sampled_cfg_distill=True,
        w_min=2.0,
        w_max=5.0,
        distill_steps_per_iter=50000,
        
        sample_s_strategy='uniform',
        heun_step_strategy='weighted',
        heun_step_multiplier=1.0,
        auxiliary_type='stop_grad',
        time_continuous=False,
        
        diffusion_training=True,
        denoising_weight=1.,
        diffusion_mult = 0.7,
        diffusion_schedule_sampler='halflognormal',
        apply_adaptive_weight=True,
        dsm_loss_target='z_0', # z_0 or z_target
        diffusion_weight_schedule="karras_weight",
    )

def create_model_and_diffusion(args, teacher=False):
    schedule_sampler = create_named_schedule_sampler(args, args.schedule_sampler, args.start_scales) 
    diffusion_schedule_sampler = create_named_schedule_sampler(args, args.diffusion_schedule_sampler, args.start_scales) 
    
    model = AudioDiffusionEDM(
        text_encoder_name=args.text_encoder_name,
        unet_model_name=None,
        unet_model_config_path=args.unet_model_config,
        sigma_data=args.sigma_data,
        freeze_text_encoder=args.freeze_text_encoder,
        teacher=teacher,
        ctm_unet_model_config_path=args.ctm_unet_model_config
    )
    diffusion = KarrasDenoiser(
        args=args, schedule_sampler=schedule_sampler,
        diffusion_schedule_sampler=diffusion_schedule_sampler,
        # feature_networks=feature_networks,
    )
    return model, diffusion

def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter,
):
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "progressive":
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1

        elif target_ema_mode == "adaptive" and scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        elif target_ema_mode == "fixed" and scale_mode == "progdist":
            distill_stage = step // distill_steps_per_iter
            scales = start_scales // (2**distill_stage)
            scales = np.maximum(scales, 2)

            sub_stage = np.maximum(
                step - distill_steps_per_iter * (np.log2(start_scales) - 1),
                0,
            )
            sub_stage = sub_stage // (distill_steps_per_iter * 2)
            sub_scales = 2 // (2**sub_stage)
            sub_scales = np.maximum(sub_scales, 1)

            scales = np.where(scales == 2, sub_scales, scales)

            target_ema = 1.0
        else:
            raise NotImplementedError

        return float(target_ema), int(scales)

    return ema_and_scales_fn


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
