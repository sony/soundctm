"""
Based on: https://github.com/crowsonkb/k-diffusion
"""
import numpy as np
import torch as th
import torch.nn as nn
from .nn import mean_flat, append_dims

import tools.torch_tools as torch_tools

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = th.sqrt(th.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat/(norm_factor+eps)

def get_weightings(weight_schedule, snrs, sigma_data, t, s):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "sq-snr":
        weightings = snrs**0.5
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    elif weight_schedule == "uniform_g":
        return 1./(1. - s / t)
    elif weight_schedule == "karras_weight":
        sigma = snrs ** -0.5
        weightings = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    elif weight_schedule == "sq-t-inverse":
        weightings = 1. / snrs ** 0.25
    else:
        raise NotImplementedError()
    return weightings

class KarrasDenoiser:
    def __init__(
        self,
        args,
        schedule_sampler,
        diffusion_schedule_sampler,
        # feature_networks=None,
    ):
        self.args = args
        self.schedule_sampler = schedule_sampler
        self.diffusion_schedule_sampler = diffusion_schedule_sampler
        # self.omega_sampler = OmegaSampler(args.omega_min, args.omega_max)
        # self.feature_networks = feature_networks
        self.num_timesteps = args.start_scales
        self.dist = nn.MSELoss(reduction='none')

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas

    def get_c_in(self, sigma):
        return 1 / (sigma**2 + self.args.sigma_data**2) ** 0.5

    def get_scalings(self, sigma):
        c_skip = self.args.sigma_data**2 / (sigma**2 + self.args.sigma_data**2)
        c_out = sigma * self.args.sigma_data / (sigma**2 + self.args.sigma_data**2) ** 0.5
        return c_skip, c_out

    def get_scalings_t(self, t, s): # TODO: check what this is later
        c_skip = th.zeros_like(t)
        c_out = ((t ** 2 + self.args.sigma_data ** 2) / (s ** 2 + self.args.sigma_data ** 2)) ** 0.5
        return c_skip, c_out

    def get_scalings_for_generalized_boundary_condition(self, t, s):
        if self.args.parametrization.lower() == 'euler':
            c_skip = s / t
        elif self.args.parametrization.lower() == 'variance':
            c_skip = (((s - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2) / ((t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2)).sqrt()
        elif self.args.parametrization.lower() == 'euler_variance_mixed':
            c_skip = s / (t + 1.) + \
                     (((s - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2) /
                      ((t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2)).sqrt() / (t + 1.)
        c_out = (1. - s / t)
        return c_skip, c_out

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.args.sigma_data**2 / (
            (sigma - self.args.sigma_min) ** 2 + self.args.sigma_data**2
        )
        c_out = (
            (sigma - self.args.sigma_min)
            * self.args.sigma_data
            / (sigma**2 + self.args.sigma_data**2) ** 0.5 
        )
        return c_skip, c_out

    def calculate_adaptive_weight(self, loss1, loss2, last_layer=None, allow_unused=False):
        loss1_grad = th.autograd.grad(loss1, last_layer, retain_graph=True, allow_unused=allow_unused)[0]
        loss2_grad = th.autograd.grad(loss2, last_layer, retain_graph=True)[0]
        d_weight = th.norm(loss1_grad) / (th.norm(loss2_grad) + 1e-8)
        #print("consistency gradient: ", th.norm(loss1_grad))
        #print("denoising gradient: ", th.norm(loss2_grad))
        #print("weight: ", d_weight)
        d_weight = th.clamp(d_weight, 0.0, 1e3).detach()
        return d_weight

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight

    def rescaling_t(self, t):
        rescaled_t = 1000 * 0.25 * th.log(t + 1e-44)
        return rescaled_t

    def get_t(self, ind): 
        if self.args.time_continuous:
            t = self.args.sigma_max ** (1 / self.args.rho) + ind * (
                    self.args.sigma_min ** (1 / self.args.rho) - self.args.sigma_max ** (1 / self.args.rho)
            )
            t = t ** self.args.rho
        else: # Same as EDM's eq.(5)
            t = self.args.sigma_max ** (1 / self.args.rho) + ind / (self.args.start_scales - 1) * (
                    self.args.sigma_min ** (1 / self.args.rho) - self.args.sigma_max ** (1 / self.args.rho)
            )
            t = t ** self.args.rho
        return t

    def get_num_heun_step(self, step):
        if self.args.num_heun_step_random:
            #if step % self.args.g_learning_period == 0:
            if self.args.time_continuous:
                num_heun_step = np.random.rand() * self.args.num_heun_step / self.args.start_scales
            else:
                if self.args.heun_step_strategy == 'uniform':
                    num_heun_step = np.random.randint(1, 1+self.args.num_heun_step)
                elif self.args.heun_step_strategy == 'weighted':
                    p = np.array([i ** self.args.heun_step_multiplier for i in range(1, 1+self.args.num_heun_step)])
                    p = p / sum(p)
                    num_heun_step = np.random.choice([i+1 for i in range(len(p))], size=1, p=p)[0]
            # else:
            #    if self.args.time_continuous:
            #        num_heun_step = np.random.rand() / self.args.d_learning_period +\
            #                        (self.args.d_learning_period - 1) / self.args.d_learning_period
            #    else:
                    #num_heun_step = np.random.randint((self.args.d_learning_period - 1) * self.args.start_scales //
                    #                                  self.args.d_learning_period, 1+self.args.num_heun_step)
            #        num_heun_step = self.args.num_heun_step
        else:
            if self.args.time_continuous:
                num_heun_step = self.args.num_heun_step / self.args.start_scales
            else:
                num_heun_step = self.args.num_heun_step
        return num_heun_step

    @th.no_grad()
    def heun_solver(self, x, ind, teacher_model, dims, cond, num_step=1):
        for k in range(num_step):
            t = self.get_t(ind + k) 
            denoiser = self.denoise_fn(teacher_model, x, t, cond=cond, s=None, ctm=False, teacher=True) # D_{\theta}
            d = (x - denoiser) / append_dims(t, dims)
            
            t2 = self.get_t(ind + k + 1) 
            x_phi_ODE_1st = x + d * append_dims(t2 - t, dims) 
            denoiser_2 = self.denoise_fn(teacher_model, x_phi_ODE_1st, t2, cond=cond, s=None, ctm=False, teacher=True)            
            next_d = (x_phi_ODE_1st - denoiser_2) / append_dims(t2, dims)
            x_phi_ODE_2nd = x + (d + next_d) * append_dims((t2 - t) / 2, dims)
            x = x_phi_ODE_2nd
        return x

    @th.no_grad()
    def heun_solver_cfg(self, x, ind, guidance_scale, teacher_model, dims, cond, num_step=1):
        for k in range(num_step):
            t = self.get_t(ind + k)
            t_in = th.cat([t] * 2)
            model_input = th.cat([x] * 2) 
            cond_cfg = cond + ([""] * len(cond))
            denoiser = self.denoise_fn(teacher_model, model_input, t_in, cond=cond_cfg, s=None, ctm=False, teacher=True) # D_{\theta}
            denoised_text, denoised_uncond = denoiser.chunk(2)
            denoised = denoised_uncond + append_dims(guidance_scale, dims) * (denoised_text - denoised_uncond)
            d = (x - denoised) / append_dims(t, dims)
            
            t2 = self.get_t(ind + k + 1) 
            t2_in = th.cat([t2] * 2)
            x_phi_ODE_1st = x + d * append_dims(t2 - t, dims) 
            model_input_2 = th.cat([x_phi_ODE_1st] * 2)

            denoiser_2 = self.denoise_fn(teacher_model, model_input_2, t2_in, cond=cond_cfg, s=None, ctm=False, teacher=True)
            denoised_text, denoised_uncond = denoiser_2.chunk(2)
            denoised_2 = denoised_uncond + append_dims(guidance_scale, dims) * (denoised_text - denoised_uncond)
            
                        
            next_d = (x_phi_ODE_1st - denoised_2) / append_dims(t2, dims)
            x_phi_ODE_2nd = x + (d + next_d) * append_dims((t2 - t) / 2, dims)
            x = x_phi_ODE_2nd
        return x

    # def get_gan_estimate(self, estimate, step, x_t, t, t_dt, s, model, target_model, ctm, cond):
    #     if self.args.gan_estimate_type == 'consistency':
    #         # NOTE: If we use different timestep for gan, then use here.
    #         estimate = self.denoise_fn(model, x_t, t, cond=cond, s=th.ones_like(s) * self.args.sigma_min, ctm=ctm)
    #     elif self.args.gan_estimate_type == 'enable_grad':
    #         if self.args.auxiliary_type == 'enable_grad':
    #             estimate = estimate
    #         else:
    #             estimate = self.get_estimate(step, x_t, t, t_dt, s, model, target_model, ctm=ctm,
    #                                          auxiliary_type='enable_grad')
    #     elif self.args.gan_estimate_type == 'only_high_freq':
    #         estimate = self.get_estimate(step, x_t, t, t_dt, s, model, target_model, ctm=ctm,
    #                                      type='stop_grad', auxiliary_type='enable_grad')
    #     elif self.args.gan_estimate_type == 'same':
    #         estimate = estimate
    #     return estimate

    def get_estimate(self, step, x_t, t, t_dt, s, model, target_model, ctm, cfg=None, cond=None, type=None, auxiliary_type=None):
        distiller = self.denoise_fn(model, x_t, t, cond=cond, s=s, ctm=ctm, cfg=cfg)
        if self.args.match_point == 'zs':
            return distiller
        else:
            distiller = self.denoise_fn(target_model, distiller, s, cond=cond, s=th.ones_like(s) * self.args.sigma_min, ctm=ctm, cfg=cfg)
            return distiller

    @th.no_grad()
    def get_target(self, step, x_t_dt, t_dt, s, model, target_model, ctm, cond, cfg=None):
        with th.no_grad():
            distiller_target = self.denoise_fn(target_model, x_t_dt, t_dt, cond=cond, s=s, ctm=ctm, cfg=cfg)
            if self.args.match_point == 'zs':
                return distiller_target.detach()
            else:
                distiller_target = self.denoise_fn(target_model, distiller_target, s, cond=cond, s=th.ones_like(s) * self.args.sigma_min, ctm=ctm, cfg=cfg)
                return distiller_target.detach()

    def denoise_fn(self, model, x, t, cond, s, ctm=False, cfg=None, teacher=False):
        return self.denoise(model, x, t, cond=cond, s=s, ctm=ctm, teacher=teacher, cfg=cfg)[1]

    def denoise(self, model, x_t, t, cond=None, s=None, ctm=False, teacher=False, cfg=None):
        c_in = append_dims(self.get_c_in(t), x_t.ndim)
        model_output = model(c_in * x_t, t, prompt=cond, s_tinmesteps=s, teacher=teacher, cfg=cfg)
        if ctm:
            if self.args.inner_parametrization == 'edm':
                c_skip, c_out = [
                    append_dims(x, x_t.ndim)
                    for x in self.get_scalings(t)
                ]

                model_output = c_out * model_output + c_skip * x_t # g_{\theta}, Same as EDM's eq.(7)
            elif self.args.inner_parametrization == 'scale':
                c_skip, c_out = [
                    append_dims(x, x_t.ndim)
                    for x in self.get_scalings_t(t, s)
                ]
                #print("c_skip, c_out: ", c_skip.reshape(-1), c_out.reshape(-1))
                model_output = c_out * model_output + c_skip * x_t
            elif self.args.inner_parametrization == 'no':
                model_output = model_output
            if teacher:
                if self.args.parametrization.lower() == 'euler': # NOTE: Normally, do here.
                    denoised = model_output
                elif self.args.parametrization.lower() == 'variance':
                    denoised = model_output + append_dims((self.args.sigma_min ** 2 + self.args.sigma_data ** 2
                                                        - self.args.sigma_min * t) / \
                            ((t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2), x_t.ndim) * x_t
                elif self.args.parametrization.lower() == 'euler_variance_mixed':
                    denoised = model_output + x_t - append_dims(t / (t + 1.) * (1. + (t - self.args.sigma_min) /
                                                                        ((t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2)), x_t.ndim) * x_t
                return model_output, denoised
            else:
                assert s != None
                c_skip, c_out = [
                    append_dims(x, x_t.ndim)
                    for x in self.get_scalings_for_generalized_boundary_condition(t, s, )
                ]
                denoised = c_out * model_output + c_skip * x_t # G_{\theta} Last eq of Lemma 1 on CTM paper.
        else:
            if teacher:
                c_skip, c_out = [
                    append_dims(x, x_t.ndim) for x in self.get_scalings(t)
                ]
            else:
                c_skip, c_out = [
                    append_dims(x, x_t.ndim)
                    for x in self.get_scalings_for_boundary_condition(t) # CM's boundary condition when smallestr time instant. See Appendix.C of CM paper.
                ]
            denoised = c_out * model_output + c_skip * x_t
        
        return model_output, denoised

    def get_consistency_loss(self, estimate, target, weights, loss_type='feature_space', teacher_model=None, s=None, prompt=None):
        
        estimate_out = estimate
        target_out = target

        if loss_type == 'l2':
            consistency_loss = weights * mean_flat((estimate_out - target_out) ** 2)

                
        elif loss_type == 'l1': 
            consistency_loss = weights * mean_flat(th.abs(estimate_out - target_out)) 

        elif loss_type == 'ictm': # Psuedo-Huber loss 
            c = 0.00054 * th.sqrt(th.tensor(self.args.latent_channels*self.args.latent_f_size*self.args.latent_t_size))
            consistency_loss = weights * mean_flat(th.sqrt((estimate_out - target_out) ** 2 + c ** 2) - c) 

        elif loss_type == 'feature_space':
            if self.args.match_point == 'z0':
                c_in = append_dims(self.get_c_in(th.ones_like(s) * self.args.sigma_min), estimate.ndim)
                estimate_out = teacher_model.extract_feature_space(estimate * c_in, timesteps=th.ones_like(s) * self.args.sigma_min, prompt=prompt, unet_mode = self.args.unet_mode)
                target_out = teacher_model.extract_feature_space(target * c_in, timesteps=th.ones_like(s) * self.args.sigma_min, prompt=prompt, unet_mode = self.args.unet_mode)
                
            elif self.args.match_point == 'zs':
                # print("s", s)
                c_in = append_dims(self.get_c_in(s), estimate.ndim)
                estimate_out = teacher_model.extract_feature_space(estimate * c_in, timesteps=s, prompt=prompt, unet_mode = self.args.unet_mode)
                target_out = teacher_model.extract_feature_space(target * c_in, timesteps=s, prompt=prompt, unet_mode = self.args.unet_mode)
            
            consistency_loss = 0.
            # k = 0
            for est_feature, tgt_feature in zip(estimate_out, target_out):
                if th.isnan(mean_flat((normalize_tensor(est_feature) - normalize_tensor(tgt_feature)) ** 2).mean()): 
                    consistency_loss += self.null(mean_flat((normalize_tensor(est_feature) - normalize_tensor(tgt_feature)) ** 2))
                else:
                    consistency_loss += mean_flat((normalize_tensor(est_feature) - normalize_tensor(tgt_feature)) ** 2)
                    # consistency_loss += mean_flat((est_feature - tgt_feature) ** 2)
                # k += 1
                # print("consistency_loss_{}".format(k), consistency_loss.mean())
            consistency_loss = weights * consistency_loss

        if th.isnan(consistency_loss.mean()):
            consistency_loss = self.null(consistency_loss)
        return consistency_loss

    def get_denoising_loss(self, model, x_start, consistency_loss, step, cond, loss_target):
        dsm_null_flg = False
        sigmas, denoising_weights = self.diffusion_schedule_sampler.sample(x_start.shape[0], x_start.device)
        #print("diffusion sigmas: ", sigmas)
        noise = th.randn_like(x_start)
        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        if self.args.unform_sampled_cfg_distill:
            model_estimate = self.denoise(model, x_t, sigmas, cond=cond, s=sigmas, ctm=True, teacher=False, cfg=self.sampled_cfg)[0] # g_{\theta}(z_t, cond, t, t, omega)
        else:
            model_estimate = self.denoise(model, x_t, sigmas, cond=cond, s=sigmas, ctm=True, teacher=False)[0] # g_{\theta}(z_t, cond, t, t)
        snrs = self.get_snr(sigmas)
        denoising_weights = append_dims(get_weightings(self.args.diffusion_weight_schedule, snrs, self.args.sigma_data, None, None), dims)
        # denoising_loss = mean_flat(denoising_weights * (model_estimate - x_start) ** 2)
        denoising_loss = mean_flat(denoising_weights * (model_estimate - loss_target) ** 2)
        if th.isnan(denoising_loss.mean()): 
            denoising_loss = self.null(denoising_loss)
            dsm_null_flg = True
        if not dsm_null_flg:
            if self.args.apply_adaptive_weight:
                try:
                    balance_weight = self.calculate_adaptive_weight(consistency_loss.mean(), denoising_loss.mean(),
                                                            last_layer=model.ctm_unet.conv_out.weight)
                except:
                    balance_weight = self.calculate_adaptive_weight(consistency_loss.mean(), denoising_loss.mean(),
                                                            last_layer=model.module.ctm_unet.conv_out.weight)
        else:
            balance_weight = 1.

        balance_weight = self.adopt_weight(balance_weight, step, threshold=0, value=1.)
        denoising_loss = denoising_loss * balance_weight
        return denoising_loss


    def check_isnan(self, loss):
        if th.isnan(loss.mean()):
            loss = th.zeros_like(loss)
            loss.requires_grad_(True)
        return loss
    
    def null(self, x_start):
        loss = th.zeros_like(x_start, device=x_start.device)
        loss.requires_grad_(True)
        return loss
    
    
    def get_samples(
        self,
        step,
        model,
        wavs,
        cond=None,
        model_kwargs=None,
        target_model=None,
        teacher_model=None,
        stage1_model=None,
        stft=None,
        accelerator=None,
        noise=None,
        # init_step=0,
        ctm=True,
    ):
        
        # Prepare latent representation of mel through stage1 model
        target_length = int(self.args.duration * 102.4) 
        with th.no_grad():
            mel, _, waveform = torch_tools.wav_to_fbank(wavs, target_length, stft)
            mel = mel.unsqueeze(1).to(accelerator.device)
            waveform = waveform.to(accelerator.device)
            prompt = list(cond)
            if self.args.tango_data_augment and len(cond) > 1:
                mixed_mel, _, mixed_waveform, mixed_captions = torch_tools.augment_wav_to_fbank(wavs, cond, self.args.augment_num, target_length, stft)
                mixed_mel = mixed_mel.unsqueeze(1).to(accelerator.device)
                mixed_waveform = mixed_waveform.to(accelerator.device)
                mel = th.cat([mel, mixed_mel], 0)
                waveform = th.cat([waveform, mixed_waveform], 0)
                prompt += mixed_captions
            x_start = stage1_model.get_first_stage_encoding(stage1_model.encode_first_stage(mel))
        
        # th.cuda.empty_cache()
        
        if noise is None:
            noise = th.randn_like(x_start)
        dims = x_start.ndim
        s = None
        terms = {}
        assert self.args.consistency_weight > 0.
        num_heun_step = [self.get_num_heun_step(step)] 
        num_heun_step = num_heun_step[0]

        indices, _ = self.schedule_sampler.sample_t(x_start.shape[0], x_start.device, num_heun_step, self.args.time_continuous)
        t = self.get_t(indices)
        t_dt = self.get_t(indices + num_heun_step)
        if ctm:
            new_indices = self.schedule_sampler.sample_s(self.args, x_start.shape[0], x_start.device, indices,
                                                        num_heun_step, self.args.time_continuous,
                                                        N=self.args.start_scales)
            s = self.get_t(new_indices)
        x_t = x_start + noise * append_dims(t, dims) # z_t
        if self.args.unform_sampled_cfg_distill:
            self.sampled_cfg = (self.args.w_max - self.args.w_min) * th.rand((noise.shape[0],), device=accelerator.device) + self.args.w_min
            estimate = self.get_estimate(step, x_t, t, t_dt, s, model, target_model, ctm=ctm, cond=prompt, cfg=self.sampled_cfg)
        else:
            estimate = self.get_estimate(step, x_t, t, t_dt, s, model, target_model, ctm=ctm, cond=prompt)

        if teacher_model:
            if self.args.cfg_single_distill:
                x_t_dt = self.heun_solver_cfg(x_t, indices, self.args.single_target_cfg, teacher_model, dims, cond=prompt, num_step=num_heun_step)
                # Solver(z_t, cond, t, u, \omega; \phi)
            
            elif self.args.unform_sampled_cfg_distill:
                x_t_dt = self.heun_solver_cfg(x_t, indices, self.sampled_cfg, teacher_model, dims, cond=prompt, num_step=num_heun_step)
            
            else:
                x_t_dt = self.heun_solver(x_t, indices, teacher_model, dims, cond=prompt, num_step=num_heun_step)
                # Solver(z_t, cond, t, u; \phi)
        
        else:
            with th.no_grad():
                x_t_dt = self.denoise_fn(target_model, x_t, t, cond=prompt, s=t_dt, ctm=ctm) 
        if self.args.unform_sampled_cfg_distill:
            target = self.get_target(step, x_t_dt, t_dt, s, model, target_model, ctm=ctm, cond=prompt, cfg=self.sampled_cfg) 
        
        else:
            target = self.get_target(step, x_t_dt, t_dt, s, model, target_model, ctm=ctm, cond=prompt) 

        return estimate, target, x_start, mel, waveform, prompt, t, s
    
    def get_gen_loss(
        self,
        step,
        model,
        estimate,
        target,
        x_start,
        mel,
        waveform,
        prompt,
        t,
        s,
        teacher_model,
        stage1_model,
        accelerator,
        # discriminator,
        model_kwargs,
    ):
        terms = {}
        snrs = self.get_snr(t)
        weights = get_weightings(self.args.weight_schedule, snrs, self.args.sigma_data, t, s)
        terms["consistency_loss"] = self.get_consistency_loss(estimate, target, weights, 
                                                                loss_type=self.args.loss_type, 
                                                                teacher_model=teacher_model,
                                                                s=s, prompt=prompt)
        # th.cuda.empty_cache()
        if self.args.diffusion_training:
            if self.args.dsm_loss_target == 'z_0':
                terms['denoising_loss'] = self.get_denoising_loss(model, x_start,
                                                                terms["consistency_loss"],
                                                                step, cond=prompt,
                                                                loss_target=x_start)
            elif self.args.dsm_loss_target == 'z_target':
                terms['denoising_loss'] = self.get_denoising_loss(model, x_start,
                                                                terms["consistency_loss"],
                                                                step, cond=prompt,
                                                                loss_target=target)
        # th.cuda.empty_cache()
        return terms