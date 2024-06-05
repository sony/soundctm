import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from tango_edm.audioldm.audio.stft import TacotronSTFT
from tango_edm.audioldm.utils import default_audioldm_config, get_metadata
from tango_edm.audioldm.variational_autoencoder import AutoencoderKL
from tango_edm.edm.edm_precond import EDMPrecond, VEPrecond, VPPrecond, iDDPMPrecond
from tango_edm.unet_2d_condition import UNet2DConditionModel as CTMUNet2DConditionModel
from tango_edm.unet_2d_condition_teacher import UNet2DConditionModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
)


def build_pretrained_models(name, stage1_ckpt=None):
    if stage1_ckpt == None:
        checkpoint = torch.load("ckpt/audioldm-s-full.ckpt", map_location="cpu")
    else:
        checkpoint = torch.load(stage1_ckpt, map_location="cpu")
    scale_factor = checkpoint["state_dict"]["scale_factor"].item()

    vae_state_dict = {k[18:]: v for k, v in checkpoint["state_dict"].items() if "first_stage_model." in k}

    config = default_audioldm_config(name)
    vae_config = config["model"]["params"]["first_stage_config"]["params"]
    vae_config["scale_factor"] = scale_factor

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(vae_state_dict)

    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    vae.eval()
    fn_STFT.eval()
    return vae, fn_STFT

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]




class AudioDiffusionEDM(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        unet_model_name=None,
        unet_model_config_path=None,
        sigma_data: float = 0.5,
        freeze_text_encoder: bool = True,
        precond_type: str = 'edm',
        use_fp16: bool = False,
        force_fp32: bool = False,
        teacher: bool = False,
        ctm_unet_model_config_path=None,

    ):
        super().__init__()

        assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

        self.text_encoder_name = text_encoder_name
        # self.scheduler_name = scheduler_name
        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.ctm_unet_model_config_path = ctm_unet_model_config_path
        self.sigma_data = sigma_data
        self.freeze_text_encoder = freeze_text_encoder
        # self.uncondition = uncondition
        self.precond_type = precond_type
        self.teacher = teacher 
        
        if precond_type == "edm":
            self.edm_cond = EDMPrecond(sigma_data=sigma_data)
        elif precond_type == "vp":
            self.edm_cond = VPPrecond()
        elif precond_type == "ve":
            self.edm_cond = VEPrecond()
        elif precond_type == "iddpm":
            self.edm_cond = iDDPMPrecond()
            
        

        if teacher: 
            if unet_model_config_path:
                unet_config = UNet2DConditionModel.load_config(unet_model_config_path)
                self.unet = UNet2DConditionModel.from_config(unet_config, subfolder="unet")
                self.set_from = "random"
        if not teacher: 
            ctm_unet_config = CTMUNet2DConditionModel.load_config(ctm_unet_model_config_path) # TODO: sprcify same configs as TANGO
            self.ctm_unet_config = ctm_unet_config
            self.ctm_unet = CTMUNet2DConditionModel.from_config(ctm_unet_config, subfolder="unet")

            
        if "stable-diffusion" in self.text_encoder_name:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.text_encoder_name, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_name, subfolder="text_encoder")
        elif "t5" in self.text_encoder_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name)
            
        self.dtype = torch.float16 if (use_fp16 and not force_fp32) else torch.float32
        
    def encode_text(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        if self.freeze_text_encoder:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
        else:
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        boolean_encoder_mask = (attention_mask == 1).to(device)
        return encoder_hidden_states, boolean_encoder_mask

    def compute_loss(self, latents, prompt, validation_mode=False):
        encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)


        noise = torch.randn_like(latents)

        if self.set_from == "random":
            loss_weight, sigma = self.edm_cond.set_noise(latents, validation_mode)
            
            c_skip, c_out, c_in, c_noise = self.edm_cond.network_precond(sigma)
            
            noisy_latents = latents + sigma * noise
            model_pred = self.unet(
                (c_in * noisy_latents).to(self.dtype), c_noise.squeeze().to(self.dtype), encoder_hidden_states,
                encoder_attention_mask=boolean_encoder_mask
            ).sample 
            D_x = c_skip * noisy_latents + c_out * model_pred.to(torch.float32) # D_x in eq(7)
            loss = F.mse_loss(D_x, latents, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * loss_weight.squeeze()
            loss = loss.mean()

        return loss
    def get_c_in(self, sigma):
        return 1 / (sigma**2 + self.sigma_data**2) ** 0.5

    def unrescaling_t(self, rescaled_t):
        return torch.exp(rescaled_t / 250.) - 1e-44

    def extract_feature_space(self, latents, timesteps=None, prompt=None, unet_mode = 'half'):
        
        encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)
        noisy_latents = latents
        sigma = timesteps
            # c_in = append_dims(self.get_c_in(sigma), latents.ndim)
        c_noise = sigma.log() / 4
        if unet_mode == 'half':
            fmaps = self.unet.get_feature(
                noisy_latents, c_noise.squeeze(), encoder_hidden_states,
                encoder_attention_mask=boolean_encoder_mask
            )
        elif unet_mode == 'full':
            fmaps = self.unet.get_feature_full(
                noisy_latents, c_noise.squeeze(), encoder_hidden_states,
                encoder_attention_mask=boolean_encoder_mask
            )
        return fmaps
    
    def guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=w.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=w.device) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb
        
        
    def forward(self, latents, timesteps=None, prompt=None, s_tinmesteps=None, teacher=False, cfg=None, **kwargs):

        encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)
        noisy_latents = latents 

        if teacher:
            sigma = timesteps
            # c_in = append_dims(self.get_c_in(sigma), latents.ndim)
            c_noise = sigma.log() / 4
            F_x = self.unet(
                noisy_latents, c_noise.squeeze(), encoder_hidden_states,
                encoder_attention_mask=boolean_encoder_mask
            ).sample 

        else:
            if cfg is not None:
                # print("cfg_aug", cfg)
                w_embedding = self.guidance_scale_embedding(cfg, embedding_dim=self.ctm_unet_config['block_out_channels'][0]*4)
                w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)
            t = timesteps
            t = t.log() / 4
            if s_tinmesteps != None:
                s = s_tinmesteps
                s = s.log() / 4
            
            F_x = self.ctm_unet(
                noisy_latents, 
                t.flatten(),
                encoder_hidden_states=encoder_hidden_states,
                s_timestep=None if s == None else s.flatten(),
                embedd_cfg=None if cfg == None else w_embedding,
                encoder_attention_mask=boolean_encoder_mask
            ).sample
        return F_x


    @torch.no_grad()
    def inference(
        self, 
        prompt, 
        num_steps: int = 35,
        guidance_scale: float = 3., 
        num_samples_per_prompt: int = 1,
        stocastic: bool = False,
        S_churn: float = 0.,
        S_min: float = 0.,
        S_max: float = float('inf'), 
        S_noise: float = 1.,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7.,
    ):
        """
        inference steps with 2nd order Heun solver following with EDM
        """
        device = self.text_encoder.device
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(prompt) * num_samples_per_prompt

        if classifier_free_guidance:
            prompt_embeds, boolean_prompt_mask = self.encode_text_classifier_free(prompt, num_samples_per_prompt)
        else:
            prompt_embeds, boolean_prompt_mask = self.encode_text(prompt)
            prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
            boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.edm_cond.sigma_min)
        sigma_max = min(sigma_max, self.edm_cond.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.edm_cond.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0 
        
        # Prepare inital latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(batch_size, t_steps, num_channels_latents, t_steps.dtype, device)
        # Main sampling loop.
        x_next = latents
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            if stocastic: # Algo. 2
                # Increase noise temporarily.
                gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
                t_hat = self.edm_cond.round_sigma(t_cur + gamma * t_cur)
                x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
            else: # Algo. 1
                x_hat = x_cur
                t_hat = t_cur
            latent_model_input = torch.cat([x_hat] * 2) if classifier_free_guidance else x_hat
            
            # Euler step.
            c_skip, c_out, c_in, c_noise = self.edm_cond.network_precond(t_hat)
            model_pred = self.unet(
                c_in * latent_model_input.to(torch.float32), c_noise.squeeze(), prompt_embeds, 
                encoder_attention_mask=boolean_prompt_mask).sample # F_x in eq(7)
            denoised = (c_skip * latent_model_input.to(torch.float32) + c_out * model_pred).to(torch.float64) # D_x in eq(7)
            # perform guidance
            if classifier_free_guidance:
                denoised_uncond, denoised_text = denoised.chunk(2)
                denoised = denoised_uncond + guidance_scale * (denoised_text - denoised_uncond)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            
            # Apply 2nd order correction.
            if i < num_steps - 1:
                latent_model_input = torch.cat([x_next] * 2) if classifier_free_guidance else x_next
                
                c_skip, c_out, c_in, c_noise = self.edm_cond.network_precond(t_next)
                model_pred = self.unet(
                    c_in * latent_model_input.to(torch.float32), c_noise.squeeze(), prompt_embeds,
                    encoder_attention_mask=boolean_prompt_mask).sample # F_x in eq(7)
                denoised = (c_skip * latent_model_input.to(torch.float32) + c_out * model_pred).to(torch.float64)
                # perform guidance
                if classifier_free_guidance:
                    denoised_uncond, denoised_text = denoised.chunk(2)
                    denoised = denoised_uncond + guidance_scale * (denoised_text - denoised_uncond)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.to(torch.float32)
    
    def prepare_latents(self, batch_size, t_steps, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents, 256, 16) # TODO; last two dims are hardcoded.
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # latents = latents.to(torch.float64) * t_steps[0]
        latents = latents * t_steps[0]
        return latents

    def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]
                
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = [""] * len(prompt)

        max_length = prompt_embeds.shape[1]
        uncond_batch = self.tokenizer(
            uncond_tokens, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt",
        )
        uncond_input_ids = uncond_batch.input_ids.to(device)
        uncond_attention_mask = uncond_batch.attention_mask.to(device)

        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
            )[0]
                
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)

        return prompt_embeds, boolean_prompt_mask

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next