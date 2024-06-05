import os
import blobfile as bf
import torch as th
import wandb

@th.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

class TrainLoop:
    def __init__(
        self,
        #*,
        model,
        # discriminator,
        diffusion,
        data,
        accelerator,
        opt, 
        # d_opt,
        resume_epoch=0,
        resume_step=0,
        resume_global_step=0,
        # eval_dataloader=None,
        # lr_scheduler,
        # d_lr_scheduler,
        args=None,
    ):
        self.args = args
        self.accelerator = accelerator
        self.model = model
        # self.discriminator = discriminator
        self.diffusion = diffusion # KarrasDenoiser
        
        self.train_dataloader = data
        # self.eval_dataloader = eval_dataloader
        self.batch_size = args.per_device_train_batch_size
        self.lr = args.lr
        # self.lr_scheduler = lr_scheduler
        # self.d_lr_scheduler = d_lr_scheduler
        # self.ema_rate = (
        #     [args.ema_rate]
        #     if isinstance(args.ema_rate, float)
        #     else [float(x) for x in args.ema_rate.split(",")]
        # )
        
        self.step = 0
        self.global_step = 0
        self.first_epoch = 0
        self.resume_epoch = resume_epoch
        self.resume_step = resume_step
        self.resume_global_step = resume_global_step
        self.global_batch = self.batch_size * self.accelerator.num_processes

        self.x_T = th.randn(*(self.batch_size, 
                              self.args.latent_channels, 
                              self.args.latent_t_size, 
                              self.args.latent_f_size), 
                            device=self.accelerator.device) * self.args.sigma_max

        self.opt = opt
        # self.d_opt = d_opt
        # self.ema_params = ema_model.get_param_sets()
        self.first_epoch = self.resume_epoch
        self.step = self.resume_step
        self.global_step = self.resume_global_step

    def run_loop(self):
        while not self.args.lr_anneal_steps or self.step < self.args.lr_anneal_steps:
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            # if self.step % self.args.log_interval == 0:
            #     logger.dumpkvs()
            if self.step % self.args.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.args.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self.step += 1
            self._update_ema()
        self._anneal_lr()
        # self.log_step()

    def forward_backward(self, batch, cond):
        raise NotImplementedError

    def _anneal_lr(self):
        if not self.args.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.args.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

class CTMTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        target_model,
        teacher_model,
        latent_decoder,
        stft,
        ema_scale_fn,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.training_mode = self.args.training_mode
        self.ema_scale_fn = ema_scale_fn
        self.target_model = target_model
        self.teacher_model = teacher_model
        self.latent_decoder = latent_decoder
        self.stft = stft
        self.total_training_steps = self.args.total_training_steps
        
        if teacher_model:
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

    def run_loop(self):
        for epoch in range(self.first_epoch, self.args.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                text, audios, _ = batch
                self.run_step(audios, text)
                # th.cuda.empty_cache()
                
                if (self.global_step 
                    and self.args.save_interval != -1 
                    and self.global_step % self.args.save_interval == 0
                    ):
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.sync_gradients:
                        self.save(epoch)
                        # saved = True
                        # th.cuda.empty_cache()
                    self.accelerator.wait_for_everyone()

                if self.global_step >= self.args.total_training_steps:
                    self.save(epoch)
                    break

    def run_step(self, batch, cond):
        
        if self.accelerator.is_main_process:
            result = {}
        estimate, target, x_start, mel, waveform, prompt, t, s = self.get_samples(batch, cond)
        
        if (self.step+1) % self.args.gradient_accumulation_steps != 0:
            with self.accelerator.no_sync(self.model):
                losses = self.compute_gen_loss(estimate, target, x_start, mel, waveform, prompt, t, s)
                # th.cuda.empty_cache()
                if 'consistency_loss' in list(losses.keys()):
                    loss = self.args.consistency_weight * losses["consistency_loss"].mean()
                    # print("consistency_loss: {}".format(self.args.consistency_weight * losses["consistency_loss"].mean()))
                    
                    if 'denoising_loss' in list(losses.keys()):
                        loss = loss + self.args.denoising_weight * losses['denoising_loss'].mean()
                        # print("dsm_loss: {}".format(self.args.denoising_weight * losses['denoising_loss'].mean()))
            
                    # if 'g_loss' in list(losses.keys()):
                    #     loss = loss + self.args.discriminator_weight * losses['g_loss'].mean()
                    #     # print("gen_loss: {}".format(self.args.discriminator_weight * losses['g_loss'].mean()))
                            
                    # if 'fm_loss' in list(losses.keys()):
                    #     loss = loss + self.args.fm_weight * losses['fm_loss'].mean()
                        # print("fm_loss: {}".format(self.args.fm_weight * losses['fm_loss'].mean()))
                self.accelerator.backward(loss)

        else:
            losses = self.compute_gen_loss(estimate, target, x_start, mel, waveform, prompt, t, s)
            # th.cuda.empty_cache()
            if 'consistency_loss' in list(losses.keys()):
                loss = self.args.consistency_weight * losses["consistency_loss"].mean()
                # print("consistency_loss: {}".format(self.args.consistency_weight * losses["consistency_loss"].mean()))
                
                if 'denoising_loss' in list(losses.keys()):
                    loss = loss + self.args.denoising_weight * losses['denoising_loss'].mean()
                    # print("dsm_loss: {}".format(self.args.denoising_weight * losses['denoising_loss'].mean()))
        
                # if 'g_loss' in list(losses.keys()):
                #     loss = loss + self.args.discriminator_weight * losses['g_loss'].mean()
                #     # print("gen_loss: {}".format(self.args.discriminator_weight * losses['g_loss'].mean()))
                        
                # if 'fm_loss' in list(losses.keys()):
                #     loss = loss + self.args.fm_weight * losses['fm_loss'].mean()
                    # print("fm_loss: {}".format(self.args.fm_weight * losses['fm_loss'].mean()))
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                try:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.model_grad_clip_value)
                except:
                    self.accelerator.clip_grad_norm_(self.model.module.parameters(), self.args.model_grad_clip_value)
                    
            self.opt.step()
            self.opt.zero_grad()
            # th.cuda.empty_cache()
            # self.lr_scheduler.step() 
            
            if self.accelerator.sync_gradients:
                # self._update_ema()
                if self.target_model: 
                    self._update_target_ema()
                    # th.cuda.empty_cache()
                self.global_step += 1
                if self.accelerator.is_main_process:
                    result["step"] = self.step
                    result["global_step"] = self.global_step
                    result["ctm_loss"] = losses["consistency_loss"].mean().detach().float()
                    result["lambda_ctm_loss"] = self.args.consistency_weight * result["ctm_loss"]
                    if 'denoising_loss' in list(losses.keys()):
                        result["dsm_loss"] = losses["denoising_loss"].mean().detach().float()
                        result["lambda_dsm_loss"] = self.args.denoising_weight * result["dsm_loss"]
                    else:
                        result["dsm_loss"] = 0.0
                        result["lambda_dsm_loss"] = 0.0
                    wandb.log(result)
                    self.accelerator.log(result, step=self.global_step)
                self._anneal_lr()
        self.step += 1


    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            try:
                update_ema(
                    list(self.target_model.ctm_unet.parameters()),
                    list(self.model.ctm_unet.parameters()),
                    rate=target_ema,
                )
            except:
                update_ema(
                    list(self.target_model.ctm_unet.parameters()),
                    list(self.model.module.ctm_unet.parameters()),
                    rate=target_ema,
                )

    
    def get_samples(self, batch, cond):
        estimate, target, x_start, mel, waveform, prompt, t, s = self.diffusion.get_samples(
            step = self.global_step,
            model = self.model,
            wavs = batch,
            cond = cond,
            model_kwargs = None,
            target_model = self.target_model,
            teacher_model = self.teacher_model,
            stage1_model = self.latent_decoder,
            stft=self.stft,
            accelerator = self.accelerator,
            noise=None,
            ctm = True if self.training_mode.lower() == 'ctm' else False,
        )

        return estimate, target, x_start, mel, waveform, prompt, t, s
    
    def compute_gen_loss(self, estimate, target, x_start, mel, waveform, prompt, t, s):
        losses = self.diffusion.get_gen_loss(
            step = self.global_step,
            model = self.model, # self.ddp_model
            estimate = estimate,
            target = target,
            x_start = x_start,
            mel = mel,
            waveform = waveform,
            prompt = prompt,
            t = t,
            s = s,
            teacher_model = self.teacher_model,
            stage1_model = self.latent_decoder,
            accelerator = self.accelerator,
            # discriminator = self.discriminator,
            model_kwargs = None,
        )
        
        return  losses
    
    def save(self, epoch):
        def save_checkpoint(rate):
            try:
                state_dict = self.target_model.ctm_unet.state_dict()
                # for i, (name, _value) in enumerate(self.model.ctm_unet.named_parameters()):
                #     assert name in state_dict
                #     state_dict[name] = params[i]
            except:
                state_dict = self.target_model.module.ctm_unet.state_dict()
                # for i, (name, _value) in enumerate(self.model.module.ctm_unet.named_parameters()):
                #     assert name in state_dict
                #     state_dict[name] = params[i]
                
            self.accelerator.print(f"saving model {rate}...")
            if not rate:
                filename = f"model{self.global_step:06d}.pt"
            else:
                filename = f"ema_{rate}_{self.global_step:06d}.pt"
            ema_output_dir = os.path.join(self.args.output_dir, f"{self.global_step:06d}", filename)
            os.makedirs(os.path.join(self.args.output_dir, f"{self.global_step:06d}"), exist_ok=True)
            self.accelerator.save(state_dict, ema_output_dir)

        if self.accelerator.is_main_process:
            save_checkpoint(float(self.args.ema_rate))
            self.accelerator.print("saving state...")
            progress_output_dir = os.path.join(self.args.output_dir, f"{self.global_step:06d}", f"progress_state.pth")
            progress_state_dict = {
            'completed_epochs': int(epoch),
            'completed_steps': int(self.step),
            'completed_global_steps': int(self.global_step)
            }
            self.accelerator.save(progress_state_dict, progress_output_dir)
            self.accelerator.save_state("{}/{}".format(self.args.output_dir, f"{self.global_step:06d}")) # define output dir
    



def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


# def get_blob_logdir():
#     # You can change this to be a separate path to save checkpoints to
#     # a blobstore or some external drive.
#     return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(losses, logger):
    for key, values in losses.items():
        logger.info(f"{key} mean", values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        logger.info(f"{key} std", values.std().item())
        #for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #    quartile = int(4 * sub_t / diffusion.num_timesteps)
        #    logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
