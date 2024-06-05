import numpy as np
import torch as th
from .nn import append_dims, append_zero

def karras_sample(
    diffusion,
    model,
    shape,
    steps,
    cond,
    nu,
    gamma=0.9,
    omega=None,
    progress=False,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80, 
    rho=7.0,
    sampler="determinisitc",
    ts=None,
    x_T=None,
    ctm=True,
    teacher=False,
):

    if sampler in ['determinisitc', 'cm_multistep', 'gamma_multistep']:
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    if x_T == None:
        x_T = th.randn(*shape, device=device) * sigma_max

    sample_fn = {
        "determinisitc": determinisitc_sampling,
        "cm_multistep": cm_multistep_sampling,
        "gamma_multistep": gamma_multistep_sampling,
    }[sampler]

    if sampler in ["determinisitc", "cm_multistep"]:
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=rho, steps=steps
        )
    elif sampler in ["gamma_multistep"]:
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=rho, steps=steps, gamma=gamma,
        )
    else:
        sampler_args = {}
    def denoiser(x_t, t, s=th.ones(x_T.shape[0], device=device), cond=None, ctm=False, teacher=False, cfg=None):
        denoised = diffusion.denoise_fn(model, x_t, t, cond=cond, s=s, ctm=ctm, teacher=teacher, cfg=cfg)
        return denoised

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        cond=cond,
        ctm=ctm,
        teacher=teacher,
        nu=nu,
        omega=omega,
        progress=progress,
        **sampler_args,
    )
    return x_0


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


@th.no_grad()
def determinisitc_sampling(
    denoiser,
    x,
    sigmas,
    cond,
    nu,
    omega=None,
    ctm=False,
    teacher=False,
    progress=False,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    """SoundCTM's deterministic sampling, \gamma=0-sampling)."""
    s_in = x.new_ones([x.shape[0]])
    if omega is not None:
        omega = omega * th.ones((x.shape[0],), device=x.device)
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices[:-1]:
        sigma = sigmas[i]
        if sigmas[i+1] != 0:
            x_in = th.cat([x] * 2)
            cond_cfg = cond + ([""] * len(cond))
            denoised = denoiser(x_in, th.cat([sigma * s_in] * 2), cond=cond_cfg, 
                                s=th.cat([sigmas[i + 1] * s_in] * 2), ctm=ctm, teacher=teacher, cfg=omega)
            denoised_text, denoised_uncond = denoised.chunk(2)
            denoised = denoised_uncond + nu * (denoised_text - denoised_uncond)
            x = denoised
        else:
            x_in = th.cat([x] * 2)
            cond_cfg = cond + ([""] * len(cond))
            denoised = denoiser(x_in, th.cat([sigma * s_in]*2), cond=cond_cfg, 
                                s=th.cat([sigma * s_in]*2), ctm=ctm, teacher=teacher, cfg=omega)
            denoised_text, denoised_uncond = denoised.chunk(2)
            denoised = denoised_uncond + nu * (denoised_text - denoised_uncond)
            d = to_d(x, sigma, denoised)
            dt = sigmas[i + 1] - sigma
            x = x + d * dt
    return x

@th.no_grad()
def cm_multistep_sampling(
    denoiser,
    x,
    sigmas,
    cond,
    nu,
    omega=None,
    ctm=False,
    teacher=False,
    progress=False,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    """Implements CM's multistep sampling (CTM's \gamma-sampling when \gamma=1)."""
    s_in = x.new_ones([x.shape[0]])
    if omega is not None:
        omega = omega * th.ones((x.shape[0],), device=x.device)
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)
    for i in indices[:-1]:
        sigma = sigmas[i]
        x_in = th.cat([x] * 2)
        cond_cfg = cond + ([""] * len(cond))
        denoised = denoiser(x_in, th.cat([sigma * s_in] * 2), cond=cond_cfg, 
                            s=th.cat([t_min * s_in] * 2), ctm=ctm, teacher=teacher, cfg=omega)
        denoised_text, denoised_uncond = denoised.chunk(2)
        denoised = denoised_uncond + nu * (denoised_text - denoised_uncond)
        
        if i < len(indices) - 2:
            x = denoised + th.sqrt(sigmas[i+1] ** 2 - t_min ** 2) * th.randn_like(denoised)
        else:
            x = denoised
    return x

@th.no_grad()
def gamma_multistep_sampling(
    denoiser,
    x,
    sigmas,
    cond,
    nu,
    omega,
    gamma=0.9,
    ctm=False,
    teacher=False,
    progress=False,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    """ Implements CTM's \gamma-sampling """
    s_in = x.new_ones([x.shape[0]])
    if omega is not None:
        omega = omega * th.ones((x.shape[0],), device=x.device)
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    assert gamma != 0.0 and gamma != 1.0
    for i in indices[:-1]:
        sigma = sigmas[i]
        print(sigma, sigmas[i+1], gamma)
        s = (np.sqrt(1. - gamma ** 2) * (sigmas[i + 1] - t_min) + t_min)
        
        x_in = th.cat([x] * 2)
        cond_cfg = cond + ([""] * len(cond))
        denoised = denoiser(x_in, th.cat([sigma * s_in] * 2), cond=cond_cfg, 
                            s=th.cat([s * s_in] * 2), ctm=ctm, teacher=teacher, cfg=omega)
        denoised_text, denoised_uncond = denoised.chunk(2)
        denoised = denoised_uncond + nu * (denoised_text - denoised_uncond)
        if i < len(indices) - 2:
            std = th.sqrt(sigmas[i + 1] ** 2 - s ** 2)
            x = denoised + std * th.randn_like(denoised)
        else:
            x = denoised

    return x
