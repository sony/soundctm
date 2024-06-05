import torch
import torchaudio
import random
import itertools
import numpy as np
import os
from torchaudio import transforms as T

def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode="A_weighting"):
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    elif fs == 48000:
        n_fft = 4096
    else:
        raise Exception("Invalid fs {}".format(fs))
    stride = n_fft // 2

    gain = []
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == "RMSE":
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == "A_weighting":
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception("Invalid mode {}".format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)
    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))
    return sound

def normalize_wav(waveform):
    waveform = waveform - torch.mean(waveform)
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    return waveform * 0.5


def pad_wav(waveform, segment_length):
    waveform_length = len(waveform)
    
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    else:
        pad_wav = torch.zeros(segment_length - waveform_length).to(waveform.device)
        waveform = torch.cat([waveform, pad_wav])
        return waveform
    
    
def _pad_spec(fbank, target_length=1024):
    batch, n_frames, channels = fbank.shape
    p = target_length - n_frames
    if p > 0:
        pad = torch.zeros(batch, p, channels).to(fbank.device)
        fbank = torch.cat([fbank, pad], 1)
    elif p < 0:
        fbank = fbank[:, :target_length, :]

    if channels % 2 != 0:
        fbank = fbank[:, :, :-1]

    return fbank


def read_wav_file(filename, segment_length, new_freq=16000):
    waveform, sr = torchaudio.load(filename)  # Faster!!!
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=new_freq, rolloff=0.95, resampling_method = 'sinc_interp_kaiser' )[0]
    # print("waveform_before_normalized.shape", waveform.shape)
    try:
        waveform = normalize_wav(waveform)
    except:
        print ("Exception normalizing:", filename)
        waveform = torch.ones(160000)
    waveform = pad_wav(waveform, segment_length).unsqueeze(0)
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform
    # print("waveform_after_normalized.shape", waveform.shape)
    return waveform

def read_wav_file_sa(filename, segment_length, new_freq=16000):
    waveform, sr = torchaudio.load(filename)  # Faster!!!
    resample_tf = T.Resample(sr, new_freq)
    waveform = resample_tf(waveform)
    print("waveform.shape", waveform.shape)
    # waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=new_freq)[0]
    try:
        waveform = normalize_wav(waveform)
    except:
        print ("Exception normalizing:", filename)
        waveform = torch.ones(160000)
    waveform = pad_wav(waveform, segment_length).unsqueeze(0)
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform
    return waveform


def get_mel_from_wav(audio, _stft):
    audio = torch.nan_to_num(torch.clip(audio, -1, 1))
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
    return melspec, log_magnitudes_stft, energy


def wav_to_fbank(paths, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None

    waveform = torch.cat([read_wav_file(path, target_length * 160) for path in paths], 0)  # hop size is 160

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    fbank = fbank.transpose(1, 2)
    log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    return fbank, log_magnitudes_stft, waveform


def uncapitalize(s):
    if s:
        return s[:1].lower() + s[1:]
    else:
        return ""

    
def mix_wavs_and_captions(path1, path2, caption1, caption2, target_length=1024):
    sound1 = read_wav_file(path1, target_length * 160)[0].numpy()
    sound2 = read_wav_file(path2, target_length * 160)[0].numpy()
    mixed_sound = mix(sound1, sound2, 0.5, 16000).reshape(1, -1)
    mixed_caption = "{} and {}".format(caption1, uncapitalize(caption2))
    return mixed_sound, mixed_caption


def augment(paths, texts, num_items=4, target_length=1024):
    mixed_sounds, mixed_captions = [], []
    combinations = list(itertools.combinations(list(range(len(paths))), 2))
    random.shuffle(combinations)
    if len(combinations) < num_items:
        selected_combinations = combinations
    else:
        selected_combinations = combinations[:num_items]
        
    for (i, j) in selected_combinations:
        new_sound, new_caption = mix_wavs_and_captions(paths[i], paths[j], texts[i], texts[j], target_length)
        mixed_sounds.append(new_sound)
        mixed_captions.append(new_caption)
        
    waveform = torch.tensor(np.concatenate(mixed_sounds, 0))
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform
    
    return waveform, mixed_captions

def mix_wavs_and_captions_441(path1, path2, caption1, caption2, target_length=1024, sample_rate=44100):
    sound1 = read_wav_file(path1, target_length, sample_rate)[0].numpy()
    sound2 = read_wav_file(path2, target_length, sample_rate)[0].numpy()
    mixed_sound = mix(sound1, sound2, 0.5, sample_rate).reshape(1, -1)
    mixed_caption = "{} and {}".format(caption1, uncapitalize(caption2))
    return mixed_sound, mixed_caption

def augment_441(paths, texts, num_items=4, target_length=1024, sample_rate=44100):
    mixed_sounds, mixed_captions = [], []
    combinations = list(itertools.combinations(list(range(len(paths))), 2))
    random.shuffle(combinations)
    if len(combinations) < num_items:
        selected_combinations = combinations
    else:
        selected_combinations = combinations[:num_items]
        
    for (i, j) in selected_combinations:
        new_sound, new_caption = mix_wavs_and_captions_441(paths[i], paths[j], texts[i], texts[j], target_length, sample_rate)
        mixed_sounds.append(new_sound)
        mixed_captions.append(new_caption)
        
    waveform = torch.tensor(np.concatenate(mixed_sounds, 0))
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform
    
    return waveform, mixed_captions


def augment_wav_to_fbank(paths, texts, num_items=4, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None
    
    waveform, captions = augment(paths, texts, num_items, target_length)
    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    fbank = fbank.transpose(1, 2)
    log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    return fbank, log_magnitudes_stft, waveform, captions

def seed_all(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True