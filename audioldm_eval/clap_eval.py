import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from laion_clap import CLAP_Module
from datasets import load_dataset

from audioldm_eval.audio.tools import write_json
from tools.t2a_dataset import T2APairedDataset
from tools.torch_tools import seed_all


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def get_clap_features(pairedtextloader, clap_model):
    gt_features, gen_features, text_features = [], [], []

    for captions, gt_waves, gen_waves in tqdm(pairedtextloader):
        gt_waves = gt_waves.squeeze(1).float().to(device)
        gen_waves = gen_waves.squeeze(1).float().to(device)

        with torch.no_grad():
            seed_all(0)
            gt_features += [clap_model.get_audio_embedding_from_data(
                x=gt_waves, use_tensor=True
            )]
            seed_all(0)
            gen_features += [clap_model.get_audio_embedding_from_data(
                x=gen_waves, use_tensor=True
            )]
            seed_all(0)
            text_features += [clap_model.get_text_embedding(
                captions, use_tensor=True
            )]
            # TODO: get embedding from mel

    gt_features = torch.cat(gt_features, dim=0)
    gen_features = torch.cat(gen_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    return gt_features, gen_features, text_features


class EvaluationHelper_CLAP:
    def __init__(self, sampling_rate, device, backbone="cnn14") -> None:

        self.device = device
        self.backbone = backbone
        self.sampling_rate = sampling_rate
        self.clap_model = CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny').to(device)
        self.clap_model.load_ckpt(
            'ckpt/630k-audioset-best.pt', model_id=1
        )

    def file_init_check(self, dir):
        assert os.path.exists(dir), "The path does not exist %s" % dir
        assert len(os.listdir(dir)) > 1, "There is no files in %s" % dir

    def get_filename_intersection_ratio(
        self, dir1, dir2, threshold=0.99, limit_num=None
    ):
        self.datalist1 = [os.path.join(dir1, x) for x in os.listdir(dir1)]
        self.datalist1 = sorted(self.datalist1)
        self.datalist1 = [item for item in self.datalist1 if item.endswith(".wav")]

        self.datalist2 = [os.path.join(dir2, x) for x in os.listdir(dir2)]
        self.datalist2 = sorted(self.datalist2)
        self.datalist2 = [item for item in self.datalist2 if item.endswith(".wav")]

        data_dict1 = {os.path.basename(x): x for x in self.datalist1}
        data_dict2 = {os.path.basename(x): x for x in self.datalist2}

        keyset1 = set(data_dict1.keys())
        keyset2 = set(data_dict2.keys())

        intersect_keys = keyset1.intersection(keyset2)
        if len(intersect_keys) / len(keyset1) > threshold \
            and len(intersect_keys) / len(keyset2) > threshold:
            return True
        else:
            return False

    def calculate_metrics(
        self, test_file, generate_files_path, groundtruth_path,
        mel_path, same_name, target_length=1000, limit_num=None
    ):
        # Generation, target
        seed_all(0)
        print(f"generate_files_path: {generate_files_path}")
        print(f"groundtruth_path: {groundtruth_path}")

        
        data_files = {}
        if test_file is not None:
            data_files["test"] = test_file   

        extension = test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
        
        pairedtextdataset = T2APairedDataset(raw_datasets["test"], generated_path=generate_files_path,
            target_length=target_length, mel_path=None, sample_rate=48000
        )
        # pairedtextdataset = T2APairedDataset(
        #     dataset_json_path=dataset_json_path, generated_path=generate_files_path,
        #     target_length=target_length, mel_path=mel_path, sample_rate=[16000, 48000]
        # )
        pairedtextloader = DataLoader(
            pairedtextdataset, batch_size=16, num_workers=8, shuffle=False,
            collate_fn=pairedtextdataset.collate_fn
        )

        # melpaireddataset = MelPairedDataset(
        #     generate_files_path, groundtruth_path, self._stft, self.sampling_rate,
        #     self.fbin_mean, self.fbin_std, limit_num=limit_num,
        # )
        # melpairedloader = DataLoader(
        #     melpaireddataset, batch_size=1, sampler=None, num_workers=8, shuffle=False
        # )

        out = {}

        # Get CLAP features
        print("Calculating CLAP score...")  # CLAP Score
        gt_feat, gen_feat, text_feat = get_clap_features(pairedtextloader, self.clap_model)
        # CLAP similarity calculation
        gt_text_similarity = cosine_similarity(gt_feat, text_feat, dim=1)
        gen_text_similarity = cosine_similarity(gen_feat, text_feat, dim=1)
        gen_gt_similarity = cosine_similarity(gen_feat, gt_feat, dim=1)
        gt_text_similarity = torch.clamp(gt_text_similarity, min=0)
        gen_text_similarity = torch.clamp(gen_text_similarity, min=0)
        gen_gt_similarity = torch.clamp(gen_gt_similarity, min=0)
        # Update output dict
        out.update({
            'gt_text_clap_score': gt_text_similarity.mean().item() * 100.,
            'gen_text_clap_score': gen_text_similarity.mean().item() * 100.,
            'gen_gt_clap_score': gen_gt_similarity.mean().item() * 100.
        })
        keys_list = [
            "gt_text_clap_score", "gen_text_clap_score", "gen_gt_clap_score", "frechet_audio_distance"
        ]
        result = {}
        for key in keys_list:
            result[key] = round(out.get(key, float("nan")), 4)

        json_path = generate_files_path + "_evaluation_results.json"
        write_json(result, json_path)
        return result

    def get_featuresdict(self, dataloader):
        out, out_meta = None, None

        for waveform, filename in tqdm(dataloader):
            metadict = {"file_path_": filename}
            waveform = waveform.squeeze(1).float().to(self.device)

            with torch.no_grad():
                featuresdict = self.mel_model(waveform)
                featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

            out = featuresdict if out is None else {
                k: out[k] + featuresdict[k] for k in out.keys()
            }
            out_meta = metadict if out_meta is None else {
                k: out_meta[k] + metadict[k] for k in out_meta.keys()
            }

        out = {k: torch.cat(v, dim=0) for k, v in out.items()}
        return {**out, **out_meta}

    def sample_from(self, samples, number_to_use):
        assert samples.shape[0] >= number_to_use
        rand_order = np.random.permutation(samples.shape[0])
        return samples[rand_order[: samples.shape[0]], :]

    def main(
        self, test_csv_file, generated_files_path, groundtruth_path,
        mel_path=None, target_length=1000, limit_num=None,
    ):
        self.file_init_check(generated_files_path)
        self.file_init_check(groundtruth_path)

        same_name = self.get_filename_intersection_ratio(
            generated_files_path, groundtruth_path, limit_num=limit_num
        )
        return self.calculate_metrics(
            test_csv_file, generated_files_path, groundtruth_path,
            mel_path, same_name, target_length, limit_num
        )
