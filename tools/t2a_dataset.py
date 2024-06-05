import os
import pandas as pd
import datasets

import torch
from torch.utils.data import Dataset, DataLoader

from accelerate.logging import get_logger
logger = get_logger(__name__)

from tools import torch_tools

TARGET_LENGTH = 1024


class Text2AudioDataset(Dataset):
    def __init__(
        self, dataset, prefix, text_column, audio_column,
        num_examples=-1, target_length=1024, augment=False
    ):
        super().__init__()
        inputs = list(dataset[text_column])
        self.captions = [prefix + inp for inp in inputs]
        self.audio_paths = list(dataset[audio_column])
        self.indices = list(range(len(self.captions)))
        self.target_length = target_length
        self.augment = augment

        self.mapper = {}
        for index, audio, caption in zip(self.indices, self.audio_paths, inputs):
            self.mapper[index] = [audio, caption]

        if num_examples != -1:
            self.captions = self.captions[:num_examples]
            self.audio_paths = self.audio_paths[:num_examples]
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.captions)

    @property
    def seg_length(self):
        return self.target_length * 160

    def __getitem__(self, index):
        indice = self.indices[index]
        caption, audio_path = self.captions[index], self.audio_paths[index]
        waveform = torch_tools.read_wav_file(audio_path, self.seg_length)
        return caption, waveform, indice

    def collate_fn(self, data):
        """ Return:
        a list of captions,
        a tensor containing the waveforms,
        a tensor containing the indices.
        """
        df = pd.DataFrame(data)
        captions, waveforms, indices = [df[i].tolist() for i in df]
        waveforms = torch.cat(waveforms, dim=0)

        if self.augment:
            num_mix_items = len(captions) // 2
            mixed_waveforms, mixed_captions = torch_tools.augment(
                waveforms, captions, num_items=num_mix_items
            )
            waveforms = torch.cat([waveforms, mixed_waveforms], dim=0)
            captions += mixed_captions

        return captions, waveforms, torch.tensor(indices)


class T2APairedDataset(Dataset):
    def __init__(
        self, dataset, generated_path, mel_path, num_examples=-1,
        target_length=1024, sample_rate=16000
    ):
        super().__init__()
        # assert os.path.isfile(dataset_json_path), f"{dataset_json_path} is not a file."
        # self.dataset = pd.read_json(dataset_json_path, lines=True)
        assert os.path.isdir(generated_path), f"{generated_path} is not a directory."
        self.generated_path = generated_path
        # assert mel_path is None or os.path.isfile(mel_path), f"{mel_path} is not a directory."
        # self.mel = torch.load(mel_path) if mel_path is not None else None

        self.captions = list(dataset["caption"])
        self.audio_paths = list(dataset["file_name"])
        self.indices = list(range(len(self.captions)))
        self.target_length = target_length

        if isinstance(sample_rate, list):
            self.sample_rate = [int(sr) for sr in sample_rate]
        else:
            self.sample_rate = int(sample_rate)
        print(f"Target sample rate: {self.sample_rate}")

        if num_examples != -1:
            self.captions = self.captions[:num_examples]
            self.audio_paths = self.audio_paths[:num_examples]
            self.indices = self.indices[:num_examples]
            # if self.mel is not None:
            #     self.mel = self.mel[:num_examples, :, :, :]

    def __len__(self):
        return len(self.captions)

    @property
    def seg_length(self):
        sr = self.sample_rate[-1] if isinstance(self.sample_rate, list) else self.sample_rate
        return int(self.target_length * sr / 100), int(1000 * sr / 100)

    def __getitem__(self, index):
        # Get ground-truth waveform
        caption, audio_path = self.captions[index], self.audio_paths[index]
        gt_waveform = torch_tools.read_wav_file(
            audio_path, self.seg_length[1], self.sample_rate
        )
        # Get generated waveform
        indice = self.indices[index]
        gen_wav_path = f"{self.generated_path}/{os.path.basename(audio_path)}"
        gen_waveform = torch_tools.read_wav_file(
            gen_wav_path, self.seg_length[0], self.sample_rate
        )
        # # Get generated Mel spectrogram
        # gen_mel = self.mel[index, :, :, :].unsqueeze(dim=0) \
        #     if self.mel is not None else None
        return caption, gt_waveform, gen_waveform # gen_mel

    def collate_fn(self, data):
        """ Return:
        a list of captions,
        a tensor containing the groundtruth waveforms,
        a tensor containing the generated waveforms.
        """
        df = pd.DataFrame(data)
        captions, gt_waveforms, gen_waveforms = [df[i].tolist() for i in df]
        gt_waveforms = torch.cat(gt_waveforms, dim=0)
        gen_waveforms = torch.cat(gen_waveforms, dim=0)

        # if gen_mel is None or None in gen_mel:
        #     gen_mel = None
        # else:
        #     gen_mel = torch.cat(gen_mel, dim=0)

        return captions, gt_waveforms, gen_waveforms


def get_dataloaders(args, accelerator):
    # Get the datasets
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    if args.test_file is not None:
        data_files["test"] = args.test_file
    else:
        if args.validation_file is not None:
            data_files["test"] = args.validation_file

    extension = args.train_file.split(".")[-1]
    raw_datasets = datasets.load_dataset(extension, data_files=data_files)
    text_column, audio_column = args.text_column, args.audio_column

    if args.prefix:
        prefix = args.prefix
    else:
        prefix = ""

    # Datasets
    with accelerator.main_process_first():
        train_dataset = Text2AudioDataset(
            raw_datasets["train"], prefix, text_column, audio_column,
            args.num_examples, target_length=TARGET_LENGTH, augment=True
        )
        eval_dataset = Text2AudioDataset(
            raw_datasets["validation"], prefix, text_column, audio_column,
            args.num_examples, target_length=TARGET_LENGTH, augment=False
        )
        test_dataset = Text2AudioDataset(
            raw_datasets["test"], prefix, text_column, audio_column,
            args.num_examples, target_length=TARGET_LENGTH, augment=False
        )

        logger.info(
            f"Num instances in train: {len(train_dataset)}, "
            f"validation: {len(eval_dataset)}, "
            f"test: {len(test_dataset)}."
        )

    # Dataloaders
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size,
        num_workers=4, collate_fn=train_dataset.collate_fn
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size,
        num_workers=4, collate_fn=eval_dataset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size,
        num_workers=4, collate_fn=test_dataset.collate_fn
    )

    return train_dataloader, eval_dataloader, test_dataloader


def get_test_dataloader(test_file, text_column, audio_column, batch_size):
    # Dataset
    extension = test_file.split(".")[-1]
    data_files = {"test": test_file}
    test_set = datasets.load_dataset(extension, data_files=data_files)["test"]
    test_dataset = Text2AudioDataset(
        test_set, text_column=text_column, audio_column=audio_column,
        num_examples=-1, prefix="", target_length=TARGET_LENGTH, augment=False
    )
    try:
        logger.info(f"Num instances in test dataset: {len(test_dataset)}.")
    except:
        print(f"Num instances in test dataset: {len(test_dataset)}.")

    # Dataloader
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=batch_size,
        num_workers=4, collate_fn=test_dataset.collate_fn
    )
    return test_dataloader
