import json
import os
import random
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_dataset as hf_load_dataset
from datasets import load_dataset_builder, load_from_disk
from torch import nn
from torch.utils.data import Dataset, IterableDataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class StreamingHFDataset(IterableDataset):
    """Dataset wrapper for streaming Hugging Face datasets."""

    def __init__(
        self,
        hf_dataset,
        *,
        dataset_length: int | None = None,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.dataset = hf_dataset
        self.dataset_length = dataset_length
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    @property
    def num_examples(self) -> int | None:
        return self.dataset_length

    def __len__(self):
        if self.dataset_length is None:
            raise TypeError("Streaming dataset does not expose a static length")
        return self.dataset_length

    def __iter__(self):
        for row in self.dataset:
            audio_info = row.get("audio")
            if audio_info is None:
                continue

            if isinstance(audio_info, dict):
                audio_array = audio_info.get("array")
                sample_rate = audio_info.get("sampling_rate")
            else:
                audio_array = getattr(audio_info, "array", None)
                sample_rate = getattr(audio_info, "sampling_rate", None)

            if audio_array is None or sample_rate is None:
                continue

            audio_tensor = torch.as_tensor(audio_array).float()
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.ndim > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            duration = audio_tensor.shape[-1] / sample_rate
            if duration > 30 or duration < 0.3:
                continue

            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                audio_tensor = resampler(audio_tensor)

            mel_spec = self.mel_spectrogram(audio_tensor)
            mel_spec = mel_spec.squeeze(0)

            text = row.get("text")
            if text is None:
                continue

            yield dict(
                mel_spec=mel_spec,
                text=text,
            )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio_path = row["audio_path"]
        text = row["text"]
        duration = row["duration"]

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])

        else:
            audio, source_sample_rate = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            if duration > 15 or duration < 0.3:
                return self.__getitem__((index + 1) % len(self.data))

            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t')

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


# Dynamic Batch Sampler


def _resolve_streaming_dataset(dataset_name: str):
    """Return Hugging Face streaming configuration if dataset should stream."""

    if os.environ.get("F5R_TTS_DISABLE_HF_STREAMING"):
        return None

    name_lower = dataset_name.lower()

    if name_lower.startswith("emilia"):
        languages = dataset_name.split("_")[1:]
        default_split = "train"
        if languages:
            lang_suffix = ".".join(lang.lower() for lang in languages)
            default_split = f"train.{lang_suffix}"

        split = os.environ.get("F5R_TTS_EMILIA_SPLIT", default_split)
        return dict(
            repo_id="amphion/Emilia-Dataset",
            split=split,
            load_kwargs={},
            builder_kwargs={},
        )

    if name_lower.startswith("wenetspeech4tts"):
        parts = dataset_name.split("_")
        tier = parts[1] if len(parts) > 1 else "Premium"
        config_name = os.environ.get("F5R_TTS_WENETSPEECH_CONFIG", tier.lower())
        split = os.environ.get("F5R_TTS_WENETSPEECH_SPLIT", "train")
        return dict(
            repo_id="amphion/WenetSpeech4TTS",
            split=split,
            load_kwargs={"name": config_name},
            builder_kwargs={"name": config_name},
        )

    if name_lower.startswith("rl"):
        repo_id = os.environ.get("F5R_TTS_RL_REPO_ID", "amphion/F5-TTS-RL")
        split = os.environ.get("F5R_TTS_RL_SPLIT", "train")

        config_name = os.environ.get("F5R_TTS_RL_CONFIG")
        if "_" in dataset_name:
            _, maybe_config = dataset_name.split("_", 1)
            if maybe_config:
                config_name = config_name or maybe_config.lower()

        load_kwargs = {}
        builder_kwargs = {}
        if config_name:
            load_kwargs["name"] = config_name
            builder_kwargs["name"] = config_name

        num_examples = os.environ.get("F5R_TTS_RL_NUM_EXAMPLES")
        if num_examples is not None:
            try:
                num_examples = int(num_examples)
            except ValueError:
                num_examples = None

        return dict(
            repo_id=repo_id,
            split=split,
            load_kwargs=load_kwargs,
            builder_kwargs=builder_kwargs,
            num_examples=num_examples,
        )

    return None


class DynamicBatchSampler(Sampler[list[int]]):
    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_last: bool = False,
        repeat_count=1, mini_repeat_count=1
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.repeat_count = repeat_count
        self.mini_repeat_count = mini_repeat_count

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)

        del indices

        # if want to have different batches between epochs, may just set a seed and log it in ckpt
        # cuz during multi-gpu training, although the batch on per gpu not change between epochs, the formed general minibatch is different
        # e.g. for epoch n, use (random_seed + n)
        random.seed(random_seed)
        random.shuffle(batches)

        # repeat
        self.batches = []
        for chunk in batches:
            for _ in range(self.repeat_count):
                batch_sub = []
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        batch_sub.append(index)
                self.batches.append(batch_sub)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset | HFDataset | StreamingHFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    streaming_cfg = _resolve_streaming_dataset(dataset_name)
    if streaming_cfg is not None:
        repo_id = streaming_cfg["repo_id"]
        split = streaming_cfg["split"]
        load_kwargs = streaming_cfg.get("load_kwargs", {})
        builder_kwargs = streaming_cfg.get("builder_kwargs", {})

        print(f"Loading Hugging Face dataset '{repo_id}' (split: '{split}') in streaming mode...")
        hf_dataset = hf_load_dataset(repo_id, split=split, streaming=True, **load_kwargs)

        dataset_length = streaming_cfg.get("num_examples")
        if dataset_length is None:
            try:
                builder = load_dataset_builder(repo_id, **builder_kwargs)
                if split in builder.info.splits:
                    dataset_length = builder.info.splits[split].num_examples
            except Exception as exc:  # pragma: no cover - network/auth dependent
                print(f"Warning: unable to determine dataset length for {repo_id}:{split} ({exc})")
                dataset_length = None

        train_dataset = StreamingHFDataset(
            hf_dataset,
            dataset_length=dataset_length,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDataset":
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
        )

    return train_dataset


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
    )
