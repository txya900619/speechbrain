#!/usr/bin/env python3
"""Recipe for training a whisper-based ASR system with librispeech.
The system employs whisper from OpenAI (https://cdn.openai.com/papers/whisper.pdf).
This recipe take the whisper encoder-decoder to fine-tune on the NLL.

If you want to only use the whisper encoder system, please refer to the recipe
speechbrain/recipes/LibriSpeech/ASR/CTC/train_with_whisper.py

To run this recipe, do the following:
> python train_with_whisper.py hparams/train_hf_whisper.yaml

Authors
 * Adel Moumen 2022, 2024
 * Titouan Parcollet 2022
"""

import logging
import os
import sys
from typing import List

import torch
from datasets import DatasetDict, load_dataset, concatenate_datasets
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main

logger = logging.getLogger(__name__)

CONFIG_NAMES = [
    "XYH-6-X",
    "XYH-6-Y",
    "lavalier",
    "ios",
    "condenser",
    "android",
]

CLEAN_MAPPING = {
    "﹖": "?",
    "！": "!",
    "％": "%",
    "（": "(",
    "）": ")",
    "，": ",",
    "：": ":",
    "；": ";",
    "？": "?",
    "—": "--",
    "─": "-",
}
ACCEPTABLE_CHARS = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "àáéìîòúāō"
    " "
    '!"%()+,-./:;=?_~'
    "‘’“”'"
    "…⋯"
    "、。『』"
    "－"
)


def clean_text(text: str | List[str]):  # 台文數字掉
    text = text.strip()
    text = text.replace("'", " ")
    text = text.replace('"', " ")
    text = text.replace("“", " ")
    text = text.replace("”", " ")
    text = text.replace(":", " ")
    text = text.replace(")", " ")
    text = text.replace("(", " ")
    text = text.strip()
    if text.endswith(","):
        text = text[:-1] + "."
    if text[-1] not in "?!.":
        text += "."
    for bad_char, good_char in CLEAN_MAPPING.items():
        text = text.replace(bad_char, good_char)
    return text


def filter_transcription(transcript):
    for c in transcript:
        if c not in ACCEPTABLE_CHARS:
            print(f"{c}\t: {transcript}")
            return False
    return True


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, bos_tokens_lens = batch.tokens_bos

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            bos_tokens = self.hparams.wav_augment.replicate_labels(bos_tokens)
            bos_tokens_lens = self.hparams.wav_augment.replicate_labels(bos_tokens_lens)

        # We compute the padding mask and replace the values with the pad_token_id
        # that the Whisper decoder expect to see.
        abs_tokens_lens = (bos_tokens_lens * bos_tokens.shape[1]).long()
        pad_mask = (
            torch.arange(abs_tokens_lens.max(), device=self.device)[None, :]
            < abs_tokens_lens[:, None]
        )
        bos_tokens[~pad_mask] = self.tokenizer.pad_token_id

        # Forward encoder + decoder
        enc_out, logits, _ = self.modules.whisper(wavs, bos_tokens)
        log_probs = self.hparams.log_softmax(logits)

        hyps = None
        if stage == sb.Stage.VALID:
            hyps, _, _, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _, _, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return log_probs, hyps, wav_lens, enc_out

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss NLL given predictions and targets."""

        (log_probs, hyps, wav_lens, enc_out) = predictions
        batch = batch.to(self.device)
        channel = batch.channel
        channel = channel.unsqueeze(1)
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        # Label Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens_eos = self.hparams.wav_augment.replicate_labels(tokens_eos)
            tokens_eos_lens = self.hparams.wav_augment.replicate_labels(tokens_eos_lens)

        nll_loss = self.hparams.nll_loss(
            log_probs,
            tokens_eos,
            length=tokens_eos_lens,
        )

        est_channel = self.modules.discriminator(self.modules.revgrad(enc_out))
        channel_loss = self.hparams.compute_d_cost(est_channel, channel)
        loss = nll_loss + channel_loss

        self.nll_loss = nll_loss.detach().cpu()
        self.channel_loss = channel_loss.detach().cpu()

        if stage != sb.Stage.TRAIN:
            tokens, tokens_lens = batch.tokens

            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode(t, skip_special_tokens=True).strip() for t in hyps
            ]

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer.batch_decode(
                target_words, skip_special_tokens=True
            )

            if hasattr(self.hparams, "normalized_transcripts"):
                predicted_words = [
                    self.tokenizer.normalize(text).split(" ")
                    for text in predicted_words
                ]

                target_words = [
                    self.tokenizer.normalize(text).split(" ") for text in target_words
                ]
            else:
                predicted_words = [text.split(" ") for text in predicted_words]
                target_words = [text.split(" ") for text in target_words]

            self.wer_metric.append(ids.tolist(), predicted_words, target_words)
            self.cer_metric.append(ids.tolist(), predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Called after ``fit_batch()``.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        outputs : list or dictionary of torch.Tensors
            Returned value of compute_forward().
        loss : torch.Tensor
            Returned value of compute_objectives().
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        old_lr = self.optimizer.param_groups[-1]["lr"]

        self.hparams.lr_annealing.step()

        if sb.utils.distributed.if_main_process():
            stage_stats = {
                "loss": loss,
                "nll_loss": self.nll_loss,
                "d_loss": self.channel_loss,
            }
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "lr": old_lr,
                },
                train_stats=stage_stats,
            )

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"], num_to_keep=16
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

    def _save_intra_epoch_ckpt(self):
        """Saves a CKPT with specific intra-epoch flag."""
        self.checkpointer.save_and_keep_only(
            end_of_epoch=False,
            num_to_keep=12,
            ckpt_predicate=lambda c: sb.core.INTRA_EPOCH_CKPT_FLAG in c.meta,
            meta={sb.core.INTRA_EPOCH_CKPT_FLAG: True},
            verbosity=logging.DEBUG,
        )


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    config_names = [
        config_name
        for config_name in CONFIG_NAMES
        if config_name != hparams["dataset_config_name"]
    ]

    datasets = {}

    for i, config_name in enumerate(config_names):
        dataset: DatasetDict = load_dataset(hparams["dataset_name"], config_name)
        for split in dataset.keys():
            dataset_split = dataset[split]
            dataset_split = dataset_split.add_column(
                "channel", [i] * len(dataset_split)
            )

            if split not in datasets:
                datasets[split] = []

            datasets[split].append(dataset_split)

    for split in datasets.keys():
        datasets[split] = concatenate_datasets(datasets[split])

    datasets = DatasetDict(datasets)

    datasets = datasets.rename_column("台羅數字調", "sentence")  # for TAT

    all_column_names = set()
    for column_names in datasets.column_names.values():
        all_column_names.update(column_names)
    datasets = datasets.remove_columns(
        all_column_names - set(["audio", "sentence", "duration", "channel"])
    )
    datasets = datasets.map(
        lambda x: {"sentence": clean_text(x["sentence"])}, num_proc=4
    )
    datasets = datasets.filter(
        lambda x: filter_transcription(x["sentence"]), num_proc=4
    )

    train_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        datasets["train"]
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        datasets["validation"]
    )
    test_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        datasets["test"]
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    valid_data = valid_data.filtered_sorted(sort_key="duration")
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("audio")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(audio):
        sig = torch.FloatTensor(audio["array"])
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("sentence")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(sentence):
        wrd = sentence
        yield wrd
        tokens_list = tokenizer.encode(wrd, add_special_tokens=False)
        yield tokens_list
        tokens_list = tokenizer.build_inputs_with_special_tokens(tokens_list)
        tokens_bos = torch.LongTensor(tokens_list[:-1])
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens", "channel"],
    )

    return train_data, valid_data, test_data


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Defining tokenizer and loading it
    tokenizer = hparams["whisper"].tokenizer

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
    )

    # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for Whisper.
    asr_brain.tokenizer = tokenizer

    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        # Training
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_loader_kwargs"],
            valid_loader_kwargs=hparams["valid_loader_kwargs"],
        )

    # Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)

    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], "wer.txt"
    )
    asr_brain.evaluate(
        test_data,
        test_loader_kwargs=hparams["test_loader_kwargs"],
        min_key="WER",
    )
