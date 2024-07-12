# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Perform preprocessing and raw feature extraction for Swahili IPA dataset."""

import os
import re
from string import punctuation

import numpy as np
import soundfile as sf
from dataclasses import dataclass

from gruut import sentences

from tensorflow_tts.processor.base_processor import BaseProcessor
from tensorflow_tts.utils.utils import PROCESSOR_FILE_NAME

valid_symbols = [
    "f",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "s",
    "t",
    "t͡ʃ",
    "u",
    "v",
    "w",
    "x",
    "z",
    "ð",
    "ŋ",
    "ɑ",
    "ɓ",
    "ɔ",
    "ɗ",
    "ɛ",
    "ɠ",
    "ɣ",
    "ɾ",
    "ʃ",
    "ʄ",
    "θ",
    "ᵐɓ",
    "ᵑg",
    "ᶬv",
    "ⁿz",
    "ⁿɗ",
    "ⁿɗ͡ʒ",
]

_punctuation = "!,.?;:"
_sil = "@SIL"
_eos = "@EOS"
_pad = "@PAD"
_ipa = ["@" + s for s in valid_symbols]

SWAHILI_IPA_SYMBOLS = [_pad] + _ipa + list(_punctuation) + [_sil] + [_eos]


@dataclass
class SwahiliIPAProcessor(BaseProcessor):

    mode: str = "train"
    train_f_name: str = "train.txt"
    positions = {
        "file": 0,
        "text": 1,
        "speaker_name": 2,
    }  # positions of file,text,speaker_name after split line
    f_extension: str = ".wav"
    cleaner_names: str = None

    def create_items(self):
        with open(os.path.join(self.data_dir, self.train_f_name), mode="r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(self.delimiter)
                wav_path = os.path.join(self.data_dir, parts[self.positions["file"]])
                wav_path = (
                    wav_path + self.f_extension if wav_path[-len(self.f_extension) :] != self.f_extension else wav_path
                )
                text = parts[self.positions["text"]]
                speaker_name = parts[self.positions["speaker_name"]]
                self.items.append([text, wav_path, speaker_name])

    def get_one_sample(self, item):
        text, wav_path, speaker_name = item
        audio, rate = sf.read(wav_path, dtype="float32")

        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": wav_path.split("/")[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def setup_eos_token(self):
        return None

    def save_pretrained(self, saved_path):
        os.makedirs(saved_path, exist_ok=True)
        self._save_mapper(os.path.join(saved_path, PROCESSOR_FILE_NAME), {})

    def text_to_sequence(self, text):
        if self.mode == "train":  # in train mode text should be already transformed to phonemes
            return self.symbols_to_ids(self.clean_g2p(text.split()))
        else:
            return self.inference_text_to_seq(text)

    def inference_text_to_seq(self, text: str):
        return self.symbols_to_ids(self.text_to_ph(text))

    def symbols_to_ids(self, symbols_list: list):
        return [self.symbol_to_id[s] for s in symbols_list]

    def text_to_ph(self, text: str):
        phn_arr = []
        for words in sentences(text, lang="sw"):
            for word in words:
                if word.is_major_break or word.is_minor_break:
                    phn_arr += [word.text]
                elif word.phonemes:
                    phonemes = word.phonemes[:]

                    # NOTE: gruut doesn't handle "ng'" /ŋ/
                    # we need to fix e.g. ng'ombe -> /ŋombe/ instead of /ᵑgombe/
                    NG_GRAPHEME = "ng'"
                    NG_PRENASALIZED_PHONEME = "ᵑg"
                    NG_PHONEME = "ŋ"
                    if NG_GRAPHEME in word.text and NG_PHONEME in valid_symbols:
                        ng_graphemes = re.findall(f"{NG_GRAPHEME}?", word.text)
                        ng_phonemes_idx = [i for i, p in enumerate(phonemes) if p == NG_PRENASALIZED_PHONEME]
                        assert len(ng_graphemes) == len(ng_phonemes_idx)
                        for i, g in zip(ng_phonemes_idx, ng_graphemes):
                            phonemes[i] = NG_PHONEME if g == NG_GRAPHEME else phonemes[i]

                    phn_arr += phonemes

        return self.clean_g2p(phn_arr)

    def clean_g2p(self, g2p_text: list):
        data = []
        for txt in g2p_text:
            if txt in punctuation:
                data.append(txt)
            elif txt != " ":
                data.append("@" + txt)
        return data
