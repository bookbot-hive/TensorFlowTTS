# Implementing Custom Processor

There are only a handful of languages supported by the original TensorFlowTTS repository. Supporting a new language requires the implementation of a custom processor, which is crucial when designing text-to-speech models. 

## Processor and Phonemization

A processor serves as a tokenizer that helps encode texts into IDs, i.e. numerical representation of texts that you're going to synthesize. Naively, we can represent each character of say, the English language, as one unique ID and have characters as our 'tokens'. This is indeed the default behavior of the [LJSpeech Processor](https://github.com/w11wo/TensorFlowTTS/blob/master/tensorflow_tts/processor/ljspeech.py).

Other processors, however, choose to utilize grapheme-to-phoneme converters, and have phonemes as their tokens instead of individual raw characters. This intuitively makes sense, since phonemization in languages like English isn't straightforward, and mapping phonemes to their corresponding mel-spectrogram makes much more sense.

However, this is only possible if a grapheme-to-phoneme (g2p) converter for a particular language exists. High-resource languages like English has tools such as [g2pE](https://github.com/Kyubyong/g2p) and [gruut](https://github.com/rhasspy/gruut), Korean has [g2pK](https://github.com/Kyubyong/g2pK), etc. Alternatively, you might be able to build your own g2p converter if you at least have a lexicon (see [ipa-dict](https://github.com/open-dict-data/ipa-dict)), which you can then take and use to build g2p models via [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/index.html#first-steps-train-g2p), for example.

What if you don't, though? High-to-mid-resource might be able to get away with existing curated tools and/or lexicons, but low-resource languages might be a bit more problematic. In that case, probably the best bet is a character-based tokenizer, where your graphemes (characters) serve as proxy-phonemes. [Meyer et al. (2022)](https://arxiv.org/abs/2207.03546) trained text-to-speech models on low-resource African languages, and for certain languages where a lexicon is unavailable, they used proxy-phonemes:

> Two languages (ewe and yor) were aligned via forced alignment from scratch. Using only the found audio and transcripts (i.e., without a pre-trained acoustic model), an acoustic model was trained and the data aligned with the Montreal Forced Aligner. Graphemes were used as a proxy for phonemes in place of G2P data.

As a practical example, I will be attempting to implement a character-based tokenizer for Javanese. Let's get started!

## Processor

### Vocabulary

You first have to create a new processor class under [tensorflow_tts/processor](https://github.com/w11wo/TensorFlowTTS/tree/master/tensorflow_tts/processor), which I'll call `JavaneseCharacterProcessor`. I really like how the [LibriTTS Processor](https://github.com/w11wo/TensorFlowTTS/blob/master/tensorflow_tts/processor/libritts.py) works -- it is well integrated with Montreal Forced Aligner for duration extraction and multi-speaker models. Because of that, I'll be basing my new processor on that file.

!!! note

    This was also how I based both [`EnglishIPAProcessor`](https://github.com/w11wo/TensorFlowTTS/blob/master/tensorflow_tts/processor/english_ipa.py) and [`IndonesianIPAProcessor`](https://github.com/w11wo/TensorFlowTTS/blob/master/tensorflow_tts/processor/indonesian_ipa.py).


To begin, we have to define a hard-coded list of symbols (vocabulary) that the processor will support. This is highly dependent on the dataset that you're using, as we would want to cover all possible graphemes (so none of them become unknowns). In my case, there's the 26 latin alphabet characters A-Z, plus two additional E's with diacritics: é and è.

```py title="tensorflow_tts/processor/javanese_char.py"
valid_symbols = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "è",
    "é",
]
```

Be sure to inspect your dataset and cover all possible tokens.

Next would be punctuations. Again, this is dependent on how you would like to design the processor and which punctuations you would like to keep. I normally only go for `! , . ? ; :` and nothing else. Anything other than those will simply be ignored. Note that this will be important during inference -- anything outside of the vocabulary will be ignored!

```py title="tensorflow_tts/processor/javanese_char.py"
_punctuation = "!,.?;:"
```

There are also additional special tokens such as `@SIL`, `@EOS`, `@PAD` for silence, end-of-sentence, and padding tokens, respectively.

```py title="tensorflow_tts/processor/javanese_char.py"
_sil = "@SIL"
_eos = "@EOS"
_pad = "@PAD"
_char = ["@" + s for s in valid_symbols]

JAVANESE_CHARACTER_SYMBOLS = [_pad] + _char + list(_punctuation) + [_sil] + [_eos]
```

### Metadata Format

An important method in `LibriTTSProcessor`-based processors is how metadata is formatted and later read. Sticking to the original implementation, I will keep the default format of having a metadata called `train.txt`, which is populated with `|`-delimited lines. Each line has 3 columns (in this particular order): 

1. Path to file
2. Text read
3. Speaker name

Moreover, audios are expected to be of `.wav` format. These are then implemented as class variables inside the processor and influences how the `create_items` and `get_one_sample` methods behave. `create_items` "creates" each training sample, and `get_one_sample` loads each training sample in their right formats (e.g. read audio via SoundFile, convert text to sequence of IDs, etc.)

```py title="tensorflow_tts/processor/javanese_char.py"
@dataclass
class JavaneseCharacterProcessor(BaseProcessor):

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
        with open(
            os.path.join(self.data_dir, self.train_f_name), mode="r", encoding="utf-8"
        ) as f:
            for line in f:
                parts = line.strip().split(self.delimiter)
                wav_path = os.path.join(self.data_dir, parts[self.positions["file"]])
                wav_path = (
                    wav_path + self.f_extension
                    if wav_path[-len(self.f_extension) :] != self.f_extension
                    else wav_path
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

    ...

```

### Text-to-Sequence

Then, the only remaining crucial feature to implement is how we will be converting texts into sequence of IDs. Again, following the original implementation of `LibriTTSProcessor`, our processor will have two modes: train and eval. For training, we expect our text is training-ready. What I meant by that is, in the metadata file, all texts have been pre-converted into tokens and are separated by whitespaces, for example:

```
k a p i n g l i m a n i n d a a k e k a j i SIL
```

Because of that, we would not need to re-separate them and could simply encode/map them to their corresponding IDs via the `symbols_to_id` method. On the other hand, we would probably want to have a separate behavior for inference purposes.

```py title="tensorflow_tts/processor/javanese_char.py"
@dataclass
class JavaneseCharacterProcessor(BaseProcessor):

    ...

    def text_to_sequence(self, text):
        if (
            self.mode == "train"
        ):  # in train mode text should be already transformed to characters
            return self.symbols_to_ids(self.clean_char(text.split()))
        else:
            return self.inference_text_to_seq(text)

    def inference_text_to_seq(self, text: str):
        return self.symbols_to_ids(self.text_to_char(text))

    def symbols_to_ids(self, symbols_list: list):
        return [self.symbol_to_id[s] for s in symbols_list]

    ...

```

The code above should be relatively self-explanatory. On training mode, we simply split tokens by whitespace, clean them, and encode them as IDs. On inference, we first need to convert texts to characters, and only then encode them as IDs.

Converting texts to characters is quite straightforward for the case of character-based processors. We would just need to iterate through each characters and that's it! However, for more complex processors that involve g2p conversion, this is probably where you'd want to integrate it. For instance, in [`EnglishIPAProcessor`](https://github.com/w11wo/TensorFlowTTS/blob/master/tensorflow_tts/processor/english_ipa.py#L179), this is where we pass the job to gruut:

```py title="tensorflow_tts/processor/english_ipa.py"
@dataclass
class EnglishIPAProcessor(BaseProcessor):

    ...

    def text_to_ph(self, text: str):
        phn_arr = []
        for words in sentences(text):
            for word in words:
                if word.is_major_break or word.is_minor_break:
                    phn_arr += [word.text]
                elif word.phonemes:
                    phn_arr += word.phonemes

        return self.clean_g2p(phn_arr)
    
    ...

```

And finally, we need to implement how we're going to "clean" tokens, namely, separating punctuations from actual character tokens and special tokens (the latter two beginning with an `@`):

```py title="tensorflow_tts/processor/javanese_char.py"
@dataclass
class JavaneseCharacterProcessor(BaseProcessor):

    ...

    def clean_char(self, characters: list):
        data = []
        for char in characters:
            if char in _punctuation:
                data.append(char)
            elif char != " ":
                data.append("@" + char.lower())
        return data
```

One thing not to miss is adding our new processor class to [`tensorflow_tts/processor/__init__.py`](https://github.com/w11wo/TensorFlowTTS/blob/master/tensorflow_tts/processor/__init__.py):

```py title="tensorflow_tts/processor/__init__.py"
from tensorflow_tts.processor.javanese_char import JavaneseCharacterProcessor
```

## Preprocess

Once you're done with implementing the new processor class, you'll need to also register it to the pre-processor. It's fairly simple to do, with only a few additional lines to add in certain lines:

```py title="tensorflow_tts/bin/preprocess.py"
from tensorflow_tts.processor import JavaneseCharacterProcessor
from tensorflow_tts.processor.javanese_char import JAVANESE_CHARACTER_SYMBOLS
```

```diff title="tensorflow_tts/bin/preprocess.py"
    ...

    parser.add_argument(
        "--dataset",
        type=str,
        default="ljspeech",
        choices=[
            "ljspeech",
            "ljspeech_multi",
            "kss",
            "libritts",
            "baker",
            "thorsten",
            "ljspeechu",
            "synpaflex",
            "jsut",
            "indonesianipa",
            "englishipa",
+           "javanesechar", # what we're going to call our processor
        ],
        help="Dataset to preprocess.",
    )

...

def preprocess():
    """Run preprocessing process and compute statistics for normalizing."""
    config = parse_and_config()

    dataset_processor = {
        "ljspeech": LJSpeechProcessor,
        "ljspeech_multi": LJSpeechMultiProcessor,
        "kss": KSSProcessor,
        "libritts": LibriTTSProcessor,
        "baker": BakerProcessor,
        "thorsten": ThorstenProcessor,
        "ljspeechu": LJSpeechUltimateProcessor,
        "synpaflex": SynpaflexProcessor,
        "jsut": JSUTProcessor,
        "indonesianipa": IndonesianIPAProcessor,
        "englishipa": EnglishIPAProcessor,
+       "javanesechar": JavaneseCharacterProcessor,
    }

    dataset_symbol = {
        "ljspeech": LJSPEECH_SYMBOLS,
        "ljspeech_multi": LJSPEECH_SYMBOLS,
        "kss": KSS_SYMBOLS,
        "libritts": LIBRITTS_SYMBOLS,
        "baker": BAKER_SYMBOLS,
        "thorsten": THORSTEN_SYMBOLS,
        "ljspeechu": LJSPEECH_U_SYMBOLS,
        "synpaflex": SYNPAFLEX_SYMBOLS,
        "jsut": JSUT_SYMBOLS,
        "indonesianipa": INDONESIAN_IPA_SYMBOLS,
        "englishipa": ENGLISH_IPA_SYMBOLS,
+       "javanesechar": JAVANESE_CHARACTER_SYMBOLS,
    }

    dataset_cleaner = {
        "ljspeech": "english_cleaners",
        "ljspeech_multi": "english_cleaners",
        "kss": "korean_cleaners",
        "libritts": None,
        "baker": None,
        "thorsten": "german_cleaners",
        "ljspeechu": "english_cleaners",
        "synpaflex": "basic_cleaners",
        "jsut": None,
        "indonesianipa": None,
        "englishipa": None,
+       "javanesechar": None,
    }

```

## Integrating with text2mel Models

Lastly, we need to integrate our processor with existing text2mel models, such as [FastSpeech2](https://github.com/w11wo/TensorFlowTTS/blob/master/tensorflow_tts/configs/fastspeech.py). The number of tokens in the processor will be used as the vocabulary size of the text2mel model.

```py title="tensorflow_tts/configs/fastspeech.py"
from tensorflow_tts.processor.javanese_char import (
    JAVANESE_CHARACTER_SYMBOLS as javanese_char_symbols,
)
```

```diff title="tensorflow_tts/configs/fastspeech.py"

        ...

        if dataset == "ljspeech":
            self.vocab_size = vocab_size
        elif dataset == "kss":
            self.vocab_size = len(kss_symbols)
        elif dataset == "baker":
            self.vocab_size = len(bk_symbols)
        elif dataset == "libritts":
            self.vocab_size = len(lbri_symbols)
        elif dataset == "jsut":
            self.vocab_size = len(jsut_symbols)
        elif dataset == "ljspeechu":
            self.vocab_size = len(lju_symbols)
        elif dataset == "indonesianipa":
            self.vocab_size = len(indonesian_ipa_symbols)
        elif dataset == "englishipa":
            self.vocab_size = len(english_ipa_symbols)
+       elif dataset == "javanesechar":
+           self.vocab_size = len(javanese_char_symbols)
        else:
            raise ValueError("No such dataset: {}".format(dataset))

        ...
```

## AutoProcessor

A handy feature found in TensorFlowTTS is the ability to load processors from HuggingFace Hub. This allows you to do something like:

```py
from tensorflow_tts.inference import AutoProcessor

processor = AutoProcessor.from_pretrained("bookbot/lightspeech-mfa-id")
```

To allow such support for our custom processor, we simply have to add it to [`tensorflow_tts/inference/auto_processor.py`](https://github.com/w11wo/TensorFlowTTS/blob/master/tensorflow_tts/inference/auto_processor.py):

```diff title="tensorflow_tts/inference/auto_processor.py"
from tensorflow_tts.processor import (
    LJSpeechProcessor,
    KSSProcessor,
    BakerProcessor,
    LibriTTSProcessor,
    ThorstenProcessor,
    LJSpeechUltimateProcessor,
    SynpaflexProcessor,
    JSUTProcessor,
    LJSpeechMultiProcessor,
    IndonesianIPAProcessor,
    EnglishIPAProcessor,
+   JavaneseCharacterProcessor,
)

from tensorflow_tts.utils import CACHE_DIRECTORY, PROCESSOR_FILE_NAME, LIBRARY_NAME
from tensorflow_tts import __version__ as VERSION
from huggingface_hub import hf_hub_url, cached_download

CONFIG_MAPPING = OrderedDict(
    [
        ("LJSpeechProcessor", LJSpeechProcessor),
        ("LJSpeechMultiProcessor", LJSpeechMultiProcessor),
        ("KSSProcessor", KSSProcessor),
        ("BakerProcessor", BakerProcessor),
        ("LibriTTSProcessor", LibriTTSProcessor),
        ("ThorstenProcessor", ThorstenProcessor),
        ("LJSpeechUltimateProcessor", LJSpeechUltimateProcessor),
        ("SynpaflexProcessor", SynpaflexProcessor),
        ("JSUTProcessor", JSUTProcessor),
        ("IndonesianIPAProcessor", IndonesianIPAProcessor),
        ("EnglishIPAProcessor", EnglishIPAProcessor),
+       ("JavaneseCharacterProcessor", JavaneseCharacterProcessor),
    ]
)
```