# Adding British English Support to gruut

Everything here can be followed along in Google Colab!

<p align="center">
    <a href="https://colab.research.google.com/drive/1UK_-nUZnRyRZSz9Hz3NUdf-J_BuI5Rpt?usp=sharing">
        <img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
</p>

[gruut](https://github.com/rhasspy/gruut/) is a very awesome grapheme-to-phoneme converter for English, and it also includes additional goodies such as a text processor that further handles numbers, abbreviations, and other intricate details. This is why gruut was an excellent choice when we wanted to implement a custom IPA processor for English.

One missing feature, however, is the ability to phonemize British English. As of the time of writing, gruut only supports American English. [Several users](https://github.com/rhasspy/gruut/issues/29), myself included, would love to see it implemented with a similar API that we know and love. But seeing how the project has not been updated since June 2022, it looks like our only choice is to implement it ourselves -- which is what we're going to do today. Follow along as we integrate [espeak-ng](https://github.com/espeak-ng/espeak-ng)'s British English g2p into gruut. A fork that implements these changes can be found [here](https://github.com/w11wo/gruut).

!!! info 

    A lot of the steps here are based on the [official guide](https://rhasspy.github.io/gruut/#adding-a-new-language) which also seems to be outdated.

To start, we need to install `espeak-ng`, `gruut`, `phonetisaurus`, and clone the gruut repository as well. We can do these via the command line.

## Installation

```python
!apt-get -q install espeak-ng
```

    Reading package lists...
    Building dependency tree...
    Reading state information...
    The following package was automatically installed and is no longer required:
      libnvidia-common-460
    Use 'apt autoremove' to remove it.
    The following additional packages will be installed:
      espeak-ng-data libespeak-ng1 libpcaudio0 libsonic0
    The following NEW packages will be installed:
      espeak-ng espeak-ng-data libespeak-ng1 libpcaudio0 libsonic0
    0 upgraded, 5 newly installed, 0 to remove and 20 not upgraded.
    Need to get 3,957 kB of archives.
    After this operation, 10.9 MB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu bionic/main amd64 libpcaudio0 amd64 1.0-1 [6,536 B]
    Get:2 http://archive.ubuntu.com/ubuntu bionic/main amd64 libsonic0 amd64 0.2.0-6 [13.4 kB]
    Get:3 http://archive.ubuntu.com/ubuntu bionic/main amd64 espeak-ng-data amd64 1.49.2+dfsg-1 [3,469 kB]
    Get:4 http://archive.ubuntu.com/ubuntu bionic/main amd64 libespeak-ng1 amd64 1.49.2+dfsg-1 [187 kB]
    Get:5 http://archive.ubuntu.com/ubuntu bionic/universe amd64 espeak-ng amd64 1.49.2+dfsg-1 [282 kB]
    Fetched 3,957 kB in 1s (3,165 kB/s)
    Selecting previously unselected package libpcaudio0.
    (Reading database ... 124013 files and directories currently installed.)
    Preparing to unpack .../libpcaudio0_1.0-1_amd64.deb ...
    Unpacking libpcaudio0 (1.0-1) ...
    Selecting previously unselected package libsonic0:amd64.
    Preparing to unpack .../libsonic0_0.2.0-6_amd64.deb ...
    Unpacking libsonic0:amd64 (0.2.0-6) ...
    Selecting previously unselected package espeak-ng-data:amd64.
    Preparing to unpack .../espeak-ng-data_1.49.2+dfsg-1_amd64.deb ...
    Unpacking espeak-ng-data:amd64 (1.49.2+dfsg-1) ...
    Selecting previously unselected package libespeak-ng1:amd64.
    Preparing to unpack .../libespeak-ng1_1.49.2+dfsg-1_amd64.deb ...
    Unpacking libespeak-ng1:amd64 (1.49.2+dfsg-1) ...
    Selecting previously unselected package espeak-ng.
    Preparing to unpack .../espeak-ng_1.49.2+dfsg-1_amd64.deb ...
    Unpacking espeak-ng (1.49.2+dfsg-1) ...
    Setting up libsonic0:amd64 (0.2.0-6) ...
    Setting up libpcaudio0 (1.0-1) ...
    Setting up espeak-ng-data:amd64 (1.49.2+dfsg-1) ...
    Setting up libespeak-ng1:amd64 (1.49.2+dfsg-1) ...
    Setting up espeak-ng (1.49.2+dfsg-1) ...
    Processing triggers for man-db (2.8.3-2ubuntu0.1) ...
    Processing triggers for libc-bin (2.27-3ubuntu1.6) ...

```python
!pip install -q gruut phonetisaurus
```

    |████████████████████████████████| 74 kB 1.8 MB/s 
    |████████████████████████████████| 12.1 MB 11.2 MB/s 
    |████████████████████████████████| 292 kB 63.3 MB/s 
    |████████████████████████████████| 101 kB 9.6 MB/s 
    |████████████████████████████████| 15.2 MB 15.4 MB/s 
    |████████████████████████████████| 125 kB 54.2 MB/s 
    |████████████████████████████████| 1.0 MB 60.4 MB/s 
    Building wheel for gruut (setup.py) ... done
    Building wheel for gruut-ipa (setup.py) ... done
    Building wheel for gruut-lang-en (setup.py) ... done
    Building wheel for docopt (setup.py) ... done

```python
!git clone -q https://github.com/rhasspy/gruut.git
```

## Corpus

The first thing we have to decide is the corpus. What words do we want to include in the lexicon? For simplicity's sake, I chose to grab all words in the current American English lexicon, and hence make a British "counterpart" of it. Since all words are stored in a SQLite database, we just need to connect into the database via the `sqlite3` library provided in Python.

```python
import sqlite3
con = sqlite3.connect("gruut/gruut-lang-en/gruut_lang_en/lexicon.db")
cur = con.cursor()
```

After making a connection into the database and creating a cursor, we can then query all words in the database.

```python
words = sorted(list(set([row[0] for row in cur.execute("SELECT word FROM word_phonemes")])))
words[:5]
```

    ['a', "a's", 'a.m.', 'aaberg', 'aachen']

```python
len(words)
```

    124392

Then, we just need to write each word into a plain textfile, each separated by a newline break.

```python
with open("words.txt", "w") as f:
    for word in words:
        f.write(f"{word}\n")
```

A quick check of the results:

```python
!head words.txt
```

    a
    a's
    a.m.
    aaberg
    aachen
    aachener
    aah
    aaker
    aaliyah
    aalseth


```python
!tail words.txt
```

    zynda
    zysk
    zyskowski
    zyuganov
    zyuganov's
    zywicki
    éabha
    órla
    órlagh
    šerbedžija

## Building a Lexicon

Now that we have a list of words (corpus), we can simply use the pre-provided `espeak_word.sh` script that conveniently reads the list of words in a plain textfile and writes out a lexicon. The lexicon will contain the word, and its corresponding IPA phonemes. In our case, these phonemes are generated by `espeak-ng`, specifically the `en-gb` voice.

Notice that we also have to pass the argument `pos`, such that the lexicon will also contain the word's part-of-speech (POS) tag. This will help identify if a homograph is a noun, or a verb, or is of other POS tag. We will see later how this would be integrated into the resultant lexicon database.

```python
!bash gruut/bin/espeak_word.sh en-gb pos < words.txt > lexicon.txt
```

After about 3-4 hours of running, the lexicon file should be successfully generated.

```python
!head lexicon.txt
```

    a _ ˈeɪ
    a's _ ˈeɪ z
    a.m. _ ˌeɪ ˈɛ m
    aaberg _ ˈɑː b ɜː ɡ
    aachen _ ˈɑː tʃ ə n
    aachener _ ˈɑː tʃ ə n ˌə
    aah _ ˈɑː
    aaker _ ˈɑː k ə
    aaliyah _ ə l ˈiː  ə
    aalseth _ ˈɑː l s ə θ

```python
!tail lexicon.txt
```

    zynda _ z ˈɪ n d ə
    zysk _ z ˈɪ s k
    zyskowski _ z ɪ s k ˈaʊ s k ɪ
    zyuganov _ z j ˈuː ɡ ɐ n ˌɒ v
    zyuganov's _ z j ˈuː ɡ ɐ n ˌɒ v z
    zywicki _ z aɪ w ˈɪ k i
    éabha _ ɪ  ˈa b h ə
    órla _ ˈɔː l ə
    órlagh _ ˈɔː l ɑː ɡ
    šerbedžija _ ʃ ˈɜː b ɪ d ʒ ˌɪ dʒ ə

```python
!wc -l lexicon.txt
```

    125059 lexicon.txt

As expected, there are more entries in the lexicon than the original word list (125,059 from 124,392) due to homographs. A word could appear twice or more, if it's a homograph.

### Lexicon Database

We then convert the resultant lexicon text file to a SQLite database that gruut will read from later. Again, the script was conveniently provided by the authors of gruut. We simply have to run the Python utility script and pass the additional `--role` argument as we're also working with POS tags.

```python
!python3 -m gruut.lexicon2db --role --casing lower --lexicon lexicon.txt --database lexicon.db
```

## G2P Model

The last component that we have to train is a grapheme-to-phoneme (g2p) model. Here, we'll be using [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus) to generate a grapheme-phoneme alignment corpus.

```python
!phonetisaurus train --corpus g2p.corpus --model g2p.fst lexicon.txt
```

    INFO:phonetisaurus-train:2022-12-12 06:57:23:  Checking command configuration...
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  Directory does not exist.  Trying to create.
    INFO:phonetisaurus-train:2022-12-12 06:57:23:  Checking lexicon for reserved characters: '}', '|', '_'...
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  arpa_path:  train/model.o8.arpa
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  corpus_path:  train/model.corpus
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  dir_prefix:  train
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  grow:  False
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  lexicon_file:  /tmp/tmp9apulg06.txt
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  logger:  <Logger phonetisaurus-train (DEBUG)>
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  makeJointNgramCommand:  <bound method G2PModelTrainer._mitlm of <__main__.G2PModelTrainer object at 0x7ff2e63a4610>>
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  model_path:  train/model.fst
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  model_prefix:  model
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  ngram_order:  8
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  seq1_del:  False
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  seq1_max:  2
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  seq2_del:  True
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  seq2_max:  2
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  verbose:  True
    DEBUG:phonetisaurus-train:2022-12-12 06:57:23:  phonetisaurus-align --input=/tmp/tmp9apulg06.txt --ofile=train/model.corpus --seq1_del=false --seq2_del=true --seq1_max=2 --seq2_max=2 --grow=false
    INFO:phonetisaurus-train:2022-12-12 06:57:23:  Aligning lexicon...
    GitRevision: package
    Loading input file: /tmp/tmp9apulg06.txt
    Alignment failed: i t
    Alignment failed: m o h a m e d
    Alignment failed: o n e
    Alignment failed: s t
    Alignment failed: s t
    Alignment failed: t o
    Alignment failed: v i
    Alignment failed: v i
    Alignment failed: v i
    Alignment failed: w h e r e v e r
    Alignment failed: x i
    Alignment failed: x i
    Alignment failed: x i
    Starting EM...
    Finished first iter...
    Iteration: 1 Change: 2.5657
    Iteration: 2 Change: 0.00875473
    Iteration: 3 Change: 0.012403
    Iteration: 4 Change: 0.00788784
    Iteration: 5 Change: 0.00399113
    Iteration: 6 Change: 0.00240803
    Iteration: 7 Change: 0.00218391
    Iteration: 8 Change: 0.00197029
    Iteration: 9 Change: 0.00191975
    Iteration: 10 Change: 0.00169945
    Iteration: 11 Change: 0.00149059
    Last iteration: 
    DEBUG:phonetisaurus-train:2022-12-12 06:57:24:  estimate-ngram -o 8 -t train/model.corpus -wl train/model.o8.arpa
    INFO:phonetisaurus-train:2022-12-12 06:57:24:  Training joint ngram model...
    0.001	Loading corpus train/model.corpus...
    0.020	Smoothing[1] = ModKN
    0.020	Smoothing[2] = ModKN
    0.020	Smoothing[3] = ModKN
    0.020	Smoothing[4] = ModKN
    0.020	Smoothing[5] = ModKN
    0.020	Smoothing[6] = ModKN
    0.020	Smoothing[7] = ModKN
    0.020	Smoothing[8] = ModKN
    0.020	Set smoothing algorithms...
    0.020	Y 6.304348e-01
    0.020	Y 6.363636e-01
    0.020	Y 7.024504e-01
    0.020	Y 7.710983e-01
    0.020	Y 8.060942e-01
    0.020	Y 8.090737e-01
    0.020	Y 8.037634e-01
    0.021	Y 6.779026e-01
    0.021	Estimating full n-gram model...
    0.021	Saving LM to train/model.o8.arpa...
    DEBUG:phonetisaurus-train:2022-12-12 06:57:24:  phonetisaurus-arpa2wfst --lm=train/model.o8.arpa --ofile=train/model.fst
    INFO:phonetisaurus-train:2022-12-12 06:57:24:  Converting ARPA format joint n-gram model to WFST format...
    GitRevision: package
    Initializing...
    Converting...
    INFO:phonetisaurus-train:2022-12-12 06:57:24:  G2P training succeeded: train/model.fst

Using the alignment corpus, we can then train a g2p conditional random field (CRF) model using the script provided by gruut.

```python
!python3 -m gruut.g2p train --corpus g2p.corpus --output g2p/model.crf
```

    INFO:gruut.g2p:Training
    INFO:gruut.g2p:Training completed in 25.27762404100031 second(s)
    INFO:gruut.g2p:{'num': 49, 'scores': {}, 'loss': 5350.58727, 'feature_norm': 49.770636, 'error_norm': 0.506685, 'active_features': 46707, 'linesearch_trials': 1, 'linesearch_step': 1.0, 'time': 0.471}

Finally, we have to integrate the alignment corpus back into the lexicon database.

```python
!python3 -m gruut.corpus2db --corpus g2p.corpus --database lexicon.db
```

    Added 429 alignments to lexicon.db

And we're all set to integrate our new British English g2p model into gruut!

!!! note

    We did not go through the necessary steps to build a POS tagging model, simply because we can just use the existing American English POS tagging model.

## Integrating into gruut

The official guide is outdated on how we could add a new language into gruut. But after reading the source code, it seems that it's simply going to look up files in the `gruut/gruut-lang-{lang}` directory, where `{lang}` is the language that we want to support. Now, the issue is that there's already an English directory, which contains the necessary files for American English.

What we decided to do is to replace the lexicon database and the g2p model inside `gruut/gruut-lang-en/gruut-lang-en/espeak` folder, which can later be accessed by flagging `espeak=True` in the Python front-end. This is a rather hacky solution since we're assuming that anytime the end user asks for the espeak English model, we're going to return them the British English model -- regardless if they ask for `en-us`. But nonetheless, I think this would still work anyway.

A simple replacement of `gruut/gruut-lang-en/gruut_lang_en/espeak/lexicon.db` and `gruut/gruut-lang-en/gruut_lang_en/espeak/g2p/model.crf` with our newly generated files from above should do the job.

```
!cp g2p/model.crf gruut/gruut-lang-en/gruut_lang_en/espeak/g2p/
!cp lexicon.db gruut/gruut-lang-en/gruut_lang_en/espeak/
```

```diff
  gruut
  ├── ...
  ...
  ├── gruut-lang-en
  │   ├── LANGUAGE
  │   ├── README.md
  │   ├── gruut_lang_en
  │   │   ├── VERSION
  │   │   ├── __init__.py
  │   │   ├── espeak
  │   │   │   ├── g2p
+ │   │   │   │   └── model.crf
+ │   │   │   └── lexicon.db
  │   │   ├── g2p
  │   │   │   └── model.crf
  │   │   ├── lexicon.db
  │   │   └── pos
  │   │       └── model.crf
  │   └── setup.py
  ...
  └── ...
```

An example demonstrating this can be found in this [commit](https://github.com/w11wo/gruut/commit/a1201f3f0a73999f739c99c9f1f9735558eb684d).

## Test

To test, we need to re-install our local copy of gruut.

```python
!cd gruut && pip install . && pip install ./gruut-lang-en
```

At this point you might need to restart your Google Colab runtime to re-import the new gruut.

And finally, we can perform a conversion through the Python frontend like so:

```python
import gruut

text = "an accelerating runner"

for words in gruut.sentences(text, lang="en-gb", espeak=True):
    for word in words:
        print(word.phonemes)
```

    ['ˈa', 'n']
    ['ɐ', 'k', 's', 'ˈɛ', 'l', 'ə', 'ɹ', 'ˌeɪ', 't', 'ɪ', 'ŋ']
    ['ɹ', 'ˈʌ', 'n', 'ə']

Comparing that to American English

```python
for words in gruut.sentences(text, lang="en-us"): # don't set espeak=True!
    for word in words:
        print(word.phonemes)
```

    ['ə', 'n']
    ['æ', 'k', 's', 'ˈɛ', 'l', 'ɚ', 'ˌeɪ', 't', 'ɪ', 'ŋ']
    ['ɹ', 'ˈʌ', 'n', 'ɚ']

All done!