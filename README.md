# POS-Tagger-for-Punctuation-Restoration
This repository is for the paper [Incorporating External POS Tagger for Punctuation Restoration](https://arxiv.org/abs/2106.06731) in proceedings of the [*2021 Conference of the International Speech Communication Association (INTERSPEECH)*](https://www.interspeech2021.org/).

## Methods
+ Language Model -> Linear Layer
+ Language Model -> POS Fusion Layer -> Linear Layer

## Language Models
+ bert-base-uncased
+ bert-large-uncased
+ albert-base-v2
+ albert-large-v2
+ roberta-base
+ roberta-large
+ xlm-roberta-base
+ xlm-roberta-large
+ funnel-transformer/large
+ funnel-transformer/xlarge

## Directory
+ **main** - Source Code
+ **main/train.py** - Training Process
+ **main/config.py** - Training Configurations
+ **main/main.ipynb** - Inference Demo
+ **main/res/data/raw** - IWSLT Source Data
+ **main/src/models** - Models
+ **main/src/utils** - Helper Function
+ **eva.xlsx** - Evaluation Results
```
POS-Tagger-for-Punctuation-Restoration/
├── README.md
├── eva.xlsx
├── main
│   ├── config.py
│   ├── res
│   │   ├── data
│   │   │   ├── raw
│   │   │   │   ├── dev2012.txt
│   │   │   │   ├── test2011.txt
│   │   │   │   ├── test2011asr.txt
│   │   │   │   └── train2012.txt
│   │   └── settings.py
│   ├── src
│   │   ├── models
│   │   │   ├── lan_model.py
│   │   │   └── linear.py
│   │   └── utils
│   │       ├── eva.py
│   │       ├── load.py
│   │       ├── pipeline.py
│   │       └── save.py
│   ├── train.py
│   └── main.ipynb
```

## Dependencies
+ python >= 3.8.5
+ jupyterlab >= 3.1.4
+ flair >= 0.8.
+ scikit_learn >= 0.24.1
+ torch >= 1.7.1
+ tqdm >= 4.57.0
+ transformers >= 4.3.2
+ ipywidgets >= 7.6.3

## Setup
Please ensure required packages are already installed. A virtual environment is recommended.
```
$ cd POS-Tagger-for-Punctuation-Restoration
$ cd main
$ pip install pip --upgrade
$ pip install -r requirements.txt
Looking in indexes: http://mirrors.cloud.aliyuncs.com/pypi/simple/
Collecting flair==0.8
  Downloading http://mirrors.cloud.aliyuncs.com/pypi/packages/16/a9/02ab3594958a89c5477f2820a19158187e095763ab6d5d6c0aa5a896087c/flair-0.8-py3-none-any.whl (277 kB)
     |████████████████████████████████| 277 kB 23.4 MB/s
...
...
...
Installing collected packages: urllib3, numpy, idna, chardet, zipp, tqdm, smart-open, six, scipy, requests, regex, PySocks, pyparsing, joblib, decorator, click, wrapt, wcwidth, typing-extensions, tokenizers, threadpoolctl, sentencepiece, sacremoses, python-dateutil, pillow, packaging, overrides, networkx, kiwisolver, importlib-metadata, gensim, future, filelock, cycler, cloudpickle, transformers, torch, tabulate, sqlitedict, segtok, scikit-learn, mpld3, matplotlib, lxml, langdetect, konoha, janome, hyperopt, huggingface-hub, gdown, ftfy, deprecated, bpemb, flair
Successfully installed PySocks-1.7.1 bpemb-0.3.2 chardet-4.0.0 click-7.1.2 cloudpickle-1.6.0 cycler-0.10.0 decorator-4.4.2 deprecated-1.2.12 filelock-3.0.12 flair-0.8 ftfy-5.9 future-0.18.2 gdown-3.12.2 gensim-3.8.3 huggingface-hub-0.0.7 hyperopt-0.2.5 idna-2.10 importlib-metadata-3.7.3 janome-0.4.1 joblib-1.0.1 kiwisolver-1.3.1 konoha-4.6.4 langdetect-1.0.8 lxml-4.6.3 matplotlib-3.4.0 mpld3-0.3 networkx-2.5 numpy-1.19.5 overrides-3.1.0 packaging-20.9 pillow-8.1.2 pyparsing-2.4.7 python-dateutil-2.8.1 regex-2021.3.17 requests-2.25.1 sacremoses-0.0.43 scikit-learn-0.24.1 scipy-1.6.2 segtok-1.5.10 sentencepiece-0.1.95 six-1.15.0 smart-open-4.2.0 sqlitedict-1.7.0 tabulate-0.8.9 threadpoolctl-2.1.0 tokenizers-0.10.1 torch-1.7.1 tqdm-4.57.0 transformers-4.3.2 typing-extensions-3.7.4.3 urllib3-1.26.4 wcwidth-0.2.5 wrapt-1.12.1 zipp-3.4.1
```

## Run
Before training, please take a look at the **config.py** to ensure training configurations.
```
$ cd main
$ vim config.py
$ python train.py
```

## Output
If everything goes well, you should see a similar progressing shown as below.
```
Initialize...
2021-03-28 00:58:27,603 loading file /root/.flair/models/upos-english-fast/b631371788604e95f27b6567fe7220e4a7e8d03201f3d862e6204dbf90f9f164.0afb95b43b32509bf4fcc3687f7c64157d8880d08f813124c1bd371c3d8ee3f7

*Configuration*
model: linear
language model: bert-base-uncased
freeze language model: False
involve pos knowledge: None
pre-trained pos embedding: None
sequence boundary sampling: random
mask loss: False
trainable parameters: 109,485,316
model:
lan_layer.embeddings.position_ids       torch.Size([1, 512])
lan_layer.embeddings.word_embeddings.weight     torch.Size([30522, 768])
lan_layer.embeddings.position_embeddings.weight torch.Size([512, 768])
lan_layer.embeddings.token_type_embeddings.weight       torch.Size([2, 768])
lan_layer.embeddings.LayerNorm.weight   torch.Size([768])
lan_layer.embeddings.LayerNorm.bias     torch.Size([768])
lan_layer.encoder.layer.0.attention.self.query.weight   torch.Size([768, 768])
lan_layer.encoder.layer.0.attention.self.query.bias     torch.Size([768])
lan_layer.encoder.layer.0.attention.self.key.weight     torch.Size([768, 768])
lan_layer.encoder.layer.0.attention.self.key.bias       torch.Size([768])
lan_layer.encoder.layer.0.attention.self.value.weight   torch.Size([768, 768])
lan_layer.encoder.layer.0.attention.self.value.bias     torch.Size([768])
lan_layer.encoder.layer.0.attention.output.dense.weight torch.Size([768, 768])
lan_layer.encoder.layer.0.attention.output.dense.bias   torch.Size([768])
lan_layer.encoder.layer.0.attention.output.LayerNorm.weight     torch.Size([768])
lan_layer.encoder.layer.0.attention.output.LayerNorm.bias       torch.Size([768])
lan_layer.encoder.layer.0.intermediate.dense.weight     torch.Size([3072, 768])
lan_layer.encoder.layer.0.intermediate.dense.bias       torch.Size([3072])
lan_layer.encoder.layer.0.output.dense.weight   torch.Size([768, 3072])
lan_layer.encoder.layer.0.output.dense.bias     torch.Size([768])
lan_layer.encoder.layer.0.output.LayerNorm.weight       torch.Size([768])
lan_layer.encoder.layer.0.output.LayerNorm.bias torch.Size([768])
...
...
...
lan_layer.encoder.layer.11.attention.self.query.bias    torch.Size([768])
lan_layer.encoder.layer.11.attention.self.key.weight    torch.Size([768, 768])
lan_layer.encoder.layer.11.attention.self.key.bias      torch.Size([768])
lan_layer.encoder.layer.11.attention.self.value.weight  torch.Size([768, 768])
lan_layer.encoder.layer.11.attention.self.value.bias    torch.Size([768])
lan_layer.encoder.layer.11.attention.output.dense.weight        torch.Size([768, 768])
lan_layer.encoder.layer.11.attention.output.dense.bias  torch.Size([768])
lan_layer.encoder.layer.11.attention.output.LayerNorm.weight    torch.Size([768])
lan_layer.encoder.layer.11.attention.output.LayerNorm.bias      torch.Size([768])
lan_layer.encoder.layer.11.intermediate.dense.weight    torch.Size([3072, 768])
lan_layer.encoder.layer.11.intermediate.dense.bias      torch.Size([3072])
lan_layer.encoder.layer.11.output.dense.weight  torch.Size([768, 3072])
lan_layer.encoder.layer.11.output.dense.bias    torch.Size([768])
lan_layer.encoder.layer.11.output.LayerNorm.weight      torch.Size([768])
lan_layer.encoder.layer.11.output.LayerNorm.bias        torch.Size([768])
lan_layer.pooler.dense.weight   torch.Size([768, 768])
lan_layer.pooler.dense.bias     torch.Size([768])
out_layer.weight        torch.Size([4, 768])
out_layer.bias  torch.Size([4])
device: cuda
train size: 4475
val size: 635
ref test size: 27
asr test size: 27
batch size: 8
train batch: 559
val batch: 80
ref test batch: 4
asr test batch: 4
valid win size: 8
if load check point: False

Training...
Loss:0.9846:   1%|█▊                                                                                                                                                                                                    | 5/559 [00:02<03:30,  2.63it/s]
```

## Demo
Please find the inference demo in ***main.ipynb***, where we show how to employ an example checkpoint to restore punctuations for test samples.

## Note
1. It takes time to prepare POS tags for the first time running.
2. There will be a warning regarding hugging face tokenizer with parallel processing. Just ignore it or rerun the ***train.py*** with the same ***config.py***.

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTex
```
@inproceedings{shi21_interspeech,
  author={Ning Shi and Wei Wang and Boxin Wang and Jinfeng Li and Xiangyu Liu and Zhouhan Lin},
  title={{Incorporating External POS Tagger for Punctuation Restoration}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={1987--1991},
  doi={10.21437/Interspeech.2021-1708}
}
```
