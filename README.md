# POS-Tagger-for-Punctuation-Restoration
This repository is for the paper *Incorporating External POS Tagger for Punctuation Restoration*.

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
+ **main/config.py** - Training Configurations
+ **main/res/raw** - IWSLT Source Data
+ **main/src/models** - Model
+ **main/src/utils** - Helper Function
+ **reference** - Literature
+ **eva.xlsx** - Evaluation Results
```
POS-Tagger-for-Punctuation-Restoration/
├── README.md
├── eva.xlsx
├── main
│   ├── config.py
│   ├── res
│   │   ├── raw
│   │   │   ├── dev2012.txt
│   │   │   ├── test2011.txt
│   │   │   ├── test2011asr.txt
│   │   │   └── train2012.txt
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
│   └── train.py
└── reference
```

## Dependencies
+ python >= 3.8.5
+ flair >= 0.8.0.post1
+ scikit_learn >= 0.24.1
+ torch >= 1.8.1
+ tqdm >= 4.59.0
+ transformers >= 4.4.2


## Setup
Please ensure required packages are already installed. A virtual environment is recommended.
```
$ pip install pip --upgrade
$ pip install -r requirements.txt
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
## Note
1. It takes time to prepare POS tags for the first time running.
2. There will be a warning regarding hugging face tokenizer with parallel processing. Just ignore it or rerun the ***train.py*** with the same ***config.py***.

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTex
```
Submitted.
```
