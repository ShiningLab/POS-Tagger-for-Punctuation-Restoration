#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'shining.shi@alibaba-inc.com'


import os

from transformers import AutoConfig

from res.settings import *


class Config(object):
    """docstring for Config"""
    def __init__(self):
        super(Config, self).__init__()
        # Basic Settings
        # None, bert-base-uncased, bert-large-uncased
        # albert-base-v2, albert-large-v2
        # roberta-base, roberta-large
        # xlm-roberta-base, xlm-roberta-large
        # funnel-transformer/large, funnel-transformer/xlarge
        self.lan_model = 'funnel-transformer/xlarge'
        self.lan_model_config = AutoConfig.from_pretrained(self.lan_model) if self.lan_model else None
        self.pos = 'flair/upos-english-fast'
        self.X_TAG = 'X'
        # None, concat, add, xfmr
        self.use_pos = 'xfmr'
        # None, pretrained
        self.pretrained_pos_embed = 'pretrained' if self.use_pos else None
        # None, random
        self.seq_sampling = 'random'
        # loss
        self.mask_loss = False
        # linear
        self.model_name = 'linear'
        self.load_check_point = False
        self.freeze_lan_model = False
        # the same as 
        # https://github.com/xashru/punctuation-restoration/blob/master/src/argparser.py
        self.random_seed = 1
        self.BOS_TOKEN = SPECIAL_TOKENS_DICT[self.lan_model]['bos_token']
        self.EOS_TOKEN = SPECIAL_TOKENS_DICT[self.lan_model]['eos_token']
        self.UNK_TOKEN = SPECIAL_TOKENS_DICT[self.lan_model]['unk_token']
        self.PAD_TOKEN = SPECIAL_TOKENS_DICT[self.lan_model]['pad_token']
        self.NORMAL_TOKEN = 'O'
        # self.comma_token = '<COMMA>'
        self.PERIOD_TOKEN = 'PERIOD'
        self.QUESTION_TOKEN = 'QUESTION'
        # 'O' -> No punctuation
        self.label2idx_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3}
        self.idx2label_dict = {v:k for k, v in self.label2idx_dict.items()}
        # I/O
        self.TRAIN_FILE = 'train2012'
        self.VALID_FILE = 'dev2012'
        self.REF_TEST_FILE = 'test2011'
        self.ASR_TEST_FILE = 'test2011asr'
        self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data', str(self.lan_model).replace('/', '-'))
        if not os.path.exists(self.DATA_PATH): os.makedirs(self.DATA_PATH)
        self.DATA_JSON = os.path.join(self.DATA_PATH, '{}.json')
        self.RAW_PATH = os.path.join(self.RESOURCE_PATH, 'data', 'raw')
        self.RAW_TXT = os.path.join(self.RAW_PATH, '{}.txt')
        self.SAVE_PATH = os.path.join(
            self.RESOURCE_PATH, 'check_points', str(self.lan_model).replace('/', '-'), str(self.use_pos), str(self.pretrained_pos_embed), str(self.seq_sampling))
        if not os.path.exists(self.SAVE_PATH): os.makedirs(self.SAVE_PATH)
        self.SAVE_POINT = os.path.join(self.SAVE_PATH, '{}.pt'.format(self.model_name))
        # path to save test log
        self.LOG_PATH = os.path.join(
            self.RESOURCE_PATH, 'log', str(self.lan_model).replace('/', '-'), str(self.use_pos), str(self.pretrained_pos_embed), str(self.seq_sampling))
        if not os.path.exists(self.LOG_PATH): os.makedirs(self.LOG_PATH)
        self.LOG_POINT = os.path.join(self.LOG_PATH,  '{}.txt')
        # path to save test output
        self.RESULT_PATH = os.path.join(
            self.RESOURCE_PATH, 'result', str(self.lan_model).replace('/', '-'), str(self.use_pos), str(self.pretrained_pos_embed), str(self.seq_sampling))
        if not os.path.exists(self.RESULT_PATH): os.makedirs(self.RESULT_PATH)
        self.RESULT_POINT = os.path.join(self.RESULT_PATH, '{}.txt')
        # data preprocessing
        self.batch_size = 8
        self.shuffle = True
        # set 0 for data generation in the main process
        self.num_workers = 4
        self.pin_memory = True
        self.drop_last = True
        self.max_seq_len = 256
        # language model
        self.lan_hidden_size = self.lan_model_config.hidden_size if self.lan_model else None
        # model
        # 5e-6 for large, 1e-5 for base
        self.lan_learning_rate = 5e-6
        self.learning_rate = 5e-6
        self.clipping_threshold = 5.
        # embedding
        self.embedding_size = self.lan_hidden_size
        # encoder
        self.en_hidden_size = self.lan_hidden_size
        self.en_num_layers = 1
        # fusion
        self.xfmr_num_attention_heads = 16 if self.lan_hidden_size >= 1024 else 8
        self.xfmr_intermediate_size = 4096 if self.lan_hidden_size >= 1024 else 3072
        self.xfmr_hidden_dropout_prob = 0.1
        # validation
        self.valid_win_size = 8