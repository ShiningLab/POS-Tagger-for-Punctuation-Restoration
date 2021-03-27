#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'shining.shi@alibaba-inc.com'

SPECIAL_TOKENS_DICT = {}
SPECIAL_TOKENS_DICT['bert-base-uncased'] = {'bos_token': '[CLS]', 'eos_token': '[SEP]', 'unk_token': '[UNK]',  'pad_token': '[PAD]'}
SPECIAL_TOKENS_DICT['bert-large-uncased'] = {'bos_token': '[CLS]', 'eos_token': '[SEP]', 'unk_token': '[UNK]',  'pad_token': '[PAD]'}
SPECIAL_TOKENS_DICT['albert-base-v2'] = {'bos_token': '[CLS]', 'eos_token': '[SEP]', 'unk_token': '<unk>',  'pad_token': '<pad>'}
SPECIAL_TOKENS_DICT['albert-large-v2'] = {'bos_token': '[CLS]', 'eos_token': '[SEP]', 'unk_token': '<unk>',  'pad_token': '<pad>'}
SPECIAL_TOKENS_DICT['roberta-base'] = {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
SPECIAL_TOKENS_DICT['roberta-large'] = {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
SPECIAL_TOKENS_DICT['xlm-roberta-base'] = {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
SPECIAL_TOKENS_DICT['xlm-roberta-large'] = {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
SPECIAL_TOKENS_DICT['funnel-transformer/large'] = {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
SPECIAL_TOKENS_DICT['funnel-transformer/xlarge'] = {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
