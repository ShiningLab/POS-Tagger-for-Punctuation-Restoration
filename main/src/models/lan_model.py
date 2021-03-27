#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'shining.shi@alibaba-inc.com'


from transformers import *


LAN_MODELS = {
    'bert-base-uncased': BertModel
    , 'bert-large-uncased': BertModel
    , 'albert-base-v2': AlbertModel
    , 'albert-large-v2': AlbertModel
    , 'roberta-base': RobertaModel
	, 'roberta-large': RobertaModel
    , 'xlm-roberta-base': XLMRobertaModel
    , 'xlm-roberta-large': XLMRobertaModel
    , 'funnel-transformer/large': FunnelModel
    , 'funnel-transformer/xlarge': FunnelModel
}