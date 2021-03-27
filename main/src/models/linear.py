#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'shining.shi@alibaba-inc.com'


import torch
import torch.nn as nn

from .lan_model import LAN_MODELS


class LanModelGraph(nn.Module):
    """docstring for LanModelGraph"""
    def __init__(self, config):
        super(LanModelGraph, self).__init__()
        self.config = config
        self.lan_layer = LAN_MODELS[config.lan_model].from_pretrained(config.lan_model)
        # freeze the Pre-trained Language Model
        if config.freeze_lan_model:
            for param in self.lan_layer.base_model.parameters():
                param.requires_grad = False
        if config.use_pos:
            self.embedding = nn.Embedding(
                num_embeddings=self.config.tag_size
                , embedding_dim=self.config.embedding_size
                )
            if config.use_pos == 'add':
                if self.config.embedding_size != self.config.lan_hidden_size:
                    self.trans_layer = nn.Linear(
                        in_features=self.config.embedding_size
                        , out_features=self.config.lan_hidden_size)
                linear_in_features = config.lan_hidden_size
            elif config.use_pos == 'concat':
                linear_in_features = config.lan_hidden_size + config.embedding_size
            elif config.use_pos == 'xfmr':
                # language model hidden size + 512
                fusion_in_features = config.lan_hidden_size + config.embedding_size
                self.fusion_layer = nn.TransformerEncoderLayer(
                    d_model=fusion_in_features
                    , nhead=config.xfmr_num_attention_heads
                    , dim_feedforward=config.xfmr_intermediate_size
                    , dropout=config.xfmr_hidden_dropout_prob
                    , activation= 'gelu'
                    )
                # linear_in_features = config.lan_hidden_size * 2
                linear_in_features = config.lan_hidden_size + config.embedding_size
            self.out_layer = nn.Linear(
                in_features=linear_in_features
                , out_features=len(config.label2idx_dict))
        else:
            self.out_layer = nn.Linear(config.lan_hidden_size, len(config.label2idx_dict))

    def forward(self, xs, x_masks, y_tags=None):
        # xs: batch_size, max_seq_len
        # x_masks: batch_size, max_seq_len
        # batch_size, max_seq_len, lan_hidden_size
        xs = self.lan_layer(xs, attention_mask=x_masks)[0]
        if self.config.use_pos:
            # batch_size, max_seq_len, embedding_size
            y_tags = self.embedding(y_tags)
            if self.config.use_pos == 'add':
                if self.config.embedding_size != self.config.lan_hidden_size:
                    # batch_size, max_seq_len, lan_hidden_size
                    y_tags = self.trans_layer(y_tags)
                xs += y_tags
            elif self.config.use_pos == 'concat':
                # batch_size, max_seq_len, lan_hidden_size + embedding_size
                xs = torch.cat((xs, y_tags), dim=-1)
            elif self.config.use_pos == 'xfmr':
                # batch_size, max_seq_len, lan_hidden_size + embedding_size
                xs = torch.cat((xs, y_tags), dim=-1)
                xs = self.fusion_layer(xs)
        # batch_size, max_seq_len, pun_size
        ys = self.out_layer(xs)
        return ys