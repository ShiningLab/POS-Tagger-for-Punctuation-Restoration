#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'shining.shi@alibaba-inc.com'


from sklearn.metrics import precision_score, recall_score, f1_score


class Evaluater(object):
    """docstring for Evaluater"""
    def __init__(self, ys, y_masks, ys_, config):
        super(Evaluater, self).__init__()
        self.ys = [l for y in ys for l in y]
        self.ys_ = [l for y_ in ys_ for l in y_]
        self.y_masks = [m for y_mask in y_masks for m in y_mask]
        self.ys = self.compress(self.ys, self.y_masks)
        self.ys_ = self.compress(self.ys_, self.y_masks)
        self.labels = list(config.label2idx_dict.keys())[1:]

        self.comma_precision, self.period_precision, self.question_precision = self.get_precision()
        self.comma_recall, self.period_recall, self.question_recall = self.get_recall()
        self.comma_f1, self.period_f1, self.question_f1 = self.get_f1()

        self.avg_precision = precision_score(self.ys, self.ys_, average='micro', zero_division=0., labels=self.labels)
        self.avg_recall = recall_score(self.ys, self.ys_, average='micro', zero_division=0., labels=self.labels)
        self.avg_f1 = 2 * (self.avg_precision * self.avg_recall) / (self.avg_precision + self.avg_recall)
        self.key_metric = self.avg_f1
        # generate an evaluation message
        comma_msg = 'COMMA P:{:.4f} R:{:.4f} F:{:.4f}'.format(self.comma_precision, self.comma_recall, self.comma_f1)
        period_msg = 'PERIOD P:{:.4f} R:{:.4f} F:{:.4f}'.format(self.period_precision, self.period_recall, self.period_f1)
        question_msg = 'QUESTION P:{:.4f} R:{:.4f} F:{:.4f}'.format(self.question_precision, self.question_recall, self.question_f1)
        avg_msg = 'Overall P:{:.4f} R:{:.4f} F:{:.4f}'.format(self.avg_precision, self.avg_recall, self.avg_f1)
        # key metric for early stopping
        self.eva_msg = '\n' + comma_msg + '\n' + period_msg + '\n' + question_msg + '\n' + avg_msg

    def compress(self, data, mask):
        return [d for d, s in zip(data, mask) if s]

    def get_precision(self):
        return precision_score(self.ys, self.ys_, average=None, zero_division=0., labels=self.labels)

    def get_recall(self):
        return recall_score(self.ys, self.ys_, average=None, zero_division=0., labels=self.labels)

    def get_f1(self):
        return f1_score(self.ys, self.ys_, average=None, zero_division=0., labels=self.labels)