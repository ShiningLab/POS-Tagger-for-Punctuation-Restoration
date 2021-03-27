#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'shining.shi@alibaba-inc.com'


import json


def load_json(path: str):
	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)