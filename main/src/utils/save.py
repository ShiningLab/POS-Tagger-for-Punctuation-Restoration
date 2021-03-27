#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'shining.shi@alibaba-inc.com'


import json


def save_txt(path, line_list):

      with open(path, 'w', encoding='utf-8') as f:
            for line in line_list:
                  f.write(line + '\n')
      f.close()

def save_json(path:str, content):
    with open(path, 'w') as f:
        json.dump(content, f, ensure_ascii=False)