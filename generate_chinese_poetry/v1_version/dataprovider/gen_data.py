#!/usr/bin/env python
#coding=utf-8

import os
import sys
import random

from paddle.trainer.PyDataProvider2 import *


def on_init1(settings, **kwargs):
    settings.word_dict = {}
    with open(kwargs["word_dict"], "r") as fdict:
        for idx, line in enumerate(fdict):
            settings.word_dict[line.strip().decode(
                "utf8", errors="ignore").split("\t")[0]] = idx
    settings.start_id = settings.word_dict["<s>"]
    settings.end_id = settings.word_dict["<e>"]
    settings.unk_id = settings.word_dict["<unk>"]

    settings.logger.info("dict len = %d" % (len(settings.word_dict)))

    settings.input_types = [
        integer_value_sequence(len(settings.word_dict)),
        integer_value_sequence(len(settings.word_dict)),
        integer_value_sequence(len(settings.word_dict)),
    ]


@provider(
    use_seq=True,
    init_hook=on_init1,
    should_shuffle=True,
    cache=CacheType.CACHE_PASS_IN_MEM,
    pool_size=51200,
    min_pool_size=51200)
def processData(settings, file_name):
    file_list = open(file_name, 'r').readlines()
    random.shuffle(file_list)

    for file_line in file_list:
        with open(file_line.strip()) as fdata:
            for idx, line in enumerate(fdata):
                line_split = line.strip().decode(
                    "utf8", errors="ignore").split("\t")
                if len(line_split) < 3: continue
                text = line_split[2].split(".")
                word_ids = []
                for sen in text:
                    word_ids.append([
                        settings.word_dict.get(char, settings.unk_id)
                        for char in sen if len(sen)
                    ])

                if len(word_ids) < 2: continue
                for i in range(len(word_ids) - 1):
                    yield (
                        [settings.start_id] + word_ids[i] + [settings.end_id],
                        [settings.start_id] + word_ids[i + 1],
                        word_ids[i + 1] + [settings.end_id])
