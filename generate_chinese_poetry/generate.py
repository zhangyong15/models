#!/usr/bin/env python
#coding=utf-8

import pdb
import os
import sys
import gzip
import logging
import numpy as np

import reader
import paddle.v2 as paddle
from paddle.v2.layer import parse_network
from network_conf import encoder_decoder_network

logger = logging.getLogger("paddle")
logger.setLevel(logging.WARNING)


def infer_a_batch(inferer, test_batch, beam_size, id_to_text):
    beam_result = inferer.infer(input=test_batch, field=["prob", "id"])
    gen_sen_idx = np.where(beam_result[1] == -1)[0]
    assert len(gen_sen_idx) == len(test_batch) * beam_size, ("%d vs. %d" % (
        len(gen_sen_idx), len(test_batch) * beam_size))

    start_pos, end_pos = 1, 0
    for i, sample in enumerate(test_batch):
        print(" ".join([
            id_to_text[w] for w in sample[0][1:-1]
        ]))  # skip the start and ending mark when print the source sentence
        for j in xrange(beam_size):
            end_pos = gen_sen_idx[i * beam_size + j]
            print("%.4f\t%s" % (beam_result[0][i][j], " ".join(
                id_to_text[w] for w in beam_result[1][start_pos:end_pos])))
            start_pos = end_pos + 2
        print("\n")


def generate(model_path, word_dict_path, test_data_path, batch_size, beam_size,
             use_gpu):
    assert os.path.exists(model_path), "trained model does not exist."

    paddle.init(use_gpu=use_gpu, trainer_count=1)
    with gzip.open(model_path, "r") as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    id_to_text = {}
    with open(word_dict_path, "r") as f:
        for i, line in enumerate(f):
            id_to_text[i] = line.strip().split("\t")[0]

    beam_gen = encoder_decoder_network(
        word_count=len(id_to_text),
        emb_dim=512,
        encoder_depth=3,
        encoder_hidden_dim=512,
        decoder_depth=3,
        decoder_hidden_dim=512,
        is_generating=True,
        beam_size=beam_size,
        max_length=10)

    inferer = paddle.inference.Inference(
        output_layer=beam_gen, parameters=parameters)

    test_batch = []
    for idx, item in enumerate(
            reader.gen_reader(test_data_path, word_dict_path)()):
        test_batch.append([item])
        if len(test_batch) == batch_size:
            infer_a_batch(inferer, test_batch, beam_size, id_to_text)
            test_batch = []

    if len(test_batch):
        infer_a_batch(inferer, test_batch, beam_size, id_to_text)
        test_batch = []


if __name__ == "__main__":
    generate(
        model_path="trained_models/models_first_round/pass_00067.tar.gz",
        word_dict_path="data/word_dict.txt",
        test_data_path="data/input.txt",
        batch_size=4,
        beam_size=5,
        use_gpu=False)
