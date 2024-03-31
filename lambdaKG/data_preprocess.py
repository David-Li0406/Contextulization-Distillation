import argparse
import random
import sys

import numpy as np
import nlp2
from datasets import load_dataset
from itertools import groupby

from transformers import AutoTokenizer


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="train data", default='dataset/FB15k-237N/knowledge_context.txt')
    parser.add_argument("--tokenizer_config", type=str, default="facebook/bart-base")
    parser.add_argument("--output_name", type=str, default="bart_pretrain_data")
    parser.add_argument("--mask_prob", default=0.15, type=float, help="mask lm probability")
    parser.add_argument("--worker", default=10, type=int, help="multi processing worker")
    parser.add_argument("--poisson_lam", default=3, type=int, help="poisson lambda")
    input_arg, others_arg = parser.parse_known_args(args)
    input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
    others_arg = {
        k.replace("--", ""): v for k, v in zip(others_arg[:-1:2], others_arg[1::2])
    }
    return input_arg, others_arg


def main(arg=None):
    input_arg, others_arg = (
        parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    )
    tokenizer = AutoTokenizer.from_pretrained(input_arg['tokenizer_config'])
    MASKTOK = tokenizer.mask_token
    dataset = load_dataset("text", data_files={'data': input_arg['data']})

    def noisy(examples):
        try:
            _, _, _, kc = examples['text'].split(' |**| ')
            target_sent = kc
            sent = kc.split(".")
            random.shuffle(sent)
            input_sent = ".".join(sent)

            sent = input_sent.split("。")
            random.shuffle(sent)
            input_sent = "。".join(sent)

            input_sent = nlp2.split_sentence_to_array(input_sent)
            for ind, word in enumerate(input_sent):
                prob = random.random()
                if prob <= input_arg['mask_prob'] and len(word) > 0:
                    length = np.random.poisson(lam=input_arg['poisson_lam'])
                    if length == 0:
                        input_sent.insert(ind, MASKTOK)
                    else:
                        input_sent[ind:ind + length] = [MASKTOK] * len(input_sent[ind:ind + length])
            input_sent = [k for k, _ in groupby(input_sent)]  # merge_repeat
            input_sent = nlp2.join_words_to_sentence(input_sent)
            examples['text'] +=  ' |**| ' + input_sent
        except:
            pass
        return examples

    dataset = dataset.map(noisy, num_proc=input_arg['worker'])
    dataset.save_to_disk(input_arg['output_name'] + "_cache")
    with open(input_arg['data'], 'w') as f:
        for item in dataset['data']:
            f.write(item['text'] + '\n')


if __name__ == "__main__":
    main()