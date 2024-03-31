from utils import *
import json
import random
random.seed(42)
import os
from tqdm import tqdm
import time
import Levenshtein
import re
import argparse
# from fuzzywuzzy import fuzzy
# from fuzzywuzzy import process

# dataset = 'FB15k-237N'
template = 'Given a triplet ({} | {} | {}), please generate a paragraph at least 100 words to introduce "{}" and "{}" respectively and reflect their relationship {}. "{}" and "{}" must concluded in the generated text.'
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='WN18RR')
    args = parser.parse_args()

    training_samples = []
    with open(os.path.join('data/processed', args.dataset, 'train2id.txt'), encoding='utf8') as f:
        for line in f.readlines()[1:]:
            head, tail, relation = line.split(' ')
            head, tail, relation = head.strip(), tail.strip(), relation.strip()
            training_samples.append((head, relation, tail))
    id2entity, id2relation = [], []
    with open(os.path.join('data/processed', args.dataset, 'entityid2name.txt'), encoding='utf8') as f:
        for line in f.readlines()[1:]:
            if args.dataset == 'WN18RR':
                entity = line.split('\t')[1].split(',')[0].strip()
            else:
                entity = line.split('\t')[1].strip()
            id2entity.append(entity)
    with open(os.path.join('data/processed', args.dataset, 'relationid2name.txt'), encoding='utf8') as f:
        for line in f.readlines()[1:]:
            id2relation.append(line.split('\t')[1].strip())

    results = []
    for sample in tqdm(training_samples):
        time.sleep(2)
        head, relation, tail = sample
        head_name, relation_name, tail_name = id2entity[int(head)], id2relation[int(relation)], id2entity[int(tail)]
        head_name='J.G. Ballard'
        tail_name='Shanghai'
        relation_name='/people/person/place_of_birth'
        prompt = template.format(head_name, relation_name, tail_name, head_name, tail_name, relation_name, head_name, tail_name)
        try:
            ret = request_api_palm(prompt, count=1)
        except Exception as E:
            print(E)
            continue
        if not isinstance(ret, str):
            continue
        ret = ret.replace('\n', '')
        s = ' |**| '.join([head, tail, relation, ret])
        if len(s.split(' |**| ')) != 4:
            print('wrong length')
            continue
        results.append(s)
    with open(os.path.join('data/processed', args.dataset, 'knowledge_context.txt'),'w', encoding='utf8') as f:
        f.write(str(len(results))+'\n')
        for item in results:
            f.write(item+'\n')

    with open(os.path.join('data/processed', args.dataset, 'train2id_choosen.txt'), 'w', encoding='utf8') as f:
        f.write(str(len(results))+'\n')
        for item in results:
            head, tail, relation, _ = item.split(' |**| ')
            item = ' '.join([head, tail, relation])
            f.write(item+'\n')

if __name__ == '__main__':
    main()