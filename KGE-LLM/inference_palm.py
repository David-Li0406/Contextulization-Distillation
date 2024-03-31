from utils import *
import json
import random
random.seed(42)
import os
from tqdm import tqdm
import time
import Levenshtein
from sentence_transformers import SentenceTransformer, util
import torch

template_tail = '''
Predict the tail entity [MASK] from the given ({}, {}, [MASK]) by completing the sentence "what is the {} of {}? The answer is ". The answer is'''
template_tail_demo = '''
Predict the tail entity [MASK] from the given ({}, {}, [MASK]) by completing the sentence "what is the {} of {}? The answer is ". The answer is {}, so the [MASK] is {}.
'''
# template_tail_temporal = '''
# Predict the tail entity [MASK] from the given ({}, {}, [MASK], {}) by completing the sentence "{} {} to (with) what (whom) at {}? The answer is ". The answer is {}, so the [MASK] is {}.
# Predict the tail entity [MASK] from the given ({}, {}, [MASK], {}) by completing the sentence "{} {} to (with) what (whom) at {}? The answer is ". The answer is (please give a exact answer)'''
template_head = '''
Predict the head entity [MASK] from the given ([MASK], {}, {}) by completing the sentence "{} is the {} of what? The answer is ". The answer is'''
template_head_demo = '''
Predict the head entity [MASK] from the given ([MASK], {}, {}) by completing the sentence "{} is the {} of what? The answer is ". The answer is {}, so the [MASK] is {}.
'''
# template_head_temporal = '''
# Predict the head entity [MASK] from the given ([MASK], {}, {}, {}) by completing the sentence "What (Who) {} to (with) {} in {}? The answer is ". The answer is {}, so the [MASK] is {}.
# Predict the head entity [MASK] from the given ([MASK], {}, {}, {}) by completing the sentence "What (Who) {} to (with) {} in {}? The answer is ". The answer is (please give a exact answer)'''

demo_num=2
dataset = 'FB15k-237N'


def inference():
    relation2training_sample = {}
    testing_samples = []
    with open(os.path.join('data/processed', dataset, 'train2id_name.txt'), encoding='utf8') as f:
        for line in f.readlines()[1:]:
            if dataset == 'ICEWS14':
                head, tail, relation, t = line.split('|')
                head, tail, relation, t = head.strip(), tail.strip(), relation.strip(), t.strip()
            else:
                head, tail, relation = line.split('|')
                head, tail, relation = head.strip(), tail.strip(), relation.strip()
            if dataset == 'WN18RR':
                head = ' '.join(head[2:].split('_')[:-2])
                tail = ' '.join(tail[2:].split('_')[:-2])
            if relation not in relation2training_sample:
                relation2training_sample[relation] = []
            if dataset == 'ICEWS14':
                relation2training_sample[relation].append([head, tail, relation, t])
            else:
                relation2training_sample[relation].append([head, tail, relation])
    with open(os.path.join('data/processed', dataset, 'test2id_name.txt'), encoding='utf8') as f:
        for line in f.readlines()[1:]:
            if dataset == 'ICEWS14':
                head, tail, relation, t = line.split('|')
                head, tail, relation, t = head.strip(), tail.strip(), relation.strip(), t.strip()
            else:
                head, tail, relation = line.split('|')
                head, tail, relation = head.strip(), tail.strip(), relation.strip()
            if dataset == 'WN18RR':
                head = ' '.join(head[2:].split('_')[:-2])
                tail = ' '.join(tail[2:].split('_')[:-2])
            if dataset == 'ICEWS14':
                testing_samples.append([head, tail, relation, t])
            else:
                testing_samples.append([head, tail, relation])

    if dataset == 'FB15k-237':
        testing_samples = random.sample(testing_samples, int(len(testing_samples)*0.1))
    
    item_all = []
    for sample in tqdm(testing_samples):
        time.sleep(4)
        if dataset == 'ICEWS14':
            head, tail, relation, t = sample
            relation_text = relation.split('/')[-1]
            demonstration = random.sample(relation2training_sample[relation], 1)[0]
            head_demonstration, tail_demonstration, _, t_demonstration = demonstration
            prompt_tail = template_tail_temporal.format(
                head_demonstration,
                relation,
                t_demonstration,
                head_demonstration,
                relation_text,
                t_demonstration,
                tail_demonstration,
                tail_demonstration,
                head,
                relation,
                t,
                head,
                relation,
                t,
            )

            prompt_head = template_head_temporal.format(
                relation,
                tail_demonstration,
                t_demonstration,
                relation_text,
                tail_demonstration,
                t_demonstration,
                head_demonstration,
                head_demonstration,
                relation,
                tail,
                t,
                relation_text,
                tail,
                t,
            )

        else:
            head, tail, relation = sample
            relation_text = relation.split('/')[-1]
            prompt_tail, prompt_head = '', ''
            for i in range(demo_num):
                demonstration = random.sample(relation2training_sample[relation], 1)[0]
                head_demonstraton, tail_demonstration, _ = demonstration
                prompt_tail += template_tail_demo.format(
                    head_demonstraton,
                    relation,
                    relation_text,
                    head_demonstraton,
                    tail_demonstration,
                    tail_demonstration,
                )

                prompt_head += template_head_demo.format(
                    relation,
                    tail_demonstration,
                    tail_demonstration,
                    relation_text,
                    head_demonstraton,
                    head_demonstraton,
                )
                
            prompt_tail += template_tail.format(
                head,
                relation,
                relation_text,
                head
            )

            prompt_head += template_head.format(
                relation,
                tail,
                tail,
                relation_text
            )

        try:
            ret_tail = request_api_palm(prompt_tail)
            ret_head = request_api_palm(prompt_head)
        except:
            continue
        
        _ret_tail, _ret_head = [], []
        for res in ret_tail:
            if 'so the [MASK] is' in res:
                res = res.split('so the [MASK] is')[1].strip()
                if res[-1] == '.':
                    res = res[:-1]
            _ret_tail.append(res)
        for res in ret_head:
            if res == '':
                continue
            if 'so the [MASK] is' in res:
                res = res.split('so the [MASK] is')[1].strip()
                if res[-1] == '.':
                    res = res[:-1]
            _ret_head.append(res)

        ret_tail, ret_head = _ret_tail, _ret_head

        item = {
            'head': head,
            'relation': relation,
            'tail': tail,
            'prediction_tail': ret_tail,
            'prediction_head': ret_head
        }
        item_all.append(item)

    with open('{}_tail_results_palm.json'.format(dataset), 'w') as f:
        json.dump(item_all, f)



def extract():
    inference_result = json.load(open('{}_tail_results_palm.json'.format(dataset)))
    
    hit_1_tail, hit_3_tail, hit_8_tail = 0, 0, 0
    for item in inference_result:
        prediction_1, prediction_3, prediction_8 = item['prediction_tail'][:1], item['prediction_tail'][:3], item['prediction_tail'][:8]
        if item['tail'].lower() in [pred.lower() for pred in prediction_1]:
            hit_1_tail += 1
        if item['tail'].lower() in [pred.lower() for pred in prediction_3]:
            hit_3_tail += 1
        if item['tail'].lower() in [pred.lower() for pred in prediction_8]:
            hit_8_tail += 1

    hit_1_head, hit_3_head, hit_8_head = 0, 0, 0
    for item in inference_result:
        prediction_1, prediction_3, prediction_8 = item['prediction_head'][:1], item['prediction_head'][:3], item['prediction_head'][:8]
        if item['head'].lower() in [pred.lower() for pred in prediction_1]:
            hit_1_head += 1
        if item['head'].lower() in [pred.lower() for pred in prediction_3]:
            hit_3_head += 1
        if item['head'].lower() in [pred.lower() for pred in prediction_8]:
            hit_8_head += 1
    
    print('hit_1:', (hit_1_tail+hit_1_head)/(2*len(inference_result)))
    print('hit_3:', (hit_3_tail+hit_3_head)/(2*len(inference_result)))
    print('hit_8:', (hit_8_tail+hit_8_head)/(2*len(inference_result)))

def extract_leve():
    all_entity = []
    with open(os.path.join('data/processed', dataset, 'entityid2name.txt'), encoding='utf8') as f:
        for line in f.readlines()[1:]:
            _, entity = line.split('\t')
            if dataset == 'WN18RR':
                entity = entity.split(' , ')[0]
            all_entity.append(entity.strip())

    inference_result = json.load(open('{}_tail_results_palm.json'.format(dataset)))
    
    hit_1_tail, hit_3_tail, hit_8_tail = 0, 0, 0
    for item in tqdm(inference_result):
        # prediction_1, prediction_3, prediction_8 = item['prediction'][:1], item['prediction'][:3], item['prediction'][:8]
        prediction = item['prediction_tail']
        entity_matched_all = []
        for pred in prediction[:1]:
            pred = pred.lower()
            entity2score = {}
            for entity in all_entity:
                entity = entity.lower()
                entity2score[entity] = Levenshtein.distance(pred, entity)
            entity2score = sorted(entity2score.items(), key=lambda x: x[1])
            entity_matched = entity2score[0][0]
            entity_matched_all.append(entity_matched)
        if item['tail'].lower() in entity_matched_all:
            hit_1_tail += 1
            hit_3_tail += 1
            hit_8_tail += 1
            continue
        
        entity_matched_all = []
        for pred in prediction[1:3]:
            pred = pred.lower()
            entity2score = {}
            for entity in all_entity:
                entity = entity.lower()
                entity2score[entity] = Levenshtein.distance(pred, entity)
            entity2score = sorted(entity2score.items(), key=lambda x: x[1])
            entity_matched = entity2score[0][0]
            entity_matched_all.append(entity_matched)
        if item['tail'].lower() in entity_matched_all:
            hit_3_tail += 1
            hit_8_tail += 1
            continue
    
        entity_matched_all = []
        for pred in prediction[3:8]:
            pred = pred.lower()
            entity2score = {}
            for entity in all_entity:
                entity = entity.lower()
                entity2score[entity] = Levenshtein.distance(pred, entity)
            entity2score = sorted(entity2score.items(), key=lambda x: x[1])
            entity_matched = entity2score[0][0]
            entity_matched_all.append(entity_matched)
        if item['tail'].lower() in entity_matched_all:
            hit_8_tail += 1
            continue

    hit_1_head, hit_3_head, hit_8_head = 0, 0, 0
    for item in tqdm(inference_result):
        # prediction_1, prediction_3, prediction_8 = item['prediction'][:1], item['prediction'][:3], item['prediction'][:8]
        prediction = item['prediction_head']
        entity_matched_all = []
        for pred in prediction[:1]:
            pred = pred.lower()
            entity2score = {}
            for entity in all_entity:
                entity = entity.lower()
                entity2score[entity] = Levenshtein.distance(pred, entity)
            entity2score = sorted(entity2score.items(), key=lambda x: x[1])
            entity_matched = entity2score[0][0]
            entity_matched_all.append(entity_matched)
        if item['head'].lower() in entity_matched_all:
            hit_1_head += 1
            hit_3_head += 1
            hit_8_head += 1
            continue
        
        entity_matched_all = []
        for pred in prediction[1:3]:
            pred = pred.lower()
            entity2score = {}
            for entity in all_entity:
                entity = entity.lower()
                entity2score[entity] = Levenshtein.distance(pred, entity)
            entity2score = sorted(entity2score.items(), key=lambda x: x[1])
            entity_matched = entity2score[0][0]
            entity_matched_all.append(entity_matched)
        if item['head'].lower() in entity_matched_all:
            hit_3_head += 1
            hit_8_head += 1
            continue
    
        entity_matched_all = []
        for pred in prediction[3:8]:
            pred = pred.lower()
            entity2score = {}
            for entity in all_entity:
                entity = entity.lower()
                entity2score[entity] = Levenshtein.distance(pred, entity)
            entity2score = sorted(entity2score.items(), key=lambda x: x[1])
            entity_matched = entity2score[0][0]
            entity_matched_all.append(entity_matched)
        if item['head'].lower() in entity_matched_all:
            hit_8_head += 1
            continue

    print('hit_1:', (hit_1_tail+hit_1_head)/(2*len(inference_result)))
    print('hit_3:', (hit_3_tail+hit_3_head)/(2*len(inference_result)))
    print('hit_8:', (hit_8_tail+hit_8_head)/(2*len(inference_result)))
        


def extract_similarity():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_entity = []
    with open(os.path.join('data/processed', dataset, 'entityid2name.txt'), encoding='utf8') as f:
        for line in f.readlines()[1:]:
            _, entity = line.split('\t')
            if dataset == 'WN18RR':
                entity = entity.split(' , ')[0]
            all_entity.append(entity.strip())
    all_entity_embedding = model.encode(all_entity, convert_to_tensor=True)

    inference_result = json.load(open('{}_tail_results_palm.json'.format(dataset)))

    hit_1_tail, hit_3_tail, hit_8_tail = 0, 0, 0
    for item in tqdm(inference_result):
        # prediction_1, prediction_3, prediction_8 = item['prediction'][:1], item['prediction'][:3], item['prediction'][:8]
        prediction = item['prediction_tail']
        entity_matched_all = []
        for pred in prediction[:1]:
            pred = pred.lower()
            pred_embedding = model.encode(pred, convert_to_tensor=True)
            cos_scores = util.cos_sim(pred_embedding, all_entity_embedding)[0]
            top_results = torch.topk(cos_scores, k=1)
            score, idx = top_results
            entity_matched_all.append(all_entity[idx[0].item()].lower())
        if item['tail'].lower() in entity_matched_all:
            hit_1_tail += 1
            hit_3_tail += 1
            hit_8_tail += 1
            continue
        
        entity_matched_all = []
        for pred in prediction[1:3]:
            pred = pred.lower()
            pred_embedding = model.encode(pred, convert_to_tensor=True)
            cos_scores = util.cos_sim(pred_embedding, all_entity_embedding)[0]
            top_results = torch.topk(cos_scores, k=1)
            score, idx = top_results
            entity_matched_all.append(all_entity[idx[0].item()].lower())
        if item['tail'].lower() in entity_matched_all:
            hit_3_tail += 1
            hit_8_tail += 1
            continue
    
        entity_matched_all = []
        for pred in prediction[3:8]:
            pred = pred.lower()
            pred_embedding = model.encode(pred, convert_to_tensor=True)
            cos_scores = util.cos_sim(pred_embedding, all_entity_embedding)[0]
            top_results = torch.topk(cos_scores, k=1)
            score, idx = top_results
            entity_matched_all.append(all_entity[idx[0].item()].lower())
        if item['tail'].lower() in entity_matched_all:
            hit_8_tail += 1
            continue
    
    hit_1_head, hit_3_head, hit_8_head = 0, 0, 0
    for item in tqdm(inference_result):
        # prediction_1, prediction_3, prediction_8 = item['prediction'][:1], item['prediction'][:3], item['prediction'][:8]
        prediction = item['prediction_head']
        entity_matched_all = []
        for pred in prediction[:1]:
            pred = pred.lower()
            pred_embedding = model.encode(pred, convert_to_tensor=True)
            cos_scores = util.cos_sim(pred_embedding, all_entity_embedding)[0]
            top_results = torch.topk(cos_scores, k=1)
            score, idx = top_results
            entity_matched_all.append(all_entity[idx[0].item()].lower())
        if item['head'].lower() in entity_matched_all:
            hit_1_head += 1
            hit_3_head += 1
            hit_8_head += 1
            continue
        
        entity_matched_all = []
        for pred in prediction[1:3]:
            pred = pred.lower()
            pred_embedding = model.encode(pred, convert_to_tensor=True)
            cos_scores = util.cos_sim(pred_embedding, all_entity_embedding)[0]
            top_results = torch.topk(cos_scores, k=1)
            score, idx = top_results
            entity_matched_all.append(all_entity[idx[0].item()].lower())
        if item['head'].lower() in entity_matched_all:
            hit_3_head += 1
            hit_8_head += 1
            continue
    
        entity_matched_all = []
        for pred in prediction[3:8]:
            pred = pred.lower()
            pred_embedding = model.encode(pred, convert_to_tensor=True)
            cos_scores = util.cos_sim(pred_embedding, all_entity_embedding)[0]
            top_results = torch.topk(cos_scores, k=1)
            score, idx = top_results
            entity_matched_all.append(all_entity[idx[0].item()].lower())
        if item['head'].lower() in entity_matched_all:
            hit_8_head += 1
            continue

    print('hit_1:', (hit_1_tail+hit_1_head)/(2*len(inference_result)))
    print('hit_3:', (hit_3_tail+hit_3_head)/(2*len(inference_result)))
    print('hit_8:', (hit_8_tail+hit_8_head)/(2*len(inference_result)))

if __name__ == '__main__':
    inference()
    extract()
    extract_leve()
    extract_similarity()