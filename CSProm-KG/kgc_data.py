import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from helper import dataloader_output_to_tensor, get_lar_sample_bank
import random
random.seed(42)


class BaseDataset(Dataset):
    def __init__(self, configs, tok, triples, text_dict, mode='train', gt=None, lars_dict=None):
        super().__init__()
        self.configs = configs
        self.tok = tok
        self.ent_names = text_dict['ent_names']
        self.rel_names = text_dict['rel_names']
        self.ent_descs = text_dict['ent_descs']
        self.triples = triples
        self.mode = mode
        if gt is not None:
            self.train_tail_gt = gt['train_tail_gt']
            self.train_head_gt = gt['train_head_gt']
        if lars_dict is not None:
            self.name_lars = lars_dict['name_lars']
            self.desc_lars = lars_dict['desc_lars']

    def parse_ent_name(self, name):
        if self.configs.dataset == 'WN18RR':
            name = ' '.join(name.split(' , ')[:-2])
            return name
        return name or ''

    def construct_input_text(self, src_ent=None, rel=None, timestamp=None, predict='predict_tail'):
        src_name = self.ent_names[src_ent]
        rel_name = self.rel_names[rel]
        src_desc = ':' + self.ent_descs[src_ent] if self.configs.desc_max_length > 0 else ''

        timestamp = ' | ' + timestamp if timestamp else ''
        if predict == 'predict_tail':
            return src_name + ' ' + src_desc, rel_name + timestamp
        elif predict == 'predict_head':
            return src_name + ' ' + src_desc, 'reversed: ' + rel_name + timestamp
        else:
            raise ValueError('Mode is not correct!')

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = dataloader_output_to_tensor(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = dataloader_output_to_tensor(data, 'source_mask', padding_value=0)
        agg_data['ent_rel'] = dataloader_output_to_tensor(data, 'ent_rel')
        agg_data['tgt_ent'] = dataloader_output_to_tensor(data, 'tgt_ent', return_list=True)
        agg_data['triple'] = dataloader_output_to_tensor(data, 'triple', return_list=True)
        if self.mode == 'train':
            agg_data['labels'] = dataloader_output_to_tensor(data, 'labels').squeeze(-1)
        if self.mode == 'train' and self.configs.n_lar > 0:
            agg_data['lars'] = dataloader_output_to_tensor(data, 'lars')
        if self.mode == 'train' and self.configs.reconstruction:
            agg_data['kc_ids'] = dataloader_output_to_tensor(data, 'kc_ids', padding_value=0)
            agg_data['kc_mask'] = dataloader_output_to_tensor(data, 'kc_mask', padding_value=0)
            agg_data['kc_labels'] = dataloader_output_to_tensor(data, 'kc_labels', padding_value=-1)
            agg_data['ent_rel_kc'] = dataloader_output_to_tensor(data, 'ent_rel_kc')
        return agg_data


class CELossDataset(BaseDataset):
    def __init__(self, configs, tok, triples, text_dict, mode='train', gt=None, lars_dict=None, triplet2kc=None):
        super().__init__(configs, tok, triples, text_dict, mode, gt, lars_dict)
        self.all_ent = set(range(configs.n_ent))
        self.triplet2kc=triplet2kc
        self.vocab_list = list(self.tok.vocab.keys())
        # print(self.vocab_list)

    def __len__(self):
        return len(self.triples) * 2 if self.mode == 'train' else len(self.triples)

    def __getitem__(self, index):
        if self.mode == 'train':
            mode = 'predict_tail' if index % 2 == 0 else 'predict_head'
            triple = self.triples[index // 2]
        else:
            mode = self.mode
            triple = self.triples[index]

        if not self.configs.is_temporal:
            head, tail, rel = triple
            timestamp = None
        else:
            head, tail, rel, timestamp = triple

        if mode == 'predict_tail':
            src = self.construct_input_text(src_ent=head,
                                            rel=rel,
                                            timestamp=timestamp,
                                            predict='predict_tail')
            tgt_ent = tail
        elif mode == 'predict_head':
            src = self.construct_input_text(src_ent=tail,
                                            rel=rel,
                                            timestamp=timestamp,
                                            predict='predict_head')
            tgt_ent = head
        else:
            raise ValueError('Mode is not correct!')

        if self.mode == 'train' and self.configs.reconstruction:
            key = (head, tail, rel)
            if key in self.triplet2kc:
                kc = self.triplet2kc[key]
            else:
                key = random.choice(list(self.triplet2kc.keys()))
                kc = self.triplet2kc[key]
            kc_tokens = self.tok.tokenize(kc)
            reconsturction_label = self.random_word(kc_tokens)
            kc_tokens = ['<CLS>'] + kc_tokens + ['<SEP>']
            # print(kc_tokens)
            # print(self.tok.convert_ids_to_tokens(reconsturction_label))
            reconsturction_label = [-1] + reconsturction_label + [-1]
            kc_tokens = kc_tokens[:128]
            reconsturction_label = reconsturction_label[:128]
            # assert self.tokenizer.convert_tokens_to_ids(kc_tokens) == self.tokenizer.encode(kc)
            kc_ids = self.tok.convert_tokens_to_ids(kc_tokens)
            kc_mask = [1 for _ in kc_ids]
            assert len(kc_ids) == len(reconsturction_label)
            

        ent_rel = (head, rel) if mode == 'predict_tail' else (tail, rel + self.configs.n_rel)
        src, text_pair = src
        tokenized_src = self.tok(src, text_pair=text_pair, max_length=self.configs.text_len, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        if self.mode == 'train' and self.configs.n_lar > 0:
            gt = self.train_tail_gt if mode == 'predict_tail' else self.train_head_gt
            if self.configs.use_speedup:
                lar_list = list((set(self.name_lars[tgt_ent]) | set(self.desc_lars[tgt_ent])) - set(gt[ent_rel]))
                if len(lar_list) < self.configs.n_lar:
                    lars = lar_list + np.random.choice(list(self.all_ent), self.configs.n_lar - len(lar_list), replace=False).tolist()
                else:
                    lars = np.random.choice(lar_list, self.configs.n_lar, replace=False).tolist()
            else:
                name_lar_max_len = self.configs.n_lar // 2
                name_lars = list(set(self.name_lars[tgt_ent]) - set(gt[ent_rel]))
                desc_lars = list(set(self.desc_lars[tgt_ent]) - set(gt[ent_rel]))
                name_lars = name_lars if len(name_lars) < name_lar_max_len else np.random.choice(name_lars, name_lar_max_len, replace=False).tolist()
                desc_lar_max_len = self.configs.n_lar - len(name_lars)
                desc_lars = desc_lars if len(desc_lars) < desc_lar_max_len else np.random.choice(desc_lars, desc_lar_max_len, replace=False).tolist()
                if len(name_lars + desc_lars) < self.configs.n_lar:
                    other_lars = np.random.choice(list(range(self.configs.n_ent)), self.configs.n_lar - len(name_lars + desc_lars), replace=False).tolist()
                else:
                    other_lars = []
                lars = name_lars + desc_lars + other_lars

        out = {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'triple': triple,
            'ent_rel': ent_rel,
            'tgt_ent': tgt_ent,
        }

        if self.mode == 'train' and self.configs.reconstruction:
            out['kc_ids'] = kc_ids
            out['kc_mask'] = kc_mask
            out['kc_labels'] = reconsturction_label
            out['ent_rel_kc'] = list([int(item) for item in key[:3]])

        if self.mode == 'train':
            out['labels'] = [tgt_ent]
        if self.mode == 'train' and self.configs.n_lar > 0:
            out['lars'] = lars

        return out

    def random_word(self, tokens):
  
        output_label = []

        for i, token in enumerate(tokens):
            if token=='[eos]':
                output_label.append(-1)
                continue
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"

                # 10% randomly change token to random token
                elif prob < 0.9:
                    # tokens[i] = random.choice(list(self.tokenizer.vocab.items()))[0]
                    tokens[i] = random.choice(self.vocab_list)

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                try:
                    output_label.append(self.tok.convert_tokens_to_ids([token])[0])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(self.tok.convert_tokens_to_ids(["[UNK]"])[0])
                    logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return output_label


class KGCDataModule(pl.LightningDataModule):
    def __init__(self, configs, train, valid, test, text_dict, tok, gt, triplet2kc):
        super().__init__()
        self.configs = configs
        self.train = train
        self.valid = valid
        self.test = test
        # ent_names, rel_names .type: list
        self.text_dict = text_dict
        self.tok = tok
        self.gt = gt
        self.triplet2kc = triplet2kc
        self.lars_dict = None
        if configs.n_lar > 0:
            self.lars_dict = get_lar_sample_bank(configs, text_dict)

        self.train_both = CELossDataset(configs, tok, train, text_dict, 'train', self.gt, self.lars_dict, triplet2kc=triplet2kc)
        self.valid_tail = CELossDataset(configs, tok, valid, text_dict, 'predict_tail', triplet2kc=triplet2kc)
        self.valid_head = CELossDataset(configs, tok, valid, text_dict, 'predict_head', triplet2kc=triplet2kc)
        self.test_tail = CELossDataset(configs, tok, test, text_dict, 'predict_tail', triplet2kc=triplet2kc)
        self.test_head = CELossDataset(configs, tok, test, text_dict, 'predict_head', triplet2kc=triplet2kc)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_both,
                                  batch_size=self.configs.batch_size,
                                  shuffle=True,
                                  collate_fn=self.train_both.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)
        return train_loader

    def val_dataloader(self):
        valid_tail_loader = DataLoader(self.valid_tail,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_tail.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        valid_head_loader = DataLoader(self.valid_head,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_head.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        return [valid_tail_loader, valid_head_loader]

    def test_dataloader(self):
        test_tail_loader = DataLoader(self.test_tail,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_tail.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)
        test_head_loader = DataLoader(self.test_head,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_head.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)
        return [test_tail_loader, test_head_loader]
