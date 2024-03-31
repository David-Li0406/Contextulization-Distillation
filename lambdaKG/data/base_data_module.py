from functools import partial
import argparse
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import time
from collections import defaultdict
from .qa_processor import KBQADataset
from .processor import KGCDataset
from .rec_processor import KGRECDataset
from .utils import LinkGraph

class Config(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, val):
        self[name] = val

def lmap(a, b):
    return list(map(a,b))

BATCH_SIZE = 8
NUM_WORKERS = 8
class QADataModule(pl.LightningDataModule):
    r"""

    """
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = Config(vars(args)) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)


        # base setting
        self.entity2text = self.get_entity_to_text()
        self.relation2text = self.get_relation_to_text()
        self.num_entity = len(self.entity2text.keys())
        self.num_relation = len(self.relation2text.keys())
    
    def get_entity_to_text(self):
        entity2text = {}
        with open(f"./dataset/{self.args.dataset}/entity2text.txt") as file:
            for line in file.readlines():
                id_, text = line.strip().split("\t")
                entity2text[int(id_)] = text
        return entity2text
    
    def get_relation_to_text(self):
        relation2text = {}
        with open(f"./dataset/{self.args.dataset}/relation2text.txt") as file:
            for line in file.readlines():
                id_, text = line.strip().split("\t")
                relation2text[int(id_)] = text
        
        return relation2text
    
    def setup(self, stage=None):
        now_time = time.time()
        print("setup data for each process...")
        if stage == "fit":
            self.data_train = KBQADataset(self.args, mode="train")
            self.data_val = KBQADataset(self.args, mode="dev")
        else:
            self.data_test = KBQADataset(self.args, mode="test")
        
        
        entity2text = []
        with open(f"dataset/{self.args.dataset}/entity2text.txt") as file:
            for line in file.readlines():
                line = line.strip().split("\t")[1]
                entity2text.append(line)
        
        self.entity_strings = entity2text
        # self.tokenized_entities = self.tokenizer(entity2text, padding='max_length', truncation=True, max_length=self.args.max_entity_length, return_tensors="pt")

        print("finished data processing... costing {}s...".format(time.time() - now_time))
    
    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.args.batch_size, num_workers=self.args.num_workers, 
            collate_fn=partial(self.collate_fn, mode="train"), pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="dev"), pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="test"), pin_memory=True, drop_last=False)


        



class BaseKGCDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
    TODO add 1-hop neighbors to batch for future use
    self.graph = 
    """

    def __init__(self, args: argparse.Namespace = None, lama: bool=False) -> None:
        super().__init__()
        self.args = Config(vars(args)) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.graph = None


        # base setting
        if not lama:
            self.entity2text = self.get_entity_to_text()
            self.relation2text = self.get_relation_to_text()
            self.num_entity = len(self.entity2text.keys())
            self.num_relation = len(self.relation2text.keys())

        self.contextualization = args.contextualization
        if self.contextualization:
            self.triplet2kc = {}
            with open(f"./dataset/{self.args.dataset}/knowledge_context.txt") as file:
                for line in file.readlines()[1:]:
                    line_list = line.split(' |**| ')
                    if len(line_list) == 4:
                        head, tail, relation, kc = line_list
                    if len(line_list) == 5:
                        head, tail, relation, kc_tail, kc_head = line_list
                        kc = [kc_tail, kc_head]
                    if len(line_list) == 6:
                        head, tail, relation, kc_head, kc_tail, kc_relation = line_list
                        kc = kc_head + kc_tail + kc_relation
                    head, tail, relation = head.strip(), tail.strip(), relation.strip()
                    head, tail, relation = lmap(int, [head, tail, relation])
                    self.triplet2kc[(head, tail, relation)] = kc

        self.reconstruction = args.reconstruction
        if self.reconstruction:
            self.triplet2kc = {}
            with open(f"./dataset/{self.args.dataset}/knowledge_context.txt") as file:
                for line in file.readlines()[1:]:
                    head, tail, relation, kc, kc_corrupted = line.split(' |**| ')
                    head, tail, relation = head.strip(), tail.strip(), relation.strip()
                    head, tail, relation = lmap(int, [head, tail, relation])
                    self.triplet2kc[(head, tail, relation)] = [kc, kc_corrupted]
        
        self.ent_freq = defaultdict(int)
        for mode in ["train"]:
            with open(f"dataset/{self.args.dataset}/{mode}.tsv") as file:
                for line in file.readlines():
                    h, r, t = lmap(int,line.strip().split('\t'))
                    self.ent_freq[h] += 1
                    self.ent_freq[t] += 1
    
    def get_entity_to_text(self):
        entity2text = {}
        self.entity2id = {}
        with open(f"./dataset/{self.args.dataset}/entity2text.txt") as file:
            for line in file.readlines():
                id_, text = line.strip().split("\t")
                # for generation method, we need to cut the len of input text
                if self.args.model_class == "BartKGC":
                    text = text.split(";")[0]
                entity2text[int(id_)] = text
                self.entity2id[text] = int(id_)
        return entity2text
    
    def get_relation_to_text(self):
        relation2text = {}
        with open(f"./dataset/{self.args.dataset}/relation2text.txt") as file:
            for line in file.readlines():
                id_, text = line.strip().split("\t")
                relation2text[int(id_)] = text
        
        return relation2text
    
    def setup(self, stage=None):
        now_time = time.time()
        print("setup data for each process...")
        if stage == "fit":
            self.data_train = KGCDataset(self.args, mode="train")
            self.data_val = KGCDataset(self.args, mode="dev")
        else:
            self.data_test = KGCDataset(self.args, mode="test")
        self.graph = LinkGraph(self.data_train)
        
        self.filter_hr_to_t = defaultdict(list)
        self.filter_tr_to_h = defaultdict(list)


        for mode in ["train", "dev", "test"]:
            with open(f"dataset/{self.args.dataset}/{mode}.tsv") as file:
                for line in file.readlines():
                    h, r, t = lmap(int,line.strip().split('\t'))
                    self.filter_hr_to_t[(h,r)].append(t)
                    self.filter_tr_to_h[(t,r)].append(h)
                    self.ent_freq[h] += 1
                    self.ent_freq[t] += 1
        
        self.filter_hr_to_t = {k: list(set(v)) for k, v in self.filter_hr_to_t.items()}
        self.filter_tr_to_h = {k: list(set(v)) for k, v in self.filter_tr_to_h.items()}
        max_filter_ent = max(max([len(_) for _ in self.filter_hr_to_t.values()]), max([len(_) for _ in self.filter_tr_to_h.values()]))
        print("=== max filter ent {} ===".format(max_filter_ent))
        
        entity2text = []
        with open(f"dataset/{self.args.dataset}/entity2text.txt") as file:
            for line in file.readlines():
                line = line.strip().split("\t")[1]
                entity2text.append(line)
        
        self.entity_strings = entity2text
        # self.tokenized_entities = self.tokenizer(entity2text, padding='max_length', truncation=True, max_length=self.args.max_entity_length, return_tensors="pt")

        print("finished data processing... costing {}s...".format(time.time() - now_time))
    
    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.args.batch_size, num_workers=self.args.num_workers, 
            collate_fn=partial(self.collate_fn, mode="train"), pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="dev"), pin_memory=True, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="test"), pin_memory=True, drop_last=False)




    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=0, help="Number of additional processes to load data."
        )
        parser.add_argument(
            "--dataset", type=str, default="FB15k-237", help="Number of additional processes to load data."
        )
        return parser

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU in distributed settings (so don't set state `self.x = y`).
        """
        pass

class BaseKGRECDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = Config(vars(args)) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        # base setting
        self.item2text = self.get_item_to_text()
        self.relation2text = {}
        self.num_item = len(self.item2text.keys())
        self.num_entity = self.num_item
        self.num_relation = len(self.relation2text.keys())
    
    def get_item_to_text(self):
        item2text = {}
        with open(f"./dataset/{self.args.dataset}/item2text.txt") as file:
            for line in file.readlines():
                id_, text = line.strip().split("\t")
                item2text[int(id_)] = text
        return item2text
    
    def setup(self, stage=None):
        now_time = time.time()
        print("setup data for each process...")
        if stage == "fit":
            self.data_train = KGRECDataset(self.args, mode="train")
            #self.data_val = KGRECDataset(self.args, mode="dev")
        else:
            self.data_test = KGRECDataset(self.args, mode="test")

        print("finished data processing... costing {}s...".format(time.time() - now_time))
    
    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.args.batch_size, num_workers=self.args.num_workers, 
            collate_fn=partial(self.collate_fn, mode="train"), pin_memory=False, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.data_valid, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="valid"), pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="test"), pin_memory=True, drop_last=False)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=0, help="Number of additional processes to load data."
        )
        parser.add_argument(
            "--dataset", type=str, default="ml20m", help="Number of additional processes to load data."
        )
        return parser

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU in distributed settings (so don't set state `self.x = y`).
        """
        pass