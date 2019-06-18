from collections import defaultdict
import os

import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataloader import default_collate

import torchtext
from torchtext.datasets import SST

from Utils.read_data import *


class Split(object):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pack_words(self, ws):
        return pack_sequence(ws)

    def collate_fn(self, batch):
        batch = sorted(batch, key=lambda item : len(item[0]), reverse=True)

        words = pack_sequence([w for w,_ in batch])
        targets = default_collate([t for _,t in batch])

        return words, targets

class SFUDataset(object):
    def __init__(self, vocab, lower_case, data_dir="../../data"):

        self.vocab =  vocab
        self.labels = {"negative": 0, "positive": 1}
        self.scope_dict = {"B": 0, "I": 1, "O": 2}

        self.splits = {}
        self.splits_names = {}

        for split in ["train", "dev"]:
            self.splits_names[split], self.splits[split] = self.open_split(data_dir, split, lower_case)

        self.splits_names["test"], self.splits["test"] = self.open_split(data_dir, "test", lower_case, train=False)

    def open_split(self, data_dir, split, lower_case, train=True):
        text = torchtext.data.Field(lower=lower_case, include_lengths=True, batch_first=True)
        polarity_label = torchtext.data.Field(sequential=False)
        scope_label = torchtext.data.Field(sequential=True)
        relev_label = torchtext.data.Field(sequential=False)

        datafile = os.path.join(data_dir, split)
        filenames, dataset, polarities = get_dataset(datafile, train=train)
        scopes = [[scope_bio(s) for s in review] for review in dataset]
        relev = [[relevant_negation_tags(s) for s in review] for review in dataset]
        sents = [[clean_sent(s) for s in review] for review in dataset]


        all_data = list(zip(sents, polarities, relev, scopes))
        data = torchtext.data.Dataset(all_data,
                                      fields=[("text", text), ("polarity", polarity_label), ("scope", scope_label), ("relevance", relev_label)])

        #data_split = [(torch.LongTensor(self.vocab.ws2ids(item.text)),
        #               torch.LongTensor([self.polarity_label[item.polarity]])) for item in data]

        return filenames, data

    def get_split(self, name):
        return Split(self.splits[name])



class ChallengeDataset(object):
    def __init__(self, vocab, lower_case, data_file="../data/challenge_dataset/sst-test.txt"):
        text = torchtext.data.Field(lower=lower_case, include_lengths=True, batch_first=True)
        label = torchtext.data.Field(sequential=False)
        self.test = torchtext.data.TabularDataset(data_file, format="tsv", skip_header=False, fields=[("label", label), ("text", text)])
        self.test_split = [(torch.LongTensor(vocab.ws2ids(item.text)),
                       torch.LongTensor([int(item.label)])) for item in self.test]
    def get_split(self):
        return Split(self.test_split)



class SSTDataset(object):
    def __init__(self, vocab, lower_case, data_dir="../data/datasets/en/sst-fine"):

        self.vocab = vocab
        self.splits = {}

        for name in ["train", "dev", "test"]:
            filename = os.path.join(data_dir, name) + ".txt"
            self.splits[name] = self.open_split(filename, lower_case)

        x, y = zip(*self.splits["dev"])
        y = [int(i) for i in y]
        self.labels = sorted(set(y))

    def open_split(self, data_file, lower_case):
        text = torchtext.data.Field(lower=lower_case, include_lengths=True, batch_first=True)
        label = torchtext.data.Field(sequential=False)
        data = torchtext.data.TabularDataset(data_file, format="tsv", skip_header=False, fields=[("label", label), ("text", text)])
        data_split = [(torch.LongTensor(self.vocab.ws2ids(item.text)),
                       torch.LongTensor([int(item.label)])) for item in data]
        return data_split


    def get_split(self, name):
        return Split(self.splits[name])
