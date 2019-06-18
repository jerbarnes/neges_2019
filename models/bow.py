import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC

from Utils.datasets import *
from Utils.WordVecs import *
from hierarchical_model import Vocab

import argparse

def bow(doc, vocab):
    bow_rep = np.zeros(len(vocab))
    for sent in doc:
        for tok in sent:
            if tok in vocab:
                bow_rep[vocab[tok]] += 1
    return bow_rep


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--EMBEDDINGS", "-emb", default="../../embeddings/blse/google.txt")

    args = parser.parse_args()
    print(args)


    # Get embeddings (CHANGE TO GLOVE OR FASTTEXT EMBEDDINGS)
    embeddings = WordVecs(args.EMBEDDINGS)
    w2idx = embeddings._w2idx

    # Create shared vocabulary for tasks
    vocab = Vocab(train=True)

    # Update with word2idx from pretrained embeddings so we don't lose them
    # making sure to change them by one to avoid overwriting the UNK token
    # at index 0
    with_unk = {}
    for word, idx in embeddings._w2idx.items():
        with_unk[word] = idx + 1
    vocab.update(with_unk)

    # Import datasets
    # This will update vocab with words not found in embeddings
    sfu = SFUDataset(vocab, False, "../data")

    maintask_train_iter = sfu.get_split("train")
    maintask_dev_iter = sfu.get_split("dev")
    maintask_test_iter = sfu.get_split("test")
    
    main_train_x = [[vocab.ws2ids(s) for s in doc] for doc, pol, scope, rel in maintask_train_iter]
    main_train_y = [[sfu.labels[pol]] for  doc, pol, rel, scope in maintask_train_iter]

    main_dev_x = [[vocab.ws2ids(s) for s in doc] for doc, pol, scope, rel in maintask_dev_iter]
    main_dev_y = [[sfu.labels[pol]] for  doc, pol, rel, scope in maintask_dev_iter]

    main_test_x = [[vocab.ws2ids(s) for s in doc] for doc, pol, scope, rel in maintask_test_iter]


    train_X = [x for x, _, _, _ in maintask_train_iter]
    train_X = [bow(x, vocab) for x in train_X]
    dev_X = [x for x, _, _, _ in maintask_dev_iter]
    dev_X = [bow(x, vocab) for x in dev_X]
    test_X = [x for x, _, _, _ in maintask_test_iter]
    test_X = [bow(x, vocab) for x in test_X]

    train_y = [i[0] for i in main_train_y]
    dev_y = [i[0] for i in main_dev_y]

    clf = LinearSVC()
    clf.fit(train_X, train_y)

    pred = clf.predict(dev_X)
    acc = accuracy_score(dev_y, pred)
    print("Dev acc: {0:.3f}".format(acc))
