
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from Utils.WordVecs import WordVecs
from Utils.utils import *
import numpy as np


import matplotlib.pyplot as plt

from tqdm import tqdm

from collections import defaultdict
from Utils.datasets import *
from torch.utils.data import DataLoader

import os
import argparse
import pickle

from hierarchical_model import *


class Vocab(defaultdict):
    def __init__(self, train=True):
        super().__init__(lambda : len(self))
        self.train = train
        self.UNK = "UNK"
        # set UNK token to 0 index
        self[self.UNK]

    def ws2ids(self, ws):
        """ If train, you can use the default dict to add tokens
            to the vocabulary, given these will be updated during
            training. Otherwise, we replace them with UNK.
        """
        if self.train:
            return [self[w] for w in ws]
        else:
            return [self[w] if w in self else 0 for w in ws]

    def ids2sent(self, ids):
        idx2w = dict([(i, w) for w, i in self.items()])
        return [idx2w[int(i)] if i in idx2w else "UNK" for i in ids]


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def train_model(vocab,
                new_matrix,
                tag_to_ix,
                num_labels,
                embedding_dim,
                hidden_dim,
                num_lstm_layers,
                train_embeddings,
                maintask_trainX,
                maintask_trainY,
                maintask_devX,
                maintask_devY,
                auxiliary_trainX,
                auxiliary_trainY,
                auxiliary_devX,
                auxiliary_devY,
                AUXILIARY_TASK=None,
                epochs=10,
                sentiment_learning_rate=0.001,
                auxiliary_learning_rate=0.0001,
                BATCH_SIZE=50,
                number_of_runs=5,
                random_seeds=[123, 456, 789, 101112, 131415],
                FINE_GRAINED="fine"
                ):

    # Save the model parameters
    param_file = (dict(vocab.items()),
                  new_matrix.shape,
                  tag_to_ix,
                  num_labels,
                  None)

    basedir = os.path.join("saved_models",
                           "SFU",
                           args.AUXILIARY_TASK)
    outfile = os.path.join(basedir,
                           "params.pkl")
    print("Saving model parameters to " + outfile)
    os.makedirs(basedir, exist_ok=True)

    with open(outfile, "wb") as out:
        pickle.dump(param_file, out)

    for i, run in enumerate(range(number_of_runs)):

        model = Hierarchical_Model(vocab,
                                   new_matrix,
                                   tag_to_ix,
                                   num_labels,
                                   embedding_dim,
                                   hidden_dim,
                                   1,
                                   train_embeddings=train_embeddings)

        # Set our optimizers
        sentiment_params = list(model.word_embeds.parameters()) + \
                           list(model.lstm1.parameters()) +\
                           list(model.lstm2.parameters()) +\
                           list(model.linear.parameters())

        auxiliary_params = list(model.word_embeds.parameters()) + \
                           list(model.lstm1.parameters()) +\
                           list(model.hidden2tag.parameters()) +\
                           [model.transitions]

        sentiment_optimizer = torch.optim.Adam(sentiment_params, lr=sentiment_learning_rate)
        auxiliary_optimizer = torch.optim.Adam(auxiliary_params, lr=auxiliary_learning_rate)

        print("RUN {0}".format(run + 1))
        best_dev_acc = 0.0

        # set random seed for reproducibility
        np.random.seed(random_seeds[i])
        torch.manual_seed(random_seeds[i])

        for j, epoch in enumerate(range(epochs)):

            # If AUXILIARY_TASK is None, defaults to single task
            if AUXILIARY_TASK not in ["None", "none", 0, None]:

                print("epoch {0}: ".format(epoch + 1), end="")
                for k in tqdm(range(len(auxiliary_trainX))):
                    # Step 1. Remember that Pytorch accumulates gradients.
                    # We need to clear them out before each instance
                    model.zero_grad()

                    # Step 2. Get our inputs ready for the network, that is,
                    # turn them into Tensors of word indices.
                    document = auxiliary_trainX[k]
                    targets = auxiliary_trainY[k]

                    # Step 3. Run our forward pass.
                    loss = model.neg_log_likelihood_document(document, targets)

                    # Step 4. Compute the loss, gradients, and update the parameters by
                    # calling optimizer.step()
                    loss.backward()
                    auxiliary_optimizer.step()

                #model.eval_aux(auxiliary_testX, auxiliary_testY,
                #               taskname=AUXILIARY_TASK)

            batch_losses = 0
            num_batches = 0
            model.train()

            print("epoch {0}".format(epoch + 1))

            for k in tqdm(range(len(maintask_trainX))):
                model.zero_grad()

                doc = maintask_trainX[k]
                target = maintask_trainY[k]

                loss = model.pooled_sentiment_loss(doc, target)
                batch_losses += loss.data
                num_batches += 1

                loss.backward()
                sentiment_optimizer.step()

            print()
            print("loss: {0:.3f}".format(batch_losses / num_batches))
            model.eval()
            f1, acc, preds, ys = model.eval_sent(maintask_trainX, maintask_trainY)
            f1, acc, preds, ys = model.eval_sent(maintask_devX, maintask_devY)


            if acc > best_dev_acc:
                best_dev_acc = acc
                print("NEW BEST DEV ACC: {0:.3f}".format(acc))


                basedir = os.path.join("saved_models", "SFU",
                                       AUXILIARY_TASK,
                                       "{0}".format(run + 1))
                outname = "epochs:{0}-lstm_dim:{1}-lstm_layers:{2}-devacc:{3:.3f}".format(epoch + 1, model.lstm1.hidden_size, model.lstm1.num_layers, acc)
                modelfile = os.path.join(basedir,
                                         outname)
                os.makedirs(basedir, exist_ok=True)
                print("saving model to {0}".format(modelfile))
                torch.save(model.state_dict(), modelfile)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_LAYERS", "-nl", default=1, type=int)
    parser.add_argument("--HIDDEN_DIM", "-hd", default=100, type=int)
    parser.add_argument("--BATCH_SIZE", "-bs", default=50, type=int)
    parser.add_argument("--EMBEDDING_DIM", "-ed", default=300, type=int)
    parser.add_argument("--TRAIN_EMBEDDINGS", "-te", action="store_false")
    parser.add_argument("--AUXILIARY_TASK", "-aux", default="negation_scope")
    parser.add_argument("--EMBEDDINGS", "-emb",
                        default="../../embeddings/neges.txt")
    parser.add_argument("--SENTIMENT_LR", "-slr", default=0.001, type=float)
    parser.add_argument("--AUXILIARY_LR", "-alr", default=0.0001, type=float)
    parser.add_argument("--FINE_GRAINED", "-fg",
                        default="fine",
                        help="Either 'fine' or 'binary' (defaults to 'fine'.")

    args = parser.parse_args()
    print(args)

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"


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


    # Set all relevant auxiliary task parameters to None
    tag_2_idx = {"B": 0, "I": 1, "O": 2}
    tag_2_idx[START_TAG] = len(tag_2_idx)
    tag_2_idx[STOP_TAG] = len(tag_2_idx)

    maintask_trainX = [[vocab.ws2ids(s) for s in doc] for doc, pol, scope, rel in maintask_train_iter]
    maintask_trainY = [[sfu.labels[pol]] for  doc, pol, rel, scope in maintask_train_iter]
    auxiliary_trainY = [[[tag_2_idx[w] for w in s] for s in scope] for doc, pol, rel, scope in maintask_train_iter]

    maintask_devX = [[vocab.ws2ids(s) for s in doc] for doc, pol, scope, rel in maintask_dev_iter]
    maintask_devY = [[sfu.labels[pol]] for  doc, pol, rel, scope in maintask_dev_iter]
    auxiliary_devY = [[[tag_2_idx[w] for w in s] for s in scope] for doc, pol, rel, scope in maintask_dev_iter]

    # Get new embedding matrix so that words not included in pretrained embeddings have a random embedding

    #diff = len(vocab) - embeddings.vocab_length
    #print(diff)
    UNK_embedding = np.zeros((1, 300))
    #new_embeddings = np.zeros((diff, args.EMBEDDING_DIM))
    new_matrix = np.concatenate((UNK_embedding, embeddings._matrix))


    train_model(vocab,
                new_matrix,
                tag_2_idx,
                len(sfu.labels),
                args.EMBEDDING_DIM,
                args.HIDDEN_DIM,
                args.NUM_LAYERS,
                args.TRAIN_EMBEDDINGS,
                maintask_trainX,
                maintask_trainY,
                maintask_devX,
                maintask_devY,
                maintask_trainX,
                auxiliary_trainY,
                maintask_devX,
                auxiliary_devY,
                AUXILIARY_TASK=args.AUXILIARY_TASK,
                epochs=10,
                sentiment_learning_rate=args.SENTIMENT_LR,
                auxiliary_learning_rate=args.AUXILIARY_LR,
                BATCH_SIZE=50,
                number_of_runs=5,
                random_seeds=[123, 456, 789, 101112, 131415],
                FINE_GRAINED=args.FINE_GRAINED
                )
