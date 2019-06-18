import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import DataLoader
from collections import Counter

from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC

from Utils.datasets import *
from Utils.WordVecs import *
from hierarchical_training import *

import argparse

def bio_classification_report(y_true, y_pred):
    """
    Taken from: https://nbviewer.jupyter.org/github/tpeng/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb

    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset)


class Hierarchical_Model(nn.Module):

    def __init__(self, word2idx,
                 embedding_matrix,
                 tag_2_idx,
                 sentiment_label_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers=2,
                 lstm_dropout=0.2,
                 word_dropout=0.5,
                 train_embeddings=False,
                 START_TAG="<START>",
                 STOP_TAG="<STOP>"):
        super(Hierarchical_Model, self).__init__()
        self.vocab = word2idx
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2idx)

        self.tag_2_idx = tag_2_idx
        self.tagset_size = len(tag_2_idx)
        self.lstm_dropout = lstm_dropout
        self.word_dropout = word_dropout
        self.sentiment_criterion = nn.CrossEntropyLoss()

        self.START_TAG = START_TAG
        self.STOP_TAG = STOP_TAG

        weight = torch.FloatTensor(embedding_matrix)
        self.word_embeds = nn.Embedding.from_pretrained(weight, freeze=False)
        self.word_embeds.requires_grad = train_embeddings
        #self.word_embeds = nn.Embedding(len(vocab), embedding_dim)

        self.lstm1 = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=1,
                            bidirectional=True)

        self.lstm2 = nn.LSTM(hidden_dim*2,
                            hidden_dim,
                            num_layers=1,
                            bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_2_idx[self.START_TAG], :] = -10000
        self.transitions.data[:, tag_2_idx[self.STOP_TAG]] = -10000

        # Set up layers for sentiment prediction
        self.word_dropout = nn.Dropout(word_dropout)
        self.batch_norm = nn.BatchNorm1d(embedding_dim)
        self.linear = nn.Linear(hidden_dim*2, sentiment_label_size)



    def max_pool(self, sentence):
        x = torch.tensor(sentence)
        emb = self.word_embeds(x).unsqueeze(1)
        emb = self.word_dropout(emb)

        int_output, (hn, cn) = self.lstm1(emb)
        pooled, _ = int_output.max(dim=0)
        return pooled

    def get_document_rep(self, doc):
        sentence_reps = []
        for sent in doc:
            sentence_reps.append(self.max_pool(sent))

        sentence_reps = torch.cat(sentence_reps).unsqueeze(1)
        o, (h, c) = self.lstm2(sentence_reps)
        
        pooled, _ = o.max(dim=0)
        output = self.linear(pooled)
        return output

    def predict_sentiment(self, x):
        scores = self.get_document_rep(x)
        probs = F.softmax(scores, dim=1)
        preds = probs.argmax(dim=1)
        return preds.numpy()

    def pooled_sentiment_loss(self, doc, label):
        if type(label) == list:
            label = torch.tensor(label)
        pred = self.get_document_rep(doc)
        loss = self.sentiment_criterion(pred, label)
        return loss

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_2_idx[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_2_idx[self.STOP_TAG]]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, hidden = self.lstm1(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim * 2)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_2_idx[self.START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_2_idx[self.STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_2_idx[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_2_idx[self.STOP_TAG]]
        best_tag_id = self.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_2_idx[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        sentence = torch.tensor(sentence)
        tags = torch.tensor(tags)
        
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def neg_log_likelihood_document(self, document, tags):
        loss = 0
        for sent, tag in zip(document, tags):

            # CLEAN UP THE REGEX
            if len(sent) == len(tag):
                loss += self.neg_log_likelihood(sent, tag)
        return loss / len(document)

    def forward(self, sentence):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def argmax(self, vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    # Compute log sum exp in a numerically stable way for the forward algorithm
    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
            torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def eval_aux(self, X, Y):
        idx2label = dict([(i,w) for w,i in self.tag_2_idx.items()])
        preds = []
        ys = []

        with torch.no_grad():
            for doc, tags in zip(X, Y):
                for i, sent in enumerate(doc):
                    x = torch.tensor(sent)
                    y = tags[i]

                    score, pred = self.forward(x)

                    #NEED TO CLEAN UP REGEX TO REMOVE ANNS
                    if len(pred) == len(y):
                        preds.append([idx2label[i] for i in pred])
                        ys.append([idx2label[i] for i in y])

        print(bio_classification_report(ys, preds))

    def eval_sent(self, docs, ys):
        preds = []

        with torch.no_grad():
            for doc in docs:
                pred = self.predict_sentiment(doc)
                preds.append(pred)
        f1 = f1_score(ys, preds, average="macro")
        acc = accuracy_score(ys, preds)
        print("Sentiment F1: {0:.3f}".format(f1))
        print("Sentiment Acc: {0:.3f}".format(acc))
        return f1, acc, preds, ys


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_LAYERS", "-nl", default=1, type=int)
    parser.add_argument("--HIDDEN_DIM", "-hd", default=100, type=int)
    parser.add_argument("--BATCH_SIZE", "-bs", default=50, type=int)
    parser.add_argument("--EMBEDDING_DIM", "-ed", default=300, type=int)
    parser.add_argument("--TRAIN_EMBEDDINGS", "-te", action="store_true")
    parser.add_argument("--AUXILIARY_TASK", "-aux", default="negation_scope")
    parser.add_argument("--EMBEDDINGS", "-emb", default="../../embeddings/google.txt")

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
    maintask_test_iter = sfu.get_split("test")

    maintask_loader = DataLoader(maintask_train_iter,
                                 batch_size=args.BATCH_SIZE,
                                 collate_fn=maintask_train_iter.collate_fn,
                                 shuffle=True)

    tag_2_idx = {START_TAG:3, STOP_TAG:4, "B":0, "I":1, "O":2}
    
    main_train_x = [[vocab.ws2ids(s) for s in doc] for doc, pol, scope, rel in maintask_train_iter]
    main_train_y = [[sfu.labels[pol]] for  doc, pol, rel, scope in maintask_train_iter]
    aux_train_y = [[[tag_2_idx[w] for w in s] for s in scope] for doc, pol, rel, scope in maintask_train_iter]

    main_dev_x = [[vocab.ws2ids(s) for s in doc] for doc, pol, scope, rel in maintask_dev_iter]
    main_dev_y = [[sfu.labels[pol]] for  doc, pol, rel, scope in maintask_dev_iter]
    aux_dev_y = [[[tag_2_idx[w] for w in s] for s in scope] for doc, pol, rel, scope in maintask_dev_iter]

    main_test_x = [[vocab.ws2ids(s) for s in doc] for doc, pol, scope, rel in maintask_test_iter]

    diff = len(vocab) - embeddings.vocab_length - 1
    UNK_embedding = np.zeros((1, 300))
    new_embeddings = np.zeros((diff, args.EMBEDDING_DIM))
    new_matrix = np.concatenate((UNK_embedding, embeddings._matrix, new_embeddings))


    def bow(doc, vocab):
        bow_rep = np.zeros(len(vocab))
        for sent in doc:
            for tok in sent:
                if tok in vocab:
                    bow_rep[vocab[tok]] += 1
        return bow_rep

    train_X = [x for x, _, _, _ in maintask_train_iter]
    dev_X = [x for x, _, _, _ in maintask_dev_iter]
    test_X = [x for x, _, _, _ in maintask_test_iter]

    c = Counter()
    for d in train_X + dev_X + test_X:
        for  s in d:
            c.update(s)

    voc = {}
    for w, i in c.items():
        voc[w] = len(voc)
    

    train_y = [i[0] for i in main_train_y]
    dev_y = [i[0] for i in main_dev_y]

    clf = LinearSVC()
    clf.fit(train_X, train_y)
    clf.score(dev_X, dev_y)
