import argparse
import pickle

from hierarchical_training import *
from hierarchical_model import *
from Utils.datasets import *

def get_best_run(weightdir):
    """
    This returns the best dev f1, parameters, and weights from the models
    found in the weightdir.
    """
    best_params = []
    best_acc = 0.0
    best_weights = ''
    for file in os.listdir(weightdir):
        epochs = int(re.findall('[0-9]+', file.split('-')[-4])[0])
        lstm_dim = int(re.findall('[0-9]+', file.split('-')[-3])[0])
        lstm_layers = int(re.findall('[0-9]+', file.split('-')[-2])[0])
        acc = float(re.findall('0.[0-9]+', file.split('-')[-1])[0])
        if acc > best_acc:
            best_params = [epochs, lstm_dim, lstm_layers]
            best_acc = acc
            weights = os.path.join(weightdir, file)
            best_weights = weights
    return best_acc, best_params, best_weights

def test_model(aux_task, num_runs=5, metric="acc", FINE_GRAINED="fine"):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--AUXILIARY_TASK", "-aux", default="negation_scope")
    parser.add_argument("--NUM_RUNS", "-nr", default=5, type=int)
    parser.add_argument("--METRIC", "-m", default="acc")
    parser.add_argument("--FINE_GRAINED", "-fg",
                        default="fine",
                        help="Either 'fine' or 'binary' (defaults to 'fine'.")
    args = parser.parse_args()

    # f1s, accs, preds, ys = test_model(args.AUXILIARY_TASK,
    #                                   num_runs=args.NUM_RUNS,
    #                                   metric=args.METRIC,
    #                                   FINE_GRAINED=args.FINE_GRAINED)

    aux_task = args.AUXILIARY_TASK
    num_runs = args.NUM_RUNS
    metric = args.METRIC

    f1s = []
    accs = []
    preds = []
    ys = []

    print("opening model params...")
    with open(os.path.join("saved_models_random_embs",
                           "SFU",
                           aux_task,
                           "params.pkl"), "rb") as infile:
        params = pickle.load(infile)

    (w2idx,
     matrix_shape,
     tag_to_ix,
     len_labels,
     task2label2id) = params

    vocab = Vocab(train=False)
    vocab.update(w2idx)

    sfu = SFUDataset(vocab, False, "../data")
    maintask_dev_iter = sfu.get_split("dev")
    maintask_devX = [[vocab.ws2ids(s) for s in doc] for doc, pol, scope, rel in maintask_dev_iter]
    maintask_devY = [[sfu.labels[pol]] for  doc, pol, rel, scope in maintask_dev_iter]

    maintask_test_iter = sfu.get_split("test")

    maintask_testX = [[vocab.ws2ids(s) for s in doc] for doc, pol, scope, rel in maintask_test_iter]


    new_matrix = np.zeros(matrix_shape)

    idx_2_label = {0: "negative", 1: "positive"}


    print("finding best weights for runs 1 - {0}".format(num_runs))
    for i in range(num_runs):
        run = i + 1
        weight_dir = os.path.join("saved_models_random_embs",
                                  "SFU",
                                  aux_task,
                                  str(run))
        best_acc, (epochs, lstm_dim, lstm_layers), best_weights =\
                                                   get_best_run(weight_dir)

        model = Hierarchical_Model(vocab,
                                   new_matrix,
                                   tag_to_ix,
                                   2,
                                   300,
                                   lstm_dim,
                                   1,
                                   train_embeddings=False)



        model.load_state_dict(torch.load(best_weights))
        model.eval()

        # DEV ACC
        print("DEV")
        f1, acc, preds, ys = model.eval_sent(maintask_devX, maintask_devY)

        # print("Run {0}".format(run))
        # pred = []
        # for k in tqdm(range(len(maintask_testX))):
        #     p = model.predict_sentiment(maintask_testX[k])
        #     p = idx_2_label[int(p)]
        #     fullname = sfu.splits_names["test"][k]
        #     splt = fullname.split("/")
        #     filename = splt[-1].split(".")[0]
        #     domain = splt[-2]
        #     pred.append((filename, domain, p))
        # print()

        # # print challenge predictions to check
        # prediction_dir = os.path.join("predictions", "SFU", aux_task)
        # os.makedirs(prediction_dir, exist_ok=True)
        # with open(os.path.join(prediction_dir, "run{0}_pred.txt".format(run)), "w") as out:
        #     for filename, domain, p in pred:
        #         out.write("{0}\t{1}\t{2}\n".format(filename, domain, p))

        prediction_dir = os.path.join("predictions", "SFU", aux_task)
        os.makedirs(prediction_dir, exist_ok=True)
        with open(os.path.join(prediction_dir, "run{0}_pred.txt".format(run)), "w") as out:
            for p in preds:
                out.write("{0}\n".format(p[0]))

        with open(os.path.join(prediction_dir, "gold.txt"), "w") as out:
            for line in maintask_devY:
                out.write("{0}\n".format(line[0]))



