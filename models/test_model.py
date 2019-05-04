import argparse
import pickle

from hard_parameter_bilstm_crf import *
from bilstm_crf import *
from hierarchical_model import *
from Utils.sst import *

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


    f1s = []
    accs = []
    preds = []
    ys = []

    print("opening model params...")
    with open(os.path.join("saved_models",
                           "SST-{0}".format(FINE_GRAINED),
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

    datadir = "../data/datasets/en/sst-{0}".format(args.FINE_GRAINED)
    sst = SSTDataset(vocab, False, datadir)
    maintask_test_iter = sst.get_split("test")

    chfile = "../data/challenge_dataset/sst-{0}.txt".format(args.FINE_GRAINED)
    challenge_dataset = ChallengeDataset(vocab, False, chfile)
    challenge_test = challenge_dataset.get_split()

    new_matrix = np.zeros(matrix_shape)


    print("finding best weights for runs 1 - {0}".format(num_runs))
    for i in range(num_runs):
        run = i + 1
        weight_dir = os.path.join("saved_models",
                                  "SST-{0}".format(FINE_GRAINED),
                                  aux_task,
                                  str(run))
        best_acc, (epochs, lstm_dim, lstm_layers), best_weights =\
                                                   get_best_run(weight_dir)

        model = Hierarchical_Model(vocab,
                                   new_matrix,
                                   tag_to_ix,
                                   len_labels,
                                   task2label2id,
                                   300,
                                   lstm_dim,
                                   1,
                                   train_embeddings=True)



        model.load_state_dict(torch.load(best_weights))
        model.eval()

        print("Run {0}".format(run))
        f1, acc, pred, y = model.eval_sent(maintask_test_iter, batch_size=50)
        print()

        f1s.append(f1)
        accs.append(acc)
        preds.append(pred)
        ys.append(y)

        print("Eval on challenge data")
        chf1, chacc, chpred, chy = model.eval_sent(challenge_test, batch_size=1)
        print()

        # print challenge predictions to check
        prediction_dir = os.path.join("predictions", "SST-{0}".format(FINE_GRAINED), aux_task)
        os.makedirs(prediction_dir, exist_ok=True)
        with open(os.path.join(prediction_dir, "run{0}_challenge_pred.txt".format(run)), "w") as out:
            for line in chpred:
                out.write("{0}\n".format(line))
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    print("#"*20 + "FINAL" + "#"*20)

    if metric == "f1":
        print("MEAN F1: {0:.3f}".format(mean_f1))
        print("STD F1: {0:.3f}".format(std_f1))

    if metric == "acc":
        print("MEAN ACC: {0:.2f} ({1:.1f})".format(mean_acc * 100, std_acc * 100))
    return f1s, accs, preds, ys


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--AUXILIARY_TASK", "-aux", default="negation_scope")
    parser.add_argument("--NUM_RUNS", "-nr", default=5, type=int)
    parser.add_argument("--METRIC", "-m", default="acc")
    parser.add_argument("--FINE_GRAINED", "-fg",
                        default="fine",
                        help="Either 'fine' or 'binary' (defaults to 'fine'.")
    args = parser.parse_args()

    f1s, accs, preds, ys = test_model(args.AUXILIARY_TASK,
                                      num_runs=args.NUM_RUNS,
                                      metric=args.METRIC,
                                      FINE_GRAINED=args.FINE_GRAINED)

