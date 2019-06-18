from collections import Counter

def majority_voting(runs):
    num_ex = len(runs[0])
    final_preds = []
    for i in range(num_ex):
        preds = []
        for run in runs:
            num, domain, pred = run[i].strip().split("\t")
            preds.append(pred)
        c = Counter(preds)
        maj, _ = c.most_common(1)[0]
        final_preds.append(num + "\t" + domain + "\t" + maj + "\n")
    return final_preds

runs = [open("predictions/SFU/negation_scope/run{0}_pred.txt".format(i)).readlines() for i in range(1,6)]

final_preds = majority_voting(runs)

with open("predictions/SFU/negation_scope/LTG-OSLO_subtaskB.txt", "w") as out:
    for l in final_preds:
        out.write(l)
