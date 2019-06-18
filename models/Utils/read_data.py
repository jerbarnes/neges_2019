from xml.etree import ElementTree as ET
import os
import re

"""
Script to read tokens, negative expressions, and
negation scopes from XML files.
"""


def proc_negexp(neg_exp, sents, file, i):
    negexp = []
    for s in neg_exp:
        if s.tag == "event":
            for ss in s:
                token = ss.get("wd")
                if token != None:
                    sents[file][i]["tokens"].append(token + "_NEG")
                    #sents[file][i]["negs"].append(token + "_NEG")
                    negexp.append(token)
        else:
            token = s.get("wd")
            if token != None:
                sents[file][i]["tokens"].append(token + "_NEG")
                #sents[file][i]["negs"].append(token + "_NEG")
                negexp.append(token)
    return negexp

def proc_event(event, sents, file, i):
    for ss in event:
        if ss.tag == "neg_structure":
            proc_neg_structure(ss, sents, file, i)
        if ss.tag == "negexp":
            proc_negexp(ss, sents, file, i)
        else:
            token = ss.get("wd")
            if token != None:
                sents[file][i]["tokens"].append(token)
                #sents[file][i]["negs"].append(token)

def proc_scope(scope, sents, file, i):
    for s in scope:
        if s.tag not in ["negexp", "neg_structure", "event"]:
            token = s.get("wd")
            if token != None:
                sents[file][i]["tokens"].append(token)
                #sents[file][i]["negs"].append(token)
        if s.tag == "negexp":
            proc_negexp(s, sents, file, i)
        if s.tag == "event":
            proc_event(s, sents, file, i)
        if s.tag == "neg_structure":
            proc_neg_structure(s, sents, file, i)
    #sents[file][i]["negs"].append(")")
    sents[file][i]["tokens"].append(")))")


def proc_other(element, sents, file, i):
    token = element.get("wd")
    if token != None:
        sents[file][i]["tokens"].append(token)

def proc_neg_structure(neg_structure, sents, file, i):
    sents[file][i]["tokens"].append("(((")
    attrib = neg_structure.attrib
    attrib_text = ""
    if "polarity" in attrib:
        attrib_text += "polarity=" + attrib["polarity"] + "|"
    else:
        attrib_text += "Polarity=NONE|"

    if "change" in attrib:
        attrib_text += "change=" + attrib["change"] + "|"
    else:
        attrib_text += "change=None|"

    if "value" in attrib:
        attrib_text += "value=" + attrib["value"] + "|"
    else:
        attrib_text += "value=None|"

    if "polarity_modifier" in attrib:
        attrib_text += "polarity_modifier=" + attrib["polarity_modifier"] + "|"
    else:
        attrib_text += "polarity_modifier=None|"

    sents[file][i]["tokens"].append(attrib_text)

    #sents[file][i]["tokens"].append("(")
    for subelem in neg_structure:
        if subelem.tag == "neg_structure":
            proc_neg_structure(subelem, sents, file, i)
        if subelem.tag == "scope":
            proc_scope(subelem, sents, file, i)
        if subelem.tag == "negexp":
            proc_negexp(subelem, sents, file, i)
        if subelem.tag == "event":
            proc_event(subelem, sents, file, i)
        if subelem.tag not in ["negexp", "event", "neg_structure", "scope"]:
            token = subelem.get("wd")
            sents[file][i]["tokens"].append(token)


def read_file(file, sents, train=True):
    #print(file)
    tree = ET.parse(file)
    root = tree.getroot()

    if train:
        polarity = root.attrib["polarity"]
    else:
        polarity = None

    sents[file] = {}
    for i, sent in enumerate(root):
        sents[file][i] = {}
        sents[file][i]["tokens"] = []
        #sents[file][i]["negs"] = []
        #sents[file][i]["scopes"] = []
        for elem in sent:
            d[elem.tag](elem, sents, file, i)
    return sents, polarity

def read_dir(DIR, sents, train=True):
    polarities = []
    for file in os.listdir(DIR):
        sents, polarity = read_file(os.path.join(DIR, file), sents, train=train)
        polarities.append(polarity)

    return sents, polarities

def get_dataset(base_dir, train=True):

    filenames = []
    dataset = []
    polarities = []
    for DIR in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, DIR)):
            sents = {}
            sents, pol = read_dir(os.path.join(base_dir, DIR), sents, train=train)

            for name, review in sents.items():
                review_text = []
                for sent in review.values():
                    review_text.append(" ".join(sent["tokens"]))
                dataset.append(review_text)
                filenames.append(name)
            polarities.extend(pol)

    return filenames, dataset, polarities

def scope_bio(sent):
    bio = []
    in_scope = False
    first = True
    for w in sent.split():
        if w == "(((":
            continue
        if w == ")))":
            in_scope = False
            continue
        if w.startswith("polarity="):
            in_scope = True
            first = True
            continue
        if in_scope == True:
            if first == True:
                #bio.append(w + "_B")
                bio.append("B")
                first = False
            else:
                #bio.append(w + "_I")
                bio.append("I")
        else:
            #bio.append(w + "_O")
            bio.append("O")
    return bio

def relevant_negation_tags(sent):
    if "change=yes" in sent:
        return 1
    else:
        return 0

def clean_sent(sent):
    sent = re.sub("_NEG", "", sent)
    return re.sub("\)\)\)", "", re.sub("\(\(\( polarity=[A-Za-z]+\|change=[A-Za-z]+\|value=[A-Za-z]+\|polarity_modifier=[A-Za-z]+\|", "", sent)).split()


#
others = ['c', 'a', 'f', 'd', 'n', 'p', 'v', 's', 'r', 'z', 'i', 'w', 'word']

d = dict()
d["neg_structure"] = proc_neg_structure
d["event"] = proc_event
d["negexp"] = proc_negexp
d["scope"] = proc_negexp
for o in others:
    d[o] = proc_other

if __name__ == "__main__":

    filenames, dataset, polarities = get_dataset("../../data/train")
    scopes = [[scope_bio(s) for s in review] for review in dataset]
    relev = [[relevant_negation_tags(s) for s in review] for review in dataset]
    sents = [[clean_sent(s) for s in review] for review in dataset]



    # dev_dataset, dev_polarities = get_dataset("subtaskB_training_and_dev_sets/corpus_SFU_Review_SP_NEG_subtaskB/dev")
    # dev_scopes = [[scope_bio(s) for s in review] for review in dev_dataset]
    # dev_relev = [[relevant_negation_tags(s) for s in review] for review in dev_dataset]
    # dev_sents = [[clean_sent(s) for s in review] for review in dev_dataset]

