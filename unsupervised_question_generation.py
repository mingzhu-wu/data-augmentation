import spacy
import nltk
from allennlp.predictors.predictor import Predictor
import data_augmentation
import multiprocessing
import json
import sys
import gzip
from nltk.tokenize.treebank import TreebankWordDetokenizer
from termcolor import colored

predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
answer_categories = {"GPE": "PLACE", "LOC": "PLACE", "FAC": "PLACE", }


# High Level Answer Category
def get_mask(label):
    if label in ['PERSON', 'NORP', 'ORG']:
        mask = 'PERSON/NORP/ORG'
    elif label in ['GPE', 'LOC', 'FAC']:
        mask = 'PLACE'
    elif label in ['PRODUCT', 'EVENT', 'WORKOFART', 'LAW', 'LANGUAGE']:
        mask = 'THING'
    elif label in ['TIME', 'DATA']:
        mask = 'TEMPORAL'
    elif label in ['PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
        mask = 'NUMERIC'
    else:
        mask = label
    return mask


def find_clause_contain_ne(tree, ne):
    clause = " ".join(tree.leaves())
    for st in tree.subtrees():
        if st.label() == "S" or st.label() == 'SBAR':
            # find the most recent and only one clause for each ne, if one ne appears in two clauses, only keep one.
            if ne in " ".join(st.leaves()):
                clause = TreebankWordDetokenizer().detokenize(tree.leaves())
    return clause


def generate_qa_pairs(sentence):
    cloze_qas = []
    ne_list = data_augmentation.spacy_ents(sentence)
    # if no named entity found in this sentence, just return
    if not ne_list:
        return []

    parses = predictor.predict(sentence)["trees"]
    tree = nltk.tree.ParentedTree.fromstring(parses)
    for ne in ne_list:
        clause = find_clause_contain_ne(tree, ne[0]).replace(ne[0], get_mask(ne[1]))
        cloze_qas.append((clause, ne[0]))
    return cloze_qas


def process_context(context, id):
    print(colored(("example " + str(id)), 'yellow'))
    ss = nltk.tokenize.sent_tokenize(context)
    cloze_qas_context = []
    for sent in ss:
        cloze_qas_context.extend(generate_qa_pairs(sent))
    with open("context_"+str(id)+".txt", 'w') as f_cloze:
        json.dump({"id":id, "context": context, "cloze_qas": cloze_qas_context}, f_cloze, indent=4, ensure_ascii=False)


if __name__ == '__main__':

    res = []
    pool = multiprocessing.Pool(processes=data_augmentation.ALLOWED_PARALLEL_PROCESS)
    context_index = 0
    with gzip.open(sys.argv[1], 'rt') as f_src:
        next(f_src)
        for line in f_src:
            example = json.loads(line)
            context_index += 1
            process_context(example['context'], context_index)
            break
            #pool.apply_async(process_context, (example['context'], ))

        #pool.close()
        #pool.join()



