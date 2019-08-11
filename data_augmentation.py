import json
import multiprocessing
import spacy
import nltk
import sys
import gzip
from termcolor import colored
from stanfordcorenlp import StanfordCoreNLP

ALLOWED_PARALLEL_PROCESS = 8
UKP_SERVER = 'http://krusty.ukp.informatik.tu-darmstadt.de'
UKP_SERVER_NED = "http://ned.ukp.informatik.tu-darmstadt.de"
LOCALHOST = "http://localhost"
nlp = spacy.load('en_core_web_sm')


def spacy_ents(text):
    ent_list = []
    doc = nlp(text)

    for ent in doc.ents:
        ent_list.append((ent.text, ent.label_))
    return ent_list


def get_ner_doc(doc):
    sta_nlp = StanfordCoreNLP(UKP_SERVER_NED, 9000)
    ner_token_list = []
    try:
        ner_token_list.extend(sta_nlp.ner(doc))
    except BaseException as e:
        ss = nltk.tokenize.sent_tokenize(doc.replace("\n", ' ').replace('\xa0', ' '))
        part_doc1 = " ".join(ss[:int(len(ss)/2)])
        part_doc2 = " ".join(ss[int(len(ss)/2):])
        ner_token_list.extend(get_ner_doc(part_doc1))
        ner_token_list.extend(get_ner_doc(part_doc2))

    return ner_token_list


# merge the tokens in standford corenlp ner result to named entities.
def get_merged_ner(orig_list):
    merged_ner_list = []
    for i in range(len(orig_list)):
        en = orig_list[i]
        current_index = i
        # merge entity_list next to each other with the same NER
        if en[1] != 'O':
            merged_ner_list.append(en)
            if current_index > 0:
                former_ne = orig_list[current_index - 1]
                if en[1] == former_ne[1]:
                    orig_list[current_index] = (former_ne[0] + ' ' + en[0], en[1])
                    merged_ner_list.append(orig_list[current_index])
                    merged_ner_list.remove(en)
                    merged_ner_list.remove(former_ne)
                elif former_ne[0].lower() == "the":
                    orig_list[current_index] = ("the" + ' ' + en[0], en[1])
                    merged_ner_list.append(orig_list[current_index])
                    merged_ner_list.remove(en)
                else:
                    continue

    return merged_ner_list


def is_valid_ne(ner):
    special_symbol = ['(', ')', '[', '±', ']', '+', '\xa0', '&', '\n', '-RRB-', '-LRB-', '-']
    if ner[0].startswith('A '):
        return False
    if any(sb in ner[0] for sb in special_symbol):
        return False
    if ner[0] == '\n' or ner[0] == ' ':
        return False
    #if ner[1] in ('NUMBER', 'CARDINAL', 'PERCENT', 'ORDINAL', 'DATE', 'TIME', 'MONEY', 'QUANTITY'):
    # if ner[1] == 'CARDINAL':
    #     return False
    else:
        return True


# write the named entities recognized by stanford corenlp to the record.
def write_ner_coren(nerlist, ner_dict):
    for ner in nerlist:
        if is_valid_ne(ner):
            ne_name = ner[0].replace("The ", "the") if ner[0].startswith("The ") else ner[0]
            ner_dict.setdefault(ne_name, ner[1])


def get_ner_spacy_stanford(context):
    spa_ner_list = spacy_ents(context)
    ner_dict = {}

    try:
        sta_ner_token_list = get_ner_doc(context)
    except BaseException as e:
        print("context is too long to get stanford named entities")
        sta_ner_token_list = []

    sta_ner_list = get_merged_ner(sta_ner_token_list)
    write_ner_coren(sta_ner_list, ner_dict)
    write_ner_coren(spa_ner_list, ner_dict)
    #print(ner_dict)

    return ner_dict


def process_example(example_dict, doc_index):
    print(colored(("example " + str(doc_index)), 'yellow'))
    context = example_dict['context']
    ner_dict = get_ner_spacy_stanford(context)

    augment_context = " [EST] "
    for ne_name in ner_dict.keys():
        augment_context += ne_name + " [ESP] "

    augment_context = augment_context[:-7] + " [EED]"
    example_dict['context'] = context + augment_context
    return example_dict


if __name__ == '__main__':
    src_path = sys.argv[1]
    res = []
    pool = multiprocessing.Pool(processes=ALLOWED_PARALLEL_PROCESS)
    doc_index = 0
    with gzip.open(src_path, 'rt') as f_src, open(src_path+"_arg", 'w') as f_out:
        json.dump(next(f_out))
        for line in f_src:
            example = json.loads(line)
            res.append(process_example(example, doc_index))
            #res.append(pool.apply_async(process_example, (example, doc_index, )))
            doc_index += 1
        # pool.close()
        # pool.join()
   # print("the end!")

#    with gzip.open(src_path+'_aug', 'wt') as f_out:
        f_out.write("\n")
        for new_example in res:
            json.dump(new_example, f_out)
            #json.dump(new_example.get(timeout=1), fw)
            f_out.write('\n')

