import json
import spacy
import nltk
from termcolor import colored
from stanfordcorenlp import StanfordCoreNLP

ALLOWED_PARALLEL_PROCESS = 8
UKP_SERVER = 'http://krusty.ukp.informatik.tu-darmstadt.de'
UKP_SERVER_NED = "http://ned.ukp.informatik.tu-darmstadt.de"
LOCALHOST = "http://localhost"
WH_WORDS = ["what", "which", "who", "whom", "whose", "where", "why", "how", "when"]
HOW_QUESTIONS = ["how far", "how long", "how many", "how much", "how old"]
nlp = spacy.load('en_core_web_sm')
sta_nlp = StanfordCoreNLP(UKP_SERVER_NED, 9000)


# get the named entities recognized by SpaCy
def spacy_ents(text):
    ent_list = []
    doc = nlp(text)

    for ent in doc.ents:
        ent_list.append((ent.text, ent.label_))
    return ent_list


# give a doc with multiple sentences, return the token list with NE tags identified by Stanford NER
def get_ner_doc_stanford(context):
    sta_nlp = StanfordCoreNLP(UKP_SERVER_NED, 9000)
    ner_token_list = []
    try:
        ner_token_list.extend(sta_nlp.ner(context))
    except BaseException as e:
        ss = nltk.tokenize.sent_tokenize(context.replace("\n", ' ').replace('\xa0', ' '))
        part_doc1 = " ".join(ss[:int(len(ss)/2)])
        part_doc2 = " ".join(ss[int(len(ss)/2):])
        ner_token_list.extend(get_ner_doc_stanford(part_doc1))
        ner_token_list.extend(get_ner_doc_stanford(part_doc2))

    return ner_token_list


# given a context, return the named entities identified by both SpaCy and Stanford
def get_ner_spacy_stanford(context):
    spa_ner_list = spacy_ents(context)
    ner_dict = {}

    try:
        sta_ner_token_list = get_ner_doc_stanford(context)
    except BaseException as e:
        print("context is too long to get stanford named entities")
        sta_ner_token_list = []

    sta_ner_list = get_merged_ner(sta_ner_token_list)
    write_ner_coren(sta_ner_list, ner_dict)
    write_ner_coren(spa_ner_list, ner_dict)
    # print(ner_dict)

    return ner_dict


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


# write the named entities to the record.
def write_ner_coren(nerlist, ner_dict):
    for ner in nerlist:
        if is_valid_ne(ner):
            ne_name = ner[0].replace("The ", "the") if ner[0].startswith("The ") else ner[0]
            ner_dict.setdefault(ne_name, ner[1])


# check whether a given named entity is valid or not, exclude weird or special symbols
def is_valid_ne(ner):
    special_symbol = ['(', ')', '[', '±', ']', '+', '\xa0', '&', '\n', '-RRB-', '-LRB-', '-']
    if ner[0].startswith('A '):
        return False
    if any(sb in ner[0] for sb in special_symbol):
        return False
    if ner[0] == '\n' or ner[0] == ' ':
        return False
    else:
        return True


# given a answer,
def analyse_non_ne_answer(answer):
    try:
        ans_tree = nltk.tree.ParentedTree.fromstring(StanfordCoreNLP(UKP_SERVER_NED, 9000).parse(answer))
        ans_type = ans_tree[0].label()
    except BaseException as e:
        #raise e
        print(e)
        # if exception or no type found, mark the type as x (unknown)
        ans_type = "x"
    return ans_type


def get_question_type(question="", question_tokens=[]):
    qs_type = "others"
    if not question_tokens:
        question_tokens = sta_nlp.word_tokenize(question.lower())
    for key_word in WH_WORDS:
        if key_word in question_tokens:
            qs_type = key_word
            if key_word == "how":
                for how_word in HOW_QUESTIONS:
                    if how_word in question.lower():
                        qs_type = how_word
            break
    return qs_type


