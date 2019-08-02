import gzip
import sys
import json
import data_augmentation
from termcolor import colored
import multiprocessing
import nltk
from stanfordcorenlp import StanfordCoreNLP

UKP_SERVER_NED = "http://ned.ukp.informatik.tu-darmstadt.de"


def analyse_none_ne_answer(answer):
    try:
        ans_tree = nltk.tree.ParentedTree.fromstring(StanfordCoreNLP(UKP_SERVER_NED, 9000).parse(answer))
        ans_type = ans_tree[0].label()
    except BaseException as e:
        print(e)
        ans_type = ""
    return ans_type


def analyse_one_example(example, index):
    print(colored(("example " + str(index)), 'yellow'))
    number_of_answers = 0
    ner_type_count = {"None-NER": {}}
    number_of_answer_not_ne = 0
    len_ans_dict = {}

    context = example["context"]
    ner_context = data_augmentation.get_ner_spacy_stanford(context)
    ner_context_lowercase = {k.lower(): v for k, v in ner_context.items()}
    for qa in example["qas"]:
        for answer in qa["answers"]:
            number_of_answers += 1
            len_ans_dict.setdefault(len(answer.split()), 0)
            len_ans_dict[len(answer.split())] += 1
            # if answer.startswith("The "):
            #    answer = answer.replace("The ", "the ")

            if answer.lower() in ner_context_lowercase.keys():
                ner_type_count.setdefault(ner_context_lowercase[answer.lower()], 0)
                ner_type_count[ner_context_lowercase[answer.lower()]] += 1
            else:
                number_of_answer_not_ne += 1
                ans_type = analyse_none_ne_answer(answer).lower()
                ner_type_count["None-NER"].setdefault(ans_type, 0)
                ner_type_count["None-NER"][ans_type] += 1

    return number_of_answers, ner_type_count, number_of_answer_not_ne, len_ans_dict


if __name__ == '__main__':
    sum_number_of_answers = 0
    sum_number_of_answer_not_ne = 0
    sum_type_count = {}
    sum_type_count_none_ne = {}
    sum_len_answer = {}
    none_ne_answers = []
    i = 0
    res = []
    pool = multiprocessing.Pool(processes=data_augmentation.ALLOWED_PARALLEL_PROCESS)
    with gzip.open(sys.argv[1], 'rt') as f_src:
        data = [json.loads(line) for line in f_src]
        print(len(data))
        for example in data[1:]:
            i += 1
            res.append(pool.apply_async(analyse_one_example, (example, i, )))
        pool.close()
        pool.join()

    for r in res:
        num_ans, type_count, num_not_ne, len_answer_dict = r.get(timeout=1)
        sum_number_of_answers += num_ans
        sum_number_of_answer_not_ne += num_not_ne
        for len, c in len_answer_dict.items():
            sum_len_answer.setdefault(len, 0)
            sum_len_answer[len] += c
        for k, v in type_count.items():
            if k != "None-NER":
                sum_type_count.setdefault(k, 0)
                sum_type_count[k] += v
            else:
                for k1, v1 in v.items():
                    sum_type_count_none_ne.setdefault(k1, 0)
                    sum_type_count_none_ne[k1] += v1

    print("number of all answers and non named entity answers", sum_number_of_answers, sum_number_of_answer_not_ne)
    print("Ratio of non-named entities in answer: %.2f%%" % (round(sum_number_of_answer_not_ne/sum_number_of_answers, 3)*100))

    separate_ratio = {k: round(v/sum_number_of_answers, 3) for k, v in sum_type_count.items()}
    [print("Ratio of "+k+" in answers is: %.2f%%" % (v*100)) for k, v in sorted(separate_ratio.items(), \
                                                               key=lambda d: d[1], reverse=True)]

    separate_ratio1 = {k1: round(v1/sum_number_of_answers, 3) for k1, v1 in sum_type_count_none_ne.items()}
    [print("Ratio of "+k1+" in answers is: %.2f%%" % (v1*100)) for k1, v1 in sorted(separate_ratio1.items(), \
                                                                    key=lambda d: d[1], reverse=True)]

    len_ratio = {l: round(c/sum_number_of_answers, 3) for l, c in sum_len_answer.items()}
    [print("number of answers with length "+str(k)+" is:  %.2f%%" % (v*100)) for k, v in len_ratio.items()]







