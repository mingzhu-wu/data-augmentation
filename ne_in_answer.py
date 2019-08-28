import gzip
import sys
import json
from termcolor import colored
import multiprocessing
import pandas
import string
import re
import text_analyse

UKP_SERVER_NED = "http://ned.ukp.informatik.tu-darmstadt.de"


# code from mrqa_official_eval
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def analyse_answer_type(example, index):
    print(colored(("example " + str(index)), 'yellow'))
    number_of_answers = 0
    ner_type_count = {"Non-NER": {}}
    number_of_answer_not_ne = 0
    len_ans_dict = {}

    context = example["context"]
    ner_context = text_analyse.get_ner_spacy_stanford(context)
    # ner_context_lowercase = {k.lower(): v for k, v in ner_context.items()}
    ner_context_normalize = {normalize_answer(k): v for k, v in ner_context.items()}
    for qa in example["qas"]:
        number_of_answers += 1
        len_ans = 0

        if not any(normalize_answer(ans) in ner_context_normalize.keys() for ans in qa["answers"]):
            number_of_answer_not_ne += 1
            ans_type = text_analyse.analyse_non_ne_answer(normalize_answer(qa["answers"][0]))
            ner_type_count["Non-NER"].setdefault(ans_type, 0)
            ner_type_count["Non-NER"][ans_type] += 1
            len_ans += len(qa["answers"][0].split())
        else:
            for answer in qa["answers"]:
                if normalize_answer(answer) in ner_context_normalize.keys():
                    len_ans += len(answer.split())
                    ner_type_count.setdefault(ner_context_normalize[normalize_answer(answer)], 0)
                    ner_type_count[ner_context_normalize[normalize_answer(answer)]] += 1
                    break
        len_ans_dict.setdefault(len_ans, 0)
        len_ans_dict[len_ans] += 1

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
    pool = multiprocessing.Pool(processes=text_analyse.ALLOWED_PARALLEL_PROCESS)
    with gzip.open(sys.argv[1], 'rt') as f_src:
        data = [json.loads(line) for line in f_src]
        print(len(data))
        for example in data[1:]:
            i += 1
            res.append(pool.apply_async(analyse_answer_type, (example, i,)))
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
            if k != "Non-NER":
                sum_type_count.setdefault(k, 0)
                sum_type_count[k] += v
            else:
                for k1, v1 in v.items():
                    sum_type_count_none_ne.setdefault(k1, 0)
                    sum_type_count_none_ne[k1] += v1

    print("number of all answers and non named entity answers", sum_number_of_answers, sum_number_of_answer_not_ne)
    print("Ratio of non-named entities in answer: %.2f%%" % (round(sum_number_of_answer_not_ne/sum_number_of_answers, 3)*100))

    separate_ratio = {k: round(v/sum_number_of_answers *100, 2) for k, v in sorted(sum_type_count.items(), key=lambda d:d[0])}
    separate_ratio["non-named entity"] = round(sum_number_of_answer_not_ne/sum_number_of_answers, 3)*100
    separate_ratio["Total Number of answers"] = sum_number_of_answers
    # [print("Ratio of "+k+" in answers is: %.2f%%" % (v*100)) for k, v in sorted(separate_ratio.items(), \
    #                                                          key=lambda d: d[1], reverse=True)]
    data_name = sys.argv[2]
    writer = pandas.ExcelWriter(data_name+".xlsx", engine='xlsxwriter')

    pandas.Series(separate_ratio).to_frame(data_name).to_excel(writer, sheet_name="NE answers")

    separate_ratio1 = {k1: round(v1/sum_number_of_answers *100, 2) for k1, v1 in sorted(sum_type_count_none_ne.items(), key=lambda d:d[0])}
    # [print("Ratio of "+k1+" in answers is: %.2f%%" % (v1*100)) for k1, v1 in sorted(separate_ratio1.items(), \
    #                                                               key=lambda d: d[1], reverse=True)]

    pandas.Series(separate_ratio1).to_frame(data_name).to_excel(writer, sheet_name="non-NE answers")

    len_ratio = {l: round(c/sum_number_of_answers *100, 2) for l, c in sorted(sum_len_answer.items(), key=lambda d:d[0])}
    #[print("number of answers with length "+str(k)+" is:  %.2f%%" % (v*100)) for k, v in len_ratio.items()]
    #
    pandas.Series(len_ratio).to_frame(data_name).to_excel(writer, sheet_name="length")
    writer.save()
    writer.close()







