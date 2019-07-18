import gzip
import sys
import json
import data_augmentation
from termcolor import colored
import multiprocessing


def analyse_one_example(example, index):
    print(colored(("example " + str(index)), 'yellow'))
    context = example["context"]
    tag_qid = {}
    ner_context = data_augmentation.get_ner_spacy_stanford(context)
    ner_context_lowercase = {k.lower(): v for k, v in ner_context.items()}

    for qa in example["qas"]:
        for answer in qa["answers"]:
            if answer.lower() in ner_context_lowercase.keys():
                tag = ner_context_lowercase[answer.lower()]
                tag_qid.setdefault(tag, [])
                tag_qid[tag].append((qa["qid"], answer))

    return tag_qid


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=data_augmentation.ALLOWED_PARALLEL_PROCESS)
    i = 0
    res = []
    output_dict = {}
    with gzip.open(sys.argv[1], 'rt') as f_src:
        next(f_src)
        for line in f_src:
            example = json.loads(line)
            i += 1
            res.append(pool.apply_async(analyse_one_example, (example, i, )))
        pool.close()
        pool.join()

    for r in res:
        for k, v in r.get(timeout=1).items():
            output_dict.setdefault(k, [])
            output_dict[k].extend(v)

    with open(sys.argv[2], 'w') as f_out:
        json.dump(output_dict, f_out, indent=4)
    print(output_dict)

#     for r in res:
#         num_ans, type_count, num_not_ne, len_answer_dict = r.get(timeout=1)
#         sum_number_of_answers += num_ans
#         sum_number_of_answer_not_ne += num_not_ne
#         for len, c in len_answer_dict.items():
#             sum_len_answer.setdefault(len, 0)
#             sum_len_answer[len] += c
#         for k, v in type_count.items():
#             sum_type_count.setdefault(k, 0)
#             sum_type_count[k] += v
#
#     separate_ratio = {k: round(v/sum_number_of_answers, 3) for k, v in sum_type_count.items()}
#
#     print("number of all answers and non named entity answers", sum_number_of_answers, sum_number_of_answer_not_ne)
#     print("Ratio of non-named entities in answer: %.2f%%" % (round(sum_number_of_answer_not_ne/sum_number_of_answers, 3)*100))
#     [print("Ratio of "+k+" in answers is: %.2f%%" % (v*100)) for k, v in sorted(separate_ratio.items(), \
#                                                                     key=lambda d: d[1], reverse=True)]
#
#     len_ratio = {l: round(c/sum_number_of_answers, 3) for l, c in sum_len_answer.items()}
#     [print("number of answers with length "+str(k)+" is:  %.2f%%" % (v*100)) for k, v in len_ratio.items()]







