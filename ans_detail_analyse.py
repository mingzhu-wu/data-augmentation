import gzip
import sys
import json
from termcolor import colored
import multiprocessing
import text_analyse


def analyse_one_example(example, index):
    print(colored(("example " + str(index)), 'yellow'))
    context = example["context"]
    tag_qid = {}
    ner_context = text_analyse.get_ner_spacy_stanford(context)
    ner_context_lowercase = {k.lower(): v for k, v in ner_context.items()}

    for qa in example["qas"]:
        for answer in qa["answers"]:
            if answer.lower() in ner_context_lowercase.keys():
                tag = ner_context_lowercase[answer.lower()]
            else:
                tag = text_analyse.analyse_none_ne_answer(answer).lower()
            tag_qid.setdefault(tag, [])
            tag_qid[tag].append((qa["qid"], answer))

    return tag_qid


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=text_analyse.ALLOWED_PARALLEL_PROCESS)
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
    print("The end!")





