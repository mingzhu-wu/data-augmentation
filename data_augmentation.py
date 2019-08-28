import json
import multiprocessing
import spacy
import sys
import gzip
from termcolor import colored
import text_analyse

ALLOWED_PARALLEL_PROCESS = 8
UKP_SERVER = 'http://krusty.ukp.informatik.tu-darmstadt.de'
UKP_SERVER_NED = "http://ned.ukp.informatik.tu-darmstadt.de"
LOCALHOST = "http://localhost"
nlp = spacy.load('en_core_web_sm')


def process_example(example_dict, doc_index):
    print(colored(("example " + str(doc_index)), 'yellow'))
    context = example_dict['context']
    ner_dict = text_analyse.get_ner_spacy_stanford(context)

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

