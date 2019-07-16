import sys
import json
import os
import gzip
import json



def extract_cloze_questions(path, out_path):
    with open(out_path, 'w') as f_out:
        for file in os.listdir(path):
            print(file)

            with open(path+'/'+file, 'r') as f_cloze:
                for qa in json.load(f_cloze)['cloze_qas']:
                    f_out.write(qa[0])
                    f_out.write('\n')


def extract_natural_questions(src, target):
    with gzip.open(path, 'rt') as f_src, open(target, 'w') as f_out:
        next(f_src)
        for line in f_src:
            for qa in json.loads(line)["qas"]:
                question = qa['question']
                f_out.write(question)
                f_out.write('\n')


if __name__ == '__main__':
    path = sys.argv[1]
    out_path = sys.argv[2]
    if sys.argv[3] == 'cloze':
        extract_cloze_questions(path, out_path)
    if sys.argv[3] == 'natural':
        extract_natural_questions(path, out_path)

