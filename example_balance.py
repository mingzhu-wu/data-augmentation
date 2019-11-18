import copy
import gzip
import json
import pandas
import random

INDEX_OLD = 0
INDEX_NEW = 1
INDEX_DIFF = 2
WH_WORDS = ["what", "which", "who", "whom", "whose", "where", "why", "how", "when"]
HOW_QUESTIONS = ["how far", "how long", "how many", "how much", "how old"]


def get_statistics():
    df = pandas.read_excel("balance.xlsx")
    dd = {}
    for row in df.to_records(index=False).tolist():
        row = list(row)
        # exclude float nan values
        if row[0] == row[0]:
            dd[row[0]] = {"SQuAD": row[1:4], "HotpotQA": row[4:7], "NewsQA": row[7:10], "TriviaQA": row[10:13], \
                          "SearchQA": row[13:16], "NaturalQuestions": row[16:19]}
    return dd


# get the number of examples need to be reduced in each dataset
def get_diff_distribute_per_dataset(ds, name):
    dis = {}
    for q_type, distribute in ds.items():
        dis[q_type] = distribute[name][INDEX_DIFF]

    return dis


def get_question_type(question="", question_tokens=[]):
    qs_type = "others"
    for key_word in WH_WORDS:
        if key_word in question_tokens:
            qs_type = key_word
            if key_word == "how":
                for how_word in HOW_QUESTIONS:
                    if how_word in question.lower():
                        qs_type = how_word
            break
    return qs_type


def generate_balance_set(name, distri, folder):
    print(folder, name)
    with gzip.open("data_train/" + name + ".jsonl.gz", 'rt') as f_in, \
            gzip.open("data_train/"+folder + "/" + name + ".jsonl.gz", "wt") as f_out:
        f_out.write(next(f_in)) # header
        lines = f_in.readlines()
        random.shuffle(lines)
        for line in lines:
            example = json.loads(line.strip())
            qas = copy.deepcopy(example['qas'])
            for qa in example["qas"]:
                question = qa["question"]
                question_tokens = [token[0].lower() for token in qa["question_tokens"]]
                qa_type = get_question_type(question.lower(), question_tokens)

                if distri[qa_type] > 0:
                    qas.remove(qa)
                    distri[qa_type] -= 1

            # if not all the questions in this example are deleted
            if qas:
                example['qas'] = qas
                f_out.write(json.dumps(example) + "\n")


def generate_random_set(name, total_examples, balanced_examples, folder):
    print(folder, name)
    deleted_number = 0
    with gzip.open("data_train/"+ name + ".jsonl.gz", 'rt') as f_in, \
            gzip.open("data_train/" + folder + "/" + name + ".jsonl.gz", "wt") as f_out:
        f_out.write(next(f_in)) # header
        lines = f_in.readlines()
        random.shuffle(lines)
        for line in lines:
            example = json.loads(line.strip())
            qas = copy.deepcopy(example['qas'])

            for qa in example["qas"]:
                if random.randint(0, total_examples) <= balanced_examples+1000 and deleted_number < balanced_examples:
                    qas.remove(qa)
                    deleted_number += 1

            # if not all the questions in this example are deleted
            if qas:
                example['qas'] = qas
                f_out.write(json.dumps(example) + "\n")
        print(deleted_number, balanced_examples)


if __name__ == '__main__':
    ds = get_statistics()
    old_total = 0
    removed_total = 0
    for q_type, type_distribute in ds.items():
        for name, set_distribute in type_distribute.items():
            old_total += set_distribute[INDEX_OLD]
            removed_total += set_distribute[INDEX_DIFF]

    print(ds, old_total, removed_total)
    datasets = {"SQuAD": 86588, "HotpotQA": 72928, "NewsQA": 74160, "TriviaQA": 61688, "SearchQA": 117384, "NaturalQuestions": 104071}
    for set_name, total in datasets.items():
        set_dis = get_diff_distribute_per_dataset(ds, set_name)
        print(set_dis)
        # this line shows how I get the reduced number for randomly remove
        # number_reduced = sum(set_dis.values())
        generate_balance_set(set_name, set_dis, "balanced")
        generate_random_set(set_name, total, round(removed_total/old_total*total), "random")


