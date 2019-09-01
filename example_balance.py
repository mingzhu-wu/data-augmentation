import pandas
import gzip
import json
import random
import copy

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
        # exlcude fload nan values
        if row[0] == row[0]:
            dd[row[0]] = {"SQuAD": row[1:4], "HotpotQA": row[4:7], "NewsQA": row[7:10], "TriviaQA": row[10:13], \
                          "SearchQA": row[13:16], "NaturalQuestions": row[16:19]}
    return dd


def check_under_present_set(distri):
    number_to_increase = 0
    dataset_need_to_decrease = 0
    for dataset, number_list in distri.items():
        if number_list[INDEX_DIFF] < 0:
            # if this type of questions are under presented in this data set, keep the original number
            # and reduce less in the over presented data set
            number_list[INDEX_NEW] = number_list[INDEX_OLD]
            number_to_increase += abs(number_list[INDEX_DIFF])
            number_list[INDEX_DIFF] = 0
        else:
            dataset_need_to_decrease += 1
    return number_to_increase, dataset_need_to_decrease


def update_set_number(lst, delta):
    lst[INDEX_NEW] += delta
    lst[INDEX_DIFF] -= delta


def balance_examples(balance_number, dataset_number, distri):
    # sort the distribution using the last item of the value
    for dataset, number_list in sorted(distri.items(), key=lambda d: d[1][INDEX_DIFF], reverse=True):
        if number_list[INDEX_DIFF] <= 0:
            continue
        # if this dataset is way more over presented, just reduce the question type in this data set
        if number_list[INDEX_DIFF] > balance_number * dataset_number:
            update_set_number(number_list, balance_number)
            balance_number = 0
            break
        # evenly increase
        else:
            part = round(balance_number / dataset_number)
            if number_list[INDEX_DIFF] > part:
                update_set_number(number_list, part)
                balance_number -= part
            else:
                dataset_number -= 1

    if balance_number > 0:
        balance_examples(balance_number, dataset_number, distri)


# get the number of examples need to be reduced in each dataset
def get_distribute_per_dataset(ds, name):
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
            # a list of dict, qa is a dict, the latter is a list
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

            # a list of dict, qa is a dict, the latter is a list
            for qa in example["qas"]:
                if random.randint(0, total_examples) <= balanced_examples and deleted_number < balanced_examples:
                    qas.remove(qa)
                    deleted_number += 1

            # if not all the questions in this example are deleted
            if qas:
                example['qas'] = qas
                f_out.write(json.dumps(example) + "\n")
        print(deleted_number, balanced_examples)


if __name__ == '__main__':
    ds = get_statistics()
    output = {}
    for q_type, distribute in ds.items():
        under_presented_examples, over_presented_set_numbers = check_under_present_set(distribute)
        balance_examples(under_presented_examples, over_presented_set_numbers, distribute)

    for q_type, new_distribute in ds.items():
        output.setdefault(q_type, [])
        for num_list in new_distribute.values():
            output[q_type].extend(num_list)

    columns = 6 * ["old", "final", "diff"]
    new_df = pandas.DataFrame.from_dict(output, orient="index", columns=columns)
    new_df.to_csv("output.csv", sep="\t")

    print(ds)
    datasets = {"SQuAD": 86588, "HotpotQA": 72928, "NewsQA": 74160, "TriviaQA": 61688, "SearchQA": 117384, "NaturalQuestions": 104071}
    for set_name, total in datasets.items():
        set_dis = get_distribute_per_dataset(ds, set_name)
        number_reduced = sum(set_dis.values())
        #generate_balance_set(set_name, set_dis, "balanced")
        generate_random_set(set_name, total, number_reduced, "random")



















