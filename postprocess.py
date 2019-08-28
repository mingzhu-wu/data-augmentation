import pandas
import sys
import time
import text_analyse


def process_gold_type(df):
    gold_type = df["gold-type"].tolist()
    f1 = df["f_score"].tolist()
    local_counter = {}

    for i in range(0, len(df)):
        local_counter.setdefault(gold_type[i], {})
        local_counter[gold_type[i]]["total"] = local_counter[gold_type[i]].get("total", 0) + 1
        local_counter[gold_type[i]]["F1"] = local_counter[gold_type[i]].get("F1", 0) + f1[i]
    return local_counter


def process_gold_length(df):
    gold_ans = df["gold"].tolist()
    f1 = df["f_score"].tolist()
    counter = {}

    for i in range(0, len(df)):
        gold_len = len(gold_ans[i].split(";")[0].split())
        counter.setdefault(gold_len, {})
        counter[gold_len]["total"] = counter[gold_len].get("total", 0) + 1
        counter[gold_len]["F1"] = counter[gold_len].get("F1", 0) + f1[i]

    return counter


def process_question_type(df):
    q_counter = {}
    questions = df["question"].tolist()
    f1 = df["f_score"].tolist()
    for i in range(0, len(df)):
        q_type = text_analyse.get_question_type(questions[i])
        q_counter.setdefault(q_type, {})
        q_counter[q_type]["total"] = q_counter[q_type].get("total", 0) + 1
        q_counter[q_type]["F1"] = q_counter[q_type].get("F1", 0) + f1[i]

    print(time.process_time() - start, q_counter)
    return q_counter


def counter_to_table(counter, table, i, ex_number):
    for k, v in counter.items():
        table.setdefault(k, 2 * i * [0])
        f1_score = v.get("F1", 0)
        table[k].extend(
            ["{:.2f}".format(v["total"] / ex_number * 100), "{:.2f}".format(f1_score / v["total"] * 100)])
    # fill empty field with 0
    for n in table.keys() - counter.keys():
        table[n].extend([0, 0])


if __name__ == "__main__":
    datasets = ["SQuAD", "HotpotQA", "NewsQA", "TriviaQA-web", "SearchQA", "NaturalQuestionsShort"]
    out_domain_datasets = ["DROP", "RACE", "BioASQ", "TextbookQA", "RelationExtraction", "DuoRC"]
    datasets.extend(out_domain_datasets)
    columns = 12 * ["Percentage", "F1"]
    ans_type_table = {}
    len_table = {}
    question_table = {}
    i = 0
    for dataset in datasets:
        print(dataset)
        df = pandas.read_csv("../predicts-pretrained/extended-logs/pred-"+dataset+".csv", sep="\t")
        number_examples = len(df)
        #df = pandas.read_csv("../predicts-BERTLarge/Logs/pred-"+dataset+".csv", sep="\t")
        counter_to_table(process_gold_type(df), ans_type_table, i, number_examples)
        start = time.process_time()
        counter_to_table(process_gold_length(df), len_table, i, number_examples)
        counter_to_table(process_question_type(df), question_table, i, number_examples)
        i += 1

    new_df = pandas.DataFrame.from_dict(ans_type_table, orient="index", columns=columns)
    new_df.to_csv(sys.argv[1]+"-type.csv", sep="\t")

    len_df = pandas.DataFrame.from_dict(len_table, orient="index", columns=columns)
    len_df.to_csv(sys.argv[1]+"-len.csv", sep="\t")

    len_df = pandas.DataFrame.from_dict(question_table, orient="index", columns=columns)
    len_df.to_csv(sys.argv[1]+"-question.csv", sep="\t")
