import pandas
import sys


def process_gold_type(df):
    gold_type = df["gold-type"].tolist()
    f1 = df["f_score"].tolist()
    local_counter = {}

    for i in range(0, len(df)):
        local_counter.setdefault(gold_type[i], {})
        local_counter[gold_type[i]]["total"] = local_counter[gold_type[i]].get("total", 0) + 1
        f1[i] = "positive" if f1[i] >= 0.5 else "negative"
        local_counter[gold_type[i]][f1[i]] = local_counter[gold_type[i]].get(f1[i], 0) + 1
    print(local_counter)
    return local_counter


def process_gold_length(df):
    gold_ans = df["gold"].tolist()
    f1 = df["f_score"].tolist()
    counter = {}

    for i in range(0, len(df)):
        gold_len = len(gold_ans[i].split(";")[0].split())
        counter.setdefault(gold_len, {})
        counter[gold_len]["total"] = counter[gold_len].get("total", 0) + 1
        f1[i] = "positive" if f1[i] >= 0.5 else "negative"
        counter[gold_len][f1[i]] = counter[gold_len].get(f1[i], 0) + 1
    return counter


def counter_to_table(counter, table):
    for k, v in counter.items():
        table.setdefault(k, 2 * i * [0])
        true_number = v.get("positive", 0)
        table[k].extend(
            ["{:.2f}".format(v["total"] / len(df) * 100), "{:.2f}".format(true_number / v["total"] * 100)])
    # fill empty field with 0
    for n in table.keys() - counter.keys():
        table[n].extend([0, 0])


if __name__ == "__main__":
    datasets = ["SQuAD", "HotpotQA", "NewsQA", "TriviaQA-web", "SearchQA", "NaturalQuestionsShort"]
    out_domain_datasets = ["DROP", "RACE", "BioASQ", "TextbookQA", "RelationExtraction", "DuoRC"]
    datasets.extend(out_domain_datasets)
    columns = 12 * ["Percentage", "F1>0.5"]
    final_table = {}
    len_table = {}
    i = 0
    for dataset in datasets:
        df = pandas.read_csv("../predicts-pretrained/extended-logs/pred-"+dataset+".csv", sep="\t")
        type_counter = process_gold_type(df)
        counter_to_table(type_counter, final_table)
        counter_to_table(process_gold_length(df), len_table)
        i += 1
    #print(final_table)

    new_df = pandas.DataFrame.from_dict(final_table, orient="index", columns=columns)
    new_df.to_csv(sys.argv[1]+"-type.csv")

    len_df = pandas.DataFrame.from_dict(len_table, orient="index", columns=columns)
    len_df.to_csv(sys.argv[1]+"-len.csv")
