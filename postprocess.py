import pandas
import sys


def process(path):
    df = pandas.read_csv(path, sep="\t")
    gold_ans = df["gold"].tolist()
    gold_type = df["gold-type"].tolist()
    f1 = df["f_score"].tolist()
    local_counter = {}

    for i in range(0, len(df)):
        local_counter.setdefault(gold_type[i], {})
        local_counter[gold_type[i]]["total"] = local_counter[gold_type[i]].get("total", 0) + 1
        f1[i] = "positive" if f1[i] >= 0.5 else "negative"
        local_counter[gold_type[i]][f1[i]] = local_counter[gold_type[i]].get(f1[i], 0) + 1
    print(path, local_counter)
    return local_counter, len(df)


if __name__ == "__main__":
    in_domain_datasets = ["SQuAD", "HotpotQA", "NewsQA", "TriviaQA-web", "SearchQA", "NaturalQuestionsShort"]
    out_domain_datasets = ["DROP", "RACE", "BioASQ", "TextbookQA", "RelationExtraction", "DuoRC"]
    columns = ["Percentage", "F1>", "Percentage", "F1", "Percentage", "F1", "Percentage", "F1", "Percentage", "F1", "Percentage", "F1"]
    final_table = {}
    i = 0
    for ds in in_domain_datasets:
        counter, example_number = process("../predicts-pretrained/extended-logs/pred-"+ds+".csv")
        for k, v in counter.items():
            final_table.setdefault(k, 2*i*[0])
            true_number = v.get("positive", 0)
            final_table[k].extend(["{:.2f}%".format(v["total"]/example_number*100), "{:.2f}%".format(true_number/v["total"]*100)])
        # fill empty field with 0
        for n in final_table.keys() - counter.keys():
            final_table[n].extend([0, 0])
        i += 1
    print(final_table)

    new_df = pandas.DataFrame.from_dict(final_table, orient="index", columns=columns)
    new_df.to_csv(sys.argv[1])
