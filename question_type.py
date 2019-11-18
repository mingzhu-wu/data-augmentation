import gzip
import json
import pandas
import sys
import text_analyse
import postprocess


def process_example(example, counter):
    qa_num = 0
    for qa in example["qas"]:
        qa_num += 1
        question = qa["question"]
        question_tokens = [token[0].lower() for token in qa["question_tokens"]]
        q_type = text_analyse.get_question_type(question.lower(), question_tokens)
        counter.setdefault(q_type, {"F1": 0})
        counter[q_type]["total"] = counter[q_type].get("total", 0) + 1
    return qa_num


if __name__ == "__main__":
    datasets = ["SQuAD", "HotpotQA", "NewsQA", "TriviaQA", "SearchQA", "NaturalQuestions"]
    final_table = {}
    for dataset in datasets:
        print(dataset)
        i = 0
        counter = {}
        with gzip.open("data_train/"+sys.argv[1]+"/"+dataset+".jsonl.gz", 'rt') as f_in:
            next(f_in)
            qa_sum = 0
            for line in f_in:
                example = json.loads(line)
                qa_sum += process_example(example, counter)
            print(qa_sum)
        print(counter)
        postprocess.counter_to_table(counter, final_table, i, qa_sum)
        i += 1

    columns = 6 * ["Percentage", "F1"]
    new_df = pandas.DataFrame.from_dict(final_table, orient="index", columns=columns)
    new_df.to_csv("data/"+sys.argv[1] + "-train.csv", sep="\t")

