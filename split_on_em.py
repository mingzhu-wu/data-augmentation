import copy
import os
import gzip
import json
import pandas
import shutil
import sys


if __name__ == "__main__":
    datasets = ["SQuAD", "HotpotQA", "NewsQA", "TriviaQA", "SearchQA", "NaturalQuestions"]
    src_folder = sys.argv[1]
    out_folder = sys.argv[2]

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)

    for dataset in datasets:
        print(dataset)
        df = pandas.read_csv("predicts-random/Logs/pred-"+dataset+".csv", sep="\t")
        qids = df["qid"].to_list()
        em = df["em"].to_list()
        easy_example_ids = [qids[i] for i in range(len(qids)) if em[i]]
        # hard_example_ids = [qid for qid in qids if qid not in easy_example_ids]

        f_out_easy = gzip.open(os.path.join(out_folder, dataset + "Easy.jsonl.gz"), "at")
        f_out_easy.write(json.dumps({"header": {"dataset": dataset+"Easy", "mrqa_split": "dev"}}) + "\n")
        f_out_hard = gzip.open(os.path.join(out_folder, dataset + "Hard.jsonl.gz"), "at")
        f_out_hard.write(json.dumps({"header": {"dataset": dataset+"Hard", "mrqa_split": "dev"}}) + "\n")

        with gzip.open(os.path.join(src_folder, dataset + ".jsonl.gz")) as f_in:
            next(f_in)
            lines = f_in.readlines()
            for line in lines:
                example = json.loads(line.strip())
                info_per_context = {}
                for qa in example["qas"]:
                    qa_id = qa["qid"]
                    if qa_id in easy_example_ids:
                        info_per_context.setdefault(f_out_easy, [])
                        info_per_context[f_out_easy].append(qa["qid"])
                    else:
                        info_per_context.setdefault(f_out_hard, [])
                        info_per_context[f_out_hard].append(qa["qid"])

                for f_out, ids in info_per_context.items():
                    new_content = copy.deepcopy(example)
                    new_content["qas"] = []
                    for qa in example["qas"]:
                        if qa["qid"] in ids:
                            new_content["qas"].append(qa)
                    f_out.write(json.dumps(new_content) + "\n")

