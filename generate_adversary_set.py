"""
This script is used for generate adversary datasets. For example, given original SQuAD training set, generate a new
SQuAD-adv set in which questions only have one wh-word instead of a whole question sentence.
"""
import os
import gzip
import json
import shutil
import text_analyse

if __name__ == "__main__":
    datasets = ["SQuAD", "HotpotQA", "NewsQA", "TriviaQA-web", "SearchQA", "NaturalQuestionsShort"]
    # datasets = ["TriviaQA-web", "NaturalQuestionsShort"]
    src_folder = "data_dev_in_domain"
    out_folder = "data_adversary_dev"

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)

    for dataset in datasets:
        print(dataset)
        f_out = gzip.open(os.path.join(out_folder, dataset + ".jsonl.gz"), "at")
        f_out.write(json.dumps({"header": {"dataset": dataset+"Advr", "mrqa_split": "train"}}) + "\n")
        with gzip.open(os.path.join(src_folder, dataset+".jsonl.gz")) as f_in:
            next(f_in)
            lines = f_in.readlines()
            for line in lines:
                content = json.loads(line.strip())
                for qa in content["qas"]:
                    question = qa["question"]
                    question_tokens = [token[0].lower() for token in qa["question_tokens"]]
                    q_type = text_analyse.get_question_type(question.lower(), question_tokens)
                    if q_type == "others":
                        # if there is no WH word in this sentence, keep the first tokens.
                        qa["question"] = qa["question_tokens"][0][0]
                        qa["question_tokens"] = qa["question_tokens"][0:1]
                    else:
                        qa["question"] = q_type
                        qa["question_tokens"] = [[q_type, 0]]
                f_out.write(json.dumps(content) + "\n")
        f_out.close()