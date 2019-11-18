import copy
import os
import gzip
import json
import logging
import shutil

## merge "whom" and "whose" to "who", cause they don't accout for much

WH_WORDS = {"what": "what", "which": "which", "who": "who", "whom": "who", "whose": "who", \
            "where": "where", "why": "why", "how": "how", "when": "when", "others": "others"}


def get_question_type(question_tokens=[]):
    qs_type = "others"
    for key_word in WH_WORDS.keys():
        if key_word in question_tokens:
            qs_type = WH_WORDS[key_word]
            break
    return qs_type


def write_questions(question_type, question_ids, content, folder):
    new_content = copy.deepcopy(content)
    new_content["qas"] = []
    with gzip.open(os.path.join(folder, question_type+".jsonl.gz"), "at") as f_out:
        for qa in content["qas"]:
            if qa["qid"] in question_ids:
                new_content["qas"].append(qa)
        f_out.write(json.dumps(new_content) + "\n")


def generate_question_set(content, out_folder):
    question_info = {}
    for qa in content["qas"]:
        question_tokens = [token[0].lower() for token in qa["question_tokens"]]
        qa_type = get_question_type(question_tokens)
        question_info.setdefault(qa_type, [])
        question_info[qa_type].append(qa["qid"])

    for qa_type, ids in question_info.items():
        write_questions(qa_type, ids, content, out_folder)


if __name__ == "__main__":
    datasets = ["SQuAD", "HotpotQA", "NewsQA", "TriviaQA", "SearchQA", "NaturalQuestions"]
    # datasets = ["test", "SQuAD"]
    src_folder = "data_dev_in_domain"
    out_folder = "question_dev"

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)

    for q_type in set(WH_WORDS.values()):
        with gzip.open(os.path.join(out_folder, q_type + ".jsonl.gz"), "at") as f_out:
            f_out.write(json.dumps({"header": {"dataset": q_type, "mrqa_split": "dev"}}) + "\n")

    final_table = {}
    for dataset in datasets:
        logging.info(dataset)
        print(dataset)
        with gzip.open(os.path.join(src_folder, dataset+".jsonl.gz")) as f_in:
            next(f_in)
            lines = f_in.readlines()
            for line in lines:
                example = json.loads(line.strip())
                generate_question_set(example, out_folder)

