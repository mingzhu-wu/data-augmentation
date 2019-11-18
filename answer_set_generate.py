import copy
import os
import gzip
import json
import logging
import shutil

ANSWER_TYPES = {"PERSON": ["PERSON"], "DATE": ["DATE"], "ORG": ["ORG", "ORGANIZATION"],
                "LOC": ["LOC", "FAC", "LOCATION"],
                "GPE": ["GPE", "COUNTRY", "CITY", "STATE_OR_PROVINCE"], "NORP": ["NORP", "NATIONALITY", "RELIGION"],
                "NUMBER": ["NUMBER", "MONEY", "CARDINAL", "PERCENT", "DURATION", "QUANTITY", "ORDINAL", "TIME", "SET"],
                "MISC": ["MISC", "WORK_OF_ART", "TITLE", "EVENT", "PRODUCT", "CAUSE_OF_DEATH", "LAW", "IDEOLOGY", "URL",
                         "CRIMINAL_CHARGE", "LANGUAGE"],
                "np": ["np"], "frag": ["frag"], "frag": ["frag"], "s": ["s", "sinv", "sbar", "sbarq"],
                "x": ["pp", "adjp", "advp", "vp", "x", "ucp", ]
                }


def get_answer_type(qid, ans_info):
    ans_type = "x"
    for answer_type, ans_ids in ans_info.items():
        for ans_id_pair in ans_ids:
            if qid in ans_id_pair[0]:
                for key, value in ANSWER_TYPES.items():
                    if answer_type in value:
                        ans_type = key
    return ans_type


def write_questions(ans_type, question_ids, content, folder):
    new_content = copy.deepcopy(content)
    new_content["qas"] = []
    with gzip.open(os.path.join(folder, ans_type + ".jsonl.gz"), "at") as f_out:
        for qa in content["qas"]:
            if qa["qid"] in question_ids:
                new_content["qas"].append(qa)
        f_out.write(json.dumps(new_content) + "\n")


def generate_answer_set(content, out_folder, ans_info):
    ans_info_per_context = {}
    for qa in content["qas"]:
        qa_id = qa["qid"]
        ans_type = get_answer_type(qa_id, ans_info)
        ans_info_per_context.setdefault(ans_type, [])
        ans_info_per_context[ans_type].append(qa["qid"])
    for ans_type, ids in ans_info_per_context.items():
        write_questions(ans_type, ids, content, out_folder)


if __name__ == "__main__":
    datasets = ["SQuAD", "HotpotQA", "NewsQA", "TriviaQA", "SearchQA", "NaturalQuestions"]
    # datasets = ["test", "SQuAD"]
    src_folder = "data_dev_in_domain"
    out_folder = "answer_dev"

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)

    for a_type in set(ANSWER_TYPES.keys()):
        with gzip.open(os.path.join(out_folder, a_type + ".jsonl.gz"), "at") as f_out:
            f_out.write(json.dumps({"header": {"dataset": a_type, "mrqa_split": "dev"}}) + "\n")

    final_table = {}
    for dataset in datasets:
        logging.info(dataset)
        print(dataset)
        # load all the answer type information
        with open(os.path.join("NETag-dev", "NETag_" + dataset + ".json")) as f_info:
            ans_info = json.load(f_info)
        with gzip.open(os.path.join(src_folder, dataset + ".jsonl.gz")) as f_in:
            next(f_in)
            lines = f_in.readlines()
            for line in lines:
                example = json.loads(line.strip())
                generate_answer_set(example, out_folder, ans_info)
