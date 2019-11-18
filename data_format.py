"""
This script is used to format unsupervised qa dataset published by
https://github.com/facebookresearch/UnsupervisedQA?fbclid=IwAR0atDkusuT6Rc2-d1QAblETUAB7_rc7Ws3-x7quc4LpphcEeOT-XOZwaL0
to MRQA compatible qa dataset for augmented training
"""
import json
import sys
from spacy.lang.en import English
import uuid

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)


def get_context_tokens(context):
    tokens = []
    for token in tokenizer(context):
        tokens.append([str(token), token.idx])
    return tokens


def get_detected_answers(ans_list, context):
    detected_ans = []
    ans_dict = {}
    context_tokens = tokenizer(context)
    for ans in ans_list:
        ans_dict["text"] = ans["text"]
        ans_dict.setdefault("char_spans", [])
        ans_dict.setdefault("token_spans", [])
        ans_dict["char_spans"].append([ans["answer_start"], ans["answer_start"]+len(ans["text"])-1])
        for token in context_tokens:
            if token.idx == ans["answer_start"]:
                token_start = token.i
                token_end = token_start + len(tokenizer(ans["text"])) - 1
        ans_dict["token_spans"].append([token_start, token_end])
        detected_ans.append(ans_dict)

    return detected_ans


def get_formulated_qas(qas_list, context):
    new_qas = []
    new_qa = {}
    for qa in qas_list:
        new_qa["id"] = uuid.uuid1().hex
        new_qa["qid"] = qa["id"]
        new_qa["question"] = qa["question"]
        new_qa["question_tokens"] = get_context_tokens(qa["question"])
        new_qa["detected_answers"] = get_detected_answers(qa["answers"], context)
        new_qa["answers"] = [ans["text"] for ans in qa["answers"]]
        new_qas.append(new_qa)

    return new_qas


def data_format(file, out_file):
    new_format = {}
    with open(file, 'r') as f_src, open(out_file, 'w') as f_out:
        json.dump({"header": {"dataset": "SQuAD", "split": "train"}}, f_out)
        f_out.write("\n")
        content = json.load(f_src)
        print(len(content), type(content))
        for example in content["data"]:
            for paraph in example["paragraphs"]:
                try:
                    new_format["id"] = ""
                    new_format["context"] = paraph["context"]
                    new_format["context_tokens"] = get_context_tokens(paraph["context"])
                    new_format["qas"] = get_formulated_qas(paraph["qas"], paraph["context"])
                except BaseException as e:
                    print(e, paraph["context"])
                    new_format = {}
                    continue
                json.dump(new_format, f_out)
                f_out.write("\n")
                new_format = {}


if __name__ == '__main__':
    data_format(sys.argv[1], sys.argv[2])