"""
This script is used to generate adversary dataset based on the original dataset.
For every example, we find the sentence which has the max lexical overlap with the question.
And we remove all the other sentences in the context and keep only this lexical overlapped one.
Thus generate a new adversary set.
"""
import copy
import gzip
import json
import nltk
import os
import pandas
import shutil
import torch
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English
import uuid

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)
nlp.add_pipe(nlp.create_pipe('sentencizer'))
model = SentenceTransformer("bert-base-nli-mean-tokens")


def get_context_tokens(context):
    tokens = []
    for token in tokenizer(context):
        tokens.append([str(token), token.idx])
    return tokens


def get_detected_answers(ans, context):
    # print(ans)
    # print(context)
    detected_ans = []
    ans_dict = {}
    context_tokens = tokenizer(context)
    ans_dict["text"] = ans
    ans_dict.setdefault("char_spans", [])
    ans_dict.setdefault("token_spans", [])

    # find all the appearance of this answer in the context
    ans_starts = [i for i in range(len(context)) if context.lower().startswith(ans.lower(), i)]

    for token in context_tokens:
        if token.idx in ans_starts:
            token_start = token.i
            token_end = token_start + len(tokenizer(ans)) - 1
            ans_dict["char_spans"].append([token.idx, token.idx + len(ans) - 1])
            ans_dict["token_spans"].append([token_start, token_end])
            detected_ans.append(ans_dict)

    return detected_ans


def get_all_sentences(context):
    doc = nlp(context)
    ss = []
    for sent in doc.sents:
        ss.append(sent.text)
    return ss


def find_lexical_overlap_sentence(sents, q_tokens):
    # find the highest lexical overlap sentence by counting the common words
    question_tokens = [token[0] for token in q_tokens]
    overlap_length = 0
    max_overlap_sentence = ""
    # if two sentences have the same overlap with the question, only keeps one
    for sent in sents:
        ss = []
        for token in tokenizer(sent):
            ss.append(str(token))
        # exclude "the" when count lexical overlap
        overlap = [t for t in ss if t in question_tokens and t != "the"]
        if len(overlap) > overlap_length:
            overlap_length = len(overlap)
            max_overlap_sentence = sent
    return max_overlap_sentence


def find_lexical_overlap_sentence_embedding(sents, question):
    # find the highest lexical overlap sentence by consine similarity of the sentence embeddings.
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
    sentence_embeddings = model.encode(sents)
    question_embedding = model.encode([question])[0]
    max_sim = 0
    most_similar_sentence = ""
    # if two sentences have the same overlap with the question, only keeps one
    for sentence, sent_embedding in zip(sents, sentence_embeddings):
        cos_sim = cos(torch.FloatTensor(sent_embedding), torch.FloatTensor(question_embedding))
        if cos_sim >= max_sim:
            max_sim = cos_sim
            most_similar_sentence = sentence
    # print(question, most_similar_sentence, max_sim)
    return most_similar_sentence


def generate_new_example(content, qa):
    sentences = get_all_sentences(content['context'])
    sentence_to_keep = find_lexical_overlap_sentence_embedding(sentences, qa["question"])
    new_content = copy.deepcopy(content)
    new_content['context'] = sentence_to_keep
    new_qa = copy.deepcopy(qa)
    new_qa["answers"] = []
    new_qa["detected_answers"] = []
    has_answer_flag = False
    for answer in qa['answers']:
        if answer.lower() in sentence_to_keep.lower():
            detected_answer = get_detected_answers(answer, sentence_to_keep)
            # if answer is "7" and there is only 1997 in the sentence, then detected answer will be empty
            if detected_answer and detected_answer not in new_qa["detected_answers"]:
                count["has_anwer"] += 1
                has_answer_flag = True
                new_qa["detected_answers"] = detected_answer
                new_qa["answers"].append(answer)

    # key = "has_answer" if has_answer_flag else "no_answer"
    # if qa["qid"] in easy_example_ids:
    #     count["easy"][key] += 1
    # else:
    #     count["hard"][key] += 1

    new_content["qas"] = [new_qa]
    new_content["context_tokens"] = get_context_tokens(sentence_to_keep)
    return new_content


if __name__ == '__main__':
    # datasets = ["SQuAD", "HotpotQA", "NewsQA", "TriviaQA-web", "NaturalQuestionsShort"]
    datasets = ["test"]
    src_folder = "data_train"
    out_folder = "data_sentence_similar"

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)

    for dataset in datasets:
        print(dataset)
        # df = pandas.read_csv("predicts-teacher/csv/pred-"+dataset+"-dev.csv", sep="\t")
        # qids = df["qid"].to_list()
        # em = df["em"].to_list()
        # easy_example_ids = [qids[i] for i in range(len(qids)) if em[i]]
        # hard_example_ids = [qid for qid in qids if qid not in easy_example_ids]
        # print(f"total number of examples is {len(df)}, correctly predicted is {len(easy_example_ids)}")

        # count = {"easy": {"has_answer": 0, "no_answer": 0}, "hard": {"has_answer": 0, "no_answer": 0}, "total_questions": 0}
        count = {"has_anwer": 0, "total_questions": 0}
        f_out = gzip.open(os.path.join(out_folder, dataset + ".jsonl.gz"), "at")
        f_out.write(json.dumps({"header": {"dataset": dataset + "Ovlp", "mrqa_split": "train"}}) + "\n")
        with gzip.open(os.path.join(src_folder, dataset + ".jsonl.gz")) as f_in:
            next(f_in)
            lines = f_in.readlines()
            for line in lines:
                content = json.loads(line.strip())
                for qa in content["qas"]:
                    count["total_questions"] += 1
                    reduced_content = generate_new_example(content, qa)
                    f_out.write(json.dumps(reduced_content) + "\n")
        f_out.close()

        print(count)
