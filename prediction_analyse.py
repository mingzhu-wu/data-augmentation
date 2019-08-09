import sys
import pandas
import numpy
import data_augmentation
import ne_in_answer


def get_ans_type(ans, ne_in_context):
    is_named_entity = False
    if ne_in_context.get(ans) is not None:
        ans_type = ne_in_context.get(ans)
        is_named_entity = True
    else:
        ans_type = ne_in_answer.analyse_non_ne_answer(ans).lower()
    return ans_type, is_named_entity


if __name__ == "__main__":
    input_file = sys.argv[1]
    df = pandas.read_csv(input_file, sep="\t")
    new_columns = ["qid", "question", "prediction", "isNE", "pred-type", "gold", "isNE", "gold-type", "em", "f_score", "context"]
    values = []
    i = 0
    for row in df.values:
        print(i)
        i += 1
        predict = row[3]
        gold = row[4]
        context = row[-1]
        ner_context = data_augmentation.get_ner_spacy_stanford(context)
        ner_context_normalize = {ne_in_answer.normalize_answer(k): v for k, v in ner_context.items()}
        pred_type, pred_flag = get_ans_type(ne_in_answer.normalize_answer(predict), ner_context_normalize)
        gold_type, gold_flag = get_ans_type(ne_in_answer.normalize_answer(gold), ner_context_normalize)
        new_row = row.tolist()[1:]
        new_row.insert(3, pred_flag)
        new_row.insert(4, pred_type)
        new_row.insert(6, gold_flag)
        new_row.insert(7, gold_type)
        values.append(new_row)

    new_df = pandas.DataFrame(values)
    new_df.columns = new_columns
    new_df.to_csv(input_file, sep="\t")

