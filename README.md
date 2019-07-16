## Data Augmentation
For a given QA dataset, identify all the named entities appearing in the contexts and append these named entities to the end of 
each context:

`python3 data_augmentation.py SQuAD.jsonl.gz`

## Analyse entity types in answers
For a given QA dataset, analyse the answer types and lengths, eg: how many tokens does the answer have and whether the answer
is a named entity or not:

`python3 ne_in_answer.py SQuAD.jsonl.gz`

## Cloze generation
Automatically generate cloze QA pairs from the contexts in a given QA dataset. First extract all the contexts by:

`python3 extract_context.py SQuAD.jsonl.gz`

Then generate cloze QA pairs based on these contexts:

`python3 cloze_question_generation.py contexts`  where contexts is the directory contains all the extracted contexts.

