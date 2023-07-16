import pandas as pd
import json
import re
from functools import reduce

from transformers import AutoTokenizer
from itertools import combinations


def extract_raw_sentences(df, columns):
    raw_sentences = reduce(lambda x, y: x + y,
        [re.split(r"[\n\.]", df.iloc[i][col])
                for i in range(df.shape[0])
                for col in columns
                 if not pd.isnull(df.iloc[i][col])
                ]
    )
    sentences = [
        x.strip().lower()
        for x in (set(raw_sentences) - set([""]))
    ]
    return sentences


def current_nohit_list(model_name):
    job_terms = [
        "html", "databricks", "python", "css", "api", "postgresql", "database", 
        "mysql", "clojure", "java", "javascript", "angular", "idempotent", "azure",
        "github", "git", "concurrency", "asyncio", "dbutils", "ipython", "docker",
        "pipeline", "sklearn", "tensorflow", "pytorch", "numpy", "pandas", "ec2", "ecs",
        "aws", "sagemaker", "nginx", "redis", "cli", "auc", "xgboost", "repository",
        "pyspark", "nlp", "spacy",
        ]
    hits = []
    no_hits = []
    vocabulary = vocabulary_of_model(model_name)
    for term in job_terms:
        for token in vocabulary:
            if term == token.strip("#"):
                hits.append([term, token])

    no_hits = list(set(job_terms) - set([x[0] for x in hits]))
    return no_hits


def vocabulary_of_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocabulary = tokenizer.get_vocab()
    return (vocabulary.keys())


def sequence_from_sentence(sentence):
    words = re.split(r"[^a-zA-Z0-9]", sentence)
    words = [x for x in words if x]
    return words


def find_nohit_sentences(sentences, nohit_list):
    # DRAFT !!!
    ...
    
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Look for both solo and ## prefixed tokens, 
    nohit_set = set(nohit_list) # | set([f"##{x}" for x in nohit_list])

    nohit_sentences = []
    for sentence in sentences:
        sequence = sequence_from_sentence(sentence)
        if found := list(set(sequence) & nohit_set):
            nohit_sentences.append([sentence, found])

    return nohit_sentences
    # tokens = tokenizer.tokenize(sentence)
    # unmatched = nohit_set - set(tokens)
    # print(unmatched)


def make_positive_pairs_from_groups(*clusters):
    """
    Given one or more clusters, create a positive-pair dataset from these.

    Saw reference to https://huggingface.co/datasets/embedding-data/sentence-compression as an example.
    """
    dataset = []
    for cluster in clusters:
        if len(cluster) < 2:
            print("could not use this cluster", cluster)
        combos = combinations(cluster, 2)
        for combo in combos:
            # dataset.append(json.dumps({"set": combo}))
            dataset.append({"set": combo})

    return dataset


