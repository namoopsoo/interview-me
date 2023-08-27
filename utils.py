import pandas as pd
import torch
import requests
import json
import re
import yaml
from functools import reduce
from transformers import AutoTokenizer, AutoModel
from itertools import combinations
from sentence_transformers.util import semantic_search, cos_sim


from nltk.corpus import stopwords as sw_nltk
import spacy
en = spacy.load("en_core_web_sm")
sw_spacy = en.Defaults.stop_words


def dont_stop(s):
    """Remove stop words from a sentence

    Example
    -------
    >>> s = "So I was talking with someone about you know coffee making methods and umm yea I learned one more"
    >>> ut.dont_stop(s)
    'talking know coffee making methods umm yea learned'
    """
    stop_words = set(sw_nltk.words("english")) | set(sw_spacy)
    return " ".join([x for x in s.split() if x.lower() not in stop_words])


def read_yaml(loc):
    with open(loc) as fd:
        return yaml.safe_load(fd)


def remove_punctuation(s):
    """Replace punctuation with spaces, collapsing consecutive spaces.

    Example
    ------
    >>> s = "·And, here: look!? (aws/stuff) • yea "
    >>> ut.remove_punctuation(s)
    'And here look aws stuff yea'
    """

    punctuation_characters_regex = "[\"·•:?!,./()’“]"
    one = re.sub(punctuation_characters_regex, " ", s)
    return re.sub(r"\s+", " ", one).strip()


def extract_raw_sentences(df, columns):
    """Cut up strings in rows into separate sentences, splitting by the given columns and by punctuation.
    """
    delimeter_characters_regex = "[·•]"
    raw_sentences = reduce(
        lambda x, y: x + y,
        filter(
            None,

            [
                re.split(
                    r"\. |\.\n|\? |\?\n", 
                    re.sub(
                        delimeter_characters_regex,
                        ". ",
                        df.iloc[i][col]
                        ))

                for i in range(df.shape[0])
                for col in columns
                if not pd.isnull(df.iloc[i][col])
                ]

            )
        )
    sentences = list(set(
        [
            x.strip().lower()
            for x in (set(raw_sentences) - set([""]))
        ]
    ))
    return sentences


def get_nohit_job_terms():
    job_terms = [
        "html", "databricks", "python", "css", "api", "postgresql", "database", 
        "mysql", "clojure", "java", "javascript", "angular", "idempotent", "azure",
        "github", "git", "concurrency", "asyncio", "dbutils", "ipython", "docker",
        "pipeline", "sklearn", "tensorflow", "pytorch", "numpy", "pandas", "ec2", "ecs",
        "aws", "sagemaker", "nginx", "redis", "cli", "auc", "xgboost", "repository",
        "pyspark", "nlp", "spacy",
        ]
    return job_terms


def current_nohit_list(model_name):
    """
    """
    job_terms = get_nohit_job_terms()
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


 
def build_my_blurb(experiences_dict):
    """ Build a list of raw sentences from a dictionary with narratives about past experiences,
    """

    sentences = []
    for project, detail in experiences_dict.items():
        one_liners = detail.get("one-liners", [])
        sentences.extend(one_liners)

        stories = detail.get("stories", [])
        if stories:
            stories_split = reduce(
                lambda x, y: x + y, 
                [
                    re.split(r"\. |\.\n", x)
                    for x in stories]
            )
            stories_split = [x for x in stories_split
                             if x and x.strip().strip("\.")]
            sentences.extend(stories_split)
    return sentences


def filter_pandas_multiple_contains(df, column, vec, case=False):
    """filter dataframe for column containing any string from list vec given.

    Example
    >>> vec = [
    ... {"title": "Software Engineer yea"},
    ... {"title": "Some Scientist"},
    ... {"title": "Product Manager"},
    ... {"title": "Industrial Designer"}
    ]
    >>> df = pd.DataFrame.from_records(vec)
    >>> df
                       title
    0  Software Engineer yea
    1         Some Scientist
    2        Product Manager
    3    Industrial Designer
    >>> import utils as ut
    >>> ut.filter_pandas_multiple_contains(df, "title", ["engineer", "scientist"])
                       title
    0  Software Engineer yea
    1         Some Scientist
    """
    query = " or ".join(
            [f"{column}.str.contains('{x}', case={case})"
             for x in vec])
    return df.query(query)


def vec_to_embeddings(model_id, texts, hf_token=None, return_tokenizer_output=False):
    if hf_token:
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        headers = {"Authorization": f"Bearer {hf_token}"}

        response = requests.post(
            api_url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
        output = response.json()
        return torch.FloatTensor(output)
    else:
        # Load AutoModel from huggingface model repository
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Tokenize sentences
        encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

        sentence_embeddings = encoded_to_embeddings(encoded_input, model_id)

        if return_tokenizer_output:
            return encoded_input, sentence_embeddings
        else:
            return sentence_embeddings


def encoded_to_embeddings(encoded_input, model_id):
    # NOTE this assumes encoding was done w/ tokenizer that works with that model.
    model = AutoModel.from_pretrained(model_id)
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]



def cluster():
    ...


def search(model_id, vec1, vec2, top_k=10):
    """
    """
    # model_id =  "sentence-transformers/all-MiniLM-L6-v2"
    # some = random.choices(raw_sentences, k=1000)
    embeddings1 = vec_to_embeddings(model_id, vec1)
    embeddings2 = vec_to_embeddings(model_id, vec2)

    hits = semantic_search(embeddings1, embeddings2, top_k=top_k)
    return hits

def search_results_to_pdf(hits, left, right, preview=True):
    if preview:
        k = 5
        n = 5
    else:
        k = len(hits)
        n = len(hits[0])
    vec = []
    for i, row in enumerate(hits[:k]):
        if preview:
            print(f"({i})", "matching,", left[i][:20], ":")
            hmm = [
                    [right[x["corpus_id"]][:20], x["corpus_id"], x["score"]] for x in row[:n] ]
            print(hmm, "\n\n")
        vec.extend([{
          "query": left[i],
          "result": right[x["corpus_id"]],
          "score": x["score"],
        } for x in row])

    return pd.DataFrame.from_records(vec)
