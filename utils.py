import pandas as pd
import re
from functools import reduce


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
