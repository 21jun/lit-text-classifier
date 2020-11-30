import re

import pandas as pd
import torch
from torch.functional import norm

from torchtext import data
from torchtext.data import Field

from transformers import BertTokenizer, BertModel

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'


class SMSDataset(data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        # https://pytorch.org/text/stable/data.html#fields
        TEXT = Field(sequential=True,
                     use_vocab=True,
                     lower=True,
                     tokenize=tokenizer,
                     batch_first=True)

        LABEL = Field(sequential=False,
                      use_vocab=False,
                      preprocessing=lambda x: 1 if x == "spam" else 0,
                      batch_first=True,
                      is_target=True)
        self.fields = [('text', TEXT), ('label', LABEL)]

    def __getitem__(self, index: int):
        pass

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        #cls == SMSDataset

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


class DataFrameDataset(data.Dataset):

    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.target if not is_test else None
            text = row.text
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        #cls == DataFrameDataset

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


def normalize_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"\#", "", text)
    text = re.sub(r"http\S+", "URL", text)  # remove URL addresses
    text = re.sub(r"@", "", text)
    text = re.sub(r"[^A-Za-z0-9()!?\'\`\"]", " ", text)
    text = re.sub("\s{2,}", " ", text)
    return text


def main():

    df = pd.read_csv("../data/spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df = df.rename(columns={'v1': 'target', 'v2': 'text'})
    labels = df['target']
    texts = df['text']
    texts = texts.apply(normalize_text)

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    TEXT = Field(sequential=True,
                 use_vocab=True,
                 lower=True,
                 tokenize=tokenizer,
                 batch_first=True)

    LABEL = Field(sequential=False,
                  use_vocab=False,
                  preprocessing=lambda x: 1 if x == "spam" else 0,
                  batch_first=True,
                  is_target=True)

    fields = [('text', TEXT), ('label', LABEL)]

    (train_dataset) = DataFrameDataset.splits(fields=fields, train_df=df)

    print(train_dataset)

    # dataset = SMSDataset(texts=texts, labels=labels,
    #                      tokenizer=tokenizer, max_len=100)


if __name__ == "__main__":
    main()
