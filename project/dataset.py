import re

import pandas as pd
import torch
from torch.utils.data import DataLoader

from torchtext import data
from torchtext.data import Field

import pytorch_lightning as pl

from sklearn.model_selection import train_test_split


from transformers import BertTokenizer, BertModel

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'


class DataFrameDataset(data.Dataset):

    def __init__(self, df, fields, tokenizer, max_len, is_test=False, **kwargs):
        self.tokenizer = tokenizer
        self.max_len = max_len
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
    def splits(cls, fields, tokenizer, max_len, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        # cls == DataFrameDataset

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field,
                             max_len=max_len, tokenizer=tokenizer, is_test=False)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field,
                           max_len=max_len, tokenizer=tokenizer, is_test=False)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field,
                            max_len=max_len, tokenizer=tokenizer, is_test=True)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        tweet = str(self.examples[item].text)
        target = self.examples[item].label

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class SMSDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, max_len, tokenizer, encoding='latin-1'):
        super().__init__()
        self.data_dir = data_dir
        self.encoding = encoding
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        self.df = pd.read_csv(self.data_dir, encoding=self.encoding)
        self.df = self.df[['v1', 'v2']]
        self.df = self.df.rename(columns={'v1': 'target', 'v2': 'text'})
        self.df['text'] = self.df['text'].apply(normalize_text)

        self.df = self.df[:100]

        TEXT = Field(sequential=True,
                     use_vocab=False,
                     # lower=True,
                     # tokenize=False # set False: using Dataset encode_plus instead
                     )

        LABEL = Field(sequential=False,
                      use_vocab=False,
                      preprocessing=lambda x: 1 if x == "spam" else 0,
                      is_target=True)

        fields = [('text', TEXT), ('label', LABEL)]

        self.train_df, self.test_df = train_test_split(self.df, test_size=0.3)
        self.val_df, self.test_df = train_test_split(
            self.test_df, test_size=0.5)

        self.train_ds, self.val_ds, self.test_ds = DataFrameDataset.splits(
            fields=fields, tokenizer=self.tokenizer, max_len=self.max_len, train_df=self.train_df, val_df=self.val_df, test_df=self.test_df)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)


def normalize_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"\#", "", text)
    text = re.sub(r"http\S+", "URL", text)  # remove URL addresses
    text = re.sub(r"@", "", text)
    text = re.sub(r"[^A-Za-z0-9()!?\'\`\"]", " ", text)
    text = re.sub("\s{2,}", " ", text)
    return text


def main():

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    datamodule = SMSDataModule("../data/spam.csv", 8,
                               max_len=100, tokenizer=tokenizer)
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    for i in train_loader:
        print(i)
        break


if __name__ == "__main__":
    main()
