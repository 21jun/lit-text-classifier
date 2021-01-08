import re

import pandas as pd
import torch

from torch.utils.data import DataLoader
from torchtext import data
from torchtext.data import Field
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'


class DataFrameDataset(data.Dataset):

    def __init__(self, df, fields, tokenizer, max_len, is_test=False, **kwargs):
        self.tokenizer = tokenizer
        self.max_len = max_len
        examples = []

        for i, row in df.iterrows():
            label = row.target if not is_test else None
            text = row.text

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids = encoding['input_ids'].flatten()
            attention_mask = encoding['attention_mask'].flatten()

            examples.append(data.Example.fromlist(
                [text, input_ids, attention_mask, label], fields))

        super().__init__(examples, fields, **kwargs)

    @ staticmethod
    def sort_key(ex):
        return len(ex.text)

    @ classmethod
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

        text = str(self.examples[item].text)
        target = self.examples[item].label
        input_ids = self.examples[item].input_ids
        attention_mask = self.examples[item].attention_mask

        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': torch.tensor(target, dtype=torch.long)
        }


class SMSDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, max_len, sample_size=None, encoding='latin-1'):
        super().__init__()
        self.data_dir = data_dir
        self.encoding = encoding
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.max_len = max_len
        self.sample_size = sample_size

    def prepare_data(self):
        pass

    @staticmethod
    def sample(df, max_len, tokenizer):
        TEXT = Field(sequential=True,
                     # tokenize=self.tokenizer.encode,
                     use_vocab=False)

        INPUT_IDS = Field(sequential=False,
                          use_vocab=False)

        ATTENTION_MASK = Field(sequential=False,
                               use_vocab=False)

        LABEL = Field(sequential=False,
                      use_vocab=False,
                      preprocessing=lambda x: 1 if x == "spam" else 0,
                      is_target=True)

        fields = [('text', TEXT), ('input_ids', INPUT_IDS),
                  ('attention_mask', ATTENTION_MASK), ('label', LABEL)]

        ds = DataFrameDataset(df=df, fields=fields, tokenizer=tokenizer,
                              max_len=max_len, is_test=True)

        return ds

    def setup(self, stage=None):

        self.df = pd.read_csv(self.data_dir, encoding=self.encoding)
        self.df = self.df[['v1', 'v2']]
        self.df = self.df.rename(columns={'v1': 'target', 'v2': 'text'})
        self.df['text'] = self.df['text'].apply(normalize_text)

        # self.df = self.df[:50]

        """
        Attributes:
            sequential: Whether the datatype represents sequential data. If False,
                no tokenization is applied. Default: True.
            use_vocab: Whether to use a Vocab object. If False, the data in this
                field should already be numerical. Default: True.
        """
        # TEXT 필드는 학습에 이용되지는 않지음
        # string 이지만 vocab을 사용하지 않기에 use_vocab=False
        TEXT = Field(sequential=True,
                     # tokenize=self.tokenizer.encode,
                     use_vocab=False)

        INPUT_IDS = Field(sequential=False,
                          use_vocab=False)

        ATTENTION_MASK = Field(sequential=False,
                               use_vocab=False)

        LABEL = Field(sequential=False,
                      use_vocab=False,
                      preprocessing=lambda x: 1 if x == "spam" else 0,
                      is_target=True)

        fields = [('text', TEXT), ('input_ids', INPUT_IDS),
                  ('attention_mask', ATTENTION_MASK), ('label', LABEL)]

        # For sampling
        if self.sample_size:
            self.df = self.df[:self.sample_size]

        self.train_df, self.test_df = train_test_split(self.df, test_size=0.3)
        self.val_df, self.test_df = train_test_split(
            self.test_df, test_size=0.5)

        self.train_ds, self.val_ds, self.test_ds = DataFrameDataset.splits(
            fields=fields, tokenizer=self.tokenizer, max_len=self.max_len, train_df=self.train_df, val_df=self.val_df, test_df=self.test_df)

    def train_dataloader(self):
        # BERT는 고정길이 (여기선 100)의 입력을 받기때문에 BucketIterator 등을 사용하지 않음
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


# for Test
def main():
    datamodule = SMSDataModule("../../data/spam.csv", 8,
                               max_len=100)
    datamodule.setup()

    train_loader = datamodule.train_dataloader()

    print(train_loader)
    for sample in train_loader:
        print(sample['attention_mask'].size())
        print(sample['input_ids'].size())
        print(sample['targets'].size())
        break


if __name__ == "__main__":
    main()
