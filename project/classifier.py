from argparse import ArgumentParser
import re

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torchtext
from torchtext import data
from torchtext.data import Field

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification.accuracy import Accuracy

from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertModel

# from project.dataset import SMSDataModule

from dataset import SMSDataModule


PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'


class LitBertClassifier(pl.LightningModule):
    def __init__(self, n_classes):
        super().__init__()
        self.save_hyperparameters()

        self.acc = Accuracy()
        self.loss_fn = nn.CrossEntropyLoss()

        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):

        # print(input_ids.size(), attention_mask.size())

        # print(input_ids, attention_mask)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        pooled_output = outputs[1]

        output = self.drop(pooled_output)
        return self.out(output)

    def training_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = self.loss_fn(outputs, targets)

        acc = self.acc(preds, targets)
        # self.log("acc", acc, prog_bar=True) # Too noisy
        return {"loss": loss, "acc": acc}

    def training_epoch_end(self, outputs):

        accs = [x['acc'] for x in outputs]
        epoch_acc = sum(accs) / len(accs)
        self.log("epoch_acc", epoch_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = self.loss_fn(outputs, targets)

        acc = self.acc(preds, targets)
        # self.log("acc", acc, prog_bar=True) # Too noisy
        return {"loss": loss, "acc": acc}

    def validation_epoch_end(self, outputs):

        accs = [x['acc'] for x in outputs]
        epoch_acc = sum(accs) / len(accs)
        self.log("val_acc", epoch_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = self.loss_fn(outputs, targets)

        acc = self.acc(preds, targets)
        # self.log("acc", acc, prog_bar=True) # Too noisy
        return {"loss": loss, "acc": acc}

    def test_epoch_end(self, outputs):

        accs = [x['acc'] for x in outputs]
        epoch_acc = sum(accs) / len(accs)
        self.log("test_acc", epoch_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


class TweetDataset(data.Dataset):

    def __init__(self, tweets, targets, tokenizer, max_len):
        self.tweets = tweets
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

        TEXT = Field(sequential=True,
                     use_vocab=True,
                     lower=True,
                     batch_first=True)

        LABEL = Field(sequential=False,
                      use_vocab=False,
                      preprocessing=lambda x: int(x),  # String 을 Int 로
                      batch_first=True)

        self.fields = fields = [('text', TEXT), ('label', LABEL)]

        examples = []
        for text, label in zip(tweets, targets):
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields)

    def __len__(self):
        return len(self.tweets)

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


class LitTweetDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, max_len):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.max_len = max_len

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print("setup starts")
        self.df = pd.read_csv(self.data_dir)
        self.df = self.df
        self.df['tweet'] = LitTweetDataModule.normalize_text(self.df['tweet'])

        self.df_train, self.df_test = train_test_split(self.df, test_size=0.3)
        self.df_val, self.df_test = train_test_split(
            self.df_test, test_size=0.5)

        self.train = TweetDataset(self.df_train.tweet.to_numpy(),
                                  self.df_train.label,
                                  tokenizer=self.tokenizer,
                                  max_len=self.max_len
                                  )
        self.val = TweetDataset(self.df_val.tweet.to_numpy(),
                                self.df_val.label,
                                tokenizer=self.tokenizer,
                                max_len=self.max_len
                                )

        self.test = TweetDataset(self.df_test.tweet.to_numpy(),
                                 self.df_test.label,
                                 tokenizer=self.tokenizer,
                                 max_len=self.max_len
                                 )

    @staticmethod
    def normalize_text(text):
        text = text.str.lower()  # lowercase
        text = text.str.replace(r"\#", "")  # replaces hashtags
        text = text.str.replace(r"http\S+", "URL")  # remove URL addresses
        text = text.str.replace(r"@user", "[USER]")
        text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
        text = text.str.replace("\s{2,}", " ")
        return text

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int)
    # https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech?select=train.csv
    parser.add_argument(
        '--data_dir', default='../data/TWEET/train.csv', type=str)
    parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--gpus', default=0, type=int, help=False)
    parser.add_argument('--weights_summary', default='full', type=str)
    # parser.add_argument('--learning_rate', default=0.001, type=float)
    # parser.add_argument('--max_epochs', default=5, type=int, help=False)

    # parser = pl.Trainer.add_argparse_args(parser)
    parser = LitBertClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    tweet = SMSDataModule("data/spam.csv", 8,
                          max_len=100, tokenizer=tokenizer)

    # tweet = LitTweetDataModule(args.data_dir, args.batch_size, args.max_len)

    # ------------
    # model
    # ------------
    model = LitBertClassifier(args.n_classes)

    # tweet

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, tweet)

    # ------------
    # testing
    # ------------
    trainer.test(datamodule=tweet)


if __name__ == '__main__':
    # print(pd.read_csv("./TWEET/train.csv"))

    cli_main()
