from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification.accuracy import Accuracy

from transformers import BertModel


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
