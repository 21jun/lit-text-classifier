import torch
import pytorch_lightning as pl
import pandas as pd
from argparse import ArgumentParser
from project.classifier import LitBertClassifier
from project.datasets.dataset import SMSDataModule, DataFrameDataset
from transformers import BertTokenizer

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'


def make_sample(text_list: list):
    df = pd.DataFrame({"text": text_list})
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    sample_ds = SMSDataModule.sample(df, 100, tokenizer)
    return sample_ds


def sample_inference(model, ds):
    for d in ds:
        print(d.text)
        outputs = model(d.input_ids.unsqueeze(
            0), d.attention_mask.unsqueeze(0))
        _, preds = torch.max(outputs, dim=1)
        print(preds)


def cli_main():
    pl.seed_everything(1234)
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        '--model', default='save/checkpoints/lightning_logs/version_0/checkpoints/epoch=0-step=487.ckpt', type=str)
    args = parser.parse_args()

    # spam(1), ham(0), spam, ham, spam
    text_list = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.",
        "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info",
        "Will ? b going to esplanade fr home?",
        "This is the 2nd time we have tried 2 contact u. U have won the 鶯750 Pound prize. 2 claim is easy, call 087187272008 NOW1! Only 10p per minute. BT-national-rate."
    ]

    model = LitBertClassifier.load_from_checkpoint(args.model)
    ds = make_sample(text_list)

    sample_inference(model, ds)


if __name__ == '__main__':

    cli_main()
