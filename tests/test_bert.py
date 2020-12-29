import pytorch_lightning as pl
from project.classifier import LitBertClassifier
from project.datasets.dataset import SMSDataModule
from easydict import EasyDict


def test_lit_text_classifier():

    # No GPUs
    args = EasyDict(
        {
            "data_dir": "data/spam.csv",
            "batch_size": 8,
            "max_len": 100,
            "n_classes": 2,
            "max_epochs": 2,
            "gpus": 0,
            "weights_summary": 'full',
        }
    )

    dm = SMSDataModule(data_dir=args.data_dir, batch_size=args.batch_size,
                       max_len=args.max_len, sample_size=100)

    model = LitBertClassifier(args.n_classes)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)

    results = trainer.test(datamodule=dm)

    assert results[0]['test_acc'] > 0.7


if __name__ == "__main__":
    test_lit_text_classifier()
