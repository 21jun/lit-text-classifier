import pytorch_lightning as pl
from argparse import ArgumentParser
from project.classifier import LitBertClassifier
from project.datasets.dataset import SMSDataModule


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument(
        '--data_dir', default='data/spam.csv', type=str)
    parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--gpus', default=0, type=int, help=False)
    parser.add_argument('--weights_summary', default='full', type=str)

    # parser = pl.Trainer.add_argparse_args(parser)
    parser = LitBertClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    # https://www.kaggle.com/uciml/sms-spam-collection-dataset
    dm = SMSDataModule(data_dir=args.data_dir, batch_size=args.batch_size,
                       max_len=args.max_len)

    # ------------
    # model
    # ------------
    model = LitBertClassifier(args.n_classes)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    trainer.test(datamodule=dm)


if __name__ == '__main__':

    cli_main()
