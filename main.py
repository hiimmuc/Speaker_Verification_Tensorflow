import argparse

from model import *
from trainer import Trainer
from ultis import load_dataset


def main(args):
    # load dataset
    train_dataset = load_dataset(args, mode='train')
    dev_dataset = load_dataset(args, mode='dev')
    test_dataset = load_dataset(args, mode='test')

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train(val_on_train=False)
        trainer.plot_graph()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Control flow
    parser.add_argument("--do_train", default=False,
                        action='store_true', help="Train model?")
    parser.add_argument("--do_eval", default=False,
                        action='store_true', help="valid model?")
    # hyperparameters
    parser.add_argument("--model", default="resnet34", type=str,
                        help="The backbone model. Available: resnet18, resnet34, resnet50, resnet101, resnet152,  vgg-m, vgg-custom, vgg16")
    parser.add_argument("--epochs", default=25, type=int,
                        help="Number of training epochs")
    parser.add_argument("--optimizer", default="Adam", type=str,
                        help="Optimizer for training")
    parser.add_argument("--train_batch_size", default=32,
                        type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64,
                        type=int, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=3e-4,
                        type=float, help="The initial learning rate")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Momentum for SGD optimizer")
    parser.add_argument("--nesterov", default=True, type=bool,
                        help="Nesterov for SGD optimizer")
    # Load and save config
    parser.add_argument("--data_dir", default="./dataset",
                        type=str, help="The input data dir")
    parser.add_argument("--save_dir", default="./backup", type=str,
                        help="saved directory for weight, logging files")
    parser.add_argument("--use_pretrained", default=False, type=bool,
                        help="decide wether train from scratch or saved checkpoints")

    args = parser.parse_args()
    main(args)
