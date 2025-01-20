import torch
import argparse
from model import train

FType = torch.FloatTensor
LType = torch.LongTensor


def main_train(args):
    the_train = train.MEGAD(args)
    the_train.train()

if __name__ == '__main__':
    data = 'disney'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=data)
    parser.add_argument('--clusters', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--neg_size', type=int, default=2)
    parser.add_argument('--hist_len', type=int, default=10)
    parser.add_argument('--save_step', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--directed', type=bool, default=False)
    args = parser.parse_args()
    main_train(args)
