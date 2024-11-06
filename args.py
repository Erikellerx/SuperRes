import argparse
import torch

def get_args():
    
    parser = argparse.ArgumentParser(description='Super Resolution')
    parser.add_argument('--data-dir', type=str, default="F:\superRes\datasets\DIV2K")
    parser.add_argument('--batchsize', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test_batchsize', type=int, default=16,
                        help='input batch size for testing (default: 16)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model', type=str, default='SRCNN')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args