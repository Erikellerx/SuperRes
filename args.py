import argparse
import torch

from baseline import model_names

def get_args():
    parser = argparse.ArgumentParser(
        description='Super Resolution'
    )
    parser.add_argument('experiment_name', type=str,
                        help='name of the experiment (required)')
    
    parser.add_argument('--model', type=str, default='SRCNN', choices=model_names,
                        help='model name (default: SRCNN)')
    parser.add_argument('--data-dir', type=str, default="./data/DIV2K",
                        help='path to dataset (default: ./data/DIV2K)')
    parser.add_argument('--log-dir', type=str, default="./logs",
                        help='path to log (default: ./logs)')
    parser.add_argument('--batch', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test_batch', type=int, default=16,
                        help='input batch size for testing (default: 16)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', action='store_true',
                        help='resume from existing log directory?')
    
    parser.add_argument('--criterion', type=str, default='mse', choices=['mse', 'l1'],
                        help='loss function (default: mse)')
    
    parser.add_argument('--optimizer', type=str, default='adam+1e-4+1e-8+0',
                        help="""
                        optimizer (default: adam+1e-4+1e-8+0)
                        Two options available: sgd for adam.
                        For SGD: format is sgd+[lr]+[momentum]+[weight_decay]+[nesterov (True|False)]
                        For Adam: format is adam+[lr]+[eps]+[weight_decay]
                        """)
    
    parser.add_argument('--scheduler', type=str, default='none',
                        help="""
                        scheduler (default: none)
                        Two options available: step, cosine
                        For step: format is step+[step_size]+[gamma]
                        For cosine: format is cosine+[t_max]+[eta_min]
                        """)
    
    # Method specific parameters
    parser.add_argument('--gray-scale', action='store_true',
                        help='convert images to grayscale?')

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args