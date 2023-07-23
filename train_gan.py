"""
Adapted from the original content available on the Github repository:

https://github.com/ETZET/MCMC_GAN

Changes made in this version:

1) Dependency on the "wandb" library is removed.

2) This script reads data from a ".mat" file.

3) Minor changes are performed on the parser.

"""
import os
import pickle
import argparse
import torch
import numpy as np
from generative_model import WGAN_SIMPLE
import scipy.io


def train_wgan_simple(args):
    """
    user training program
    :param args: Namespace, provide training parameters
    """

    print("Reading Data...")
    mat_file = scipy.io.loadmat(args.input_path)
    data = mat_file["ensemble_data"]
    data = np.reshape(data, (data.shape[0],-1))

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("currently using device:", device)

    config = dict(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        training_epoch=args.epochs,
        batch_size=args.batch_size,
        architecture="WGAN_MLP",
        data=args.input_path,
        device=device
    )

    # initialize model
    model = WGAN_SIMPLE(ndim=data.shape[1], device=device,nhid=300)

    model.optimize(data,output_path=args.output_path,
                   batch_size=config['batch_size'], epochs=config['training_epoch'],
                   lr=config['learning_rate'], beta1=config['momentum'],
                   device=device)



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=str, required=True, help="Input path")
    parser.add_argument("-o", "--output-path", type=str,required=True, help="Output path")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("-b", "--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--momentum", type=float, default=0.5, help="Momentum")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    # optimization
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    train_wgan_simple(args)
