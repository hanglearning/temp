import os
import torch
import argparse
from datetime import datetime
import pickle
from pytorch_lightning import seed_everything
from model.cifar10.cnn import CNN as CIFAR_CNN
from model.cifar10.mlp import MLP as CIFAR_MLP
from model.mnist.cnn import CNN as MNIST_CNN
from model.mnist.mlp import MLP as MNIST_MLP
from server import Server
from client import Client
from util import create_model
import wandb
from dataset.datasource import DataLoaders
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
from pathlib import Path

models = {
    'cifar10': {
        'cnn': CIFAR_CNN,
        'mlp': CIFAR_MLP
    },
    'mnist': {
        'cnn': MNIST_CNN,
        'mlp': MNIST_MLP
    }
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="mnist|cifar10",
                        type=str, default="mnist")
    parser.add_argument('--arch', type=str, default='cnn', help='cnn|mlp')
    parser.add_argument('--dataset_mode', type=str,
                        default='non-iid', help='non-iid|iid')
    parser.add_argument('--rate_unbalance', type=float, default=1.0)
    parser.add_argument('--num_clients', type=int, default=12)
    parser.add_argument('--rounds', type=int, default=25)
    parser.add_argument('--prune_step', type=float, default=0.2)
    parser.add_argument('--prune_threshold', type=float, default=0.8)
    parser.add_argument('--server_prune', type=bool, default=False)
    parser.add_argument('--server_prune_step', type=float, default=0.2)
    parser.add_argument('--server_prune_freq', type=int, default=10)
    parser.add_argument('--frac_clients_per_round', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--n_class', type=int, default=3)
    parser.add_argument('--eita', type=float, default=0.5,
                        help="accuracy threshold")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="accuracy reduction factor")
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--train_verbose', type=bool, default=False)
    parser.add_argument('--test_verbose', type=bool, default=False)
    parser.add_argument('--prune_verbose', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--fast_dev_run', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    
    parser.add_argument('--start_diff', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default="Adam", help="SGD|Adam")
    parser.add_argument('--diff_freq', type=int, default=2)
    parser.add_argument('--rewind', type=int, default=0)
    parser.add_argument('--reinit', type=int, default=1)
    parser.add_argument('--project_name', type=str, default="CELL_dummy")
    parser.add_argument('--run_note', type=str, default="")
    parser.add_argument('--POLL', type=int, default=0)
    parser.add_argument('--no_prune', type=int, default=0)
    parser.add_argument('--stand_alone', type=int, default=0)
    parser.add_argument('--stand_alone_prune', type=int, default=0)

    parser.add_argument('--noise_variance', type=int, default=1, help="noise variance level of the injected Gaussian Noise")
    parser.add_argument('--n_malicious', type=int, default=0, help="number of malicious nodes in the network")

    parser.add_argument('-lb', '--logs_base_folder', type=str, default="/content/drive/MyDrive/POLL", help='base folder dir to store running logs')

    args = parser.parse_args()

    seed_everything(seed=args.seed, workers=True)

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {args.device}")

    model = create_model(cls=models[args.dataset]
                         [args.arch], device=args.device)

    exe_date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    log_dirpath = f"{args.logs_base_folder}/POLL_BASE/{exe_date_time}"
    os.makedirs(log_dirpath)
    
    args.log_dir = log_dirpath
    model_save_path = f"{args.log_dir}/models/globals"
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    torch.save(model, f"{model_save_path}/comm_0")

    train_loaders, test_loaders, global_test_loader = DataLoaders(num_users=args.num_clients,
                                              dataset_name=args.dataset,
                                              n_class=args.n_class,
                                              nsamples=args.n_samples,
                                              log_dirpath=log_dirpath,
                                              mode=args.dataset_mode,
                                              batch_size=args.batch_size,
                                              rate_unbalance=args.rate_unbalance,
                                              num_workers=args.num_workers
                                              )
    clients = []
    n_malicious = args.n_malicious
    for i in range(args.num_clients):
        malicious = True if args.num_clients - i <= n_malicious else False
        client = Client(i + 1, args, malicious, train_loaders[i], test_loaders[i], global_test_loader)
        clients.append(client)

    if args.POLL:
        run_name = "POLL" # synchronous pruninng
    elif args.no_prune:
        run_name = "NOPRUNE" # Pure FedAvg
    elif args.stand_alone:
        run_name = "STANDALONE" # Pure Centralized
    elif args.stand_alone_prune:
        run_name = "STANDALONE_PRUNE" # Centeralized with PoLL style pruninng
    else:
        run_name = "CELL"
    
    wandb.login()
    wandb.init(project=args.project_name, entity="hangchen")
    wandb.run.name = f"{run_name}_samples_{args.n_samples}_freq_{args.diff_freq}_n_clients_{args.num_clients}_mali_{args.n_malicious}_optim_{args.optimizer}_seed_{args.seed}_{args.run_note}_{exe_date_time}"
    wandb.config.update(args)

    server = Server(args, model, clients, global_test_loader)

    if args.no_prune:
        args.rounds += 1

    for i in range(args.rounds):
        server.update()

