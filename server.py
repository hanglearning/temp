import wandb
from typing import List, Dict, Tuple
import torch.nn.utils.prune as prune
import numpy as np
import random
import os
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from util import get_prune_params, super_prune, fed_avg, l1_prune, create_model, copy_model, get_prune_summary
from util import test_by_data_set
from util import *



class Server():
    """
        Central Server
    """

    def __init__(
        self,
        args,
        model,
        clients,
        global_test_loader
    ):
        super().__init__()
        self.clients = clients
        self.num_clients = len(self.clients)
        self.args = args
        self.model = model
        self.init_model = copy_model(model, self.args.device)

        self.global_test_loader = global_test_loader

        self.elapsed_comm_rounds = 0
        self.curr_prune_step = 0.00

    def aggr(
        self,
        models,
        clients,
        *args,
        **kwargs
    ):
        weights_per_client = np.array(
            [client.num_data for client in clients], dtype=np.float32)
        weights_per_client /= np.sum(weights_per_client)

        aggr_model = fed_avg(
            models=models,
            weights=weights_per_client,
            device=self.args.device
        )
        pruned_percent = get_prune_summary(aggr_model, name='weight')['global']
        # pruned by the earlier zeros in the model
        l1_prune(aggr_model, amount=pruned_percent, name='weight')

        return aggr_model

    def update(
        self,
        *args,
        **kwargs
    ):
        """
            Interface to server and clients
        """
        self.elapsed_comm_rounds += 1
        print('-----------------------------', flush=True)
        print(
            f'| Communication Round: {self.elapsed_comm_rounds}  | ', flush=True)
        print('-----------------------------', flush=True)

        # global_model pruned at fixed freq
        # with a fixed pruning step
        if (self.args.server_prune == True and
                (self.elapsed_comm_rounds % self.args.server_prune_freq) == 0):
            # prune the model using super_mask
            self.curr_prune_step += self.args.prune_step
            super_prune(
                model=self.model,
                init_model=self.init_model,
                amount=self.curr_prune_step,
                name='weight'
            )
            # reinitialize model with std.dev of init_model
            source_params = dict(self.init_model.named_parameters())
            for name, param in self.model.named_parameters():
                std_dev = torch.std(source_params[name].data)
                param.data.copy_(std_dev*torch.sign(source_params[name].data))

        client_idxs = np.random.choice(
            self.num_clients, int(
                self.args.frac_clients_per_round*self.num_clients),
            replace=False,
        )
        clients = [self.clients[i] for i in client_idxs]

        # upload model to selected clients
        self.upload(clients)

        # call training loop on all clients
        for client in clients:
            if self.args.stand_alone:
                client.update_standalone()
            elif self.args.stand_alone_prune:
                client.update_standalone_prune()
            else:
                client.update()

        if self.args.stand_alone or self.args.stand_alone_prune:
            import sys
            sys.exit()

        # download models from selected clients
        models, accs = self.download(clients)

        avg_accuracy = np.mean(accs, axis=0, dtype=np.float32)
        print('-----------------------------', flush=True)
        print(f'| Average Accuracy: {avg_accuracy}  | ', flush=True)
        print('-----------------------------', flush=True)
        if self.args.no_prune:
            # notice that if args.no_prune, this returns the average global test accuracy on local test sets
            wandb.log({"client_fedavg_global_test_acc": avg_accuracy, "comm_round": self.elapsed_comm_rounds - 1})
        if not self.args.no_prune: # CELL and POLL
            wandb.log({"client_avg_acc": avg_accuracy, "comm_round": self.elapsed_comm_rounds})

        # average accuracy is the accuracy AFTER training, that's okay, because it's the ticket model

        # compute average-model
        aggr_model = self.aggr(models, clients)

        model_save_path = f"{self.args.log_dir}/models/globals_0"
        trainable_model_weights = get_trainable_model_weights(aggr_model)
        with open(f"{model_save_path}/{self.elapsed_comm_rounds}.pkl", 'wb') as f:
            pickle.dump(trainable_model_weights, f)

        # test on global test set
        aggr_model_acc = test_by_data_set(aggr_model,
                               self.global_test_loader,
                               self.args.device,
                               self.args.test_verbose)['Accuracy'][0]
        print(f'global test set accuracy: {aggr_model_acc}')
        wandb.log({f"comm_round": self.elapsed_comm_rounds, "global_test_acc": aggr_model_acc})

        # copy aggregated-model's params to self.model (keep buffer same)
        source_params = dict(aggr_model.named_parameters())
        for name, param in self.model.named_parameters():
            param.data.copy_(source_params[name])

    def download(
        self,
        clients,
        *args,
        **kwargs
    ):
        # downloaded models are non pruned (taken care of in fed-avg)
        uploads = [client.upload() for client in clients]
        models = [upload["model"] for upload in uploads]
        accs = [upload["acc"] for upload in uploads]
        return models, accs

    def save(
        self,
        *args,
        **kwargs
    ):
        # """
        #     Save model,meta-info,states
        # """
        # eval_log_path1 = f"./log/full_save/server/round{self.elapsed_comm_rounds}_model.pickle"
        # eval_log_path2 = f"./log/full_save/server/round{self.elapsed_comm_rounds}_dict.pickle"
        # if self.args.report_verbosity:
        #     log_obj(eval_log_path1, self.model)
        pass

    def upload(
        self,
        clients,
        *args,
        **kwargs
    ) -> None:
        """
            Upload global model to clients
        """
        for client in clients:
            # make pruning permanent and then upload the model to clients
            model_copy = copy_model(self.model, self.args.device)
            init_model_copy = copy_model(self.init_model, self.args.device)

            params = get_prune_params(model_copy, name='weight')
            for param, name in params:
                prune.remove(param, name)

            init_params = get_prune_params(init_model_copy)
            for param, name in init_params:
                prune.remove(param, name)
            # call client method
            client.download(model_copy, init_model_copy)
