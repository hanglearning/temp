import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
import numpy as np
import os
from typing import Dict
import copy
import math
import wandb
from torch.nn.utils import prune
from util import *
from util import train as util_train
from util import test as util_test
import time


class Client():
    def __init__(
        self,
        idx,
        args,
        is_malicious,
        train_loader=None,
        test_loader=None,
        global_test_loader=None,
        **kwargs
    ):
        self.idx = idx
        self.args = args
        self.is_malicious = is_malicious
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.global_test_loader = global_test_loader

        self.eita_hat = self.args.eita
        self.eita = self.eita_hat
        self.alpha = self.args.alpha
        self.num_data = len(self.train_loader)

        self.elapsed_comm_rounds = 0
        self.last_prune_diff = args.start_diff

        self.accuracies = []
        self.losses = []
        self.prune_rates = []
        self.cur_prune_rate = 0.00

        self.model = None
        self.global_model = None
        self.global_init_model = None

    def poison_model(self):
        for layer, module in self.model.named_children():
            for name, weight_params in module.named_parameters():
                if "weight" in name:
                    noise = self.args.noise_variance * torch.randn(weight_params.size())
                    # variance_of_noise = torch.var(noise)
                    weight_params.add_(noise.to(self.args.device))
        print(f"Device {self.idx} has poisoned its model.")

    def update_standalone(self) -> None:

        self.model = self.global_model

        for i in range(self.args.rounds):
            
            print("\nRound", self.elapsed_comm_rounds + 1)
            
            self.train(self.elapsed_comm_rounds)
            print(f"\n{self.idx} Evaluating Trained Model")
            metrics = self.eval(self.model)
            print(f'Trained model accuracy: {metrics["Accuracy"][0]}')

            wandb.log({f"{self.idx}_acc": metrics["Accuracy"][0], "comm_round": self.elapsed_comm_rounds + 1})

            self.elapsed_comm_rounds += 1

    def update_standalone_prune(self) -> None:

        self.model = self.global_model

        for i in range(self.args.rounds):

            start_diff = 0
            curr_diff = round(min(self.args.prune_threshold, start_diff + (self.elapsed_comm_rounds // self.args.diff_freq) * self.args.prune_step), 2)
            
            print("\nRound", self.elapsed_comm_rounds + 1)

            l1_prune(model=self.model,
                    amount=curr_diff,
                    name='weight',
                    verbose=self.args.prune_verbose)

            print(f"{self.idx} pruned {curr_diff} in round {self.elapsed_comm_rounds + 1}.")

            prune_rate = get_prune_summary(model=self.model,name='weight')['global']
            print(f"Sparcity {1 - get_prune_summary(model=self.model,name='weight')['global']}")

            if self.args.reinit:
                if self.elapsed_comm_rounds > 0 and self.elapsed_comm_rounds % self.args.diff_freq == 0 and self.elapsed_comm_rounds < self.args.diff_freq * int(self.args.prune_threshold/self.args.prune_step):
                    # reinitialize model with init_params
                    source_params = dict(self.global_init_model.named_parameters())
                    for name, param in self.global_model.named_parameters():
                        param.data.copy_(source_params[name].data)
                    print(f"{self.idx} reinited in round {self.elapsed_comm_rounds + 1}.")

            self.train(self.elapsed_comm_rounds)

            print(f"\n{self.idx} Evaluating Trained Model")
            metrics = self.eval(self.model)
            print(f'Trained model accuracy: {metrics["Accuracy"][0]}')

            wandb.log({f"{self.idx}_acc": metrics["Accuracy"][0], "comm_round": self.elapsed_comm_rounds + 1})
            wandb.log(
            {f"{self.idx}_percent_pruned": prune_rate})

            self.elapsed_comm_rounds += 1

    def update(self) -> None:
        """
            Interface to Server
        """
        print(f"\n----------Client:{self.idx} Update---------------------")

        print(f"Evaluating Global model ")
        metrics = self.eval(self.global_model)
        accuracy = metrics['Accuracy'][0]
        print(f'Global model accuracy: {accuracy}')

        prune_rate = get_prune_summary(model=self.global_model,
                                       name='weight')['global']
        print('Global model prune percentage: {}'.format(prune_rate))

        if self.args.POLL:
                    
            curr_diff = round(min(self.args.prune_threshold, self.args.start_diff + ((self.elapsed_comm_rounds + 1) // self.args.diff_freq) * self.args.prune_step), 2)

            reinit = False
            if curr_diff > self.last_prune_diff:
                reinit = True
                self.last_prune_diff = curr_diff
            
            params_pruned = get_prune_params(self.global_model, name='weight')
            for param, name in params_pruned:
                prune.remove(param, name)
            
            l1_prune(model=self.global_model,
                    amount=curr_diff,
                    name='weight',
                    verbose=self.args.prune_verbose)
            print(f"{self.idx} pruned {curr_diff} in round {self.elapsed_comm_rounds + 1}.")
            
            prune_rate = get_prune_summary(model=self.global_model,name='weight')['global']
            print(f"Sparcity {round(1 - get_prune_summary(model=self.global_model,name='weight')['global'], 2)}")

            # how about do not reinit
            if self.args.reinit and reinit:
                # reinitialize model with init_params
                source_params = dict(self.global_init_model.named_parameters())
                for name, param in self.global_model.named_parameters():
                    param.data.copy_(source_params[name].data)
                print(f"{self.idx} reinited in round {self.elapsed_comm_rounds + 1}.")

            self.prune_rates.append(prune_rate)
            self.model = self.global_model

        elif self.args.no_prune:
            self.prune_rates.append(1.0)
            self.model = self.global_model

            # in this mode, also need to test for global model on individual test set
            if self.elapsed_comm_rounds: # skip initial model
                print(f"\nEvaluating the latest global model on local test set")
                acc = self.eval(self.model)["Accuracy"][0]
                print(f'Global model on local test set accuracy: {acc}')
                wandb.log({"comm_round": self.elapsed_comm_rounds, "global_model_local_set_acc": acc})

        else:

            if self.cur_prune_rate < self.args.prune_threshold:
                if accuracy > self.eita:
                    self.cur_prune_rate = min(self.cur_prune_rate + self.args.prune_step,
                                            self.args.prune_threshold)
                    if self.cur_prune_rate > prune_rate:
                        # print("pruned amount by weights before l1:", get_pruned_amount_by_weights(self.global_model))
                        # print("get_pruned_amount_by_weights", get_pruned_amount_by_weights(self.global_model), "==", prune_rate)
                        # print("get_pruned_amount_by_mask", get_pruned_amount_by_mask(self.global_model), "==", prune_rate)
                        l1_prune(model=self.global_model,
                                amount=self.cur_prune_rate - prune_rate,
                                name='weight',
                                verbose=self.args.prune_verbose)
                        # print("get_pruned_amount_by_mask", get_pruned_amount_by_mask(self.global_model), "==", prune_rate)
                        # print("===============================")
                        # print("self.cur_prune_rate", self.cur_prune_rate)
                        # print("prune_rate", prune_rate)
                        # print("amount", self.cur_prune_rate - prune_rate)
                        # print("pruned amount by weights:", get_pruned_amount_by_weights(self.global_model))
                        # print(get_pruned_amount_by_weights(self.global_model), "==", self.cur_prune_rate - prune_rate)
                        # print("===============================")
                        self.prune_rates.append(self.cur_prune_rate)
                        # print()
                    else:
                        self.prune_rates.append(prune_rate)
                    # reinitialize model with init_params
                    source_params = dict(self.global_init_model.named_parameters())
                    for name, param in self.global_model.named_parameters():
                        param.data.copy_(source_params[name].data)

                    self.model = self.global_model
                    self.eita = self.eita_hat

                else:
                    self.eita *= self.alpha
                    self.model = self.global_model
                    self.prune_rates.append(prune_rate)
            else:
                if self.cur_prune_rate > prune_rate:
                    l1_prune(model=self.global_model,
                            amount=self.cur_prune_rate-prune_rate,
                            name='weight',
                            verbose=self.args.prune_verbose)
                    self.prune_rates.append(self.cur_prune_rate)
                else:
                    self.prune_rates.append(self.cur_prune_rate)
                self.model = self.global_model

        print(f"\nTraining local model")

        start_time = time.time()
        self.train(self.elapsed_comm_rounds)
        training_time = round(time.time() - start_time, 4)
        print(f"Training time {training_time}s")
        wandb.log({f"comm_round": self.elapsed_comm_rounds + 1})
        wandb.log({f"{self.idx}_training_time": training_time})

        if self.args.POLL and self.is_malicious:
            print(f"\nBefore poisoning model, evaluating Trained Model")
            metrics = self.eval(self.model)
            print(f'Trained model accuracy: {metrics["Accuracy"][0]}')
            self.poison_model()

        if not self.args.no_prune:
            # when no_prune mode (pure FedAvg), test global model above
            print(f"\nEvaluating Trained Model")
            metrics = self.eval(self.model)
            print(f'Trained model accuracy: {metrics["Accuracy"][0]}')

        
        wandb.log({f"{self.idx}_cur_prune_rate": self.cur_prune_rate})
        wandb.log({f"{self.idx}_eita": self.eita})
        wandb.log(
            {f"{self.idx}_percent_pruned": self.prune_rates[-1]})

        for key, thing in metrics.items():
            if(isinstance(thing, list)):
                wandb.log({f"{self.idx}_{key}": thing[0]})
            else:
                wandb.log({f"{self.idx}_{key}": thing})

        if (self.elapsed_comm_rounds+1) % self.args.save_freq == 0:
            self.save(self.model)

        self.elapsed_comm_rounds += 1

    def train(self, round_index):
        """
            Train NN
        """
        losses = []

        for epoch in range(self.args.epochs):
            if self.args.train_verbose:
                print(
                    f"Client={self.idx}, epoch={epoch}, round:{round_index}")

            metrics = util_train(self.model,
                                 self.train_loader,
                                 self.args.optimizer,
                                 self.args.lr,
                                 self.args.device,
                                 self.args.fast_dev_run,
                                 self.args.train_verbose)
            losses.append(metrics['Loss'][0])

            if self.args.fast_dev_run:
                break
        self.losses.extend(losses)

    @torch.no_grad()
    def download(self, global_model, global_init_model, *args, **kwargs):
        """
            Download global model from server
        """
        self.global_model = global_model
        self.global_init_model = global_init_model

        params_to_prune = get_prune_params(self.global_model)
        for param, name in params_to_prune:
            weights = getattr(param, name)
            masked = torch.eq(weights.data, 0.00).sum().item()
            # masked = 0.00
            prune.l1_unstructured(param, name, amount=int(masked))

        params_to_prune = get_prune_params(self.global_init_model)
        for param, name in params_to_prune:
            weights = getattr(param, name)
            masked = torch.eq(weights.data, 0.00).sum().item()
            # masked = 0.00
            prune.l1_unstructured(param, name, amount=int(masked))

    def eval(self, model):
        """
            Eval self.model
        """
        eval_score = util_test(model,
                               self.test_loader,
                               self.args.device,
                               self.args.fast_dev_run,
                               self.args.test_verbose)
        self.accuracies.append(eval_score['Accuracy'][0])
        # TODO - test on global test set
        return eval_score

    def save(self, *args, **kwargs):
        pass

    def upload(self, *args, **kwargs) -> Dict[nn.Module, float]:
        """
            Upload self.model
        """
        upload_model = copy_model(model=self.model, device=self.args.device)
        params_pruned = get_prune_params(upload_model, name='weight')
        for param, name in params_pruned:
            prune.remove(param, name)
        return {
            'model': upload_model,
            'acc': self.accuracies[-1]
        }
