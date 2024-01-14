import os
import os.path
import re

import torch
from torch import nn
from torch.optim import Optimizer


class ModelCheckPoint:
    """ Helper class to save and load intermediary model states.

        Can be used to save :class:`torch.nn.Module` models with their optimizer :class:`torch.optim.Optimizer` for a given epoch. This is helpful if you intend to
        resume the training process starting from a previous epoch.
        It saves the model with `model_state_dict`, `optimizer_state_dict`, train and validation losses
        to a directory in the form::
            Models
            └── LID-DS-2019
                ├── CVE-2014-0160
                │    └── transformer
                │        └── ngram_length11_thread_awareTrue_anomaly_scoreMEAN_batch_size512
                │            ├── epochs1.model
                │            └── epochs2.model
                └── CVE-2017-7529
                     └── transformer
                         └── ngram_length11_thread_awareTrue_anomaly_scoreMEAN_batch_size512
                             ├── epochs2.model
                             └── epochs6.model

        Note:
            If run on a cluster, you might want to run all epochs on one node. Let's say node1 runs only until
            epoch 5 and node2 runs until epoch 10. Even thought it is possible to share the Models folder,
            node2 can not use the checkpoint at epoch 5 of node1 since node1 has not finished yet (if they are started
            at the same time).

    """

    def __init__(
            self,
            scenario_name: str,
            lid_ds_version_name: str,
            algorithm: str,
            algo_config: dict,
            models_dir: str = "Models",
    ):
        """
        Args:
            algo_config (dict): will be used to construct the model name. should be unique for each configuration.
            models_dir (str): base dir to save models.
        """
        self.model_path_base = os.path.join(models_dir, lid_ds_version_name, scenario_name, algorithm)
        self.model_name = '_'.join(''.join((key, str(val))) for (key, val) in algo_config.items())
        self.epochs_dir = os.path.join(self.model_path_base, self.model_name)

        os.makedirs(self.epochs_dir, exist_ok=True)

    def load(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            epoch: int = -1
    ) -> tuple[int, dict[int, float], dict[int, float], dict]:
        """ Load the recent checkpoint states to the given model and optimizer from a checkpoint

        If there exists a checkpoint with specified epoch it will be loaded. Else, the checkpoint with the highest epoch
        will be loaded and the epoch number will be returned. If there are no previous checkpoints nothing will be
        loaded and the returned epoch number is 0.

        Args:
            model (nn.Module): pytorch model
            optimizer (Optimizer): model optimizer
            epoch (int): epoch to load

        Returns:
            tuple: (last_epoch, train_losses, val_losses)
            last_epoch: same as `epoch` if checkpoint found, else the highest available epoch number
            losses: dictionaries of form {epoch: loss}
        """
        train_losses = {}
        val_losses = {}
        checkpoint = None

        saved_epochs = [f for f in os.listdir(self.epochs_dir) if f.endswith(".model")]
        saved_epochs = [int(re.findall(r'\d+', saved_epoch)[0]) for saved_epoch in saved_epochs]
        last_epoch = max(saved_epochs, default=0)

        if saved_epochs and last_epoch > epoch:
            last_epoch = max(e for e in saved_epochs if e <= epoch)
        if last_epoch > 0:
            epoch_path = os.path.join(self.epochs_dir, f"epochs{last_epoch}.model")
            checkpoint = torch.load(
                epoch_path,
                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            train_losses = checkpoint["train_losses"]
            val_losses = checkpoint["val_losses"]
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return last_epoch, train_losses, val_losses, checkpoint

    def save(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            epoch: int,
            train_losses: dict[int, float],
            val_losses: dict[int, float], **kwargs):
        """ Saves the model and optimizer states.

        Args:
            model (nn.Module): pytorch model
            optimizer (Optimizer): model optimizer
            epoch (int): epoch to load
            train_losses: list of train_losses up to this epoch
            val_losses: list of validation losses up to this epoch
        """
        epoch_path = os.path.join(self.epochs_dir, f"epochs{epoch}.model")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses
            } | kwargs,
            epoch_path
        )

    def load_epoch(self, epoch) -> dict:
        """ Load the  checkpoint for specified epoch

        Args:
            epoch (int): epoch to load

        Returns:
            dict: checkpoint
        """
        epoch_path = os.path.join(self.epochs_dir, f"epochs{epoch}.model")
        return torch.load(
            epoch_path,
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
