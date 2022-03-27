import os
import math
from numpy import full_like
import torch
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from ..model import *
from .model import *
from .dataset import IemoH5LastLayerDataset, IEMOCAPDataset, collate_fn

import wandb

class DownstreamExpert_orig(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert_orig, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        DATA_ROOT = self.datarc['root']
        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant")
        if self.fold is None:
            self.fold = "fold1"

        print(f"[Expert] - using the testing fold: \"{self.fold}\". Ps. Use -o config.downstream_expert.datarc.test_fold=fold2 to change test_fold in config.")

        train_path = os.path.join(
            DATA_ROOT, 'meta_data', self.fold.replace('fold', 'Session'), 'train_meta_data.json')
        print(f'[Expert] - Training path: {train_path}')

        test_path = os.path.join(
            DATA_ROOT, 'meta_data', self.fold.replace('fold', 'Session'), 'test_meta_data.json')
        print(f'[Expert] - Testing path: {test_path}')
        
        dataset = IEMOCAPDataset(DATA_ROOT, train_path, self.datarc['pre_load'])
        trainlen = int((1 - self.datarc['valid_ratio']) * len(dataset))
        lengths = [trainlen, len(dataset) - trainlen]
        
        torch.manual_seed(0)
        self.train_dataset, self.dev_dataset = random_split(dataset, lengths)

        self.test_dataset = IEMOCAPDataset(DATA_ROOT, test_path, self.datarc['pre_load'])

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = dataset.class_num,
            **model_conf,
        )
        self.objective = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))


    def get_downstream_name(self):
        return self.fold.replace('fold', 'emotion')


    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, filenames, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)

        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)

        labels = torch.LongTensor(labels).to(features.device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        records["filename"] += filenames
        records["predict"] += [self.test_dataset.idx2emotion[idx] for idx in predicted_classid.cpu().tolist()]
        records["truth"] += [self.test_dataset.idx2emotion[idx] for idx in labels.cpu().tolist()]

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'emotion-{self.fold}/{mode}-{key}',
                average,
                global_step=global_step
            )

            # Wandb logging
            full_key = f'emotion-{self.fold}/{mode}-{key}'
            wandb.log({full_key: average})

            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["predict"])]
                file.writelines(line)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["truth"])]
                file.writelines(line)

        return save_names


# A subclass of original DownstreamExpert_orig to allow overwriting relevant properties and interface methods to adapt the expert as we wish, e.g. for loading from the pre-saved last_layer H5 Dataset instead
class DownstreamExpert(DownstreamExpert_orig):
    """
    Comment out this class and rename DownstreamExpert_orig above to "DownstreamExpert" to regain original functionality.
    Or maybe just add code to the bottom of this script? lol.
    """
    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        # Run super of the ORIGINAL downstream expert - we don't want to waste the time processing the undesired Dataset needlessly
        super(DownstreamExpert_orig, self).__init__()

        # Set basic necessary properties
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        IEMOCAP_HOME = self.datarc['root']
        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant")
        if self.fold is None:
            self.fold = "fold1"

        print(f"[MODIFIED Expert] - using the testing fold: \"{self.fold}\". Ps. Use -o config.downstream_expert.datarc.test_fold=fold2 to change test_fold in config.")        
        
        # Create desired Dataset iterators
        dataset = IemoH5LastLayerDataset(IEMOCAP_HOME, self.fold, 'train', False) #self.datarc['pre_load'])
        trainlen = int((1 - self.datarc['valid_ratio']) * len(dataset))
        lengths = [trainlen, len(dataset) - trainlen]
        
        torch.manual_seed(0)
        self.train_dataset, self.dev_dataset = random_split(dataset, lengths)

        self.test_dataset = IemoH5LastLayerDataset(IEMOCAP_HOME, self.fold, 'test', False) #self.datarc['pre_load'])

        # Select out the desired model
        # TODO: What exactly is projector??
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = dataset.class_num,
            **model_conf,
        )

        # Set Loss objective
        self.objective = nn.CrossEntropyLoss()
        
        # Some other properties
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))

    
    # Don't need no FUCKING filenames.
    def forward(self, mode, features, labels, records, **kwargs):        
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)

        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)

        labels = torch.LongTensor(labels).to(features.device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        # Don't need no FUCKING filenames.
        # records["filename"] += filenames
        records["predict"] += [self.test_dataset.idx2emotion[idx] for idx in predicted_classid.cpu().tolist()]
        records["truth"] += [self.test_dataset.idx2emotion[idx] for idx in labels.cpu().tolist()]

        return loss



