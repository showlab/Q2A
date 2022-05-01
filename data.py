import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class EncodedAssistQA(Dataset):
    def __init__(self, cfg, is_train):
        super().__init__()
        # for cvpr challenge, root should be the folder of train / test set
        if cfg.DATASET.USAGE == "loveu":
            root = cfg.DATASET.TRAIN if is_train else cfg.DATASET.VAL
            samples = []
            for t in os.listdir(root):
                sample = torch.load(os.path.join(root, t, cfg.INPUT.QA), map_location="cpu")
                for s in sample:
                    s["video"] = os.path.join(root, t, cfg.INPUT.VIDEO)
                    s["script"] = os.path.join(root, t, cfg.INPUT.SCRIPT)
                samples.extend(sample)
            self.samples = samples
        
        # NOTE: This is the data splitting method used in https://arxiv.org/abs/2203.04203, not the cvpr challenge!
        if cfg.DATASET.USAGE == "assistq":
            samples = []
            for split in [cfg.DATASET.TRAIN, cfg.DATASET.VAL]:
                for t in os.listdir(split):
                    sample = torch.load(os.path.join(split, t, cfg.INPUT.QA), map_location="cpu")
                    for s in sample:
                        s["video"] = os.path.join(split, t, cfg.INPUT.VIDEO)
                        s["script"] = os.path.join(split, t, cfg.INPUT.SCRIPT)
                    samples.extend(sample)
            import random
            random.shuffle(samples)
            num_val = int(len(samples)*0.2)
            self.samples = samples[num_val:] if is_train else samples[:num_val]
    
    def __getitem__(self, index):
        sample = self.samples[index]
        video = torch.load(sample["video"], map_location="cpu")
        script = torch.load(sample["script"], map_location="cpu")
        question = sample["question"]
        actions = sample["answers"]
        label = torch.tensor(sample['correct']) - 1 # NOTE here
        return video, script, question, actions, label

    def __len__(self, ):
        return len(self.samples)

    @staticmethod
    def collate_fn(samples):
        return samples

class EncodedAssistQADataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def train_dataloader(self): 
        cfg = self.cfg
        trainset = EncodedAssistQA(cfg, is_train=True)
        return DataLoader(trainset, batch_size=cfg.SOLVER.BATCH_SIZE, collate_fn=EncodedAssistQA.collate_fn,
            shuffle=True, drop_last=True, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)

    def val_dataloader(self):
        cfg = self.cfg
        valset = EncodedAssistQA(cfg, is_train=False)
        return DataLoader(valset, batch_size=cfg.SOLVER.BATCH_SIZE, collate_fn=EncodedAssistQA.collate_fn,
            shuffle=False, drop_last=False, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)
    
def build_data(cfg):
    return EncodedAssistQADataModule(cfg)