import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class EncodedAssistQA(Dataset):
    def __init__(self, cfg, is_train):
        super().__init__()
        # for loveu@cvpr2022. paper is updated for this.
        root = cfg.DATASET.TRAIN if is_train else cfg.DATASET.VAL
        samples = []
        for t in os.listdir(root):
            sample = torch.load(os.path.join(root, t, cfg.INPUT.QA), map_location="cpu")
            for s in sample:
                s["video"] = os.path.join(root, t, cfg.INPUT.VIDEO)
                s["script"] = os.path.join(root, t, cfg.INPUT.SCRIPT)
            samples.extend(sample)
        self.samples = samples
        
    def __getitem__(self, index):
        sample = self.samples[index]
        video = torch.load(sample["video"], map_location="cpu")
        script = torch.load(sample["script"], map_location="cpu")
        question = sample["question"]
        actions = sample["answers"]
        meta = {'question': sample['src_question'], 'folder': sample['folder']}
        if 'correct' in sample:
            label = torch.tensor(sample['correct']) - 1 # NOTE here, start from 1
        else:
            label = None
        return video, script, question, actions, label, meta
        
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