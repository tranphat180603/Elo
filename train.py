#TODO: FINE-TUNING SCRIPT FOR MINICPM
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from abc import dataclass
import json
@dataclass
class Config():
    def __init__(self) -> None:
        vlm_model_name = "",
        


class Dataset:
    super.__init__(Dataset)
    def __init__(self, json_file, img_dir) -> object:
        with open(f"{json_file}", "r"):
            self.ds = json.loads(json_file)
        
    def __len__(self):
        pass
    def __get__item(self):
        pass

class DataLoader:
    super.__init__(DataLoader)
    def __init__(self) -> object:
        pass
    def _getlen():
        pass

class VLModel():
    super.__init__(nn.Module)
    def __init__(self) -> None:
        pass
    def forward() -> None:
        pass

def main():
    pass