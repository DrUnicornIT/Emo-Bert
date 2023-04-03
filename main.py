import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, AutoTokenizer, AutoModelWithLMHead, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import os
import pandas as pd
from sklearn.metrics import  classification_report

print(torch.__version__)

class Distilroberta_base():
    name_model = "distilroberta-base"

    def printLog(self):
        print("Name model: %s" %{str(name_model)})

    def tokenizer(self):
        return AutoTokenizer.from_pretrained(name_model)
    def model(self):
        model = AutoModelWithLMHead.from_pretrained("distilroberta-base")
        # model = torch.load("./distilroberta-base/pytorch_model.bin")
        return model.base_model

    def example_tokenizer(self, text):
        enc = self.tokenizer().encode_plus(text)
        print(text)
        print(enc)

if __name__ == '__main__':
    name_model = 'distilroberta-base'
    processors = {
        "RoBert": Distilroberta_base
    }

    processor = processors["RoBert"]()

    processor.printLog()
    # model = processor.model()
    processor.example_tokenizer("I love you.")