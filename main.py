import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, AutoTokenizer, AutoModelWithLMHead, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tqdm.notebook import tqdm



import os
import pandas as pd
from sklearn.metrics import  classification_report
import matplotlib.pyplot as plt
from datasets import load_dataset

import pickle

# use cpu to train
device = torch.device("cpu")


def load_datasethug():
    dataset = load_dataset("dair-ai/emotion")
    print(dataset)
    text = []
    label = []

    for line in dataset['train']:
        text.append(line['text'])
        label.append(line['label'])
    dataset = pd.DataFrame({'text': text, 'label': label}, columns=['text', 'label'])
    dataset.to_csv('data/DairAIEmo/train.csv', index=False)

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained("tokenizer/Distilroberta")

def load_model(model_name):
    model = AutoModelWithLMHead.from_pretrained(name_model)
    model.save_pretrained('model/Distilroberta' , from_pt=True)
class LoaderDairAIEmo():
    def __init__(self):

        super().__init__()
        emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        enum = {
            "sadness": 0,
            "joy": 1,
            "love": 2,
            "anger": 3,
            "fear": 4,
            "surprise": 5
        }
        def load_from_pickle(directory):
            return pickle.load(open(directory, "rb"))

        data = load_from_pickle(directory="data/DairAIEmo/merged_training.pkl")

        data = data[data["emotions"].isin(emotions)]
        print(type(data))
        data.emotions.value_counts().plot.bar()
        plt.show()

        print(data.count())
        print(data.head(10))
        print(data.emotions.unique())

#_____________________main______________________#
class EmoDataset(Dataset):

    def __init__(self, path):

        super().__init__()
        self.emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        self.enum = {
            "sadness": 0,
            "joy": 1,
            "love": 2,
            "anger": 3,
            "fear": 4,
            "surprise": 5
        }
        self.text_column = "text"
        self.label_column = "label"
        self.data = pd.read_csv(path, header=0, names=[self.text_column, self.label_column],
                               engine="python")
    def list_emotions(self):
        return self.emotions
    def __getitem__(self, idx):
        return self.data.loc[idx, self.text_column], self.data.loc[idx, self.label_column]

    def __len__(self):
        return self.data.shape[0]


class TokenizersCollateFn:
    def __init__(self, path_tokenizer, max_tokens=512):
        ## RoBERTa uses BPE tokenizer similar to GPT
        t = ByteLevelBPETokenizer(
            str(path_tokenizer) + "/vocab.json",
            str(path_tokenizer) + "/merges.txt"
        )
        t._tokenizer.post_processor = BertProcessing(
            ("</s>", t.token_to_id("</s>")),
            ("<s>", t.token_to_id("<s>")),
        )
        t.enable_truncation(max_tokens)
        t.enable_padding(length=max_tokens, pad_id=t.token_to_id("<pad>"))
        self.tokenizer = t

    def __call__(self, batch):
        encoded = self.tokenizer.encode_batch([x[0] for x in batch])
        sequences_padded = torch.tensor([enc.ids for enc in encoded])
        attention_masks_padded = torch.tensor([enc.attention_mask for enc in encoded])
        labels = torch.tensor([x[1] for x in batch])

        return (sequences_padded, attention_masks_padded), labels

class Dataloader_full():
    def __init__(self, dataset, batch_size, train_path, val_path, test_path, path_tokenizer):
        super().__init__()
        self.dataset = dataset
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.path_tokenizer = path_tokenizer
        self.emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    def list_emotions(self):
        return self.emotions
    def train_dataloader(self):
        return self.create_data_loader(self.train_path, shuffle=True)

    def val_dataloader(self):
        return self.create_data_loader(self.val_path)

    def test_dataloader(self):
        return self.create_data_loader(self.test_path)

    def create_data_loader(self, ds_path: str, shuffle=False):
        return DataLoader(
            self.dataset(ds_path),
            batch_size=self.batch_size,
            collate_fn=TokenizersCollateFn(self.path_tokenizer)
        )


def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def forward(self, input):
        return mish(input)
class EmoModel(nn.Module):
    def __init__(self, base_model, n_classes, base_model_output_size=768, dropout=0.05):
        super().__init__()
        self.base_model = base_model

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, base_model_output_size),
            Mish(),
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, n_classes)
        )

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, input_ids, attention_mask, *args):

        hidden_states = self.base_model(input_ids = input_ids, attention_mask=attention_mask)
        return self.classifier(hidden_states[0][:, 0, :])

class RunModule():
    def __init__(self, base_model, dataloader, num_epochs):
        super().__init__()
        self.model = EmoModel(base_model, len(dataloader.list_emotions()))
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)
        self.dataloader = dataloader
        self.num_epochs = num_epochs

    def Training(self):
        print('Training ....')
        for epoch in tqdm(range(self.num_epochs)):
            print('Epoch {}/{}'.format(epoch + 1, self.num_epochs))
            for batch in self.dataloader.train_dataloader():
                X, y = batch
                output = self.model(X)
                loss = self.loss(output, y)

                loss.backward()
                self.optimizer.step()

                print(loss)
                break

class Distilroberta_base():

    def __init__(self):
        super().__init__()
        self.path_model = "model/Distilroberta"
        self.path_tokenizer = "tokenizer/Distilroberta"
        model = AutoModelWithLMHead.from_pretrained(self.path_model)
        self.base_model = model.base_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.path_tokenizer)

    def example_model(self, text = "I love you"):
        print("Name model: %s" % {str(name_model)})

        enc = self.tokenizer.encode_plus(text)
        print(enc["input_ids"])
        print(self.tokenizer.decode(enc["input_ids"]))
        print(f"Length: {len(enc['input_ids'])}")
        last_hidden_state = self.base_model(torch.tensor(enc["input_ids"]).unsqueeze(0))[0][0]
        print(last_hidden_state.shape)


if __name__ == '__main__':
    name_model = 'distilroberta-base'

    processors = {
        "RoBert": Distilroberta_base
    }

    datasets = {
        "Dair-AI-Emo": EmoDataset
    }

    pocessor = processors["RoBert"]()
    dataset = datasets["Dair-AI-Emo"]

    DLer = Dataloader_full(dataset = dataset,
                           batch_size = 32,
                           train_path ="data/DairAIEmo/train.csv",
                           val_path ="data/DairAIEmo/validation.csv",
                           test_path ="data/DairAIEmo/test.csv",
                           path_tokenizer="tokenizer/Distilroberta")

    #train = RunModule(base_model = pocessor.base_model, dataloader = DLer, num_epochs = 1)
    #train.Training()

    model = pocessor.base_model
    CEL = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-8)

    print('Training ....')
    for epoch in tqdm(range(2)):

        print('Epoch {}/{}'.format(epoch + 1, 1))
        model.train()
        for batch in tqdm(DLer.train_dataloader()):
            X, label = batch
            input_ids, attention_mask = X
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)

            output = model(input_ids, attention_mask)
            loss = CEL(output, label)

            loss.backward()
            optimizer.step()

    # processor.example_model("I love you.")

    # data = EmoDataset("data/train.csv")
    # print(data[1])