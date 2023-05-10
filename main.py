import pandas as pd
import numpy as np
from typing import List
from argparse import Namespace
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, AutoTokenizer, AutoModelWithLMHead, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tqdm.notebook import tqdm
import pytorch_lightning as pl


BATCH_SIZE_DEFAULT = 64

batch_size = BATCH_SIZE_DEFAULT
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from datasets import load_dataset

import pickle

#------------ CPU + GPU -------------#
if torch.cuda.is_available():
    device = torch.device(0)
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


#_____________________main______________________#

class ContrastiveDataBuilder(object):

    def __init__(self, path):
        self.data = []
        self.path = path
        self.save_path = "contrastive/train_cl.csv"

    def _read_file(self, type='SR_text'):
        dataframe = pd.read_csv(self.path)
        for id in dataframe.index:
            self.data.append([dataframe['text'][id], 0])
            self.data.append([dataframe[type][id], 0])
        self.data = self.data[:int(len(self.data) / batch_size) * batch_size]

    def _write_file(self, _index=False):
        pd.DataFrame(self.data, columns=["text", "label"]).to_csv(self.save_path, index=False)

    def load(self):
        self._read_file()
        self._write_file()


#---------------------- BUILD DATASET ----------------------------#
class EmoDataset(Dataset):

    def __init__(self, path):
        super().__init__()
        self.text_column = "text"
        self.label_column = "label"
        self.data = pd.read_csv(path, header=0, names=[self.text_column, self.label_column],
                                engine="python")

        self.emotions = list(set(self.data[self.label_column]))

        self.enum = {emo:id for id, emo in enumerate(self.emotions)}
        # print(self.enum)
    def list_emotions(self):
        return self.emotions

    def list_enum(self):
        return self.enum

    def __getitem__(self, idx):
        return self.data.loc[idx, self.text_column], self.enum[self.data.loc[idx, self.label_column]]

    def __len__(self):
        return self.data.shape[0]

class ContrastiveDataset(Dataset):

    def __init__(self, path):
        super().__init__()
        self.text_column = "text"
        self.label_column = "label"
        self.data = pd.read_csv(path)

    def __getitem__(self, idx):
        return self.data.loc[idx, self.text_column], self.data.loc[idx, self.label_column]

    def __len__(self):
        return self.data.shape[0]

#------------------Tokenizer----------------------#
class RoBERTaTokenizersCollateFn:
    def __init__(self, max_tokens=128):
        ## RoBERTa uses BPE tokenizer similar to GPT
        path_tokenizer = 'tokenizer/Distilroberta'
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


#---------------------MODEL PYTORCHLIGHTNING------------------------------#
@torch.jit.script
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

    def forward(self, input_, *args):
        X, attention_mask = input_
        hidden_states = self.base_model(X, attention_mask=attention_mask)

        return self.classifier(hidden_states[0][:, 0, :])


class ContrastiveModule(pl.LightningModule):
    def __init__(self, hparams, TokenizerFn):
        super().__init__()
        self.hparams.update(vars(hparams))
        self.model = EmoModel(AutoModelWithLMHead.from_pretrained(self.hparams.name_path).base_model, self.hparams.num_label)
        self.TokenizersCollateFn = TokenizerFn

    def loss(self, output_layer):

        def _dot_simililarity_dim1(x, y):
            v = torch.matmul(torch.unsqueeze(x, 1), torch.unsqueeze(y, 2))
            return v

        def _dot_simililarity_dim2(x, y):
            v = torch.tensordot(torch.unsqueeze(x, 1), torch.unsqueeze(torch.transpose(y, 0, 1), 0), dims=2)
            return v

        def get_negative_mask(batch_size):
            negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
            for i in range(batch_size):
                negative_mask[i, i] = 0
                negative_mask[i, i + batch_size] = 0
            return torch.tensor(negative_mask)

        criterion = torch.nn.CrossEntropyLoss()
        negative_mask = get_negative_mask(int(batch_size / 2))
        output_layer = output_layer.cpu()
        output_layer = output_layer.detach().numpy()
        zis = output_layer[::2]  # z0 z2 z4
        zjs = output_layer[1::2]  # z1 z3 z5

        zis = torch.from_numpy(zis)
        zjs = torch.from_numpy(zjs)
        zis = torch.nn.functional.normalize(zis, p=2, dim=1)

        zjs = normalized = torch.nn.functional.normalize(zjs, p=2, dim=1)
        l_pos = _dot_simililarity_dim1(zis, zjs)
        l_pos = torch.reshape(l_pos, (int(batch_size / 2), 1))
        l_pos /= 0.1

        negatives = torch.cat([zjs, zis], dim=0)
        loss = 0

        for positives in [zis, zjs]:
            l_neg = _dot_simililarity_dim2(positives, negatives)

            labels = torch.zeros(int(batch_size / 2), dtype=torch.long)

            l_neg = torch.masked_select(l_neg, negative_mask)
            l_neg = torch.reshape(l_neg, (int(batch_size / 2), -1))
            l_neg /= 0.1

            logits = torch.cat([l_pos, l_neg], axis=1)  # [N,K+1]
            loss += criterion(input=logits, target=labels)

        loss_tf = loss / (batch_size)
        loss_np = loss_tf.numpy()
        loss_torch = torch.tensor(loss_np, dtype=torch.float32, requires_grad=True)

        return loss_torch

    def step(self, batch, step_name="train"):
        if len(batch) != batch_size:
            return
        X, y = batch
        loss = self.loss(self.forward(X), self.hparams.batch_size)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        return {("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                "progress_bar": {loss_key: loss}}

    def forward(self, X, *args):
        return self.model(X, *args)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def train_dataloader(self):
        return self.create_data_loader(self.hparams.train_path, shuffle=False)

    def create_data_loader(self, ds_path: str, shuffle=False):
        return DataLoader(
            ContrastiveDataset(ds_path),
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            collate_fn=self.TokenizersCollateFn()
        )

    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.accumulate_grad_batches * self.hparams.epochs

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps(),
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


class TrainingModule(pl.LightningModule):
    def __init__(self, hparams, Tokenizer, model = None):
        super().__init__()
        self.hparams.update(vars(hparams))

        if model == None:
            self.model = EmoModel(AutoModelWithLMHead.from_pretrained(self.hparams.name_path).base_model, self.hparams.num_labels)
        else:
            self.model = model
        self.TokenizersCollateFn = Tokenizer
        self.loss = nn.CrossEntropyLoss()

    def step(self, batch, step_name="train"):
        X, y = batch
        loss = self.loss(self.forward(X), y)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        return {("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                "progress_bar": {loss_key: loss}}

    def forward(self, X, *args):
        return self.model(X, *args)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def validation_end(self, outputs: List[dict]):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def train_dataloader(self):
        return self.create_data_loader(self.hparams.train_path, shuffle=True)

    def val_dataloader(self):
        return self.create_data_loader(self.hparams.val_path)

    def test_dataloader(self):
        return self.create_data_loader(self.hparams.test_path)

    def create_data_loader(self, ds_path: str, shuffle=False):
        return DataLoader(
            EmoDataset(ds_path),
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            collate_fn=self.TokenizersCollateFn()
        )

    # @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.accumulate_grad_batches * self.hparams.epochs

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps(),
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]



if __name__ == '__main__':

    train_cl = False
    name_model = 'distilroberta-base'
    num_label = 6

    path_data = "data/DAIR-AI/"

    Tokenizer = RoBERTaTokenizersCollateFn
    CL = ContrastiveDataBuilder(path_data + "CL/train.csv")
    module = None
    if train_cl == True:
        CL.load()
        print('Training Contrastive Learning ....')

        hparams = Namespace(
            name_path="model/Distilroberta",
            train_path="contrastive/train_cl.csv",
            batch_size=64,
            warmup_steps=100,
            epochs=1,
            lr=1e-4,
            num_label = 6,
            accumulate_grad_batches=1
        )
        module = ContrastiveModule(hparams, Tokenizer)
        trainer = pl.Trainer(max_epochs=hparams.epochs,
                             accumulate_grad_batches=hparams.accumulate_grad_batches)

        trainer.fit(module)

    print('Training Tuning ....')

    hparams = Namespace(
        name_path="model/Distilroberta",
        train_path= path_data + "Tuning/train.csv",
        val_path= path_data + "Tuning/dev.csv",
        test_path=path_data + "Tuning/test.csv",
        batch_size=64,
        warmup_steps=100,
        epochs=1,
        num_labels = 6,
        lr=1e-4,
        accumulate_grad_batches=1
    )
    print(module)
    module = TrainingModule(hparams, Tokenizer, module)

    trainer = pl.Trainer(max_epochs=hparams.epochs,
                         accumulate_grad_batches=hparams.accumulate_grad_batches)

    trainer.fit(module)

    with torch.no_grad():
        progress = ["/", "-", "\\", "|", "/", "-", "\\", "|"]
        module = module.to(device)
        module.eval()
        true_y, pred_y = [], []
        for i, batch_ in enumerate(module.val_dataloader()):
            (X, attn), y = batch_
            batch = (X.to(device), attn.to(device))
            print(progress[i % len(progress)], end="\r")
            output = module(batch)
            y_pred = torch.argmax(output, dim=1)
            true_y.extend(y.cpu())
            pred_y.extend(y_pred.cpu())

    print("\n" + "_" * 80)

    print(classification_report(true_y, pred_y))
