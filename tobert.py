from functools import partial

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics import Accuracy, Metric
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


class ToBERTModel(torch.nn.Module):
    def __init__(self, num_labels, device):
        super(ToBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.trans = torch.nn.TransformerEncoderLayer(d_model=768, nhead=2)
        self.fc = torch.nn.Linear(768, 30)
        self.classifier = torch.nn.Linear(30, num_labels)
        self.device = device

    def forward(self, ids, mask, token_type_ids, length):
        _, pooled_out = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )

        chunks_emb = pooled_out.split_with_sizes(length)
        batch_emb_pad = torch.nn.utils.rnn.pad_sequence(
            chunks_emb, padding_value=0, batch_first=True
        )
        batch_emb = batch_emb_pad.transpose(0, 1)  # (B,L,D) -> (L,B,D)
        padding_mask = np.zeros([batch_emb.shape[1], batch_emb.shape[0]])
        for idx in range(len(padding_mask)):
            padding_mask[idx][length[idx] :] = 1  # padding key = 1 ignored

        padding_mask = torch.tensor(padding_mask).to(self.device, dtype=torch.bool)
        trans_output = self.trans(batch_emb, src_key_padding_mask=padding_mask)
        mean_pool = torch.mean(trans_output, dim=0)  # Batch size, 768
        fc_output = self.fc(mean_pool)
        relu_output = F.relu(fc_output)
        logits = self.classifier(relu_output)
        return logits


class ChunkDataset(Dataset):
    def __init__(self, text, labels, tokenizer, chunk_len=200, overlap_len=50):
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.overlap_len = overlap_len
        self.chunk_len = chunk_len

    def __len__(self):
        return len(self.labels)

    def chunk_tokenizer(self, tokenized_data, targets):
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []

        previous_input_ids = tokenized_data["input_ids"]
        previous_attention_mask = tokenized_data["attention_mask"]
        previous_token_type_ids = tokenized_data["token_type_ids"]
        remain = tokenized_data.get("overflowing_tokens")

        input_ids_list.append(torch.tensor(previous_input_ids, dtype=torch.long))
        attention_mask_list.append(
            torch.tensor(previous_attention_mask, dtype=torch.long)
        )
        token_type_ids_list.append(
            torch.tensor(previous_token_type_ids, dtype=torch.long)
        )
        targets_list.append(torch.tensor(targets, dtype=torch.long))

        if remain:  # if there is any overflowing tokens
            # remain = torch.tensor(remain, dtype=torch.long)
            idxs = range(len(remain) + self.chunk_len)
            idxs = idxs[
                (self.chunk_len - self.overlap_len - 2) :: (
                    self.chunk_len - self.overlap_len - 2
                )
            ]
            input_ids_first_overlap = previous_input_ids[-(self.overlap_len + 1) : -1]
            start_token = [101]
            end_token = [102]

            for i, idx in enumerate(idxs):
                previous_idx = idx
                if i == 0:
                    input_ids = input_ids_first_overlap + remain[:idx]
                elif i == len(idxs):
                    input_ids = remain[idx:]
                elif previous_idx >= len(remain):
                    break
                else:
                    input_ids = remain[(previous_idx - self.overlap_len) : idx]
                # previous_idx = idx
                nb_token = len(input_ids) + 2
                attention_mask = np.ones(self.chunk_len)
                attention_mask[nb_token : self.chunk_len] = 0
                token_type_ids = np.zeros(self.chunk_len)
                input_ids = start_token + input_ids + end_token
                if self.chunk_len - nb_token > 0:
                    padding = np.zeros(self.chunk_len - nb_token)
                    input_ids = np.concatenate([input_ids, padding])
                input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
                attention_mask_list.append(
                    torch.tensor(attention_mask, dtype=torch.long)
                )
                token_type_ids_list.append(
                    torch.tensor(token_type_ids, dtype=torch.long)
                )
                targets_list.append(torch.tensor(targets, dtype=torch.long))
        return {
            "ids": input_ids_list,
            "mask": attention_mask_list,
            "token_type_ids": token_type_ids_list,
            "targets": targets_list,
            "len": [torch.tensor(len(targets_list), dtype=torch.long)],
        }

    def __getitem__(self, index):
        text = " ".join(str(self.text[index]).split())
        targets = self.labels[index]

        data = self.tokenizer.encode_plus(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.chunk_len,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
            return_overflowing_tokens=True,
        )

        chunk_token = self.chunk_tokenizer(data, targets)
        return chunk_token


def chunk_collate_fn(batches):
    """
    Create batches for ChunkDataset
    """
    return [
        {key: torch.stack(value) for key, value in batch.items()} for batch in batches
    ]


def create_dataloader(
    dataset_class,
    text_set,
    label_set,
    tokenizer,
    max_length,
    batch_size,
    num_workers,
    dev_run,
):
    """
    Create appropriate dataloaders for the given data
    :param dataset_class: Dataset to use as defined in datasets.py
    :param text_set: dict of lists of texts for train/dev/test splits, keys=['train', 'dev', 'test']
    :param label_set: dict of lists of labels for train/dev/test splits, keys=['train', 'dev', 'test']
    :param tokenizer: tokenizer of choice e.g. LongformerTokenizer, BertTokenizer
    :param max_length: maximum length of sequence e.g. 512
    :param batch_size: batch size for dataloaders
    :param num_workers: number of workers for dataloaders
    :return: set of dataloaders for train/dev/test splits, keys=['train', 'dev', 'test']
    """
    dataloaders = {}
    loader = partial(
        DataLoader,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=chunk_collate_fn,
    )
    for split in ["train", "dev", "test"]:
        dataset = dataset_class(
            text_set[split], label_set[split], tokenizer, max_length
        )
        if dev_run:
            subset_indices = np.random.choice(len(dataset), batch_size).tolist()
            dataset = Subset(dataset, subset_indices)
        if split == "train":
            shuffle = True
        else:
            shuffle = False
        dataloaders[split] = loader(dataset, shuffle=shuffle)
    return dataloaders


def step(model, batch, device):
    ids = [data["ids"] for data in batch]
    mask = [data["mask"] for data in batch]
    token_type_ids = [data["token_type_ids"] for data in batch]
    targets = [data["targets"][0] for data in batch]
    length = [data["len"] for data in batch]
    ids = torch.cat(ids)
    mask = torch.cat(mask)
    token_type_ids = torch.cat(token_type_ids)
    targets = torch.stack(targets)
    length = torch.cat(length)
    length = [x.item() for x in length]
    ids = ids.to(device)
    mask = mask.to(device)
    token_type_ids = token_type_ids.to(device)
    targets = targets.to(device)
    logits = model(ids, mask, token_type_ids, length)
    return logits, targets


def train_step(model, batch, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    logits, targets = step(model, batch, device)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_step(model, loader, criterion, device):
    model.eval()
    loss = 0.0
    for batch in loader:
        logits, targets = step(model, batch, device)
        loss += criterion(logits, targets).item()
    loss = loss / len(loader)
    return loss


def train_model(
    model, criterion, optimizer, loader, dev_loader, max_epochs, model_path, device
):
    best_val_loss = float("inf")
    total_steps = max_epochs * len(loader)
    progress_bar = tqdm(total=total_steps, desc="Training")
    best_val_loss = float("inf")
    for _ in range(max_epochs):
        train_loss = 0.0
        for i, batch in enumerate(loader):
            train_loss += train_step(model, batch, optimizer, criterion, device)
            if i % 32 == 0:
                val_loss = eval_step(model, dev_loader, criterion, device)
                progress_bar.set_postfix(
                    {"train_loss": train_loss / (i + 1), "val_loss": val_loss}
                )
            progress_bar.update()
        train_loss /= len(loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
    progress_bar.close()


@torch.no_grad()
def evaluate(model, loader, metric: Metric, device):
    metric.to(device)
    for batch in loader:
        logits, targets = step(model, batch, device)
        preds = torch.argmax(logits, dim=1)
        metric.update(preds, targets.long())
    return metric.compute()


def main():
    seed = 42
    dev_run = False
    max_length = 200
    max_epochs = 2
    num_classes = 2
    batch_size = 4
    num_workers = 0
    test_size = 0.2
    model_path = "models/hyperpartisian.pt"

    torch.manual_seed(seed)

    df = pl.read_csv("data/raw/hyperpartisan.csv")
    df = df.rename({"Article ID": "id", "Content": "text"})
    df = df.with_columns(label=pl.col("Hyperpartisan").cast(pl.Int32))
    df = df.select("id", "label", "text")

    train, temp = train_test_split(df, test_size=test_size, random_state=seed)
    dev, test = train_test_split(temp, test_size=0.5, random_state=seed)

    text_set = {
        "train": train["text"].to_list(),
        "dev": dev["text"].to_list(),
        "test": test["text"].to_list(),
    }
    label_set = {
        "train": train["label"].to_list(),
        "dev": dev["label"].to_list(),
        "test": test["label"].to_list(),
    }
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataloaders = create_dataloader(
        dataset_class=ChunkDataset,
        text_set=text_set,
        label_set=label_set,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        num_workers=num_workers,
        dev_run=dev_run,
    )
    train_loader = dataloaders["train"]
    dev_loader = dataloaders["dev"]
    test_loader = dataloaders["test"]

    device = torch.device("mps" if torch.cuda.is_available() else "mps")
    model = ToBERTModel(num_labels=num_classes, device=device)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        dev_loader,
        max_epochs,
        model_path,
        device,
    )

    model = ToBERTModel(num_labels=num_classes, device=device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

    task = "binary" if num_classes == 2 else "multiclass"
    metric = Accuracy(task=task, num_classes=num_classes)
    accuracy = evaluate(model, test_loader, metric, device)
    test_loss = eval_step(model, test_loader, criterion, device)

    print(f"Test loss: {test_loss}, Test accuracy: {accuracy}")


if __name__ == "__main__":
    main()
