# Classify Parliamentary Motions {#methods}



## Download the data

## Import libraries and set device


```{.python .fold-show}
import pandas as pd
import torch
from sklearn import metrics
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, AutoTokenizer, AdamW

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
```



## Read data


```{.python .fold-show}
df = pd.read_feather("motioner_2018_2021.feather")
df = df[df["single_party_authors"] == True]
# df = df[df["subtyp"] == "Enskild motion"]
label_mapping = {
    0: "V",
    1: "S",
    2: "MP",
    3: "C",
    4: "L",
    5: "M",
    6: "KD",
    7: "SD",
    8: "independent",
}
label_mapping = {v: k for k, v in label_mapping.items()}
df["label"] = df["party"].map(label_mapping)
df = df.reset_index(drop=True)
```

### Split into train and validation


```{.python .fold-show}
df_train = df.sample(frac=0.85, random_state=5)
df_valid = df.drop(df_train.index)
```

## Create Dataset and DataLoader


```{.python .fold-show}
class MotionerDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        df_row = self.df.iloc[index]

        label = df_row["label"]
        text = df_row["text"]

        tokenized_text = self.tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt",
        )

        label = torch.tensor(label)
        tokenized_text["label"] = label

        return tokenized_text
```

### DataLoader

```{.python .fold-show}
train_dataset = MotionerDataset(df=df_train)
valid_dataset = MotionerDataset(df=df_valid)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, collate_fn=custom_collate_fn, shuffle=True, num_workers=4
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=16, collate_fn=custom_collate_fn, shuffle=False, num_workers=4
)
```

### Data collator with padding


```{.python .fold-show}
def custom_collate_fn(data):
    tokens = [sample["input_ids"][0] for sample in data]
    attention_masks = [sample["attention_mask"][0] for sample in data]
    labels = [sample["label"] for sample in data]

    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
    padded_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    labels = torch.stack(labels)  # List of B 1-length vectors to single vector of dimension B

    batch = {"input_ids": padded_tokens, "attention_mask": attention_masks, "labels": labels}
    return batch
```

## Training loop


```{.python .fold-show}
log_list = []
for epoch in range(4):
    print(f"epoch: {epoch + 1} started")
    running_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        optim.zero_grad()
        # [batch_size, 1, seq_len] -> [batch_size, seq_len]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)

        loss = loss_fn(outputs["logits"], labels)
        running_loss += loss.item()

        if i % 50 == 49:
            print(f"iter: {i+1}, loss: {running_loss/50:.8f}, lr: {scheduler.get_last_lr()}")
            log_list.append({"iter": i + 1, "loss": running_loss / 50})
            running_loss = 0

        loss.backward()
        optim.step()
        scheduler.step()
```
