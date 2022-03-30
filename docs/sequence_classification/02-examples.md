# (PART\*) Examples {-}



# Sentiment Analysis on ScandiSent

## Download the data

## Import libraries and set device


```{.python .fold-show}
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, AutoTokenizer, AdamW
```


```{.python .fold-show}
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
```

```
## cuda
```

## Read data

We split the data the same way @Isbister2021ShouldWS did.


```{.python .fold-show}
df = pd.read_csv("ScandiSent/sv.csv")
df_train = df[:7500]  # First 7500 train set
df_valid = df[7500:]  # Last 2500 evaluation
```

Let us take a look at the ScandiSent data again.



```{=html}
<div id="htmlwidget-2285a7120f768f2891e9" class="reactable html-widget" style="width:auto;height:auto;"></div>
<script type="application/json" data-for="htmlwidget-2285a7120f768f2891e9">{"x":{"tag":{"name":"Reactable","attribs":{"data":{"text":["En gång i tiden hade de ett mycket bra kort, ikanokortet. Tyvärr har kortet försämrats oerhört mycket så nu är det totalt ointressant. Banken känns oerhört omodern. Långa väntetider i telefon till kundservice, och när de väl svarar ger de tyvärr inte alls något professionellt intryck. Jag är väldigt glad att jag inte längre är kund hos dem.","App ur funktion, ett antal falsklarm. Får ingen återkoppling vare sig på e-mail eller via telefon trots upprepade försök att maila och ringa. Det bara görs ”ärenden” av våra frågor om vad som är fel. Vi kan inte använda larmet nu då vi inte vet vad som är fel. Så arrogant bemötande!!!","Jag trodde att räntevillkoren på mina tidigare lån var bland marknadens bästa, eftersom jag håller mig uppdaterad, MEN Northmills erbjudande var överlägset, så det är den bästa deal jag har gjort med en bank.","Väldigt hjälpsam kundservis. Bra varor med topp kvalité och snabbservis.","Har försökt bli kund hos dem och hade en fråga innan. Deras kundservice har inte fungerat denna eller förra månaden. Vågar inte bli kund här även om det är billigt","Trevlig kommunikation, snabb och bra service! Rekommenderas!","Wow vilket jobb ni gjort hemma hos oss! Förstår inte hur ni lyckas få saker så otroligt rena! Jag är så glad och tacksam.","Dom påstår att man kan ha kostnadsfri profil. Men när jag sa åt dom att sluta ringa mig hela tiden och sluta försöka sälja på mig dyra abbonemang, då skrev han att han tar bort min profil. Fast jag sa att jag ville ha min kostnadsfria profil kvar. Väldigt oseriöst. Bluff företag.","Inte så bra lim och utlovad tejp var inte med","Klockren snabb och personlig hjälp vid beställning. Rekommenderas verkligen!"],"label":[0,0,1,1,0,1,1,0,0,1]},"columns":[{"accessor":"text","name":"text","type":"character","cell":[{"name":"div","attribs":{"style":{"text-decoration":"underline","text-decoration-style":"dotted","text-decoration-color":"#FF6B00","cursor":"info","white-space":"nowrap","overflow":"hidden","text-overflow":"ellipsis"}},"children":[{"name":"WidgetContainer","attribs":{"key":"adbce2cc59c2892fad21b38376f194cc"},"children":[{"name":"Fragment","attribs":[],"children":[{"name":"span","attribs":{"id":"htmlwidget-03d87778c38d3dbc0351","width":960,"height":500,"className":"tippy html-widget"},"children":[]},{"name":"script","attribs":{"type":"application/json","data-for":"htmlwidget-03d87778c38d3dbc0351"},"children":["{\"x\":{\"opts\":{\"content\":\"En gång i tiden hade de ett mycket bra kort, ikanokortet. Tyvärr har kortet försämrats oerhört mycket så nu är det totalt ointressant. Banken känns oerhört omodern. Långa väntetider i telefon till kundservice, och när de väl svarar ger de tyvärr inte alls något professionellt intryck. Jag är väldigt glad att jag inte längre är kund hos dem.\"},\"text\":\"En gång i tiden hade de ett mycket bra kort, ikanokortet. Tyvärr har kortet försämrats oerhört mycket så nu är det totalt ointressant. Banken känns oerhört omodern. Långa väntetider i telefon till kundservice, och när de väl svarar ger de tyvärr inte alls något professionellt intryck. Jag är väldigt glad att jag inte längre är kund hos dem.\"},\"evals\":[],\"jsHooks\":[]}"]}]}]}]},{"name":"div","attribs":{"style":{"text-decoration":"underline","text-decoration-style":"dotted","text-decoration-color":"#FF6B00","cursor":"info","white-space":"nowrap","overflow":"hidden","text-overflow":"ellipsis"}},"children":[{"name":"WidgetContainer","attribs":{"key":"0187ee6aff70331751bd5a1a6e77c297"},"children":[{"name":"Fragment","attribs":[],"children":[{"name":"span","attribs":{"id":"htmlwidget-d8cbd8cc01bfe7fc5063","width":960,"height":500,"className":"tippy html-widget"},"children":[]},{"name":"script","attribs":{"type":"application/json","data-for":"htmlwidget-d8cbd8cc01bfe7fc5063"},"children":["{\"x\":{\"opts\":{\"content\":\"App ur funktion, ett antal falsklarm. Får ingen återkoppling vare sig på e-mail eller via telefon trots upprepade försök att maila och ringa. Det bara görs ”ärenden” av våra frågor om vad som är fel. Vi kan inte använda larmet nu då vi inte vet vad som är fel. Så arrogant bemötande!!!\"},\"text\":\"App ur funktion, ett antal falsklarm. Får ingen återkoppling vare sig på e-mail eller via telefon trots upprepade försök att maila och ringa. Det bara görs ”ärenden” av våra frågor om vad som är fel. Vi kan inte använda larmet nu då vi inte vet vad som är fel. Så arrogant bemötande!!!\"},\"evals\":[],\"jsHooks\":[]}"]}]}]}]},{"name":"div","attribs":{"style":{"text-decoration":"underline","text-decoration-style":"dotted","text-decoration-color":"#FF6B00","cursor":"info","white-space":"nowrap","overflow":"hidden","text-overflow":"ellipsis"}},"children":[{"name":"WidgetContainer","attribs":{"key":"585678e0d1e9e9e004523dd9210cf20c"},"children":[{"name":"Fragment","attribs":[],"children":[{"name":"span","attribs":{"id":"htmlwidget-a778c3512f3e3e702008","width":960,"height":500,"className":"tippy html-widget"},"children":[]},{"name":"script","attribs":{"type":"application/json","data-for":"htmlwidget-a778c3512f3e3e702008"},"children":["{\"x\":{\"opts\":{\"content\":\"Jag trodde att räntevillkoren på mina tidigare lån var bland marknadens bästa, eftersom jag håller mig uppdaterad, MEN Northmills erbjudande var överlägset, så det är den bästa deal jag har gjort med en bank.\"},\"text\":\"Jag trodde att räntevillkoren på mina tidigare lån var bland marknadens bästa, eftersom jag håller mig uppdaterad, MEN Northmills erbjudande var överlägset, så det är den bästa deal jag har gjort med en bank.\"},\"evals\":[],\"jsHooks\":[]}"]}]}]}]},{"name":"div","attribs":{"style":{"text-decoration":"underline","text-decoration-style":"dotted","text-decoration-color":"#FF6B00","cursor":"info","white-space":"nowrap","overflow":"hidden","text-overflow":"ellipsis"}},"children":[{"name":"WidgetContainer","attribs":{"key":"b5c91be9b059d59428f8865419316a32"},"children":[{"name":"Fragment","attribs":[],"children":[{"name":"span","attribs":{"id":"htmlwidget-66616113b1f325eab9a4","width":960,"height":500,"className":"tippy html-widget"},"children":[]},{"name":"script","attribs":{"type":"application/json","data-for":"htmlwidget-66616113b1f325eab9a4"},"children":["{\"x\":{\"opts\":{\"content\":\"Väldigt hjälpsam kundservis. Bra varor med topp kvalité och snabbservis.\"},\"text\":\"Väldigt hjälpsam kundservis. Bra varor med topp kvalité och snabbservis.\"},\"evals\":[],\"jsHooks\":[]}"]}]}]}]},{"name":"div","attribs":{"style":{"text-decoration":"underline","text-decoration-style":"dotted","text-decoration-color":"#FF6B00","cursor":"info","white-space":"nowrap","overflow":"hidden","text-overflow":"ellipsis"}},"children":[{"name":"WidgetContainer","attribs":{"key":"9d2cc7af5acba1246cc2ad1046f099fb"},"children":[{"name":"Fragment","attribs":[],"children":[{"name":"span","attribs":{"id":"htmlwidget-4be2fc3476501c10826b","width":960,"height":500,"className":"tippy html-widget"},"children":[]},{"name":"script","attribs":{"type":"application/json","data-for":"htmlwidget-4be2fc3476501c10826b"},"children":["{\"x\":{\"opts\":{\"content\":\"Har försökt bli kund hos dem och hade en fråga innan. Deras kundservice har inte fungerat denna eller förra månaden. Vågar inte bli kund här även om det är billigt\"},\"text\":\"Har försökt bli kund hos dem och hade en fråga innan. Deras kundservice har inte fungerat denna eller förra månaden. Vågar inte bli kund här även om det är billigt\"},\"evals\":[],\"jsHooks\":[]}"]}]}]}]},{"name":"div","attribs":{"style":{"text-decoration":"underline","text-decoration-style":"dotted","text-decoration-color":"#FF6B00","cursor":"info","white-space":"nowrap","overflow":"hidden","text-overflow":"ellipsis"}},"children":[{"name":"WidgetContainer","attribs":{"key":"64864c238ad0fbf1fb008a6df55b14ac"},"children":[{"name":"Fragment","attribs":[],"children":[{"name":"span","attribs":{"id":"htmlwidget-a11eaf68a6446c77ab27","width":960,"height":500,"className":"tippy html-widget"},"children":[]},{"name":"script","attribs":{"type":"application/json","data-for":"htmlwidget-a11eaf68a6446c77ab27"},"children":["{\"x\":{\"opts\":{\"content\":\"Trevlig kommunikation, snabb och bra service! Rekommenderas!\"},\"text\":\"Trevlig kommunikation, snabb och bra service! Rekommenderas!\"},\"evals\":[],\"jsHooks\":[]}"]}]}]}]},{"name":"div","attribs":{"style":{"text-decoration":"underline","text-decoration-style":"dotted","text-decoration-color":"#FF6B00","cursor":"info","white-space":"nowrap","overflow":"hidden","text-overflow":"ellipsis"}},"children":[{"name":"WidgetContainer","attribs":{"key":"231429d362aa9765ffddb644415c1267"},"children":[{"name":"Fragment","attribs":[],"children":[{"name":"span","attribs":{"id":"htmlwidget-9801691ec7119851bef9","width":960,"height":500,"className":"tippy html-widget"},"children":[]},{"name":"script","attribs":{"type":"application/json","data-for":"htmlwidget-9801691ec7119851bef9"},"children":["{\"x\":{\"opts\":{\"content\":\"Wow vilket jobb ni gjort hemma hos oss! Förstår inte hur ni lyckas få saker så otroligt rena! Jag är så glad och tacksam.\"},\"text\":\"Wow vilket jobb ni gjort hemma hos oss! Förstår inte hur ni lyckas få saker så otroligt rena! Jag är så glad och tacksam.\"},\"evals\":[],\"jsHooks\":[]}"]}]}]}]},{"name":"div","attribs":{"style":{"text-decoration":"underline","text-decoration-style":"dotted","text-decoration-color":"#FF6B00","cursor":"info","white-space":"nowrap","overflow":"hidden","text-overflow":"ellipsis"}},"children":[{"name":"WidgetContainer","attribs":{"key":"4d8aaaedf86439c1c6b6c6a895d5327f"},"children":[{"name":"Fragment","attribs":[],"children":[{"name":"span","attribs":{"id":"htmlwidget-8c1e87f547a2abc8f9b3","width":960,"height":500,"className":"tippy html-widget"},"children":[]},{"name":"script","attribs":{"type":"application/json","data-for":"htmlwidget-8c1e87f547a2abc8f9b3"},"children":["{\"x\":{\"opts\":{\"content\":\"Dom påstår att man kan ha kostnadsfri profil. Men när jag sa åt dom att sluta ringa mig hela tiden och sluta försöka sälja på mig dyra abbonemang, då skrev han att han tar bort min profil. Fast jag sa att jag ville ha min kostnadsfria profil kvar. Väldigt oseriöst. Bluff företag.\"},\"text\":\"Dom påstår att man kan ha kostnadsfri profil. Men när jag sa åt dom att sluta ringa mig hela tiden och sluta försöka sälja på mig dyra abbonemang, då skrev han att han tar bort min profil. Fast jag sa att jag ville ha min kostnadsfria profil kvar. Väldigt oseriöst. Bluff företag.\"},\"evals\":[],\"jsHooks\":[]}"]}]}]}]},{"name":"div","attribs":{"style":{"text-decoration":"underline","text-decoration-style":"dotted","text-decoration-color":"#FF6B00","cursor":"info","white-space":"nowrap","overflow":"hidden","text-overflow":"ellipsis"}},"children":[{"name":"WidgetContainer","attribs":{"key":"433065f0aed7ad6bb721a64a1fc7b050"},"children":[{"name":"Fragment","attribs":[],"children":[{"name":"span","attribs":{"id":"htmlwidget-26588b905a413972d17c","width":960,"height":500,"className":"tippy html-widget"},"children":[]},{"name":"script","attribs":{"type":"application/json","data-for":"htmlwidget-26588b905a413972d17c"},"children":["{\"x\":{\"opts\":{\"content\":\"Inte så bra lim och utlovad tejp var inte med\"},\"text\":\"Inte så bra lim och utlovad tejp var inte med\"},\"evals\":[],\"jsHooks\":[]}"]}]}]}]},{"name":"div","attribs":{"style":{"text-decoration":"underline","text-decoration-style":"dotted","text-decoration-color":"#FF6B00","cursor":"info","white-space":"nowrap","overflow":"hidden","text-overflow":"ellipsis"}},"children":[{"name":"WidgetContainer","attribs":{"key":"74fe95608e7a1c250962a3f6c772620a"},"children":[{"name":"Fragment","attribs":[],"children":[{"name":"span","attribs":{"id":"htmlwidget-cf74ef4d6582b4c6df9c","width":960,"height":500,"className":"tippy html-widget"},"children":[]},{"name":"script","attribs":{"type":"application/json","data-for":"htmlwidget-cf74ef4d6582b4c6df9c"},"children":["{\"x\":{\"opts\":{\"content\":\"Klockren snabb och personlig hjälp vid beställning. Rekommenderas verkligen!\"},\"text\":\"Klockren snabb och personlig hjälp vid beställning. Rekommenderas verkligen!\"},\"evals\":[],\"jsHooks\":[]}"]}]}]}]}],"html":true},{"accessor":"label","name":"label","type":"numeric"}],"defaultPageSize":5,"paginationType":"numbers","showPageInfo":true,"minRows":1,"highlight":true,"striped":true,"theme":{"color":"#2f2f2f","borderColor":"#c5c5c5"},"dataKey":"780f044fdfaf4b517a109c33e8f57316"},"children":[]},"class":"reactR_markup"},"evals":[],"jsHooks":[]}</script>
```

## Create Dataset and DataLoader


```{.python .fold-show}
class SentimentDataset(Dataset):
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
train_dataset = SentimentDataset(df=df_train)
valid_dataset = SentimentDataset(df=df_valid)

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


:::help
[Everything you always wanted to know about padding and truncation.](https://huggingface.co/docs/transformers/preprocessing#everything-you-always-wanted-to-know-about-padding-and-truncation)
:::
