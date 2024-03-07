## Requirements

```
git clone https://github.com/anarlavrenov/n1
```

Команди для завантаження моделі, вагів оптимізатора (для донавчання), та словника

```
gdown --no-check-certificate "https://drive.google.com/uc?export=download&id=1FGcIjKJ9Grk7fwb6RRWHL8NGC7gdEBuO" -O model.pth

gdown --no-check-certificate "https://drive.google.com/uc?export=download&id=1vUs6NsXLS7Pe0DEG3XKu36JU1pI770x8" -O optimizer_state_dict.pth

gdown --no-check-certificate "https://drive.google.com/uc?export=download&id=1-031D5om2D5oHYbSN60tqlYw1b5TQ4Zi" -O vocab.pth
```


Дана модель побудована на базі архітектури трансформера: 

```
num_layers_encoder = 6
num_layers_decoder = 6
d_model = 512
nhead = 8
dff = 1024
ntokens = len(vocab) + 2
dropout = 0.5
```

Ваги до моделі ініціалізовані росподілом Xavier 
```py
for param in model.parameters():
  if param.dim() > 1:
    torch.nn.init.xavier_uniform_(param)
```

    
Модель навчена на 32986 - статей з класного [датасету](https://huggingface.co/datasets/d0p3/ukr-pravda-news-summary) українських хлопців
