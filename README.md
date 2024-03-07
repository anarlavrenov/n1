# N1

![image](https://github.com/anarlavrenov/n1/assets/90818739/7bc4f5f7-dc7e-4c21-a1dd-56e34df0be92)

N1 є українскьою трансформерною моделлю сумаризації українського тексту, розробленою [Анаром Лавреновим](www.linkedin.com/in/anar-lavrenov). N1 має на меті заохотити вітчизняних Дата Сайнтистів до розвитку українскього напряму ШІ. 

Модель навчена на 32986 статтях з класного [датасету](https://huggingface.co/datasets/d0p3/ukr-pravda-news-summary) українських хлопців.

Тести показали високу якість роботи N1 на тестових даних. 

Окрім задачі сумаризації, дана модель, за допомогою блока енкодера, може надавати якісні ембединги для задач NLP на базі українскього тексту.

## Використання

```
git clone https://github.com/anarlavrenov/n1
```
Весь необхідний код для інференсу моделі знаходиться в [inference.py](https://github.com/anarlavrenov/n1/edit/master/inference.py)

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
