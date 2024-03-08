# N1

![image](https://github.com/anarlavrenov/n1/assets/90818739/7bc4f5f7-dc7e-4c21-a1dd-56e34df0be92)

N1 є українскьою трансформерною моделлю сумаризації українського тексту, розробленою [Анаром Лавреновим, Head of AI компанії SPUNCH](https://www.linkedin.com/mynetwork/). N1 має на меті заохотити вітчизняних Дата Сайнтистів до розвитку українскього напряму ШІ. 

Модель навчена на 32986 статтях з класного [датасету](https://huggingface.co/datasets/d0p3/ukr-pravda-news-summary) українських хлопців та має 295,170,389 параметрів, що робить її перспективною для донавчання на більшій кільскості даних.

Модель показали непогані результати роботи на тестових даних. Хоча точність сумаризації не можна назвати сто відсотковою, проте дана версія моделі добре розуміє контекст та робить його стислий переказ.

Декілька прикладів:

```
Текст: """Держави, що входять у так звану коаліцію винищувачів, розглядають Румунію як можливе місце для навчання українських льотчиків керуванню винищувачами F-16. Про це двоє американських джерел повідомили Politico, пише 'Європейська правда'. За даними співрозмовників видання, 'коаліція винищувачів' працює над домовленістю про проведення авіаційних навчань на одному з полігонів у Румунії. Ймовірно, навчання вестиме компанія Lockheed Martin, яка виробляє F-16. Європейські посадовці публічно не підтверджували й не спростовували, що пропозиція навчати українських льотчиків у Румунії справді обговорюється. Військово-повітряні сили Румунії мають на озброєнні 17 вживаних літаків F-16, придбаних у Португалії, і планують придбати ще 32 літаки у Норвегії. Проте нещодавно Бухарест схвалив план придбання більш досконалих F-35. Країна відіграє важливу роль у місії НАТО з патрулювання повітряного простору – міжнародній оперативній групі, яка займається постійним патрулюванням європейського неба з метою швидкого реагування на порушення повітряного простору. У Міністерстві оборони Румунії на запит Politico не підтвердили й не спростували, що розглядають можливість навчання українських льотчиків на своїй території, та зауважили, що 'вітають ініціативу створення коаліції країн-членів НАТО з метою підготовки льотчиків на F-16'. Нагадаємо, питання надання Україні винищувачів F-16 обговорювалось у межах засідання Контактної групи з питань оборони України (формат 'Рамштайн') минулого тижня. Після неї стало відомо, що до липня партнери України планують затвердити програму навчання українських пілотів, інженерів та техніків на винищувачах F-16, самі навчання проходитимуть у спеціально створеному центрі у одній з європейських країн."""

Сумаризація: "Держави - члени НАТО оголосили про початок навчань українських льотчиків на винищувачах F-16, які будуть передані в Румунії."
```

```
Текст: """Лише за минулу добу українські захисники знищили близько 500 окупантів, 7 танків, 10 ББМ, 15 артилерійських систем та гелікоптер. Джерело: Генеральний штаб ЗСУ у Facebook Деталі: Загальні бойові втрати противника з 24.02.22 по 11.04.23 орієнтовно склали: особового складу – близько 179 320 (+500) осіб ліквідовано, танків – 3 644 (+7) од, бойових броньованих машин – 7 038 (+10) од, артилерійських систем – 2 765 (+15) од, РСЗВ – 535 (+1) од, засобів ППО – 282 (+0) од, літаків – 307 (+0) од, гелікоптерів – 293 (+1) од, БПЛА оперативно-тактичного рівня – 2 332 (+9) од, крилатих ракет – 911 (+0) од, кораблів/катерів – 18 (+0) од, автомобільної техніки та автоцистерн – 5 620 (+13) од, спеціальної техніки – 316 (+5) од. Дані уточнюються."""

Сумаризація: "Українські захисники знищили близько 200 російських окупантів, 1 гелікоптер та 10 танків."
```

Варто зазначити, що, окрім задачі сумаризації, дана модель, за допомогою блока енкодера, може надавати якісні ембединги для задач NLP на базі українскього тексту.

## Використання

```
git clone https://github.com/anarlavrenov/n1
```
Весь необхідний код для інференсу моделі знаходиться в [n1_inference.py](https://github.com/anarlavrenov/n1/edit/master/inference.py), код тренування моделі - в [n1_training.ipynb](https://github.com/anarlavrenov/n1/edit/master/n1_training.ipynb)

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
