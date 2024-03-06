import dill
import torch
from pprint import pprint
from utils import PositionalEncoding, \
    Encoder, Decoder, Transformer, tokenizer, summarize

device = "cuda" if torch.cuda.is_available() else "cpu"


# Завантаження моделі
"""
Команда для завантаження моделі з Drive:
gdown --no-check-certificate "https://drive.google.com/uc?export=download&id=1FGcIjKJ9Grk7fwb6RRWHL8NGC7gdEBuO" -O model.pth
"""
model = torch.load(
    "path/to/your/project/model.pth", map_location=torch.device("cpu")) # Наполегливо рекомендується використання GPU
# для інференсу


# Завантаження вагів оптимізатора

"""
Команда для завантаження вагів оптимізатора з Drive:
# gdown --no-check-certificate "https://drive.google.com/uc?export=download&id=1vUs6NsXLS7Pe0DEG3XKu36JU1pI770x8" -O optimizer_state_dict.pth
"""
# optimizer_state_dict = torch.load(
#     "path/to/your/project/optimizer_state_dict.pth")
# optimizer = torch.optim.AdamW(model.parameters())
# optimizer.load_state_dict(optimizer_state_dict)

# Завантаження словника

"""
Команда для завантаження словника з Drive:
# gdown --no-check-certificate "https://drive.google.com/uc?export=download&id=1-031D5om2D5oHYbSN60tqlYw1b5TQ4Zi" -O vocab.pth
"""
with open("path/to/your/project/vocab.pth",
          "rb") as f:
    vocab = dill.load(f)


text = "Держави, що входять у так звану коаліцію винищувачів, розглядають Румунію як можливе місце для навчання " \
       "українських льотчиків керуванню винищувачами F-16. Про це двоє американських джерел повідомили Politico, " \
       "пише 'Європейська правда'. За даними співрозмовників видання, 'коаліція винищувачів' працює над домовленістю " \
       "про проведення авіаційних навчань на одному з полігонів у Румунії. Ймовірно, навчання вестиме компанія " \
       "Lockheed Martin, яка виробляє F-16. Європейські посадовці публічно не підтверджували й не спростовували, " \
       "що пропозиція навчати українських льотчиків у Румунії справді обговорюється. Військово-повітряні сили Румунії " \
       "мають на озброєнні 17 вживаних літаків F-16, придбаних у Португалії, і планують придбати ще 32 літаки у " \
       "Норвегії. Проте нещодавно Бухарест схвалив план придбання більш досконалих F-35. Країна відіграє важливу роль " \
       "у місії НАТО з патрулювання повітряного простору – міжнародній оперативній групі, яка займається постійним " \
       "патрулюванням європейського неба з метою швидкого реагування на порушення повітряного простору. У " \
       "Міністерстві оборони Румунії на запит Politico не підтвердили й не спростували, що розглядають можливість " \
       "навчання українських льотчиків на своїй території, та зауважили, що 'вітають ініціативу створення коаліції " \
       "країн-членів НАТО з метою підготовки льотчиків на F-16'. Нагадаємо, питання надання Україні винищувачів F-16 " \
       "обговорювалось у межах засідання Контактної групи з питань оборони України (формат 'Рамштайн') минулого " \
       "тижня. Після неї стало відомо, що до липня партнери України планують затвердити програму навчання українських " \
       "пілотів, інженерів та техніків на винищувачах F-16, самі навчання проходитимуть у спеціально створеному " \
       "центрі у одній з європейських країн. "

ids = summarize(text, model=model, tokenizer=tokenizer, vocab=vocab)
summary = " ".join([vocab.get_itos()[word] for word in ids[1:]])

pprint(summary)
