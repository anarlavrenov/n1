import dill
import torch
from utils import PositionalEncoding, \
    Encoder, Decoder, Transformer, tokenizer, translate

"""
Команда для завантаження моделі з Drive:

# gdown --no-check-certificate "https://drive.google.com/uc?export=download&id=1eiciYYyS3iy81-nsaWtCmwzQujfZlLZ6" -O n1_ukr_en_model.pth
"""

"""
Команда для завантаження оптимізатора з Drive:

# gdown --no-check-certificate "https://drive.google.com/uc?export=download&id=1vUs6NsXLS7Pe0DEG3XKu36JU1pI770x8" -O n1_ukr_en_optimizer.pth
"""

"""
Команди для завантаження словників з Drive:

# gdown --no-check-certificate "https://drive.google.com/uc?export=download&id=1-1fZz_K8OneDd-roRCDkcwO3z5y0ChS4" -O vocab_translation_en.pkl
"""

"""
# gdown --no-check-certificate "https://drive.google.com/uc?export=download&id=1-0yL0gNYuBTQdPRgkt7VZ2wShF_80zHr" -O vocab_translation_uk.pkl
"""

model = torch.load("n1_ukr_en_model.pth",
                   map_location=torch.device('cpu'))

with open("vocab_translation_en.pkl", "rb") as f:
  vocab_en = dill.load(f)

with open("vocab_translation_uk.pkl", "rb") as f:
  vocab_uk = dill.load(f)


res = translate("Я можу бути і серйозним і смішним", model=model, vocab_en=vocab_en, vocab_uk=vocab_uk)

print(" ".join([vocab_en.get_itos()[word] for word in res[1:]]))
