import torch
import math
from typing import Callable, List
import torchtext
import spacy

device = "cuda" if torch.cuda.is_available() else "cpu"


class PositionalEncoding(torch.nn.Module):  # Джерело: PyTorch documentation

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Encoder(torch.nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int,
                 dff: int, ntokens: int, dropout: float = 0.5):
        super(Encoder, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=ntokens,
                                            embedding_dim=d_model,
                                            padding_idx=0)

        self.pos_encoding = PositionalEncoding(d_model=d_model,
                                               dropout=dropout)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model,
                                                         nhead=nhead,
                                                         dim_feedforward=dff,
                                                         dropout=dropout,
                                                         norm_first=True)

        self.encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                   num_layers=num_layers)

        self.d_model = d_model

        self.linear_glu = torch.nn.Linear(in_features=d_model,
                                          out_features=d_model * 2)

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # src -> seq_len, batch_size, d_model
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)

        src = torch.nn.functional.glu(self.linear_glu(src), dim=-1)  # Застосування GLU

        if mask is None:
            mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=len(src)).to(device)

        encoder_output = self.encoder(src, mask)

        return encoder_output  # -> Tensor shape: seq_len, batch_size, ntokens


class Decoder(torch.nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int,
                 dff: int, ntokens: int, dropout: float = 0.5):
        super(Decoder, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=ntokens,
                                            embedding_dim=d_model,
                                            padding_idx=0)

        self.pos_encoding = PositionalEncoding(d_model=d_model,
                                               dropout=dropout)

        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model,
                                                         nhead=nhead,
                                                         dim_feedforward=dff,
                                                         dropout=dropout,
                                                         norm_first=True)

        self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoder_layer,
                                                   num_layers=num_layers)

        self.fc = torch.nn.Linear(in_features=d_model,
                                  out_features=ntokens)

        self.d_model = d_model

        self.linear_glu = torch.nn.Linear(in_features=d_model,
                                          out_features=d_model * 2)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):

        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)

        tgt = torch.nn.functional.glu(self.linear_glu(tgt), dim=-1)  # Застосування GLU

        if tgt_mask is None:
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(len(tgt)).to(device)

        if memory_mask is None:
            memory_mask = torch.zeros((tgt.size(1), memory.size(0))).to(device)

        decoder_output = self.decoder(tgt, memory,
                                      tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)

        output = self.fc(decoder_output)

        return output


class Transformer(torch.nn.Module):
    def __init__(self, num_layers_encoder: int, num_layers_decoder: int, d_model: int, nhead: int,
                 dff: int, ntokens: int, dropout: float = 0.5):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers_encoder, d_model, nhead, dff, ntokens)
        self.decoder = Decoder(num_layers_decoder, d_model, nhead, dff, ntokens)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        memory = self.encoder(src)
        decoder_output = self.decoder(tgt, memory)

        return decoder_output


def tokenizer(text: str) -> List[str]:
    nlp = spacy.load("uk_core_news_trf")  # Джерело й подяка: https://spacy.io/models/uk
    return [tok.text for tok in nlp.tokenizer(text)]


def summarize(string: str, model: torch.nn.Module,
              tokenizer: Callable[[str], List[str]], vocab: torchtext.vocab.Vocab,
              repetition_penalty: float = 1.2, maxlen_title: int = 33) -> torch.Tensor:
    model.eval()

    start_token = [len(vocab)]
    end_token = [len(vocab) + 1]

    string = torch.IntTensor(
        start_token + [vocab(tokenizer(word))[0] for word in string.split()] + end_token).unsqueeze(0).to(device)
    output = torch.IntTensor(start_token).unsqueeze(0).to(device)

    with torch.no_grad():

        for i in range(maxlen_title):

            prediction = model(string.permute(1, 0), output.permute(1, 0))

            prediction = prediction[-1:, :, :]

            if i > 1:
                # repetition penalty
                for token_id in set(output.squeeze().tolist()):
                    prediction[0, 0, token_id] /= repetition_penalty

            predicted_id = torch.argmax(prediction, dim=-1)

            if predicted_id[0] == end_token[0]:
                return output.squeeze(0)

            output = torch.cat([output, predicted_id.permute(1, 0)], dim=-1)

        return output.squeeze(0)
