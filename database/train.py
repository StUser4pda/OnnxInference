# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import os

# ---------------- Dataset ----------------

EMB_DIM=512
HID_DIM=512

class StressDataset(Dataset):
    def __init__(self, path, letter2id=None, ph2id=None):
        self.data = []
        words, phonemes = [], []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                word, ph_str = line.split("\t")
                ph_seq = ph_str.split()
                self.data.append((word, ph_seq))
                words.append(word)
                phonemes.extend(ph_seq)

        # build vocabularies (reuse provided mappings if given)
        if letter2id is None:
            letters = set("".join([w for w, _ in self.data]))
            self.letter2id = {"<pad>": 0, "<sos>": 1, "<unk>": 2}
            for i, c in enumerate(sorted(letters)):
                self.letter2id[c] = i + 3
            self.letter2id["<eos>"] = len(self.letter2id)
        else:
            self.letter2id = letter2id
        self.id2letter = {i: c for c, i in self.letter2id.items()}

        if ph2id is None:
            ph_set = sorted(set(phonemes) | {"ˈ"})
            self.ph2id = {"<pad>": 0, "<sos>": 1, "<unk>": 2}
            for i, p in enumerate(ph_set):
                self.ph2id[p] = i + 3
            self.ph2id["<eos>"] = len(self.ph2id)
        else:
            self.ph2id = ph2id
        self.id2ph = {i: p for p, i in self.ph2id.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word, ph_seq = self.data[idx]
        unk_letter = self.letter2id.get("<unk>")
        src = []
        for c in word:
            if c not in self.letter2id:
                print(f"[WARN] Unknown letter '{c}' mapped to <unk>")
            src.append(self.letter2id.get(c, unk_letter))
        unk_ph = self.ph2id.get("<unk>")
        trg = [self.ph2id["<sos>"]]
        for p in ph_seq:
            if p not in self.ph2id:
                print(f"[WARN] Unknown phoneme '{p}' mapped to <unk>")
            trg.append(self.ph2id.get(p, unk_ph))
        trg.append(self.ph2id["<eos>"])
        return torch.tensor(src), torch.tensor(trg)


def collate_fn(batch):
    srcs, trgs = zip(*batch)
    src_lens = [len(s) for s in srcs]
    trg_lens = [len(t) for t in trgs]
    src_pad = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    trg_pad = nn.utils.rnn.pad_sequence(trgs, batch_first=True, padding_value=0)
    return src_pad, torch.tensor(src_lens), trg_pad, torch.tensor(trg_lens)

# ---------------- Seq2Seq Model ----------------


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim=EMB_DIM, hid_dim=HID_DIM):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc_h = nn.Linear(hid_dim*2, hid_dim)
        self.fc_c = nn.Linear(hid_dim*2, hid_dim)

    def forward(self, src, src_len):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        cell_cat = torch.cat((cell[-2], cell[-1]), dim=1)
        hidden = torch.tanh(self.fc_h(hidden_cat)).unsqueeze(0)
        cell = torch.tanh(self.fc_c(cell_cat)).unsqueeze(0)
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim=EMB_DIM, hid_dim=HID_DIM):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        pred = self.fc_out(output.squeeze(1))
        return pred, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, device):
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.device = device

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.dec.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        enc_outputs, hidden, cell = self.enc(src, src_len)
        input = trg[:, 0]  # <sos>
        for t in range(1, trg_len):
            output, hidden, cell = self.dec(input, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if random.random() < teacher_forcing_ratio else top1
        return outputs

# ---------------- Helper functions ----------------


def decode_prediction(output, id2ph):
    pred_ids = output.argmax(1).cpu().numpy()
    phonemes = []
    for idx in pred_ids:
        if id2ph[idx] == "<eos>":
            break
        if id2ph[idx] != "<pad>":
            phonemes.append(id2ph[idx])
    return "".join(phonemes)

# The get_tf_by_loss function is removed.

# ---------------- Training ----------------


def train_model(
    train_path="train.txt",
    val_path="validate.txt",
    save_path="seq2seq.pt",
    vocab_path="vocabs.pkl",
    epochs=40,
    batch_size=2048,
    lr=1e-3,
    initial_tf=0.5,  # New parameter for initial teacher forcing ratio
    tf_decay_rate=0.1,  # New parameter for decay rate per epoch
    resume=True
):

    train_dataset = StressDataset(train_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = StressDataset(val_path, letter2id=train_dataset.letter2id, ph2id=train_dataset.ph2id)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Use the letter2id and ph2id from the training dataset for the entire model
    enc = Encoder(len(train_dataset.letter2id)).to(device)
    dec = Decoder(len(train_dataset.ph2id)).to(device)
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    best_loss = float("inf")
    start_epoch = 1  # Keep track of where we are in the training process

    if resume and os.path.exists(save_path):
        print(f"Resuming from {save_path}")
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_loss = checkpoint['best_loss']
        # The new logic needs to save and load the epoch number to continue the decay
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded best loss: {best_loss:.8f}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Optimizer state loaded, learning rate set to {lr}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0
        # Calculate teacher forcing ratio for the current epoch
        tf = max(0.0, initial_tf - (epoch - 1) * tf_decay_rate)
        print(f"Teacher forcing ratio for epoch {epoch}: {tf:.2f}")

        for batch_idx, (src, src_len, trg, _) in enumerate(train_dataloader):
            src, src_len, trg = src.to(device), src_len.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, src_len, trg, teacher_forcing_ratio=tf)
            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(output_flat, trg_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            if batch_idx == 0:
                print("\n--- Example predictions ---")
                for i in range(min(3, src.size(0))):
                    start_idx = i * (trg.size(1)-1)
                    end_idx = (i+1) * (trg.size(1)-1)
                    pred_ph = decode_prediction(output_flat[start_idx:end_idx], train_dataset.id2ph)
                    trg_ph = "".join([train_dataset.id2ph[t.item()] for t in trg[i, 1:] if train_dataset.id2ph[t.item()] not in ["<pad>", "<eos>"]])
                    print(f"Word: {''.join([train_dataset.id2letter[c.item()] for c in src[i]])}")
                    print(f"Target: {trg_ph}")
                    print(f"Pred  : {pred_ph}\n")

        avg_loss = total_loss / len(train_dataloader)

        # --- Validation Loop ---
        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for src, src_len, trg, _ in val_dataloader:
                src, src_len, trg = src.to(device), src_len.to(device), trg.to(device)

                # Use a fixed teacher forcing ratio of 0.0 for validation/inference
                output = model(src, src_len, trg, teacher_forcing_ratio=0.0)

                output_dim = output.shape[-1]
                output_flat = output[:, 1:].reshape(-1, output_dim)
                trg_flat = trg[:, 1:].reshape(-1)

                loss = criterion(output_flat, trg_flat)
                val_total_loss += loss.item()

        avg_val_loss = val_total_loss / len(val_dataloader)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.8f} | Validation Loss: {avg_val_loss:.8f}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.8f}")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            print(f"Validation loss improved, updating best_loss to {best_loss:.8f}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'epoch': epoch
            }, save_path)

    with open(vocab_path, "wb") as f:
        pickle.dump((train_dataset.letter2id, train_dataset.ph2id, train_dataset.id2ph), f)

    print(f"✅ Training finished. Model saved to {save_path}, vocabs to {vocab_path}")


if __name__ == "__main__":
    train_model(resume=True)
