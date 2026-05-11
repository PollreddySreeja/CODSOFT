"""
Vocabulary builder for the captioning dataset.
"""

import os
import re
import pickle
from collections import Counter


class Vocabulary:
    def __init__(self, min_freq=3, pad_token="<pad>", start_token="<start>",
                 end_token="<end>", unk_token="<unk>"):
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.word2idx = {pad_token: 0, start_token: 1, end_token: 2, unk_token: 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self._next_idx = 4

    def __len__(self):
        return len(self.word2idx)

    def build(self, captions):
        counter = Counter()
        for cap in captions:
            counter.update(self._tokenise(cap))
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = self._next_idx
                self.idx2word[self._next_idx] = word
                self._next_idx += 1
        print(f"[Vocabulary] Built with {len(self)} words (min_freq={self.min_freq})")

    def encode_caption(self, caption, max_len=52):
        tokens = self._tokenise(caption)
        indices = [self.word2idx[self.start_token]]
        for token in tokens[:max_len - 2]:
            indices.append(self.word2idx.get(token, self.word2idx[self.unk_token]))
        indices.append(self.word2idx[self.end_token])
        while len(indices) < max_len:
            indices.append(self.word2idx[self.pad_token])
        return indices

    def decode_indices(self, indices, skip_special=True):
        special = {self.pad_token, self.start_token, self.end_token}
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.unk_token)
            if skip_special and word in special:
                continue
            if word == self.end_token:
                break
            words.append(word)
        return " ".join(words)

    @staticmethod
    def _tokenise(text):
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
