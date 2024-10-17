import torch
from lstm import LSTM
import numpy as np

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

with open('data/text8', 'r', encoding='utf-8') as f:
    text = f.read()

text = text[:100000]

chars = sorted(list(set(text)))
char_to_idx = {char: i for i, char in enumerate(chars)}
idx_to_char = {i: char for i, char in enumerate(chars)}
vocab_size = len(chars)

model = LSTM(vocab_size, 512, 8, 64).to(device)

model.load_state_dict(torch.load('model_1.pth'))


start_text = 'anarchism originated as a'

chars = [char_to_idx[char] for char in start_text]
input_seq = torch.tensor(chars).unsqueeze(0).to(device)
hidden = model.init_hidden(1)

output_chars = start_text

for i in range(20):
    with torch.no_grad():
        output, hidden = model(input_seq, hidden)
        prob = torch.softmax(output[-1], dim=0).cpu().numpy()
        next_char_idx = np.random.choice(len(prob), p=prob)
        next_char = idx_to_char[next_char_idx]
        output_chars += next_char
        input_seq = torch.tensor([next_char_idx]).unsqueeze(0).to(device)


print(output_chars)