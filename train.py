from data.dataset import TextDataset
from lstm import LSTM
import torch.nn as nn
import torch
from tqdm import tqdm

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

with open('data/text8', 'r', encoding='utf-8') as f:
    text = f.read()

text = text[:100000]

chars = sorted(list(set(text)))
char_to_idx = {char: i for i, char in enumerate(chars)}
idx_to_char = {i: char for i, char in enumerate(chars)}
vocab_size = len(chars)

input_seq = [char_to_idx[char] for char in text]
dataset = TextDataset(input_seq, 10)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

model = LSTM(vocab_size, 1024, 16, 64).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    # Initialize hidden state with the correct batch size
    for i, (input_seq, target_seq) in enumerate(tqdm(data_loader, desc=f'Epoch {epoch+1}')):
        batch_size = input_seq.size(0)  # Get the current batch size
        hidden = model.init_hidden(batch_size=batch_size)  # Initialize hidden state with current batch size

        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        # Detach hidden state to prevent backpropagating through the entire history
        hidden = tuple([h.detach() for h in hidden])

        output, hidden = model(input_seq, hidden)

        # Flatten target_seq to match the shape of output
        loss = criterion(output, target_seq.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # Clear the gradients for the next step

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{10}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), f'model_{epoch+1}.pth')




        

