import numpy as np
import logging
from embeddings import OneHotEncoder
from gru_network import GRUNetwork
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TextPredictor:
    def __init__(self, text_path, hidden_size, learning_rate=0.01, chunk_size=1024*1024):
        self.text_path = text_path
        self.encoder = OneHotEncoder()
        self.input_size = len(self.encoder.characters)  # Dynamically set input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.chunk_size = chunk_size
        self.model = GRUNetwork(self.input_size, self.hidden_size)
        logging.info("Initialized TextPredictor with input size %d, hidden size %d, learning rate %f", 
                     self.input_size, self.hidden_size, self.learning_rate)

    def load_text_in_chunks(self):
        logging.info("Loading text from %s in chunks", self.text_path)
        with open(self.text_path, 'r') as file:
            while True:
                chunk = file.read(self.chunk_size)
                if not chunk:
                    break
                logging.debug("Loaded chunk: %s", chunk[:50])  # Log the first 50 characters of the chunk
                yield chunk

    def prepare_data(self, text):
        logging.info("Preparing data for training")
        def encode_generator(text):
            for char in tqdm(text, desc="Encoding"):
                yield self.encoder.encode(char)
        
        inputs = np.array(list(encode_generator(text[:-1]))).reshape(-1, self.input_size, 1)
        targets = np.array(list(encode_generator(text[1:]))).reshape(-1, self.input_size, 1)
        logging.debug("Prepared %d input-target pairs", len(inputs))
        return inputs, targets

    def train(self, epochs=5):
        for epoch in range(epochs):
            total_loss = 0
            logging.info("Starting epoch %d/%d", epoch + 1, epochs)
            for chunk in self.load_text_in_chunks():
                inputs, targets = self.prepare_data(chunk)
                for i in tqdm(range(len(inputs)), desc=f"Epoch {epoch + 1}/{epochs}"):
                    input_seq = inputs[i].reshape(-1, 1)  # Shape: (input_size, 1)
                    target_seq = targets[i].reshape(-1, 1)  # Shape: (input_size, 1)

                    # Forward pass
                    final_state = self.model.forward(input_seq)  # Shape: (hidden_size, 1)
                    logging.debug("Forward pass completed for input index %d", i)

                    # Compute loss (mean squared error)
                    loss = 0.5 * np.sum((final_state - target_seq) ** 2)
                    total_loss += loss
                    logging.debug("Loss computed for input index %d: %f", i, loss)

                    # Backward pass
                    dL_dh_T = final_state - target_seq  # Shape: (hidden_size, 1)
                    self.model.backward(dL_dh_T)
                    logging.debug("Backward pass completed for input index %d", i)

                    # Update parameters
                    self.model.update_parameters(self.learning_rate)
                    logging.debug("Parameters updated for input index %d", i)

            logging.info("Epoch %d/%d completed, Total Loss: %f", epoch + 1, epochs, total_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

# Example usage
if __name__ == "__main__":
    text_predictor = TextPredictor('./text8', hidden_size=52)
    text_predictor.train(epochs=5)