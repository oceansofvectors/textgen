class OneHotEncoder:
    def __init__(self):
        # Include the space character in the supported character set
        self.characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
        # Create a mapping from character to index
        self.char_to_index = {char: idx for idx, char in enumerate(self.characters)}
        # Create a mapping from index to character
        self.index_to_char = {idx: char for idx, char in enumerate(self.characters)}

    def encode(self, char):
        # Check if the character is in the supported character set
        if char not in self.char_to_index:
            raise ValueError(f"Character '{char}' is not in the supported character set.")
        # Initialize a zero vector of size num_chars
        one_hot_vector = [0] * len(self.characters)
        # Set the index corresponding to the character to 1
        one_hot_vector[self.char_to_index[char]] = 1
        return one_hot_vector

    def decode(self, one_hot):
        # Find the index with the value 1
        index = one_hot.index(1)
        # Return the character corresponding to the index
        return self.index_to_char[index]

# Example usage:
if __name__ == "__main__":
    encoder = OneHotEncoder()
    encoded_char = encoder.encode('A')
    print(f"One-hot encoding for 'A': {encoded_char}")
    decoded_char = encoder.decode(encoded_char)
    print(f"Decoded character: {decoded_char}")
