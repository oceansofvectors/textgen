import numpy as np

class GRUNetwork:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # GRU Parameters
        self.Wx = np.random.randn(hidden_size, input_size)  # For update gate z
        self.Uz = np.random.randn(hidden_size, hidden_size)
        self.Wr = np.random.randn(hidden_size, input_size)  # For reset gate r
        self.Ur = np.random.randn(hidden_size, hidden_size)
        self.Wh = np.random.randn(hidden_size, input_size)  # For candidate hidden state h_hat
        self.Uh = np.random.randn(hidden_size, hidden_size)
        self.bias_z = np.random.randn(hidden_size, 1)
        self.bias_r = np.random.randn(hidden_size, 1)
        self.bias_h = np.random.randn(hidden_size, 1)
        
        # Output Layer Parameters
        self.Wy = np.random.randn(input_size, hidden_size)  # Maps hidden state to output
        self.bias_y = np.random.randn(input_size, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update_gate(self, h, x):
        z = self.sigmoid(np.dot(self.Wx, x) + 
                        np.dot(self.Uz, h) + self.bias_z)
        return z

    def reset_gate(self, h, x):
        r = self.sigmoid(np.dot(self.Wr, x) + 
                        np.dot(self.Ur, h) + self.bias_r)
        return r

    def candidate_hidden_state(self, h, x, r):
        h_hat = np.tanh(np.dot(self.Wh, x) + 
                       np.dot(self.Uh, r * h) + self.bias_h)
        return h_hat

    def final_hidden_state(self, z, h, h_hat):
        h_new = z * h + (1 - z) * h_hat
        return h_new

    def forward(self, input_seq):
        """
        Processes a single time-step input.
        
        Parameters:
        - input_seq (np.ndarray): Shape (input_size, 1)
        
        Returns:
        - output (np.ndarray): Output vector of shape (input_size, 1)
        - h_new (np.ndarray): Updated hidden state
        """
        if input_seq.shape != (self.input_size, 1):
            raise ValueError(f"Input shape mismatch: expected ({self.input_size}, 1), got {input_seq.shape}")

        # Initialize lists to store intermediate variables if not already
        if not hasattr(self, 'x_s'):
            self.x_s = []
            self.h_s = [np.zeros((self.hidden_size, 1))]  # h_s[0] is h_0
            self.z_s = []
            self.r_s = []
            self.h_hat_s = []
            self.output_s = []

        h = self.h_s[-1]
        x = input_seq

        z = self.update_gate(h, x)
        r = self.reset_gate(h, x)
        h_hat = self.candidate_hidden_state(h, x, r)
        h_new = self.final_hidden_state(z, h, h_hat)
        y = np.dot(self.Wy, h_new) + self.bias_y  # Compute output

        # Store variables for backpropagation
        self.x_s.append(x)
        self.z_s.append(z)
        self.r_s.append(r)
        self.h_hat_s.append(h_hat)
        self.h_s.append(h_new)
        self.output_s.append(y)

        return y, h_new  # Return output and hidden state

    def backward(self, dL_dy):
        # Initialize gradients with zeros
        dL_dWx = np.zeros_like(self.Wx)
        dL_dUz = np.zeros_like(self.Uz)
        dL_dbias_z = np.zeros_like(self.bias_z)

        dL_dWr = np.zeros_like(self.Wr)
        dL_dUr = np.zeros_like(self.Ur)
        dL_dbias_r = np.zeros_like(self.bias_r)

        dL_dWh = np.zeros_like(self.Wh)
        dL_dUh = np.zeros_like(self.Uh)
        dL_dbias_h = np.zeros_like(self.bias_h)
        
        # Gradients for Output Layer
        dL_dWy = np.zeros_like(self.Wy)
        dL_dbias_y = np.zeros_like(self.bias_y)

        dL_dh_next = np.zeros((self.hidden_size, 1))  # Initialize to zero
        T = len(self.x_s)

        for t in reversed(range(T)):
            # Output layer gradients
            y_t = self.output_s[t]  # Shape: (input_size, 1)
            dL_dy_t = dL_dy[t].reshape(-1, 1)  # Shape: (input_size, 1)
            dL_dWy += np.dot(dL_dy_t, self.h_s[t+1].T)  # (input_size,1)*(1,hidden_size)
            dL_dbias_y += dL_dy_t  # (input_size,1)
            dL_dh = np.dot(self.Wy.T, dL_dy_t) + dL_dh_next  # (hidden_size,1)

            x_t = self.x_s[t]  # Shape: (input_size, 1)
            h_t = self.h_s[t+1]  # Shape: (hidden_size, 1)
            h_prev = self.h_s[t]  # Shape: (hidden_size, 1)
            z_t = self.z_s[t]  # Shape: (hidden_size, 1)
            r_t = self.r_s[t]  # Shape: (hidden_size, 1)
            h_hat_t = self.h_hat_s[t]  # Shape: (hidden_size, 1)

            # Gradients of the loss w.r.t h_t components
            # Gradient w.r.t h_hat_t
            dL_dh_hat = dL_dh * (1 - z_t)  # Shape: (hidden_size, 1)
            dL_dh_hat_preact = dL_dh_hat * (1 - h_hat_t ** 2)  # Derivative of tanh

            # Gradient w.r.t z_t
            dL_dz = dL_dh * (h_prev - h_hat_t)  # Shape: (hidden_size, 1)
            dL_dz_preact = dL_dz * z_t * (1 - z_t)  # Derivative of sigmoid

            # Gradient w.r.t r_t
            dL_dr = np.dot(self.Uh.T, dL_dh_hat_preact) * h_prev  # Shape: (hidden_size, 1)
            dL_dr_preact = dL_dr * r_t * (1 - r_t)  # Derivative of sigmoid

            # Accumulate gradients w.r.t parameters
            dL_dWh += np.dot(dL_dh_hat_preact, x_t.T)  # (hidden_size,1) * (1, input_size)
            dL_dUh += np.dot(dL_dh_hat_preact, (r_t * h_prev).T)  # (hidden_size,1) * (1, hidden_size)
            dL_dbias_h += dL_dh_hat_preact  # (hidden_size,1)

            dL_dWx += np.dot(dL_dz_preact, x_t.T)  # (hidden_size,1) * (1, input_size)
            dL_dUz += np.dot(dL_dz_preact, h_prev.T)  # (hidden_size,1) * (1, hidden_size)
            dL_dbias_z += dL_dz_preact  # (hidden_size,1)

            dL_dWr += np.dot(dL_dr_preact, x_t.T)  # (hidden_size,1) * (1, input_size)
            dL_dUr += np.dot(dL_dr_preact, h_prev.T)  # (hidden_size,1) * (1, hidden_size)
            dL_dbias_r += dL_dr_preact  # (hidden_size,1)

            # Accumulate gradient w.r.t h_prev
            dL_dh_prev = (
                np.dot(self.Uz.T, dL_dz_preact) +
                np.dot(self.Ur.T, dL_dr_preact) +
                np.dot(self.Uh.T, (dL_dh_hat_preact * r_t))
            )

            # Update dL_dh_next for the next iteration
            dL_dh_next = dL_dh_prev

        # Store gradients for parameter updates
        self.dL_dWx = dL_dWx
        self.dL_dUz = dL_dUz
        self.dL_dbias_z = dL_dbias_z

        self.dL_dWr = dL_dWr
        self.dL_dUr = dL_dUr
        self.dL_dbias_r = dL_dbias_r

        self.dL_dWh = dL_dWh
        self.dL_dUh = dL_dUh
        self.dL_dbias_h = dL_dbias_h

        # Gradients for Output Layer
        self.dL_dWy = dL_dWy
        self.dL_dbias_y = dL_dbias_y

    def update_parameters(self, learning_rate):
        # Update GRU parameters
        self.Wx -= learning_rate * self.dL_dWx
        self.Uz -= learning_rate * self.dL_dUz
        self.bias_z -= learning_rate * self.dL_dbias_z

        self.Wr -= learning_rate * self.dL_dWr
        self.Ur -= learning_rate * self.dL_dUr
        self.bias_r -= learning_rate * self.dL_dbias_r

        self.Wh -= learning_rate * self.dL_dWh
        self.Uh -= learning_rate * self.dL_dUh
        self.bias_h -= learning_rate * self.dL_dbias_h

        # Update Output Layer parameters
        self.Wy -= learning_rate * self.dL_dWy
        self.bias_y -= learning_rate * self.dL_dbias_y