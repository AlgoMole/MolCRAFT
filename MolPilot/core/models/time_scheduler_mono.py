import torch
import torch.nn as nn
import torch.nn.functional as F

class StrictlyMonotonicNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, num_hidden_layers=2, output_dim=2):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
            self.hidden_layers.append(nn.Softplus())  # Ensures strictly positive activations
            input_dim = hidden_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.softplus = nn.Softplus()  # Ensure strictly positive outputs

    def forward(self, x):
        z = x
        for layer in self.hidden_layers:
            z = layer(z)
        z = self.softplus(self.output_layer(z))  # Strictly positive outputs
        return z

class ConstrainedFunction(nn.Module):
    def __init__(self, strictly_monotonic_nn):
        super().__init__()
        self.strictly_monotonic_nn = strictly_monotonic_nn
        self.num_features = strictly_monotonic_nn.output_layer.out_features

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        # print('x', x.requires_grad, x.device)
        h_x = self.strictly_monotonic_nn(x).squeeze(-1)  # Compute h(x)
        # print('h_x', h_x.requires_grad, h_x.device)
        g_x = torch.sigmoid(h_x)  # Constrain g(x) to (0, 1)
        # print('g_x', g_x.requires_grad, g_x.device)
        output = x + (1 - x) * x * g_x  # Constrained and monotonic output
        # print('output', output.requires_grad, output.device)
        return output



# Example usage and testing
if __name__ == "__main__":
    torch.manual_seed(0)
    # Example Usage
    strictly_monotonic_nn = StrictlyMonotonicNN(input_dim=1, hidden_dim=16, num_hidden_layers=2)
    f = ConstrainedFunction(strictly_monotonic_nn)

    # Test input
    x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    output = f(x)
    print(output)
    
    x = torch.linspace(0, 1, 100)  # Input values from 0 to 1
    y = f(x.unsqueeze(-1)).detach().numpy()  # Compute outputs

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(x.numpy(), y[:, 0], label="Output Dimension 1")
    plt.plot(x.numpy(), y[:, 1], label="Output Dimension 2")
    plt.title("Learnable 2D Monotonic Function with Constraints", fontsize=14)
    plt.xlabel("Input (x)", fontsize=12)
    plt.ylabel("Output (y)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()