import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, n_channel: int):
        super(VAE, self).__init__()

        self.encode_conv = nn.Sequential(
            nn.Conv2d(n_channel, 32, kernel_size=3, stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(6 * 6 * 64, 32)
        self.fc_log_var = nn.Linear(6 * 6 * 64, 32)
        self.fc_z = nn.Linear(32, 6 * 6 * 64)

        self.decode_deconv = nn.Sequential(
            nn.Linear(32, 6 * 6 * 64),
            nn.Unflatten(-1, (64, 6, 6)),
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(1, n_channel, 2, stride=1),
        )

    def encode(self, x):
        x = self.encode_conv(x)
        return x
    
    def sample(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        return mu, log_var, z

    def decode(self, x):
        out = torch.tanh(self.decode_deconv(x))

        return out