import torch
from torch import nn
import pytorch_lightning as pl



class ElectricityLoadModel(pl.LightningModule):
    """Encoder network for encoder-decoder forecast model."""

    def __init__(self, hist_len=168, fct_len=24, input_size=4, num_layers=1, hidden_units=8, lr=1e-3):
        super().__init__()

        self.hist_len = hist_len
        self.fct_len = fct_len
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.lr = lr

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_units,
                            num_layers=self.num_layers,
                            batch_first=True)

        self.mu = nn.Linear(in_features=self.hidden_units, out_features=1)
        self.sigma_raw = nn.Linear(in_features=self.hidden_units, out_features=1)
        self.sigma = nn.Softplus()

    def forward(self, x, hidden=None):
        output, (hh, cc) = self.lstm(x)
        mu = self.mu(output)
        sigma = self.sigma(self.sigma_raw(output))
        return output, mu, sigma, hh, cc

    def training_step(self, batch, batch_idx):
        combined_input = torch.cat(batch, dim=1)

        # Pushing through the network
        out, mu, sigma, hh, cc = self(combined_input[:, :-1, :])

        return self.loss(mu, sigma, combined_input[:, 1:, [0]])

    def validation_step(self, batch, batch_idx):

        combined_input = torch.cat(batch, dim=1)

        # Pushing through the network
        out, mu, sigma, hh, cc = self(combined_input[:, :-1, :])

        loss = self.loss(mu, sigma, combined_input[:, 1:, [0]])
        self.log('val_logprob', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def sample(self, x, samples):
        # Handle single stream or multiple streams
        x = x.view(-1, self.hist_len, 4)

        # Initial pass - `mu(T+1)` and `sigma(T+1)` are ready
        out, mu, sigma, hh_initial, cc_initial = self(x)

        # Sample from the distribution
        gaussian = torch.distributions.normal.Normal(mu[:, -1, -1], sigma[:, -1, -1])
        initial_sample = gaussian.sample((samples,))

        # Getting calendar features
        calendar_features = x[:, -self.fct_len:, 1:]

        all_samples = []

        # Iterating over samples
        for sample in range(samples):
            current_sample = initial_sample[sample]

            # We want output to be `(fct_len, batch_size, samples)`
            local_samples = [current_sample.view(1, -1, 1)]

            # Initialize hidden and cell state to encoder output
            hh, cc = hh_initial, cc_initial

            # Iterating over time steps
            for step in range(self.fct_len - 1):
                # Input tensor for this step
                step_in = torch.cat([current_sample.view(-1, 1, 1),
                                     calendar_features[:, [step], :]], dim=-1)

                step_out, mu, sigma, hh, cc = self(step_in, (hh, cc))

                # Sampling the next step value
                gaussian = torch.distributions.normal.Normal(mu[:, -1, -1], sigma[:, -1, -1])
                current_sample = gaussian.sample((1,))

                # Storing the samples for this time step
                local_samples.append(current_sample.unsqueeze(-1))

            # Storing all samples for this sample
            all_samples.append(torch.cat(local_samples, dim=0))
        return torch.cat(all_samples, -1)

    def loss(self, mu, sigma, y):
        # Distribution with generated `mu` and `sigma`
        gaussian = torch.distributions.normal.Normal(mu, sigma)

        # Log-likelihood
        L = gaussian.log_prob(y)

        return -torch.mean(L)
