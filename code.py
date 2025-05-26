import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

# Dataset Preparation
latent_data = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)  # Load the dataset

scaler = StandardScaler()
latent_data_normalized = scaler.fit_transform(latent_data)

# Define the VAE Architecture
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2)  # Two sets of parameters for mean and variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) #compute the standard deviation (std) from the logarithm of the variance (log_var)
        eps = torch.randn_like(std) #normal distribution
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = encoded[:, :self.latent_dim], encoded[:, self.latent_dim:] #extract mean and log Variance
        z = self.reparameterize(mu, log_var) #sample from the latent space distribution defined by the mean and logarithm of the variance
        decoded = self.decoder(z)
        return decoded, mu, log_var

# Define the Loss Function
def vae_loss(recon_x, x, mu, log_var, beta):
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss

# Training Setup
input_dim = latent_data.shape[1]
latent_dim = 6  
beta = 0.04
learning_rate = 1e-4
num_epochs = 100
batch_size = 35
log_interval = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Training Loop
vae = VAE(input_dim, latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Convert numpy array to PyTorch TensorDataset
tensor_data = torch.tensor(latent_data, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        data = data[0].to(device).float()
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(data)
        loss = vae_loss(recon_batch, data, mu, log_var, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % log_interval == 0:
        print('Epoch {} - Loss: {:.4f}'.format(epoch, train_loss / len(dataset)))

#Generate New Samples
num_samples_to_generate = 3000
random_samples = torch.randn(num_samples_to_generate, latent_dim).to(device)
generated_samples = vae.decoder(random_samples)
generated_samples = generated_samples.cpu().detach().numpy()

#Save the Generated Samples with ID column
ids = np.arange(1, num_samples_to_generate + 1).reshape(-1, 1)
joint_columns = [f'joint_{i}' for i in range(1, input_dim + 1)]
headers = ['id'] + joint_columns
generated_data_with_id = np.hstack((ids, generated_samples))

id_format = '%d'

# Save the generated samples in a csv file
np.savetxt('generated_samples.csv', generated_data_with_id, delimiter=',', header=','.join(headers), fmt=[id_format] + ['%.8f']*input_dim, comments='')

# Save the Model
torch.save(vae.state_dict(), 'vae_model.pth')