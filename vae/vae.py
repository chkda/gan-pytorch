import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

class ImageDataset(Dataset):

    def __init__(self, folder_path, image_size=64):

        self.folder_path = Path(folder_path)
        valid_extensions = {".jpg", ".jpeg", ".png"}
        self.image_files = [
            f for f in self.folder_path.iterdir() 
            if f.suffix.lower() in valid_extensions
        ]

        self.transform =  transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def get_image_dataloader(
    folder_path,
    batch_size=32,
    image_size=64,
    shuffle=True,
    num_workers=4
    ):

    dataset = ImageDataset(folder_path, image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader

def convlayer_enc(n_input, n_output, k_size, stride, padding, bn=False):
    block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
    if bn:
        block.append(nn.BatchNorm2d(n_output))
    block.append(nn.LeakyReLU(0.2, inplace=True))
    return block

def convlayer_dec(n_input, n_output, k_size, stride, padding):
    block = [
        nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding,bias=False),
        nn.BatchNorm2d(n_output),
        nn.ReLU(inplace=True)
    ]
    return block

class VAE(nn.Module):

    def __init__(self, latent_dim=128, no_of_sample=10, batch_size=32, channels=3):
        super().__init__()

        self.latent_dim = latent_dim
        self.no_of_sample = no_of_sample
        self.batch_size = batch_size
        self.channels = channels

        self.encoder = nn.Sequential(
            *convlayer_enc(self.channels, 64, 4, 2, 2),
            *convlayer_enc(64, 128, 4, 2, 2),
            *convlayer_enc(128, 256, 4, 2, 2, bn=True),
            *convlayer_enc(256, 512, 4, 2, 2, bn=True),
            nn.Conv2d(512, self.latent_dim*2, 4, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder = nn.Sequential(
            *convlayer_dec(self.latent_dim, 512, 4, 2, 1),
            *convlayer_dec(512, 256, 4, 2, 1),
            *convlayer_dec(256, 128, 4, 2, 1),
            *convlayer_dec(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, self.channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = x[:, :self.latent_dim, :, :]
        logvar = x[:, self.latent_dim:, :, :]
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z.view(-1, 3 * 64 * 64)

    def reparameterize(self, mu, logvar):
        if self.training:
            sample_z = []
            for _ in range(self.no_of_sample):
                std = logvar.mul(0.5).exp_()
                eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
                sample_z.append(eps.mul(std).add_(mu))
            return sample_z
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if self.training:
            return [self.decode(i) for i in z], mu, logvar
        return self.decode(z), mu, logvar

    def loss_func(self, recon_x, x, mu, logvar):
        if self.training:
            BCE = 0
            for recon_x_one in recon_x:
                BCE += F.binary_cross_entropy(recon_x_one, x.view(-1, 3 * 64 * 64))
            BCE /= len(recon_x)
        else:
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3 * 64 * 64))
        
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= self.batch_size * 3 * 64 * 64
        return BCE + KLD, BCE, KLD

def log_reconstructions(model, data, epoch, num_images=8):
    model.eval()
    with torch.no_grad():
        recon_batch, _, _ = model(data[:num_images])
        recon_batch = recon_batch.view(-1, 3, 64, 64)
        comparision = torch.cat([
            data[:num_images],
            recon_batch
        ])

        grid = torchvision.utils.make_grid(comparison, nrow=num_images, normalize=True)
        images = wandb.Image(
            grid.cpu(),
            caption=f'Epoch {epoch}: Top: Original, Bottom: Reconstructed'
        )
        wandb.log({"reconstructions": images})




def train(folder_path):
    lr = 0.001
    epochs = 50
    latent_dim = 32
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_image_dataloader(folder_path)
    model = VAE(latent_dim=latent_dim, batch_size=batch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    x = next(iter(dataloader))

    wandb.init(
        project="vae-celeba-faces",
        config = {
            "lr":lr,
            "epochs":epochs,
            "latent_dim":latent_dim,
            "batch_size":batch_size,
        }
    )

    global_step = 0
    for epoch in range(epochs):
        model.train()
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)

            loss, bce_loss, kld_loss = model.loss_func(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                wandb.log({
                    "loss":loss.item(),
                    "bce_loss":bce_loss.item(),
                    "kld_loss":kld_loss.item(),
                },
                step=global_step)
            global_step += 1
            

        if epoch % 5 == 0:
            log_reconstructions(model, x, epoch)
    
    wandb.finish()

if __name__ == "__main__":
    train("img_align_celeba")




