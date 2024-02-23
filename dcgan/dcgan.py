import wandb
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

class Generator(nn.Module):

    def __init__(self, latent_dim, num_feature_maps, channels=3):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_feature_maps = num_feature_maps
        self.channels = channels
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, num_feature_maps*8,
                               kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(num_feature_maps*8),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(num_feature_maps*8, num_feature_maps*4,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feature_maps*4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(num_feature_maps*4, num_feature_maps*2,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feature_maps*2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(num_feature_maps*2, num_feature_maps,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(num_feature_maps, channels,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        img = self.model(noise)
        return img


class Discriminator(nn.Module):

    def __init__(self, num_feature_maps, channels=3):
        super().__init__()

        self.num_feature_maps = num_feature_maps
        self.channels = channels
        self.model = nn.Sequential(
            nn.Conv2d(channels, num_feature_maps,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_feature_maps, num_feature_maps*2,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_feature_maps*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_feature_maps*2, num_feature_maps*4,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_feature_maps*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_feature_maps*4, num_feature_maps*8,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_feature_maps*8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_feature_maps*8, 1,
                      kernel_size=4, stride=1, padding=0,
                      bias=False),
            nn.Flatten(),
        )

    def forward(self, img):
        logits = self.model(img)
        return logits


def get_mnist_dataloaders(batch_size=32, num_workers=1,
                          validation_fraction=None,
                          train_transforms=None, test_transforms=None):
    
    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.MNIST(root="data", train=True,
                                   download=True, transform=train_transforms)
    
    valid_dataset = datasets.MNIST(root="data", train=True,
                                   transform=test_transforms)

    test_dataset = datasets.MNIST(root="data", train=False,
                                  transform=test_transforms)

    if validation_fraction is not None:
        validation_size = int(validation_fraction * 60000)
        train_indices = torch.arange(0, 60000 - validation_size)
        valid_indices = torch.arange(60000 - validation_size, 60000)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      sampler=train_sampler)

        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      sampler=valid_sampler)

    else:
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=True)

    if validation_fraction is None:
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, valid_dataloader, test_dataloader


def get_celeba_dataloaders(batch_size=32, num_workers=1,
                          train_transforms=None, test_transforms=None):
    
    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.CelebA(root="data", split="train",
                                   download=True, transform=train_transforms)

    valid_dataset = datasets.CelebA(root="data", split="valid",
                                   transform=train_transforms)

    test_dataset = datasets.CelebA(root="data", split="test",
                                  transform=test_transforms)

    train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=True)

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader


def train(config):
    if config["use_wandb"]:
        wandb.init(
            project=config["wandb_project_name"],
            config=config,
            sync_tensorboard=True,
        )
    writer = SummaryWriter()
    custom_transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((160,160)),
        torchvision.transforms.Resize([config["image_height"], config["image_width"]]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)) ## For MNIST
    ])

    train_dataloader, test_dataloader = get_mnist_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        train_transforms=custom_transforms,
        test_transforms=custom_transforms,
    )

    generator = Generator(latent_dim=100, num_feature_maps=64, channels=1)
    discriminator = Discriminator(num_feature_maps=64, channels=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator.to(device)
    discriminator.to(device)

    optimizer_g = optim.Adam(generator.parameters(), betas=(0.5, 0.999),lr=config["generator_lr"])
    optimizer_d = optim.Adam(generator.parameters(), betas=(0.5, 0.999),lr=config["discriminator_lr"])

    # print("--------Generator-----------")
    # print(generator)

    # print("--------Discriminator-------")
    # print(discriminator)

    latent_dim = 100
    loss_func = F.binary_cross_entropy_with_logits
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    global_step = 0
    for epoch in range(config["num_epochs"]):
        generator.train()
        discriminator.train()

        for batch_idx, (images, _) in enumerate(train_dataloader):
            batch_size = images.size(0)
            real_images = images.to(device)
            real_labels = torch.ones(batch_size, device=device)

            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator.forward(noise)
            fake_labels = torch.zeros(batch_size, device=device)

            flipped_fake_labels = real_labels

            discriminator_real_pred = discriminator.forward(real_images).view(-1)
            discriminator_real_loss = loss_func(discriminator_real_pred, real_labels)

            discriminator_fake_pred = discriminator.forward(fake_images.detach()).view(-1)
            discriminator_fake_loss = loss_func(discriminator_fake_pred, fake_labels)

            discriminator_loss = 0.5*(discriminator_fake_loss +  discriminator_real_loss)
        
            optimizer_d.zero_grad()
            discriminator_loss.backward()
            optimizer_d.step()

            discriminator_fake_pred = discriminator.forward(fake_images).view(-1)
            generator_loss = loss_func(discriminator_fake_pred, flipped_fake_labels)

            optimizer_g.zero_grad()
            generator_loss.backward()
            optimizer_g.step()

            writer.add_scalar("discriminator_loss", discriminator_loss.cpu().mean(), global_step)
            writer.add_scalar("generator_loss", generator_loss.cpu().mean(), global_step)
            global_step += 1

        with torch.no_grad():
            generated_images = generator.forward(fixed_noise).detach().cpu()
            # images_grid = torchvision.utils.make_grid(generated_image, normalize=True)
            generated_images = generated_images.repeat_interleave(3, dim=1)
            writer.add_images("generated_image", generated_images, global_step)
    
    writer.close()



if __name__ == "__main__":
    config = {"batch_size":64, 
              "num_workers":1, 
              "image_height":64,
              "image_width":64, 
              "generator_lr": 0.0002,
              "discriminator_lr": 0.0002,
              "num_epochs": 20,
              "use_wandb": False,
              "wandb_project_name":"DCGAN"}
    
    train(config)

