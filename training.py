import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.main(x)

class SketchyDataset(Dataset):
    def __init__(self, photo_dir, sketch_dir, transform):
        self.photo_paths = []
        self.sketch_paths = []
        self.transform = transform

        for category in os.listdir(photo_dir):
            photo_category_path = os.path.join(photo_dir, category)
            sketch_category_path = os.path.join(sketch_dir, category)

            if os.path.isdir(photo_category_path) and os.path.isdir(sketch_category_path):
                photo_files = sorted(glob.glob(os.path.join(photo_category_path, "*.jpg")))
                sketch_files = sorted(glob.glob(os.path.join(sketch_category_path, "*.png")))

                self.photo_paths.extend(photo_files)
                self.sketch_paths.extend(sketch_files)

    def __len__(self):
        return min(len(self.photo_paths), len(self.sketch_paths))

    def __getitem__(self, idx):
        photo = Image.open(self.photo_paths[idx]).convert('RGB')
        sketch = Image.open(self.sketch_paths[idx]).convert('RGB')

        if self.transform:
            photo = self.transform(photo)
            sketch = self.transform(sketch)

        return photo, sketch

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_data(photo_dir, sketch_dir):
    dataset = SketchyDataset(photo_dir, sketch_dir, transform)
    return DataLoader(dataset, batch_size=16, shuffle=True)

def train(generator, discriminator, dataloader, num_epochs, device):
    criterion = nn.MSELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, (photos, sketches) in enumerate(dataloader):
            photos = photos.to(device)
            sketches = sketches.to(device)

            fake_sketches = generator(photos)

            optimizer_d.zero_grad()
            real_loss = criterion(discriminator(sketches), torch.ones_like(discriminator(sketches)))
            fake_loss = criterion(discriminator(fake_sketches.detach()), torch.zeros_like(discriminator(fake_sketches)))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            g_loss = criterion(discriminator(fake_sketches), torch.ones_like(discriminator(fake_sketches)))
            g_loss.backward()
            optimizer_g.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    photo_dir = "./sketches/photo"
    sketch_dir = "./sketches/sketch"

    data_loader = load_data(photo_dir, sketch_dir)

    train(generator, discriminator, data_loader, num_epochs=10, device=device)

    torch.save(generator.state_dict(), "generator.pth")
