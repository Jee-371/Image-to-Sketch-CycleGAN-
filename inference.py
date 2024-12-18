import torch
from torchvision import transforms
from PIL import Image
from training import Generator

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def infer(generator, input_image_path, output_image_path):
    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = data_transform(input_image).unsqueeze(0).to(device)
    generator.eval()
    with torch.no_grad():
        output_tensor = generator(input_tensor).cpu().squeeze(0)
    output_image = transforms.ToPILImage()(output_tensor)
    output_image.save(output_image_path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    generator.load_state_dict(torch.load("generator.pth", map_location=device))

    input_image_path = "download.jpg"
    output_image_path = "output.jpg"
    infer(generator, input_image_path, output_image_path)
    print(f"Output saved to {output_image_path}")
