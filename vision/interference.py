import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from wire_dataset import WireDataset
from unet import UNet

def pred_show_image_grid(data_path, model_pth, device, num_display=5):
    # Učitavanje modela
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=device))
    model.eval()
    
    image_dataset = WireDataset(data_path, test=True)
    images, orig_masks, pred_masks = [], [], []

    with torch.inference_mode():
        # Testiranje i prikaz samo prvih num_display fotografija
        for idx, (img, orig_mask) in enumerate(image_dataset):
            if idx >= num_display:
                break

            img_batch = img.float().to(device).unsqueeze(0)
            
            # Forward pass + Sigmoid + Threshold
            output = model(img_batch)
            probs = torch.sigmoid(output)
            pred_mask = (probs > 0.5).float()

            # Priprema az vizualizaciju
            images.append(img.permute(1, 2, 0).cpu())
            orig_masks.append(orig_mask.permute(1, 2, 0).cpu())
            pred_masks.append(pred_mask.squeeze(0).permute(1, 2, 0).cpu())

    # Prikaz 3 reda: original image / ground trouth / prediction
    fig, axes = plt.subplots(3, num_display, figsize=(num_display * 3, 9))
    
    for i in range(num_display):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f"Input {i+1}")
        axes[1, i].imshow(orig_masks[i], cmap="gray")
        axes[1, i].set_title(f"GT {i+1}")
        axes[2, i].imshow(pred_masks[i], cmap="gray")
        axes[2, i].set_title(f"Pred {i+1}")

        for row in range(3):
            axes[row, i].axis("off")
    
    plt.tight_layout()
    plt.show()


def single_image_inference(image_pth, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    img = transform(Image.open(image_pth)).float().to(device).unsqueeze(0)
   
    with torch.inference_mode():
        output = model(img)
        pred_mask = torch.sigmoid(output)
        pred_mask = (pred_mask > 0.5).float()

    # Prijelaz na CPU za matplotlib
    img_plot = img.squeeze(0).cpu().permute(1, 2, 0)
    mask_plot = pred_mask.squeeze(0).cpu().permute(1, 2, 0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_plot)
    ax[0].set_title("Original Image (512x512px)")
    ax[0].axis("off")
    ax[1].imshow(mask_plot, cmap="gray")
    ax[1].set_title("Predicted Wire Mask")
    ax[1].axis("off")
    plt.show()


if __name__ == "__main__":
    SINGLE_IMG_PATH = "./data/test_data/0433.png"
    DATA_PATH = "./data"
    MODEL_PATH = "./models/unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prikaz prvih 5 slika
    pred_show_image_grid(DATA_PATH, MODEL_PATH, device, num_display=5)

    # Prikaz jedne slike
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)
