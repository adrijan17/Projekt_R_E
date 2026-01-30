import torch
import csv
from tqdm import tqdm

from wire_dataset import WireDataset
from unet import UNet


def basic_metrics(pred, target, eps=1e-7):
    pred = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return dice.item(), iou.item(), accuracy.item(), precision.item()


def evaluate_test_set_csv(
    data_path,
    model_pth,
    device,
    output_csv="evaluation.csv",
    threshold=0.5
):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=device))
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    test_dataset = WireDataset(data_path, test=True)

    total_loss = 0.0
    dice_sum = iou_sum = acc_sum = prec_sum = 0.0

    with torch.inference_mode():
        for img, gt_mask in tqdm(test_dataset, desc="Evaluating"):
            img = img.unsqueeze(0).float().to(device)
            gt_mask = gt_mask.float().to(device)

            logits = model(img)
            loss = criterion(logits, gt_mask.unsqueeze(0))
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            pred = (probs > threshold).float()

            dice, iou, acc, prec = basic_metrics(
                pred.squeeze(0),
                gt_mask
            )

            dice_sum += dice
            iou_sum += iou
            acc_sum += acc
            prec_sum += prec

    n = len(test_dataset)

    # CSV datoteka
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "number_of_samples",
            "loss",
            "dice",
            "iou",
            "accuracy",
            "precision",
            "threshold"
        ])
        writer.writerow([
            n,
            total_loss / n,
            dice_sum / n,
            iou_sum / n,
            acc_sum / n,
            prec_sum / n,
            threshold
        ])


if __name__ == "__main__":
    DATA_PATH = "./data"
    MODEL_PATH = "./models/unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate_test_set_csv(data_path=DATA_PATH, model_pth=MODEL_PATH, device=device)
