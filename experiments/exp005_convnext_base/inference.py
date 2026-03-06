import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

# train.py에서 모델과 데이터셋 import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import StructureDataset, ConvNeXtClassifier, DATA_ROOT, IMG_SIZE

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_ds = StructureDataset(
        csv_path=os.path.join(DATA_ROOT, "sample_submission.csv"),
        data_dir=os.path.join(DATA_ROOT, "test"),
        transform=transform,
        is_test=True,
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    model = ConvNeXtClassifier(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=DEVICE, weights_only=True))
    model.eval()

    all_ids, all_probs = [], []
    with torch.no_grad():
        for images, sample_ids in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_ids.extend(sample_ids)
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs)
    submission = pd.DataFrame({
        "id": all_ids,
        "unstable_prob": all_probs[:, 0],
        "stable_prob": all_probs[:, 1],
    })

    save_path = os.path.join(SAVE_DIR, "submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head())


if __name__ == "__main__":
    main()
