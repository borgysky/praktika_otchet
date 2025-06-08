import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from model import UNet
import argparse
from PyQt5.QtCore import QObject, pyqtSignal
from PIL import Image
from test import dice_coefficient, iou_score

class dataset(Dataset):
    def __init__(self, image_dir, threshold=0.5, images=None):
        self.image_dir = image_dir
        self.threshold = threshold
        if images is not None:
            self.images = images
        else:
            self.images = [
                img for img in os.listdir(image_dir)
                if img.lower().endswith(('.jpg', '.jpeg'))
            ]
        
        if not self.images:
            raise ValueError(f"Папка {image_dir} не содержит изображений формата .jpg или .jpeg")
        
        for img_name in self.images:
            img_path = os.path.join(image_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                raise ValueError(f"Папка содержит поврежденные изображения")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(path).convert("L")
        if img is None:
            raise ValueError(f"Ошибка загрузки изображения: {path}")
        img = img.resize((624, 320))
        img = np.array(img).astype(np.float32) / 255.0
        mask = (img < self.threshold).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        return torch.tensor(img), torch.tensor(mask)

class Trainer(QObject):
    epoch_start_signal = pyqtSignal(int, int)
    epoch_complete_signal = pyqtSignal(int, float, float, float, float)
    batch_progress_signal = pyqtSignal(int, int, float)
    training_complete_signal = pyqtSignal(str)

    def __init__(self, image_dir, save_path, batch_size=1, epochs=25, lr=1e-4, threshold=0.3):
        super().__init__()
        self.image_dir = image_dir
        self.save_path = save_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold

    def run(self):
        self.training_complete_signal.emit(f"Загрузка датасета из: {self.image_dir}")
        all_images = [
            img for img in os.listdir(self.image_dir)
            if img.lower().endswith(('.jpg', '.jpeg'))
        ]
        np.random.seed(42) 
        all_images = np.array(all_images)
        np.random.shuffle(all_images)
        train_size = int(0.8 * len(all_images))
        train_images = all_images[:train_size].tolist()
        test_images = all_images[train_size:].tolist()
        
        train_dataset = dataset(self.image_dir, threshold=self.threshold, images=train_images)
        test_dataset = dataset(self.image_dir, threshold=self.threshold, images=test_images)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet().to(device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            total_loss = 0
            self.epoch_start_signal.emit(epoch + 1, self.epochs)
            model.train()
            for i, (imgs, masks) in enumerate(train_dataloader, 1):
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = criterion(preds, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                self.batch_progress_signal.emit(i, len(train_dataloader), loss.item())
            
            avg_loss = total_loss / len(train_dataloader)
            
            model.eval()
            test_loss = 0
            dice_scores = []
            iou_scores = []
            with torch.no_grad():
                for imgs, masks in test_dataloader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    preds = model(imgs)
                    loss = criterion(preds, masks)
                    test_loss += loss.item()
                    pred_mask = (preds > self.threshold).float()
                    dice_scores.append(dice_coefficient(pred_mask, masks).item())
                    iou_scores.append(iou_score(pred_mask, masks).item())
            
            avg_test_loss = test_loss / len(test_dataloader) if len(test_dataloader) > 0 else 0
            avg_dice = np.mean(dice_scores) if dice_scores else 0
            avg_iou = np.mean(iou_scores) if iou_scores else 0
            self.epoch_complete_signal.emit(epoch + 1, avg_loss, avg_test_loss, avg_dice, avg_iou)

        torch.save(model.state_dict(), self.save_path)
        self.training_complete_signal.emit(f"Модель сохранена в: {self.save_path}")

def main():
    parser = argparse.ArgumentParser(description="Обучение нейросети UNet")
    parser.add_argument('--data', required=True, help="Путь к папке с изображениями")
    parser.add_argument('--output', required=True, help="Путь для сохранения модели (model.pth)")
    parser.add_argument('--batch-size', type=int, default=4, help="Размер батча (по умолчанию: 4)")
    parser.add_argument('--epochs', type=int, default=25, help="Количество эпох (по умолчанию: 25)")
    args = parser.parse_args()

    trainer = Trainer(args.data, args.output, batch_size=args.batch_size, epochs=args.epochs)
    trainer.run()

if __name__ == "__main__":
    main()