import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_x, image_y = 224, 224  # ResNet expects 224x224 images
batch_size = 64
train_dir = "chords"
num_of_classes = 7

# 데이터 로더 설정
train_transforms = transforms.Compose([
    transforms.Resize((image_x, image_y)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop((image_x, image_y), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ResNet normalization
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Pretrained ResNet 모델 로드 및 수정
model = models.resnet18(pretrained=True)  # Load ResNet18 model
for param in model.parameters():
    param.requires_grad = False  # Freeze pretrained layers

# Replace the final layer for custom classification
model.fc = nn.Linear(model.fc.in_features, num_of_classes)
model = model.to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Only train the new classifier layer

# 학습 및 검증 루프
num_epochs = 5
best_accuracy = 0.0
save_path = "guitar_learner_resnet.pth"

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 검증 단계
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # 모델 저장
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved with accuracy: {best_accuracy:.2f}%")

print(f"Training completed. Best validation accuracy: {best_accuracy:.2f}%")
