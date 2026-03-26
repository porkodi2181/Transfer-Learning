# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model.

## DESIGN STEPS
### STEP 1:
Import required libraries.Then dataset is loaded and define the training and testing dataset.

### STEP 2:
initialize the model,loss function,optimizer. CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.

### STEP 3:
Train the model with training dataset.

### STEP 4:
Evaluate the model with testing dataset.

### STEP 5:
Make Predictions on New Data.



## PROGRAM

```python
# Load Pretrained Model and Modify for Transfer Learning

from torchvision.models import VGG19_Weights
model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes

num_classes = len(train_dataset.classes)
model.classifier[6] = nn.Linear(4096, num_classes)

# Include the Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.001)

# Train the model

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
    # Plot training and validation loss
    print("Name: PORKODI B")
    print("Register Number: 212224240114")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

<img width="759" height="749" alt="image" src="https://github.com/user-attachments/assets/4e4580e3-1665-46b9-988d-7e3fd84ce0e3" />



### Confusion Matrix
<img width="693" height="619" alt="image" src="https://github.com/user-attachments/assets/1b4118bb-65e9-43ab-bf1a-6329ee5cba11" />

<img width="506" height="224" alt="image" src="https://github.com/user-attachments/assets/b98725bc-e187-49f5-ac0a-3589eda78c47" />




### Classification Report

<img width="445" height="399" alt="image" src="https://github.com/user-attachments/assets/11456037-57f9-460b-821e-469875e5ed31" />


### New Sample Prediction
<img width="428" height="402" alt="image" src="https://github.com/user-attachments/assets/7950a35f-9940-43f8-a3ab-12b114d46703" />


## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
