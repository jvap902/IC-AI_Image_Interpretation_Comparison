from tqdm import tqdm
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset
from src.config import device, batch_size
from .evaluator import getFeatureTensors

class LinearHead(nn.Module):
    """Simple linear classification head for the final features."""
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc(x)

def trainHead(head, train_features, epochs):    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(head.parameters(), lr=0.01) # Increased learning rate is common for linear probes

    # 3. Training loop (Linear Probe)
    head.train()
    for epoch in tqdm(range(epochs), desc=f"Training Head"):
        for inputs, labels in train_features:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = head(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
    
    return head

def newTrainedHead(modelc, num_classes, epochs=10):
    
    train_features = getFeatureTensors(modelc, modelc.train_loader)
    train_labels = Tensor([label for _, label in modelc.train_loader.dataset])
    
    train_features_dataset = TensorDataset(train_features, train_labels)
    train_features_loader = DataLoader(train_features_dataset, batch_size=batch_size, shuffle=True)
    
    new_head = LinearHead(input_dim=train_features.size(1), num_classes=num_classes).to(device)
    
    new_head = trainHead(new_head, train_features_loader, epochs=epochs)
    
    modelc.setHead(new_head)