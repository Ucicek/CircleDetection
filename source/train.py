import os
import torch
from tqdm import tqdm
import yaml

from utils import get_custom_optimizer, get_dataloader, get_network, get_optimizer
from loss import mae_loss


def train(model, config, epochs, device):
    train_loader = get_dataloader(config, train=True, generate=True)
    val_loader = get_dataloader(config, train=False, generate=False )
    optimizer = get_custom_optimizer(config, model)
    lr_reduction = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, verbose=True)

    criterion_mae = mae_loss
    criterion_l1 = torch.nn.L1Loss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        # Training loop
        train_loss = training_loop(model, train_loader, optimizer, criterion_mae, criterion_l1, device)

        # Validation loop
        val_loss = validation_loop(model, val_loader, criterion_mae, criterion_l1, device)

        # Learning rate adjustment
        lr_reduction.step(val_loss)

        if val_loss < best_val_loss:
            checkpoint_dir = config['TRAIN']['CHECKPOINT']
            checkpoint_filename = f"model_checkpoint_val_loss_{val_loss:.4f}.pth"
            complete_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save the model
            print(f"Saving model with val loss: {val_loss:.4f} to {complete_checkpoint_path}")
            torch.save(model.state_dict(), complete_checkpoint_path)

            best_val_loss = val_loss


def training_loop(model, dataloader, optimizer, criterion_mae, criterion_l1, device):
    model.train()
    running_loss = 0.0
    i = 0
    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion_mae(outputs, targets) + criterion_l1(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Train Loss: {epoch_loss:.4f}')
    return epoch_loss

def validation_loop(model, dataloader, criterion_mae, criterion_l1, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion_mae(outputs, targets) + criterion_l1(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Val Loss: {epoch_loss:.4f}')
    return epoch_loss

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model = get_network(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config.get('TRAIN', {}).get('EPOCHS', 5)  # Default to 5 epochs if not specified
    train(model, config, epochs, device)

if __name__ == "__main__":
    main()
