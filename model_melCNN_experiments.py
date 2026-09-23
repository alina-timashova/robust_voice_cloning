import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import librosa
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from data_prep_parallel import load_split


ROOTS = {
    'adult': os.path.expanduser('~/nn-dataprep/codecfake_preprocessed'),
    'child': os.path.expanduser('~/nn-dataprep/samromur_preprocessed_sp'),
    'dac_adult': os.path.expanduser('~/nn-dataprep/dac_test_adult'),
    'dac_child': os.path.expanduser('~/nn-dataprep/dac_test_child'),
}

EXPERIMENTS = {
    '1': {
        'train': [('adult', 'clean')],
        'eval': [('adult', 'clean')],
        'tests': {
            'clean': [('adult', 'clean')],
            'compressed': [('adult', 'compressed')],
            'child_clean': [('child', 'clean')],
            'clean_adult_DAC': [('dac_adult', 'clean')]
        },
    },

    '2': {
        'train': [('adult', 'clean'), ('adult', 'compressed')],
        'eval': [('adult', 'clean'), ('adult', 'compressed')],
        'tests': {
            'adult_compressed': [('adult', 'compressed')],
            'child_compressed': [('child', 'compressed')]
        },
    },

    '3': {
        'train': [('adult', 'clean'), ('adult', 'compressed'),
                  ('child', 'clean'), ('child', 'compressed')],
        'eval': [('adult', 'clean'), ('adult', 'compressed'),
                  ('child', 'clean'), ('child', 'compressed')],
        'tests': {
            'all_compressed_ages': [('adult', 'clean'), ('adult', 'compressed'),
                                    ('child', 'clean'), ('child', 'compressed')],
            'DAC': [('dac_adult', 'clean'), ('dac_adult', 'compressed'),
                    ('dac_child', 'clean'), ('dac_child', 'compressed')],
        },
    },
}

ACTIVE_EXPERIMENT = '1' # choose experiment you are running
FINAL_TEST_RUN = True # False if you are doing hyperparameter tuning, True if you are ready to do test set 
                        # NOTE: test set can be used only ONCE, do NOT use test set if you are planning to do any changes to a model

TEST_ONLY = True

TARGET_FRAMES = 126
N_MELS = 80
BATCH_SIZE = 64
NUM_EPOCHS = 100
LR = 5e-4

def mel_transform(audio, target_frames=126):

  mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=80)
  mel_db = librosa.power_to_db(mel, ref=np.max) # normalize

  # Crop if too long
  if mel_db.shape[1] > target_frames:
    mel_db = mel_db[:, :target_frames]

  # Pad if too short
  elif mel_db.shape[1] < target_frames:
    padding = target_frames - mel_db.shape[1]
    mel_db = np.pad(
        mel_db,
        ((0, 0), (0, padding)),
        mode='constant',
        constant_values=mel_db.min()
    )

  return mel_db.astype(np.float32)

def mel_to_tensor(mel):
    # NumPy array: (N, 80, 126)
    tensor = torch.from_numpy(mel).float()

    # Add channel dimension: (N, 1, 80, 126)
    tensor = tensor.unsqueeze(1)

    return tensor

def load_sources(split, sources, transform=None):
    '''
    Load and concatenate one or more (dataset, version) sources for a split
    Returns X (list), y (list) with all versions stacked
    '''

    X_all, y_all, ids_all = [], [], []

    for dataset, version in sources:
        root = ROOTS[dataset]
        X, y, ids = load_split(split, version, root, transform=transform)
        X_all.extend(X)
        y_all.extend(y)
        ids_all.extend(ids)
        print(f'Loaded {len(X):>6} from {dataset}/{split}/{version}')

    return X_all, y_all, ids_all

class MFM(nn.Module):
    '''
    Max-Feature-Map activation
    '''
    def forward(self, x):
        c = x.size(1)

        if c % 2 != 0:
            raise ValueError("MFM needs even number of channels")

        x1 = x[:, :c // 2, :, :]
        x2 = x[:, c // 2:, :, :]

        return torch.max(x1, x2)

def calculate_eer(y_true, y_score):

    # ROC curve
    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score
    )

    # False Rejection Rate
    fnr = 1 - tpr

    # Find where FPR and FNR intersect
    eer = brentq(
        lambda x: 1. - x - interp1d(
            fpr, tpr
        )(x),
        fpr[0],
        fpr[-1]
    )

    return eer

class LCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * 2,
            kernel_size=3,
            padding=1
        )

        self.bn = nn.BatchNorm2d(out_channels * 2)

        self.mfm = MFM()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mfm(x)
        x = self.pool(x)

        return x

class LCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            LCNNBlock(1, 32),
            LCNNBlock(32, 64),
            LCNNBlock(64, 128),
            LCNNBlock(128, 256)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)

        # Calculate loss
        loss = criterion(outputs, y_batch)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def predict(model, loader, device):
    model.eval()

    all_predictions = []
    all_labels = []
    all_scores = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            probabilities = torch.softmax(outputs, dim=1)

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y_batch.numpy()
            )

            # Probability of fake class
            all_scores.extend(
                probabilities[:, 1].cpu().numpy()
            )

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_scores)
    )

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp = EXPERIMENTS[ACTIVE_EXPERIMENT]

    ckpt = f'best_lcnn_exp{ACTIVE_EXPERIMENT}.pth'
    model = LCNN(num_classes=2).to(device)

    if not TEST_ONLY:

        X_train, y_train, _ = load_sources('train', exp['train'], transform=mel_transform)
        X_eval, y_eval, _ = load_sources('eval', exp['eval'], transform=mel_transform)

        X_train = mel_to_tensor(np.array(X_train))
        X_eval = mel_to_tensor(np.array(X_eval))

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_eval = torch.tensor(y_eval, dtype=torch.long)
        print('shapes of X tensors', X_train.shape, X_eval.shape)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        eval_loader = DataLoader(TensorDataset(X_eval, y_eval), batch_size=BATCH_SIZE, shuffle=False)

        # class weights
        counts = np.bincount(y_train.numpy(), minlength=2).astype(np.float64) # counts how many time each label, 1 and 0, appears in dataset
                                                                                # minlength=2 guarantees that array always has 2 entries: if class is not present then has 0 as entry
                                                                                # astype(np.float64) for later division
        weights = counts.sum() / (2.0 * np.maximum(counts, 1)) # total / (number of classes * count of that class) -> returns array of numbers [weight_0, weight_1]
                                                                # NOTE: weights correspond to label based on POSITION, first position 0 -> real, second position 1 -> fake
                                                                # necessary to keep labels unchanged, i.e. do NOT reassign 1 to real and 0 to fake
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f'Class counts (real, fake) = {counts.astype(int)} | weights = {weights.round(3)}')

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        best_eval_loss = float('inf')
        patience = 3
        epochs_no_improve = 0

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader,
                                                    criterion, optimizer, device)
            eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)
            scheduler.step()

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt)
            else:
                epochs_no_improve += 1

            print(f'Epoch {epoch+1}/{NUM_EPOCHS} | '
                f'Train loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                f'Eval loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}')

            if epochs_no_improve >= patience:
                print(f'early stopping at epoch {epoch+1}')
                break

    else:
        print('TEST_ONLY=True -> skipping training')

    model.load_state_dict(torch.load(ckpt, map_location=device))

    if FINAL_TEST_RUN:
        for test_name, test_sources in exp['tests'].items():
            try:
                X_test, y_test, _ = load_sources('test', test_sources, transform=mel_transform)
                X_test = mel_to_tensor(np.array(X_test))
                y_test = torch.tensor(y_test, dtype=torch.long)
                test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

                y_true, y_pred, y_score = predict(model, test_loader, device)
                print(f"\n=== TEST: {test_name} (experiment #{ACTIVE_EXPERIMENT}) ===")
                print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
                print(f"Precision: {precision_score(y_true, y_pred):.4f}")
                print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
                print(f"F1-score:  {f1_score(y_true, y_pred):.4f}")
                print("\nConfusion Matrix:")
                print(confusion_matrix(y_true, y_pred))
                eer = calculate_eer(y_true, y_score)
                print(f"EER: {eer:.4f}  ({eer * 100:.2f}%)")
                
            except Exception as e:
                print(f'\n TEST: {test_name} SKIPPED ({e})') # if test set is not ready yet
                continue 

if __name__ == '__main__':
    main()