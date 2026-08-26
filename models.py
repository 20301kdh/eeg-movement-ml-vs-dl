# models.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import CSP

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def run_stage2_csp_svm(epochs):
    print("  [ML] 머신러닝 (CSP + SVM) 평가 시작")
    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1]

    csp = CSP(n_components=8, log=True, norm_trace=False)
    svm = SVC(kernel='rbf', C=0.5, gamma='scale')
    clf = Pipeline([('CSP', csp), ('SVM', svm)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
    print(f"  -> CSP+SVM 평균 분류 정확도: {scores.mean() * 100:.2f}%\n")

class SimpleEEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes):
        super(SimpleEEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 80), padding=(0, 40), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.pooling2 = nn.AvgPool2d((1, 4))
        self.dropout = nn.Dropout(p=0.5)

        def _get_size():
            with torch.no_grad():
                x = torch.zeros(1, 1, n_channels, n_times)
                x = self.pooling2(self.conv2(self.conv1(x)))
                return x.numel()
        self.fc = nn.Linear(_get_size(), n_classes)

    def forward(self, x):
        x = torch.nn.functional.elu(self.batchnorm1(self.conv1(x)))
        x = torch.nn.functional.elu(self.batchnorm2(self.conv2(x)))
        x = self.pooling2(x)
        x = self.dropout(x)
        return self.fc(x.view(x.size(0), -1))

def run_stage3_eegnet(epochs):
    print("  [DL] 딥러닝 (PyTorch EEGNet) 평가 시작")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1]
    n_samples, n_channels, n_times = X.shape

    unique_labels = np.unique(y)
    y_mapped = np.array([np.where(unique_labels == label)[0][0] for label in y])

    # 1. 데이터 누수 방지를 위한 엄격한 Train/Val 분할 (8:2)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
    )

    # 2. Train 데이터 기준으로만 fit_transform 수행 (Val은 transform만)
    scaler = StandardScaler()
    
    X_train_reshaped = X_train.transpose(0, 2, 1).reshape(-1, n_channels)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape[0], n_times, n_channels).transpose(0, 2, 1)
    
    X_val_reshaped = X_val.transpose(0, 2, 1).reshape(-1, n_channels)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape[0], n_times, n_channels).transpose(0, 2, 1)

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(1), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SimpleEEGNet(n_channels, n_times, len(unique_labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    max_epochs = 80
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                _, predicted = torch.max(model(batch_x).data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        if (epoch+1) % 20 == 0 or epoch == 0:
            print(f"    - Epoch [{epoch+1:02d}/{max_epochs}] Loss: {total_loss/len(train_loader):.4f} | Val Acc: {100*correct/total:.2f}%")
