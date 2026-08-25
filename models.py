import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import CSP

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def run_stage2_csp_svm(epochs):
    print("\n" + "="*60)
    print("[분기 1] 단일 피험자 특화 머신러닝 (CSP + SVM) 시작")
    print("="*60)

    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1]

    csp = CSP(n_components=8, log=True, norm_trace=False)
    svm = SVC(kernel='rbf', C=0.5, gamma='scale')
    clf = Pipeline([('CSP', csp), ('SVM', svm)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
    print(f"CSP+SVM 평균 분류 정확도: {scores.mean() * 100:.2f}%\n")

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
    print("\n" + "="*60)
    print("[분기 2] 다중 피험자 통합 딥러닝 (PyTorch EEGNet) 시작")
    print("="*60)

    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1]
    n_samples, n_channels, n_times = X.shape

    X_reshaped = X.transpose(0, 2, 1).reshape(-1, n_channels)
    X = StandardScaler().fit_transform(X_reshaped).reshape(n_samples, n_times, n_channels).transpose(0, 2, 1)

    unique_labels = np.unique(y)
    y_mapped = np.array([np.where(unique_labels == label)[0][0] for label in y])

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y_mapped, dtype=torch.long))
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SimpleEEGNet(n_channels, n_times, len(unique_labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    max_epochs = 80
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
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
                _, predicted = torch.max(model(batch_x).data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  - Epoch [{epoch+1:02d}/{max_epochs}] Loss: {total_loss/len(train_loader):.4f} | Val Acc: {100*correct/total:.2f}%")
