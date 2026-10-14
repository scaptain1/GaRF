"""GaRF:three-class toxicity classification.
"""
import argparse
import html
import json
import os
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LAYERS = (8, 12, 16, -1, -2, -3, -4, -5)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_toxicity_text(text):
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', 'URL', text)
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    return re.sub(r'\s+', ' ', text).strip()


class ExplicitNormAwareClassifier(nn.Module):
    def __init__(self, num_layers=8, cls_dim=1024, hidden_dim=512,
                 fusion_dim=256, num_classes=3):
        super().__init__()
        self.direction_dim = num_layers * cls_dim
        self.direction_proj = nn.Sequential(
            nn.Linear(self.direction_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(0.45), nn.Linear(hidden_dim, fusion_dim), nn.ReLU())
        self.norm_proj = nn.Sequential(nn.Linear(num_layers, fusion_dim), nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(fusion_dim, num_classes))

    def forward(self, x):
        direction = x[:, :self.direction_dim]
        norms = x[:, self.direction_dim:]
        return self.classifier(self.direction_proj(direction) * self.norm_proj(norms))


def extract_features(texts, model_name, device, batch_size, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModelForSequenceClassification.from_pretrained(
        model_name, output_hidden_states=True).to(device).eval()
    hidden_dim = encoder.config.hidden_size
    features = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc='Extracting CLS features'):
            inputs = tokenizer(texts[start:start + batch_size], return_tensors='pt',
                               padding=True, truncation=True, max_length=max_length)
            outputs = encoder(**{k: v.to(device) for k, v in inputs.items()})
            cls = torch.stack([outputs.hidden_states[layer][:, 0, :] for layer in LAYERS], dim=1)
            norms = torch.linalg.vector_norm(cls, dim=2)
            directions = cls / (norms.unsqueeze(-1) + 1e-8)
            features.append(torch.cat([directions.flatten(start_dim=1), norms], dim=1).cpu())
    del encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return torch.cat(features, dim=0), hidden_dim


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += len(y)
    return loss_sum / len(loader), correct / count


def save_plots(history, cm, names, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history['train_accuracy'], label='Train accuracy')
    axes[0].plot(history['val_accuracy'], label='Validation accuracy')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[1].plot(history['train_loss'], label='Train loss')
    axes[1].plot(history['val_loss'], label='Validation loss')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / 'training_curves.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(names)), names, rotation=30, ha='right')
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='Toxicity_Data_Shuffled1.csv')
    parser.add_argument('--model-name', default='Xuhui/ToxDect-roberta-large')
    parser.add_argument('--output-dir', default='results/garf_multilayer_toxicity')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--extract-batch-size', type=int, default=32)
    parser.add_argument('--train-batch-size', type=int, default=64)
    parser.add_argument('--max-length', type=int, default=128)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print('Using device:', device)

    df = pd.read_csv(args.data)
    if not {'text', 'label'}.issubset(df.columns):
        raise ValueError('Input CSV must contain text and label columns')
    if df[['text', 'label']].isna().any().any():
        raise ValueError('Input CSV contains missing text/label values; check original dataset')
    texts = [preprocess_toxicity_text(t) for t in df['text'].tolist()]
    le = LabelEncoder()
    labels = le.fit_transform(df['label'].tolist())
    if len(le.classes_) != 3:
        raise ValueError(f'Expected three classes, found {list(le.classes_)}')
    print('Rows:', len(df), '| Label mapping:', dict(enumerate(le.classes_.tolist())))

    # Reconstruct the same stratified 80/10%-of-remainder split as the notebook.
    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2,
                                               random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1,
                                         random_state=42, stratify=labels[train_val_idx])
    np.savez(output / 'split_indices.npz', train_idx=train_idx,
             val_idx=val_idx, test_idx=test_idx)
    pd.DataFrame({
        'sample_id': [f'toxicity_row_{i:06d}' for i in test_idx],
        'original_row_index': test_idx,
        'raw_text': df['text'].iloc[test_idx].to_numpy(),
        'text': np.asarray(texts)[test_idx],
        'label': labels[test_idx],
        'label_name': df['label'].iloc[test_idx].to_numpy(),
    }).to_csv(output / 'held_out_test_samples.csv', index=False)

    x, hidden_dim = extract_features(texts, args.model_name, device,
                                      args.extract_batch_size, args.max_length)
    y = torch.tensor(labels, dtype=torch.long)
    def loader(idx, shuffle=False):
        return DataLoader(TensorDataset(x[idx], y[idx]), batch_size=args.train_batch_size,
                          shuffle=shuffle)
    train_loader, val_loader, test_loader = (loader(train_idx, True), loader(val_idx), loader(test_idx))
    print('Train / validation / test:', len(train_idx), len(val_idx), len(test_idx))
    print('Feature shape:', tuple(x.shape))

    head = ExplicitNormAwareClassifier(cls_dim=hidden_dim, num_classes=len(le.classes_)).to(device)
    weights = compute_class_weight('balanced', classes=np.arange(len(le.classes_)), y=labels[train_idx])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(head.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    for epoch in range(args.epochs):
        tl, ta = run_epoch(head, train_loader, criterion, device, optimizer)
        vl, va = run_epoch(head, val_loader, criterion, device)
        scheduler.step(vl)
        for key, val in zip(history, (tl, vl, ta, va)):
            history[key].append(val)
        print(f'Epoch {epoch+1:02d}/{args.epochs}: train_loss={tl:.4f}, train_acc={ta:.4f}, '
              f'val_loss={vl:.4f}, val_acc={va:.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}', flush=True)

    head.eval()
    preds, probs = [], []
    with torch.inference_mode():
        for xb, _ in test_loader:
            logits = head(xb.to(device))
            probabilities = logits.softmax(dim=1).cpu()
            preds.extend(probabilities.argmax(dim=1).tolist())
            probs.extend(probabilities.tolist())
    truth = labels[test_idx]
    report = classification_report(truth, preds, labels=np.arange(len(le.classes_)),
                                   target_names=le.classes_.tolist(), output_dict=True, zero_division=0)
    cm = confusion_matrix(truth, preds, labels=np.arange(len(le.classes_)))
    print('\n', classification_report(truth, preds, labels=np.arange(len(le.classes_)),
                                      target_names=le.classes_.tolist(), zero_division=0))
    print('Confusion matrix:\n', cm)
    pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(output / 'confusion_matrix.csv')
    pd.DataFrame({'true_label': truth, 'predicted_label': preds,
                  **{f'prob_{name}': np.asarray(probs)[:, i] for i, name in enumerate(le.classes_)}}
                 ).to_csv(output / 'test_predictions.csv', index=False)
    pd.DataFrame(history).to_csv(output / 'training_history.csv', index=False)
    with open(output / 'classification_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    with open(output / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump({**vars(args), 'layers': list(LAYERS), 'class_names': le.classes_.tolist(),
                   'hidden_size': hidden_dim}, f, indent=2)
    torch.save({'model_state_dict': head.cpu().state_dict(), 'class_names': le.classes_.tolist(),
                'hidden_size': hidden_dim, 'layers': LAYERS, 'model_name': args.model_name},
               output / 'garf_classifier.pt')
    save_plots(history, cm, le.classes_, output)
    print(f'\nAccuracy: {accuracy_score(truth, preds):.4f} | Macro-F1: {f1_score(truth, preds, average="macro"):.4f}')
    print('Saved results to:', output.resolve())


if __name__ == '__main__':
    main()
