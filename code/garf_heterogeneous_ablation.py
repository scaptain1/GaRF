"""GaRF heterogeneous-dataset ablation study (A1–A10, five seeds).

"""
import argparse
from pathlib import Path

_parser = argparse.ArgumentParser(description="GaRF heterogeneous ablation experiments")
_parser.add_argument("--data", default="Toxicity_Data_Shuffled1.csv", help="CSV with text and label columns")
_parser.add_argument("--output-dir", default="results/heterogeneous_ablation", help="Output directory")
_args = _parser.parse_args()
OUTPUT_DIR = Path(_args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)




import random
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


SEEDS = [42, 123, 2024, 3407, 9999]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device :", device)

if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))




MODEL_NAME = "Xuhui/ToxDect-roberta-large"

SELECTED_LAYERS = [8, 12, 16, -1, -2, -3, -4, -5]

MAX_LENGTH = 128

EXTRACT_BATCH = 32
TRAIN_BATCH = 64

EPOCHS = 10
LR = 2e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

NUM_CLASSES = 3


print("\nModel          :", MODEL_NAME)
print("Layers         :", SELECTED_LAYERS)
print("Max length     :", MAX_LENGTH)
print("Extract batch  :", EXTRACT_BATCH)
print("Train batch    :", TRAIN_BATCH)
print("Epochs         :", EPOCHS)
print("Learning rate  :", LR)
print("Weight decay   :", WEIGHT_DECAY)
print("Dropout        :", DROPOUT)
print("Seeds          :", SEEDS)



import re
import html


def preprocess_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'https?://\S+|www\.\S+', 'URL', text)
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r'\s+', ' ', text).strip()
    return text



df = pd.read_csv(_args.data)


texts = [
    preprocess_text(text)
    for text in df["text"].tolist()
]


labels = df["label"].tolist()



print("Dataset shape :", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nTotal samples :", len(texts))



from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(labels)

print("Label mapping:")
for i, name in enumerate(label_encoder.classes_):
    print(f"{i} = {name}")

print("\ny shape:", y.shape)
print("Class counts:", np.bincount(y))



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True
).to(device)


model.eval()

for param in model.parameters():
    param.requires_grad = False



print("Model       :", MODEL_NAME)
print("Hidden size :", model.config.hidden_size)
print("Layers      :", model.config.num_hidden_layers)
print("Parameters  :", f"{sum(p.numel() for p in model.parameters()):,}")
print("Trainable   :", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print("Frozen      :", not any(p.requires_grad for p in model.parameters()))




all_raw = []

model.eval()

with torch.no_grad():

    for start in range(0, len(texts), EXTRACT_BATCH):

        batch_texts = texts[start:start + EXTRACT_BATCH]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs)

        hidden_states = outputs.hidden_states

       
        batch_layers = torch.stack(
            [hidden_states[i][:, 0, :] for i in SELECTED_LAYERS],
            dim=1
        )

        all_raw.append(batch_layers.cpu())

        if start % (EXTRACT_BATCH * 50) == 0:
            print(f"Processed {start}/{len(texts)}")



X_raw_layers = torch.cat(all_raw, dim=0)


print("\nRaw layer shape:", X_raw_layers.shape)



X_magnitude = torch.linalg.vector_norm(
    X_raw_layers,
    ord=2,
    dim=2
)


X_direction = X_raw_layers / (
    X_magnitude.unsqueeze(-1) + 1e-12
)


print("Raw       :", X_raw_layers.shape)
print("Direction :", X_direction.shape)
print("Magnitude :", X_magnitude.shape)




direction_norms = torch.linalg.vector_norm(
    X_direction,
    ord=2,
    dim=2
)

print("\nDirection norm mean :", direction_norms.mean().item())
print("Direction norm min  :", direction_norms.min().item())
print("Direction norm max  :", direction_norms.max().item())

assert X_raw_layers.shape == (12388, 8, 1024)
assert X_direction.shape == (12388, 8, 1024)
assert X_magnitude.shape == (12388, 8)

print("\n✓ Feature extraction complete.")




indices = np.arange(len(y))


trainval_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    stratify=y,
    random_state=42
)


train_idx, val_idx = train_test_split(
    trainval_idx,
    test_size=0.10,
    stratify=y[trainval_idx],
    random_state=42
)



print("Total :", len(indices))
print("Train :", len(train_idx))
print("Val   :", len(val_idx))
print("Test  :", len(test_idx))

print("\nClass order:")
print("0 = Explicit")
print("1 = Implicit")
print("2 = Non-toxic")

print("\nTrain:", np.bincount(y[train_idx]))
print("Val  :", np.bincount(y[val_idx]))
print("Test :", np.bincount(y[test_idx]))




class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=y[train_idx]
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)

print("Class weights:", class_weights.cpu().numpy())



X_last_raw = X_raw_layers[:, -1, :]           # [N, 1024]

X_last_norm = X_last_raw / (
    torch.linalg.vector_norm(
        X_last_raw, dim=1, keepdim=True
    ) + 1e-12
)

X_last_mag = torch.linalg.vector_norm(
    X_last_raw,
    dim=1,
    keepdim=True
)



X_multi_raw = X_raw_layers.reshape(
    len(y), -1
)                                               # [N, 8192]


X_multi_dir = X_direction.reshape(
    len(y), -1
)                                               # [N, 8192]


X_multi_mag = X_magnitude                       # [N, 8]



print("A1/A2 Last raw       :", X_last_raw.shape)
print("A3    Last normalized:", X_last_norm.shape)
print("A4    Last magnitude :", X_last_mag.shape)

print("A5    Multi raw      :", X_multi_raw.shape)
print("A6    Multi direction:", X_multi_dir.shape)
print("A7    Multi magnitude:", X_multi_mag.shape)

print("\nFusion inputs for A8-A10:")
print("Direction :", X_multi_dir.shape)
print("Magnitude :", X_multi_mag.shape)



class LinearHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, NUM_CLASSES)

    def forward(self, x):
        return self.fc(x)



class LastLayerMLP(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)




class LastMagnitudeMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)




class MultiLayerMLP(nn.Module):
    def __init__(self, input_dim=8192):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)




class MultiMagnitudeMLP(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(32, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)




class FusionHead(nn.Module):

    def __init__(self, fusion="multiply"):
        super().__init__()

        
        self.direction_proj = nn.Sequential(
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(512, 256),
            nn.ReLU()
        )

       
        self.magnitude_proj = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU()
        )

        self.fusion = fusion

       
        if fusion == "concat":
            classifier_dim = 512
        else:
            classifier_dim = 256

        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(classifier_dim, NUM_CLASSES)
        )

    def forward(self, direction, magnitude):

        d = self.direction_proj(direction)
        m = self.magnitude_proj(magnitude)

        if self.fusion == "concat":
            h = torch.cat([d, m], dim=1)

        elif self.fusion == "add":
            h = d + m

        elif self.fusion == "multiply":
            h = d * m

        else:
            raise ValueError("Unknown fusion method")

        return self.classifier(h)


print("✓ All ablation heads defined.")




def make_loader(X, indices, shuffle=False):
    dataset = TensorDataset(
        X[indices].float(),
        torch.tensor(y[indices], dtype=torch.long)
    )

    return DataLoader(
        dataset,
        batch_size=TRAIN_BATCH,
        shuffle=shuffle
    )




def train_single_model(model, X, seed):

    set_seed(seed)
    model = model.to(device)

    train_loader = make_loader(X, train_idx, shuffle=True)
    val_loader   = make_loader(X, val_idx, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):

     
        model.train()

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

        
        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    return model




def train_fusion_model(model, direction, magnitude, seed):

    set_seed(seed)
    model = model.to(device)

    train_dataset = TensorDataset(
        direction[train_idx].float(),
        magnitude[train_idx].float(),
        torch.tensor(y[train_idx], dtype=torch.long)
    )

    val_dataset = TensorDataset(
        direction[val_idx].float(),
        magnitude[val_idx].float(),
        torch.tensor(y[val_idx], dtype=torch.long)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_BATCH,
        shuffle=False
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):

 
        model.train()

        for d, m, yb in train_loader:

            d = d.to(device)
            m = m.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(d, m)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

       
        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            for d, m, yb in val_loader:

                d = d.to(device)
                m = m.to(device)
                yb = yb.to(device)

                logits = model(d, m)
                loss = criterion(logits, yb)

                val_loss += loss.item() * d.size(0)

        val_loss /= len(val_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    return model


print("✓ Training functions ready.")




def compute_metrics(y_true, y_pred, y_prob):

    

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = \
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

    _, _, weighted_f1, _ = \
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )


  

    y_true_bin = label_binarize(
        y_true,
        classes=[0, 1, 2]
    )

    macro_auc = roc_auc_score(
        y_true_bin,
        y_prob,
        average="macro",
        multi_class="ovr"
    )


    

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2]
    )


   

    implicit_total = cm[1].sum()
    implicit_recall = (
        cm[1, 1] / implicit_total
        if implicit_total > 0 else 0
    )


   

    mcr = (
        cm[1, 2] / implicit_total
        if implicit_total > 0 else 0
    )




    benign_total = cm[2].sum()

    benign_fpr = (
        (cm[2, 0] + cm[2, 1]) / benign_total
        if benign_total > 0 else 0
    )


    return {
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Macro ROC-AUC": macro_auc,
        "Implicit Recall": implicit_recall,
        "MCR": mcr,
        "Benign FPR": benign_fpr
    }




def evaluate_single_model(model, X):

    test_loader = make_loader(
        X,
        test_idx,
        shuffle=False
    )

    model.eval()

    all_pred = []
    all_prob = []
    all_true = []

    with torch.no_grad():

        for xb, yb in test_loader:

            xb = xb.to(device)

            logits = model(xb)

            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)

            all_pred.append(pred.cpu())
            all_prob.append(prob.cpu())
            all_true.append(yb)

    y_pred = torch.cat(all_pred).numpy()
    y_prob = torch.cat(all_prob).numpy()
    y_true = torch.cat(all_true).numpy()

    return compute_metrics(
        y_true,
        y_pred,
        y_prob
    )




def evaluate_fusion_model(model, direction, magnitude):

    test_dataset = TensorDataset(
        direction[test_idx].float(),
        magnitude[test_idx].float(),
        torch.tensor(
            y[test_idx],
            dtype=torch.long
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_BATCH,
        shuffle=False
    )

    model.eval()

    all_pred = []
    all_prob = []
    all_true = []

    with torch.no_grad():

        for d, m, yb in test_loader:

            d = d.to(device)
            m = m.to(device)

            logits = model(d, m)

            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)

            all_pred.append(pred.cpu())
            all_prob.append(prob.cpu())
            all_true.append(yb)

    y_pred = torch.cat(all_pred).numpy()
    y_prob = torch.cat(all_prob).numpy()
    y_true = torch.cat(all_true).numpy()

    return compute_metrics(
        y_true,
        y_pred,
        y_prob
    )


print("✓ Evaluation functions ready.")




all_results = []


def save_result(ablation, seed, metrics):

    row = {
        "Ablation": ablation,
        "Seed": seed
    }

    row.update(metrics)
    all_results.append(row)




for seed in SEEDS:

    print("\n" + "=" * 70)
    print(f"SEED {seed}")
    print("=" * 70)


  

    print("A1 — Raw last-layer CLS + Linear")

    set_seed(seed)
    model = LinearHead(1024)

    model = train_single_model(
        model,
        X_last_raw,
        seed
    )

    metrics = evaluate_single_model(
        model,
        X_last_raw
    )

    save_result("A1 Raw last CLS + Linear", seed, metrics)

    del model
    torch.cuda.empty_cache()


  

    print("A2 — Raw last-layer CLS + MLP")

    set_seed(seed)
    model = LastLayerMLP(1024)

    model = train_single_model(
        model,
        X_last_raw,
        seed
    )

    metrics = evaluate_single_model(
        model,
        X_last_raw
    )

    save_result("A2 Raw last CLS + MLP", seed, metrics)

    del model
    torch.cuda.empty_cache()


    

    print("A3 — Normalized last-layer CLS")

    set_seed(seed)
    model = LastLayerMLP(1024)

    model = train_single_model(
        model,
        X_last_norm,
        seed
    )

    metrics = evaluate_single_model(
        model,
        X_last_norm
    )

    save_result("A3 Normalized last CLS", seed, metrics)

    del model
    torch.cuda.empty_cache()


  

    print("A4 — Last-layer magnitude")

    set_seed(seed)
    model = LastMagnitudeMLP()

    model = train_single_model(
        model,
        X_last_mag,
        seed
    )

    metrics = evaluate_single_model(
        model,
        X_last_mag
    )

    save_result("A4 Last magnitude", seed, metrics)

    del model
    torch.cuda.empty_cache()


    

    print("A5 — Raw multi-layer CLS")

    set_seed(seed)
    model = MultiLayerMLP(8192)

    model = train_single_model(
        model,
        X_multi_raw,
        seed
    )

    metrics = evaluate_single_model(
        model,
        X_multi_raw
    )

    save_result("A5 Raw multi-layer CLS", seed, metrics)

    del model
    torch.cuda.empty_cache()


    

    print("A6 — Multi-layer direction")

    set_seed(seed)
    model = MultiLayerMLP(8192)

    model = train_single_model(
        model,
        X_multi_dir,
        seed
    )

    metrics = evaluate_single_model(
        model,
        X_multi_dir
    )

    save_result("A6 Multi-layer direction", seed, metrics)

    del model
    torch.cuda.empty_cache()




    print("A7 — Multi-layer magnitude")

    set_seed(seed)
    model = MultiMagnitudeMLP(8)

    model = train_single_model(
        model,
        X_multi_mag,
        seed
    )

    metrics = evaluate_single_model(
        model,
        X_multi_mag
    )

    save_result("A7 Multi-layer magnitude", seed, metrics)

    del model
    torch.cuda.empty_cache()


    

    print("A8 — Direction + Magnitude (Concat)")

    set_seed(seed)
    model = FusionHead(fusion="concat")

    model = train_fusion_model(
        model,
        X_multi_dir,
        X_multi_mag,
        seed
    )

    metrics = evaluate_fusion_model(
        model,
        X_multi_dir,
        X_multi_mag
    )

    save_result("A8 Concat", seed, metrics)

    del model
    torch.cuda.empty_cache()


    

    print("A9 — Direction + Magnitude (Add)")

    set_seed(seed)
    model = FusionHead(fusion="add")

    model = train_fusion_model(
        model,
        X_multi_dir,
        X_multi_mag,
        seed
    )

    metrics = evaluate_fusion_model(
        model,
        X_multi_dir,
        X_multi_mag
    )

    save_result("A9 Add", seed, metrics)

    del model
    torch.cuda.empty_cache()



    print("A10 — Direction × Magnitude (GaRF)")

    set_seed(seed)
    model = FusionHead(fusion="multiply")

    model = train_fusion_model(
        model,
        X_multi_dir,
        X_multi_mag,
        seed
    )

    metrics = evaluate_fusion_model(
        model,
        X_multi_dir,
        X_multi_mag
    )

    save_result("A10 GaRF", seed, metrics)

    del model
    torch.cuda.empty_cache()


print("\n✓ ALL 10 ABLATIONS × 5 SEEDS COMPLETE")




results_df = pd.DataFrame(all_results)

display_df = results_df.copy()

metric_cols = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Macro ROC-AUC",
    "Implicit Recall",
    "MCR",
    "Benign FPR"
]

display_df[metric_cols] = (
    display_df[metric_cols] * 100
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(
    display_df.round(2).to_string(index=False)
)
results_df.to_csv(OUTPUT_DIR / "ablation_all_seeds.csv", index=False)




metric_cols = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Macro ROC-AUC",
    "Implicit Recall",
    "MCR",
    "Benign FPR"
]

results_percent = results_df.copy()
results_percent[metric_cols] *= 100



mean_df = (
    results_percent
    .groupby("Ablation", sort=False)[metric_cols]
    .mean()
)




std_df = (
    results_percent
    .groupby("Ablation", sort=False)[metric_cols]
    .std()
)




summary_df = pd.DataFrame(index=mean_df.index)

for col in metric_cols:

    summary_df[col] = [
        f"{m:.2f} ± {s:.2f}"
        for m, s in zip(
            mean_df[col],
            std_df[col]
        )
    ]


print("\n" + "=" * 150)
print("HETEROGENEOUS DATASET — 5-SEED RESULTS")
print("=" * 150)

print(
    summary_df.to_string()
)
summary_df.to_csv(OUTPUT_DIR / "ablation_mean_std.csv")





journal_metrics = [
    "Accuracy",
    "Macro F1",
    "Implicit Recall",
    "MCR",
    "Benign FPR"
]

journal_table = summary_df[journal_metrics]

print("\nHETEROGENEOUS DATASET — JOURNAL TABLE\n")

print(
    journal_table.to_string()
)
journal_table.to_csv(OUTPUT_DIR / "ablation_journal_table.csv")





X_no_projection = (
    X_direction *
    X_magnitude.unsqueeze(-1)
).reshape(len(y), -1)


print("Direction      :", X_direction.shape)
print("Magnitude      :", X_magnitude.shape)
print("No projection  :", X_no_projection.shape)




max_difference = (
    X_no_projection - X_multi_raw
).abs().max().item()

print("\nMax difference from raw multi-layer CLS:",
      max_difference)

print("\n✓ No-projection representation ready.")




class FixedProjectionGaRF(nn.Module):
    def __init__(self, seed):
        super().__init__()

       
        g = torch.Generator()
        g.manual_seed(seed)

        Wd = torch.randn(8192, 256, generator=g)
        Wd = Wd / torch.linalg.vector_norm(
            Wd, dim=0, keepdim=True
        )

        self.register_buffer("Wd_fixed", Wd)

       
        Wm = torch.randn(8, 256, generator=g)
        Wm = Wm / torch.linalg.vector_norm(
            Wm, dim=0, keepdim=True
        )

        self.register_buffer("Wm_fixed", Wm)

        
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, direction, magnitude):

       
        d = torch.relu(direction @ self.Wd_fixed)
        m = torch.relu(magnitude @ self.Wm_fixed)

       
        h = d * m

        return self.classifier(h)




test_model = FixedProjectionGaRF(seed=42).to(device)

trainable = sum(
    p.numel() for p in test_model.parameters()
    if p.requires_grad
)

fixed = sum(
    b.numel() for b in test_model.buffers()
)

print(test_model)

print("\nTrainable parameters :", trainable)
print("Fixed projection parameters :", fixed)

with torch.no_grad():
    out = test_model(
        X_multi_dir[:4].to(device),
        X_multi_mag[:4].to(device)
    )

print("Output shape :", out.shape)
print("\n✓ Fixed-projection GaRF ready.")




fixed_projection_results = []

for seed in SEEDS:

    print("\n" + "=" * 65)
    print(f"FIXED PROJECTION GARF | SEED = {seed}")
    print("=" * 65)

    set_seed(seed)

    
    model_fixed = FixedProjectionGaRF(seed=seed).to(device)

    
    model_fixed = train_fusion_model(
        model_fixed,
        X_multi_dir,
        X_multi_mag,
        seed
    )

  
    metrics = evaluate_fusion_model(
        model_fixed,
        X_multi_dir,
        X_multi_mag
    )

    metrics["Seed"] = seed
    fixed_projection_results.append(metrics)

    print(
        f"Accuracy={metrics['Accuracy']*100:.2f}% | "
        f"Macro-F1={metrics['Macro F1']*100:.2f}% | "
        f"Implicit Recall={metrics['Implicit Recall']*100:.2f}% | "
        f"MCR={metrics['MCR']*100:.2f}% | "
        f"Benign FPR={metrics['Benign FPR']*100:.2f}%"
    )




fixed_df = pd.DataFrame(fixed_projection_results)

print("\n" + "=" * 65)
print("FIXED-PROJECTION GARF — 5-SEED SUMMARY")
print("=" * 65)

for metric in [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Macro ROC-AUC",
    "Implicit Recall",
    "MCR",
    "Benign FPR"
]:
    mean = fixed_df[metric].mean() * 100
    std  = fixed_df[metric].std(ddof=1) * 100

    print(f"{metric:20s}: {mean:.2f} ± {std:.2f}")
fixed_df.to_csv(OUTPUT_DIR / "fixed_projection_all_seeds.csv", index=False)

