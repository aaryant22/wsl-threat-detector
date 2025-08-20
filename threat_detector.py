# autoencoder_pipeline.py

import os
import json
import pickle
import string
import urllib.parse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =====================================================
# 🔹 Load badwords from file
# =====================================================

def load_badwords(filepath="badwords.txt"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Badwords file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


# =====================================================
# 🔹 Feature Extraction
# =====================================================

def count_badwords(text: str, badwords_list) -> int:
    text_lower = text.lower()
    return sum(1 for w in badwords_list if w in text_lower)


def extract_features(df: pd.DataFrame, badwords_list) -> pd.DataFrame:
    def count_features(path_str, body_str):
        # both raw and decoded versions
        text_raw = path_str + body_str
        text_decoded = urllib.parse.unquote(text_raw)
        text = text_raw + text_decoded   # merged view

        features = {}
        features['single_q'] = text.count("'")
        features['double_q'] = text.count('"')
        features['dashes'] = text.count("-")
        # braces including round, square, curly
        features['braces'] = sum(text.count(c) for c in "()[]{}")
        features['spaces'] = text.count(" ")
        features['percentages'] = text.count("%")
        features['semicolons'] = text.count(";")
        features['angle_brackets'] = text.count("<") + text.count(">")
        # non-alphanumeric & not common URL chars
        safe_chars = string.ascii_letters + string.digits + "/?&=._-"
        features['special_chars'] = sum(1 for c in text if c not in safe_chars)
        features['path_length'] = len(path_str)
        features['body_length'] = len(body_str)
        return features

    feature_rows, badword_counts = [], []
    for _, row in df.iterrows():
        path_str, body_str = str(row['path']), str(row['body'])
        feats = count_features(path_str, body_str)
        feature_rows.append(feats)

        if 'badwords_count' in df.columns:
            badword_counts.append(row['badwords_count'])
        else:
            badword_counts.append(count_badwords(path_str + " " + body_str, badwords_list))

    feature_df = pd.DataFrame(feature_rows)
    feature_df['badwords_count'] = badword_counts

    preserved_cols = df[['method', 'path', 'body']].reset_index(drop=True)
    return pd.concat([preserved_cols, feature_df], axis=1)


# =====================================================
# 🔹 Autoencoder
# =====================================================

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=24, latent_dim=6):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# =====================================================
# 🔹 ThreatDetectorAE class
# =====================================================

class ThreatDetectorAE:
    def __init__(self, 
                 badwords_file="badwords.txt", 
                 model_path="model.pth", 
                 scaler_path="scaler.pkl", 
                 feature_cols_path="feature_cols.json", 
                 training_data="training_data.csv",
                 hidden_dim=24, latent_dim=6, 
                 epochs=300, batch_size=32, lr=0.001, weight_decay=1e-5):

        self.badwords_list = load_badwords(badwords_file)
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_cols_path = feature_cols_path
        self.training_data = training_data

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Try to load model, otherwise train
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_cols_path):
            print("✅ Loading existing model...")
            self._load()
        else:
            print("🚀 Training new model...")
            self._train_and_save()


    def _train_and_save(self):
        df = pd.read_csv(self.training_data)
        df = extract_features(df, self.badwords_list)

        self.feature_cols = [c for c in df.columns if c not in ["method", "path", "body", "class"]]
        X_train = df[self.feature_cols].values

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)

        input_dim = X_train.shape[1]
        dataset = TensorDataset(torch.FloatTensor(X_train))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = SimpleAutoencoder(input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for (batch,) in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss={total_loss/len(dataloader):.6f}")

        # Compute threshold
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_train).to(self.device)
            recon = self.model(X_tensor)
            losses = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
        # self.threshold = np.mean(losses) + 2*np.std(losses)
        self.threshold = np.max(losses)
        print(f"📌 Threshold set at {self.threshold:.6f}")
        with open("threshold.pkl", "wb") as f:
            pickle.dump(self.threshold, f)

        # Save everything
        torch.save(self.model.state_dict(), self.model_path)
        pickle.dump(self.scaler, open(self.scaler_path, "wb"))
        json.dump(self.feature_cols, open(self.feature_cols_path, "w"))


    def _load(self):
        # Load feature cols
        with open(self.feature_cols_path, "r") as f:
            self.feature_cols = json.load(f)
        # Load scaler
        self.scaler = pickle.load(open(self.scaler_path, "rb"))
        # Load model
        input_dim = len(self.feature_cols)
        self.model = SimpleAutoencoder(input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        with open("threshold.pkl", "rb") as f:
            self.threshold = pickle.load(f)
            print(f"📌 Threshold set at {self.threshold:.6f}")
            

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict whether logs are threats based on reconstruction loss.
        Returns df with is_threat + threat_level columns.
        """
        X = extract_features(df, self.badwords_list)
        X_features = X[self.feature_cols].values
        X_scaled = self.scaler.transform(X_features)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            recon = self.model(X_tensor)
            losses = torch.mean((X_tensor - recon) ** 2, dim=1).cpu().numpy()

        # is threat if above threshold
        is_threat = losses > self.threshold

        # threat level assignment
        bins = np.linspace(min(losses), max(losses), 5)  # 4 bins
        levels = ["low", "medium", "high", "critical"]
        threat_levels = []

        for loss, threat in zip(losses, is_threat):
            if not threat:
                threat_levels.append("none")
            else:
                bin_idx = np.digitize(loss, bins) - 1
                threat_levels.append(levels[min(bin_idx, 3)])

        df_out = X.copy()
        df_out["is_threat"] = is_threat
        df_out["reconstruction_loss"] = losses
        df_out["threat_level"] = threat_levels
        return df_out

    
if __name__ == "__main__":
    detector = ThreatDetectorAE(epochs=300)
    # predicting on new logs
    test_logs_df = pd.read_csv("testing_data.csv")
    results = detector.predict(test_logs_df)
    print(results.describe())
