import firebase_admin
from firebase_admin import credentials, firestore
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset
import os

def remove_existing_model_files():
    model_dir = "./model"  # 指定模型文件夹路径
    model_path = os.path.join(model_dir, "bert_role_classifier.pth")
    
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"✅ 已删除旧模型文件: {model_path}")
    else:
        print(f"ℹ️ 未找到旧模型文件，无需删除。")

remove_existing_model_files()

# 初始化 Firebase
cred = credentials.Certificate('../config/dayofftest1-firebase-adminsdk-xfpl4-f64d9dc336.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# 讀取 Firebase `Dialogue` 資料
def load_dialogue_data():
    docs = db.collection("Dialogue").stream()
    conversations = []

    for doc in docs:
        data = doc.to_dict()
        for message in data.get("messages", []):
            role = int(message["role"])  # ✅ 将布尔值转换为整数
            conversations.append((message["text"], role))

    return conversations


data = load_dialogue_data()


texts, labels = zip(*[(text, role) for text, role in data])


# 切分訓練集與測試集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 下載 BERT 分詞器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


class RoleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

train_dataset = RoleDataset(train_texts, train_labels, tokenizer)
test_dataset = RoleDataset(test_texts, test_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn.init as init

class BertForRoleClassification(nn.Module):
    def __init__(self, num_labels):
        super(BertForRoleClassification, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=num_labels)
        
        # 手动初始化分类层权重
        init.xavier_uniform_(self.bert.classifier.weight)
        init.zeros_(self.bert.classifier.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


model = BertForRoleClassification(num_labels=2).to(device)



optimizer = optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

def train_model(model, train_loader, optimizer, loss_fn, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

train_model(model, train_loader, optimizer, loss_fn)

from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import accuracy_score, classification_report
import torch

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []  # ✅ 正確初始化
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 確保 test_loader 內有數據
    if not predictions:
        print("⚠️ 測試數據為空，請檢查 `test_loader` 是否正確加載！")
        return

    print("Accuracy:", accuracy_score(true_labels, predictions))
    print("Classification Report:\n", classification_report(
        true_labels, predictions, target_names=["scammer", "victim"], labels=[0, 1]
    ))

evaluate_model(model, test_loader)



torch.save(model.state_dict(), "model/bert_role_classifier.pth")
print("模型已儲存！")



