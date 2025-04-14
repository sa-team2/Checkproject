import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import firebase_admin
from firebase_admin import credentials, firestore
from imblearn.over_sampling import SMOTE
import numpy as np

# ========== 1. 初始化 Firebase ========== 
cred = credentials.Certificate("../config/dayofftest1-firebase-adminsdk-xfpl4-f64d9dc336.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


model_save_path = "bert"  # 設定儲存的資料夾名稱

# 清空資料夾內容（但保留資料夾本身）
import os
import shutil
if os.path.exists(model_save_path):
    for filename in os.listdir(model_save_path):
        file_path = os.path.join(model_save_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"成功刪除檔案：{file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"成功刪除資料夾：{file_path}")
        except Exception as e:
            print(f"❌ 無法刪除 {file_path}：{e}")
else:
    os.makedirs(model_save_path)
    print(f"📁 資料夾不存在，已建立：{model_save_path}")





# ========== 2. 從 Firestore 讀取詐騙數據 ========== 
def get_data_from_firestore():
    collection_ref = db.collection("FraudDefine")
    docs = collection_ref.stream()
    data, labels = [], []
    for doc in docs:
        doc_data = doc.to_dict()
        data.append(doc_data["Keyword"])
        result = doc_data.get('Result', 0)  # 默認為 0，若缺失結果
        labels.append(result)  # 這裡需要將 Result 添加到 labels 中
    print(f"獲取 {len(data)} 條詐騙數據")
    return data, labels

texts, labels = get_data_from_firestore()

# ========== 3. 文本編碼（使用 BERT Tokenizer） ========== 
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

def encode_texts(texts, tokenizer, max_length=32):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

encoded_texts = encode_texts(texts, tokenizer)

# ========== 4. 使用 SMOTE 進行過採樣 ========== 
# 將文本編碼的 "input_ids" 轉換為 NumPy 陣列進行 SMOTE
input_ids_np = encoded_texts['input_ids'].numpy()

# 使用 SMOTE 進行過採樣處理
smote = SMOTE(sampling_strategy=1, random_state=42)
input_ids_resampled, labels_resampled = smote.fit_resample(input_ids_np, np.array(labels))

# ========== 5. 更新 Attention Mask ========== 
# 重新創建一個與過採樣後 input_ids 相對應的 attention_mask
attention_mask_resampled = np.zeros_like(input_ids_resampled)  # 先創建空的 attention_mask
for i in range(len(input_ids_resampled)):
    # 為每個過採樣的樣本生成 attention_mask，根據原始的 input_ids 的 padding 填充情況
    attention_mask_resampled[i] = encoded_texts['attention_mask'][i % len(encoded_texts['attention_mask'])].numpy()

# 轉回 PyTorch 張量格式
input_ids_resampled = torch.tensor(input_ids_resampled)
attention_mask_resampled = torch.tensor(attention_mask_resampled)

# 更新 DataLoader
class FraudDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

# 創建過採樣後的數據集和數據加載器
# 更新的部分：確保 labels 轉換為 LongTensor 類型
dataset_resampled = FraudDataset(input_ids_resampled, attention_mask_resampled, torch.tensor(labels_resampled, dtype=torch.long))
dataloader_resampled = DataLoader(dataset_resampled, batch_size=8, shuffle=True)

# ========== 6. 加載 BERT 模型 ========== 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2).to(device)

# ========== 7. 設定優化器與損失函數 ========== 
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss()

# ========== 8. 訓練模型 ========== 
def train(model, dataloader, optimizer, loss_fn, epochs=2):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # 打印模型的輸出和損失
            # print(f"Logits: {logits.shape}, Labels: {labels.shape}")
            
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

train(model, dataloader_resampled, optimizer, loss_fn)

# ==========  儲存模型與 tokenizer ========== 

# 儲存模型
model.save_pretrained(model_save_path)

# 儲存 tokenizer
tokenizer.save_pretrained(model_save_path)

print(f"模型和 tokenizer 已保存到 {model_save_path} 資料夾。")


# ========== 9. 詐騙檢測函數 ========== 
def predict(text, model, tokenizer):
    model.eval()
    encoded = encode_texts([text], tokenizer)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    
    fraud_prob = probs[0][1].item() * 100  # 詐騙的機率
    return fraud_prob

# ========== 10. 測試詐騙辨識 ========== 
test_texts = [
    "這是穩賺不賠的投資機會，快來加入！",
    "您的信用卡有異常交易，請點擊以下連結確認。",
    "今天的天氣很好，適合出去散步。",
    "您有未支付的帳單，請立即支付以免影響信用。",
    "我們的網站正在進行維護，請稍後再試。",
    "您的訂單已經成功處理，請耐心等待配送。",
    "咕嚕",
    "您的賬戶正在被異常登錄，請立即修改密碼。",
    "要不要出去玩",
    "感謝您的訂購，我們已經開始處理您的訂單。",
    "你好球球",
    "4486",
    "今天上課的老師出的作業很多，機車，作業很多不想寫",
    "麥當勞公司是一家美國跨國速食連鎖企業，於1940年由理查和莫里斯·麥當勞兄弟在美國加利福尼亞州聖貝納迪諾市成立，當時是一家餐廳。他們將自己的業務重新命名為漢堡攤，後來將公司轉變為加盟連鎖企業，並於1953年在亞利桑那州鳳凰城的一個地點推出了金色拱門標誌。1955年，商人雷·克洛克作為加盟連鎖代理加入公司，並於1961年買下麥當勞兄弟的全部股份。總部先前位於伊利諾州奧克布魯克，於2018年6月遷至芝加哥。麥當勞也是一家房地產公司，擁有約70%的餐廳建築等（出租給其加盟連鎖商）。"
]

for text in test_texts:
    prob = predict(text, model, tokenizer)
    print(f"【{text}】詐騙可能性: {prob:.2f}%")
