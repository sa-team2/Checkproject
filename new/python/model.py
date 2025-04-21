import os
import shutil
import torch
from transformers import pipeline, BertTokenizer, BertModel
from firebase_admin import credentials, firestore
import firebase_admin

# ========== 第一步：清空 emomodel 資料夾 ==========
emomodel_dir = './emomodel'

if os.path.exists(emomodel_dir):
    for filename in os.listdir(emomodel_dir):
        file_path = os.path.join(emomodel_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print(f"已刪除: {file_path}")
        except Exception as e:
            print(f"刪除 {file_path} 時出錯: {e}")
else:
    os.makedirs(emomodel_dir)

print("已清空 emomodel 資料夾。")

# ========== 第二步：儲存 Zero-shot 模型 ==========
print("正在下載並儲存 Zero-shot 模型...")
classifier = pipeline("zero-shot-classification", model="IDEA-CCNL/Erlangshen-Roberta-110M-NLI")
classifier.model.save_pretrained(emomodel_dir)
classifier.tokenizer.save_pretrained(emomodel_dir)
print("Zero-shot 模型已成功儲存！")

# ========== 第三步：初始化 Firebase ==========
cred = credentials.Certificate('../config/dayofftest1-firebase-adminsdk-xfpl4-f64d9dc336.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# ========== 第四步：下載 BERT 模型 ==========
print("正在下載 BERT 模型...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', clean_up_tokenization_spaces=False)
model_type = BertModel.from_pretrained('bert-base-chinese')

# ========== 第五步：取得 Firestore 詐騙描述 ==========
def fraud_type_descriptions():
    descriptions = {}
    docs = db.collection('Statistics').stream()
    for doc in docs:
        data = doc.to_dict()
        fraud_type = data.get('Type')
        fraud_define = data.get('Define')
        if fraud_type and fraud_define:
            descriptions[fraud_type] = fraud_define
    return descriptions

# ========== 第六步：轉換文字為 BERT 向量 ==========
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model_type(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

print("正在計算詐騙類型嵌入...")
fraud_type_embeddings = {key: get_embedding(desc) for key, desc in fraud_type_descriptions().items()}

# ========== 第七步：儲存模型狀態與嵌入到 model 資料夾 ==========
model_save_path = './model/bert_fraud_model.pth'

# 如果檔案已存在，先刪掉
if os.path.exists(model_save_path):
    os.remove(model_save_path)
    print("已刪除舊的 bert_fraud_model.pth")

# 儲存模型與嵌入
torch.save({
    'model_state_dict': model_type.state_dict(),
    'fraud_type_embeddings': fraud_type_embeddings
}, model_save_path)

print("BERT 模型與嵌入已成功儲存到 model 資料夾！")
