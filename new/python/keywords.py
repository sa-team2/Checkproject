import firebase_admin
from firebase_admin import  firestore
from transformers import BertTokenizer, BertModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re



app = firebase_admin.get_app('app')  # 根據名稱獲取已初始化的實例
db = firestore.client(app)

DEFAULT_TYPE = "未知"  # 默認類型

# 載入詐騙類型的模型2和嵌入
checkpoint = torch.load('./model/bert_fraud_model.pth')
model_type = BertModel.from_pretrained('bert-base-chinese')
model_type.load_state_dict(checkpoint['model_state_dict'])
model_type.eval()
fraud_type_embeddings = checkpoint['fraud_type_embeddings']


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', clean_up_tokenization_spaces=False)

def get_bert_embedding(text): 
    """通過 BERT 獲取文本的嵌入向量"""
    # Tokenize the text with padding, truncation, and max length of 128 tokens
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128, add_special_tokens=True)
    outputs = model_type(**inputs)
    
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def predict_fraud_types(keywords):
    """使用 BERT 模型預測關鍵字的類型"""
    global fraud_type_embeddings  # 已有的詐騙類型嵌入

    predicted_types = []
    for keyword in keywords:
        keyword_embedding = get_bert_embedding(keyword)

        # 計算與所有詐騙類型的相似度
        similarities = {
            fraud_type: np.dot(embedding.flatten(), keyword_embedding.flatten())
            for fraud_type, embedding in fraud_type_embeddings.items()
        }

        # 按相似度排序，選出最相似的類型
        sorted_types = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        if sorted_types and sorted_types[0][1] > 0.5:  # 閾值為 0.5
            predicted_types.append((keyword, sorted_types[0][0]))
        else:
            predicted_types.append((keyword, DEFAULT_TYPE))

    return predicted_types




import google.generativeai as genai

# 初始化 Gemini（確保你已經設置好 API Key）
genai.configure(api_key="")  
model_gemini = genai.GenerativeModel("models/gemini-1.5-pro-latest")

def find_suspicious_keywords(text):
    """使用 Gemini 抓出1~2個可疑關鍵字（不需分類）"""
    prompt = f"""
你是一個專業的語意分析、情緒分類、對於詐騙很敏銳的 AI，請分析以下訊息是否為詐騙，但要避免將「防詐騙提醒」誤判為詐騙。

### 訊息如下：
{text}

### 任務：
1. 不論是否為詐騙，請找出1-2個**詐騙關鍵字或關鍵句**。
2. 使用格式：suspicious_keywords=關鍵字1, 關鍵字2
"""

    try:
        response = model_gemini.generate_content(prompt)
        content = response.text

        match = re.search(r"suspicious_keywords\s*=\s*(.+)", content)
        if match:
            keywords_str = match.group(1)
            suspicious_keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
        else:
            suspicious_keywords = []
    except Exception as e:
        print(f"Gemini 分析出錯: {e}")
        suspicious_keywords = []

    return suspicious_keywords






def get_and_match_keywords_with_details(text):
    """
    從 Gemini 分析原文取得關鍵字 → 與 Firebase 資料庫比對 → 回傳匹配或 BERT 預測的關鍵字與類型
    並取得對應 Remind 和 Prevent 資訊
    """
    result = []
    keywords_data = []

    # Step 1: 讀取 Firebase FraudDefine 資料表中的所有關鍵字與類型
    try:
        keywords_ref = db.collection('FraudDefine')
        docs = keywords_ref.stream()
        for doc in docs:
            keyword_info = doc.to_dict()
            if 'Keyword' in keyword_info and 'Type' in keyword_info:
                keywords_data.append({
                    'keyword': keyword_info['Keyword'],
                    'type': keyword_info['Type']
                })
    except Exception as e:
        print(f"讀取 FraudDefine 錯誤: {e}")
        return result

    # Step 2: 透過 Gemini 從原文中提取詐騙關鍵字
    suspicious_keywords = []
    try:
        suspicious_keywords = find_suspicious_keywords(text)
    except Exception as e:
        print(f"從 Gemini 抓取關鍵字失敗: {e}")
        return result

    if not suspicious_keywords:
        return result

    # Step 3: 將抓出的關鍵字與資料庫關鍵字比對
    matched = []

    for keyword in suspicious_keywords:
        match = next((kd for kd in keywords_data if keyword in kd['keyword'] or kd['keyword'] in keyword), None)
        if match:
            matched.append({
                'keyword': keyword,
                'type': match['type']
            })
        else:
            # 這裡直接使用 BERT 預測這個關鍵字的類型
            predicted_type = predict_fraud_types([keyword])[0][1]
            matched.append({
                'keyword': keyword,
                'type': predicted_type if predicted_type else DEFAULT_TYPE
            })

    # Step 4: 加上 Remind 和 Prevent 資訊（根據 type）
    for item in matched:
        try:
            snapshot = db.collection('Statistics').where('Type', '==', item['type']).stream()
            doc = next(snapshot, None)
            if doc:
                data = doc.to_dict()
                item['Remind'] = data.get('Remind', '')
                item['Prevent'] = data.get('Prevent', '')
            else:
                item['Remind'] = ''
                item['Prevent'] = ''
        except Exception as e:
            print(f"查詢 Statistics 集合時出錯: {e}")
            item['Remind'] = ''
            item['Prevent'] = ''

    return matched
