import firebase_admin
from firebase_admin import credentials, firestore
from transformers import BertTokenizer, BertModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



app = firebase_admin.get_app('app')  # 根據名稱獲取已初始化的實例
db = firestore.client(app)

DEFAULT_TYPE = "未知"  # 默认类型

# 加载詐騙類型的模型2和嵌入
checkpoint = torch.load('bert_fraud_model.pth')
model_type = BertModel.from_pretrained('bert-base-chinese')
model_type.load_state_dict(checkpoint['model_state_dict'])
model_type.eval()
fraud_type_embeddings = checkpoint['fraud_type_embeddings']


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', clean_up_tokenization_spaces=False)

def get_bert_embedding(text): 
    """通过 BERT 获取文本的嵌入向量"""
    # Tokenize the text with padding, truncation, and max length of 128 tokens
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128, add_special_tokens=True)
    outputs = model_type(**inputs)
    
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def predict_fraud_types(keywords):
    """使用 BERT 模型预测关键词的类型"""
    global fraud_type_embeddings  # 已有的诈骗类型嵌入

    predicted_types = []
    for keyword in keywords:
        keyword_embedding = get_bert_embedding(keyword)

        # 计算与所有诈骗类型的相似度
        similarities = {
            fraud_type: np.dot(embedding.flatten(), keyword_embedding.flatten())
            for fraud_type, embedding in fraud_type_embeddings.items()
        }

        # 按相似度排序，选出最相似的类型
        sorted_types = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        if sorted_types and sorted_types[0][1] > 0.5:  # 阈值为 0.5
            predicted_types.append((keyword, sorted_types[0][0]))
        else:
            predicted_types.append((keyword, DEFAULT_TYPE))

    return predicted_types



def find_suspicious_keywords(text, keywords_data):
    """使用 TF-IDF 提取最可疑关键词"""
    
    # 这里我们不拆分为单个词，而是直接将文本传给 TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])  # 直接使用原始文本
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    # 选取 TF-IDF 分数最高的两个关键词
    top_indices = tfidf_scores.argsort()[-2:][::-1]
    suspicious_keywords = [feature_names[i] for i in top_indices]

    # 检查关键词是否已在数据库中
    classified_keywords = [
        (kw, kd['type']) for kw in suspicious_keywords
        for kd in keywords_data if kw == kd['keyword']
    ]

    # 若不在数据库中，进行预测
    if not classified_keywords:
        predicted_types = predict_fraud_types(suspicious_keywords)
        classified_keywords = [(kw, pt if pt else DEFAULT_TYPE) for kw, pt in predicted_types]

    return classified_keywords




def get_and_match_keywords_with_details(text):
    """从 Firebase 获取关键字和类型，匹配或预测文本中的关键字及其类型，并获取 Remind 和 Prevent 信息"""
    matched = []
    keywords_data = []

    # 从 Firebase 获取关键字和类型
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
        print(f"Failed to fetch keywords from Firebase: {e}")
        return matched

    # 匹配关键字
    matched = [
        {'keyword': kd['keyword'], 'type': kd['type']}
        for kd in keywords_data if kd['keyword'] in text
    ]

    # 如果没有匹配到关键字，则进行预测
    if not matched:
        suspicious_keywords = find_suspicious_keywords(text, keywords_data)
        predicted_types = predict_fraud_types([kw for kw, _ in suspicious_keywords])

        matched.extend({
            'keyword': kw,
            'type': pt if pt else DEFAULT_TYPE
        } for kw, pt in predicted_types)

    # 获取每个匹配或预测的关键字类型的 Remind 和 Prevent 信息
    for item in matched:
        try:
            snapshot = db.collection('Statistics').where('Type', '==', item['type']).stream()
            doc = next(snapshot, None)  # 获取第一个匹配文档，若没有则返回 None

            if doc:
                data = doc.to_dict()
                item['Remind'] = data.get('Remind', '')
                item['Prevent'] = data.get('Prevent', '')
            else:
                item['Remind'] = ''
                item['Prevent'] = ''
        except Exception as e:
            print(f"查询 Statistics 集合时出错: {e}")
            item['Remind'] = ''
            item['Prevent'] = ''

    return matched

