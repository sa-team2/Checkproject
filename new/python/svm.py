import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from flask import Flask, request, jsonify

# 初始化 Flask 应用
app = Flask(__name__)

# 初始化 Firebase Admin
cred = credentials.Certificate('../config/dayofftest1-firebase-adminsdk-xfpl4-23ed2646dd.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# 初始化 BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')

#獲得資料表
def get_data_from_firestore():
    collection_ref = db.collection('FraudDefine')
    docs = collection_ref.stream()
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    return data

def encode_with_bert(data):
    """
    使用 BERT 对关键词列表进行编码，返回嵌入向量。
    """
    inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():  # 不需要反向传播
        outputs = bert_model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)  # 平均池化
    return embeddings.cpu().numpy()

def update_model():
    # 从 Firestore 获取数据
    firestore_data = get_data_from_firestore()
    df = pd.DataFrame(firestore_data)
    df.dropna(subset=['Keyword'], inplace=True)

    data = df['Keyword'].tolist()

    # 计算 TF-IDF 特征
    global vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, max_df=1.0, min_df=1)
    X_tfidf = vectorizer.fit_transform(data).toarray()  # 生成 TF-IDF 特征矩阵

    # 计算 BERT 特征
    X_bert = encode_with_bert(data)  # 使用 BERT 生成文本嵌入向量

    print("TF-IDF shape:", X_tfidf.shape)
    print("BERT shape:", X_bert.shape)
    # 结合 TF-IDF 和 BERT 特征
    X = np.hstack((X_tfidf, X_bert))  # 将 TF-IDF 和 BERT 特征水平拼接
    print("Combined shape:", X.shape)  # 打印确认

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  
    print("Shape after scaling:", X_scaled.shape)

    # PCA 降维
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    print("Shape of X after PCA:", X_pca.shape)  # 这将告诉您 PCA 后的形状

    # KMeans 聚类
    silhouette_scores = []
    for n_clusters in range(2, 10):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, clusters)
        silhouette_scores.append(score)

    best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    # One-Class SVM 模型训练
    global ocsvm_models
    ocsvm_models = []
    for cluster in set(clusters):
        X_cluster = X_pca[clusters == cluster]
        ocsvm = OneClassSVM(kernel='rbf', nu=0.4, gamma='scale')
        ocsvm.fit(X_cluster)
        ocsvm_models.append(ocsvm)

    # 保存模型
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca.pkl')
    for i, model in enumerate(ocsvm_models):
        joblib.dump(model, f'ocsvm_model_{i}.pkl')



#以下是判斷類型的模型
    # -------------------------------------------------------------

def update_model2():
    # 加载 BERT Tokenizer 和 Model
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', clean_up_tokenization_spaces=False)
    model_type = BertModel.from_pretrained('bert-base-chinese')

    # 从 Firestore 获取诈骗类型及描述
    def fraud_type_descriptions():
        descriptions = {}
        docs = db.collection('Statistics').stream()
        for doc in docs:
            data = doc.to_dict()
            descriptions[data['Type']] = data['Define']
        return descriptions

    # 获取文本的 BERT 嵌入
    def get_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model_type(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    # 获取所有诈骗类型描述并计算其嵌入
    fraud_type_embeddings = {key: get_embedding(desc) for key, desc in fraud_type_descriptions().items()}

    # 保存模型状态和嵌入
    torch.save({
        'model_state_dict': model_type.state_dict(),
        'fraud_type_embeddings': fraud_type_embeddings
    }, 'bert_fraud_model.pth')

if __name__ == "__main__":
    update_model()
    update_model2()
