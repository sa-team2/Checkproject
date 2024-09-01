import pandas as pd
from google.cloud import firestore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Union
import os

app = FastAPI()

class TextSample(BaseModel):
    sample: str

# 设置服务账户密钥文件路径
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../config/test-bc002-firebase-adminsdk-47w0c-20f1ea4f43.json"

# 初始化 Firestore 客户端
db = firestore.Client()

def fetch_data_from_firestore() -> pd.DataFrame:
    collection_ref = db.collection('sa').document('keyword').collection('data')
    docs = collection_ref.stream()
    
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    
    return pd.DataFrame(data)

def fetch_fraud_keywords() -> List[str]:
    collection_ref = db.collection('sa').document('keyword').collection('data')
    docs = collection_ref.where('是否是詐騙', '==', 1).stream()
    
    keywords = []
    for doc in docs:
        data = doc.to_dict()
        keyword = data.get('關鍵字')
        if keyword:
            keywords.append(keyword)
    
    return keywords

def save_new_fraud_keyword(keyword: str):
    collection_ref = db.collection('sa').document('keyword').collection('data')
    # 检查关键字是否已存在
    query = collection_ref.where('關鍵字', '==', keyword).stream()
    if not any(True for _ in query):
        collection_ref.add({'關鍵字': keyword, '類型': '新檢測', '是否是詐騙': 1})

# 从 Firestore 读取数据
df = fetch_data_from_firestore()
df.dropna(subset=['關鍵字', '是否是詐騙'], inplace=True)  # 处理缺失值

data = df['關鍵字'].tolist()  # 提取文本数据
labels = df['是否是詐騙'].tolist()  # 提取标签数据

# 将文本数据转化为 TF-IDF 特征向量
vectorizer = TfidfVectorizer(max_features=1000, max_df=1.0, min_df=1)
X = vectorizer.fit_transform(data).toarray()

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 降维
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 使用 K-means 进行聚类
silhouette_scores = []
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, clusters)
    silhouette_scores.append(score)

best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# 为每个聚类训练 One-Class SVM 模型
ocsvm_models = []
cluster_keywords = {i: [] for i in set(clusters)}

for cluster in set(clusters):
    X_cluster = X_pca[clusters == cluster]
    ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
    ocsvm.fit(X_cluster)
    ocsvm_models.append(ocsvm)
    # 存储每个聚类中标记为诈骗的关键字
    cluster_keywords[cluster] = df[df['是否是詐騙'] == 1]['關鍵字'].tolist()

# 从 Firestore 获取诈骗关键字
fraud_keywords = fetch_fraud_keywords()

def classify_sample(new_sample: str) -> Dict[str, Union[str, List[str]]]:
    # 将输入样本转化为向量
    new_sample_vector = vectorizer.transform([new_sample]).toarray()
    new_sample_scaled = scaler.transform(new_sample_vector)
    new_sample_pca = pca.transform(new_sample_scaled)

    # 使用已训练的 One-Class SVM 模型进行预测
    predictions = [ocsvm.predict(new_sample_pca)[0] for ocsvm in ocsvm_models]

    if any(pred == 1 for pred in predictions):
        # 如果判断为诈骗，返回相关的关键字和类型
        cluster_index = predictions.index(1)
        sample_keywords = vectorizer.inverse_transform(new_sample_vector)[0]
        matched_keywords = [kw for kw in sample_keywords if kw in fraud_keywords]
        
        # 将新匹配到的诈骗关键字保存到 Firestore
        for kw in matched_keywords:
            save_new_fraud_keyword(kw)
        
        return {
            "result": "詐騙",
            "keywords": cluster_keywords[cluster_index],
            "matched_keywords": matched_keywords
        }
    else:
        return {
            "result": "非詐騙",
            "keywords": [],
            "matched_keywords": []
        }

@app.post("/classify/")
def classify_text(sample: TextSample) -> Dict[str, Union[str, List[str]]]:
    try:
        result = classify_sample(sample.sample)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))