from flask import Flask, request, jsonify
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

app = Flask(__name__)

# 读取CSV文件并初始化模型
df = pd.read_csv('鎖定關鍵字.csv')  # 替换成实际的路径
df.dropna(subset=['關鍵字'], inplace=True)

data = df['關鍵字'].tolist()  # 将文字转换成TF-IDF特征向量
vectorizer = TfidfVectorizer(max_features=1000, max_df=1.0, min_df=1)
X = vectorizer.fit_transform(data).toarray()

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 使用K-means进行聚类
silhouette_scores = []
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, clusters)
    silhouette_scores.append(score)

best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# 在每个聚类上分别训练One-Class SVM模型
ocsvm_models = []
for cluster in set(clusters):
    X_cluster = X_pca[clusters == cluster]
    ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
    ocsvm.fit(X_cluster)
    ocsvm_models.append(ocsvm)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    new_sample = data.get('text', '')

    new_sample_vector = vectorizer.transform([new_sample]).toarray()
    new_sample_scaled = scaler.transform(new_sample_vector)
    new_sample_pca = pca.transform(new_sample_scaled)
    
    predictions = [ocsvm.predict(new_sample_pca)[0] for ocsvm in ocsvm_models]
    
    result = "非詐騙" if any(pred == 1 for pred in predictions) else "詐騙"

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
