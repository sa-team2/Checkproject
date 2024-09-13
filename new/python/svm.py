from flask import Flask, request, jsonify
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import re

app = Flask(__name__)

# 读取CSV文件并初始化模型
df = pd.read_csv('C:/Users/a0311/OneDrive/桌面/專題/new/new/python/鎖定關鍵字.csv')
df.dropna(subset=['關鍵字'], inplace=True)

# 获取关键字、标签和类型
keywords = df['關鍵字'].tolist()
labels = df['是否是詐騙'].tolist()  # 假设标签列名为 '是否是詐騙'
types = df['類型'].tolist()  # 假设类型列名为 '類型'

# 初始化TF-IDF特征向量，使用 ngram_range=(1, 2) 来捕捉单词对的特征
vectorizer = TfidfVectorizer(max_features=1000, max_df=1.0, min_df=1, ngram_range=(1, 2))
X = vectorizer.fit_transform(keywords).toarray()

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
    new_sample = data.get('text', '').lower()  # 转换为小写以避免大小写敏感问题

    # 处理新样本
    new_sample_vector = vectorizer.transform([new_sample]).toarray()
    new_sample_scaled = scaler.transform(new_sample_vector)
    new_sample_pca = pca.transform(new_sample_scaled)
    
    # 使用One-Class SVM模型进行预测
    predictions = [ocsvm.predict(new_sample_pca)[0] for ocsvm in ocsvm_models]
    svm_result = any(pred == 1 for pred in predictions)
    
    # 关键字匹配和标签检查
    is_fraud = False
    matched_keywords = []
    
    # 调试信息
    print(f"新样本: {new_sample}")
    print(f"One-Class SVM 预测结果: {svm_result}")
    
    # 标志是否找到匹配的关键字
    keyword_found = False
    
    for keyword, label, type_ in zip(keywords, labels, types):
        # 使用正则表达式进行部分匹配
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        if pattern.search(new_sample):  # 正则表达式匹配
            keyword_found = True
            print(f"匹配到关键字: {keyword} (标签: {label}, 类型: {type_})")  # 调试输出
            if label == 1:
                is_fraud = True
            matched_keywords.append({'MatchKeyword': keyword, 'MatchType': type_})

    # 打印是否找到关键字
    print(f"是否找到关键字: {keyword_found}")

    # 综合判断结果
    if not keyword_found:  # 如果没有找到匹配的关键字
        result = "非詐騙"
    elif svm_result and not is_fraud:
        result = "非詐騙"
    else:
        result = "詐騙"
    
    # 返回结果中加入匹配到的关键字和类型
    return jsonify({'FraudResult': result, 'MatchKeywords': matched_keywords})

if __name__ == '__main__':
    # 设置应用在 localhost:5000 运行
    app.run(debug=True, host='127.0.0.1', port=5000)
