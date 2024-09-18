import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from firebase_admin import credentials, firestore
import firebase_admin

# 初始化 Firebase Admin
cred = credentials.Certificate('../config/dayofftest1-firebase-adminsdk-xfpl4-cdd57f1038.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

def get_data_from_firestore():
    collection_ref = db.collection('FraudDefine')
    docs = collection_ref.stream()
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    return data

def update_model():
    # 从 Firestore 获取数据
    firestore_data = get_data_from_firestore()
    df = pd.DataFrame(firestore_data)
    df.dropna(subset=['Keyword'], inplace=True)

    data = df['Keyword'].tolist()
    vectorizer = TfidfVectorizer(max_features=1000, max_df=1.0, min_df=1)
    X = vectorizer.fit_transform(data).toarray()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    silhouette_scores = []
    for n_clusters in range(2, 20):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, clusters)
        silhouette_scores.append(score)

    best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    ocsvm_models = []
    for cluster in set(clusters):
        X_cluster = X_pca[clusters == cluster]
        ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
        ocsvm.fit(X_cluster)
        ocsvm_models.append(ocsvm)

    # 保存模型
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca.pkl')
    for i, model in enumerate(ocsvm_models):
        joblib.dump(model, f'ocsvm_model_{i}.pkl')

if __name__ == "__main__":
    update_model()
