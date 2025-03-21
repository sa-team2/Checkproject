from flask import Flask, request, jsonify
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS

# 初始化 Flask 應用
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}, r"/process_report": {"origins": "http://localhost:5173"}})

# 初始化 Firebase
cred = credentials.Certificate('../config/dayofftest1-firebase-adminsdk-xfpl4-2f3127e656.json')
firebase_admin.initialize_app(cred , name='app')
fapp = firebase_admin.get_app('app')  # 確保使用已初始化的 'app' 實例
db = firestore.client(fapp)  # 使用這個實例連接 Firestore

from check_text import check_text_for_lineid_and_url

# 初始化 Tokenizer
DEFAULT_TYPE = "未知"  # 默认类型

# 加載判斷是否是詐騙的模型1和處理器
def load_models():
    models = []
    i = 0
    while True:
        try:
            model = joblib.load(f'ocsvm_model_{i}.pkl')
            models.append(model)
            i += 1
        except FileNotFoundError:
            break
    return models

ocsvm_models = load_models()
vectorizer = joblib.load('vectorizer.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

# Sigmoid 函数
def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))

# 主預測函数
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text', '')
    image_urls = data.get('image_urls', [])
    ocr_texts = []
    
    if not image_urls:
        combined_text = input_text
    else:
        from ocr import process_images
        ocr_texts, ocr_results = process_images(image_urls)
        combined_text = input_text + ' ' + ' '.join(ocr_texts)

    text, lineid_data, url_data = check_text_for_lineid_and_url(combined_text)

    if not lineid_data and not url_data:
        from keywords import get_and_match_keywords_with_details, get_bert_embedding
        matched_keywords = get_and_match_keywords_with_details(text)
        bert_embedding = get_bert_embedding(text)
        new_sample_vector = vectorizer.transform([text]).toarray()
        combined_features = np.concatenate((new_sample_vector, bert_embedding), axis=1)
        new_sample_scaled = scaler.transform(combined_features)
        new_sample_pca = pca.transform(new_sample_scaled)

        scores = [ocsvm.decision_function(new_sample_pca)[0] for ocsvm in ocsvm_models]
        probabilities = sigmoid(np.array(scores))
        fraud_probability = (1 - np.mean(probabilities)) * 100
        print(fraud_probability)

        return jsonify({
            'result': '詐騙' if fraud_probability >= 50 else '非詐騙',
            'matched_keywords': matched_keywords,
            'ocr_results': ocr_results if image_urls else {},
            'FraudRate': fraud_probability
        })
    else:
        matched_keywords = []
        if lineid_data:
            for lineid in lineid_data:
                matched_keywords.append({
                    'keyword': lineid.get('LineID', '无 LineID'),
                    'type': lineid.get('Type', '无类型'),
                    'Remind': lineid.get('GoverURL', '无 GoverURL'),
                    'Prevent': lineid.get('Prevent', '无 Prevent')
                })
        
        if url_data:
            for url in url_data:
                matched_keywords.append({
                    'keyword': url.get('url', '无 URL'),
                    'type': url.get('Type', '无类型'),
                    'Remind': url.get('GoverURL', '无 GoverURL'),
                    'Prevent': url.get('Prevent', '无 Prevent')
                })

        return jsonify({
            'result': '詐騙',
            'matched_keywords': matched_keywords,
            'ocr_results': ocr_results if image_urls else {},
            'FraudRate': 100
        })

@app.route("/report_fraud", methods=["POST"])
def report_fraud():
    data = request.json
    text = data.get('text', '')
    result = data.get('result', '非詐騙')
    fraud_rate = data.get('FraudRate', 0)
    matched_keywords = data.get('matched_keywords', [])

    doc_ref = db.collection('Report').add({
        'MSG': text,
        'result': result,
        'FraudRate': fraud_rate,
        'matched_keywords': matched_keywords,
        'createdAt': firestore.SERVER_TIMESTAMP
    })

    return jsonify({"message": "回報成功", "doc_id": doc_ref.id})

# 新增的處理報告端點，專門給 AdminPreview.jsx 使用
@app.route("/process_report", methods=["POST"])
def process_report():
    data = request.json
    reports = data.get('reports', [])
    
    if not reports:
        return jsonify({"error": "No reports provided"}), 400
    
    results = []
    
    for report in reports:
        try:
            report_id = report.get('id')
            report_text = report.get('text', '')
            
            if not report_id:
                results.append({
                    "status": "error",
                    "message": "Missing report ID"
                })
                continue
                
            if not report_text:
                # 嘗試從資料庫獲取報告文本
                try:
                    report_ref = db.collection('Report').document(report_id)
                    report_doc = report_ref.get()
                    
                    if not report_doc.exists:
                        results.append({
                            "id": report_id,
                            "status": "error",
                            "message": "Report not found"
                        })
                        continue
                    
                    report_data = report_doc.to_dict()
                    report_text = report_data.get('Report', report_data.get('MSG', ''))
                except Exception as e:
                    results.append({
                        "id": report_id,
                        "status": "error",
                        "message": f"Error fetching report: {str(e)}"
                    })
                    continue
            
            if not report_text:
                results.append({
                    "id": report_id,
                    "status": "error",
                    "message": "No text found in report"
                })
                continue
            
            # 嘗試檢測 URL，但不直接標記為詐騙
            try:
                import re
                url_pattern = re.compile(r'https?://\S+')
                url_matches = url_pattern.findall(report_text)
                
                # 收集URL作為關鍵字，但不直接判定為詐騙
                matched_keywords = []
                if url_matches:
                    for url in url_matches:
                        matched_keywords.append({
                            'keyword': url,
                            'type': 'URL'  # 改為中性的標籤
                        })
                
                # 無論是否有URL，都使用ML模型進行評估
                try:
                    # 確保文本不為空且含有有效詞彙
                    cleaned_text = report_text.strip()
                    if not cleaned_text:
                        results.append({
                            "id": report_id,
                            "status": "error",
                            "message": "Text is empty after cleaning"
                        })
                        continue
                    
                    # 預設一些基本文本，確保向量化不會失敗
                    if len(cleaned_text.split()) < 3:
                        cleaned_text = cleaned_text + " this is additional text to ensure vectorization works"
                    
                    from keywords import get_and_match_keywords_with_details, get_bert_embedding
                    keyword_matches = get_and_match_keywords_with_details(cleaned_text)
                    matched_keywords.extend(keyword_matches)  # 合併關鍵詞結果
                    bert_embedding = get_bert_embedding(cleaned_text)
                    
                    try:
                        # 嘗試向量化
                        new_sample_vector = vectorizer.transform([cleaned_text]).toarray()
                        
                        # 繼續處理
                        combined_features = np.concatenate((new_sample_vector, bert_embedding), axis=1)
                        new_sample_scaled = scaler.transform(combined_features)
                        new_sample_pca = pca.transform(new_sample_scaled)
                        
                        scores = [ocsvm.decision_function(new_sample_pca)[0] for ocsvm in ocsvm_models]
                        probabilities = sigmoid(np.array(scores))
                        base_fraud_probability = (1 - np.mean(probabilities)) * 100
                        
                        # 如果有URL，適當調整詐騙率，但不直接設為100%
                        final_fraud_probability = base_fraud_probability
                        if url_matches:
                            # 提高詐騙率，但保留一些不確定性
                            # 根據URL數量微調增加的幅度
                            url_bonus = min(len(url_matches) * 15, 30)  # 最多增加30%
                            final_fraud_probability = min(base_fraud_probability + url_bonus, 95)
                        
                        fraud_probability = final_fraud_probability
                        fraud_result = '詐騙' if fraud_probability >= 50 else '非詐騙'
                        
                    except Exception as ve:
                        print(f"Vectorization error: {ve}")
                        # 如果向量化失敗，使用關鍵詞匹配結果
                        if matched_keywords:
                            fraud_result = '詐騙'
                            fraud_probability = 80
                        else:
                            fraud_result = '非詐騙'
                            fraud_probability = 20
                except Exception as ml_error:
                    print(f"ML error: {ml_error}")
                    # 使用簡單的判斷邏輯
                    if any(kw in report_text.lower() for kw in ['騙', '詐騙', 'scam', 'fraud']):
                        fraud_result = '詐騙'
                        fraud_probability = 90
                        matched_keywords.append({
                            'keyword': '詐騙相關詞語',
                            'type': '詐騙關鍵詞'
                        })
                    else:
                        # 即使有URL也不直接判定為詐騙
                        if url_matches:
                            fraud_result = '可疑'
                            fraud_probability = 60
                        else:
                            fraud_result = '非詐騙'
                            fraud_probability = 10
            except Exception as detection_error:
                print(f"Detection error: {detection_error}")
                # 最終的 fallback
                fraud_result = '未知'
                fraud_probability = 50
                matched_keywords = []
            
            # 更新 Report 文檔的 PythonResult 欄位
            try:
                simplified_match = [
                    {
                        'MatchKeyword': item.get('keyword', ''),
                        'MatchType': item.get('type', '未知')
                    } for item in matched_keywords
                ]
                
                report_ref = db.collection('Report').document(report_id)
                report_ref.update({
                    'PythonResult': {
                        'FraudResult': fraud_result,
                        'FraudRate': fraud_probability,
                        'Match': simplified_match
                    }
                })
            except Exception as db_error:
                print(f"Database update error: {db_error}")
            
            # 添加處理結果
            results.append({
                "id": report_id,
                "status": "success",
                "result": fraud_result,
                "FraudRate": fraud_probability,
                "matched_keywords": matched_keywords,
                "row": {
                    "Report": report_text[:100] + ('...' if len(report_text) > 100 else '')
                }
            })
            
        except Exception as e:
            print(f"Unexpected error processing report {report.get('id', 'unknown')}: {e}")
            results.append({
                "id": report.get('id', "unknown"),
                "status": "error",
                "message": str(e)
            })
    
    return jsonify({
        "message": f"Processed {len(results)} reports",
        "results": results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
