import re
import firebase_admin
from firebase_admin import credentials, firestore
from urllib.parse import urlparse

# 初始化 Firebase Admin SDK
app = firebase_admin.get_app('app')  # 根據名稱獲取已初始化的實例
db = firestore.client(app)

# URL 的正則表達式
url_pattern = r'https?://[^\s]+'

def normalize_url(url):
    # 解析 URL，仅保留域名部分（包括协议）
    parsed_url = urlparse(url)
    domain_prefix = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return domain_prefix

def check_text_for_lineid_and_url(text):
    # 提取 URL
    urls = re.findall(url_pattern, text)
    urls = [normalize_url(url) for url in urls]

    # 提取 LineID（這裡假設是純文本，無固定格式，只需要匹配字母和數字的組合）
    line_ids = re.findall(r'[a-zA-Z0-9]+', text)
    
    # 查詢 Firebase 資料庫以檢查 URL 和 LineID
    url_info = []
    LineID_info = []

    # 查詢 FraudLine 集合中的資料
    if urls:
        for url in urls:
            
            # 查询数据库中是否有匹配的 URL 前缀
            query = db.collection('FraudURLtest').stream()
            for doc in query:
                db_url = doc.to_dict().get('URL')
                db_domain_prefix = normalize_url(db_url)
                
                # 进行前缀匹配
                if url == db_domain_prefix:
                    url_info.append({
                        'url': url,
                        'GoverURL': doc.to_dict().get('GoverURL'),
                        'Type': doc.to_dict().get('type'),
                        'Prevent':'此為警政署公布詐騙網站，可即時撥打165反詐騙諮詢專線，第一時間協助您辨明查證，降低受詐機率!'
                    })
                    break  # 找到匹配的就跳出
    
    if line_ids:
        for line_id in line_ids:
            # 查詢 FraudLine 集合是否有包含此 LineID 的資料
            query = db.collection('FraudLinetest').where('LineID', '==', line_id).get()
            for doc in query:
                LineID_info.append({
                    'LineID': line_id,
                    'GoverURL': doc.to_dict().get('GoverURL'),
                    'Type':'政府公開詐騙Line_ID',
                    'Prevent':'警政署公布「千萬別加好友」的詐騙LINE ID，這些就是各種詐騙手法所使用的帳號，如民眾發現詐騙LINE ID，亦請儘速向165反詐騙諮詢專線反映。遇到可疑 LINE ID或網址，可先利用165全民防騙網 ( https://165.npa.gov.tw ) 上方查詢功能，第一時間協助您辨明查證，降低受詐機率!'
                })

    # 去除文字中的 URL
    #text_without_urls = re.sub(url_pattern, '[URL REMOVED]', text)

    # 返回去除 URL 的文字和包含的 LineID 或 URL 和其對應 GoverURL
    return text, LineID_info, url_info


    
