import re
import requests
from urllib.parse import urlparse

# URL 和 LineID 的 API 地址
url_api = "https://od.moi.gov.tw/api/v1/rest/datastore/A01010000C-002150-013"
line_api = "https://od.moi.gov.tw/api/v1/rest/datastore/A01010000C-001277-053"

# URL 的正則表達式
url_pattern = r'https?://[^\s]+'

def fetch_api_data(api_url):
    """
    從指定 API 獲取資料
    """
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        # 提取 API 回傳的 records 資料
        return data.get("result", {}).get("records", [])
    except requests.exceptions.RequestException as e:
        print(f"無法從 API 獲取資料: {e}")
        return []

def check_text_for_lineid_and_url(text):
    # 提取 URL
    urls = re.findall(url_pattern, text)
    # 提取 LineID（假設是純文字，無固定格式，只需要匹配字母和數字的組合）
    line_ids = re.findall(r'[a-zA-Z0-9_]+', text)
    
    # 獲取 API 資料
    url_api_data = fetch_api_data(url_api)
    line_api_data = fetch_api_data(line_api)
    
    url_info = []
    lineid_info = []

    # 查詢 URL
    if urls and url_api_data:
        for url in urls:
            text = text.replace(url, '')
            for record in url_api_data:
                # 使用 `WEBURL` 作為匹配欄位
                db_url = record.get("WEBURL")
                if db_url and not db_url.startswith(('http://', 'https://')):
                    db_url = f"https://{db_url}"  # 补全为 https://
                if db_url:
                    # 進行 URL 包含匹配
                    if db_url in url:  # 判断 db_url 是否是 url 的一部分
                        url_info.append({
                            'url': url,
                            'GoverURL': record.get("WEBURL"),
                            'Type': '政府公開詐騙網站',
                            'Prevent': '此為警政署公布詐騙網站，可即時撥打165反詐騙諮詢專線，第一時間協助您辨明查證，降低受詐機率！'
                        })
                        break  # 找到匹配的就跳出

    # 查詢 LineID
    if line_ids and line_api_data:
        for line_id in line_ids:
            for record in line_api_data:
                # 使用 `LineID` 欄位匹配
                db_line_id = record.get("帳號")
                if db_line_id and line_id == db_line_id:
                    lineid_info.append({
                        'LineID': line_id,
                        'GoverURL': record.get("帳號"),
                        'Type': '政府公開詐騙 Line_ID',
                        'Prevent': '警政署公布「千萬別加好友」的詐騙 LINE ID，這些就是各種詐騙手法所使用的帳號。如發現詐騙 LINE ID，請即時向 165 反詐騙諮詢專線反映。'
                    })
                    text = text.replace(line_id, '')
                    break  # 找到匹配的就跳出

    # 返回文字、LineID 資訊和 URL 資訊
    return text, lineid_info, url_info
