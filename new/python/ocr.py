from paddleocr import PaddleOCR
import cv2
import numpy as np
import requests


# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)

# 用于跟蹤處理過的圖片 URL
processed_urls = set()


def download_image_from_url(image_url):
    """从 URL 下载图片并转为 OpenCV 格式"""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        if 'image' not in response.headers.get('Content-Type', ''):
            raise Exception(f"URL does not point to an image: {image_url}")

        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise Exception(f"Failed to decode image: {image_url}")
        return image
    except Exception as e:
        print(f"Failed to download or decode image {image_url}: {e}")
        return None
    

def load_image_from_path(image_url):
    """从本地路径加载图像并转为 OpenCV 格式"""
    try:
        image = cv2.imdecode(np.fromfile(image_url, dtype=np.uint8), -1)
        if image is None:
            raise Exception(f"Failed to load image from path: {image_url}")
        return image
    except Exception as e:
        print(f"Error loading image from local path: {e}")
        raise  # 抛出异常，便于更好地调试



# 增強圖片
def enhance_image(image, scale_factor=4.0):
    """增強圖片（調整對比度、亮度和銳化）"""
    alpha = 1.5  # 對比度
    beta = 0     # 亮度
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

# OCR 識别
def perform_ocr(image):
    """执行 OCR 识别"""
    result = ocr.ocr(image, cls=True)
    return result

# 提取 OCR 識别結果中的文字
def extract_text_from_ocr(result):
    """提取 OCR 識别結果中的文字"""
    if not result:
        return ''
    text = ''
    for line in result:
        for word_info in line:
            text += word_info[1][0] + ' '
    return text.strip()

def is_url(path):
    """判断传递的路径是否为 URL"""
    return path.startswith(('http://', 'https://'))


def process_images(image_urls):
    """批量处理图像 URLs，并返回 OCR 结果"""
    ocr_texts = []
    ocr_results = {}

    for image_url in image_urls:
        if (image_url in processed_urls) and is_url(image_url):
            print(f"Skipping already processed URL: {image_url}")
            continue
        processed_urls.add(image_url)

        try:
            if is_url(image_url):
                # 处理 URL
                image = download_image_from_url(image_url)
            else:
                image = load_image_from_path(image_url)  # 加载本地文件

            if image is None:
                continue
            enhanced_image = enhance_image(image)
            ocr_data = perform_ocr(enhanced_image)
            ocr_text = extract_text_from_ocr(ocr_data)
            ocr_texts.append(ocr_text)
            ocr_results[image_url] = ocr_text

        except Exception as e:
            print(f"Failed to process image {image_url}: {e}")
            ocr_results[image_url] = str(e)

        if not is_url(image_url):
            break

    return ocr_texts, ocr_results