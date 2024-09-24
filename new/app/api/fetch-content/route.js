import { NextResponse } from 'next/server';
import puppeteer from 'puppeteer';
import fetch from 'node-fetch'; 
import admin from 'firebase-admin';
import serviceAccount from '../../../config/dayofftest1-firebase-adminsdk-xfpl4-cdd57f1038.json'; 

// 初始化 Firebase Admin SDK
if (!admin.apps.length) {
    admin.initializeApp({
        credential: admin.credential.cert(serviceAccount),
    });
}

const db = admin.firestore();

async function sendImageUrlToPythonService(text, imageUrls) {
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text, image_urls: imageUrls }), // 傳遞文字和圖片 URL 列表給 Python
        });

        if (!response.ok) {
            throw new Error('呼叫 Python 服務失敗');
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('呼叫 Python 服務時出錯:', error.message);
        throw error;
    }
}

export async function POST(request) {
    const { url, text } = await request.json();

    console.log(`處理 URL: ${url}`);

    let browser;

    try {
        if (url) {
            // 處理 URL 
            browser = await puppeteer.launch({ headless: true });
            const page = await browser.newPage();
            await page.goto(url, { waitUntil: 'domcontentloaded' });

            const content = await page.evaluate(() => {
                return document.body.innerText;
            });

            const imageUrls = await page.evaluate(() => {
                return Array.from(document.querySelectorAll('img')).map(img => img.src);
            });

            console.log(`找到圖片 URL: ${imageUrls}`);

            // 發送文字和圖片 URL 到 Python 進行處理
            const pythonResult = await sendImageUrlToPythonService(content, imageUrls);

            const simplifiedPythonResult = {
                FraudResult: pythonResult.result || '未檢測到',
                Match: (pythonResult.matched_keywords || []).map(item => ({
                    MatchKeyword: item.keyword || '無關鍵詞',
                    MatchType: item.type || '無類型'
                })),
                FraudRate: pythonResult.FraudRate || 0 

            };

            // 儲存到 Firebase
            const docRef = await db.collection('Outcome').add({
                DetectionType: 1, // 指示資料來源類型
                Content: content, 
                PythonResult: simplifiedPythonResult, // 確保資料結構簡單
                URL:url,
                TimeStamp: admin.firestore.FieldValue.serverTimestamp(),
            });

            console.log('Python 結果:', pythonResult);

            const result = {
                matched_keywords: pythonResult.matched_keywords || [], 
                result: pythonResult.result || '未檢測到',
                FraudRate: pythonResult.FraudRate || 0
            };

            return NextResponse.json({
                success: true,
                content,
                pythonResult: result,
            });
        } else if (text) {
            // 處理簡訊
            const imageUrls = []; 

            // 發送文字和圖片 URL 到 Python 進行處理
            const pythonResult = await sendImageUrlToPythonService(text, imageUrls);

            const simplifiedPythonResult = {
                FraudResult: pythonResult.result || '未檢測到',
                Match: (pythonResult.matched_keywords || []).map(item => ({
                    MatchKeyword: item.keyword || '無關鍵詞',
                    MatchType: item.type || '無類型'
                })),
                FraudRate: pythonResult.FraudRate || 0 

            };

            // 儲存到 Firebase
            const docRef = await db.collection('Outcome').add({
                DetectionType: 2, // 指示資料來源類型
                Content: text, 
                PythonResult: simplifiedPythonResult, // 確保資料結構簡單
                TimeStamp: admin.firestore.FieldValue.serverTimestamp(),
            });

            console.log('Python 結果:', pythonResult);

            const result = {
                matched_keywords: pythonResult.matched_keywords || [],
                result: pythonResult.result || '未檢測到', 
                FraudRate: pythonResult.FraudRate || 0
            };

            return NextResponse.json({
                success: true,
                pythonResult: result,
            });
        } else {
            return NextResponse.json({
                success: false,
                message: '請求缺少有效資料'
            });
        }
    } catch (error) {
        console.error('處理失敗:', error.message);
        return NextResponse.json({ success: false, message: error.message });
    } finally {
        if (browser) {
            try {
                await browser.close();
            } catch (closeError) {
                console.error('關閉瀏覽器失敗:', closeError.message);
            }
        }
    }
}
