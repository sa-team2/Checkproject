import { NextResponse } from 'next/server';
import puppeteer from 'puppeteer';
import fetch from 'node-fetch'; // 确保你安装了 node-fetch
import admin from 'firebase-admin';
import serviceAccount from '../../../config/dayofftest1-firebase-adminsdk-xfpl4-cdd57f1038.json'; // 确保路径正确

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
            body: JSON.stringify({ text, image_urls: imageUrls }), // 传递文本和图片 URL 列表给 Python
        });

        if (!response.ok) {
            throw new Error('Failed to call Python service');
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error calling Python service:', error.message);
        throw error;
    }
}

export async function POST(request) {
    const { url, text } = await request.json();

    console.log(`Processing URL: ${url}`);
    console.log(`Processing Text: ${text}`);

    let browser;

    try {
        if (url) {
            // 处理 URL 的情况
            browser = await puppeteer.launch({ headless: true });
            const page = await browser.newPage();
            await page.goto(url, { waitUntil: 'domcontentloaded' });

            const content = await page.evaluate(() => {
                return document.body.innerText;
            });

            const imageUrls = await page.evaluate(() => {
                return Array.from(document.querySelectorAll('img')).map(img => img.src);
            });

            console.log(`Found image URLs: ${imageUrls}`);

            // 发送文本和图片 URL 到 Python 后端进行处理
            const pythonResult = await sendImageUrlToPythonService(content, imageUrls);

            const simplifiedPythonResult = {
                result: pythonResult.result || '未检测到',
                matched_keywords: (pythonResult.matched_keywords || []).map(item => ({
                    keyword: item.keyword || '无关键词',
                    type: item.type || '无类型'
                })),
                FraudRate: pythonResult.FraudRate || 0 // 添加 FraudRate

            };

            // 存储到 Firebase
            const docRef = await db.collection('Outcome').add({
                detection_type: 1, // 指示数据来源类型
                url,
                content,
                pythonResult: simplifiedPythonResult, // 确保数据结构简单
                timestamp: admin.firestore.FieldValue.serverTimestamp(),
            });

            console.log('Python Result:', pythonResult);

            const result = {
                matched_keywords: pythonResult.matched_keywords || [], // 如果 undefined，使用空数组
                result: pythonResult.result || '未检测到',// 如果 undefined，使用默认值
                FraudRate: pythonResult.FraudRate || 0
            };

            return NextResponse.json({
                success: true,
                content,
                pythonResult: result,
            });
        } else if (text) {
            // 处理简讯文本的情况
            const imageUrls = []; 

            // 发送文本和图片 URL 到 Python 后端进行处理
            const pythonResult = await sendImageUrlToPythonService(text, imageUrls);

            const simplifiedPythonResult = {
                result: pythonResult.result || '未检测到',
                matched_keywords: (pythonResult.matched_keywords || []).map(item => ({
                    keyword: item.keyword || '无关键词',
                    type: item.type || '无类型'
                })),
                FraudRate: pythonResult.FraudRate || 0 // 添加 FraudRate

            };

            // 存储到 Firebase
            const docRef = await db.collection('Outcome').add({
                detection_type: 2, // 指示数据来源类型
                content: text, // 将文本内容存储为 content
                pythonResult: simplifiedPythonResult, // 确保数据结构简单
                timestamp: admin.firestore.FieldValue.serverTimestamp(),
            });

            console.log('Python Result:', pythonResult);

            const result = {
                matched_keywords: pythonResult.matched_keywords || [], // 如果 undefined，使用空数组
                result: pythonResult.result || '未检测到', // 如果 undefined，使用默认值
                FraudRate: pythonResult.FraudRate || 0
            };

            return NextResponse.json({
                success: true,
                pythonResult: result,
            });
        } else {
            return NextResponse.json({
                success: false,
                message: '请求缺少有效数据'
            });
        }
    } catch (error) {
        console.error('处理失败:', error.message);
        return NextResponse.json({ success: false, message: error.message });
    } finally {
        if (browser) {
            try {
                await browser.close();
            } catch (closeError) {
                console.error('Failed to close browser:', closeError.message);
            }
        }
    }
}
