import { NextResponse } from 'next/server';
import puppeteer from 'puppeteer';
import axios from 'axios';
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
        const response = await fetch('http://localhost:5000/predict', { // Python 后端地址
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
    const { url } = await request.json();

    console.log(`Processing URL: ${url}`);

    let browser;

    try {
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

        const expirationTime = admin.firestore.Timestamp.fromDate(new Date(Date.now() + 60 * 60 * 1000));

        const simplifiedPythonResult = {
            result: pythonResult.result || '未检测到',
            matched_keywords: (pythonResult.matched_keywords || []).map(item => ({
                keyword: item.keyword || '无关键词',
                type: item.type || '无类型'
            })) // 确保 matched_keywords 是数组，并给每个对象提供默认值
        };


        // 确保 matchedKeywords 字段不为 undefined
        const docRef = await db.collection('Outcome').add({
            url,
            content,
            pythonResult: simplifiedPythonResult, // 确保数据结构简单
            timestamp: admin.firestore.FieldValue.serverTimestamp(),
        });

        console.log('Python Result:', pythonResult);

        const result = {
            matched_keywords: pythonResult.matched_keywords || [], // 如果 undefined，使用空数组
            result: pythonResult.result || '未检测到' // 如果 undefined，使用默认值
        };

        return NextResponse.json({
            success: true,
            message: '内容和图片 URL 已成功处理',
            documentId: docRef.id,
            content,
            pythonResult: result,
        });
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
