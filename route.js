import { NextResponse } from 'next/server';
import axios from 'axios';
import puppeteer from 'puppeteer';
import Tesseract from 'tesseract.js';
import sharp from 'sharp';
import admin from 'firebase-admin';
import serviceAccount from '../../../config/test-bc002-firebase-adminsdk-47w0c-20f1ea4f43.json'; // 确保路径正确

// 初始化 Firebase Admin SDK
if (!admin.apps.length) {
    admin.initializeApp({
        credential: admin.credential.cert(serviceAccount),
    });
}

const db = admin.firestore();

async function fetchImage(url) {
    try {
        const response = await axios.get(url, { responseType: 'arraybuffer' });
        return Buffer.from(response.data);
    } catch (error) {
        console.error(`Failed to fetch image from ${url}:`, error.message);
        throw error;
    }
}

async function processImage(imageUrl) {
    try {
        const imageBuffer = await fetchImage(imageUrl);
        const format = await sharp(imageBuffer).metadata().then(metadata => metadata.format);
        console.log(`Processing format: ${format}`);

        const processedImageBuffer = await sharp(imageBuffer)
            .resize({ width: 800 })
            .grayscale()
            .toBuffer();

        // 保持 Tesseract.js 的现有代码不变
        const worker = await Tesseract.createWorker("chi_tra", 1, {workerPath: "./node_modules/tesseract.js/src/worker-script/node/index.js"});

        const { data: { text } } = await worker.recognize(processedImageBuffer);
        await worker.terminate();

        return text;
    } catch (error) {
        console.error(`Failed to process image ${imageUrl}:`, error.message);
        return ''; // 返回空字符串以避免影响后续结果
    }
}

// 修改后的 sendTextToPythonService 函数
async function sendTextToPythonService(text, additionalText) {
    try {
        const response = await fetch('http://localhost:5000/predict', { // 假设 Python 后端运行在 localhost:5000
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text + "\n\n" + additionalText }), // 合并文本
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

        const ocrTexts = await Promise.all(imageUrls
            .filter(imageUrl => !imageUrl.startsWith('data:image')) // Skip base64 images
            .map(imageUrl => processImage(imageUrl))
        );

        const ocrText = ocrTexts.join('\n\n');

        // 调用 Python 服务处理合并后的文本
        const pythonResult = await sendTextToPythonService(content, ocrText);

        const expirationTime = admin.firestore.Timestamp.fromDate(new Date(Date.now() + 60 * 60 * 1000));

        // 确保 matchedKeywords 字段不为 undefined
        const docRef = await db.collection('webContent').add({
            url,
            content,
            ocrText,
            pythonResult, // 保存 Python 处理结果
            matchedKeywords: pythonResult.matchedKeywords || [], // 如果 undefined，使用空数组
            timestamp: admin.firestore.FieldValue.serverTimestamp(),
        });

        return NextResponse.json({
            success: true,
            message: '内容和 OCR 文本成功保存',
            documentId: docRef.id,
            content,
            ocrText,
            pythonResult,
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
