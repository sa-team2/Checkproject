import { NextResponse } from 'next/server';
import puppeteer from 'puppeteer';
import fetch from 'node-fetch';
import admin from 'firebase-admin';
import serviceAccount from '../../../config/dayofftest1-firebase-adminsdk-xfpl4-f64d9dc336.json';
import fs from 'fs/promises'; // 用于读取文件内容
import mime from 'mime-types'; // 用于检测文件的 MIME 类型

// 初始化 Firebase Admin SDK
if (!admin.apps.length) {
    admin.initializeApp({
        credential: admin.credential.cert(serviceAccount),
    });
}

export const config = {
    api: {
        bodyParser: false,
    },
};

const db = admin.firestore();

// 发送数据到 Python 服务
async function sendImageUrlToPythonService(text, imageUrls) {
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text, image_urls: imageUrls }),
        });

        if (!response.ok) {
            throw new Error('调用 Python 服务失败');
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('调用 Python 服务时出错:', error.message);
        throw error;
    }
}

// 读取文件内容（针对文本文件）
async function readFileContent(buffer) {
    return buffer.toString('utf-8'); // 将 Buffer 转换为字符串
}

export async function POST(request) {
    let browser;

    try {
        // 解析请求中的文件和其他数据
        const contentType = request.headers.get('content-type');

        if (contentType.includes('application/json')) {
            // 处理 JSON 请求
            const json = await request.json();
            const { url, text } = json;

            // 处理 URL
            if (url) {
                console.log(`处理 URL: ${url}`);
                browser = await puppeteer.launch({ headless: true });
                const page = await browser.newPage();
                await page.goto(url, { waitUntil: 'domcontentloaded' });

                const content = await page.evaluate(() => document.body.innerText);
                const imageUrls = await page.evaluate(() =>
                    Array.from(document.querySelectorAll('img')).map(img => img.src)
                );

                console.log(`找到图片 URL: ${imageUrls}`);

                // 发送文本和图片 URL 到 Python 处理
                const pythonResult = await sendImageUrlToPythonService(content, imageUrls);

                const simplifiedPythonResult = {
                    FraudResult: pythonResult.result || '未检测到',
                    Match: (pythonResult.matched_keywords || []).map(item => ({
                        MatchKeyword: item.keyword || '无关键词',
                        MatchType: item.type || '无类型',
                    })),
                    FraudRate: pythonResult.FraudRate || 0,
                };

                // 保存结果到 Firebase
                const docRef = await db.collection('Outcome').add({
                    DetectionType: 1,
                    Content: content,
                    PythonResult: simplifiedPythonResult,
                    URL: url,
                    TimeStamp: admin.firestore.FieldValue.serverTimestamp(),
                });

                const ID = docRef.id;

                return NextResponse.json({
                    ID,
                    success: true,
                    content,
                    pythonResult: simplifiedPythonResult,
                });

            // 处理文本信息
            } else if (text) {
                console.log('处理文本信息');
                const imageUrls = []; // 没有图片时传递空列表
                const pythonResult = await sendImageUrlToPythonService(text, imageUrls);

                const simplifiedPythonResult = {
                    FraudResult: pythonResult.result || '未检测到',
                    Match: (pythonResult.matched_keywords || []).map(item => ({
                        MatchKeyword: item.keyword || '无关键词',
                        MatchType: item.type || '无类型',
                    })),
                    FraudRate: pythonResult.FraudRate || 0,
                };

                // 保存到 Firebase
                const docRef = await db.collection('Outcome').add({
                    DetectionType: 2,
                    Content: text,
                    PythonResult: simplifiedPythonResult,
                    TimeStamp: admin.firestore.FieldValue.serverTimestamp(),
                });

                const ID = docRef.id;

                return NextResponse.json({
                    ID,
                    success: true,
                    pythonResult: simplifiedPythonResult,
                });
            }
            else {
                return NextResponse.json({
                    success: false,
                    message: '无法识别的文件类型',
                });
            }
        } 

        // 处理文件上传
        else if (contentType.includes('multipart/form-data')) {
            const formData = await request.formData(); // 使用 formData 方法
            const file = formData.get('file'); // 获取文件
            console.log("form data", formData, file); // 打印 formData 和文件

            // 使用 Buffer 对象
            const uploadedFileBuffer = await file.arrayBuffer(); // 获取文件的 Buffer
            const uploadedFileName = file.name; // 获取文件名
            console.log(`处理上传文件: ${uploadedFileName}`); // 打印文件名

            // 检查文件类型
            const mimeType = mime.lookup(uploadedFileName); // 使用文件名获取 MIME 类型
            console.log(`文件 MIME 类型: ${mimeType}`);

            if (typeof mimeType === 'string') {
                // 如果是图片文件，直接发送给 Python 服务处理
                if (mimeType.startsWith('image/')) {
                    console.log('检测到图片文件，发送到 Python 服务处理');
                    const fs = require('fs');
                    const path = require('path');
                    const filePath = path.resolve(__dirname, `C:\\Users\\a0311\\OneDrive\\桌面\\專題\\new\\new\\uploads`, uploadedFileName); // 设置文件存储路径
                    console.log(__dirname); // 查看當前目錄

                    fs.writeFileSync(filePath, Buffer.from(uploadedFileBuffer)); // 写入文件
                    console.log(`文件已保存至: ${filePath}`);
                
                    // 发送文件路径到 Python 服务处理
                    const pythonResult = await sendImageUrlToPythonService('',filePath);
                    
                    // 处理完后删除文件
                    fs.unlinkSync(filePath); // 删除文件
                    console.log(`文件已删除: ${filePath}`);
                    console.log(pythonResult.ocr_results);

                    const simplifiedPythonResult = {
                        FraudResult: pythonResult.result || '未检测到',
                        Match: (pythonResult.matched_keywords || []).map(item => ({
                            MatchKeyword: item.keyword || '无关键词',
                            MatchType: item.type || '无类型',
                        })),
                        FraudRate: pythonResult.FraudRate || 0,
                    };
    
                    const docRef = await db.collection('Outcome').add({
                        DetectionType: 4, // 代表图片上传
                        PythonResult: simplifiedPythonResult,
                        TimeStamp: admin.firestore.FieldValue.serverTimestamp(),
                    });
    
                    const ID = docRef.id;
    

                    return NextResponse.json({
                        ID,
                        success: true,
                        pythonResult: simplifiedPythonResult,
                    });

                } else if (mimeType.startsWith('text/')) {
                    // 如果是文本文件，读取文件内容并发送给 Python 服务
                    const text = await readFileContent(Buffer.from(uploadedFileBuffer));
                    console.log(`读取到的文本内容: ${text}`);

                    const imageUrls = []; // 没有图片时传递空列表
                    const pythonResult = await sendImageUrlToPythonService(text, imageUrls);

                    const simplifiedPythonResult = {
                        FraudResult: pythonResult.result || '未检测到',
                        Match: (pythonResult.matched_keywords || []).map(item => ({
                            MatchKeyword: item.keyword || '无关键词',
                            MatchType: item.type || '无类型',
                        })),
                        FraudRate: pythonResult.FraudRate || 0,
                    };

                    // 保存到 Firebase
                    const docRef = await db.collection('Outcome').add({
                        DetectionType: 3,
                        Content: text,
                        PythonResult: simplifiedPythonResult,
                        TimeStamp: admin.firestore.FieldValue.serverTimestamp(),
                    });

                    const ID = docRef.id;

                    return NextResponse.json({
                        ID,
                        success: true,
                        pythonResult: simplifiedPythonResult,
                    });

                } else {
                    return NextResponse.json({
                        success: false,
                        message: '不支持的文件类型',
                    });
                }
            } else {
                return NextResponse.json({
                    success: false,
                    message: '无法识别的文件类型',
                });
            }
        }

    } catch (error) {
        console.error('处理失败:', error.message);
        return NextResponse.json({ success: false, message: error.message });
    } finally {
        if (browser) {
            try {
                await browser.close();
            } catch (closeError) {
                console.error('关闭浏览器失败:', closeError.message);
            }
        }
    }
}
