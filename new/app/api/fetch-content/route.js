import { NextResponse } from 'next/server';
import puppeteer from 'puppeteer';
import fetch from 'node-fetch';
import admin from 'firebase-admin';
import serviceAccount from '../../../config/dayofftest1-firebase-adminsdk-xfpl4-f64d9dc336.json';
import fs from 'fs';
import mime from 'mime-types';
import { spawn } from 'child_process';
import path from 'path';

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

async function sendImageUrlToPythonService(text, imageUrls) {
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, image_urls: imageUrls }),
        });

        if (!response.ok) {
            throw new Error('调用 Python 服务失败');
        }

        return await response.json();
    } catch (error) {
        console.error('调用 Python 服务时出错:', error.message);
        throw error;
    }
}

async function readFileContent(buffer) {
    return buffer.toString('utf-8');
}

function createResponse(success, data = {}, message = '') {
    return NextResponse.json({ success, ...data, message });
}

async function saveToFirestore(detectionType, content, pythonResult) {
    console.log(pythonResult); // 现在应该输出实际的数据而不是 Promise
    const docRef = await db.collection('Outcome').add({
        DetectionType: detectionType,
        Content: content,
        PythonResult: pythonResult,
        TimeStamp: admin.firestore.FieldValue.serverTimestamp(),
    });

    return docRef.id;
}


async function processPythonResult(pythonResult) {
    // 使用 for...of 循环，确保每个处理步骤是同步顺序的
    const matches = [];
    for (const item of (pythonResult.matched_keywords || [])) {
        const fraudTypeDetails = await getFraudTypeDetails(item.type);
        matches.push({
            MatchKeyword: item.keyword || '无关键词',
            MatchType: item.type || '无类型',
            Remind: fraudTypeDetails.Remind,
            Prevent: fraudTypeDetails.Prevent,
        });
    }

    // 构造返回对象
    const simplifiedPythonResult = {
        FraudResult: pythonResult.result || '未检测到',
        FraudRate: pythonResult.FraudRate || 0,
        Match: matches,
    };

    // 确保返回的对象是标准的 JSON 格式
    return JSON.parse(JSON.stringify(simplifiedPythonResult));
}



async function getFraudTypeDetails(type) {
    try {
        console.log('Searching for type:', type);
        const snapshot = await db.collection('FraudTypeDefine').where('Type', '==', type).get();
        console.log('Searching for type:', snapshot.docs[0]);

        if (!snapshot.empty) {
            const doc = snapshot.docs[0];
            return {
                Remind: doc.data().Remind || ' ',
                Prevent: doc.data().Prevent || ' ',
            };
        } else {
            return {
                Remind: ' ',
                Prevent: ' ',
            };
        }
    } catch (error) {
        console.error('查询 FraudTypeDefine 时出错:', error.message);
        return {
            Remind: ' ',
            Prevent: ' ',
        };
    }
}

//--------------------------------------------------------------------------------------------------

export async function POST(request) {
    let browser;
    try {
        const contentType = request.headers.get('content-type');
        console.log(contentType);

        if (contentType.includes('application/json')) {
            const json = await request.json();
            const { url, text } = json;

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
                const pythonResult = await sendImageUrlToPythonService(content, imageUrls);
                const simplifiedPythonResult = await processPythonResult(pythonResult);
                const ID = await saveToFirestore(1, content, simplifiedPythonResult);

                return createResponse(true, {
                    ID, content, pythonResult: simplifiedPythonResult
                });

            } else if (text) {
                console.log('处理文本信息', text);
                const pythonResult = await sendImageUrlToPythonService(text, []);
                const simplifiedPythonResult = await processPythonResult(pythonResult);
                const ID = await saveToFirestore(2, text, simplifiedPythonResult);

                return createResponse(true, { ID, pythonResult: simplifiedPythonResult });
            }

        } else if (contentType.includes('multipart/form-data')) {
            const formData = await request.formData();
            const file = formData.get('file');
            const uploadedFileBuffer = await file.arrayBuffer();
            const uploadedFileName = file.name;

            console.log(`处理上传文件: ${uploadedFileName}`);
            const mimeType = mime.lookup(uploadedFileName);
            console.log(`文件 MIME 类型: ${mimeType}`);

            if (mimeType?.startsWith('image/')) {
                const filePath = path.resolve(__dirname, `../../../../../uploads`, uploadedFileName);
                fs.writeFileSync(filePath, Buffer.from(uploadedFileBuffer));
                console.log(`文件已保存至: ${filePath}`);

                const pythonResult = await sendImageUrlToPythonService('', filePath);
                fs.unlinkSync(filePath);
                const simplifiedPythonResult = await processPythonResult(pythonResult);
                const ID = await saveToFirestore(4, '', simplifiedPythonResult);

                return createResponse(true, { ID, pythonResult: simplifiedPythonResult });

            } else if (mimeType?.startsWith('text/')) {
                const text = await readFileContent(Buffer.from(uploadedFileBuffer));
                const pythonResult = await sendImageUrlToPythonService(text, []);
                const simplifiedPythonResult = await processPythonResult(pythonResult);
                const ID = await saveToFirestore(3, text, simplifiedPythonResult);

                return createResponse(true, { ID, pythonResult: simplifiedPythonResult });

            } else if (mimeType === 'application/vnd.ms-excel' || mimeType === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') {
                const filePath = path.resolve(__dirname, `../../../../../uploads`, uploadedFileName);
                fs.writeFileSync(filePath, Buffer.from(uploadedFileBuffer));

                const pythonProcess = spawn('python', ['schedule_update.py', filePath], {
                    cwd: path.join(__dirname, '../../../../../python')
                });

                pythonProcess.stdout.on('data', (data) => {
                    console.log(`Python 输出: ${data.toString()}`);
                });

                const simplifiedPythonResult = {};
                const ID = await saveToFirestore(5, '', simplifiedPythonResult);

                return createResponse(true, { ID, pythonResult: simplifiedPythonResult });
            }

            return createResponse(false, {}, '不支持的文件类型');
        }

        return createResponse(false, {}, '无法识别的文件类型');

    } catch (error) {
        console.error('处理失败:', error.message);
        return createResponse(false, {}, error.message);
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
