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

async function UrlContent(url) {
    console.log(`处理 URL: ${url}`);
    const browser = await puppeteer.launch({ headless: true });
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    
    // 获取页面文本内容
    const content = await page.evaluate(() => document.body.innerText);
    
    // 获取所有图片的 URL
    const imageUrls = await page.evaluate(() =>
        Array.from(document.querySelectorAll('img')).map(img => img.src)
    );
    
    console.log(`找到图片 URL: ${imageUrls}`);
    await browser.close();
    
    return { content, imageUrls };
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
    // 直接构造返回对象
    const matches = (pythonResult.matched_keywords || []).map(item => ({
        MatchKeyword: item.keyword || '无关键词',
        MatchType: item.type || '无类型',
        Remind: item.Remind || '',
        Prevent: item.Prevent || '',
    }));

    const simplifiedPythonResult = {
        FraudResult: pythonResult.result || '未检测到',
        FraudRate: pythonResult.FraudRate || 0,
        Match: matches,
    };

    // 确保返回的对象是标准的 JSON 格式
    return JSON.parse(JSON.stringify(simplifiedPythonResult));
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
                const result = await UrlContent(url);
                const pythonResult = await sendImageUrlToPythonService(result.content+url, result.imageUrls);
                const simplifiedPythonResult = await processPythonResult(pythonResult);
                const ID = await saveToFirestore(1, result.content, simplifiedPythonResult);

                return createResponse(true, {
                    ID,  pythonResult: simplifiedPythonResult
                });

            } 
            
            else if (text) {
                // 檢查 text 裡是否包含 URL
                const urlPattern = /(https?:\/\/[^\s]+)/g;
                const containsUrl = urlPattern.test(text);
                const urls = text.match(urlPattern); 
                let pythonResult;
                let allContent = text;          // 用于收集所有文本内容
                let allImageUrls = [];        // 用于收集所有图片链接
                if (containsUrl) {
                    console.log('处理文本信息', text);
                    for (const url of urls) {
                        const result = await UrlContent(url); // 每个 URL 传入 scrapeUrlContent
                        allContent += result.content + '\n';       
                        allImageUrls = allImageUrls.concat(result.imageUrls);  // 合并图片链接数组
                    }
                    pythonResult = await sendImageUrlToPythonService(allContent, allImageUrls);
                    
                } else {
                    console.log('处理文本信息', text);
                    pythonResult = await sendImageUrlToPythonService(text, []);
                    
                }
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

           
            if (mimeType.trim() === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') {
                console.log("MIME type matches after trimming.");
            } else {
                console.log("MIME type does not match.");
            }
            
            if (mimeType?.startsWith('image/')) {
                const filePath = path.resolve(__dirname, `../../../../../uploads`, uploadedFileName);
                fs.writeFileSync(filePath, Buffer.from(uploadedFileBuffer));
                console.log(`文件已保存至: ${filePath}`);

                const pythonResult = await sendImageUrlToPythonService('', [filePath]);
                fs.unlinkSync(filePath);
                const simplifiedPythonResult = await processPythonResult(pythonResult);
                const ID = await saveToFirestore(4, '', simplifiedPythonResult);

                return createResponse(true, { ID, pythonResult: simplifiedPythonResult });

            } else if (mimeType?.startsWith('text/plain')) {
                const text = await readFileContent(Buffer.from(uploadedFileBuffer));
                const pythonResult = await sendImageUrlToPythonService(text, []);
                const simplifiedPythonResult = await processPythonResult(pythonResult);
                const ID = await saveToFirestore(3, text, simplifiedPythonResult);

                return createResponse(true, { ID, pythonResult: simplifiedPythonResult });

            } else if (mimeType === 'application/vnd.ms-excel' || mimeType.trim() === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' || mimeType === 'text/csv') {
                const filePath = path.resolve(__dirname, `../../../../../uploads`, uploadedFileName);
                fs.writeFileSync(filePath, Buffer.from(uploadedFileBuffer));

                const pythonPath = 'python'; // 或 'python3'，或者写上绝对路径
                const pythonProcess = spawn(pythonPath, ['schedule_update.py', filePath], {
                    cwd: path.join(__dirname, '../../../../../python'),
                    stdio: 'pipe'  // 确保子进程输出通过管道传回
                });
                pythonProcess.stdout.on('data', (data) => {
                    console.log(`Python Output: ${data.toString()}`);
                });
                
                // 监听错误输出
                pythonProcess.stderr.on('data', (data) => {
                    console.error(`Python Error: ${data.toString()}`);
                });
                
                // 监听脚本结束
                pythonProcess.on('close', (code) => {
                    console.log(`Python script exited with code ${code}`);
                });

                
                pythonProcess.stderr.on('data', (data) => {
                    console.error(`Python 错误: ${data.toString()}`);
                });
                return createResponse(true, {}, '上傳成功');
            
            }else{
                return createResponse(false, {}, '不支持的文件类型');
                }
            
        }
        else{return createResponse(false, {}, '无法识别的文件类型');
    }
        

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
