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
    if (!/^https?:\/\//.test(url)) {
        url = `https://${url}`; // 补全为 https://
    }
    console.log(`处理 URL: ${url}`);

    await page.goto(url, { waitUntil: 'domcontentloaded' });

    // 检查是否有 <article> 元素并根据情况提取内容
    const { content, imageUrls } = await page.evaluate(() => {
        let targetElement = document.querySelector('article');
        if (!targetElement) {
            targetElement = document.body; // 如果没有 <article>，默认抓取整个 body
        }

        // 提取文本内容
        const content = targetElement.innerText;

        // 提取图片 URL
        const imageUrls = Array.from(targetElement.querySelectorAll('img'))
            .map(img => img.src)
            .filter(src => {
                // 排除包含 "icon" 的 URL
                if (src.includes('icon')) return false;
                if (src.includes('title')) return false;
                if (src.includes('logo')) return false;
                if (src.endsWith('.svg')) return false;
                // 排除分辨率过小的缩略图 (例如宽度或高度小于50px)
                const imgElement = document.querySelector(`img[src="${src}"]`);
                if (imgElement && (imgElement.width < 50 || imgElement.height < 50)) return false;
                return true;
            });

        return { content, imageUrls };
    });

    console.log(`提取的内容: ${content}`);
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
        Emotion: pythonResult.Emotion || '',
    };

    // 确保返回的对象是标准的 JSON 格式
    return JSON.parse(JSON.stringify(simplifiedPythonResult));
}


//--------------------------------------------------------------------------------------------------
//Cors不允許問題處理
const corsHeaders = {
    'Access-Control-Allow-Origin': '*', // 前端地址
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',  // 允许的 HTTP 方法
    'Access-Control-Allow-Headers': 'Content-Type',  // 允许的头部
    'Access-Control-Allow-Credentials': 'true', // 如果需要支持 Cookies 或认证
  };
  
  export async function OPTIONS() {
    return new NextResponse(null, {
      status: 204,
      headers: corsHeaders,
    });
  }

//--------------------------------------------------------------

export async function POST(request) {
    
    let browser;
    try {

        const contentType = request.headers.get('content-type');
        console.log(contentType);

        
        if (contentType.includes('application/json')) {
            const json = await request.json();
            const { url, text, from } = json;

            if (url) {
                const result = await UrlContent(url);
                const pythonResult = await sendImageUrlToPythonService(result.content+url, result.imageUrls);
                const simplifiedPythonResult = await processPythonResult(pythonResult);
                if (from !== undefined) {
                    console.log(`請求來自: ${from}`);
                  }
                  let ID = null;
                  if (from === undefined) {
                    ID = await saveToFirestore(1, result.content, simplifiedPythonResult);
                    console.log(`已儲存到資料庫`);
                  }
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
                if (from !== undefined) {
                    console.log(`請求來自: ${from}`);
                  }
                  let ID = null;
                  if (from === undefined) {
                    ID = await saveToFirestore(2, text, simplifiedPythonResult);
                    console.log(`已儲存到資料庫`);
                  }
            
                return createResponse(true, {
                    ID, 
                    pythonResult: simplifiedPythonResult
                },  corsHeaders);  // 这里加上corsHeaders
            }
            

        } else if (contentType.includes('multipart/form-data')) {
            const formData = await request.formData();
            const from = formData.get('from');         // 拿來源
            const files = formData.getAll('files[]'); // 取得所有檔案
            let uploadedFileBuffer, uploadedFileName, mimeType;
            let filePaths = []; // 用來儲存所有圖片的路徑

            // 先處理所有文件
            for (const file of files) {
                uploadedFileBuffer = await file.arrayBuffer();
                uploadedFileName = file.name;

                console.log(`处理上传文件: ${uploadedFileName}`);
                mimeType = mime.lookup(uploadedFileName);
                console.log(`文件 MIME 类型: ${mimeType}`);

                if (mimeType?.startsWith('image/')) {
                    const filePath = path.resolve(__dirname, `../../../../../uploads`, uploadedFileName);
                    fs.writeFileSync(filePath, Buffer.from(uploadedFileBuffer));
                    console.log(`文件已保存至: ${filePath}`);
                    filePaths.push(filePath);

                }
            }

            // 只對圖片發送到 Python 服務
            if (mimeType?.startsWith('image/')) {
                console.log("filePath",filePaths)

                const pythonResult = await sendImageUrlToPythonService('', filePaths);
                const content = pythonResult?.content || '';  // 提取 content 欄位
                for (const filePath of filePaths) {
                fs.unlinkSync(filePath);
                }
                const simplifiedPythonResult = await processPythonResult(pythonResult);
                console.log(`從 Python 獲得的內容: ${content}`);
                let ID = null;
                if (from == null) {
                    ID = await saveToFirestore(4, content, simplifiedPythonResult);
                    console.log(`已儲存到資料庫`);
                  }
                else{
                    console.log(`請求來自: ${from}`);
                }
                

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
