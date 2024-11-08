const express = require('express');
const cors = require('cors');
const axios = require('axios');
const cheerio = require('cheerio');

const app = express();
const port = 917;

app.use(cors());

let cachedArticles = [];  // 用來緩存爬取的資料

async function getPage() {
    try {
        const res = await axios.get("https://tw.news.yahoo.com/tag/%E8%A9%90%E9%A8%99");
        const $ = cheerio.load(res.data);
        const articles = [];

        // 遍歷並抓取前 6 筆資料的標題和簡介
        $('div.Cf').each((index, element) => {
            if (index < 6) {  // 只抓取前 6 筆資料
                const title = $(element).find('h3.Mb\\(5px\\) a').text();
                const description = $(element).find('p.Mt\\(8px\\)').text();
                const link = `https://tw.news.yahoo.com${$(element).find('h3.Mb\\(5px\\) a').attr('href')}`;
                const img = $(element).find('img').attr('src');
                articles.push({ title, description, link, img });
            } else {
                return false;
            }
        });

        cachedArticles = articles;  // 將爬取的資料儲存到緩存變數中
        return articles;
    } catch (error) {
        console.error('Error fetching page:', error);
        return [];
    }
}

// 在伺服器啟動時自動爬取一次資料
(async () => {
    await getPage();  // 伺服器啟動時預先抓取一次資料
    app.listen(port, () => {
        console.log(`Server is running at http://localhost:${port}`);
    });
})();

// 路由：處理前端請求並返回已經爬取的資料
app.get('/', (req, res) => {
    res.json(cachedArticles);  // 直接返回緩存中的資料
});
