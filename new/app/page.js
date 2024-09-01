"use client"; 
import { useState } from 'react';

export default function Home() {
    const [url, setUrl] = useState('');
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleSubmit = async (event) => {
        event.preventDefault();
        setResult(null);
        setError(null);

        try {
            const response = await fetch('/api/fetch-content', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url }),
            });

            if (!response.ok) {
                throw new Error('Failed to fetch content');
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError(err.message);
        }
    };

    return (
        <div>
            <h1>網址檢測</h1>
            <form onSubmit={handleSubmit}>
                <label htmlFor="url">輸入網址:</label>
                <input
                    type="text"
                    id="url"
                    name="url"
                    required
                    placeholder="https://example.com"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                />
                <button type="submit">檢測</button>
            </form>
            <div id="results">
                {error && (
                    <div>
                        <p style={{ color: 'red' }}>錯誤: {error}</p>
                    </div>
                )}
                {result && !error && (
                    <div>
                        <p>檢測完成</p>
                        <h3>抓取的內容:</h3>
                        <p>{result.content || '無內容'}</p>
                        <h3>OCR 識別文字:</h3>
                        <p>{result.ocrText || '無文字識別'}</p>
                        <h3>Python 服務結果:</h3>
                        <p>{result.pythonResult ? result.pythonResult.result : '無結果'}</p>
                    </div>
                )}
            </div>
        </div>
    );
}
