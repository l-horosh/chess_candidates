from flask import Flask, request, jsonify
import requests
import base64
import os
import cv2
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 

ROBOFLOW_API_KEY = "zipFoxXwSowFhMRI79Sv"
PROJECT_NAME = "candidates-chess" 
VERSION = "2"

CSS_STYLE = '''
<style>
    body { 
        font-family: 'Segoe UI', sans-serif; 
        background: radial-gradient(circle at center, #1a1a2e 0%, #0f0f1a 100%);
        color: #d4af37; 
        text-align: center; 
        margin: 0;
        padding: 20px;
        min-height: 100vh;
    }
    .box { 
        background: rgba(30, 30, 46, 0.9); 
        border: 2px solid #d4af37; 
        padding: 40px; 
        border-radius: 15px; 
        display: inline-block; 
        box-shadow: 0 0 40px rgba(212, 175, 55, 0.2);
        margin-top: 30px;
    }
    h1 { letter-spacing: 4px; text-transform: uppercase; color: #f1c40f; text-shadow: 2px 2px #000; }
    .stats {
        background: #000;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px dashed #d4af37;
        font-size: 24px;
    }
    img { 
        max-width: 90%; 
        max-height: 60vh;
        border: 3px solid #d4af37; 
        border-radius: 10px;
        object-fit: contain;
    }
    button { 
        background: linear-gradient(90deg, #d4af37 0%, #f1c40f 100%);
        color: black; border: none; padding: 15px 40px; 
        cursor: pointer; font-weight: bold; border-radius: 30px; 
        font-size: 18px; transition: 0.3s;
    }
    button:hover { transform: scale(1.05); box-shadow: 0 0 15px #f1c40f; }
</style>
'''

@app.route('/')
def home():
    return f'''
    <html>
    <head>{CSS_STYLE}</head>
    <body>
        <div class="box">
            <h1>CogniMetrics AI</h1>
            <p>Predictive Behavioral Analytics Dashboard</p>
            <form action="/classify" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*,video/*" required style="margin: 20px 0; color: white;"><br>
                <button type="submit">Analyze Player State</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route("/classify", methods=['POST'])
def classify():
    file = request.files['file']
    filename = file.filename.lower()
    file_bytes = np.frombuffer(file.read(), np.uint8)
    
    try:
        # 1. Читаем картинку или 1 кадр из видео
        if filename.endswith(('.mp4', '.avi', '.mov')):
            temp = "temp.mp4"
            with open(temp, "wb") as f: f.write(file_bytes)
            cap = cv2.VideoCapture(temp)
            success, frame = cap.read()
            cap.release()
            os.remove(temp)
            img = frame
        else:
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 2. Отправляем картинку в Roboflow (просим JSON)
        _, buffer = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(buffer).decode('ascii')
        
        # Обрати внимание: мы убрали format=image. Теперь получаем умный JSON!
        url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}&confidence=15"
        response = requests.post(url, data=img_b64, headers={"Content-Type": "application/x-www-form-urlencoded"})
        data = response.json()

        # 3. Рисуем рамки и считаем проценты САМИ
        predictions = data.get('predictions', [])
        stats_html = ""

        if not predictions:
            stats_html = "<div class='stats' style='color: #e74c3c;'>No player detected. Confidence < 15%</div>"
        else:
            for pred in predictions:
                x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                cls = pred['class']
                conf = int(pred['confidence'] * 100) # Переводим в проценты!
                
                # Задаем цвета
                if cls == "Deep_Focus":
                    color = (0, 0, 255) # Красный для OpenCV (BGR)
                    color_css = "#ff4d4d"
                else:
                    color = (255, 0, 255) # Фиолетовый
                    color_css = "#d633ff"

                # Рисуем саму рамку
                cv2.rectangle(img, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), color, 4)
                
                # Пишем текст НА картинке
                label = f"{cls}: {conf}%"
                cv2.putText(img, label, (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                
                # Добавляем красивую панель с процентами НАД картинкой
                stats_html += f"<div class='stats' style='color: {color_css};'>Detected State: <b>{cls}</b> | Confidence: <b>{conf}%</b></div>"

        # 4. Кодируем готовую картинку с нашими рамками обратно для HTML
        _, result_buffer = cv2.imencode('.jpg', img)
        res_img_b64 = base64.b64encode(result_buffer).decode('utf-8')
        
        return f'''
        <html>
        <head>{CSS_STYLE}</head>
        <body>
            <div class="box">
                <h2>Live Feed Analysis</h2>
                {stats_html}
                <img src="data:image/jpeg;base64,{res_img_b64}">
                <br><br>
                <a href="/" style="color: #d4af37; text-decoration: none; font-weight: bold; border: 1px solid #d4af37; padding: 10px 20px; border-radius: 5px;">← ANALYZE NEW FEED</a>
            </div>
        </body>
        </html>
        '''
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
