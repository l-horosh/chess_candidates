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

def process_and_draw(frame):
    # Уменьшаем для скорости
    height, width = frame.shape[:2]
    scale = 640 / width
    small_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    
    _, buffer = cv2.imencode('.jpg', small_frame)
    img_base64 = base64.b64encode(buffer).decode('ascii')
    
    # Запрос JSON (не картинки!), чтобы самим рисовать цвета
    url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}"
    resp = requests.post(url, data=img_base64, headers={"Content-Type": "application/x-www-form-urlencoded"}).json()
    
    for pred in resp.get('predictions', []):
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        cls = pred['class']
        
        # Выбираем цвет: BGR формат
        if cls == "Deep_Focus":
            color = (0, 0, 255) # Красный
        elif cls == "Normal_Focus":
            color = (128, 0, 128) # Фиолетовый
        else:
            color = (0, 255, 0) # Зеленый для остального
            
        # Рисуем рамку
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        cv2.rectangle(small_frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(small_frame, f"{cls}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, final_buffer = cv2.imencode('.jpg', small_frame)
    return base64.b64encode(final_buffer).decode('utf-8')

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>CogniMetrics AI | Chess Elite</title>
        <style>
            body { font-family: 'Georgia', serif; background-color: #1a1a1a; color: #d4af37; text-align: center; padding: 50px; }
            .box { background: #2c2c2c; border: 2px solid #d4af37; padding: 40px; border-radius: 5px; display: inline-block; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
            h1 { letter-spacing: 3px; text-transform: uppercase; }
            input { background: #3d3d3d; color: white; border: 1px solid #d4af37; padding: 10px; margin: 10px; }
            button { background: #d4af37; color: black; border: none; padding: 15px 30px; cursor: pointer; font-weight: bold; text-transform: uppercase; transition: 0.3s; }
            button:hover { background: #f1c40f; box-shadow: 0 0 10px #d4af37; }
        </style>
    </head>
    <body>
        <div class="box">
            <h1>CogniMetrics AI</h1>
            <p>Advanced Posture & Focus Analysis for Grandmasters</p>
            <form action="/classify" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*,video/*" required><br>
                <button type="submit">Start Grandmaster Analysis</button>
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
    results = []

    try:
        if filename.endswith(('.mp4', '.avi', '.mov')):
            temp = "temp_vid.mp4"
            with open(temp, "wb") as f: f.write(file_bytes)
            cap = cv2.VideoCapture(temp)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Берем 1 кадр в секунду (для 10 секунд будет 10 кадров)
            for i in range(10):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * fps))
                success, frame = cap.read()
                if not success: break
                results.append(process_and_draw(frame))
            cap.release()
            os.remove(temp)
        else:
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            results.append(process_and_draw(img))

        imgs_html = "".join([f'<div style="margin:10px;"><img src="data:image/jpeg;base64,{img}" style="width:100%; border:1px solid #d4af37;"></div>' for img in results])
        
        return f'''
        <body style="background:#1a1a1a; color:#d4af37; font-family:sans-serif; text-align:center; padding:20px;">
            <h2>Analysis Timeline</h2>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap:10px;">
                {imgs_html}
            </div>
            <br><a href="/" style="color:#d4af37; text-decoration:none; border:1px solid #d4af37; padding:10px;">New Analysis</a>
        </body>
        '''
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
