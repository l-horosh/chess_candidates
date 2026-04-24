from flask import Flask, request, jsonify, send_file
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
    body { font-family: 'Georgia', serif; background: radial-gradient(circle at center, #10101a 0%, #000 100%); color: #d4af37; text-align: center; margin: 0; padding: 20px; }
    .box { background: rgba(25, 25, 35, 0.9); border: 1px solid #d4af37; padding: 30px; border-radius: 8px; display: inline-block; box-shadow: 0 0 50px rgba(212, 175, 55, 0.1); margin-top: 20px; }
    h1 { letter-spacing: 6px; text-transform: uppercase; color: #f1c40f; }
    img { max-width: 95%; max-height: 70vh; border: 2px solid #d4af37; object-fit: contain; }
    button { background: #d4af37; color: black; border: none; padding: 15px 45px; cursor: pointer; font-weight: bold; text-transform: uppercase; }
</style>
'''

@app.route('/')
def home():
    return f'<html><head>{CSS_STYLE}</head><body><div class="box"><h1>CogniMetrics AI</h1><p>BPI High-Frequency Analytics</p><form action="/classify" method="post" enctype="multipart/form-data"><input type="file" name="file" accept="image/*,video/*" required style="color:white;"><br><br><button type="submit">Run Analysis</button></form></div></body></html>'

@app.route("/classify", methods=['POST'])
def classify():
    file = request.files['file']
    filename = file.filename.lower()
    try:
        if filename.endswith(('.mp4', '.avi', '.mov')):
            temp_in, temp_out = "in.mp4", "out.mp4"
            file.save(temp_in)
            cap = cv2.VideoCapture(temp_in)
            w, h = int(cap.get(3)), int(cap.get(4))
            fps = cap.get(5)
            if fps == 0 or np.isnan(fps): fps = 30
            
            # Пишем видео с нормальной скоростью (fps/2 т.к. берем каждый 2-й кадр)
            out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps/2, (w, h))
            
            f_idx = 0
            player_timers = [0.0] * 5 
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                f_idx += 1
                
                # Берем КАЖДЫЙ 2-Й КАДР для плавности
                if f_idx % 2 != 0: continue 
                
                # Оптимизация: сжимаем кадр для API до 640px, чтобы не висло
                small_frame = cv2.resize(frame, (640, int(h * (640/w))))
                _, buf = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                
                # Порог уверенности ставим 10% (confidence=10)
                url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}&confidence=10"
                try:
                    resp = requests.post(url, data=base64.b64encode(buf).decode('ascii'), headers={"Content-Type": "application/x-www-form-urlencoded"}).json()
                    predictions = sorted(resp.get('predictions', []), key=lambda p: p['x'])
                    
                    for i, p in enumerate(predictions):
                        if i >= 5: break
                        x, y, pw, ph = int(p['x']*(w/640)), int(p['y']*(h/int(h*(640/w)))), int(p['width']*(w/640)), int(p['height']*(h/int(h*(640/w))))
                        cls = p['class']
                        
                        # Симуляция времени (1 сек видео = 3 мин матча)
                        if cls == "Deep_Focus":
                            player_timers[i] += (2/fps) * 3.0
                            color = (0, 0, 255)
                        else:
                            player_timers[i] = max(0, player_timers[i] - 0.5)
                            color = (255, 0, 255)

                        t = player_timers[i]
                        if t <= 5.0: bpi = max(2, int(12 - (t * 2)))
                        elif t <= 25.0: bpi = int(2 + ((t-5)/20)*43)
                        else: bpi = min(98, int(45 + (t-25)*5))

                        x1, y1, x2, y2 = int(x-pw/2), int(y-ph/2), int(x+pw/2), int(y+ph/2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
                        
                        label = f"{cls} | BPI: {bpi}%"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)
                        ty = y1 - 10 if y1 > 45 else y2 + 35
                        cv2.rectangle(frame, (x1, ty-th-5), (x1+tw, ty+5), (0,0,0), -1)
                        cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2, cv2.LINE_AA)

                    cv2.putText(frame, "LIVE ANALYTICS ACTIVE", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                except: pass
                
                out.write(frame)
                if f_idx > 200: break # Лимит 100 кадров (около 7 сек анализа)
                
            cap.release(); out.release()
            return send_file(temp_out, as_attachment=True)

        else:
            # ДЛЯ ФОТО (confidence=10 чтобы наверняка!)
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            _, buf = cv2.imencode('.jpg', img)
            url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}&confidence=10"
            data = requests.post(url, data=base64.b64encode(buf).decode('ascii'), headers={"Content-Type": "application/x-www-form-urlencoded"}).json()
            stats_html = ""
            for p in data.get('predictions', []):
                x, y, pw, ph = int(p['x']), int(p['y']), int(p['width']), int(p['height'])
                cls = p['class']
                color = (0, 0, 255) if cls == "Deep_Focus" else (255, 0, 255)
                x1, y1, x2, y2 = int(x-pw/2), int(y-ph/2), int(x+pw/2), int(y+ph/2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
                label = f"STATE: {cls}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1.0, 2)
                ty = y1 - 15 if y1 > 50 else y2 + 40
                cv2.rectangle(img, (x1, ty-th-5), (x1+tw, ty+5), (0,0,0), -1)
                cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 2, cv2.LINE_AA)
                stats_html += f"<div class='stats'>{cls} DETECTED</div>"
            _, res_buf = cv2.imencode('.jpg', img)
            return f'<html><head>{CSS_STYLE}</head><body><div class="box"><h2>Static Analysis</h2>{stats_html}<img src="data:image/jpeg;base64,{base64.b64encode(res_buf).decode("utf-8")}"><br><br><a href="/" style="color:#d4af37;">← NEW SCAN</a></div></body></html>'

    except Exception as e: return str(e), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
