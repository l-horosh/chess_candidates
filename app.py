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
    img { max-width: 95%; max-height: 70vh; border: 2px solid #d4af37; object-fit: contain; box-shadow: 0 10px 30px rgba(0,0,0,0.8); }
    button { background: #d4af37; color: black; border: none; padding: 15px 45px; cursor: pointer; font-weight: bold; font-size: 16px; transition: 0.4s; text-transform: uppercase; }
    button:hover { background: #fff; box-shadow: 0 0 20px #d4af37; }
    .stats { font-family: 'Courier New', monospace; font-size: 20px; padding: 10px; border: 1px dashed #d4af37; margin-bottom: 20px; }
</style>
'''

@app.route('/')
def home():
    return f'<html><head>{CSS_STYLE}</head><body><div class="box"><h1>CogniMetrics AI</h1><p>Time-Series Posture Analytics</p><form action="/classify" method="post" enctype="multipart/form-data"><input type="file" name="file" accept="image/*,video/*" required style="color:white;"><br><br><button type="submit">Analyze Player State</button></form></div></body></html>'

@app.route("/classify", methods=['POST'])
def classify():
    file = request.files['file']
    filename = file.filename.lower()
    
    try:
        # ==========================================
        # ОБРАБОТКА ВИДЕО
        # ==========================================
        if filename.endswith(('.mp4', '.avi', '.mov')):
            temp_in, temp_out = "in.mp4", "out.mp4"
            file.save(temp_in)
            cap = cv2.VideoCapture(temp_in)
            w, h = int(cap.get(3)), int(cap.get(4))
            fps = cap.get(5)
            if fps == 0 or np.isnan(fps): fps = 30
            
            out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps/5, (w, h))
            
            f_idx = 0
            # Массив для хранения времени Deep Focus индивидуально для каждого игрока (до 5 человек)
            player_timers = [0.0] * 5 
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                f_idx += 1
                if f_idx % 5 != 0: continue 
                
                real_seconds_passed = 5 / fps
                simulated_minutes_added = real_seconds_passed * 3.0 
                
                _, buf = cv2.imencode('.jpg', frame)
                url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}&confidence=15"
                try:
                    resp = requests.post(url, data=base64.b64encode(buf).decode('ascii'), headers={"Content-Type": "application/x-www-form-urlencoded"}).json()
                    
                    # Сортируем предсказания слева направо (по координате X), чтобы точно знать, где Player 1, а где Player 2
                    predictions = sorted(resp.get('predictions', []), key=lambda p: p['x'])
                    
                    for i, p in enumerate(predictions):
                        if i >= 5: break # Ограничение на 5 игроков
                        
                        x, y, pw, ph = int(p['x']), int(p['y']), int(p['width']), int(p['height'])
                        cls = p['class']
                        
                        # Обновляем таймер конкретного игрока
                        if cls == "Deep_Focus":
                            player_timers[i] += simulated_minutes_added
                            color = (0, 0, 255) # Красный
                        else:
                            player_timers[i] = max(0, player_timers[i] - (simulated_minutes_added * 2))
                            color = (255, 0, 255) # Фиолетовый

                        t = player_timers[i]
                        # Логика BPI для конкретного игрока
                        if t == 0: bpi = 12 
                        elif t <= 5.0: bpi = max(2, int(12 - (t * 2)))
                        elif t <= 25.0: bpi = int(2 + ((t - 5) / 20) * 43)
                        else: bpi = min(98, int(45 + ((t - 25) * 5)))

                        # Отрисовка рамки
                        x1, y1 = int(x-pw/2), int(y-ph/2)
                        x2, y2 = int(x+pw/2), int(y+ph/2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
                        
                        # Отрисовка ПРОФЕССИОНАЛЬНОГО текста (с черной подложкой)
                        label = f"{cls} | BPI: {bpi}%"
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)
                        
                        # Если текст не влезает сверху, рисуем его снизу рамки
                        text_y = y1 - 10 if y1 > 40 else y2 + 30
                        
                        # Черный фон для текста (чтобы не было каши)
                        cv2.rectangle(frame, (x1, text_y - text_height - 5), (x1 + text_width, text_y + 5), (0, 0, 0), -1)
                        # Сам текст
                        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2, cv2.LINE_AA)

                    # Общая плашка скорости симуляции (только одна, в углу)
                    cv2.putText(frame, "SIMULATION SPEED: 1 sec = 3 mins", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    
                except: pass
                
                out.write(frame)
                if f_idx > 150: break 
                
            cap.release(); out.release()
            return send_file(temp_out, as_attachment=True)

        # ==========================================
        # ОБРАБОТКА ФОТО (Без изменений, статика)
        # ==========================================
        else:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            _, buf = cv2.imencode('.jpg', img)
            url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}&confidence=15"
            data = requests.post(url, data=base64.b64encode(buf).decode('ascii'), headers={"Content-Type": "application/x-www-form-urlencoded"}).json()

            stats_html = ""
            for p in data.get('predictions', []):
                x, y, pw, ph = int(p['x']), int(p['y']), int(p['width']), int(p['height'])
                cls = p['class']
                
                color = (0, 0, 255) if cls == "Deep_Focus" else (255, 0, 255)
                c_css = "#ff4d4d" if cls == "Deep_Focus" else "#d633ff"
                
                x1, y1 = int(x-pw/2), int(y-ph/2)
                x2, y2 = int(x+pw/2), int(y+ph/2)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
                
                # Текст с подложкой для фото
                label = f"State: {cls}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.9, 2)
                text_y = y1 - 10 if y1 > 40 else y2 + 30
                cv2.rectangle(img, (x1, text_y - th - 5), (x1 + tw, text_y + 5), (0, 0, 0), -1)
                cv2.putText(img, label, (x1, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.9, color, 2, cv2.LINE_AA)
                
                stats_html += f"<div class='stats' style='color:{c_css};'>DETECTED POSTURE: {cls}</div>"

            _, res_buf = cv2.imencode('.jpg', img)
            return f'<html><head>{CSS_STYLE}</head><body><div class="box"><h2>Static Image Analysis</h2><p style="font-size:14px; color:#aaa;">*BPI calculation requires live video feed</p>{stats_html}<img src="data:image/jpeg;base64,{base64.b64encode(res_buf).decode("utf-8")}"><br><br><a href="/" style="color:#d4af37;">← NEW SCAN</a></div></body></html>'

    except Exception as e: return str(e), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
