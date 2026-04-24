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
    button { background: #d4af37; color: black; border: none; padding: 15px 45px; cursor: pointer; font-weight: bold; text-transform: uppercase; }
</style>
'''

@app.route('/')
def home():
    return f'<html><head>{CSS_STYLE}</head><body><div class="box"><h1>CogniMetrics AI</h1><p>Demo Simulation Mode (Max 40 sec)</p><form action="/classify" method="post" enctype="multipart/form-data"><input type="file" name="file" accept="image/*,video/*" required style="color:white;"><br><br><button type="submit">Start Analysis</button></form></div></body></html>'

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
            
            # Увеличиваем стабильность: берем fps/2
            out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps/2, (w, h))
            
            f_idx = 0
            # Память: [таймер, инерция, координаты, класс]
            memory = [[0.0, 0, None, "Normal_Focus"] for _ in range(5)]
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                f_idx += 1
                if f_idx % 2 != 0: continue 

                # Сжимаем сильнее для скорости
                small_frame = cv2.resize(frame, (480, int(h * (480/w))))
                _, buf = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                
                url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}&confidence=10"
                try:
                    resp = requests.post(url, data=base64.b64encode(buf).decode('ascii'), headers={"Content-Type": "application/x-www-form-urlencoded"}).json()
                    preds = sorted(resp.get('predictions', []), key=lambda p: p['x'])
                    
                    for m in memory: m[1] = max(0, m[1] - 1)
                    
                    for i, p in enumerate(preds):
                        if i >= 5: break
                        scale_x, scale_y = w/480, h/int(h*(480/w))
                        memory[i][1] = 20 # УВЕЛИЧИЛИ ИНЕРЦИЮ ДО 20 КАДРОВ
                        memory[i][2] = (int((p['x']-p['width']/2)*scale_x), int((p['y']-p['height']/2)*scale_y), 
                                        int((p['x']+p['width']/2)*scale_x), int((p['y']+p['height']/2)*scale_y))
                        memory[i][3] = p['class']

                    for i, (timer, persistence, coords, cls) in enumerate(memory):
                        if persistence > 0 and coords:
                            if cls == "Deep_Focus":
                                memory[i][0] += (2/fps) * 5.0 # Ускорили накопление BPI для демо
                                color = (0, 0, 255)
                            else:
                                memory[i][0] = max(0, memory[i][0] - 0.4)
                                color = (255, 0, 255)

                            t = memory[i][0]
                            # BPI Logic
                            if t <= 5.0: bpi = max(2, int(12 - t))
                            elif t <= 20.0: bpi = int(2 + ((t-5)/15)*43)
                            else: bpi = min(98, int(45 + (t-20)*6))

                            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 4, cv2.LINE_AA)
                            label = f"{cls} | BPI: {bpi}%"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)
                            ty = coords[1]-10 if coords[1]>40 else coords[3]+35
                            cv2.rectangle(frame, (coords[0], ty-th-5), (coords[0]+tw, ty+5), (0,0,0), -1)
                            cv2.putText(frame, label, (coords[0], ty), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2, cv2.LINE_AA)

                    cv2.putText(frame, "PREDICTIVE BPI ANALYTICS", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                except: pass
                
                out.write(frame)
                # ЛИМИТ 600 КАДРОВ = ~40 СЕКУНД ГОТОВОГО ВИДЕО. БОЛЬШЕ RENDER НЕ ВЫДЕРЖИТ!
                if f_idx > 600: break 
                
            cap.release(); out.release()
            return send_file(temp_out, as_attachment=True)

        else:
            # ФОТО (без изменений)
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            _, buf = cv2.imencode('.jpg', img)
            url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}&confidence=10"
            data = requests.post(url, data=base64.b64encode(buf).decode('ascii'), headers={"Content-Type": "application/x-www-form-urlencoded"}).json()
            for p in data.get('predictions', []):
                x, y, pw, ph = int(p['x']), int(p['y']), int(p['width']), int(p['height'])
                cls = p['class']
                color = (0, 0, 255) if cls == "Deep_Focus" else (255, 0, 255)
                x1, y1, x2, y2 = int(x-pw/2), int(y-ph/2), int(x+pw/2), int(y+ph/2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
                label = f"STATE: {cls}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1.0, 2)
                ty = y1-15 if y1>50 else y2+40
                cv2.rectangle(img, (x1, ty-th-5), (x1+tw, ty+5), (0,0,0), -1)
                cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 2, cv2.LINE_AA)
            _, res_buf = cv2.imencode('.jpg', img)
            return f'<html><head>{CSS_STYLE}</head><body><div class="box"><img src="data:image/jpeg;base64,{base64.b64encode(res_buf).decode("utf-8")}"><br><br><a href="/" style="color:#d4af37;">← BACK</a></div></body></html>'

    except Exception as e: return str(e), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
