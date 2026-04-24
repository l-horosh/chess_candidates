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
    return f'<html><head>{CSS_STYLE}</head><body><div class="box"><h1>CogniMetrics AI</h1><p>Dual-Mode Blunder Prediction (Exhaustion & Negligence)</p><form action="/classify" method="post" enctype="multipart/form-data"><input type="file" name="file" accept="image/*,video/*" required style="color:white;"><br><br><button type="submit">Start Predictive Analysis</button></form></div></body></html>'

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
            
            out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps/2, (w, h))
            
            f_idx = 0
            # Память: [deep_timer, normal_timer, persistence, coords, last_cls]
            memory = [[0.0, 0.0, 0, None, "Normal_Focus"] for _ in range(5)]
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                f_idx += 1
                if f_idx % 2 != 0: continue 

                small_frame = cv2.resize(frame, (480, int(h * (480/w))))
                _, buf = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                
                url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}&confidence=10"
                try:
                    resp = requests.post(url, data=base64.b64encode(buf).decode('ascii'), headers={"Content-Type": "application/x-www-form-urlencoded"}).json()
                    preds = sorted(resp.get('predictions', []), key=lambda p: p['x'])
                    
                    for m in memory: m[2] = max(0, m[2] - 1)
                    
                    for i, p in enumerate(preds):
                        if i >= 5: break
                        scale_x, scale_y = w/480, h/int(h*(480/w))
                        memory[i][2] = 15 # Инерция 15 кадров
                        memory[i][3] = (int((p['x']-p['width']/2)*scale_x), int((p['y']-p['height']/2)*scale_y), 
                                        int((p['x']+p['width']/2)*scale_x), int((p['y']+p['height']/2)*scale_y))
                        memory[i][4] = p['class']

                    for i, (deep_t, norm_t, persistence, coords, cls) in enumerate(memory):
                        if persistence > 0 and coords:
                            # 1 реальная сек видео = 3 минуты матча
                            sim_step = (2/fps) * 3.0 
                            
                            if cls == "Deep_Focus":
                                memory[i][0] += sim_step # Растет время напряжения
                                memory[i][1] = max(0, memory[i][1] - (sim_step * 2)) # Падает время расслабленности
                                color = (0, 0, 255)
                            else:
                                memory[i][1] += sim_step # Растет время расслабленности
                                memory[i][0] = max(0, memory[i][0] - (sim_step * 2)) # Падает напряжение
                                color = (255, 0, 255)

                            # --- НОВАЯ СЛОЖНАЯ МАТЕМАТИКА BPI ---
                            d_t, n_t = memory[i][0], memory[i][1]
                            
                            # Риск от переутомления (Deep Focus)
                            bpi_deep = 0
                            if d_t > 25: bpi_deep = (d_t - 25) * 6
                            elif d_t > 5: bpi_deep = (d_t - 5) * 1.5
                            else: bpi_deep = - (5 - d_t) # Риск падает, пока фокус свежий
                            
                            # Риск от потери концентрации (Normal Focus)
                            bpi_norm = 0
                            if n_t > 8: # Если расслаблен больше 8 минут симуляции
                                bpi_norm = (n_t - 8) * 8 # Резкий рост риска "зевка"
                            
                            bpi = min(98, max(2, 12 + int(bpi_deep + bpi_norm)))
                            # ------------------------------------

                            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 4, cv2.LINE_AA)
                            label = f"{cls} | BPI: {bpi}%"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)
                            ty = coords[1]-10 if coords[1]>40 else coords[3]+35
                            cv2.rectangle(frame, (coords[0], ty-th-5), (coords[0]+tw, ty+5), (0,0,0), -1)
                            cv2.putText(frame, label, (coords[0], ty), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2, cv2.LINE_AA)

                    cv2.putText(frame, "PREDICTIVE ENGINE: ACTIVE", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                except: pass
                
                out.write(frame)
                if f_idx > 600: break 
                
            cap.release(); out.release()
            return send_file(temp_out, as_attachment=True)

        else:
            # ФОТО (без BPI, только статус)
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
