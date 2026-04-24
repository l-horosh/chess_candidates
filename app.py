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
    body { 
        font-family: 'Georgia', serif; 
        background: radial-gradient(circle at center, #10101a 0%, #000 100%);
        color: #d4af37; text-align: center; margin: 0; padding: 20px;
    }
    .box { 
        background: rgba(25, 25, 35, 0.9); border: 1px solid #d4af37; 
        padding: 30px; border-radius: 8px; display: inline-block; 
        box-shadow: 0 0 50px rgba(212, 175, 55, 0.1); margin-top: 20px;
    }
    h1 { letter-spacing: 6px; text-transform: uppercase; color: #f1c40f; }
    img { 
        max-width: 95%; max-height: 70vh; border: 2px solid #d4af37; 
        object-fit: contain; box-shadow: 0 10px 30px rgba(0,0,0,0.8);
    }
    button { 
        background: #d4af37; color: black; border: none; padding: 15px 45px; 
        cursor: pointer; font-weight: bold; border-radius: 2px; 
        font-size: 16px; transition: 0.4s; text-transform: uppercase;
    }
    button:hover { background: #fff; box-shadow: 0 0 20px #d4af37; }
</style>
'''

def get_bpi(cls, conf):
    # Логика: если Deep Focus, риск сразу высокий. Если Normal - умеренный.
    if cls == "Deep_Focus":
        return min(98, int(70 + (conf * 25)))
    return max(12, int(conf * 40))

@app.route('/')
def home():
    return f'<html><head>{CSS_STYLE}</head><body><div class="box"><h1>CogniMetrics AI</h1><p>BPI: Individual Player Risk Analytics</p><form action="/classify" method="post" enctype="multipart/form-data"><input type="file" name="file" accept="image/*,video/*" required style="color:white;"><br><br><button type="submit">Analyze Live Feed</button></form></div></body></html>'

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
            out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(5)/5, (w, h))
            
            f_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                f_idx += 1
                if f_idx % 5 != 0: continue # Ускоряем обработку
                
                _, buf = cv2.imencode('.jpg', frame)
                url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}&confidence=15"
                resp = requests.post(url, data=base64.b64encode(buf).decode('ascii'), headers={"Content-Type": "application/x-www-form-urlencoded"}).json()
                
                for p in resp.get('predictions', []):
                    x, y, pw, ph = int(p['x']), int(p['y']), int(p['width']), int(p['height'])
                    cls, conf = p['class'], p['confidence']
                    bpi = get_bpi(cls, conf)
                    
                    color = (0, 0, 255) if cls == "Deep_Focus" else (255, 0, 255)
                    x1, y1 = int(x - pw/2), int(y - ph/2)
                    x2, y2 = int(x + pw/2), int(y + ph/2)
                    
                    # Четкая рамка
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
                    
                    # Исправляем вылет текста за границы (проверка y1)
                    text_y = y1 - 15 if y1 > 50 else y1 + 40
                    label = f"{cls} | BPI: {bpi}%"
                    cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 2, cv2.LINE_AA)
                
                out.write(frame)
                if f_idx > 200: break
                
            cap.release(); out.release()
            return send_file(temp_out, as_attachment=True)

        else:
            # ОБРАБОТКА ФОТО
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            _, buf = cv2.imencode('.jpg', img)
            url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}&confidence=15"
            data = requests.post(url, data=base64.b64encode(buf).decode('ascii'), headers={"Content-Type": "application/x-www-form-urlencoded"}).json()

            stats_html = ""
            for p in data.get('predictions', []):
                x, y, pw, ph = int(p['x']), int(p['y']), int(p['width']), int(p['height'])
                cls, conf = p['class'], p['confidence']
                bpi = get_bpi(cls, conf)
                
                color = (0, 0, 255) if cls == "Deep_Focus" else (255, 0, 255)
                x1, y1 = int(x - pw/2), int(y - ph/2)
                x2, y2 = int(x + pw/2), int(y + ph/2)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
                text_y = y1 - 15 if y1 > 50 else y1 + 40
                cv2.putText(img, f"{cls} {bpi}%", (x1, text_y), cv2.FONT_HERSHEY_COMPLEX, 1.2, color, 2, cv2.LINE_AA)
                
                c_css = "#ff4d4d" if cls == "Deep_Focus" else "#d633ff"
                stats_html += f"<div class='stats' style='color:{c_css};'>PLAYER: {cls} | BLUNDER RISK: {bpi}%</div>"

            _, res_buf = cv2.imencode('.jpg', img)
            return f'<html><head>{CSS_STYLE}</head><body><div class="box"><h2>Cognitive Report</h2>{stats_html}<img src="data:image/jpeg;base64,{base64.b64encode(res_buf).decode("utf-8")}"><br><br><a href="/" style="color:#d4af37;">← NEW SCAN</a></div></body></html>'

    except Exception as e: return str(e), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
