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
        background: radial-gradient(circle at center, #1a1a2e 0%, #05050a 100%);
        color: #d4af37; 
        text-align: center; 
        margin: 0; padding: 20px; min-height: 100vh;
    }
    .box { 
        background: rgba(20, 20, 30, 0.95); 
        border: 2px solid #d4af37; 
        padding: 40px; border-radius: 12px; 
        display: inline-block; 
        box-shadow: 0 0 40px rgba(212, 175, 55, 0.15);
        margin-top: 30px;
    }
    h1 { letter-spacing: 4px; text-transform: uppercase; color: #f1c40f; text-shadow: 1px 1px #000; }
    .stats {
        background: #000; padding: 15px; border-radius: 8px;
        margin-bottom: 20px; border: 1px solid #d4af37; font-size: 22px;
        font-family: 'Courier New', monospace; font-weight: bold;
    }
    img { 
        max-width: 90%; max-height: 60vh;
        border: 2px solid #d4af37; border-radius: 5px;
        object-fit: contain; box-shadow: 0 5px 15px rgba(0,0,0,0.5);
    }
    button { 
        background: linear-gradient(90deg, #d4af37 0%, #f1c40f 100%);
        color: black; border: none; padding: 15px 40px; 
        cursor: pointer; font-weight: bold; border-radius: 30px; 
        font-size: 18px; transition: 0.3s; text-transform: uppercase;
        font-family: 'Georgia', serif;
    }
    button:hover { transform: scale(1.05); box-shadow: 0 0 20px rgba(241, 196, 15, 0.5); }
</style>
'''

@app.route('/')
def home():
    return f'''
    <html><head>{CSS_STYLE}</head><body>
        <div class="box">
            <h1>CogniMetrics AI</h1>
            <p style="font-style: italic; color: #a0a0a0;">Grandmaster Blunder Probability Index (BPI)</p>
            <form action="/classify" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*,video/*" required style="margin: 20px 0; color: white;"><br>
                <button type="submit">Run BPI Analysis</button>
            </form>
        </div>
    </body></html>
    '''

@app.route("/classify", methods=['POST'])
def classify():
    file = request.files['file']
    filename = file.filename.lower()
    
    try:
        # === ВИДЕО: РАСЧЕТ BPI В ДИНАМИКЕ ===
        if filename.endswith(('.mp4', '.avi', '.mov')):
            temp_in = "temp_in.mp4"
            temp_out = "temp_out.mp4"
            file.save(temp_in)
            
            cap = cv2.VideoCapture(temp_in)
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps): fps = 30
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_out, fourcc, fps/5, (width, height))
            
            frame_count = 0
            deep_focus_streak = 0 # СЧЕТЧИК УСТАЛОСТИ
            
            while True:
                success, frame = cap.read()
                if not success: break
                frame_count += 1
                if frame_count % 5 != 0: continue
                
                _, buffer = cv2.imencode('.jpg', frame)
                url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}&confidence=15"
                
                try:
                    resp = requests.post(url, data=base64.b64encode(buffer).decode('ascii'), headers={"Content-Type": "application/x-www-form-urlencoded"}).json()
                    
                    frame_has_deep_focus = False
                    for pred in resp.get('predictions', []):
                        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                        cls = pred['class']
                        
                        if cls == "Deep_Focus":
                            color = (0, 0, 255)
                            frame_has_deep_focus = True
                        else:
                            color = (255, 0, 255)
                            
                        # Красивые, четкие рамки (LINE_AA + FONT_HERSHEY_COMPLEX)
                        cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), color, 3, cv2.LINE_AA)
                        cv2.putText(frame, cls, (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2, cv2.LINE_AA)
                    
                    # Логика BPI (Blunder Probability Index)
                    if frame_has_deep_focus:
                        deep_focus_streak += 1
                    else:
                        deep_focus_streak = max(0, deep_focus_streak - 1)
                        
                    # Каждое попадание в Deep Focus прибавляет 15% к вероятности ошибки! (Специально для 10-сек демо)
                    bpi = min(98, 12 + (deep_focus_streak * 15)) 
                    
                    # Отрисовка BPI на экране видео
                    bpi_color = (0, 0, 255) if bpi > 60 else (0, 255, 255)
                    cv2.putText(frame, f"BPI (Blunder Risk): {bpi}%", (40, 60), cv2.FONT_HERSHEY_COMPLEX, 1.2, bpi_color, 3, cv2.LINE_AA)
                    
                except: pass
                
                out.write(frame)
                if frame_count > 150: break
                
            cap.release()
            out.release()
            if os.path.exists(temp_in): os.remove(temp_in)
            
            return send_file(temp_out, as_attachment=True, download_name="bpi_analysis.mp4")

        # === ФОТО: СТАТИЧНЫЙ BPI ===
        else:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            _, buffer = cv2.imencode('.jpg', img)
            url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}&confidence=15"
            data = requests.post(url, data=base64.b64encode(buffer).decode('ascii'), headers={"Content-Type": "application/x-www-form-urlencoded"}).json()

            predictions = data.get('predictions', [])
            stats_html = ""

            if not predictions:
                stats_html = "<div class='stats' style='color: #e74c3c;'>No player detected.</div>"
            else:
                highest_bpi = 12
                main_state = "Normal_Focus"
                
                for pred in predictions:
                    x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                    cls = pred['class']
                    
                    if cls == "Deep_Focus":
                        color = (0, 0, 255) 
                        main_state = "Deep_Focus"
                        highest_bpi = 84 # Если на фото Deep Focus, сразу ставим высокий риск ошибки
                    else:
                        color = (255, 0, 255) 
                    
                    # Четкие шахматные шрифты
                    cv2.rectangle(img, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), color, 3, cv2.LINE_AA)
                    cv2.putText(img, cls, (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                color_css = "#ff4d4d" if highest_bpi > 60 else "#f1c40f"
                stats_html = f"<div class='stats' style='color: {color_css};'>Current State: {main_state} | <b>Blunder Risk (BPI): {highest_bpi}%</b></div>"

            _, result_buffer = cv2.imencode('.jpg', img)
            res_img_b64 = base64.b64encode(result_buffer).decode('utf-8')
            
            return f'''
            <html><head>{CSS_STYLE}</head><body>
                <div class="box">
                    <h2>Live Cognitive Report</h2>
                    {stats_html}
                    <img src="data:image/jpeg;base64,{res_img_b64}">
                    <br><br>
                    <a href="/" style="color: #d4af37; text-decoration: none; font-weight: bold; border: 1px solid #d4af37; padding: 10px 20px; border-radius: 5px;">← ANALYZE NEXT MOVE</a>
                </div>
            </body></html>
            '''
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
