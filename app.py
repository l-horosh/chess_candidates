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
        background: radial-gradient(circle at center, #2c3e50 0%, #000000 100%);
        color: #d4af37; 
        text-align: center; 
        margin: 0;
        padding: 20px;
        min-height: 100vh;
    }
    .box { 
        background: rgba(44, 44, 44, 0.9); 
        border: 2px solid #d4af37; 
        padding: 30px; 
        border-radius: 15px; 
        display: inline-block; 
        box-shadow: 0 0 30px rgba(212, 175, 55, 0.3);
        margin-top: 50px;
    }
    h1 { letter-spacing: 5px; text-transform: uppercase; color: #f1c40f; text-shadow: 2px 2px #000; }
    img { 
        max-width: 90%; 
        max-height: 60vh;
        border: 3px solid #d4af37; 
        border-radius: 10px;
        object-fit: contain;
    }
    button { 
        background: #d4af37; color: black; border: none; padding: 15px 40px; 
        cursor: pointer; font-weight: bold; border-radius: 30px; 
        font-size: 18px; transition: 0.3s;
    }
    button:hover { background: #fff; transform: scale(1.05); }
    .footer { margin-top: 20px; color: #7f8c8d; font-size: 12px; }
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
            <p>Elite Behavioral Analytics | Nano Banan Edition</p>
            <form action="/classify" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*,video/*" required style="margin: 20px 0; color: white;"><br>
                <button type="submit">Analyze Performance</button>
            </form>
            <div class="footer">Powered by DeepFocus YOLOv11</div>
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

        _, buffer = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(buffer).decode('ascii')
        
        url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}&format=image&stroke=5&confidence=15"
        
        # Вот она, исправленная строчка с одинарными скобками!
        response = requests.post(url, data=img_b64, headers={"Content-Type": "application/x-www-form-urlencoded"})
        
        res_img_b64 = base64.b64encode(response.content).decode('utf-8')
        
        return f'''
        <html>
        <head>{CSS_STYLE}</head>
        <body>
            <div class="box">
                <h2>Analysis Result</h2>
                <img src="data:image/jpeg;base64,{res_img_b64}">
                <br><br>
                <a href="/" style="color: #d4af37; text-decoration: none; font-weight: bold;">← BACK TO DASHBOARD</a>
            </div>
        </body>
        </html>
        '''
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
