from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from datetime import datetime
import json
from time import time as current_time
from werkzeug.utils import secure_filename

import deepfake_detector

app = Flask(__name__)

# Config
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'static/videos')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the service."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }), 200

@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video uploads and trigger deepfake detection."""
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        timestamp = int(current_time())
        unique_filename = f"uploaded_video_{timestamp}_{filename}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(video_path)

        output_filename = f"processed_{timestamp}_{filename}"
        video_path2 = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        try:
            # Service layer call
            result_from_det = deepfake_detector.run(video_path, video_path2)
            
            video_info = {
                'name': filename,
                'size': f"{os.path.getsize(video_path) / 1024:.2f} KB",
                'user': 'Guest', 
                'source': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'per': result_from_det
            }
            video_info_json = json.dumps(video_info)
            
            return redirect(url_for('result', video_info=video_info_json, video_path2=video_path2))
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return render_template('index.html', error="Failed to process video.")

    return redirect(request.url)

@app.route('/result')
def result():
    """Render the result page."""
    video_info_json = request.args.get('video_info')
    video_path2 = request.args.get('video_path2')  

    if not video_info_json or not video_path2:
        return redirect(url_for('index'))

    try:
        video_info = json.loads(video_info_json)
    except json.JSONDecodeError:
        return redirect(url_for('index'))

    return render_template('result.html', video_url=video_path2, video_info=video_info)

if __name__ == '__main__':
    # Disable reloader to prevent connection reset on video writes
    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() in ('true', '1')
    app.run(debug=debug_mode, use_reloader=False)
