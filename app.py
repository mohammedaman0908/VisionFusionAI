import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import uuid
from fusion_model import ImageFusionModel
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-for-image-fusion")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure upload settings
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize fusion model
fusion_model = ImageFusionModel()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')

@app.route('/fuse', methods=['POST'])
def fuse_images():
    """Handle image fusion request"""
    try:
        # Check if files were uploaded
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        
        if len(files) < 2:
            return jsonify({'error': 'At least 2 images are required for fusion'}), 400
        
        if len(files) > 3:
            return jsonify({'error': 'Maximum 3 images allowed for fusion'}), 400
        
        # Validate and save uploaded files
        uploaded_files = []
        session_id = str(uuid.uuid4())
        
        for i, file in enumerate(files):
            if file.filename == '':
                return jsonify({'error': f'Empty filename for image {i+1}'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': f'Invalid file type for {file.filename}. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
            
            # Create secure filename with session ID
            filename = f"{session_id}_input_{i+1}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filepath)
            
            logging.debug(f"Saved uploaded file: {filepath}")
        
        # Perform image fusion
        logging.info(f"Starting fusion process for {len(uploaded_files)} images")
        
        result_filename = f"{session_id}_fused.png"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # Call fusion model
        metrics = fusion_model.fuse_images(uploaded_files, result_path)
        
        # Prepare response with relative paths
        input_urls = [f"/static/uploads/{os.path.basename(f)}" for f in uploaded_files]
        result_url = f"/static/results/{result_filename}"
        
        response_data = {
            'success': True,
            'input_images': input_urls,
            'fused_image': result_url,
            'metrics': metrics,
            'message': 'Image fusion completed successfully'
        }
        
        logging.info(f"Fusion completed successfully. Result saved to: {result_path}")
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Error during image fusion: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'error': f'Fusion process failed: {str(e)}',
            'success': False
        }), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB per file.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
