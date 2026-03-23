import os
import sys
try:
    from flask import Flask, request, render_template, send_from_directory, jsonify, redirect, url_for, flash
    from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
except ModuleNotFoundError:
    print("Error: missing required package 'flask'.")
    print("Install it in your active environment with:")
    print("    python -m pip install flask")
    print("Or install all requirements:")
    print("    python -m pip install -r requirements.txt")
    sys.exit(1)
from werkzeug.utils import secure_filename
import torch
from src.model import DilatedAttentionNetwork
import predict_custom
import database
try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:
    hf_hub_download = None

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("MODEL_PATH", "output/best_model.pth")
OUTPUT_FOLDER = "output/custom_predictions"
ALLOWED_EXT = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, full_name, email):
        self.id = id
        self.username = username
        self.full_name = full_name
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    user_data = database.get_user(user_id)
    if user_data:
        return User(user_data['id'], user_data['username'], user_data['full_name'], user_data['email'])
    return None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def get_model_path():
    """Resolve local model path, downloading from Hugging Face if configured."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    repo_id = os.getenv("HF_MODEL_REPO_ID")
    filename = os.getenv("HF_MODEL_FILENAME", "best_model.pth")
    if repo_id and hf_hub_download is not None:
        print(f"Downloading model from Hugging Face repo: {repo_id}")
        return hf_hub_download(repo_id=repo_id, filename=filename)

    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Set HF_MODEL_REPO_ID and HF_MODEL_FILENAME for automatic download."
    )


def load_model():
    model = DilatedAttentionNetwork(num_classes=7).to(DEVICE)
    model_path = get_model_path()
    state = torch.load(model_path, map_location=DEVICE)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    return model




# Load model once
try:
    MODEL = load_model()
    print("Model loaded successfully for web server.")
except Exception as e:
    MODEL = None
    print(f"Warning: could not load model at startup: {e}")

# Initialize DB on import so Gunicorn deployments also create schema.
database.init_db()


def _to_image_url(image_value):
    if not image_value:
        return image_value
    if image_value.startswith("http://") or image_value.startswith("https://"):
        return image_value
    return f"/predictions/{image_value}"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_data = database.verify_user(username, password)
        if user_data:
            user = User(user_data['id'], user_data['username'], user_data['full_name'], user_data['email'])
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        full_name = request.form['full_name']
        
        if database.create_user(username, email, password, full_name):
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        flash('Username already exists')
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    user = database.get_user(current_user.id)
    history = database.get_user_history(current_user.id)
    for item in history:
        item["input_image_url"] = _to_image_url(item.get("input_image"))
        item["output_image_url"] = _to_image_url(item.get("output_image"))
    return render_template('dashboard.html', user=user, history=history)

@app.route('/analysis/<analysis_id>', endpoint='analysis')
@login_required
def analysis(analysis_id):
    item = database.get_history_item(current_user.id, analysis_id)
    if not item:
        # User requested an analysis that doesn't exist or doesn't belong to them.
        return "Analysis not found", 404
    item["input_image_url"] = _to_image_url(item.get("input_image"))
    item["output_image_url"] = _to_image_url(item.get("output_image"))

    return render_template('analysis.html', item=item)


@app.route('/app')
@login_required
def app_page():
    return render_template('app.html')


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    # Accept optional uploaded file. If provided, run prediction on it; otherwise use configured image.
    try:
        if MODEL is None:
            return jsonify({'error': 'Model is not loaded. Check model path or HF model config.'}), 500
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            filename = secure_filename(file.filename)
            upload_dir = os.path.join('tmp_uploads')
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            upload_path = os.path.join(upload_dir, filename)
            file.save(upload_path)
            # Use predict_custom helper with the pre-loaded MODEL for speed
            input_fname, out_fname, stats = predict_custom.predict_image(input_image_path=upload_path, model=MODEL)
        else:
            input_fname, out_fname, stats = predict_custom.predict_image(model=MODEL)

        # Save to database
        database.add_prediction(current_user.id, input_fname, out_fname, stats)

        return jsonify({
            'input_url': f'/predictions/{input_fname}', 
            'result_url': f'/predictions/{out_fname}',
            'stats': stats
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predictions/<path:filename>')
def serve_prediction(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    # Create output folder if missing
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    app.run(host='0.0.0.0', port=5000, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
