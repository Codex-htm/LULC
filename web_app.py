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

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "output/best_model.pth"
OUTPUT_FOLDER = "output/custom_predictions"
ALLOWED_EXT = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this in production

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


def load_model():
    model = DilatedAttentionNetwork(num_classes=7).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=DEVICE)
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
    return render_template('dashboard.html', user=user, history=history)

@app.route('/analysis/<analysis_id>', endpoint='analysis')
@login_required
def analysis(analysis_id):
    item = database.get_history_item(current_user.id, analysis_id)
    if not item:
        # User requested an analysis that doesn't exist or doesn't belong to them.
        return "Analysis not found", 404

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
    
    # Initialize Database
    database.init_db()
    print("Database initialized.")

    app.run(host='0.0.0.0', port=5000, debug=True)
