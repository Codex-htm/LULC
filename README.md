# Land Use Land Cover (LULC) Project

This project is a Land Use Land Cover classification application using a Deep Learning model (Dilated Attention Network). It provides a web interface for users to upload satellite images and get classification maps.

## Features
- **Authentication**: Secure Login and Registration system.
- **Dashboard**: View your prediction history.
- **Interactive App**: Upload images and visualize results.

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Standard Laptop (CPU only is fine, the code automatically detects if GPU is missing)
- CUDA capable GPU (Optional - only needed for faster training, not required for running the likely)

## 🛠️ Step-by-Step Setup Guide (For Beginners)

If you have nothing installed on your laptop, follow these steps first:

### 1. Install Python
1.  Download Python from [python.org](https://www.python.org/downloads/).
2.  Run the installer. **IMPORTANT**: Check the box that says **"Add Python to PATH"** before clicking Install.

## 🚀 Installation & Running


### Next Steps (For both options)

1.  **Create a Virtual Environment** (Keeps things clean):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
    *(You will see `(venv)` appear at the start of your command line)*

2.  **Install Dependencies** (This installs the AI libraries):
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This might take 5-10 minutes depending on your internet speed.*

3.  **Run the App**:
    ```bash
    python web_app.py
    ```

4.  **Open the App**:
    - Go to your browser (Chrome/Edge) and type: `http://localhost:5000`
    - **Login Credentials**:
        - Username: `demo_user`
        - Password: `password123`

    > **Note**: The first time you run `web_app.py`, it may take a few minutes to download the ResNet50 model weights. Please be patient.

## Usage

1.  **Register**: Create a new account.
2.  **Login**: Log in with your credentials.
    - *Default Demo User*:
        - Username: `demo_user`
        - Password: `password123`
3.  **Dashboard**: Check your past predictions.
4.  **App**: Upload an image to classify land cover types.

## Project Structure
- `web_app.py`: Main Flask application.
- `database.py`: Database interactions (SQLite).
- `src/`: Source code for the model.
- `templates/`: HTML templates.
- `static/`: CSS and JS files.
- `output/`: Stores model predictions.
- `inputDATA/`: Sample input data.

## Troubleshooting
- **Model not found**: Ensure `output/best_model.pth` exists. If not, you may need to train the model first using `train.py`.
- **Database errors**: Delete `lulc.db` to reset the database if you encounter schema issues.

## Production Deployment (Recommended)

This app is now compatible with both SQLite (local) and PostgreSQL (cloud). For production, use:

- **App Hosting**: Render (or Koyeb/Fly.io)
- **Database**: Neon PostgreSQL (or Supabase PostgreSQL)
- **Model Weights**: Hugging Face model repository

### 1) Set Environment Variables

Use values from `.env.example`:

- `SECRET_KEY`: strong random string
- `DATABASE_URL`: cloud Postgres connection string
- `HF_MODEL_REPO_ID`: Hugging Face repo id, e.g. `yourname/lulc-model`
- `HF_MODEL_FILENAME`: model filename in HF repo, e.g. `best_model.pth`
- `FLASK_DEBUG=false`
- `CLOUDINARY_URL`: Cloudinary API URL, e.g. `cloudinary://API_KEY:API_SECRET@CLOUD_NAME`

Optional:
- `MODEL_PATH`: local fallback path, default `output/best_model.pth`

### 2) Deploy to Render

1. Push this repo to GitHub.
2. In Render, create a **Web Service** from the repo.
3. Render will detect `requirements.txt` and `Procfile`.
4. Add the environment variables above.
5. Deploy.

Start command used:
```bash
gunicorn web_app:app
```
**CRITICAL**: Render may default to `gunicorn app:app`. To fix this ModuleNotFoundError, explicitly set the **Start Command** in Render to `gunicorn web_app:app`.

### 3) Database Notes

- Local dev still uses SQLite if `DATABASE_URL` is not set.
- Production should use PostgreSQL to avoid local file limitations and data loss.

### 4) Prediction Image Storage

Current app stores generated images on local filesystem (`output/custom_predictions`).
For fully durable cloud storage (perfect for Render), we have integrated Cloudinary!
Add `CLOUDINARY_URL` to your Render environment variables, and the app will automatically upload images and store their long-lived URLs in your database.
