# 🚀 Deployment Guide for Plant Disease Predictor

## Option 1: Streamlit Cloud (Recommended - FREE)

### Prerequisites
- GitHub account
- Your code pushed to a GitHub repository

### Steps:

1. **Prepare Your Repository**
   - Ensure all files are in your GitHub repository:
     - `app1.py` (main application)
     - `requirements.txt` (Python dependencies)
     - `packages.txt` (system dependencies)
     - `plant_disease_model_15_class.h5` (model file)
     - `.streamlit/config.toml` (configuration)

2. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Click "Sign in with GitHub"
   - Click "New app"
   - Fill in the details:
     - Repository: `your-username/your-repo-name`
     - Branch: `main` (or your branch name)
     - Main file path: `app1.py`
   - Click "Deploy!"

3. **Wait for Deployment**
   - Initial deployment takes 5-10 minutes
   - Streamlit Cloud will install all dependencies
   - Your app will be live at: `https://your-app-name.streamlit.app`

### Important Notes for Streamlit Cloud:
- ✅ Free tier includes 1GB RAM
- ✅ Model file (plant_disease_model_15_class.h5) must be in repository
- ✅ Maximum file size: 100MB per file
- ⚠️ If model file is too large, consider using Git LFS

---

## Option 2: Heroku

### Prerequisites
- Heroku account
- Heroku CLI installed

### Steps:

1. **Create Additional Files**

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

Create `Procfile`:
```
web: sh setup.sh && streamlit run app1.py
```

2. **Deploy to Heroku**
```bash
heroku login
heroku create your-app-name
git push heroku main
heroku open
```

---

## Option 3: AWS EC2

### Steps:

1. **Launch EC2 Instance**
   - Choose Ubuntu Server
   - t2.medium or larger (for TensorFlow)
   - Open port 8501 in security group

2. **SSH into Instance**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Install Dependencies**
```bash
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
```

4. **Run Application**
```bash
streamlit run app1.py --server.port 8501 --server.address 0.0.0.0
```

5. **Access App**
   - Visit: `http://your-ec2-ip:8501`

---

## Option 4: Google Cloud Run

### Steps:

1. **Create Dockerfile**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD streamlit run app1.py --server.port 8080 --server.address 0.0.0.0
```

2. **Deploy**
```bash
gcloud run deploy plant-disease-app --source .
```

---

## Option 5: Local Network Deployment

### For sharing on your local network:

```bash
streamlit run app1.py --server.address 0.0.0.0 --server.port 8501
```

Access from other devices: `http://your-local-ip:8501`

---

## Troubleshooting

### Model File Too Large for GitHub
If your model file exceeds 100MB:

1. **Use Git LFS**
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add plant_disease_model_15_class.h5
git commit -m "Add model with LFS"
git push
```

2. **Or Use Cloud Storage**
   - Upload model to Google Drive/Dropbox
   - Modify app1.py to download model on startup:
```python
import gdown
if not os.path.exists('plant_disease_model_15_class.h5'):
    gdown.download('your-google-drive-link', 'plant_disease_model_15_class.h5')
```

### Memory Issues
- Upgrade to Streamlit Cloud paid tier
- Use AWS/GCP with more RAM
- Optimize model (quantization, pruning)

### Slow Loading
- Enable caching with @st.cache_resource
- Compress model file
- Use CDN for static assets

---

## Quick Start Commands

### Initialize Git Repository
```bash
cd "c:\Users\nanda\OneDrive\Desktop\plant disease app"
git init
git add .
git commit -m "Initial commit"
```

### Create GitHub Repository
```bash
# Create repo on GitHub, then:
git remote add origin https://github.com/your-username/plant-disease-app.git
git branch -M main
git push -u origin main
```

### Deploy to Streamlit Cloud
- Visit: https://share.streamlit.io
- Connect your GitHub repository
- Deploy!

---

## Recommended: Streamlit Cloud

For this application, **Streamlit Cloud** is the best option because:
- ✅ Free and easy to use
- ✅ Automatic HTTPS
- ✅ Built-in CI/CD
- ✅ Perfect for Streamlit apps
- ✅ No server management needed

Your app will be live at: `https://your-app-name.streamlit.app`
