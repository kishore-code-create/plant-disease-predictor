# 🌿 Plant Disease Predictor

An advanced AI-powered web application for detecting plant diseases from leaf images using deep learning.

🌐 **Live App:** https://plant-disease-predictor-sem-v.streamlit.app/

## Features

- 🔍 Real-time disease detection for 15 different plant conditions
- 📦 Batch processing for multiple images
- 📊 Detailed disease information and treatment recommendations
- 📜 Prediction history tracking
- 🎨 Image enhancement tools
- 📈 Confidence scoring and visualization

## Supported Plants

- 🌶️ Bell Peppers (2 conditions)
- 🥔 Potatoes (3 conditions)
- 🍅 Tomatoes (10 conditions)

## Deployment

### Streamlit Cloud Deployment

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set main file path: `app1.py`
7. Click "Deploy"

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app1.py
```

## Model

The application uses a Convolutional Neural Network (CNN) trained on thousands of plant leaf images.

- **Model Type:** CNN
- **Input Size:** 150x150 pixels
- **Classes:** 15
- **Framework:** TensorFlow/Keras

## Important Notes

⚠️ This tool is for preliminary assessment only. Always consult with agricultural experts for final diagnosis.

## Contact

📧 Email: nandakishoredevarashetti@gmail.com

## License

© 2024 All Rights Reserved
