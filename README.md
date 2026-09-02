# Creative Scoring App

A machine learning application to score, analyze, and visualize performance predictions for digital creatives across YouTube, Meta, and Google Ads. 

This project uses computer vision, CLIP embeddings, and calibrated ML models to predict top-performing creatives and provide feature-level explanations (like Grad-CAM heatmaps).

---

## 🚀 Setup Instructions

Due to GitHub's file size limits, this project is split into two parts: the **Code** (hosted here) and the **Heavy Data** (hosted securely on Google Drive). 

Follow these steps exactly to get the app running on your local machine.

### 1. Clone the Repository
First, pull the code and the Git LFS (Large File Storage) models:
```bash
git clone [https://github.com/rishabtej02-sketch/creative-scoring.git](https://github.com/rishabtej02-sketch/creative-scoring.git)
cd creative-scoring
git lfs pull
```

### 2. Set Up the Python Environment
Create a virtual environment and install the required dependencies:
```bash
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Mac/Linux: source .venv/bin/activate

pip install -r requirements_ads.txt
```

### 3. Download and Place the Heavy Data
The raw images, thumbnails, and large `.npy` files are too big for GitHub. 
1. Download the `creative-scoring-bigdata` folder from the provided Google Drive link.
2. Extract the files and place them directly into the `data/` folder in this repository. 
3. Your final path structure must look exactly like this or the app will crash:
   * `data/google_ads/`
   * `data/meta_ads/`
   * `data/thumbnails/`
   * `data/clip_embeddings_paid.npy`
   * *(Any other zip files provided in the Drive)*

### 4. Configure Secrets (API Keys)
This app requires a Google Cloud API key to function, which is kept out of version control for security.
1. Create a folder named `.streamlit` in the root of the project.
2. Inside that folder, create a file named `secrets.toml`.
3. Paste the GCP API Key provided to you privately into `secrets.toml`.

Your structure should look like this:
```text
creative-scoring/
├── .streamlit/
│   └── secrets.toml
├── data/
│   ├── google_ads/
│   └── ...
├── src/
└── ...
```

---

## 💻 Running the App

Once the data is in place and the secrets are configured, you can launch the Streamlit dashboard:

```bash
streamlit run src/app.py
```

## 📂 Project Structure
* `src/` - Core application scripts, ML pipelines, and Streamlit UI components.
* `data/` - Datasets, extracted text (OCR), embeddings, and sample images.
* `models/` - Trained models (`.pkl`), CNN weights, and calibration files (managed by Git LFS).
* `figs/` - Generated Grad-CAM heatmaps and score distribution plots.
