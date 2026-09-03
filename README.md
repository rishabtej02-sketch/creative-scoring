# Creative Scoring App

Machine learning app that scores digital creatives (YouTube, Meta, Google Ads) for expected performance before spend.

Uses computer vision + CLIP embeddings + calibrated ML models to predict top-performing creatives and explain them via Grad-CAM heatmaps.

---

## 🚀 Setup Instructions

Split into two parts: **Code** (GitHub) + **Heavy Data** (Google Drive).

### 1. Clone the repo

```bash
git clone https://github.com/rishabtej02-sketch/creative-scoring.git
cd creative-scoring
git lfs pull
```

### 2. Python environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download heavy data

Google Drive link: **https://drive.google.com/drive/folders/1dW38QPqrzP2m-kMpujnMBBm0Q1QJ8Sws?usp=drive_link**

Place downloaded files so final structure matches:

```
creative-scoring/
├── data/
│   ├── google_ads/                    ← from Drive
│   ├── meta_ads/                      ← from Drive
│   ├── thumbnails/                    ← from Drive
│   └── clip_embeddings_paid.npy       ← from Drive
```

### 4. Verify setup

```bash
python src/verify_parity.py
```
Should print `max abs diff < 1e-6`. If yes → setup correct.

### 5. Configure API keys

Create `.env` in project root:
```
GROQ_API_KEY=xxx
GOOGLE_API_KEY=xxx
OPENROUTER_API_KEY=xxx
```
(Keys shared privately by owner.)

---

## 💻 Run the app

```bash
streamlit run src/app.py
```

---

## 📂 Project Structure

* `src/` — Core scripts, ML pipeline, Streamlit UI
* `data/` — Datasets, OCR, embeddings, samples
* `models/` — Trained models (`.pkl`), CNN weights (Git LFS)
* `figs/` — Grad-CAM heatmaps, score plots
