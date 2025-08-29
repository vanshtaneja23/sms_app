
# SMS Spam Detector (Flask + Heroku)

A tiny web app and JSON API that classifies SMS messages as **spam** or **ham** using your trained scikit-learn model.

## Quick start (local)

1) Install Python 3.11 (or update `runtime.txt` to your version).
2) Create and activate a virtual environment.
3) `pip install -r requirements.txt`
4) Put your trained artifacts in the project root:
   - `model-2.pkl`
   - `vectorizer.pkl`
5) `python app.py`
6) Open http://localhost:5000

## API
`POST /api/predict` with JSON `{"message": "text"}` returns `{ "label": "spam", "spam_probability": 0.97 }`

## Deploy to Heroku

- You need a Heroku account and the Heroku CLI installed.
- Heroku no longer has a free dyno. Use the **Eco** or **Basic** dyno.

```bash
heroku login
heroku create sms-spam-<your-unique-suffix> --stack heroku-22
# Commit the repo
git init
git add .
git commit -m "Initial commit: SMS spam detector"
# Push to Heroku
heroku git:remote -a sms-spam-<your-unique-suffix>
git push heroku HEAD:main
# (If main branch doesn't exist yet, create it: git branch -M main)
# Scale a web dyno
heroku ps:scale web=1
# Open
heroku open
```

### Add your model files
Ensure `model-2.pkl` and `vectorizer.pkl` are committed. If the pickle fails to load due to version mismatch, align `scikit-learn`, `numpy`, and `scipy` versions in `requirements.txt` with those used to train the model.

## Environment variables (optional)
You can override paths:
- `MODEL_PATH` (default: `model-2.pkl`)
- `VECTORIZER_PATH` (default: `vectorizer.pkl`)

## Notes
- If you want CORS for a frontend, add `flask-cors` and enable it.
- To keep dyno from sleeping on Eco, consider pinging the app externally (per Heroku ToS) or upgrade the plan.
