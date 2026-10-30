# Customer Segmentation System

## Run (no database, no pandas needed)
    cd backend
    pip install flask flask-cors
    python app.py            # API on http://127.0.0.1:5000
Then open `frontend/index.html` (or run `python -m http.server 8000` inside `frontend`).

## Retrain (needs pandas + scikit-learn + joblib)
    cd backend
    pip install pandas numpy scikit-learn joblib
    python train_model.py

## API
- `/segments?segment=Champions&limit=100`  customers
- `/summary`                               per-segment size and average RFM
- `/top-customers?n=20`                    top spenders
- `/metrics`                               model quality
- `/predict?recency=30&frequency=12&monetary=45000`  segment for a new customer
