import os, sys, time, json, socket
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Optional: local process memory (Linux) via resource; psutil fallback
peak_rss_kb = None
def get_peak_rss_kb():
    global peak_rss_kb
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux, ru_maxrss is in kilobytes
        return int(usage.ru_maxrss)
    except Exception:
        try:
            import psutil
            process = psutil.Process()
            return int(process.memory_info().rss / 1024)
        except Exception:
            return None

def now_iso():
    import datetime as dt
    return dt.datetime.utcnow().isoformat() + "Z"

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("rf", "xgb", "cnn"):
        print("Usage: python benchmark.py [rf|xgb|cnn]")
        sys.exit(2)

    model_type = sys.argv[1]
    mem_limit = os.getenv("MEM_LIMIT", "")
    cpu_limit = os.getenv("CPU_LIMIT", "")
    run_id    = os.getenv("RUN_ID", "")
    host      = socket.gethostname()

    y_test = np.load("data/y_test.npy")

    start_load = time.time()
    if model_type == "rf":
        import joblib
        model = joblib.load("models/rf_model.pkl")
        X_test = np.load("data/X_test_pca.npy")

    elif model_type == "xgb":
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model("models/xgb_model.json")
        X_test = np.load("data/X_test_pca.npy")

    else:  # cnn
        import tensorflow as tf
        model = tf.keras.models.load_model("models/cnn_model.h5")
        X_test_scaled = np.load("data/X_test_scaled.npy")
        # reshape for 1D-CNN: [samples, features, 1]
        X_test = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

        # Optional: small warm-up to reduce TF first-call overhead
        _ = model.predict(X_test[:4], verbose=0)

    end_load = time.time()

    # Inference timing
    start_inf = time.time()
    if model_type == "cnn":
        y_proba = model.predict(X_test, verbose=0).ravel()
        y_pred  = (y_proba > 0.5).astype(int)
    else:
        # scikit/xgb
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba > 0.5).astype(int)
    end_inf = time.time()

    # Metrics
    acc   = float(accuracy_score(y_test, y_pred))
    prec  = float(precision_score(y_test, y_pred, zero_division=0))
    rec   = float(recall_score(y_test, y_pred, zero_division=0))
    f1    = float(f1_score(y_test, y_pred, zero_division=0))
    auc   = float(roc_auc_score(y_test, y_proba))

    total_latency_s = end_inf - start_inf
    per_sample_ms   = (total_latency_s / len(X_test)) * 1000.0

    peak_rss_kb = get_peak_rss_kb()

    result = {
        "timestamp_utc": now_iso(),
        "run_id": run_id,
        "host": host,
        "model": model_type,
        "mem_limit": mem_limit,
        "cpu_limit": cpu_limit,
        "n_samples": int(len(X_test)),
        "load_time_s": float(end_load - start_load),
        "inference_time_s": float(total_latency_s),
        "latency_per_sample_ms": float(per_sample_ms),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "peak_rss_kb": peak_rss_kb
    }

    # Print one-line JSON (easy to parse later)
    print(json.dumps(result))
