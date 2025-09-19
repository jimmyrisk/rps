import os, sqlite3, joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = os.environ.get("DATA_PATH", "/data")
DB_PATH = os.path.join(DATA_PATH, "rps.db")
MODEL_PATH = os.path.join(DATA_PATH, "model.pkl")

# load plays and create toy features: last N outcomes / moves (very simple)
with sqlite3.connect(DB_PATH) as conn:
    rows = conn.execute("SELECT user_move, bot_move, outcome FROM plays ORDER BY ts").fetchall()

if len(rows) < 20:
    print("Not enough data to train; need at least 20 rows")
    raise SystemExit(0)

move_to_ix = {"rock":0, "paper":1, "scissors":2}
X, y = [], []
for user_move, bot_move, outcome in rows:
    # Super simple: predict user's next move from bot_move one-hot (toy)
    f = np.zeros(3)
    f[move_to_ix[bot_move]] = 1.0
    X.append(f)
    y.append(move_to_ix[user_move])

X = np.array(X); y = np.array(y)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=100)
clf.fit(X_tr, y_tr)
acc = accuracy_score(y_te, clf.predict(X_te))
print({"accuracy": float(acc), "n": int(len(y))})
joblib.dump(clf, MODEL_PATH)

# If using MLflow later, you can log metrics and the artifact here.