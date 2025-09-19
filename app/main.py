from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, time, joblib, sqlite3
from prometheus_fastapi_instrumentator import Instrumentator

DATA_PATH = os.environ.get("DATA_PATH", "/data")
DB_PATH = os.path.join(DATA_PATH, "rps.db")
MODEL_PATH = os.path.join(DATA_PATH, "model.pkl")

app = FastAPI(title="RPS API")

class Play(BaseModel):
    user_move: str  # "rock" | "paper" | "scissors"

MOVES = ["rock", "paper", "scissors"]

# Initialize DB if missing
os.makedirs(DATA_PATH, exist_ok=True)
with sqlite3.connect(DB_PATH) as conn:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS plays (
          ts REAL, user_move TEXT, bot_move TEXT, outcome INTEGER
        )
        """
    )

Instrumentator().instrument(app).expose(app)  # /metrics

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/play")
def play(p: Play):
    if p.user_move not in MOVES:
        raise HTTPException(400, "invalid move")

    # naive policy: if model exists, use it; else random
    import random
    bot_move = random.choice(MOVES)

    # If model exists, adjust bot_move using predicted user next move
    try:
        model = joblib.load(MODEL_PATH)
        # toy features: last move one-hot; here just placeholder zeros
        import numpy as np
        x = np.zeros((1, 3))
        proba = model.predict_proba(x)[0]  # [p(rock), p(paper), p(scissors)]
        # counter the most likely user move
        likely = MOVES[int(proba.argmax())]
        counter = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
        bot_move = counter[likely]
    except Exception:
        pass

    # compute outcome: 1 win, 0 tie, -1 loss (from bot perspective)
    outcome = 0
    win = {("rock", "scissors"), ("paper", "rock"), ("scissors", "paper")}
    if (bot_move, p.user_move) in win:
        outcome = 1
    elif bot_move == p.user_move:
        outcome = 0
    else:
        outcome = -1

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO plays(ts, user_move, bot_move, outcome) VALUES (?,?,?,?)",
            (time.time(), p.user_move, bot_move, outcome),
        )
    return {"bot_move": bot_move, "outcome": outcome}