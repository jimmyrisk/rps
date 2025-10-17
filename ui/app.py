# ui/app.py
import os, uuid, requests, streamlit as st, hashlib, random, datetime
import math

API = os.getenv("API_BASE", "http://rps-app.mlops-poc.svc.cluster.local")

# Generate deterministic 3-digit code from session ID
def generate_player_code(session_id: str) -> str:
    """Generate a deterministic 3-digit code from session ID"""
    hash_object = hashlib.sha256(session_id.encode())
    hex_dig = hash_object.hexdigest()
    # Use first 3 characters, convert to uppercase for readability
    return hex_dig[:3].upper()

# --- Branding ---
st.set_page_config(page_title="RPS Quest", page_icon="⚔️", layout="wide")

st.markdown("""
<style>
  :root { --rps-score-size: clamp(1.1rem, 2.6vw, 1.6rem); }

  /* === Layout: left controls + centered combat === */
  .rps-controls{width:100%;text-align:left;}
  .rps-controls .stButton > button{width:170px;} /* not full width */
  .rps-controls .rps-badge{width:170px;text-align:center;}

  .rps-combat{width:clamp(380px,45vw,720px);height:180px;margin:0 auto;
    display:flex;align-items:center;justify-content:center;
    border:1px dashed var(--primary-color);border-radius:.75rem;opacity:.9;}

  /* Game over state */
  .rps-game-over{text-align:center;margin:2rem 0;}
  .rps-game-over h2{color:var(--primary-color);margin-bottom:1rem;}

  /* Damage badges with color coding */
  .rps-badge { 
    display:inline-block; padding:.30rem .6rem; margin:0; 
    border-radius:9999px; font-weight:600; border: 1px solid var(--primary-color); 
    font-size: calc(1rem + 2px); text-align:center;
    /* Dynamic coloring based on damage value */
    background: var(--dmg-bg-color, var(--secondary-background-color)); 
    color: var(--dmg-text-color, var(--text-color));
    transition: background-color 0.2s ease, color 0.2s ease;
    white-space: nowrap;   /* keep “DMG (damage) 20” on one line */
    min-width: 64px;            
  }

  /* HP Bar Styles */
  .rps-hp {
    margin: 1rem 0;
    border-radius: 8px;
    background: var(--secondary-background-color);
    padding: 1rem;
  }

  .rps-hp-row {
    display: grid;
    grid-template-columns: 4.5rem 1fr 6rem 3rem;
    gap: 0.75rem;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .rps-hp-row:last-child {
    margin-bottom: 0;
  }

  .rps-hp-label {
    font-weight: 600;
    font-size: 0.9rem;
  }

  .rps-hp-track {
    height: 24px;
    background: var(--secondary-background-color);
    border-radius: 12px;
    border: 2px solid var(--primary-color);
    overflow: hidden;
    position: relative;
  }

  .rps-hp-fill {
    height: 100%;
    border-radius: 10px;
    transition: all 0.3s ease;
  }

  .rps-hp-num {
    font-weight: bold;
    font-size: 0.85rem;
    text-align: right;
    min-width: 4rem;
  }

  .rps-hp-damage {
    font-size: 0.75rem;
    font-weight: bold;
    text-align: center;
    min-width: 2.5rem;
  }

  .rps-hp-blip {
    color: #ef4444;
    background: rgba(239, 68, 68, 0.1);
    padding: 0.1rem 0.3rem;
    border-radius: 4px;
    font-size: 0.7rem;
  }
                    
    /* Combat UI */
    .rps-combat-grid{
    display:grid;
    grid-template-columns:1fr auto 1fr;
    align-items:center;
    justify-items:center;
    gap:1rem;
    width:100%;
    }
    .rps-combat-side{
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    }
    .rps-combat-emoji{ font-size:2.25rem; line-height:1; }
    .rps-combat-vs{ font-size:0.9rem; opacity:.7; user-select:none; }

    /* Damage badges */
    .rps-dmg-badge{
    position: relative;                /* needed for ::after line positioning */
    display: inline-block;
    padding: .10rem .4rem;
    border-radius: .5rem;
    font-weight: 700;
    font-size: .8rem;
    border: 1px solid var(--primary-color);
    margin-top: .25rem;
    line-height: 1.1;
    white-space: nowrap;   /* keep “DMG (damage) 20” on one line */
    min-width: 64px;
    text-align: center;
    }
    .rps-dmg-badge.losing{
    text-decoration: none;             /* don’t rely on this anymore */
    }
    /* the actual strike line (always visible on any bg) */
    .rps-dmg-badge.losing::after{
    content: "";
    position: absolute;
    left: .35rem;                      /* inset a touch from the pill edges */
    right: .35rem;
    top: 50%;
    height: 2px;                       /* nice, bold */
    transform: translateY(-50%);
    background: currentColor;          /* matches the text color for contrast */
    opacity: .95;                      /* punchy and visible */
    pointer-events: none;
    }
    /* Triangle controls */
    .rps-tri-wrap { position: relative; max-width: 560px; margin: 1.25rem auto; }
    .rps-tri-svg  { position: absolute; inset: 0; pointer-events: none; z-index: 10; }

    /* Arrow look */
    .rps-tri-svg .edge { stroke: var(--text-color); stroke-width: 2; marker-end: url(#rpsArrow); opacity: .85; }
    .rps-tri-svg .cap  { font-size: 4px; fill: var(--text-color); opacity: .9; }

    /* keep badges tight and non-wrapping (you already had this) */
    .rps-badge { white-space: nowrap; min-width: 64px; }

    /* optional: compact the side columns a bit */
    .rps-side-col { display:flex; align-items:center; justify-content:center; }
    }
""", unsafe_allow_html=True)

# ---------- helpers ----------
@st.cache_data(ttl=30)
def load_policies():
    # returns a list of {"id": ..., "label": ...}
    try:
        r = requests.get(f"{API}/policies", timeout=5)
        r.raise_for_status()
        data = r.json()
        out = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    out.append({"id": item, "label": item.replace("_"," ").title()})
                elif isinstance(item, dict):
                    pid = item.get("id") or item.get("name") or item.get("policy")
                    if not pid:  # skip bad entries
                        continue
                    label = item.get("label") or pid.replace("_"," ").title()
                    out.append({"id": pid, "label": label})
        if out:
            return out
    except Exception:
        pass
    # fallback if API is unreachable
    return [
        {"id":"brian","label":"🧠 Brian (Neural Network)"},
        {"id":"forrest","label":"🌲 Forrest (XGBoost)"},
        {"id":"logan","label":"🪵 Logan (Logistic Regression)"},
    ]


def fetch_json(url, **params):
    """Enhanced fetch with debugging and error handling"""
    import time
    start_time = time.time()
    
    try:
        r = requests.get(url, params=params, timeout=6)
        elapsed = time.time() - start_time
        
        # Log API call details for debugging
        if st.session_state.get("show_debug"):
            st.sidebar.write(f"🔍 API Call: {url}")
            st.sidebar.write(f"⏱️ Time: {elapsed:.2f}s")
            st.sidebar.write(f"📊 Status: {r.status_code}")
            if params:
                st.sidebar.write(f"📋 Params: {params}")
        
        if r.ok:
            data = r.json()
            if st.session_state.get("show_debug"):
                st.sidebar.write(f"✅ Response keys: {list(data.keys()) if isinstance(data, dict) else 'Non-dict response'}")
            return data
        else:
            if st.session_state.get("show_debug"):
                st.sidebar.write(f"❌ Error: {r.status_code} - {r.text[:100]}")
            return None
    except requests.RequestException as e:
        elapsed = time.time() - start_time
        if st.session_state.get("show_debug"):
            st.sidebar.write(f"🔍 API Call: {url}")
            st.sidebar.write(f"⏱️ Time: {elapsed:.2f}s")
            st.sidebar.write(f"💥 Exception: {str(e)[:100]}")
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        if st.session_state.get("show_debug"):
            st.sidebar.write(f"🔍 API Call: {url}")
            st.sidebar.write(f"⏱️ Time: {elapsed:.2f}s")
            st.sidebar.write(f"💥 Other Error: {str(e)[:100]}")
        return None

def fetch_round(game_id):
    return fetch_json(f"{API}/round_points", game_id=game_id) or {"round_no": None, "round_points": {}}

def fetch_stats(game_id):
    return fetch_json(f"{API}/stats", game_id=game_id) or {}


def parse_iso8601(ts: str | None):
    """Parse ISO-8601 timestamps like '2025-10-10T08:00:00Z' into aware datetimes."""
    if not ts:
        return None
    try:
        normalized = ts.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        return datetime.datetime.fromisoformat(normalized)
    except Exception:
        return None

def fetch_bot_predictions(game_id):
    """Fetch current bot's probability predictions for the active game"""
    return fetch_json(f"{API}/predict/current_bot/{game_id}") or {}

def bot_avatar(policy):
    """Return bot-specific avatar emoji based on policy"""
    return {"brian": "🧠", "forrest": "🌲", "logan": "🪵"}.get(policy.lower(), "🤖")

def damage_badge_colors(values: list) -> dict:
    """Return color mapping for damage values. Lowest=white, highest=green, middle=yellow."""
    if not values:
        return {}
    
    sorted_vals = sorted(set(values))
    if len(sorted_vals) == 1:
        # All same value - use white
        return {sorted_vals[0]: ("var(--secondary-background-color)", "var(--text-color)")}
    elif len(sorted_vals) == 2:
        # Two values - lowest=white, highest=green
        return {
            sorted_vals[0]: ("var(--secondary-background-color)", "var(--text-color)"),
            sorted_vals[1]: ("#10b981", "#ffffff")  # Green background, white text
        }
    else:
        # Three values - lowest=white, middle=yellow, highest=green
        return {
            sorted_vals[0]: ("var(--secondary-background-color)", "var(--text-color)"),
            sorted_vals[1]: ("#fbbf24", "#000000"),  # Yellow background, black text
            sorted_vals[2]: ("#10b981", "#ffffff")   # Green background, white text
        }

def dmg_from_value(v: float) -> int:
    """Convert 1.0..2.0 round multiplier -> 10..20 damage (rounded)."""
    try:
        return int(round(10 * float(v)))
    except Exception:
        return 10

# --- display math helpers ---
def hp_from_points(x: float) -> int:
    """Convert 0..10 points -> 100..0 hit points (rounded, clamped)."""
    try:
        v = 100 - 10 * float(x or 0)
    except Exception:
        v = 100
    return max(0, min(100, int(round(v))))

def hp_color(hp: int) -> str:
    """Return color based on HP value using theme colors."""
    if hp >= 100:
        return "#ffffff"  # White for perfect health
    elif hp >= 80:
        return "#10b981"  # Green (same as highest damage value)
    elif hp >= 50:
        return "#fbbf24"  # Yellow
    elif hp >= 25:
        return "#f97316"  # Orange
    else:
        return "#ef4444"  # Red

def value_badge(label: str, v: float) -> str:
    """Reused as damage badge in RPS Quest."""
    try:
        fv = float(v)
    except Exception:
        fv = 1.0
    dmg = dmg_from_value(fv)
    # emphasize when original value meaningfully > 1.0
    cls = "rps-badge emph" if fv >= 1.4 else "rps-badge"
    return f'<div class="{cls}"><span class="small">DMG</span>{dmg}</div>'


# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    grafana_url = os.getenv("GRAFANA_DASHBOARD_URL")
    if grafana_url:
        if hasattr(st, "link_button"):
            st.link_button(
                "📈 Open Monitoring Dashboard",
                grafana_url,
                use_container_width=True,
                help="Opens the Grafana dashboard for live A/B metrics and service health.",
            )
        else:
            st.markdown(f"[📈 Open Monitoring Dashboard]({grafana_url})")
    else:
        st.caption("Set GRAFANA_DASHBOARD_URL to enable the monitoring shortcut.")
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex

    # Generate deterministic player code from session ID
    player_code = generate_player_code(st.session_state.session_id)
    name = st.text_input("Your name", value=f"Player{player_code}")

    # Fetch policy list from API (generic), with a safe fallback
    try:
        resp = requests.get(f"{API}/policies", timeout=5)
        data = resp.json() if resp.ok else []
    except Exception:
        data = []

    # Normalize into [{"id":..., "label":...}] and add emojis
    pols = []
    emoji_map = {"brian": "🧠", "forrest": "🌲", "logan": "🪵"}
    
    for item in data:
        if isinstance(item, str):
            emoji = emoji_map.get(item.lower(), "")
            label = item.replace("_", " ").title()
            if emoji:
                label = f"{emoji} {label}"
            pols.append({"id": item, "label": label})
        elif isinstance(item, dict):
            pid = item.get("id") or item.get("name") or item.get("policy")
            if pid:
                emoji = emoji_map.get(pid.lower(), "")
                label = item.get("label") or pid.replace("_", " ").title()
                # Add emoji if not already in label
                if emoji and emoji not in label:
                    label = f"{emoji} {label}"
                pols.append({"id": pid, "label": label})
    if not pols:
        pols = [
            {"id": "brian", "label": "🧠 Brian (Neural Network)"},
            {"id": "forrest", "label": "🌲 Forrest (XGBoost)"},
            {"id": "logan", "label": "🪵 Logan (Logistic Regression)"},
        ]

    # Use buttons for model selection instead of dropdown
    st.subheader("🤖 Choose Your AI Opponent")
    
    # Initialize selected policy randomly if not set
    if "policy" not in st.session_state:
        st.session_state["policy"] = random.choice([p["id"] for p in pols])
    
    # Create buttons for each model
    cols = st.columns(len(pols))
    for i, pol in enumerate(pols):
        with cols[i]:
            is_selected = st.session_state.get("policy") == pol["id"]
            button_type = "primary" if is_selected else "secondary" 
            if st.button(pol["label"], key=f"policy_{pol['id']}", type=button_type):
                st.session_state["policy"] = pol["id"]
                st.rerun()
    
    # Bot Win Rate Statistics - showing how often each bot wins games
    st.markdown("### 🤖 Bot Performance")
    st.caption("All statistics reflect standard difficulty. Easy mode is retired but still stored in historical data for compatibility.")
    
    try:
        metrics_reset_raw = os.getenv("UI_METRICS_RESET_TS", "2025-10-10T08:00:00Z")
        metrics_reset_dt = parse_iso8601(metrics_reset_raw)

        games_resp = fetch_json(
            f"{API}/games",
            limit=500,
        )

        if games_resp and isinstance(games_resp, dict):
            raw_games = games_resp.get("games", []) or []
            stats = {pol["id"]: {"wins": 0, "games": 0} for pol in pols}

            for game in raw_games:
                policy = game.get("policy")
                if policy not in stats:
                    continue

                created_dt = parse_iso8601(game.get("created_ts"))
                if metrics_reset_dt and created_dt and created_dt < metrics_reset_dt:
                    continue

                stats[policy]["games"] += 1
                if game.get("winner") == "bot":
                    stats[policy]["wins"] += 1

            for pol in pols:
                bot_id = pol["id"]
                bot_name = pol["label"].split(" ", 1)[1] if " " in pol["label"] else pol["label"]
                bot_emoji = pol["label"].split(" ", 1)[0] if " " in pol["label"] else "🤖"

                bot_stats = stats.get(bot_id, {"wins": 0, "games": 0})
                wins = bot_stats["wins"]
                games = bot_stats["games"]
                win_rate = (wins / games) if games else 0.0

                st.text(f"{bot_emoji} {bot_name}: {win_rate:.1%} ({wins}/{games} games won)")

            st.caption(f"Window start · {metrics_reset_raw}")
        else:
            st.text("🤖 No bot data available yet - play some games!")

    except Exception as e:
        if st.session_state.get("show_debug"):
            st.sidebar.write(f"💥 Bot stats error: {str(e)[:50]}")
        st.text("🤖 Bot statistics will appear after playing games")
    
    # # User Statistics
    # st.markdown("### 📊 Bot Win Rate Statistics")
    
    # # Fetch bot win rates from API with caching
    # try:
    #     session_stats = fetch_json(f"{API}/stats", session_id=st.session_state.session_id)
        
    #     if session_stats and isinstance(session_stats, dict):
    #         wins = session_stats.get("wins", 0)
    #         losses = session_stats.get("losses", 0)
    #         rounds = session_stats.get("rounds", 0)
    #         games = session_stats.get("games", 0)
            
    #         if rounds > 0:
    #             win_rate = wins / rounds
    #             st.text(f"🎯 Your Overall Performance:")
    #             st.text(f"   Win Rate: {win_rate:.1%} ({wins}/{rounds} rounds)")
    #             st.text(f"   Games Played: {games}")
    #             st.text(f"   Record: {wins}W - {losses}L")
    #         else:
    #             st.text("🎯 Play some games to see your statistics!")
    #     else:
    #         st.text("🎯 Statistics will appear here after playing games")
    # except Exception as e:
    #     if st.session_state.get("show_debug"):
    #         st.sidebar.write(f"� Win rates exception: {str(e)[:50]}")
    #     st.info("🚧 Win rate tracking will be displayed here once loaded")
    
    policy = st.session_state.get("policy", "brian")

    def start_game():
        r = requests.post(
            f"{API}/start_game",
            json={
                "session_id": st.session_state.session_id,
                "user_name": name,
                "bot_policy": policy,
            },
            timeout=6,
        )
        r.raise_for_status()
        st.session_state.game = r.json()
        st.session_state.last = None
        st.session_state.stats = fetch_stats(st.session_state.game["game_id"])
        st.session_state.roundinfo = fetch_round(st.session_state.game["game_id"])
        # Reset game finished notification for auto-refresh
        st.session_state["game_finished_notified"] = False

    col1, col2 = st.columns(2)
    col1.button("Start new game", on_click=start_game, type="primary")
    col2.button("Reset session ID", on_click=lambda: st.session_state.update({"session_id": uuid.uuid4().hex}))

    st.divider()


    # # Session stats (persist for the whole connection)
    # st.subheader("Session Stats")
    # try:
    #     sess = fetch_json(f"{API}/stats", session_id=st.session_state.session_id) or {}
    # except Exception:
    #     sess = {}

    # wins   = int(sess.get("wins") or sess.get("win_count") or 0)
    # losses = int(sess.get("losses") or sess.get("loss_count") or 0)
    # ties   = int(sess.get("ties") or sess.get("draws") or 0)
    # games  = int(sess.get("games") or sess.get("total_games") or (wins+losses+ties))
    # win_pct = (wins / games * 100) if games else 0.0

    # cA, cB, cC = st.columns(3)
    # cA.metric("Wins", wins)
    # cB.metric("Losses", losses)
    # cC.metric("Win %", f"{win_pct:.0f}%")

    # ---------- LEADERBOARD SECTION ----------
    st.subheader("🏆 Player Win Streaks")
    
    # Get policies for opponent filter buttons
    pols = load_policies()
    
    # Opponent filter buttons - optimized for better performance
    st.write("**Filter by opponent:**")
    
    # Initialize selected opponent filter
    if "leaderboard_opponent" not in st.session_state:
        st.session_state["leaderboard_opponent"] = None
    
    # Efficient button layout for opponent filter buttons
    max_buttons_per_row = 3  # Adjusted for 3 ML bots only
    filter_cols = st.columns(min(len(pols), max_buttons_per_row))
    
    # Initialize with a random opponent if not set (no "All" option)
    current_selection = st.session_state.get("leaderboard_opponent")
    if current_selection is None:
        st.session_state["leaderboard_opponent"] = random.choice([pol["id"] for pol in pols])
        current_selection = st.session_state["leaderboard_opponent"]
    
    # Individual opponent buttons with proper emojis
    emoji_map = {"brian": "🧠", "forrest": "🌲", "logan": "🪵"}
    
    for i, pol in enumerate(pols):
        if i < len(filter_cols):
            with filter_cols[i]:
                bot_emoji = emoji_map.get(pol["id"].lower(), "🤖")
                bot_name = pol["id"].title()  # Brian, Forrest, Logan
                button_label = f"{bot_emoji} {bot_name}"
                
                if st.button(button_label, key=f"opponent_{pol['id']}", 
                           use_container_width=True,
                           type="primary" if current_selection == pol["id"] else "secondary"):
                    if current_selection != pol["id"]:  # Only rerun if selection changes
                        st.session_state["leaderboard_opponent"] = pol["id"]
                        st.rerun()
    
    selected_opponent = st.session_state.get("leaderboard_opponent")
    
    # Auto-refresh checkbox
    auto_refresh = st.checkbox("🔄 Auto-refresh after games", value=True, key="leaderboard_auto_refresh")
    
    # Manual refresh button with cache clearing
    if st.button("🔄 Refresh Now", key="leaderboard_refresh"):
        # Clear any cached data to force fresh API calls
        st.rerun()
    
    # Fetch and display win streaks leaderboard with proper consecutive wins
    try:
        streak_data = fetch_json(
            f"{API}/win_streaks", 
            opponent=selected_opponent,
            limit=10  # Always show top 10
        )
        
        if streak_data == "Loading...":
            st.write("📊 Loading leaderboard...")
        elif streak_data is None:
            st.write("⚠️ Leaderboard unavailable")
        elif not isinstance(streak_data, dict) or "players" not in streak_data:
            if st.session_state.get("show_debug"):
                st.sidebar.write(f"⚠️ Leaderboard data format error: {type(streak_data)}")
            st.write("⚠️ Leaderboard data format error")
        elif not streak_data["players"]:
            st.write("📊 No players found")
        else:
            players = streak_data["players"]
            st.write("**Top 10 Win Streaks**")
            
            # Table header for streaks
            col1, col2, col3 = st.columns([3, 1, 2])
            with col1:
                st.write("**Name**")
            with col2:
                st.write("**Streak**")
            with col3:
                st.write("**Date/Time**")
            
            # Display players with actual win streaks
            for i, player in enumerate(players, 1):
                col1, col2, col3 = st.columns([3, 1, 2])
                with col1:
                    st.write(f"{i}. {player['username']}")
                with col2:
                    streak = player.get('streak', 0)
                    st.write(f"{streak}")
                with col3:
                    # Format date/time nicely
                    try:
                        dt = datetime.datetime.fromisoformat(player.get('last_win_date', '').replace('Z', '+00:00'))
                        formatted_date = dt.strftime('%m/%d %H:%M')
                    except:
                        formatted_date = player.get('last_win_date', 'N/A')[:10] if player.get('last_win_date') else 'N/A'
                    st.write(formatted_date)
            
            # Show filter info - now always shows a specific opponent
            emoji_map = {"brian": "🧠", "forrest": "🌲", "logan": "🪵"}
            opponent_emoji = emoji_map.get(selected_opponent, "🤖")
            filter_text = f"{opponent_emoji} vs {selected_opponent.title()}"
            st.caption(f"Showing win streaks for: {filter_text}")
            
    except Exception as e:
        if st.session_state.get("show_debug"):
            st.sidebar.write(f"💥 Leaderboard exception: {str(e)[:50]}")
        st.write("⚠️ Leaderboard unavailable")

    # Debug toggle (stores in session_state)
    show_debug = st.toggle("Show debug details", value=st.session_state.get("show_debug", False))
    st.session_state["show_debug"] = show_debug
    
    # Debug probabilities toggle
    show_debug_probs = st.toggle("Show bot predictions", value=st.session_state.get("show_debug_probs", False))
    st.session_state["show_debug_probs"] = show_debug_probs

    # Enhanced debug panel
    if show_debug:
        st.write("🔧 **Debug Panel**")
        st.write("**Session State Keys:**", list(st.session_state.keys()))
        st.write("**API Base:**", API)
        
        # Add debug buttons for testing API endpoints
        if st.button("🧪 Test Bot Win Rates API"):
            result = fetch_json(f"{API}/bot_win_rates")
            st.sidebar.json(result)
        
        if st.button("🧪 Test Win Streaks API"):
            result = fetch_json(f"{API}/win_streaks", limit=5)
            st.sidebar.json(result)
            
        # Show recent API calls log
        if "api_debug_log" not in st.session_state:
            st.session_state["api_debug_log"] = []
        
        if st.session_state["api_debug_log"]:
            st.write("**Recent API Calls:**")
            for log_entry in st.session_state["api_debug_log"][-5:]:  # Show last 5
                st.text(log_entry)

# ---------- MAIN CONTENT LAYOUT ----------
# Single column layout for main game area 
main_left = st.container()

# ---------- MAIN GAME AREA ----------
with main_left:
    st.title("RPS Quest")
    st.caption("Deal DMG on win • -5 HP to both on tie • 🪨 beats ✂️ • 📄 beats 🪨 • ✂️ beats 📄")

    game = st.session_state.get("game")
    if not game:
        st.info("Use the **Settings** sidebar to start a game.")
        st.stop()

    gid = game["game_id"]
if "roundinfo" not in st.session_state:
    st.session_state.roundinfo = fetch_round(gid)
if "stats" not in st.session_state:
    st.session_state.stats = fetch_stats(gid)

# Animation state management
if "animation_state" not in st.session_state:
    st.session_state.animation_state = "idle"  # idle, countdown_1, countdown_2, countdown_3, shoot
if "pending_user_move" not in st.session_state:
    st.session_state.pending_user_move = None
if "animation_start_time" not in st.session_state:
    st.session_state.animation_start_time = None

# ---------- PLAY PANEL ----------
play = st.container()
with play:
    # Use stats from session_state (like the working version)
    stats = st.session_state.stats or {}
    
    # Check if game is finished
    game_finished = stats.get("winner") is not None
    if game_finished:
        st.success(f"🎉 Game Over! Winner: {stats.get('winner', 'Unknown')}")
        st.info("Start a new game from the sidebar to continue playing.")

    # Robustly read points from /stats, regardless of shape
    def _points(d: dict, who: str) -> float:
        if not isinstance(d, dict):
            return 0.0
        # flat keys
        for k in (f"{who}_score", f"{who}_points"):
            try:
                v = d.get(k, None)
                if v is not None:
                    return float(v)
            except Exception:
                pass
        # nested keys: {"user": {"score": .., "points": ..}}
        try:
            nested = d.get(who) or {}
            for k in ("score", "points"):
                v = nested.get(k, None)
                if v is not None:
                    return float(v)
        except Exception:
            pass
        return 0.0

    # Convert points → HP (exactly like the old scoreboard but with HP bars)
    user_pts = stats.get("user_score", 0)      # Use the same approach as old code
    bot_pts = stats.get("bot_score", 0)        # Use the same approach as old code
    you_hp = hp_from_points(bot_pts)           # your HP goes down as bot_pts rises
    bot_hp = hp_from_points(user_pts)          # bot HP goes down as user_pts rises
    
    # Get last round data for damage indicators, debug, and combat display
    last = st.session_state.get("last") or {}
    
    # Debug HP calculation (only show in debug mode)
    if st.session_state.get("show_debug"):
        st.sidebar.write(f"🔍 HP Debug: user_pts={user_pts:.1f}, bot_pts={bot_pts:.1f}")
        st.sidebar.write(f"🔍 HP Values: you={you_hp}, bot={bot_hp}")
        st.sidebar.write(f"🔍 Stats: {stats}")
        st.sidebar.write(f"🔍 Last result: {last.get('result', 'None')}")
        st.sidebar.write(f"🔍 Deltas: user={last.get('user_delta', 0)}, bot={last.get('bot_delta', 0)}")

    # 1) HP Bars - using inline styles with dedicated damage space
    you_color = hp_color(you_hp)
    bot_color = hp_color(bot_hp)
    
    # Damage blips with consistent spacing
    last = st.session_state.get("last") or {}
    
    # HP damage blips based on opponent's point gain (since HP is inverted)
    # Your HP goes down when bot gains points, bot's HP goes down when you gain points
    user_delta = float(last.get("user_delta") or 0)  # Your point gain
    bot_delta = float(last.get("bot_delta") or 0)    # Bot's point gain
    
    # HP damage = opponent's points gained * 10 (since HP is inverted)
    dmg_you = int(round(10 * bot_delta)) if bot_delta > 0 else 0    # Your HP loss from bot gaining points
    dmg_bot = int(round(10 * user_delta)) if user_delta > 0 else 0  # Bot's HP loss from you gaining points
    
    # Show damage blips when there's actual damage
    you_blip = f'<span class="rps-hp-blip">-{dmg_you}</span>' if dmg_you > 0 else ''
    bot_blip = f'<span class="rps-hp-blip">-{dmg_bot}</span>' if dmg_bot > 0 else ''
    
    # Get names for health bar labels
    player_name = st.session_state.get("username", "You")
    bot_name = game.get("bot_policy", "Bot").title()
    bot_emoji = bot_avatar(game.get("bot_policy", "unknown"))
    
    st.markdown(f"""
    <div class="rps-hp">
      <div class="rps-hp-row">
        <div class="rps-hp-label">🙂 {player_name}</div>
        <div class="rps-hp-track" style="width: {you_hp}%; max-width: 100%;">
          <div class="rps-hp-fill" style="background: {you_color};"></div>
        </div>
        <div class="rps-hp-num">{you_hp}/100</div>
        <div class="rps-hp-damage">{you_blip}</div>
      </div>
      <div class="rps-hp-row">
        <div class="rps-hp-label">{bot_emoji} {bot_name}</div>
        <div class="rps-hp-track" style="width: {bot_hp}%; max-width: 100%;">
          <div class="rps-hp-fill" style="background: {bot_color};"></div>
        </div>
        <div class="rps-hp-num">{bot_hp}/100</div>
        <div class="rps-hp-damage">{bot_blip}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add the old-style scoreboard for debugging to see if THAT updates
    if st.session_state.get("show_debug"):
        st.markdown(f"""
        <div style="background: var(--secondary-background-color); padding: 1rem; margin: 1rem 0; border-radius: 0.5rem;">
            <strong>Debug Info:</strong><br>
            Callback count: {st.session_state.get("debug_callback_count", 0)}<br>
            Last update: {st.session_state.get("debug_last_update", "None")}<br>
            Current stats: {stats}<br>
            <br>
            <strong>HP Debug:</strong><br>
            You HP: {you_hp} (from bot_pts: {bot_pts})<br>
            Bot HP: {bot_hp} (from user_pts: {user_pts})<br>
            You color: {you_color}<br>
            Bot color: {bot_color}<br>
            <br>
            <strong>CSS Variable Test:</strong><br>
            <div style="width: 200px; height: 20px; background: #333; margin: 5px 0;">
                <div style="--test-width: {you_hp}; width: calc(var(--test-width) * 1%); height: 100%; background: red;">CSS Var Test</div>
            </div>
            <br>
            <strong>Inline Style Test:</strong><br>
            <div style="width: 200px; height: 20px; background: #333; margin: 5px 0;">
                <div style="width: {you_hp}%; height: 100%; background: {you_color};">Inline Test</div>
            </div>
            <br>
            <strong>Current HP Bar HTML (what we're generating):</strong><br>
            <code>
            &lt;div class="rps-hp-track" style="--hp-percent:{you_hp}; --hp-color:{you_color};"&gt;<br>
            &nbsp;&nbsp;&lt;div class="rps-hp-fill"&gt;&lt;/div&gt;<br>
            &lt;/div&gt;
            </code>
            <br>
            <strong>Old-style scoreboard:</strong><br>
            You <span style="color: var(--primary-color); font-weight: bold;">{stats.get("user_score", 0)}</span> — 
            Bot <span style="font-weight: bold;">{stats.get("bot_score", 0)}</span>
            {f'<br><strong>Game over — {stats["winner"]} wins!</strong>' if stats.get("winner") else ""}
        </div>
        """, unsafe_allow_html=True)


    # 2) Combat Area - Player vs Bot with choice emojis
    result = (last.get("result") or "").lower()

    if result == "win":
        player_face = "🙂"
    elif result == "loss":
        player_face = "😔"
    elif result in ("draw", "tie"):
        player_face = "😐"
    else:
        player_face = "🙂"  # Default for new game

    user_choice = (last.get("user_move") or "").lower()
    bot_choice  = (last.get("bot_move")  or "").lower()

    def move_to_emoji(move):
        return {"rock":"🪨", "paper":"📄", "scissors":"✂️"}.get(move, "")

    user_emoji = move_to_emoji(user_choice)
    bot_emoji  = move_to_emoji(bot_choice)

    def dmg(v):
        try:
            return int(round(float(v) * 10))
        except Exception:
            return 10

    # Helper: is this side a non-winner (i.e., should be struck)?
    def is_non_winner(side: str, res: str) -> bool:
        if res in ("draw", "tie"):   # both are non-winning
            return True
        if res == "win":             # user won -> bot is non-winner
            return side == "bot"
        if res in ("loss", "lose"):  # user lost -> user is non-winner (handle both API formats)
            return side == "user"
        return False                 # no result yet

    # Build damage badges (or invisible placeholders on first render)
    user_dmg_html = ""
    bot_dmg_html  = ""

    if user_choice and bot_choice and "round_points" in (last or {}):
        rp = last.get("round_points") or {}
        user_damage_val = dmg(float(rp.get(user_choice, 1.0)))
        bot_damage_val  = dmg(float(rp.get(bot_choice, 1.0)))

        # Color scale based on last round’s R/P/S damages
        all_damages = [dmg(float(rp.get("rock", 1.0))), dmg(float(rp.get("paper", 1.0))), dmg(float(rp.get("scissors", 1.0)))]
        damage_colors_last = damage_badge_colors(all_damages)

        u_bg, u_text = damage_colors_last.get(user_damage_val, ("var(--secondary-background-color)", "var(--text-color)"))
        b_bg, b_text = damage_colors_last.get(bot_damage_val,  ("var(--secondary-background-color)", "var(--text-color)"))

        u_lose = is_non_winner("user", result)
        b_lose = is_non_winner("bot",  result)

        def badge(val, bg, text, losing):
            cls = "rps-dmg-badge" + (" losing" if losing else "")
            # Force unique key to prevent Streamlit caching issues
            unique_id = f"dmg-{val}-{losing}-{hash(str((val, bg, text, losing)))}"
            return f'<div class="{cls}" id="{unique_id}" style="background:{bg};color:{text};">DMG {val}</div>'

        user_dmg_html = badge(user_damage_val, u_bg, u_text, u_lose)
        bot_dmg_html  = badge(bot_damage_val,  b_bg, b_text, b_lose)
    else:
        # First render: reserve space without showing text (prevents layout jump and any raw HTML flashes)
        placeholder = '<div class="rps-dmg-badge" style="visibility:hidden;">DMG 0</div>'
        user_dmg_html = placeholder
        bot_dmg_html  = placeholder

    # Render the combat grid (ONLY keep this block; remove the old `<br>`-based combat block)
    bot_avatar_emoji = bot_avatar(game.get("bot_policy", "unknown"))
    bot_name = game.get("bot_policy", "Bot").title()
    player_name = st.session_state.get("username", "You")
    
    st.markdown(f"""
    <div class="rps-combat">
    <div class="rps-combat-grid">
        <div class="rps-combat-side">
        <div style="text-align: center; font-weight: bold; margin-bottom: 0.5rem; color: var(--text-color);">{player_name}</div>
        <div class="rps-combat-emoji">{player_face}{("-" + user_emoji) if user_emoji else ""}</div>
        {user_dmg_html}
        </div>
        <div class="rps-combat-vs">VS</div>
        <div class="rps-combat-side">
        <div style="text-align: center; font-weight: bold; margin-bottom: 0.5rem; color: var(--text-color);">{bot_name}</div>
        <div class="rps-combat-emoji">{(bot_emoji + "-") if bot_emoji else ""}{bot_avatar_emoji}</div>
        {bot_dmg_html}
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)


    # 3) Picks + Outcome (centered)
    def pretty_move(m):
        m = (m or "").lower()
        return {"rock":"🪨 Rock","paper":"📄 Paper","scissors":"✂️ Scissors"}.get(m, "—")

    # def outcome_block():
    #     res = (last.get("result") or "").lower()
    #     if res == "win":   return '<div class="rps-outcome win"><span class="chip">You Win!</span></div>'
    #     if res == "loss":  return f'<div class="rps-outcome"><span class="chip">{game.get("bot_policy","Bot")} Wins!</span></div>'
    #     if res in ("draw","tie"): return '<div class="rps-outcome tie"><span class="chip">Tie!</span></div>'
    #     return ""

    # Center the picks display
    st.markdown(f'''
    <div style="text-align: center; margin: 1rem 0;">
      <div class="rps-picks">
        You Picked: <b>{pretty_move(last.get("user_move"))}</b>
        &nbsp;·&nbsp;
        Bot Picked: <b>{pretty_move(last.get("bot_move"))}</b>
      </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # # Center the outcome
    # outcome_html = outcome_block()
    # if outcome_html:
    #     st.markdown(f'<div style="text-align: center;">{outcome_html}</div>', unsafe_allow_html=True)

    # 4) Game state handling - buttons or game over
    game_over = you_hp <= 0 or bot_hp <= 0 or game_finished
    
    if game_over:
        # Auto-refresh leaderboard if enabled and this is the first time seeing this game end
        if (st.session_state.get("leaderboard_auto_refresh", True) and 
            not st.session_state.get("game_finished_notified", False)):
            st.session_state["game_finished_notified"] = True
            # Force a refresh by clearing any cached data (streamlit will rerun)
            if "leaderboard_last_refresh" in st.session_state:
                del st.session_state["leaderboard_last_refresh"]
        
        # Game over state - show winner and restart options
        winner_text = "You Win!" if bot_hp <= 0 else ("Bot Wins!" if you_hp <= 0 else f"{stats.get('winner', 'Someone')} Wins!")
        
        st.markdown(f"""
        <div class="rps-game-over">
            <h2>🎉 Game Over!</h2>
            <h3>{winner_text}</h3>
            <p>Final Scores: You {user_pts:.1f} - Bot {bot_pts:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Restart options (centered)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            def restart_game():
                st.session_state.update({
                    "game": None, 
                    "last": None, 
                    "stats": None, 
                    "roundinfo": None,
                    "game_finished_notified": False
                })
            st.button("🎮 Start New Game", key="restart_game", use_container_width=True, on_click=restart_game)
            
            # Quick opponent selection
            pols = load_policies()
            labels = [p["label"] for p in pols]
            ids = [p["id"] for p in pols]
            current_idx = next((i for i, p in enumerate(pols) if p["id"] == st.session_state.get("policy")), 0)
            
            new_opponent_idx = st.selectbox("Choose New Opponent", range(len(labels)), 
                                          format_func=lambda i: labels[i], index=current_idx, key="new_opponent")
            if st.button("🔄 Change Opponent & Start", key="change_opponent", use_container_width=True):
                st.session_state["policy"] = ids[new_opponent_idx]
                st.session_state.update({"game": None, "last": None, "stats": None, "roundinfo": None})
                st.rerun()
    else:
        # Normal game state - show damage and buttons
        #st.markdown('<div style="text-align: center; margin: 1.5rem 0;"><h3>Damage This Round</h3></div>', unsafe_allow_html=True)
        
        rp = (st.session_state.roundinfo or {}).get("round_points", {}) or {}

        def dmg(v):
            try:
                return int(round(float(v) * 10))
            except Exception:
                return 10

        # Calculate damage values and colors
        rock_dmg = dmg(rp.get("rock", 1.0))
        paper_dmg = dmg(rp.get("paper", 1.0))
        scissors_dmg = dmg(rp.get("scissors", 1.0))
        
        damage_colors = damage_badge_colors([rock_dmg, paper_dmg, scissors_dmg])

        def do_play(move: str):
            r = requests.post(f"{API}/play", json={"game_id": gid, "user_move": move}, timeout=6)
            r.raise_for_status()
            st.session_state.last = r.json()
            # Refresh stats immediately (like the working version)
            old_stats = st.session_state.stats.copy() if st.session_state.stats else {}
            st.session_state.stats = fetch_stats(gid)
            new_stats = st.session_state.stats or {}
            
            # Debug: Force a debug update to see if this callback is running
            st.session_state.debug_callback_count = st.session_state.get("debug_callback_count", 0) + 1
            st.session_state.debug_last_update = f"Old: {old_stats.get('user_score', 'N/A')}-{old_stats.get('bot_score', 'N/A')}, New: {new_stats.get('user_score', 'N/A')}-{new_stats.get('bot_score', 'N/A')}"
            
            # Fetch next round's values unless the game finished
            if not st.session_state.stats.get("winner"):
                st.session_state.roundinfo = fetch_round(gid)

        # Debug Probabilities Section (controlled by sidebar toggle)
        if st.session_state.get("show_debug_probs", False):
            st.markdown("**🤖 Bot Prediction Probabilities**")
            try:
                bot_preds = fetch_bot_predictions(gid)
                if "probabilities" in bot_preds:
                    probs = bot_preds["probabilities"]
                    pred_move = bot_preds.get("predicted_move", "unknown")
                    bot_policy = bot_preds.get("bot_policy", "unknown")
                    model_type = bot_preds.get("model_type", "unknown")
                    
                    st.markdown(f"**Bot:** {bot_policy.title()} (*{model_type}*)")
                    
                    # Display probabilities as bars
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🪨 Rock", f"{probs['rock']:.1%}")
                    with col2:
                        st.metric("📄 Paper", f"{probs['paper']:.1%}")
                    with col3:
                        st.metric("✂️ Scissors", f"{probs['scissors']:.1%}")
                    
                    st.markdown(f"**Predicted:** {pred_move.title()} (Round {bot_preds.get('round_number', '?')})")
                elif "error" in bot_preds:
                    st.warning(f"⚠️ {bot_preds['error']}")
                else:
                    st.info("🤖 Bot predictions not available for this policy")
            except Exception as e:
                st.error(f"⚠️ Failed to fetch bot predictions: {e}")

        def damage_badge(damage_val, move_name):
            bg_color, text_color = damage_colors.get(damage_val, ("var(--secondary-background-color)", "var(--text-color)"))
            return (
                f'<div class="rps-badge" '
                f'style="--dmg-bg-color:{bg_color}; --dmg-text-color:{text_color}; background:{bg_color}; color:{text_color}; margin-top:8px;">'
                f'DMG&nbsp;{damage_val}</div>'
            )

        # # Create centered button layout with current round damage values next to buttons
        # st.markdown('<div style="margin: 1.5rem 0;">', unsafe_allow_html=True)
        # col1, col2, col3 = st.columns([1, 2, 1])  # Center the buttons
        # with col2:
        #     # Rock
        #     bcol, dcol = st.columns([3, 1])
        #     with bcol:
        #         st.button(f"🪨 Rock", key=f"btn_rock", on_click=do_play, args=("rock",), use_container_width=True)
        #     with dcol:
        #         st.markdown(damage_badge(rock_dmg, "rock"), unsafe_allow_html=True)
            
        #     # Paper  
        #     bcol, dcol = st.columns([3, 1])
        #     with bcol:
        #         st.button(f"📄 Paper", key=f"btn_paper", on_click=do_play, args=("paper",), use_container_width=True)
        #     with dcol:
        #         st.markdown(damage_badge(paper_dmg, "paper"), unsafe_allow_html=True)
            
        #     # Scissors
        #     bcol, dcol = st.columns([3, 1])
        #     with bcol:
        #         st.button(f"✂️ Scissors", key=f"btn_scissors", on_click=do_play, args=("scissors",), use_container_width=True)
        #     with dcol:
        #         st.markdown(damage_badge(scissors_dmg, "scissors"), unsafe_allow_html=True)
        # st.markdown('</div>', unsafe_allow_html=True)

        def triangle_controls():
            # Wrapper so the SVG can overlay the whole control area
            st.markdown('<div class="rps-tri-wrap">', unsafe_allow_html=True)

            # ── Top Row: Rock centered, damage on its right ─────────────────────────────
            top = st.columns([1, 2, 1])
            with top[1]:
                inner = st.columns([5, 2])            # [button][dmg to the side]
                with inner[0]:
                    st.button("🪨 Rock", key="btn_rock", on_click=do_play, args=("rock",), use_container_width=True)
                with inner[1]:
                    st.markdown(f'<div class="rps-side-col">{damage_badge(rock_dmg, "rock")}</div>', unsafe_allow_html=True)

            # ── SVG overlay with arrows + “beats” text (non-interactive) ────────────────
            st.markdown("""
            <svg class="rps-tri-svg" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
            <defs>
                <marker id="rpsArrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
                <path d="M0,0 L8,4 L0,8 z" fill="currentColor"></path>
                </marker>
            </defs>

            <!-- Rock (50,16) -> Scissors (86,86) -->
            <line class="edge" x1="50" y1="16" x2="86" y2="86"></line>
            <text class="cap" x="69" y="46">beats</text>

            <!-- Scissors (86,86) -> Paper (14,86) -->
            <line class="edge" x1="86" y1="86" x2="14" y2="86"></line>
            <text class="cap" x="50" y="92" text-anchor="middle">beats</text>

            <!-- Paper (14,86) -> Rock (50,16) -->
            <line class="edge" x1="14" y1="86" x2="50" y2="16"></line>
            <text class="cap" x="31" y="46">beats</text>
            </svg>
            """, unsafe_allow_html=True)

            # ── Bottom Row: Paper left (damage outside), Scissors right (damage outside) ─
            bot = st.columns([1, 1, 1])

            # Paper (left): damage on the far left, button to the right
            with bot[0]:
                inner = st.columns([2, 5])            # [dmg][button]
                with inner[0]:
                    st.markdown(f'<div class="rps-side-col">{damage_badge(paper_dmg, "paper")}</div>', unsafe_allow_html=True)
                with inner[1]:
                    st.button("📄 Paper", key="btn_paper", on_click=do_play, args=("paper",), use_container_width=True)

            # Scissors (right): button first, damage on the far right
            with bot[2]:
                inner = st.columns([5, 2])            # [button][dmg]
                with inner[0]:
                    st.button("✂️ Scissors", key="btn_scissors", on_click=do_play, args=("scissors",), use_container_width=True)
                with inner[1]:
                    st.markdown(f'<div class="rps-side-col">{damage_badge(scissors_dmg, "scissors")}</div>', unsafe_allow_html=True)

            # Accessible text for screen readers
            st.markdown(
                '<span class="sr-only">Rock beats Scissors. Scissors beats Paper. Paper beats Rock.</span>',
                unsafe_allow_html=True
            )

            st.markdown('</div>', unsafe_allow_html=True)

        # Use the triangle
        triangle_controls()




# ---------- BOTTOM ROW: Game / Last / Recent+Session ----------
if st.session_state.get("show_debug"):
    st.markdown('<div class="rps-details-gap"></div>', unsafe_allow_html=True)
    st.markdown('<div class="rps-details">', unsafe_allow_html=True)

    # Enhanced debug layout with bot analysis
    colA, colB, colC, colD = st.columns(4)
    
    with colA:
        st.subheader("Game")
        st.write(f"Target: **{game['target_score']}**")
        st.write(f"Policy: **{game['bot_policy']}**")
        st.caption(f"Session `{st.session_state.session_id[:8]}…` · Game `{gid[:8]}…`")
        
    with colB:
        st.subheader("Bot Analysis")
        
        # Get current round info
        roundinfo = st.session_state.get("roundinfo", {})
        round_no = roundinfo.get("round_no", 1) if roundinfo else 1
        
        # Check if we're in opening gambit phase (first 3 rounds)
        if round_no <= 3:
            st.write("**Phase:** Opening Gambit")
            st.write("**Probabilities:** N/A (using gambit)")
            st.write("**Values:** N/A (gambit override)")
            
            # Show which gambit is being used if available in last result
            last = st.session_state.get("last")
            if last and "opening_name" in last:
                st.write(f"**Gambit:** {last['opening_name']}")
        else:
            st.write(f"**Phase:** ML Prediction (Round {round_no})")
            
            # Fetch bot predictions and values
            try:
                bot_preds = fetch_bot_predictions(gid)
                if "probabilities" in bot_preds:
                    probs = bot_preds["probabilities"]
                    st.write("**User Move Probabilities:**")
                    st.write(f"🪨 Rock: {probs['rock']:.1%}")
                    st.write(f"📄 Paper: {probs['paper']:.1%}")
                    st.write(f"✂️ Scissors: {probs['scissors']:.1%}")
                    
                    # Calculate expected values for each bot move
                    stats = st.session_state.stats or {}
                    our_score = stats.get("bot_score", 0)  # bot's score
                    opponent_score = stats.get("user_score", 0)  # user's score
                    
                    rp = roundinfo.get("round_points", {}) if roundinfo else {}
                    round_values = {
                        "rock": rp.get("rock", 1.0),
                        "paper": rp.get("paper", 1.0), 
                        "scissors": rp.get("scissors", 1.0)
                    }
                    
                    st.write("**Expected Values:**")
                    
                    # Import the value calculation function (we'll need to make it available)
                    # For now, let's do a simplified calculation here
                    move_values = {}
                    for bot_move in ["rock", "paper", "scissors"]:
                        # What user move does this bot move beat?
                        beaten_user_move = {"rock": "scissors", "paper": "rock", "scissors": "paper"}[bot_move]
                        # What user move beats this bot move?
                        losing_user_move = {"rock": "paper", "paper": "scissors", "scissors": "rock"}[bot_move]
                        
                        # Expected value calculation (updated with tie bonus)
                        p_beat = probs.get(beaten_user_move, 0.0)
                        p_lose = probs.get(losing_user_move, 0.0)
                        p_tie = probs.get(bot_move, 0.0)  # Probability of tying
                        our_damage = round_values.get(bot_move, 1.0)
                        opponent_damage = round_values.get(losing_user_move, 1.0)
                        
                        # Win bonuses
                        win_bonus = 10.0 if (our_score + our_damage) >= 10.0 else 0.0
                        lose_penalty = 10.0 if (opponent_score + opponent_damage) >= 10.0 else 0.0
                        tie_bonus = 0.5  # Both players gain 0.5 points on ties
                        
                        expected_value = p_beat * (our_damage + win_bonus) - p_lose * (opponent_damage + lose_penalty) + p_tie * tie_bonus
                        move_values[bot_move] = expected_value
                    
                    # Display values with best choice highlighted
                    best_move = max(move_values.keys(), key=lambda m: move_values[m])
                    for move, value in move_values.items():
                        emoji = {"rock": "🪨", "paper": "📄", "scissors": "✂️"}[move]
                        marker = " ⭐" if move == best_move else ""
                        st.write(f"{emoji} {move.title()}: {value:.2f}{marker}")
                        
                elif "error" in bot_preds:
                    st.write(f"**Error:** {bot_preds['error']}")
                else:
                    st.write("**Probabilities:** Not available")
                    st.write("**Values:** Cannot calculate")
                    
            except Exception as e:
                st.write(f"**Error fetching bot data:** {e}")
                
    with colC:
        st.subheader("Last round")
        last = st.session_state.get("last")
        if last and "round_points" in last:
            rpp = last["round_points"]
            st.write(f"Played values: R={rpp['rock']:.1f} · P={rpp['paper']:.1f} · S={rpp['scissors']:.1f}")
        st.json(last or {"hint":"click a move"})
        
    with colD:
        st.subheader("Recent rounds")
        try:
            recent = fetch_json(f"{API}/recent", game_id=gid, limit=5) or []
            for r in recent:
                rp = r.get("round_points", {})
                rp_str = f" [R={rp.get('rock','?'):.1f}, P={rp.get('paper','?'):.1f}, S={rp.get('scissors','?'):.1f}]"
                st.write(f"R{r.get('round_no', '?')}: **{r['user_move']}** vs **{r['bot_move']}** → {r['result']}")
                st.caption(f"{rp_str} | +you {r['user_delta']:.1f} / +bot {r['bot_delta']:.1f}")
        except Exception as e:
            st.caption(f"(no recent yet) {e}")

        st.subheader("Session stats")
        try:
            sess = fetch_json(f"{API}/stats", session_id=st.session_state.session_id) or {}
            wins = sess.get("wins", 0)
            losses = sess.get("losses", 0)
            games = sess.get("games", 0)
            st.write(f"Games: {games} | W: {wins} | L: {losses}")
            if games > 0:
                win_pct = wins / games * 100
                st.write(f"Win rate: {win_pct:.1f}%")
        except Exception as e:
            st.caption(f"(session stats error) {e}")

    st.markdown('</div>', unsafe_allow_html=True)
