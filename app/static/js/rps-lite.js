'use strict';

(function() {
  const root = document.getElementById('rps-lite-app');
  if (!root) {
    return;
  }

  const PST_TIMEZONE = 'America/Los_Angeles';
  const SESSION_STORAGE_KEY = 'rps-lite-session';
  const SESSION_DAY_KEY = 'rps-lite-session-day';

  function getPstDateParts(date = new Date()) {
    try {
      const formatter = new Intl.DateTimeFormat('en-US', {
        timeZone: PST_TIMEZONE,
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      });
      const parts = formatter.formatToParts(date);
      const lookup = {};
      parts.forEach(({ type, value }) => {
        if (type !== 'literal') {
          lookup[type] = value;
        }
      });
      return {
        year: Number(lookup.year),
        month: Number(lookup.month),
        day: Number(lookup.day),
        hour: Number(lookup.hour),
        minute: Number(lookup.minute),
        second: Number(lookup.second)
      };
    } catch (err) {
      console.warn('[rps-lite] Failed to compute PST date parts:', err);
      return null;
    }
  }

  function getPstDateKey(date = new Date()) {
    const parts = getPstDateParts(date);
    if (!parts) {
      return null;
    }
    const y = String(parts.year).padStart(4, '0');
    const m = String(parts.month).padStart(2, '0');
    const d = String(parts.day).padStart(2, '0');
    return `${y}-${m}-${d}`;
  }

  function secondsUntilNextPstMidnight(date = new Date()) {
    const parts = getPstDateParts(date);
    if (!parts) {
      return 24 * 60 * 60;
    }
    const secondsPerDay = 24 * 60 * 60;
    const secondsPassed = (parts.hour * 3600) + (parts.minute * 60) + parts.second;
    const remaining = secondsPerDay - secondsPassed;
    return remaining > 0 ? remaining : secondsPerDay;
  }

  function ensureDailySessionFreshness() {
    if (!window?.localStorage) {
      return null;
    }
    try {
      const todayKey = getPstDateKey();
      if (!todayKey) {
        return null;
      }
      const storedDay = localStorage.getItem(SESSION_DAY_KEY);
      if (storedDay && storedDay !== todayKey) {
        localStorage.removeItem(SESSION_STORAGE_KEY);
      }
      localStorage.setItem(SESSION_DAY_KEY, todayKey);
      return todayKey;
    } catch (err) {
      console.warn('[rps-lite] Unable to rotate session code daily:', err);
      return null;
    }
  }

  ensureDailySessionFreshness();

  let initialSessionId = null;
  try {
    initialSessionId = localStorage.getItem(SESSION_STORAGE_KEY);
  } catch (err) {
    console.warn('[rps-lite] Unable to read stored session id:', err);
  }

  const apiRoot = (root.dataset.apiBase || '').replace(/\/?$/, '');
  const defaultDebugEnabled = root.dataset.defaultDebug === 'true';
  const metricsResetIso = root.dataset.metricsReset || '';
  const metricsResetDate = parseIsoTimestamp(metricsResetIso);
  const byId = (id) => document.getElementById(id);
  const queryAll = (selector) => Array.from(document.querySelectorAll(selector));

  const elements = {
    nameInput: byId('player-name'),
    playerCode: byId('player-code'),
  policyButtons: byId('policy-buttons'),
    startBtn: byId('start-game'),
    endBtn: byId('end-game'),
    resetBtn: byId('reset-session'),
    status: byId('status-banner'),
  dataHealthBanner: byId('data-health-banner'),
    monitoringButton: byId('open-monitoring'),
    monitoringHint: byId('monitoring-hint'),
  leaderboardControls: byId('leaderboard-controls'),
  leaderboardList: byId('leaderboard-list'),
  leaderboardMeta: byId('leaderboard-meta'),
  leaderboardRefresh: byId('leaderboard-refresh'),
    toggleDebug: byId('toggle-debug'),
    debugTestBotRates: byId('debug-test-bot-rates'),
    debugTestStreaks: byId('debug-test-streaks'),
    scoreboard: byId('scoreboard'),
    targetScore: byId('target-score'),
    gameId: byId('game-id'),
    emptyState: byId('empty-state'),
    playerHpFill: byId('player-hp-fill'),
    botHpFill: byId('bot-hp-fill'),
    playerHpLabel: byId('player-hp'),
    botHpLabel: byId('bot-hp'),
    playerNameDisplay: byId('player-name-display'),
    playerDamage: byId('player-damage'),
    botDamage: byId('bot-damage'),
    winnerBanner: byId('winner-banner'),
    battleArena: byId('battle-arena'),
    gameOverPanel: byId('game-over-leaderboard'),
    gameOverSummary: byId('game-over-summary'),
  gameOverControls: byId('leaderboard-gameover-controls'),
  gameOverList: byId('leaderboard-gameover-list'),
  gameOverMeta: byId('leaderboard-gameover-meta'),
    gameOverRefresh: byId('leaderboard-gameover-refresh'),
    roundSummary: byId('round-summary'),
    lastRoundDetail: byId('last-round-detail'),
    historyList: byId('recent-history'),
    historyCard: byId('history-card'),
    recentHistoryMeta: byId('recent-history-meta'),
    sessionStatsSummary: byId('session-stats-summary'),
    playerFace: byId('player-face'),
    botFace: byId('bot-face'),
    botName: byId('bot-name'),
    playerMove: byId('player-move'),
    botMove: byId('bot-move'),
    playerDmgChip: byId('player-dmg-chip'),
    botDmgChip: byId('bot-dmg-chip'),
    playerNameLabel: byId('player-name-label'),
    botNameLabel: byId('bot-name-label'),
    damagePills: queryAll('.dmg-pill'),
    moveControls: byId('move-controls'),
    moveTriangle: byId('move-triangle'),
    triangleSvg: byId('triangle-svg'),
    moveButtons: queryAll('button[data-move]'),
    debugPanel: byId('debug-panel'),
    analysisGrid: byId('analysis-grid'),
    analysisTarget: byId('analysis-target'),
    analysisPolicy: byId('analysis-policy'),
    analysisSession: byId('analysis-session'),
    analysisGameId: byId('analysis-game-id'),
    analysisPhase: byId('analysis-phase'),
    analysisBotProbs: byId('analysis-bot-probs'),
    analysisExpected: byId('analysis-expected'),
    analysisPredicted: byId('analysis-predicted'),
    analysisLastJson: byId('analysis-last-json'),
    analysisSessionStats: byId('analysis-session-stats'),
    analysisApiLog: byId('analysis-api-log'),
    apiDebugLog: byId('api-debug-log')
  };

  const state = {
    sessionId: initialSessionId || null,
    gameId: null,
    policy: null,
    leaderboardOpponent: null,
    leaderboardPlayers: [],
    showDebug: defaultDebugEnabled,
    botPolicies: [],
    policyStatRefs: new Map(),
    lastRound: null,
    gameFinishedNotified: false,
    roundInfo: null,
    gameStats: null,
    sessionStats: null,
    apiLog: [],
    botProbabilities: null,
    botPredictionMeta: null,
    targetScore: null,
    grafanaUrl: root.dataset.grafanaUrl || '',
  playerCode: null,
  playerName: null,
    metricsResetIso,
    metricsResetDate,
    dataHealthKey: null,
    dataHealthDetails: [],
    botRatesRefreshToken: 0,
    latestBotRates: null
  };

  const BOT_PERFORMANCE_REFRESH_MS = 60 * 1000;

    if (!state.metricsResetIso || !state.metricsResetDate) {
      updateDataHealthBanner([]);
    }

  const emojiMap = {
    brian: '🧠',
    forrest: '🌲',
    logan: '🪵'
  };

  const moveEmoji = {
    rock: '🪨',
    paper: '📄',
    scissors: '✂️'
  };

  const botNames = {
    brian: 'Brian',
    forrest: 'Forrest', 
    logan: 'Logan'
  };
  const legacyPlayerNames = new Set(['ace', 'bob', 'cal', 'dan', 'edd', 'fox', 'gus', 'hal']);
  function isLegacyPlayerName(name) {
    if (!name) return false;
    const normalized = name.trim().toLowerCase();
    return legacyPlayerNames.has(normalized);
  }

  function isSimulatedLeaderboardName(name) {
    if (!name) return false;
    const trimmed = name.trim();
    const lower = trimmed.toLowerCase();
    if (/^sim(?:\[\d+\]|\d+|\(\d+\))$/i.test(trimmed)) {
      return true;
    }
    if (lower.startsWith('sim') && /\d/.test(lower)) {
      return true;
    }
    if (lower.startsWith('simp') && /\d/.test(lower)) {
      return true;
    }
    return lower.startsWith('sim[');
  }

  function shouldExcludeLeaderboardPlayer(name) {
    return isLegacyPlayerName(name) || isSimulatedLeaderboardName(name);
  }

  function currentBotDisplayName() {
    const policyId = state.policy;
    if (!policyId) return 'Bot';
    if (botNames[policyId]) {
      return botNames[policyId];
    }
    const match = state.botPolicies.find(p => p.id === policyId);
    if (!match || !match.label) {
      return 'Bot';
    }
    const label = match.label.trim();
    if (!label) return 'Bot';
    const withoutParen = label.replace(/\s*\(.*?\)\s*$/, '').trim();
    return withoutParen || label;
  }

  function formatBotLabel(policyId, fallback = 'Bot') {
    if (!policyId) return fallback;
    if (botNames[policyId]) {
      return botNames[policyId];
    }
    const match = state.botPolicies.find(p => p.id === policyId);
    if (!match || !match.label) {
      return fallback;
    }
    const label = match.label.trim();
    if (!label) return fallback;
    const withoutParen = label.replace(/\s*\(.*?\)\s*$/, '').trim();
    return withoutParen || label;
  }

  function setupMonitoringLink() {
    if (!elements.monitoringButton || !elements.monitoringHint) return;
    if (state.grafanaUrl) {
      elements.monitoringButton.disabled = false;
      elements.monitoringHint.textContent = 'Opens Grafana metrics in a new tab.';
    } else {
      elements.monitoringButton.disabled = true;
      elements.monitoringHint.textContent = 'Set GRAFANA_DASHBOARD_URL to enable the monitoring shortcut.';
    }
  }

  function parseIsoTimestamp(value) {
    if (!value) return null;
    const normalized = value.endsWith('Z') || value.includes('+') ? value : `${value}Z`;
    const parsed = new Date(normalized);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  function generatePlayerCode(sessionId) {
    if (!sessionId) return '---';
    // Generate a 3-character code from the session ID
    // Use a simple hash to create a consistent code for the same session
    let hash = 0;
    for (let i = 0; i < sessionId.length; i++) {
      hash = ((hash << 5) - hash) + sessionId.charCodeAt(i);
      hash = hash & hash; // Convert to 32-bit integer
    }
    // Convert to base-36 and take 3 characters
    const code = Math.abs(hash).toString(36).toUpperCase().padStart(3, '0').slice(0, 3);
    return code;
  }

  function apiUrl(path, params) {
    const normalized = path.startsWith('/') ? path : `/${path}`;
    const base = apiRoot || '';
    const url = base ? `${base}${normalized}` : normalized;
    if (!params) return url;
    const usp = new URLSearchParams(params);
    return `${url}?${usp.toString()}`;
  }

  async function fetchJson(path, params) {
    const url = apiUrl(path, params);
    const started = performance.now();
    try {
      const res = await fetch(url, { headers: { Accept: 'application/json' }, cache: 'no-store' });
      const elapsed = Math.round(performance.now() - started);
      recordApiLog({
        ts: new Date(),
        method: 'GET',
        url,
        params,
        status: res.status,
        ok: res.ok,
        elapsed
      });
      if (!res.ok) throw new Error(`${res.status}`);
      return res.json();
    } catch (err) {
      const elapsed = Math.round(performance.now() - started);
      recordApiLog({
        ts: new Date(),
        method: 'GET',
        url,
        params,
        status: err?.status || 'ERR',
        ok: false,
        elapsed,
        error: err.message
      });
      debugLog(`Fetch failed ${url}: ${err.message}`);
      throw err;
    }
  }

  function setStatus(message, type = '') {
    if (!elements.status) return;
    elements.status.textContent = message || '';
    elements.status.className = `status-banner ${type}`.trim();
  }

  function setWinner(message, ok = false) {
    if (!elements.winnerBanner) return;
    elements.winnerBanner.textContent = message || '';
    elements.winnerBanner.className = `winner-banner ${ok ? 'ok' : ''}`.trim();
  }

  function recordApiLog(details) {
    state.apiLog.push(details);
    if (state.apiLog.length > 50) {
      state.apiLog.splice(0, state.apiLog.length - 50);
    }
    renderApiLog();
  }

  function updateDataHealthBanner(issues = []) {
    const messages = issues.filter(Boolean);
    const key = messages.join('|') || 'OK';
    if (state.dataHealthKey === key) {
      return false;
    }

    state.dataHealthKey = key;
    state.dataHealthDetails = messages.slice();

    if (messages.length && state.showDebug) {
      messages.forEach(msg => console.warn('[rps-lite] data quality note:', msg));
      debugLog(`ℹ️ Data quality note: ${messages.join(' | ')}`);
    }

    if (elements.dataHealthBanner) {
      elements.dataHealthBanner.hidden = true;
      elements.dataHealthBanner.textContent = '';
      elements.dataHealthBanner.removeAttribute('title');
      elements.dataHealthBanner.classList.remove('fail', 'warn');
    }

    return false;
  }

  function toggleMoveButtons(enabled) {
    elements.moveButtons.forEach(btn => {
      // If disabling, always disable
      if (!enabled) {
        btn.disabled = true;
        return;
      }
      
      // If enabling, check if damage values are meaningful (not default 10)
      const damageLoaded = elements.damagePills.some(pill => {
        const text = pill.textContent || '';
        return text.includes('DMG') && !text.includes('DMG 10');
      });
      
      btn.disabled = !damageLoaded;
    });
  }

  function setGameSectionsActive(active) {
  // Always hide game over section when switching states
  hideGameOverLeaderboard();
    
    if (active) {
      // Game is active - show game sections, hide empty state
      if (elements.emptyState) elements.emptyState.hidden = true;
      if (elements.scoreboard) elements.scoreboard.hidden = false;
      if (elements.battleArena) elements.battleArena.hidden = false;
      if (elements.moveControls) elements.moveControls.hidden = false;
      if (elements.historyCard) elements.historyCard.hidden = false;
      requestAnimationFrame(updateTriangleGeometry);
    } else {
      // No game - show empty state, hide game sections
      if (elements.emptyState) elements.emptyState.hidden = false;
      if (elements.scoreboard) elements.scoreboard.hidden = true;
      if (elements.battleArena) elements.battleArena.hidden = true;
      if (elements.moveControls) elements.moveControls.hidden = true;
      if (elements.historyCard) elements.historyCard.hidden = true;
    }
  }

  function hideGameOverLeaderboard() {
    if (elements.gameOverPanel) {
      elements.gameOverPanel.hidden = true;
    }
    if (elements.gameOverSummary) {
      elements.gameOverSummary.textContent = '';
    }
  }

  function renderGameOverLeaderboard() {
    if (!elements.gameOverList) return;
    const players = state.leaderboardPlayers || [];
    const opponent = state.leaderboardOpponent
      ? state.botPolicies.find(p => p.id === state.leaderboardOpponent)
      : null;
  const difficultyLabel = 'Win streaks (standard mode)';
  const difficultyKey = 'standard';
    const opponentLabel = opponent
      ? `${emojiMap[opponent.id] || '🤖'} ${opponent.label}`
      : 'All opponents';

    elements.gameOverList.innerHTML = '';

    if (!players.length) {
      const li = document.createElement('li');
      li.className = 'meta';
      li.textContent = opponent
  ? `No win streaks yet against ${opponent.label} in ${difficultyKey} mode. Play some games to start building streaks!`
  : `No win streak data yet for ${difficultyKey} mode. Play some games to build a streak!`;
      elements.gameOverList.appendChild(li);
    } else {
      players.slice(0, 10).forEach((player, idx) => {
        const li = document.createElement('li');
        li.className = 'leaderboard-item';
        const streak = player.streak || player.win_streak || 0;
        const lastDate = player.last_win_date ? new Date(player.last_win_date) : null;
        const meta = lastDate ? lastDate.toLocaleString() : '';
        li.innerHTML = `<span class="rank">${idx + 1}</span><div><strong>${player.username || player.name || 'Unknown'}</strong><div class="meta">${meta}</div></div><span>${streak}</span>`;
        elements.gameOverList.appendChild(li);
      });
    }

    if (elements.gameOverMeta) {
      elements.gameOverMeta.textContent = `${difficultyLabel} · ${opponentLabel}`;
    }
  }

  function showGameOverLeaderboard(summary) {
    if (!elements.gameOverPanel) return;
    renderGameOverLeaderboard();

    const playerWon = summary?.winner === 'user';
    const isTie = summary?.winner === 'tie';
    const botLabel = formatBotLabel(state.policy, 'Bot');
    const playerScore = Number(summary?.user_score ?? state.gameStats?.user_score ?? 0).toFixed(1);
    const botScore = Number(summary?.bot_score ?? state.gameStats?.bot_score ?? 0).toFixed(1);

    let message;
    if (isTie) {
      message = `It's a tie! Final score ${playerScore} – ${botScore}.`;
    } else if (playerWon) {
      message = `You win ${playerScore} – ${botScore}.`;
    } else {
      message = `${botLabel} wins ${botScore} – ${playerScore}.`;
    }

    if (elements.gameOverSummary) {
      elements.gameOverSummary.textContent = message;
    }

    elements.gameOverPanel.hidden = false;
  }

  function hpFromScore(score) {
    const val = 100 - 10 * Number(score || 0);
    return Math.max(0, Math.min(100, Math.round(val)));
  }

  function damageFromDelta(delta) {
    if (!delta) return 0;
    return Math.max(0, Math.round(Number(delta) * 10));
  }

  function setHpFill(fillEl, hp) {
    if (!fillEl) return;
    const clamped = Math.max(0, Math.min(100, Number(hp) || 0));
    fillEl.style.width = `${clamped}%`;
    fillEl.style.transform = '';
    fillEl.classList.remove('danger', 'warn');
    if (clamped <= 25) fillEl.classList.add('danger');
    else if (clamped <= 55) fillEl.classList.add('warn');
  }

  function updateHpCards(userScore, botScore, deltas = {}) {
    const youHp = hpFromScore(botScore);
    const botHp = hpFromScore(userScore);

    if (elements.playerHpLabel) elements.playerHpLabel.textContent = `${youHp} HP`;
    if (elements.botHpLabel) elements.botHpLabel.textContent = `${botHp} HP`;

    state.gameStats = {
      user_score: Number(userScore || 0),
      bot_score: Number(botScore || 0)
    };

    setHpFill(elements.playerHpFill, youHp);
    setHpFill(elements.botHpFill, botHp);

    // BUG FIX: Correct delta assignment - user_delta is USER's points, bot_delta is BOT's points
    // When displaying damage TO player, use user_delta (player lost these points)
    // When displaying damage TO bot, use bot_delta (bot lost these points)
    const playerDamageTaken = damageFromDelta(deltas.bot_delta);   // Damage TO player (bot scored)
    const botDamageTaken = damageFromDelta(deltas.user_delta);     // Damage TO bot (player scored)
    const outcome = (deltas?.result || '').toLowerCase();
    const isTieRound = outcome === 'tie' || outcome === 'draw';

    if (elements.playerDamage) {
      if (playerDamageTaken > 0) {
        elements.playerDamage.textContent = `−${playerDamageTaken}`;
        elements.playerDamage.hidden = false;
        elements.playerDamage.style.textDecoration = isTieRound ? 'line-through' : 'none';
      } else {
        elements.playerDamage.hidden = true;
        elements.playerDamage.style.textDecoration = 'none';
      }
    }

    if (elements.botDamage) {
      if (botDamageTaken > 0) {
        elements.botDamage.textContent = `−${botDamageTaken}`;
        elements.botDamage.hidden = false;
        elements.botDamage.style.textDecoration = isTieRound ? 'line-through' : 'none';
      } else {
        elements.botDamage.hidden = true;
        elements.botDamage.style.textDecoration = 'none';
      }
    }
  }

  function renderApiLog() {
    const latest = state.apiLog.slice(-12).reverse();
    [elements.apiDebugLog, elements.analysisApiLog].forEach(container => {
      if (!container) return;
      container.innerHTML = '';
      if (!latest.length) {
        const span = document.createElement('span');
        span.className = 'meta';
        span.textContent = 'No API calls logged yet.';
        container.appendChild(span);
        return;
      }
      latest.forEach(log => {
        const entry = document.createElement('div');
        entry.className = `entry ${log.ok ? 'ok' : 'fail'}`.trim();
        const ts = log.ts instanceof Date ? log.ts : new Date(log.ts);
        const params = log.params ? `?${new URLSearchParams(log.params).toString()}` : '';
        const path = (() => {
          try {
            return new URL(log.url, window.location.origin).pathname;
          } catch (_) {
            return log.url;
          }
        })();

        let debugInfo = '';
        if (state.showDebug && log.debug) {
          debugInfo = `
            <details class="api-debug-details">
              <summary>🔍 Debug Details</summary>
              <div class="debug-content">
                <div><strong>Probabilities:</strong> ${JSON.stringify(log.debug.probabilities, null, 2)}</div>
                ${log.debug.base_probabilities ? `<div><strong>Base Probabilities:</strong> ${JSON.stringify(log.debug.base_probabilities, null, 2)}</div>` : ''}
                <div><strong>Expected Values:</strong> ${JSON.stringify(log.debug.move_values, null, 2)}</div>
                <div><strong>Model:</strong> ${log.debug.model_type}</div>
                <div><strong>Source:</strong> ${log.debug.probability_source}</div>
                <div><strong>Selected:</strong> ${log.debug.selected_move}</div>
              </div>
            </details>
          `;
        }

        let requestInfo = '';
        if (state.showDebug && log.requestBody) {
          requestInfo = `<div class="request-body"><strong>Request:</strong> ${JSON.stringify(log.requestBody)}</div>`;
        }

        entry.innerHTML = `
          <strong>${ts.toLocaleTimeString()}</strong>
          <span>${log.method || 'GET'} · ${log.status} · ${log.elapsed}ms</span>
          <span>${path}${params}</span>
          ${log.error ? `<span class="meta">${log.error}</span>` : ''}
          ${requestInfo}
          ${debugInfo}
        `;
        container.appendChild(entry);
      });
    });
  }

  function renderSessionStats() {
    const targets = [elements.sessionStatsSummary, elements.analysisSessionStats];
    targets.forEach(container => {
      if (!container) return;
      container.innerHTML = '';
    });

    if (!state.sessionStats) {
      targets.forEach(container => {
        if (!container) return;
        const msg = document.createElement('p');
        msg.className = 'meta';
        msg.textContent = 'No session stats yet.';
        container.appendChild(msg);
      });
      return;
    }

    const { games = 0, rounds = 0, wins = 0, losses = 0, draws = 0 } = state.sessionStats;
    const winPct = rounds > 0 ? ((wins / rounds) * 100).toFixed(1) : '—';
    const summary = [
      { label: 'Games', value: games },
      { label: 'Rounds', value: rounds },
      { label: 'Wins', value: wins },
      { label: 'Losses', value: losses },
      { label: 'Draws', value: draws },
      { label: 'Win %', value: winPct }
    ];

    targets.forEach(container => {
      if (!container) return;
      summary.forEach(item => {
        const row = document.createElement('span');
        row.innerHTML = `<strong>${item.label}</strong><span>${item.value}</span>`;
        container.appendChild(row);
      });
    });
  }

  function updateAnalysisPanels() {
    if (!elements.analysisGrid) {
      renderApiLog();
      return;
    }

    const shouldShow = state.showDebug && (state.gameId || state.sessionId);
    elements.analysisGrid.hidden = !shouldShow;

    if (!shouldShow) {
      renderApiLog();
      return;
    }

    elements.analysisTarget.textContent = state.targetScore != null ? `${state.targetScore} pts` : '—';
    elements.analysisPolicy.textContent = formatBotLabel(state.policy, 'Bot');
    elements.analysisSession.textContent = state.sessionId ? `${state.sessionId.slice(0, 8)}…` : '—';
    elements.analysisGameId.textContent = state.gameId ? `${state.gameId.slice(0, 8)}…` : '—';

    const roundNo = state.roundInfo?.round_no;
    if (!roundNo) {
      elements.analysisPhase.textContent = 'Awaiting round data';
    } else if (roundNo <= 3) {
      elements.analysisPhase.textContent = `Round ${roundNo}`;
    } else {
      elements.analysisPhase.textContent = `ML prediction · Round ${roundNo}`;
    }

    elements.analysisBotProbs.innerHTML = '';
    const probs = state.botProbabilities?.probabilities;
    const canShowProbabilities = !!probs;
    if (canShowProbabilities) {
      ['rock', 'paper', 'scissors'].forEach(move => {
        const card = document.createElement('div');
        card.className = 'prob-card';
        card.innerHTML = `<span>${move.toUpperCase()}</span>${(probs[move] * 100).toFixed(1)}%`;
        elements.analysisBotProbs.appendChild(card);
      });
    } else {
      const span = document.createElement('span');
      span.className = 'meta';
      span.textContent = 'Probabilities unavailable';
      elements.analysisBotProbs.appendChild(span);
    }

    elements.analysisExpected.innerHTML = '';
    const roundPoints = state.roundInfo?.round_points;
  if (canShowProbabilities && roundPoints && state.gameStats) {
      const values = {
        rock: Number(roundPoints.rock ?? 1),
        paper: Number(roundPoints.paper ?? 1),
        scissors: Number(roundPoints.scissors ?? 1)
      };
      const moveInfo = {
        rock: { beats: 'scissors', loses: 'paper', icon: '🪨' },
        paper: { beats: 'rock', loses: 'scissors', icon: '📄' },
        scissors: { beats: 'paper', loses: 'rock', icon: '✂️' }
      };
      const botScore = Number(state.gameStats.bot_score || 0);
      const userScore = Number(state.gameStats.user_score || 0);
      const results = Object.keys(moveInfo).map(move => {
        const { beats, loses, icon } = moveInfo[move];
        const pBeat = Number(probs[beats] || 0);
        const pLose = Number(probs[loses] || 0);
        const pTie = Number(probs[move] || 0);
        const dmgForBot = values[move] || 1;
        const dmgForUser = values[loses] || 1;
        const winBonus = (botScore + dmgForBot) >= 10 ? 10 : 0;
        const losePenalty = (userScore + dmgForUser) >= 10 ? 10 : 0;
        const tieBonus = 0.5;
        const expected = pBeat * (dmgForBot + winBonus) - pLose * (dmgForUser + losePenalty) + pTie * tieBonus;
        return { move, icon, expected };
      });
      const best = results.reduce((acc, cur) => cur.expected > acc.expected ? cur : acc, results[0]);
      results.forEach(item => {
        const li = document.createElement('li');
        if (best && item.move === best.move) li.classList.add('best');
        li.innerHTML = `<span>${item.icon} ${item.move.charAt(0).toUpperCase() + item.move.slice(1)}</span><span>${item.expected.toFixed(2)}</span>`;
        elements.analysisExpected.appendChild(li);
      });
    } else {
      const li = document.createElement('li');
      li.className = 'meta';
      li.textContent = 'Expected values unavailable';
      elements.analysisExpected.appendChild(li);
    }

    if (state.botPredictionMeta) {
      if (state.botPredictionMeta.error) {
        elements.analysisPredicted.textContent = `⚠️ ${state.botPredictionMeta.error}`;
      } else {
        const move = state.botPredictionMeta.predicted_move ? state.botPredictionMeta.predicted_move.toUpperCase() : '?';
        const model = state.botPredictionMeta.model_type || 'unknown';
        const round = state.botPredictionMeta.round_number || '?';
        elements.analysisPredicted.textContent = `Predicted ${move} · ${model} · Round ${round}`;
      }
    } else {
      elements.analysisPredicted.textContent = 'Probabilities unavailable';
    }

    elements.analysisLastJson.textContent = state.lastRound ? JSON.stringify(state.lastRound, null, 2) : 'Awaiting data…';

    renderApiLog();
  }

  async function refreshSessionStats() {
    if (!state.sessionId) {
      state.sessionStats = null;
      renderSessionStats();
      return;
    }
    try {
      const data = await fetchJson('/stats', { session_id: state.sessionId });
      state.sessionStats = data;
    } catch (err) {
      debugLog(`session stats error: ${err.message}`);
    }
    renderSessionStats();
    updateAnalysisPanels();
  }

  function faceForResult(result, side) {
    if (!result) return side === 'player' ? '🙂' : '🤖';
    if (result === 'draw' || result === 'tie') return '😐';
    if (result === 'win') return side === 'player' ? '😄' : '😖';
    return side === 'player' ? '😖' : '😄';
  }

  function formatMove(move) {
    const key = (move || '').toLowerCase();
    if (!moveEmoji[key]) return '—';
    const label = key.charAt(0).toUpperCase() + key.slice(1);
    return `${moveEmoji[key]} ${label}`;
  }

  function renderBattle(last) {
    if (!elements.playerFace || !elements.botFace) return;

    if (!last) {
      if (state.playerName && elements.playerNameLabel) {
        elements.playerNameLabel.textContent = state.playerName;
      }
      elements.playerFace.textContent = '🙂';
      elements.botFace.textContent = emojiMap[state.policy] || '🤖';
      if (elements.botName) elements.botName.textContent = botNames[state.policy] || 'Bot';
      if (elements.playerMove) elements.playerMove.textContent = '—';
      if (elements.botMove) elements.botMove.textContent = '—';
      if (elements.playerDmgChip) elements.playerDmgChip.textContent = '0 DMG';
      if (elements.botDmgChip) elements.botDmgChip.textContent = '0 DMG';
      if (elements.roundSummary) elements.roundSummary.textContent = 'Choose your move to battle!';
      if (elements.botNameLabel) elements.botNameLabel.textContent = formatBotLabel(state.policy, 'Bot');
      return;
    }

  const outcome = (last.result || '').toLowerCase();
  const isTieRound = outcome === 'tie' || outcome === 'draw';
    elements.playerFace.textContent = faceForResult(outcome, 'player');
    elements.botFace.textContent = emojiMap[state.policy] || '🤖';
    if (elements.botName) elements.botName.textContent = botNames[state.policy] || 'Bot';
    if (elements.playerMove) elements.playerMove.textContent = formatMove(last.user_move);
    if (elements.botMove) elements.botMove.textContent = formatMove(last.bot_move);
    if (elements.botNameLabel) elements.botNameLabel.textContent = formatBotLabel(state.policy, 'Bot');
    if (state.playerName && elements.playerNameLabel) {
      elements.playerNameLabel.textContent = state.playerName;
    }

    const roundPts = last.round_points || {};
    const moves = ['rock', 'paper', 'scissors'];
    const damages = moves.map(m => damageFromDelta(roundPts[m] || 0.0));
    const palette = damagePalette(damages);

    const playerBaseDamage = damageFromDelta(roundPts[last.user_move] ?? 0.0);
    const botBaseDamage = damageFromDelta(roundPts[last.bot_move] ?? 0.0);
    const playerInflicted = damageFromDelta(last.user_delta || 0);
    const botInflicted = damageFromDelta(last.bot_delta || 0);
    const playerLosing = outcome === 'lose' || outcome === 'loss';
    const botLosing = outcome === 'win';

    updateDmgChip(
      elements.playerDmgChip,
      playerInflicted,
      playerBaseDamage,
      palette.get(playerBaseDamage),
      playerLosing ? 'losing' : 'win',
      isTieRound
    );
    updateDmgChip(
      elements.botDmgChip,
      botInflicted,
      botBaseDamage,
      palette.get(botBaseDamage),
      botLosing ? 'losing' : 'win',
      isTieRound
    );

    const roundText = buildRoundSummary(last);
    if (elements.roundSummary) elements.roundSummary.innerHTML = roundText.summary;
    if (elements.lastRoundDetail) elements.lastRoundDetail.innerHTML = roundText.detail;
  }

  function updateDmgChip(el, inflictedDamage, baseDamage, colors, losingState, isTieRound = false) {
    if (!el) return;
    const [bg, text] = colors || ['rgba(15,23,42,0.6)', 'var(--text)'];

    const displayValue = `${baseDamage} DMG`;
    el.textContent = displayValue;

  const shouldStrike = isTieRound || inflictedDamage <= 0;
  el.style.textDecoration = shouldStrike ? 'line-through' : 'none';

    el.style.background = bg;
    el.style.color = text;
    el.classList.toggle('losing', losingState === 'losing');
    el.classList.toggle('emph', Number(baseDamage) >= 14);
  }

  function buildRoundSummary(last) {
    const outcome = (last.result || '').toLowerCase();
    const userDmg = damageFromDelta(last.user_delta || 0);
    const botDmg = damageFromDelta(last.bot_delta || 0);
    const userHp = hpFromScore(last.bot_score || 0);
    const botHp = hpFromScore(last.user_score || 0);

    const botLabel = currentBotDisplayName();
    let summary = `You played <strong>${formatMove(last.user_move)}</strong> while ${botLabel} picked <strong>${formatMove(last.bot_move)}</strong>.`;
    if (outcome === 'win') summary += ' ✅ You won the round.';
    else if (outcome === 'lose' || outcome === 'loss') summary += ` ❌ ${botLabel} won the round.`;
    else summary += ` 🤝 It's a tie with ${botLabel}.`;
    // summary += ` Damage dealt — You: ${userDmg} DMG · Bot: ${botDmg} DMG.`;

    const detail = `Result: <strong>${outcome.toUpperCase() || 'PENDING'}</strong> · HP You ${userHp} · ${botLabel} ${botHp}`;

    return { summary, detail };
  }

  function addHistory(entry) {
    if (!elements.historyList) return;
    const li = document.createElement('li');
    const userDmg = damageFromDelta(entry.user_delta || 0);
    const botDmg = damageFromDelta(entry.bot_delta || 0);
    const userHp = hpFromScore(entry.bot_score || 0);
    const botHp = hpFromScore(entry.user_score || 0);
    const botLabel = currentBotDisplayName();
    li.innerHTML = `
      <strong>${(entry.result || '').toUpperCase()}</strong> · ${formatMove(entry.user_move)} vs ${formatMove(entry.bot_move)}<br />
      <span class="meta">DMG you ${userDmg} · ${botLabel} ${botDmg} · HP you ${userHp} / ${botLabel} ${botHp}</span>
    `;
    elements.historyList.prepend(li);
    const maxRows = 12;
    while (elements.historyList.children.length > maxRows) {
      elements.historyList.removeChild(elements.historyList.lastChild);
    }
    if (elements.recentHistoryMeta) {
      elements.recentHistoryMeta.textContent = `Updated ${new Date().toLocaleTimeString()}`;
    }
  }

  function clearHistory() {
    if (!elements.historyList) return;
    elements.historyList.innerHTML = '';
    if (elements.recentHistoryMeta) elements.recentHistoryMeta.textContent = '';
  }

  function damagePalette(damages) {
    const unique = Array.from(new Set(damages)).sort((a, b) => a - b);
    const map = new Map();

    const colors = {
      low: ['rgba(15,23,42,0.6)', 'var(--text)'],
      mid: ['#fbbf24', '#141414'],
      high: ['#10b981', '#0f172a']
    };

    damages.forEach(dmg => {
      let bucket = colors.low;
      if (unique.length === 1) bucket = colors.low;
      else if (unique.length === 2) bucket = dmg === unique[1] ? colors.high : colors.low;
      else if (unique.length >= 3) {
        if (dmg === unique[0]) bucket = colors.low;
        else if (dmg === unique[1]) bucket = colors.mid;
        else bucket = colors.high;
      }
      map.set(dmg, bucket);
    });

    return map;
  }

  function applyDamagePills(roundPoints = {}) {
    const moves = ['rock', 'paper', 'scissors'];
    const damages = moves.map(m => damageFromDelta(Number(roundPoints[m] || 1.0)));
    const palette = damagePalette(damages);

    elements.damagePills.forEach(pill => {
      const move = pill.dataset.dmg;
      const dmg = damageFromDelta(Number(roundPoints[move] || 1.0));
      const colors = palette.get(dmg) || ['rgba(15,23,42,0.6)', 'var(--text)'];
      pill.textContent = `DMG ${dmg}`;
      pill.style.background = colors[0];
      pill.style.color = colors[1];
      pill.style.borderColor = colors[0] === 'rgba(15,23,42,0.6)' ? 'rgba(148,163,184,0.2)' : colors[0];
    });

    // Enable move buttons now that damage values are loaded
    if (state.gameId && !state.gameFinishedNotified) {
      toggleMoveButtons(true);
    }

    requestAnimationFrame(updateTriangleGeometry);
  }

  function updateTriangleGeometry() {
    const container = elements.moveTriangle;
    const svg = elements.triangleSvg;
    if (!container || !svg) return;

    const nodes = {
      rock: container.querySelector('[data-node="rock"]'),
      paper: container.querySelector('[data-node="paper"]'),
      scissors: container.querySelector('[data-node="scissors"]')
    };

    const rect = container.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const points = {};
    Object.entries(nodes).forEach(([key, node]) => {
      if (!node) return;
      const target = node.querySelector('.move-btn') || node;
      const btnRect = target.getBoundingClientRect();
      points[key] = {
        x: btnRect.left + btnRect.width / 2 - rect.left,
        y: btnRect.top + btnRect.height / 2 - rect.top
      };
    });

    svg.setAttribute('viewBox', `0 0 ${Math.round(rect.width)} ${Math.round(rect.height)}`);

    const lines = {
      rockScissors: svg.querySelector('[data-line="rock-scissors"]'),
      scissorsPaper: svg.querySelector('[data-line="scissors-paper"]'),
      paperRock: svg.querySelector('[data-line="paper-rock"]')
    };

    const labels = {
      rockScissors: svg.querySelector('[data-label="rock-scissors"]'),
      scissorsPaper: svg.querySelector('[data-label="scissors-paper"]'),
      paperRock: svg.querySelector('[data-label="paper-rock"]')
    };

    const arrows = {
      rockScissors: svg.querySelector('[data-arrow="rock-scissors"]'),
      scissorsPaper: svg.querySelector('[data-arrow="scissors-paper"]'),
      paperRock: svg.querySelector('[data-arrow="paper-rock"]')
    };

    const setLine = (line, from, to) => {
      if (!line || !points[from] || !points[to]) return;
      line.setAttribute('x1', points[from].x);
      line.setAttribute('y1', points[from].y);
      line.setAttribute('x2', points[to].x);
      line.setAttribute('y2', points[to].y);
    };

    const setLabel = (label, from, to, offsetY = -10) => {
      if (!label || !points[from] || !points[to]) return;
      const midX = (points[from].x + points[to].x) / 2;
      const midY = (points[from].y + points[to].y) / 2 + offsetY;
      label.setAttribute('x', midX);
      label.setAttribute('y', midY);
      label.setAttribute('text-anchor', 'middle');
    };

    setLine(lines.rockScissors, 'rock', 'scissors');
    setLine(lines.scissorsPaper, 'scissors', 'paper');
    setLine(lines.paperRock, 'paper', 'rock');

    setLabel(labels.rockScissors, 'rock', 'scissors');
    setLabel(labels.scissorsPaper, 'scissors', 'paper', 14);
    setLabel(labels.paperRock, 'paper', 'rock');

    const setArrow = (arrow, from, to) => {
      if (!arrow || !points[from] || !points[to]) return;
      const midX = (points[from].x + points[to].x) / 2;
      const midY = (points[from].y + points[to].y) / 2;
      const dx = points[to].x - points[from].x;
      const dy = points[to].y - points[from].y;
      const angle = Math.atan2(dy, dx) * (180 / Math.PI);
      arrow.setAttribute('transform', `translate(${midX}, ${midY}) rotate(${angle})`);
    };

    setArrow(arrows.rockScissors, 'rock', 'scissors');
    setArrow(arrows.scissorsPaper, 'scissors', 'paper');
    setArrow(arrows.paperRock, 'paper', 'rock');
  }

  function setActivePolicy(policyId, quiet = false) {
    if (!policyId) return;
    state.policy = policyId;
    if (elements.policyButtons) {
      elements.policyButtons.querySelectorAll('.policy-button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.policy === policyId);
      });
    }
    const policy = state.botPolicies.find(p => p.id === policyId);
    if (elements.botNameLabel) elements.botNameLabel.textContent = formatBotLabel(policyId, 'Bot');
    if (!quiet && policy) {
      setStatus(`Selected ${policy.label}`);
    }
    updateAnalysisPanels();
  }

  function updatePolicyButtons(list) {
    if (!elements.policyButtons) return;
    elements.policyButtons.innerHTML = '';
    state.botPolicies = list;
    if (!state.policy && list[0]) {
      state.policy = list[0].id;
    }

    state.policyStatRefs = new Map();

    list.forEach(item => {
      const card = document.createElement('div');
      card.className = 'policy-card';

      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'policy-button';
      btn.dataset.policy = item.id;
      btn.innerHTML = `<span>${emojiMap[item.id] || '🤖'} ${item.label}</span>`;
      btn.addEventListener('click', () => setActivePolicy(item.id));

      const stats = document.createElement('div');
      stats.className = 'policy-stats';
      stats.dataset.policyStats = item.id;
      stats.innerHTML = '<span>Win Rate vs Human Players: —</span><span>—</span>';

      card.appendChild(btn);
      card.appendChild(stats);
      elements.policyButtons.appendChild(card);
      state.policyStatRefs.set(item.id, stats);
    });

    if (!state.leaderboardOpponent && list.length > 0) {
      // Default to the first available policy (brian, forrest, or logan)
      state.leaderboardOpponent = list[0].id;
    }
    renderLeaderboardControls();

    setActivePolicy(state.policy, true);
  }

  function renderLeaderboardControls() {
    const containers = [elements.leaderboardControls, elements.gameOverControls].filter(Boolean);
    if (!containers.length) return;

    containers.forEach(container => {
      container.innerHTML = '';

      state.botPolicies.forEach(pol => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = state.leaderboardOpponent === pol.id ? 'active' : '';
        btn.textContent = `${emojiMap[pol.id] || '🤖'} ${pol.label}`;
        btn.addEventListener('click', () => {
          state.leaderboardOpponent = pol.id;
          renderLeaderboardControls();
          refreshLeaderboard();
        });
        container.appendChild(btn);
      });
    });
  }

  function debugLog(message) {
    if (!state.showDebug || !elements.debugPanel) return;
    const timestamp = new Date().toISOString();
    const entry = document.createElement('div');
    entry.textContent = `[${timestamp}] ${message}`;
    elements.debugPanel.appendChild(entry);
    if (elements.debugPanel.childElementCount > 80) {
      elements.debugPanel.removeChild(elements.debugPanel.firstChild);
    }
  }

  function updateDebugPredictionInfo(debug, fullResponse) {
    if (!state.showDebug) return;
    
    // Log to console for easy inspection
    console.log('🎲 Prediction Debug Info:', debug);
    console.log('📊 Full Response:', fullResponse);
    
    // Update analysis panels if they exist
    if (elements.analysisBotProbs) {
      const probsText = debug.probabilities ? 
        Object.entries(debug.probabilities)
          .map(([move, prob]) => `${move}: ${(prob * 100).toFixed(1)}%`)
          .join(', ') :
        'N/A';
      elements.analysisBotProbs.textContent = probsText;
    }
    
    if (elements.analysisExpected) {
      const evText = debug.move_values ?
        Object.entries(debug.move_values)
          .map(([move, ev]) => `${move}: ${ev.toFixed(3)}`)
          .join(', ') :
        'N/A';
      elements.analysisExpected.textContent = evText;
    }
    
    if (elements.analysisPredicted) {
      elements.analysisPredicted.textContent = debug.selected_move || 'N/A';
    }
    
    if (elements.analysisLastJson) {
      elements.analysisLastJson.textContent = JSON.stringify(debug, null, 2);
    }
    
    // Show floating debug panel
    showFloatingDebugPanel(debug, fullResponse);
    
    // Log detailed debug info
    debugLog(`🎯 Model: ${debug.model_type} | Source: ${debug.probability_source}`);
    debugLog(`📊 Probabilities: ${JSON.stringify(debug.probabilities)}`);
    if (debug.base_probabilities && JSON.stringify(debug.base_probabilities) !== JSON.stringify(debug.probabilities)) {
      debugLog(`📊 Base Probabilities (before easy-mode): ${JSON.stringify(debug.base_probabilities)}`);
    }
    debugLog(`💰 Expected Values: ${JSON.stringify(debug.move_values)}`);
    debugLog(`🤖 Selected Move: ${debug.selected_move}`);
  }
  
  function showFloatingDebugPanel(debug, fullResponse) {
    // Create or get floating debug panel
    let panel = document.getElementById('floating-debug-panel');
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'floating-debug-panel';
      panel.className = 'floating-debug-panel';
      panel.innerHTML = `
        <div class="floating-debug-header">
          <strong>🔍 Latest Prediction</strong>
          <button class="close-btn" onclick="document.getElementById('floating-debug-panel').style.display='none'">✕</button>
        </div>
        <div class="floating-debug-content"></div>
      `;
      document.body.appendChild(panel);
    }
    
    const content = panel.querySelector('.floating-debug-content');
    if (!content) return;
    
    // Get model info from response
    const modelInfo = fullResponse.model_info || {};
    const modelAlias = modelInfo.model_alias || 'unknown';
    const modelVersion = modelInfo.model_version || '?';
    const modelRunId = modelInfo.model_run_id || null;
    const modelType = debug.model_type || 'unknown';
    const probSource = debug.probability_source || 'unknown';
    
    // Build version display string
    let versionStr = `v${modelVersion}`;
    if (modelRunId) {
      versionStr += ` (${modelRunId})`;
    }
    
    content.innerHTML = `
      <div class="debug-section">
        <div class="debug-label">Round</div>
        <div class="debug-value">${fullResponse.round || '?'}</div>
      </div>
      <div class="debug-section">
        <div class="debug-label">Model</div>
        <div class="debug-value">
          ${modelType}<br>
          <small>Alias: <strong>${modelAlias}</strong> | Version: <strong>${versionStr}</strong></small><br>
          <small>Source: ${probSource}</small>
        </div>
      </div>
      <div class="debug-section">
        <div class="debug-label">Probabilities</div>
        <div class="debug-grid">
          ${Object.entries(debug.probabilities || {}).map(([move, prob]) => `
            <div class="prob-item">
              <span>${move}</span>
              <strong>${(prob * 100).toFixed(1)}%</strong>
            </div>
          `).join('')}
        </div>
      </div>
      ${debug.base_probabilities && JSON.stringify(debug.base_probabilities) !== JSON.stringify(debug.probabilities) ? `
        <div class="debug-section">
          <div class="debug-label">Base Probabilities (before difficulty adjustment)</div>
          <div class="debug-grid">
            ${Object.entries(debug.base_probabilities).map(([move, prob]) => `
              <div class="prob-item">
                <span>${move}</span>
                <strong>${(prob * 100).toFixed(1)}%</strong>
              </div>
            `).join('')}
          </div>
        </div>
      ` : ''}
      <div class="debug-section">
        <div class="debug-label">Expected Values</div>
        <div class="debug-grid">
          ${Object.entries(debug.move_values || {}).map(([move, ev]) => `
            <div class="ev-item ${move === debug.selected_move ? 'selected' : ''}">
              <span>${move}</span>
              <strong>${ev.toFixed(3)}</strong>
            </div>
          `).join('')}
        </div>
      </div>
      <div class="debug-section">
        <div class="debug-label">Selected Move</div>
        <div class="debug-value highlight">${debug.selected_move}</div>
      </div>
    `;
    
    panel.style.display = 'block';
  }

  async function fetchPolicies() {
    try {
      const data = await fetchJson('/policies');
      const list = (data || []).map(item => typeof item === 'string' ? { id: item, label: item.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) } : {
        id: item.id || item.policy || item.name,
        label: item.label || (item.id || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
      }).filter(Boolean);

      updatePolicyButtons(list.length ? list : [
        { id: 'brian', label: 'Brian (Neural Network)' },
        { id: 'forrest', label: 'Forrest (XGBoost)' },
        { id: 'logan', label: 'Logan (Logistic Regression)' }
      ]);
    } catch (err) {
      setStatus(`Failed to load policies: ${err.message}`, 'fail');
      updatePolicyButtons([
        { id: 'brian', label: 'Brian (Neural Network)' },
        { id: 'forrest', label: 'Forrest (XGBoost)' },
        { id: 'logan', label: 'Logan (Logistic Regression)' }
      ]);
    }
  }

  async function refreshBotPerformance() {
    if (!state.botPolicies.length) return;
    const requestId = ++state.botRatesRefreshToken;
    try {
      const response = await fetchJson('/bot_win_rates', {
        easy_mode: false,
        human_only: true
      });

      if (requestId !== state.botRatesRefreshToken) {
        return;
      }

      const rates = response?.bot_rates || {};
      const serverResetIso = response?.since || state.metricsResetIso;
      const cutoff = parseIsoTimestamp(serverResetIso);
      if (serverResetIso && serverResetIso !== state.metricsResetIso) {
        state.metricsResetIso = serverResetIso;
        state.metricsResetDate = cutoff;
      }

      const warnings = [];
      if (!serverResetIso) {
        warnings.push('Metrics reset timestamp unavailable; using full history for win rates.');
      } else if (!cutoff) {
        warnings.push(`Metrics reset timestamp "${serverResetIso}" could not be parsed; win rates may be stale.`);
      }

      updateDataHealthBanner(warnings);
      state.latestBotRates = rates;

      state.botPolicies.forEach(pol => {
        const statsEl = state.policyStatRefs?.get(pol.id);
        if (!statsEl) return;

        const metrics = rates[pol.id] || {};
        const wins = Number(metrics?.bot_wins || 0);
        const total = Number(metrics?.human_games || 0);
        const winRate = typeof metrics?.bot_win_rate === 'number' ? metrics.bot_win_rate : null;
        const rateText = winRate === null ? '—' : `${winRate.toFixed(1)}%`;

        statsEl.innerHTML = `<span>Win Rate vs Human Players: ${rateText}</span><span>${wins}/${total}</span>`;
        statsEl.classList.toggle('warn', total === 0);
        statsEl.classList.remove('info');
      });
    } catch (err) {
      if (requestId !== state.botRatesRefreshToken) {
        return;
      }
      updateDataHealthBanner([`Failed to refresh /bot_win_rates: ${err.message}`]);
      state.policyStatRefs.forEach(statsEl => {
        statsEl.innerHTML = '<span class="meta">Win Rate vs Human Players: —</span><span class="meta">—</span>';
        statsEl.classList.add('warn');
        statsEl.classList.remove('info');
      });
    }
  }

  function scheduleBotPerformancePolling() {
    if (!window || typeof window.setInterval !== 'function') {
      return;
    }
    if (scheduleBotPerformancePolling.timerId) {
      clearInterval(scheduleBotPerformancePolling.timerId);
    }
    scheduleBotPerformancePolling.timerId = window.setInterval(() => {
      refreshBotPerformance().catch(err => debugLog(`bot_win_rates poll error: ${err.message}`));
    }, BOT_PERFORMANCE_REFRESH_MS);
  }

  async function refreshLeaderboard() {
    // Set a default opponent if none is selected
    if (state.leaderboardOpponent === undefined && state.botPolicies.length > 0) {
      state.leaderboardOpponent = state.botPolicies[0].id; // Default to first policy
      renderLeaderboardControls();
    }
    
    if (!state.leaderboardOpponent) {
      console.warn('No leaderboard opponent selected and no bot policies available');
      return;
    }
    try {
      const params = {
        opponent: state.leaderboardOpponent,
        limit: 10
      };

      const data = await fetchJson('/win_streaks', params);
      
      console.log('Leaderboard data received:', data);
      const players = data?.players || [];
      renderLeaderboard(players);
    } catch (err) {
      console.error('Leaderboard fetch error:', err);
      if (elements.leaderboardList) elements.leaderboardList.innerHTML = '<li class="meta">Leaderboard unavailable</li>';
      if (elements.leaderboardMeta) elements.leaderboardMeta.textContent = '';
    }
  }

  function renderLeaderboard(players) {
    const list = Array.isArray(players) ? players : [];
    const filtered = list.filter(player => {
      const candidate = (player?.username || player?.name || '').trim();
      if (!candidate) {
        return true;
      }
      return !shouldExcludeLeaderboardPlayer(candidate);
    });
    state.leaderboardPlayers = filtered.slice();

    const opponent = state.leaderboardOpponent 
      ? state.botPolicies.find(p => p.id === state.leaderboardOpponent)
      : null;
  const difficultyLabel = 'Win streaks (standard mode)';
  const difficultyKey = 'standard';
    const opponentLabel = opponent 
      ? `${emojiMap[opponent.id] || '🤖'} ${opponent.label}`
      : 'All opponents';

    if (elements.leaderboardMeta) {
      elements.leaderboardMeta.textContent = `${difficultyLabel} · ${opponentLabel}`;
    }

    if (!elements.leaderboardList) {
      renderGameOverLeaderboard();
      return;
    }

    elements.leaderboardList.innerHTML = '';

    console.log(`Rendering leaderboard with ${filtered.length} players (after filtering):`, filtered);

    if (!filtered.length) {
      const li = document.createElement('li');
      li.className = 'meta';
      li.textContent = opponent 
        ? `No win streaks yet against ${opponent.label} in ${difficultyKey} mode. Play some games to start building streaks!`
        : `No win streaks found in ${difficultyKey} mode. Play some games to build a streak!`;
      elements.leaderboardList.appendChild(li);
      renderGameOverLeaderboard();
      return;
    }

    filtered.slice(0, 10).forEach((player, idx) => {
      const li = document.createElement('li');
      li.className = 'leaderboard-item';
      const streak = player.streak || player.win_streak || 0;
      const lastDate = player.last_win_date ? new Date(player.last_win_date) : null;
      const meta = lastDate ? lastDate.toLocaleString() : '';
      li.innerHTML = `<span class="rank">${idx + 1}</span><div><strong>${player.username || player.name || 'Unknown'}</strong><div class="meta">${meta}</div></div><span>${streak}</span>`;
      elements.leaderboardList.appendChild(li);
    });

    renderGameOverLeaderboard();
  }

  async function fetchRoundPoints(gameId) {
    if (!gameId) return;
    try {
      const data = await fetchJson('/round_points', { game_id: gameId });
      if (data) {
        state.roundInfo = data;
        if (data.round_points) applyDamagePills(data.round_points);
        updateAnalysisPanels();
      }
    } catch (err) {
      debugLog(`round_points error: ${err.message}`);
    }
  }

  async function fetchRecent() {
    if (!state.gameId) return;
    try {
      const data = await fetchJson('/recent', { game_id: state.gameId, limit: 10 });
      clearHistory();
      (data || []).forEach(item => addHistory(item));
    } catch (err) {
      debugLog(`recent error: ${err.message}`);
    }
  }

  async function fetchProbabilities() {
    // Probabilities are now included in the /play response debug field
    // This function is kept for backwards compatibility but does nothing
    // The debug info is updated directly in playMove()
    if (elements.botProbs) elements.botProbs.classList.remove('active');
    return;
  }

  function updateNameFromSession() {
    const code = generatePlayerCode(state.sessionId);
    state.playerCode = code;
    const inputValue = (elements.nameInput?.value || '').trim();
    const placeholderName = (state.playerName && state.playerName.toLowerCase() !== 'player')
      ? state.playerName
      : `Player-${code}`;

    if (elements.nameInput && !inputValue) {
      elements.nameInput.placeholder = placeholderName;
    }
    if (elements.playerCode) elements.playerCode.textContent = code;
    const displayName = inputValue || placeholderName;
    if (elements.playerNameLabel) elements.playerNameLabel.textContent = displayName;
    if (elements.playerNameDisplay) elements.playerNameDisplay.textContent = displayName;
    state.playerName = displayName;
  }

  function prepareNewGameUI(data) {
    state.gameId = data.game_id;
    state.sessionId = data.session_id;
    state.policy = data.bot_policy;
    const serverName = (data.player_name || '').trim();
    if (serverName) {
      state.playerName = serverName;
      if (elements.nameInput && !elements.nameInput.value) {
        elements.nameInput.placeholder = serverName;
      }
    }
    if (state.botPolicies.length) setActivePolicy(state.policy, true);
    state.lastRound = null;
    state.gameFinishedNotified = false;
    state.roundInfo = null;
    state.botProbabilities = null;
    state.botPredictionMeta = null;
    state.targetScore = data.target_score ?? 10;
    try {
      localStorage.setItem(SESSION_STORAGE_KEY, state.sessionId);
      const todayKey = getPstDateKey();
      if (todayKey) {
        localStorage.setItem(SESSION_DAY_KEY, todayKey);
      }
    } catch (err) {
      console.warn('[rps-lite] Unable to persist session details:', err);
    }
    updateNameFromSession();
    if (elements.gameId) elements.gameId.textContent = `Game ${state.gameId.slice(0, 8)}`;
    if (elements.targetScore) elements.targetScore.textContent = state.targetScore;
    updateHpCards(data.user_score ?? 0, data.bot_score ?? 0);
    renderBattle(null);
    clearHistory();
    setWinner('');
    toggleMoveButtons(false); // Initially disabled until damage values load
    applyDamagePills(data.points || {}); // This will enable buttons when damage is applied
    if (elements.roundSummary) elements.roundSummary.textContent = 'Choose your move to battle!';
    if (elements.lastRoundDetail) elements.lastRoundDetail.innerHTML = 'Choose your move to battle!';
    
    // Ensure bot display is correct
    if (elements.botFace) elements.botFace.textContent = emojiMap[state.policy] || '🤖';
    if (elements.botName) elements.botName.textContent = botNames[state.policy] || 'Bot';
    if (elements.botNameLabel) elements.botNameLabel.textContent = formatBotLabel(state.policy, 'Bot');
    
    hideGameOverLeaderboard();
    setGameSectionsActive(true);
    fetchRoundPoints(state.gameId);
    fetchRecent();
    fetchProbabilities();
    refreshSessionStats();
    updateAnalysisPanels();
    requestAnimationFrame(updateTriangleGeometry);
    scheduleSessionCodeRotation();

    // Freeze policy selection during game
    toggleGameControls(false);
  }

  function endGame() {
    if (!state.gameId) {
      setStatus('No game to end.', 'fail');
      return;
    }
    
    try {
      // Reset game state
      state.gameId = null;
      state.policy = null;
      state.lastRound = null;
      state.gameFinishedNotified = false;
      state.roundInfo = null;
      state.botProbabilities = null;
      state.botPredictionMeta = null;
      
      // Hide game sections and show empty state
      setGameSectionsActive(false);
      toggleMoveButtons(false);
      
  // Unfreeze policy selection
      toggleGameControls(true);
      
      setStatus('Game ended. Start a new game!');
      
      // Refresh data after game ends
      setTimeout(() => {
        refreshLeaderboard().catch(err => console.warn('Leaderboard refresh failed:', err));
        refreshBotPerformance().catch(err => console.warn('Bot performance refresh failed:', err));
      }, 100);
      
    } catch (err) {
      console.error('Error ending game:', err);
      setStatus('Game ended with errors. You can start a new game.', 'warn');
    }
  }

  function toggleGameControls(enabled) {
    // Toggle policy buttons
    const policyButtons = document.querySelectorAll('.policy-button');
    policyButtons.forEach(btn => {
      btn.disabled = !enabled;
    });
    
    // Toggle start/end buttons
    if (elements.startBtn) {
      elements.startBtn.hidden = !enabled;
    }
    if (elements.endBtn) {
      elements.endBtn.hidden = enabled;
    }
  }

  async function startGame() {
    // BUG FIX: Ensure we have a valid policy selected before starting
    if (!state.policy && state.botPolicies.length > 0) {
      state.policy = state.botPolicies[0].id;
      setActivePolicy(state.policy, true);
    }
    
    const selectedPolicy = state.policy || state.botPolicies[0]?.id || 'brian';
    const playerName = (elements.nameInput?.value || '').trim();
    const difficultyMode = 'normal';

    const body = {
      session_id: state.sessionId,
      player_name: playerName || undefined,
      user_name: playerName || undefined,
      policy: selectedPolicy,
      bot_policy: selectedPolicy,
      difficulty_mode: difficultyMode
    };

    console.log(`Starting game with policy: ${selectedPolicy} (${difficultyMode})`);
    setStatus('Starting game...');
    toggleMoveButtons(false);

    try {
      const started = performance.now();
      const res = await fetch(apiUrl('/start_game'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const elapsed = Math.round(performance.now() - started);
      recordApiLog({
        ts: new Date(),
        method: 'POST',
        url: apiUrl('/start_game'),
        status: res.status,
        ok: res.ok,
        elapsed,
        requestBody: body  // Log the request for debugging
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const data = await res.json();
      prepareNewGameUI(data);
      setStatus('Game ready. Choose your move!', 'ok');
      refreshLeaderboard();
      refreshBotPerformance();
      if (elements.botNameLabel) elements.botNameLabel.textContent = formatBotLabel(state.policy, 'Bot');
    } catch (err) {
      setStatus(`Failed to start game: ${err.message}`, 'fail');
      toggleMoveButtons(false);
    }
  }

  async function playMove(move) {
    if (!state.gameId) {
      setStatus('Start a game first.', 'fail');
      return;
    }

    if (state.gameFinishedNotified) {
      setStatus('Game is finished. Start a new game to keep playing.', 'warn');
      toggleMoveButtons(false);
      return;
    }

    setStatus(`Playing ${move}...`);
    toggleMoveButtons(false);

    try {
      const body = {
        game_id: state.gameId,
        user_move: move
      };
      const started = performance.now();
      const res = await fetch(apiUrl('/play'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const elapsed = Math.round(performance.now() - started);
      recordApiLog({
        ts: new Date(),
        method: 'POST',
        url: apiUrl('/play'),
        status: res.status,
        ok: res.ok,
        elapsed,
        requestBody: body
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const data = await res.json();
      
      // Record full response for debug logging
      recordApiLog({
        ts: new Date(),
        method: 'POST',
        url: apiUrl('/play'),
        status: res.status,
        ok: res.ok,
        elapsed,
        requestBody: body,
        responseBody: data,
        debug: data.debug
      });
      
      state.lastRound = data;
      addHistory(data);
      renderBattle(data);
      updateHpCards(data.user_score, data.bot_score, data);
      setStatus(`Round result: ${data.result.toUpperCase()}`, 'ok');
      hideGameOverLeaderboard();
      fetchProbabilities();
      setGameSectionsActive(true);
      
      // Update debug panels with prediction details
      if (state.showDebug && data.debug) {
        updateDebugPredictionInfo(data.debug, data);
      }

      if (data.finished) {
  state.gameFinishedNotified = true;
        const playerWon = data.winner === 'user';
        const isTie = data.winner === 'tie';
        const botLabel = formatBotLabel(state.policy, 'Bot');
        let bannerMessage;
        if (playerWon) {
          bannerMessage = '🎉 You win the match!';
        } else if (isTie) {
          bannerMessage = "🤝 It's a tie.";
        } else {
          bannerMessage = `${botLabel} wins this time.`;
        }

        setWinner(bannerMessage, playerWon);
        if (state.policy) {
          state.leaderboardOpponent = state.policy;
          renderLeaderboardControls();
        }
        showGameOverLeaderboard(data);
        toggleMoveButtons(false);
        toggleGameControls(true);
        refreshLeaderboard();
        refreshSessionStats();
        refreshBotPerformance().catch(err => console.warn('Bot performance refresh after match failed:', err));
      } else {
  state.gameFinishedNotified = false;
  fetchRoundPoints(state.gameId);
        hideGameOverLeaderboard();
        toggleMoveButtons(true);
      }
      updateAnalysisPanels();
    } catch (err) {
      setStatus(`Failed to play move: ${err.message}`, 'fail');
      toggleMoveButtons(true);
    }
  }

  async function resetSession(options = {}) {
    const { silent = false, skipReschedule = false } = options;
    // BUG FIX: End active game before resetting session
    if (state.gameId) {
      try {
        await endGame();
      } catch (err) {
        console.warn('Failed to end game before reset:', err);
      }
    }
    
    state.sessionId = null;
    state.gameId = null;
    state.lastRound = null;
    state.gameFinishedNotified = false;
    state.roundInfo = null;
    state.botProbabilities = null;
    state.botPredictionMeta = null;
    state.gameStats = null;
    state.sessionStats = null;
    try {
      localStorage.removeItem(SESSION_STORAGE_KEY);
      localStorage.removeItem(SESSION_DAY_KEY);
    } catch (err) {
      console.warn('[rps-lite] Unable to clear stored session details:', err);
    }
    if (elements.nameInput) elements.nameInput.value = '';
    if (elements.gameId) elements.gameId.textContent = '';
    setWinner('');
    if (!silent) {
      setStatus('Session reset. Start a new game to continue.', 'ok');
    }
    toggleMoveButtons(false);
    updateHpCards(0, 0);
    renderBattle(null);
    clearHistory();
    if (elements.roundSummary) elements.roundSummary.textContent = 'Session reset.';
    if (elements.lastRoundDetail) elements.lastRoundDetail.textContent = 'Session reset.';
  if (elements.botProbs) elements.botProbs.classList.remove('active');
  hideGameOverLeaderboard();
    setGameSectionsActive(false);
    renderSessionStats();
    updateAnalysisPanels();
    updateNameFromSession();
    requestAnimationFrame(updateTriangleGeometry);
    if (!skipReschedule) {
      scheduleSessionCodeRotation();
    }
  }

  function scheduleSessionCodeRotation() {
    if (!window || typeof window.setTimeout !== 'function') {
      return;
    }
    try {
      const secondsRemaining = secondsUntilNextPstMidnight();
      const delayMs = Math.max(secondsRemaining * 1000, 60 * 1000);
      if (scheduleSessionCodeRotation.timerId) {
        clearTimeout(scheduleSessionCodeRotation.timerId);
      }
      scheduleSessionCodeRotation.timerId = window.setTimeout(async () => {
        try {
          await resetSession({ silent: true, skipReschedule: true });
          setStatus('Session refreshed for a new day. Configure an opponent to begin.', 'ok');
        } catch (err) {
          console.warn('[rps-lite] Daily session reset failed:', err);
        } finally {
          ensureDailySessionFreshness();
          scheduleSessionCodeRotation();
        }
      }, delayMs);
    } catch (err) {
      console.warn('[rps-lite] Unable to schedule session code rotation:', err);
    }
  }

  function bindEvents() {
    elements.startBtn?.addEventListener('click', startGame);
    elements.endBtn?.addEventListener('click', endGame);
    elements.resetBtn?.addEventListener('click', resetSession);

    elements.monitoringButton?.addEventListener('click', () => {
      if (!state.grafanaUrl) return;
      window.open(state.grafanaUrl, '_blank', 'noopener');
    });

    elements.moveButtons.forEach(btn => btn.addEventListener('click', () => playMove(btn.dataset.move)));
    elements.leaderboardRefresh?.addEventListener('click', () => {
      refreshLeaderboard().catch(err => console.warn('Leaderboard refresh failed:', err));
      refreshBotPerformance().catch(err => console.warn('Bot performance refresh failed:', err));
    });
    elements.gameOverRefresh?.addEventListener('click', () => {
      refreshLeaderboard().catch(err => console.warn('Leaderboard refresh failed:', err));
      refreshBotPerformance().catch(err => console.warn('Bot performance refresh failed:', err));
    });

    if (elements.toggleDebug) {
      elements.toggleDebug.checked = state.showDebug;
      elements.debugPanel?.classList.toggle('active', state.showDebug);
      elements.toggleDebug.addEventListener('change', () => {
        state.showDebug = !!elements.toggleDebug?.checked;
        elements.debugPanel?.classList.toggle('active', state.showDebug);
        updateAnalysisPanels();
      });
    } else {
      state.showDebug = defaultDebugEnabled;
      elements.debugPanel?.classList.toggle('active', state.showDebug);
    }

    elements.nameInput?.addEventListener('input', () => {
      const name = elements.nameInput?.value || elements.nameInput?.placeholder || 'You';
      if (elements.playerNameLabel) elements.playerNameLabel.textContent = name;
      if (elements.playerNameDisplay) elements.playerNameDisplay.textContent = name;
      state.playerName = name;
    });

    elements.debugTestBotRates?.addEventListener('click', async () => {
      try {
        const result = await fetchJson('/bot_win_rates');
        debugLog(`bot_win_rates → ${JSON.stringify(result).slice(0, 400)}`);
      } catch (err) {
        debugLog(`bot_win_rates error: ${err.message}`);
      }
    });

    elements.debugTestStreaks?.addEventListener('click', async () => {
      try {
        const opponent = state.leaderboardOpponent || state.botPolicies[0]?.id;
        const result = await fetchJson('/win_streaks', { opponent, limit: 5 });
        debugLog(`win_streaks → ${JSON.stringify(result).slice(0, 400)}`);
      } catch (err) {
        debugLog(`win_streaks error: ${err.message}`);
      }
    });
  }

  function initializeFromSession() {
    updateNameFromSession();
    if (state.sessionId) {
      setStatus('Session restored. Start a new game when ready.', 'ok');
    } else {
      setStatus('Configure your opponent and start a game.');
    }
    toggleMoveButtons(false);
    setGameSectionsActive(false);
    hideGameOverLeaderboard();
    if (elements.botProbs) elements.botProbs.classList.remove('active');
    renderSessionStats();
    updateAnalysisPanels();
    requestAnimationFrame(updateTriangleGeometry);
  }

  async function bootstrap() {
    toggleMoveButtons(false);
    setupMonitoringLink();
    initializeFromSession();
    await fetchPolicies();
    await refreshBotPerformance();
    await refreshLeaderboard();
    renderApiLog();
    refreshSessionStats();
    scheduleBotPerformancePolling();
  }

  window.addEventListener('resize', () => requestAnimationFrame(updateTriangleGeometry));
  window.addEventListener('orientationchange', () => requestAnimationFrame(updateTriangleGeometry));

  bindEvents();
  bootstrap();
  scheduleSessionCodeRotation();
})();
