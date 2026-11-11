/* canguard dashboard
 *
 * Structure:
 *  1. theme + helpers        5. events list + player
 *  2. tab navigation         6. diagnose chat
 *  3. websocket stream       7. evaluate tab (metrics.json)
 *  4. monitor charts
 */

"use strict";

/* ------------------------------------------------------------------ *
 * 1. Theme + helpers
 * ------------------------------------------------------------------ */

const cssVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const ATTACK_LABELS = { dos: "DoS", fuzzy: "Fuzzy", gear: "Gear", rpm: "RPM",
                        normal: "Normal" };
const attackColor = (type) => cssVar(`--c-${type}`) || cssVar("--c-normal");

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("canguard-theme", theme); } catch (_) {}
  restyleCharts();
}
try {
  applyTheme(localStorage.getItem("canguard-theme") || "light");
} catch (_) { applyTheme("light"); }

document.getElementById("theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next);
});

function chartDefaults() {
  Chart.defaults.font.family = cssVar("--font-mono");
  Chart.defaults.font.size = 11;
  Chart.defaults.color = cssVar("--muted");
  Chart.defaults.borderColor = cssVar("--line");
  Chart.defaults.animation = false;
  Chart.defaults.plugins.legend.display = false;
  Chart.defaults.plugins.tooltip.backgroundColor = cssVar("--panel");
  Chart.defaults.plugins.tooltip.titleColor = cssVar("--text");
  Chart.defaults.plugins.tooltip.bodyColor = cssVar("--text");
  Chart.defaults.plugins.tooltip.borderColor = cssVar("--line");
  Chart.defaults.plugins.tooltip.borderWidth = 1;
}

const charts = {};           // name -> Chart instance
function restyleCharts() {
  if (typeof Chart === "undefined") return;
  chartDefaults();
  Object.values(charts).forEach((c) => c && c.destroy());
  Object.keys(charts).forEach((k) => delete charts[k]);
  buildScoreChart();
  drawHeatmap();
  if (evalData) renderEvaluate(evalData);
}

const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(d));
const pct95 = (arr) => {
  if (!arr.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(0.95 * s.length))];
};

/* ------------------------------------------------------------------ *
 * 2. Tabs
 * ------------------------------------------------------------------ */

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    button.classList.add("active");
    document.getElementById(`tab-${button.dataset.tab}`).classList.add("active");
    if (button.dataset.tab === "evaluate" && !evalData) loadEvaluate();
  });
});

/* ------------------------------------------------------------------ *
 * 3. WebSocket stream
 * ------------------------------------------------------------------ */

const state = {
  playing: false, position: 0, duration: 1, speed: 1,
  source: "rpm", model: "lstm_autoencoder",
  scores: [],        // {t, score, threshold, true_attack, is_attack, type, suspect}
  frames: [],        // raw frame rows (latest last), capped
  events: [],        // closed + open alarm events
  openEvent: null,
  latencies: [],     // model inference ms (rolling)
  heatCols: [],      // {t, by_id: {}}
  heatIds: [],       // row order
  showAllFrames: false,
};

let socket = null;
function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/stream`);
  socket.onopen = () => setLive(true);
  socket.onclose = () => { setLive(false); setTimeout(connect, 2000); };
  socket.onmessage = (event) => handleMessage(JSON.parse(event.data));
}
function send(cmd, value) {
  if (socket && socket.readyState === 1) socket.send(JSON.stringify({ cmd, value }));
}
function setLive(on) {
  document.getElementById("live-pill").classList.toggle("on", on);
}

function handleMessage(msg) {
  if (msg.type === "reset") {
    Object.assign(state, {
      scores: [], frames: [], events: [], openEvent: null,
      latencies: [], heatCols: [], heatIds: [],
    });
    applyStatus(msg.status);
    renderAll();
    return;
  }
  if (msg.type !== "tick") return;
  applyStatus(msg.status);

  if (msg.stats) {
    document.getElementById("stat-fps").textContent = msg.stats.frames_per_s;
    document.getElementById("stat-load").innerHTML =
      `${fmt(msg.stats.bus_load_pct, 1)}<span class="stat-unit"> %</span>`;
    document.getElementById("stat-ids").textContent = msg.stats.active_ids;
  }
  if (msg.frames.length) {
    state.frames.push(...msg.frames);
    if (state.frames.length > 400) state.frames.splice(0, state.frames.length - 400);
    renderFrames();
  }
  if (msg.heatmap) {
    state.heatCols.push(msg.heatmap);
    if (state.heatCols.length > 48) state.heatCols.shift();
    drawHeatmap();
  }
  for (const detection of msg.detections) ingestDetection(detection);
  if (msg.detections.length) {
    updateScoreChart();
    renderBusState(msg.detections[msg.detections.length - 1]);
    const p95 = pct95(state.latencies);
    document.getElementById("stat-latency").innerHTML =
      `${fmt(p95, 1)}<span class="stat-unit"> ms</span>`;
    document.getElementById("meta-updated").textContent =
      new Date().toLocaleTimeString();
  }
}

function applyStatus(status) {
  state.playing = status.playing;
  state.position = status.position;
  state.duration = status.duration || 1;
  state.speed = status.speed;
  document.getElementById("play-button").classList.toggle("playing", state.playing);
  document.getElementById("player-time").textContent =
    `${fmt(state.position, 1)} / ${fmt(state.duration, 1)} s`;
  const frac = state.position / state.duration;
  document.getElementById("progress-fill").style.width = `${frac * 100}%`;
  document.getElementById("progress-handle").style.left = `${frac * 100}%`;
  document.getElementById("speed-pill").textContent =
    `${state.speed % 1 ? state.speed.toFixed(1) : state.speed}×`;
}

function ingestDetection(d) {
  state.scores.push(d);
  if (state.scores.length > 2000) state.scores.shift();
  state.latencies.push(d.latency_ms);
  if (state.latencies.length > 400) state.latencies.shift();

  const typeFilter = document.getElementById("filter-type").value;
  if (d.is_attack) {
    if (!state.openEvent) {
      state.openEvent = { t: d.t, type: d.attack_type, score: d.score,
                          suspect: d.suspect_id, end: d.t };
      state.events.unshift(state.openEvent);
      if (state.events.length > 60) state.events.pop();
    } else {
      state.openEvent.end = d.t;
      state.openEvent.score = Math.max(state.openEvent.score, d.score);
      if (d.attack_type !== "normal") state.openEvent.type = d.attack_type;
    }
    renderEvents(typeFilter);
    renderMarkers();
  } else if (state.openEvent) {
    state.openEvent = null;
    renderEvents(typeFilter);
  }
}

/* ------------------------------------------------------------------ *
 * 4. Monitor charts
 * ------------------------------------------------------------------ */

// Shades ground-truth attack periods and tags the latest flagged point
const shadePlugin = {
  id: "attackShade",
  beforeDatasetsDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    const points = chart.$scorePoints || [];
    if (!points.length) return;
    ctx.save();
    let runStart = null;
    const color = attackColor(state.source === "normal" ? "normal" : state.source);
    for (let i = 0; i <= points.length; i++) {
      const inAttack = i < points.length && points[i].true_attack;
      if (inAttack && runStart === null) runStart = points[i].t;
      if (!inAttack && runStart !== null) {
        const x0 = scales.x.getPixelForValue(runStart);
        const x1 = scales.x.getPixelForValue(points[Math.max(0, i - 1)].t);
        ctx.fillStyle = color + "22";
        ctx.fillRect(x0, chartArea.top, Math.max(x1 - x0, 2),
                     chartArea.bottom - chartArea.top);
        runStart = null;
      }
    }
    ctx.restore();
  },
  afterDatasetsDraw(chart) {
    const { ctx, scales, chartArea } = chart;
    const points = chart.$scorePoints || [];
    const flagged = [...points].reverse().find((p) => p.is_attack);
    if (!flagged) return;
    const x = scales.x.getPixelForValue(flagged.t);
    const y = scales.y.getPixelForValue(flagged.score);
    const label = `${ATTACK_LABELS[flagged.attack_type] || flagged.attack_type}` +
      ` · 0x${flagged.suspect_id} · ${flagged.score.toFixed(2)}`;
    ctx.save();
    ctx.font = `500 10px ${cssVar("--font-mono")}`;
    const w = ctx.measureText(label).width + 12;
    let bx = Math.min(x + 8, chartArea.right - w);
    let by = Math.max(y - 22, chartArea.top + 2);
    ctx.fillStyle = cssVar("--panel");
    ctx.strokeStyle = cssVar("--alert");
    ctx.beginPath();
    ctx.roundRect(bx, by, w, 16, 8);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = cssVar("--text");
    ctx.fillText(label, bx + 6, by + 11);
    ctx.fillStyle = cssVar("--alert");
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  },
};

function buildScoreChart() {
  const canvas = document.getElementById("score-chart");
  charts.score = new Chart(canvas, {
    type: "line",
    data: { datasets: [
      { label: "anomaly score", data: [], parsing: false,
        borderColor: cssVar("--data"), borderWidth: 2, pointRadius: 0,
        cubicInterpolationMode: "monotone", tension: 0.3 },
      { label: "threshold", data: [], parsing: false,
        borderColor: cssVar("--alert"), borderWidth: 1, borderDash: [5, 4],
        pointRadius: 0 },
    ]},
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { type: "linear", title: { display: true, text: "replay time (s)" },
             ticks: { maxTicksLimit: 8 } },
        y: { min: 0, max: 1,
             title: { display: true, text: "anomaly score" } },
      },
      plugins: { tooltip: {
        callbacks: {
          title: (items) => `t = ${items[0].parsed.x.toFixed(2)} s`,
          label: (item) => item.datasetIndex === 0
            ? ` score ${item.parsed.y.toFixed(3)}` : ` threshold ${item.parsed.y.toFixed(3)}`,
        },
      }},
    },
    plugins: [shadePlugin],
  });
  updateScoreChart();
}

function updateScoreChart() {
  const chart = charts.score;
  if (!chart) return;
  const WINDOW_S = 120;
  const points = state.scores.filter(
    (p) => p.t >= state.position - WINDOW_S);
  chart.$scorePoints = points;
  chart.data.datasets[0].data = points.map((p) => ({ x: p.t, y: p.score }));
  chart.data.datasets[1].data = points.length
    ? [{ x: points[0].t, y: points[0].threshold },
       { x: points[points.length - 1].t, y: points[points.length - 1].threshold }]
    : [];
  chart.update("none");
}

function renderBusState(d) {
  const card = document.getElementById("bus-state");
  const value = document.getElementById("bus-state-value");
  const detail = document.getElementById("bus-state-detail");
  card.classList.toggle("attack", d.is_attack);
  if (d.is_attack) {
    value.textContent = "Under attack";
    const label = ATTACK_LABELS[d.attack_type] || d.attack_type;
    const duration = state.openEvent
      ? (d.t - state.openEvent.t).toFixed(1) : "0.0";
    detail.innerHTML =
      `<span class="type-pill" style="background:${attackColor(d.attack_type)}">${label}</span>` +
      ` score ${d.score.toFixed(2)} · ID 0x${d.suspect_id} · ${duration} s`;
  } else {
    value.textContent = "Normal";
    detail.textContent = `score ${d.score.toFixed(2)} · threshold ${d.threshold.toFixed(2)}`;
  }
}

/* --- heatmap: CAN ID x time, diverging Δt deviation ----------------- */

function divergingColor(v) {
  // v in [-1, 1]: negative = faster (teal), positive = slower (purple)
  const clamp = Math.max(-1, Math.min(1, v));
  const t = Math.abs(clamp);
  const neutral = cssVar("--surface");
  const target = clamp < 0 ? cssVar("--div-fast") : cssVar("--div-slow");
  return mixColors(neutral, target, t);
}
function mixColors(a, b, t) {
  const parse = (c) => {
    if (c.startsWith("#")) {
      const n = parseInt(c.slice(1), 16);
      return [n >> 16 & 255, n >> 8 & 255, n & 255];
    }
    const m = c.match(/hsl\(([\d.]+)[ ,]+([\d.]+)%[ ,]+([\d.]+)%\)/);
    if (!m) return [128, 128, 128];
    return hslToRgb(+m[1], +m[2] / 100, +m[3] / 100);
  };
  const ca = parse(a), cb = parse(b);
  const mix = ca.map((x, i) => Math.round(x + (cb[i] - x) * t));
  return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
}
function hslToRgb(h, s, l) {
  const k = (n) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  return [f(0), f(8), f(4)].map((x) => Math.round(x * 255));
}

function drawHeatmap() {
  const canvas = document.getElementById("heatmap");
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const columns = state.heatCols;
  if (!columns.length) {
    ctx.fillStyle = cssVar("--muted");
    ctx.font = `11px ${cssVar("--font-mono")}`;
    ctx.fillText("waiting for stream…", 8, 20);
    return;
  }
  // Row order: IDs seen so far, capped to the 18 most recently active
  const idFilter = document.getElementById("filter-id").value.trim().toUpperCase();
  const seen = new Map();
  columns.forEach((col) =>
    Object.keys(col.by_id).forEach((id) => seen.set(id, true)));
  let ids = [...seen.keys()].sort();
  if (idFilter) ids = ids.filter((id) => id.includes(idFilter));
  ids = ids.slice(0, 18);

  const labelW = 42, labelH = 16;
  const gap = 3, radius = 3;
  const cellW = Math.max(4, (rect.width - labelW) / columns.length - gap);
  const cellH = Math.max(4, (rect.height - labelH) / Math.max(ids.length, 1) - gap);

  ctx.font = `10px ${cssVar("--font-mono")}`;
  ctx.fillStyle = cssVar("--muted");
  ids.forEach((id, r) => {
    ctx.fillText(id, 4, labelH + r * (cellH + gap) + cellH / 2 + 3);
  });
  columns.forEach((col, c) => {
    const x = labelW + c * (cellW + gap);
    if (c % Math.ceil(columns.length / 6) === 0) {
      ctx.fillStyle = cssVar("--muted");
      ctx.fillText(`${col.t.toFixed(0)}s`, x, 10);
    }
    ids.forEach((id, r) => {
      const v = col.by_id[id];
      const y = labelH + r * (cellH + gap);
      ctx.fillStyle = v === undefined ? cssVar("--surface") : divergingColor(v);
      ctx.beginPath();
      ctx.roundRect(x, y, cellW, cellH, radius);
      ctx.fill();
    });
  });
}
window.addEventListener("resize", drawHeatmap);

/* ------------------------------------------------------------------ *
 * 5. Events list, frames table, player
 * ------------------------------------------------------------------ */

function renderEvents(typeFilter = "") {
  const list = document.getElementById("events-list");
  const events = typeFilter
    ? state.events.filter((e) => e.type === typeFilter) : state.events;
  if (!events.length) {
    list.innerHTML = `<li class="events-empty">no events yet</li>`;
    return;
  }
  list.innerHTML = events.slice(0, 30).map((e, i) => `
    <li data-t="${e.t}">
      <span class="event-dot" style="background:${attackColor(e.type)}"></span>
      <span>${ATTACK_LABELS[e.type] || e.type} · 0x${e.suspect}</span>
      <span class="events-time">t=${e.t.toFixed(1)}s · ${(e.end - e.t).toFixed(1)}s</span>
      <span class="events-score">${e.score.toFixed(2)}</span>
    </li>`).join("");
  list.querySelectorAll("li[data-t]").forEach((li) => {
    li.addEventListener("click", () => send("seek", parseFloat(li.dataset.t)));
  });
}

function renderFrames() {
  const body = document.getElementById("frames-body");
  const idFilter = document.getElementById("filter-id").value.trim().toUpperCase();
  let rows = state.frames;
  if (idFilter) rows = rows.filter((f) => f.id.includes(idFilter));
  const n = state.showAllFrames ? 30 : 6;
  body.innerHTML = rows.slice(-n).reverse().map((f) => `
    <tr class="${f.label ? "flagged" : ""}">
      <td>${f.t.toFixed(4)}</td><td>${f.id}</td><td>${f.dlc}</td>
      <td>${f.data}</td><td>${f.label ? "T" : "R"}</td>
    </tr>`).join("");
}

function renderMarkers() {
  const wrap = document.getElementById("progress-markers");
  wrap.innerHTML = state.events.map((e) => `
    <span class="progress-marker"
          style="left:${(e.t / state.duration) * 100}%;
                 background:${attackColor(e.type)}"></span>`).join("");
}

function renderAll() {
  renderEvents();
  renderFrames();
  renderMarkers();
  updateScoreChart();
  drawHeatmap();
  document.getElementById("bus-state").classList.remove("attack");
  document.getElementById("bus-state-value").textContent = "idle";
  document.getElementById("bus-state-detail").textContent =
    "press play to start the replay";
}

document.getElementById("play-button").addEventListener("click", () =>
  send(state.playing ? "pause" : "play"));

const SPEEDS = [1, 2, 5, 10];
document.getElementById("speed-pill").addEventListener("click", () => {
  const next = SPEEDS[(SPEEDS.indexOf(state.speed) + 1) % SPEEDS.length] || 1;
  send("speed", next);
});

document.getElementById("progress").addEventListener("click", (event) => {
  const rect = event.currentTarget.getBoundingClientRect();
  const frac = (event.clientX - rect.left) / rect.width;
  send("seek", frac * state.duration);
});

document.getElementById("source-select").addEventListener("change", (e) =>
  send("source", e.target.value) || (state.source = e.target.value));
document.getElementById("model-select").addEventListener("change", (e) => {
  state.model = e.target.value;
  document.getElementById("meta-model").textContent =
    e.target.selectedOptions[0].textContent;
  send("model", e.target.value);
});
document.getElementById("source-select").addEventListener("change", (e) => {
  state.source = e.target.value;
});

document.getElementById("filter-id").addEventListener("input", () => {
  renderFrames(); drawHeatmap();
});
document.getElementById("filter-type").addEventListener("change", (e) =>
  renderEvents(e.target.value));

document.getElementById("frames-toggle").addEventListener("click", (e) => {
  state.showAllFrames = !state.showAllFrames;
  e.target.textContent = state.showAllFrames ? "Show less" : "Show all";
  renderFrames();
});

/* ------------------------------------------------------------------ *
 * 6. Diagnose chat
 * ------------------------------------------------------------------ */

const chat = document.getElementById("chat");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");

document.querySelectorAll(".example-pill").forEach((pill) =>
  pill.addEventListener("click", () => {
    chatInput.value = pill.textContent;
    chatForm.requestSubmit();
  }));

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;
  const empty = document.getElementById("chat-empty");
  if (empty) empty.remove();

  appendMessage("user", question);
  chatInput.value = "";
  const sendButton = chatForm.querySelector(".chat-send");
  sendButton.disabled = true;
  const pending = appendMessage("assistant", "…");
  try {
    const response = await fetch("/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || response.statusText);
    pending.firstChild.textContent = data.answer;
    const codes = (data.sources || [])
      .map((s) => s.code || s.source).filter(Boolean);
    if (codes.length) {
      const pills = document.createElement("div");
      pills.className = "source-pills";
      pills.innerHTML = [...new Set(codes)]
        .map((c) => `<span class="source-pill">${c}</span>`).join("");
      pending.appendChild(pills);
    }
    if (!data.llm_used && data.grounded) {
      const note = document.createElement("div");
      note.className = "msg-note";
      note.textContent = "LLM offline — raw knowledge-base entries shown.";
      pending.appendChild(note);
    }
  } catch (error) {
    pending.firstChild.textContent = `Request failed: ${error.message}`;
  } finally {
    sendButton.disabled = false;
    chat.scrollTop = chat.scrollHeight;
  }
});

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg msg-${role}`;
  const span = document.createElement("span");
  span.textContent = text;
  div.appendChild(span);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

/* ------------------------------------------------------------------ *
 * 7. Evaluate tab
 * ------------------------------------------------------------------ */

let evalData = null;
const MODEL_META = {
  isolation_forest: { label: "Isolation Forest", color: "#5b8def" },
  lstm_autoencoder: { label: "LSTM-AE", color: () => cssVar("--c-dos") },
  lightgbm: { label: "LightGBM", color: () => cssVar("--c-fuzzy") },
};
const modelColor = (name) => {
  const c = MODEL_META[name].color;
  return typeof c === "function" ? c() : c;
};

async function loadEvaluate() {
  try {
    const response = await fetch("/metrics");
    if (!response.ok) throw new Error("metrics not available yet");
    evalData = await response.json();
    renderEvaluate(evalData);
  } catch (error) {
    document.getElementById("compare-table").innerHTML =
      `<tr><td class="eval-note">Evaluation results not found — run ` +
      `python -m src.models.evaluate first (${error.message}).</td></tr>`;
  }
}

const errorBarPlugin = {
  id: "errorBars",
  afterDatasetsDraw(chart) {
    const errors = chart.$errors;
    if (!errors) return;
    const { ctx } = chart;
    ctx.save();
    ctx.strokeStyle = cssVar("--text");
    ctx.lineWidth = 1;
    chart.data.datasets.forEach((dataset, di) => {
      const meta = chart.getDatasetMeta(di);
      meta.data.forEach((bar, i) => {
        const std = errors[di][i];
        if (!std) return;
        const y0 = chart.scales.y.getPixelForValue(dataset.data[i] - std);
        const y1 = chart.scales.y.getPixelForValue(
          Math.min(1, dataset.data[i] + std));
        ctx.beginPath();
        ctx.moveTo(bar.x, y0); ctx.lineTo(bar.x, y1);
        ctx.moveTo(bar.x - 3, y0); ctx.lineTo(bar.x + 3, y0);
        ctx.moveTo(bar.x - 3, y1); ctx.lineTo(bar.x + 3, y1);
        ctx.stroke();
      });
    });
    ctx.restore();
  },
};

function renderEvaluate(data) {
  const det = data.detection;
  if (!det) return;
  const attacks = ["dos", "fuzzy", "gear", "rpm"];
  const models = Object.keys(MODEL_META);

  // --- F1 grouped bars with error bars
  if (charts.f1) charts.f1.destroy();
  const f1Canvas = document.getElementById("f1-chart");
  const datasets = models.map((m) => ({
    label: MODEL_META[m].label,
    data: attacks.map((a) => det.models[m].per_attack_f1[a].mean),
    backgroundColor: modelColor(m),
    borderRadius: 3,
  }));
  charts.f1 = new Chart(f1Canvas, {
    type: "bar",
    data: { labels: attacks.map((a) => ATTACK_LABELS[a]), datasets },
    options: {
      maintainAspectRatio: false,
      scales: { y: { min: 0, max: 1, title: { display: true, text: "F1" } } },
      plugins: {
        legend: { display: true, position: "bottom",
                  labels: { boxWidth: 10, boxHeight: 10 } },
        tooltip: { callbacks: { label: (item) => {
          const std = det.models[models[item.datasetIndex]]
            .per_attack_f1[attacks[item.dataIndex]].std;
          return ` ${item.dataset.label}: ${item.parsed.y.toFixed(3)} ± ${std.toFixed(3)}`;
        } } },
      },
    },
    plugins: [errorBarPlugin],
  });
  charts.f1.$errors = models.map((m) =>
    attacks.map((a) => det.models[m].per_attack_f1[a].std));

  // --- ROC curves
  if (charts.roc) charts.roc.destroy();
  charts.roc = new Chart(document.getElementById("roc-chart"), {
    type: "line",
    data: { datasets: models.map((m) => ({
      label: `${MODEL_META[m].label} (AUC ${det.models[m].window.roc_auc.mean.toFixed(3)})`,
      data: det.models[m].roc_curve.fpr.map((x, i) =>
        ({ x, y: det.models[m].roc_curve.tpr[i] })),
      parsing: false, borderColor: modelColor(m), borderWidth: 1.6,
      pointRadius: 0,
    })) },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { min: 0, max: 1, title: { display: true, text: "false positive rate" } },
        y: { min: 0, max: 1, title: { display: true, text: "true positive rate" } },
      },
      plugins: { legend: { display: true, position: "bottom",
                           labels: { boxWidth: 10, boxHeight: 10 } } },
    },
  });

  // --- Confusion matrix (sequential scale, values in cells)
  const cm = det.models.lightgbm.confusion_matrix;
  const wrap = document.getElementById("confusion-matrix");
  const n = cm.labels.length;
  wrap.style.gridTemplateColumns = `56px repeat(${n}, 1fr)`;
  const lo = cssVar("--seq-lo"), hi = cssVar("--seq-hi");
  let html = `<div class="confusion-head"></div>` + cm.labels.map((l) =>
    `<div class="confusion-head">${ATTACK_LABELS[l] || l}</div>`).join("");
  cm.counts.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    html += `<div class="confusion-head">${ATTACK_LABELS[cm.labels[i]] || cm.labels[i]}</div>`;
    html += row.map((v) => {
      const t = v / total;
      const bg = mixColors(lo, hi, Math.sqrt(t));
      const fg = t > 0.45 ? "#fff" : cssVar("--text");
      return `<div class="confusion-cell" style="background:${bg};color:${fg}"
                   title="${v.toLocaleString()} windows">${v.toLocaleString()}</div>`;
    }).join("");
  });
  wrap.innerHTML = html;

  // --- Latency box plot (custom floating bars + whisker plugin)
  if (charts.latency) charts.latency.destroy();
  const stats = models.map((m) => {
    const samples = [...det.models[m].latency_ms.samples].sort((a, b) => a - b);
    const q = (p) => samples[Math.floor(p * (samples.length - 1))];
    return { min: q(0.05), q1: q(0.25), med: q(0.5), q3: q(0.75), max: q(0.95) };
  });
  const boxPlugin = {
    id: "whiskers",
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      const meta = chart.getDatasetMeta(0);
      ctx.save();
      ctx.strokeStyle = cssVar("--text");
      ctx.lineWidth = 1;
      meta.data.forEach((bar, i) => {
        const s = stats[i];
        const y = (v) => chart.scales.y.getPixelForValue(v);
        ctx.beginPath();
        ctx.moveTo(bar.x, y(s.q3)); ctx.lineTo(bar.x, y(s.max));
        ctx.moveTo(bar.x - 5, y(s.max)); ctx.lineTo(bar.x + 5, y(s.max));
        ctx.moveTo(bar.x, y(s.q1)); ctx.lineTo(bar.x, y(s.min));
        ctx.moveTo(bar.x - 5, y(s.min)); ctx.lineTo(bar.x + 5, y(s.min));
        ctx.stroke();
        ctx.beginPath();
        const half = bar.width / 2;
        ctx.moveTo(bar.x - half, y(s.med)); ctx.lineTo(bar.x + half, y(s.med));
        ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1;
      });
      ctx.restore();
    },
  };
  charts.latency = new Chart(document.getElementById("latency-chart"), {
    type: "bar",
    data: {
      labels: models.map((m) => MODEL_META[m].label),
      datasets: [{
        data: stats.map((s) => [s.q1, s.q3]),
        backgroundColor: models.map((m) => modelColor(m) + "66"),
        borderColor: models.map(modelColor), borderWidth: 1,
        borderSkipped: false, borderRadius: 3, barPercentage: 0.45,
      }],
    },
    options: {
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true,
                     title: { display: true, text: "latency (ms)" } } },
      plugins: { tooltip: { callbacks: { label: (item) => {
        const s = stats[item.dataIndex];
        return ` median ${s.med.toFixed(2)} ms · IQR ${s.q1.toFixed(2)}–${s.q3.toFixed(2)}`;
      } } } },
    },
    plugins: [boxPlugin],
  });

  // --- comparison table
  const compare = document.getElementById("compare-table");
  const windowRow = (m) => det.models[m].window;
  compare.innerHTML = `
    <tr><th>model</th><th class="num">precision</th><th class="num">recall</th>
        <th class="num">F1</th><th class="num">ROC-AUC</th>
        <th class="num">train (s)</th><th class="num">infer (ms)</th></tr>` +
    models.map((m) => `
      <tr><td>${MODEL_META[m].label}</td>
        <td class="num">${windowRow(m).precision.mean.toFixed(3)} ± ${windowRow(m).precision.std.toFixed(3)}</td>
        <td class="num">${windowRow(m).recall.mean.toFixed(3)} ± ${windowRow(m).recall.std.toFixed(3)}</td>
        <td class="num">${windowRow(m).f1.mean.toFixed(3)} ± ${windowRow(m).f1.std.toFixed(3)}</td>
        <td class="num">${windowRow(m).roc_auc.mean.toFixed(3)}</td>
        <td class="num">${det.models[m].train_time_s.mean.toFixed(1)}</td>
        <td class="num">${det.models[m].latency_ms.mean.toFixed(2)}</td>
      </tr>`).join("");

  // --- episode table
  const episode = document.getElementById("episode-table");
  episode.innerHTML = `
    <tr><th>model</th><th>attack</th><th class="num">episode recall</th>
        <th class="num">latency med (s)</th><th class="num">p95 (s)</th>
        <th class="num">false alarms / h</th></tr>` +
    models.map((m) => attacks.map((a, i) => {
      const e = det.models[m].episodes.per_attack[a];
      const fa = det.models[m].episodes.false_alarms_per_hour;
      return `<tr>
        <td>${i === 0 ? MODEL_META[m].label : ""}</td>
        <td><span class="event-dot" style="background:${attackColor(a)};
             display:inline-block;margin-right:6px"></span>${ATTACK_LABELS[a]}</td>
        <td class="num">${(e.episode_recall.mean * 100).toFixed(1)} %</td>
        <td class="num">${e.latency_median_s.mean.toFixed(2)}</td>
        <td class="num">${e.latency_p95_s.mean.toFixed(2)}</td>
        <td class="num">${i === 0 ? fa.mean.toFixed(2) + " ± " + fa.std.toFixed(2) : ""}</td>
      </tr>`;
    }).join("")).join("");
}

/* ------------------------------------------------------------------ *
 * boot
 * ------------------------------------------------------------------ */

chartDefaults();
buildScoreChart();
renderAll();
connect();
