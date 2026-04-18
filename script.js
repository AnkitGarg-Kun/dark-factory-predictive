/* ═══════════════════════════════════════
   DARK FACTORY ML — JavaScript
   ML prediction now uses EXACT Python kNN model
   (loaded from model_data.json at startup)
═══════════════════════════════════════ */

// ── SAMPLE DATA (from CSV first 20 rows) ──
const SAMPLE_DATA = [
  { id:'CMP0001', type:'Hydraulic Cylinder', v:3.499, t:48.5,  p:181.8, h:2547, rul:870 },
  { id:'CMP0002', type:'Hydraulic Cylinder', v:4.875, t:90.5,  p:192.9, h:457,  rul:690 },
  { id:'CMP0003', type:'Hydraulic Cylinder', v:0.420, t:77.7,  p:286.8, h:1750, rul:670 },
  { id:'CMP0004', type:'Gear',               v:0.574, t:93.2,  p:191.1, h:1467, rul:51  },
  { id:'CMP0005', type:'Gear',               v:3.700, t:96.1,  p:202.3, h:596,  rul:343 },
  { id:'CMP0006', type:'Engine',             v:2.205, t:78.8,  p:225.4, h:2091, rul:731 },
  { id:'CMP0007', type:'Gear',               v:2.192, t:88.6,  p:265.1, h:1091, rul:325 },
  { id:'CMP0008', type:'Engine',             v:0.857, t:62.1,  p:140.3, h:1064, rul:952 },
  { id:'CMP0009', type:'Engine',             v:0.612, t:94.1,  p:50.5,  h:1391, rul:812 },
  { id:'CMP0010', type:'Hydraulic Cylinder', v:3.093, t:76.7,  p:291.4, h:2129, rul:175 },
  { id:'CMP0011', type:'Hydraulic Cylinder', v:2.590, t:88.8,  p:58.2,  h:1274, rul:744 },
  { id:'CMP0012', type:'Hydraulic Cylinder', v:3.510, t:40.9,  p:234.0, h:442,  rul:731 },
  { id:'CMP0013', type:'Hydraulic Cylinder', v:4.709, t:74.7,  p:61.5,  h:2105, rul:689 },
  { id:'CMP0014', type:'Engine',             v:1.313, t:91.3,  p:171.3, h:682,  rul:76  },
  { id:'CMP0015', type:'Hydraulic Cylinder', v:2.525, t:63.8,  p:86.8,  h:3961, rul:962 },
  { id:'CMP0016', type:'Gear',               v:1.194, t:42.2,  p:281.1, h:2594, rul:253 },
  { id:'CMP0017', type:'Engine',             v:0.276, t:79.7,  p:149.4, h:3550, rul:893 },
  { id:'CMP0018', type:'Engine',             v:4.241, t:85.1,  p:295.8, h:1108, rul:240 },
  { id:'CMP0019', type:'Engine',             v:1.747, t:50.4,  p:163.2, h:194,  rul:507 },
  { id:'CMP0020', type:'Gear',               v:2.597, t:79.9,  p:219.2, h:2616, rul:520 },
];

// ══════════════════════════════════════════════════════
// EXACT kNN MODEL — matches Python 3rul_prediction_model.py
// Uses model_data.json exported by export_model_data.py
// ══════════════════════════════════════════════════════
let MODEL = null; // loaded once from model_data.json

async function loadModel() {
  try {
    const resp = await fetch('model_data.json');
    MODEL = await resp.json();
    console.log('%c[kNN Model] Loaded successfully — ' + MODEL.knn.X_train.length + ' training points', 'color:#00d4ff');
  } catch(e) {
    console.warn('[kNN Model] Could not load model_data.json. Using fallback.', e);
  }
}
loadModel();

/**
 * Replicate Python MinMaxScaler.transform — exact same formula:
 *   X_scaled = X * scale_ + min_
 * where scale_ = 1 / (data_max - data_min)
 *       min_   = -data_min / (data_max - data_min)
 */
function minmaxScale(vib, temp, press, hrs) {
  const s = MODEL.scaler;
  const vals = [vib, temp, press, hrs];
  return vals.map((v, i) => v * s.scale[i] + s.min[i]);
}

/**
 * Replicate Python LabelEncoder + normalise by ct_max
 * classes: ["Engine", "Gear", "Hydraulic Cylinder"]
 */
function encodeComponent(type) {
  const classes = MODEL.label_encoder.classes;
  const ct_max  = MODEL.label_encoder.ct_max;
  const idx = classes.indexOf(type);
  if (idx === -1) throw new Error('Unknown component type: ' + type);
  return idx / ct_max;
}

/**
 * Replicate Python Health Index calculation:
 *   norm via scaler -> deg = (clip(norm,0)^1.3 @ W) * 2.0 -> HI = exp(-deg)
 */
function computeHealthIndex(type, vib, temp, press, hrs) {
  if (!MODEL) {
    // fallback approximation
    const Vn  = (vib - 0.11) / (5.0 - 0.11);
    const Tn  = (temp - 40)   / (100 - 40);
    const Pn  = (press - 50)  / (300 - 50);
    const Hn  = hrs           / 5000;
    const deg = 0.35*Vn + 0.25*Tn + 0.20*Pn + 0.20*Hn;
    return { HI: Math.exp(-deg * 2.0), deg };
  }
  const norm = minmaxScale(vib, temp, press, hrs);
  const W    = MODEL.weights; // [0.35, 0.25, 0.20, 0.20]
  let deg = 0;
  for (let i = 0; i < 4; i++) {
    const clipped = Math.max(0, norm[i]);
    deg += Math.pow(clipped, 1.3) * W[i];
  }
  deg *= 2.0;
  const HI = Math.exp(-deg);
  return { HI, deg };
}

/**
 * k=3 distance-weighted kNN — exact replica of Python KNeighborsRegressor
 * metric: euclidean, weights: 'distance'
 */
function computeMLRUL(type, vib, temp, press, hrs) {
  const { HI, deg } = computeHealthIndex(type, vib, temp, press, hrs);

  if (!MODEL) {
    // fallback if model not loaded yet
    const RUL_fallback = 50 + 950 * Math.pow(Math.max(0, HI), 1.4);
    const rul = Math.max(51, Math.min(999, Math.round(RUL_fallback)));
    return { rul, HI, deg };
  }

  // Build query vector: [norm_vib, norm_temp, norm_press, norm_hrs, ct_enc]
  const norm  = minmaxScale(vib, temp, press, hrs);
  const ct    = encodeComponent(type);
  const query = [...norm, ct];

  const X_train = MODEL.knn.X_train;
  const y_train = MODEL.knn.y_train;
  const k       = MODEL.knn.k; // 3

  // Compute euclidean distances to all training points
  const distances = X_train.map((row, i) => {
    let d2 = 0;
    for (let j = 0; j < query.length; j++) {
      const diff = query[j] - row[j];
      d2 += diff * diff;
    }
    return { dist: Math.sqrt(d2), idx: i };
  });

  // Sort ascending, take k nearest
  distances.sort((a, b) => a.dist - b.dist);
  const knn = distances.slice(0, k);

  // Distance-weighted average (if dist=0 → exact match)
  let weightedSum = 0, totalWeight = 0;
  const hasExact = knn.some(n => n.dist === 0);
  for (const n of knn) {
    if (hasExact) {
      if (n.dist === 0) { weightedSum += y_train[n.idx]; totalWeight += 1; }
    } else {
      const w = 1 / n.dist;
      weightedSum += w * y_train[n.idx];
      totalWeight += w;
    }
  }

  let rul = weightedSum / totalWeight;
  rul = Math.max(51, Math.min(999, Math.round(rul * 10) / 10)); // same clip as Python
  rul = Math.round(rul); // display as integer

  return { rul, HI, deg };
}


// ── NAV ────────────────────────────────────
const navbar   = document.getElementById('navbar');
const hamburger= document.getElementById('hamburger');
const navLinks = document.getElementById('navLinks');

window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 50);
});
hamburger.addEventListener('click', () => {
  navLinks.classList.toggle('open');
});
// close on link click
navLinks.querySelectorAll('a').forEach(a => {
  a.addEventListener('click', () => navLinks.classList.remove('open'));
});

// ── PARTICLES ───────────────────────────────
function initParticles() {
  const container = document.getElementById('particles');
  if (!container) return;
  for (let i = 0; i < 40; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    p.style.left   = Math.random()*100 + '%';
    p.style.top    = Math.random()*100 + '%';
    p.style.animationDelay    = Math.random()*6 + 's';
    p.style.animationDuration = (4 + Math.random()*6) + 's';
    const colors = ['#00d4ff','#a855f7','#f97316'];
    p.style.background = colors[Math.floor(Math.random()*3)];
    container.appendChild(p);
  }
}
initParticles();

// ── COUNTER ANIMATION ───────────────────────
function animateCounters() {
  document.querySelectorAll('.stat-num').forEach(el => {
    const target = parseInt(el.dataset.target);
    let start = 0;
    const duration = 1800;
    const step = (timestamp) => {
      if (!start) start = timestamp;
      const progress = Math.min((timestamp - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.floor(ease * target).toLocaleString();
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

// ── REVEAL ON SCROLL ────────────────────────
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('visible');
    }
  });
}, { threshold: 0.15 });

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

// Counter trigger
const heroSection = document.querySelector('.hero');
new IntersectionObserver((entries) => {
  if (entries[0].isIntersecting) animateCounters();
}, { threshold: 0.5 }).observe(heroSection);

// ── SAMPLE TABLE ─────────────────────────────
function populateSampleTable() {
  const tbody = document.getElementById('sampleTableBody');
  if (!tbody) return;
  SAMPLE_DATA.forEach(row => {
    const typeClass = row.type === 'Hydraulic Cylinder' ? 'type-hc'
                     : row.type === 'Gear' ? 'type-gear' : 'type-engine';
    const rulClass  = row.rul < 200 ? 'style="color:#ef4444"'
                     : row.rul < 500 ? 'style="color:#eab308"' : 'style="color:#22c55e"';
    tbody.innerHTML += `
      <tr>
        <td style="font-family:var(--font-mono);font-size:0.78rem;">${row.id}</td>
        <td><span class="type-badge ${typeClass}">${row.type}</span></td>
        <td>${row.v.toFixed(3)}</td>
        <td>${row.t.toFixed(1)}</td>
        <td>${row.p.toFixed(1)}</td>
        <td>${row.h.toLocaleString()}</td>
        <td class="rul-cell" ${rulClass}>${row.rul}</td>
      </tr>`;
  });
}
populateSampleTable();

// ── DONUT CHART ─────────────────────────────
function drawDonut() {
  const canvas = document.getElementById('componentChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const cx  = canvas.width / 2, cy = canvas.height / 2;
  const R   = 100, r = 65;
  const data = [
    { label:'Hydraulic Cylinder', value:0.334, color:'#00d4ff' },
    { label:'Gear',               value:0.333, color:'#a855f7' },
    { label:'Engine',             value:0.333, color:'#f97316' },
  ];
  let angle = -Math.PI/2;
  data.forEach(d => {
    const slice = d.value * 2 * Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, R, angle, angle + slice);
    ctx.closePath();
    ctx.fillStyle = d.color;
    ctx.fill();
    angle += slice;
  });
  // hole
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, 2*Math.PI);
  ctx.fillStyle = '#080f1f';
  ctx.fill();
  // center text
  ctx.fillStyle = '#00d4ff';
  ctx.font = 'bold 28px Outfit, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('1,000', cx, cy - 8);
  ctx.fillStyle = '#94a3b8';
  ctx.font = '13px Outfit, sans-serif';
  ctx.fillText('records', cx, cy + 14);
}
drawDonut();

// ── GAUGE DRAWING ────────────────────────────
function drawGauge(canvas, rul, maxRul=999) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W/2, cy = H * 0.9;
  const R  = Math.min(W,H*2)*0.4;
  const startAngle = Math.PI;
  const endAngle   = 0;
  const frac       = Math.min(1, rul / maxRul);

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, R, startAngle, endAngle);
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 14;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Color gradient arcs (green→yellow→red from left to right = high→low RUL)
  const grad = ctx.createLinearGradient(cx-R, cy, cx+R, cy);
  grad.addColorStop(0, '#ef4444');
  grad.addColorStop(0.4,'#eab308');
  grad.addColorStop(1, '#22c55e');

  const arcEnd = Math.PI + frac * Math.PI;
  ctx.beginPath();
  ctx.arc(cx, cy, R, startAngle, arcEnd);
  ctx.strokeStyle = grad;
  ctx.lineWidth = 14;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Glow
  ctx.shadowColor = frac > 0.6 ? '#22c55e' : frac > 0.35 ? '#eab308' : '#ef4444';
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.arc(cx, cy, R, startAngle, arcEnd);
  ctx.stroke();
  ctx.shadowBlur = 0;
}

// ── kNN NEIGHBOURS CHART ──────────────────
// Shows the 3 nearest neighbours for a sample query (Engine, Vib=2.2, Temp=78.8, Press=225, Hrs=2090)
// Python output: RUL = 732.3 h
function drawModelCompChart() {
  const canvas = document.getElementById('modelCompChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const pad = { top:40, right:30, bottom:70, left:60 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top  - pad.bottom;

  ctx.clearRect(0,0,W,H);

  // Sample: Engine | Vib=2.2, Temp=78.8, Press=225, Hrs=2090 → RUL 732h
  // Illustrative nearest neighbours (realistic distances from training data)
  const neighbours = [
    { label: 'Neighbour 1', dist: 0.031, rul: 731, color: '#00d4ff' },
    { label: 'Neighbour 2', dist: 0.058, rul: 749, color: '#22c55e' },
    { label: 'Neighbour 3', dist: 0.074, rul: 708, color: '#a855f7' },
  ];
  const predictedRUL = 732; // weighted avg ≈ Python output

  const maxRUL = 999;
  const barW = (plotW / (neighbours.length + 1)) * 0.6;
  const barGap = plotW / (neighbours.length + 1);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + plotH - (i / 5) * plotH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.3)'; ctx.font = '11px Outfit'; ctx.textAlign = 'right';
    ctx.fillText(Math.round((i / 5) * maxRUL), pad.left - 6, y + 4);
  }

  // Draw neighbour bars
  neighbours.forEach((n, i) => {
    const bH = (n.rul / maxRUL) * plotH;
    const bx = pad.left + (i + 0.5) * barGap + (barGap - barW) / 2;
    const by = pad.top + plotH - bH;

    const grad = ctx.createLinearGradient(0, by, 0, by + bH);
    grad.addColorStop(0, n.color);
    grad.addColorStop(1, n.color + '33');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.roundRect(bx, by, barW, bH, [6, 6, 0, 0]); ctx.fill();

    // RUL value
    ctx.fillStyle = '#f1f5f9'; ctx.font = 'bold 13px Outfit'; ctx.textAlign = 'center';
    ctx.fillText(n.rul + ' h', bx + barW / 2, by - 22);

    // Distance
    ctx.fillStyle = n.color; ctx.font = '11px Outfit';
    ctx.fillText('d = ' + n.dist.toFixed(3), bx + barW / 2, by - 8);

    // Label
    ctx.fillStyle = '#94a3b8'; ctx.font = '11px Outfit';
    ctx.fillText(n.label, bx + barW / 2, pad.top + plotH + 16);
    ctx.fillText('1/d = ' + (1/n.dist).toFixed(1), bx + barW / 2, pad.top + plotH + 30);
  });

  // Draw predicted RUL bar (last position)
  const pBH = (predictedRUL / maxRUL) * plotH;
  const pbx = pad.left + (neighbours.length + 0.5) * barGap + (barGap - barW) / 2;
  const pby = pad.top + plotH - pBH;
  const pg = ctx.createLinearGradient(0, pby, 0, pby + pBH);
  pg.addColorStop(0, '#f97316'); pg.addColorStop(1, '#f9731622');
  ctx.fillStyle = pg;
  ctx.beginPath(); ctx.roundRect(pbx, pby, barW, pBH, [6, 6, 0, 0]); ctx.fill();
  // Glow border
  ctx.strokeStyle = '#f97316'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.roundRect(pbx, pby, barW, pBH, [6, 6, 0, 0]); ctx.stroke();

  ctx.fillStyle = '#f1f5f9'; ctx.font = 'bold 13px Outfit'; ctx.textAlign = 'center';
  ctx.fillText(predictedRUL + ' h', pbx + barW / 2, pby - 22);
  ctx.fillStyle = '#f97316'; ctx.font = '11px Outfit';
  ctx.fillText('★ Predicted', pbx + barW / 2, pby - 8);
  ctx.fillStyle = '#94a3b8'; ctx.font = '11px Outfit';
  ctx.fillText('Weighted avg', pbx + barW / 2, pad.top + plotH + 16);
  ctx.fillText('(1/d weights)', pbx + barW / 2, pad.top + plotH + 30);

  // Y-axis label
  ctx.save();
  ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = '#94a3b8'; ctx.font = '12px Outfit'; ctx.textAlign = 'center';
  ctx.fillText('RUL (hours)', 0, 0);
  ctx.restore();

  // Title note
  ctx.fillStyle = 'rgba(255,255,255,0.25)'; ctx.font = '11px Outfit'; ctx.textAlign = 'center';
  ctx.fillText('Sample: Engine | Vib=2.2, Temp=78.8°C, Press=225 bar, Hrs=2090', W / 2, pad.top - 16);
}
drawModelCompChart();

// ── RANGE SLIDER DISPLAY ─────────────────────
function setupSliders() {
  const sliders = [
    { id:'vibration',   display:'vibVal',   decimals:2 },
    { id:'temperature', display:'tempVal',  decimals:0 },
    { id:'pressure',    display:'pressVal', decimals:0 },
    { id:'opHours',     display:'hoursVal', decimals:0 },
  ];
  sliders.forEach(s => {
    const input   = document.getElementById(s.id);
    const display = document.getElementById(s.display);
    if (!input || !display) return;
    const update = () => {
      const v = parseFloat(input.value);
      display.textContent = v.toFixed(s.decimals);
      // Update gradient fill
      const min = parseFloat(input.min), max = parseFloat(input.max);
      const pct = ((v-min)/(max-min))*100;
      input.style.background =
        `linear-gradient(90deg, var(--cyan) ${pct}%, rgba(255,255,255,0.1) ${pct}%)`;
    };
    input.addEventListener('input', update);
    update();
  });
}
setupSliders();

// ── PRESETS ──────────────────────────────────
const PRESETS = {
  low:    { type:'Engine',             v:0.5,  t:45,  p:70,  h:200  },
  medium: { type:'Gear',               v:2.5,  t:70,  p:175, h:2500 },
  high:   { type:'Hydraulic Cylinder', v:4.5,  t:95,  p:280, h:4800 },
};
function loadPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  document.getElementById('compType').value = p.type;
  const setSlider = (id, val) => {
    const el = document.getElementById(id);
    if (el) { el.value = val; el.dispatchEvent(new Event('input')); }
  };
  setSlider('vibration',   p.v);
  setSlider('temperature', p.t);
  setSlider('pressure',    p.p);
  setSlider('opHours',     p.h);
  runPrediction();
}

window.addEventListener('load', drawModelCompChart);

// ── RELIABILITY MODEL ────────────────────────
function computeReliability(v, t, p, h, rul) {
  const Vn = (v - 0.1) / (5.0 - 0.1);
  const Tn = (t - 40) / (100 - 40);
  const Pn = (p - 50) / (300 - 50);
  const Hn = (h - 0) / (5000 - 0);

  const S = 0.25 * Vn + 0.25 * Tn + 0.25 * Pn + 0.25 * Hn;
  const lambda_0 = 0.001;
  const lambda = lambda_0 * (1 + S);

  const R = Math.exp(-lambda * rul);
  const F = 1 - R;

  return { S, lambda, R, F };
}

// ── MAIN PREDICTION ──────────────────────────
function runPrediction() {
  // If model hasn't loaded yet, wait and retry
  if (!MODEL) {
    const recBox = document.getElementById('recBox');
    if (recBox) {
      recBox.textContent = '⏳ Loading AI model... please wait and try again.';
      recBox.style.background = 'rgba(0,212,255,0.1)';
      recBox.style.color = '#00d4ff';
    }
    loadModel().then(() => {
      if (MODEL) runPrediction();
    });
    return;
  }

  const type = document.getElementById('compType').value;
  const v    = parseFloat(document.getElementById('vibration').value);
  const t    = parseFloat(document.getElementById('temperature').value);
  const p    = parseFloat(document.getElementById('pressure').value);
  const h    = parseFloat(document.getElementById('opHours').value);

  const { rul: mlRUL, HI, deg }          = computeMLRUL(type, v, t, p, h);
  const { R, F }                         = computeReliability(v, t, p, h, mlRUL);

  // Status
  const dot    = document.getElementById('statusDot');
  const status = HI > 0.60 ? 'Healthy' : HI > 0.35 ? 'Warning' : 'Critical';
  const color  = HI > 0.60 ? 'healthy' : HI > 0.35 ? 'warning'  : 'critical';
  const statusLabel = HI > 0.60 ? '🟢 Healthy' : HI > 0.35 ? '🟡 Warning' : '🔴 Critical';

  dot.className = 'status-dot ' + color;
  document.getElementById('statusText').textContent = statusLabel;

  // Gauge
  const gaugeCanvas = document.getElementById('gaugeCanvas');
  drawGauge(gaugeCanvas, mlRUL);

  // RUL display
  const rulEl = document.getElementById('rulValue');
  let curr = parseInt(rulEl.textContent) || 0;
  const diff = mlRUL - curr;
  const steps = 30;
  let step = 0;
  const interval = setInterval(() => {
    step++;
    const val = Math.round(curr + diff * (step/steps));
    rulEl.textContent = val;
    if (step >= steps) { clearInterval(interval); rulEl.textContent = mlRUL; }
  }, 16);

  // Details
  document.getElementById('mlRul').textContent  = mlRUL + ' hrs';
  document.getElementById('relR').textContent   = (R * 100).toFixed(2) + '%';
  document.getElementById('failF').textContent  = (F * 100).toFixed(2) + '%';

  // Recommendation
  let rec = '', recColor = '';
  if (mlRUL < 100) {
    rec = '🔴 IMMEDIATE MAINTENANCE REQUIRED';
    recColor = 'rgba(239,68,68,0.15)';
  } else if (mlRUL < 300) {
    rec = '🟠 SCHEDULE MAINTENANCE SOON';
    recColor = 'rgba(249,115,22,0.15)';
  } else if (mlRUL < 600) {
    rec = '🟡 MONITOR CLOSELY';
    recColor = 'rgba(234,179,8,0.15)';
  } else {
    rec = '🟢 COMPONENT IN GOOD HEALTH';
    recColor = 'rgba(34,197,94,0.15)';
  }
  const recBox = document.getElementById('recBox');
  recBox.textContent = rec;
  recBox.style.background = recColor;
  recBox.style.color = '#f1f5f9';

  // Timeline
  const countdown = document.getElementById('maintCountdown');
  countdown.style.display = 'block';
  const nowWidth = 5;
  const maintWidth = Math.min(95, 5 + (mlRUL / 999) * 90);
  document.getElementById('tl-maint') &&
    (document.getElementById('tl-maint').style.width = maintWidth + '%');
  document.getElementById('timelineMaint').style.cssText =
    `left:${nowWidth}%; width:${maintWidth-nowWidth}%;`;


}

// ── WINDOW RESIZE → redraw charts ────────────
window.addEventListener('resize', () => {
  drawModelCompChart();
});

// ── SCROLL SMOOTH ACTIVE LINK ─────────────────
const sections = document.querySelectorAll('section[id]');
window.addEventListener('scroll', () => {
  const scrollY = window.scrollY;
  sections.forEach(s => {
    const top = s.offsetTop - 120;
    const bot = top + s.offsetHeight;
    const link = document.querySelector(`.nav-links a[href="#${s.id}"]`);
    if (link) {
      link.style.color = scrollY >= top && scrollY < bot ? 'var(--cyan)' : '';
    }
  });
});

console.log('%c🏭 DarkFactory ML — Predictive Maintenance System', 'color:#00d4ff;font-size:16px;font-weight:bold;');
console.log('%cUniversity Project | Kaggle Data | RUL Prediction', 'color:#94a3b8;');
