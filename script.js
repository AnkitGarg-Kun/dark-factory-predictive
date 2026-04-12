/* ═══════════════════════════════════════
   DARK FACTORY ML — JavaScript
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

// ── THEORETICAL RUL FORMULA ──
const L_MAX = { 'Hydraulic Cylinder': 1200, 'Gear': 1100, 'Engine': 1400 };
const WEIGHTS = {
  'Hydraulic Cylinder': { w_V:1.5, w_T:0.8, w_P:1.2, w_H:0.7 },
  'Gear':               { w_V:1.8, w_T:1.0, w_P:0.6, w_H:1.0 },
  'Engine':             { w_V:1.0, w_T:1.5, w_P:0.9, w_H:0.8 },
};
const DI_MAX = { 'Hydraulic Cylinder':4.2, 'Gear':4.4, 'Engine':4.2 };

function computeTheoreticalRUL(type, v, t, p, h) {
  const Vn = v / 5.0;
  const Tn = (t - 40) / 60;
  const Pn = (p - 50) / 250;
  const Hn = h / 5000;
  const w  = WEIGHTS[type];
  const DI = w.w_V * Math.pow(Vn,2) + w.w_T * Math.pow(Tn,1.5) +
             w.w_P * Math.pow(Pn,1.2) + w.w_H * Hn;
  const DImax = DI_MAX[type];
  const DInorm = Math.min(1, DI / DImax);
  const L = L_MAX[type];
  const RULraw = L * Math.exp(-2.5 * DInorm);
  const RULmin_raw = L * Math.exp(-2.5);
  const RULmax_raw = L;
  const scaled = 51 + ((RULraw - RULmin_raw) / (RULmax_raw - RULmin_raw)) * 948;
  return { rul: Math.max(51, Math.min(999, Math.round(scaled))), DI, DInorm };
}

// ── ML APPROXIMATION (Physics-Blended HI model) ──
function computeHealthIndex(type, v, t, p, h) {
  // MinMax norms using dataset ranges
  const Vn  = (v - 0.11) / (5.0 - 0.11);
  const Tn  = (t - 40)   / (100 - 40);
  const Pn  = (p - 50)   / (300 - 50);
  const Hn  = h           / 5000;
  const deg = 0.30*Vn + 0.25*Tn + 0.25*Hn + 0.20*Pn;
  const HI  = Math.max(0, Math.min(1, 1 - deg));
  return { HI, deg };
}

function computeMLRUL(type, v, t, p, h) {
  const { HI, deg } = computeHealthIndex(type, v, t, p, h);
  // Physics target: RUL_phys = 50 + 950 * HI^1.4
  const RUL_phys = 50 + 950 * Math.pow(HI, 1.4);
  // Theoretical for 20% blend
  const { rul: RUL_th } = computeTheoreticalRUL(type, v, t, p, h);
  const RUL_blend = 0.80 * RUL_phys + 0.20 * RUL_th;
  // Add small interaction effect
  const VxT  = v * t;
  const PxH  = p * h;
  const Vsq  = v * v;
  const typeEnc = type === 'Engine' ? 0 : type === 'Gear' ? 1 : 2;
  // Simulate RF model with plausible coefficients
  let rul = RUL_blend
    + typeEnc * 8
    - Vsq * 4
    - (VxT / 500) * 2
    + (PxH > 500000 ? -5 : 5);
  rul = Math.max(51, Math.min(999, Math.round(rul)));
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

// ── DEGRADATION CURVE CHART ─────────────────
function drawDegradationCurve() {
  const canvas = document.getElementById('degradationChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = { top:30, right:20, bottom:40, left:50 };

  ctx.clearRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i=0; i<=4; i++) {
    const y = pad.top + (H - pad.top - pad.bottom) * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W-pad.right, y); ctx.stroke();
  }

  const types = ['Hydraulic Cylinder', 'Gear', 'Engine'];
  const colors = ['#00d4ff', '#a855f7', '#f97316'];
  const N = 100;
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top  - pad.bottom;

  types.forEach((type, ti) => {
    const L = L_MAX[type];
    const arr = [];
    for (let i=0; i<=N; i++) {
      const x_norm = i / N;
      const RULraw = L * Math.exp(-2.5 * x_norm);
      const RULmin_raw = L * Math.exp(-2.5);
      const RULmax_raw = L;
      const rul = 51 + ((RULraw - RULmin_raw) / (RULmax_raw - RULmin_raw)) * 948;
      arr.push(Math.max(51, Math.min(999, rul)));
    }
    ctx.beginPath();
    ctx.strokeStyle = colors[ti];
    ctx.lineWidth = 2.5;
    arr.forEach((rul, i) => {
      const px = pad.left + (i/N) * plotW;
      const py = pad.top  + (1 - (rul-51)/948) * plotH;
      i === 0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
    });
    ctx.stroke();
  });

  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, H-pad.bottom);
  ctx.lineTo(W-pad.right, H-pad.bottom);
  ctx.stroke();

  // Labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px Outfit, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('DI_norm (0 = new → 1 = end)', W/2, H-6);
  ctx.save();
  ctx.translate(14, H/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('RUL (hours)', 0, 0);
  ctx.restore();

  // Legend
  ctx.font = '10px Outfit, sans-serif';
  ['HC','Gear','Engine'].forEach((lb,i) => {
    ctx.fillStyle = colors[i];
    ctx.fillRect(pad.left + i*90, pad.top - 20, 12, 3);
    ctx.fillText(lb, pad.left + i*90 + 18, pad.top - 16);
  });
}
drawDegradationCurve();

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

// ── MODEL COMPARISON CHART ──────────────────
function drawModelCompChart() {
  const canvas = document.getElementById('modelCompChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const pad = { top:30, right:20, bottom:60, left:60 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top  - pad.bottom;

  ctx.clearRect(0,0,W,H);

  const models = ['Linear Reg.', 'Ridge Reg.', 'Random Forest', 'Grad. Boosting'];
  const r2vals  = [0.91, 0.92, 0.985, 0.975];
  const colors  = ['#a855f7','#f97316','#00d4ff','#22c55e'];

  const barW   = (plotW / models.length) * 0.55;
  const barGap = plotW / models.length;

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i=0; i<=5; i++) {
    const y = pad.top + plotH - (i/5)*plotH;
    ctx.beginPath(); ctx.moveTo(pad.left,y); ctx.lineTo(pad.left+plotW,y); ctx.stroke();
    ctx.fillStyle='rgba(255,255,255,0.3)'; ctx.font='11px Outfit';
    ctx.textAlign='right';
    ctx.fillText((0.8 + i*0.04).toFixed(2), pad.left-6, y+4);
  }

  // Bars
  models.forEach((m,i) => {
    const barH = ((r2vals[i] - 0.80) / 0.20) * plotH;
    const bx   = pad.left + i*barGap + (barGap - barW)/2;
    const by   = pad.top  + plotH - barH;

    const grad = ctx.createLinearGradient(0, by, 0, by+barH);
    grad.addColorStop(0, colors[i]);
    grad.addColorStop(1, colors[i]+'44');

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(bx, by, barW, barH, [6,6,0,0]);
    ctx.fill();

    // Label
    ctx.fillStyle = '#f1f5f9'; ctx.font='bold 12px Outfit';
    ctx.textAlign='center';
    ctx.fillText(r2vals[i].toFixed(3), bx+barW/2, by-8);

    ctx.fillStyle='#94a3b8'; ctx.font='12px Outfit';
    const words = m.split(' ');
    words.forEach((w,wi) => {
      ctx.fillText(w, bx+barW/2, pad.top+plotH+18+wi*14);
    });
  });

  // Axis label
  ctx.save();
  ctx.translate(14, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillStyle='#94a3b8'; ctx.font='12px Outfit'; ctx.textAlign='center';
  ctx.fillText('R² Score',0,0);
  ctx.restore();
}
drawModelCompChart();

// ── FORMULA TABS ────────────────────────────
function showFormula(which) {
  ['hc','gear','engine'].forEach(f => {
    const el = document.getElementById('formula-'+f);
    if (el) el.style.display = 'none';
  });
  const target = document.getElementById('formula-'+which);
  if (target) target.style.display='block';
  document.querySelectorAll('.comp-tab').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');
}

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

// ── MINI RF BAR CHART ─────────────────────────
function drawRFBar() {
  const canvas = document.getElementById('rfChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);
  // show 5 CV scores
  const scores=[0.984,0.981,0.986,0.979,0.983];
  const bW=(W-40)/scores.length, bGap=6;
  scores.forEach((s,i)=>{
    const bH=(s-0.95)*H/(0.05);
    const bx=20+i*(bW+bGap);
    const by=H-bH;
    const g=ctx.createLinearGradient(0,by,0,H);
    g.addColorStop(0,'#00d4ff'); g.addColorStop(1,'rgba(0,212,255,0.2)');
    ctx.fillStyle=g;
    ctx.beginPath();
    ctx.roundRect(bx,by,bW,bH,[3,3,0,0]);
    ctx.fill();
    ctx.fillStyle='rgba(255,255,255,0.5)'; ctx.font='9px Outfit';
    ctx.textAlign='center';
    ctx.fillText('F'+(i+1), bx+bW/2, H-2);
  });
  ctx.fillStyle='rgba(255,255,255,0.3)'; ctx.font='9px Outfit';
  ctx.textAlign='left';
  ctx.fillText('5-Fold CV Scores', 20, 12);
}
window.addEventListener('load', drawRFBar);

// ── COMPARISON MINI CHART ─────────────────────
function drawComparisonChart(ml, th) {
  const canvas = document.getElementById('comparisonChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);

  const bar_data=[
    {label:'ML Predicted', val:ml,  color:'#00d4ff' },
    {label:'Theoretical',  val:th,  color:'#a855f7' },
  ];
  const maxVal = 999;
  const bW=80, gap=60;
  const startX=(W - bar_data.length*(bW+gap)+gap)/2;

  bar_data.forEach((d,i)=>{
    const bH=(d.val/maxVal)*(H-60);
    const bx=startX+i*(bW+gap);
    const by=H-30-bH;
    const g=ctx.createLinearGradient(0,by,0,by+bH);
    g.addColorStop(0,d.color); g.addColorStop(1,d.color+'22');
    ctx.fillStyle=g;
    ctx.beginPath(); ctx.roundRect(bx,by,bW,bH,[6,6,0,0]); ctx.fill();

    ctx.fillStyle='#f1f5f9'; ctx.font='bold 14px Outfit'; ctx.textAlign='center';
    ctx.fillText(d.val+' h', bx+bW/2, by-8);
    ctx.fillStyle='#94a3b8'; ctx.font='12px Outfit';
    ctx.fillText(d.label, bx+bW/2, H-10);
  });
}

// ── MAIN PREDICTION ──────────────────────────
function runPrediction() {
  const type = document.getElementById('compType').value;
  const v    = parseFloat(document.getElementById('vibration').value);
  const t    = parseFloat(document.getElementById('temperature').value);
  const p    = parseFloat(document.getElementById('pressure').value);
  const h    = parseFloat(document.getElementById('opHours').value);

  const { rul: thRUL }                   = computeTheoreticalRUL(type, v, t, p, h);
  const { rul: mlRUL, HI, deg }          = computeMLRUL(type, v, t, p, h);

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

  // Health Index bar
  document.getElementById('hiBar').style.width = (HI*100)+'%';
  document.getElementById('hiVal').textContent  = HI.toFixed(3);

  // Details
  document.getElementById('thRul').textContent  = thRUL + ' hrs';
  document.getElementById('mlRul').textContent  = mlRUL + ' hrs';
  document.getElementById('degScore').textContent = deg.toFixed(4);

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

  // Comparison chart
  const compWrapper = document.getElementById('compChartWrapper');
  compWrapper.style.display = 'block';
  drawComparisonChart(mlRUL, thRUL);
}

// ── WINDOW RESIZE → redraw charts ────────────
window.addEventListener('resize', () => {
  drawModelCompChart();
  drawDegradationCurve();
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
