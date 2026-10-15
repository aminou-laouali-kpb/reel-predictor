/* Reel Predictor — interface (JS vanilla, aucune dépendance réseau ; uPlot servi localement). */
'use strict';

(() => {
  // ====================================================================== constantes
  const INDEX_KEYS = ['attention', 'action', 'tension', 'manipulation', 'emotion', 'memory'];
  const INDEX_LABELS = {
    attention: 'Attention', action: "Signal d'action", tension: 'Tension',
    manipulation: 'Manipulation', emotion: 'Émotion', memory: 'Mémoire',
  };
  const INDEX_COLORS = {
    attention: '#4cc9f0', action: '#f8961e', tension: '#ff5d73',
    manipulation: '#c77dff', emotion: '#ffd166', memory: '#7bd88f',
  };
  const RETENTION_COLOR = '#e9ecf2';
  const CONF_LABELS = { high: 'confiance élevée', medium: 'confiance moyenne', low: 'confiance faible' };
  const SEV_LABELS = { high: 'Élevée', medium: 'Moyenne', low: 'Faible' };
  const SEV_ORDER = { high: 0, medium: 1, low: 2 };
  const SEV_FILL = { high: 'rgba(255,107,107,0.16)', medium: 'rgba(245,184,74,0.13)', low: 'rgba(90,169,255,0.10)' };
  const LOW_CONTEXT_S = 5;
  const INSIGHT_FIELDS = [
    { id: 'views', label: 'Vues', kind: 'count' },
    { id: 'reach', label: 'Comptes touchés', kind: 'count' },
    { id: 'likes', label: "J'aime", kind: 'count' },
    { id: 'comments', label: 'Commentaires', kind: 'count' },
    { id: 'shares', label: 'Partages', kind: 'count' },
    { id: 'saves', label: 'Enregistrements', kind: 'count' },
    { id: 'follows', label: 'Abonnements', kind: 'count' },
    { id: 'watch_time_s', label: 'Durée de visionnage totale', kind: 'duration' },
    { id: 'avg_watch_s', label: 'Durée moyenne de visionnage', kind: 'duration' },
    { id: 'skip_rate', label: 'Taux de skip (%)', kind: 'percent' },
  ];
  const AXIS_STYLE = {
    stroke: '#8f8f8f',
    grid: { stroke: 'rgba(255,255,255,0.07)', width: 1 },
    ticks: { stroke: 'rgba(255,255,255,0.15)', width: 1, size: 4 },
    font: '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  };

  // ====================================================================== utilitaires
  const $ = (sel, root = document) => root.querySelector(sel);
  const nf0 = new Intl.NumberFormat('fr-FR', { maximumFractionDigits: 0 });
  const nf1 = new Intl.NumberFormat('fr-FR', { maximumFractionDigits: 1 });
  const nf2 = new Intl.NumberFormat('fr-FR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const hasUPlot = typeof window.uPlot === 'function';

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
      else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
      else if (k === 'dataset') Object.assign(node.dataset, v);
      else node.setAttribute(k, v === true ? '' : String(v));
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return node;
  }

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

  function fmtTime(s) {
    if (!isNum(s)) return '–';
    s = Math.max(0, s);
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${String(sec).padStart(2, '0')}`;
  }
  function fmtDuration(s) {
    if (!isNum(s)) return '–';
    if (s < 60) return `${nf1.format(s)} s`;
    const parts = [];
    let r = Math.round(s);
    const d = Math.floor(r / 86400); r -= d * 86400;
    const h = Math.floor(r / 3600); r -= h * 3600;
    const m = Math.floor(r / 60); r -= m * 60;
    if (d) parts.push(`${d} j`);
    if (h) parts.push(`${h} h`);
    if (m) parts.push(`${m} min`);
    if (r && !d) parts.push(`${r} s`);
    return parts.join(' ');
  }
  function fmtEta(s) {
    if (!isNum(s)) return '';
    if (s < 5) return 'presque terminé';
    if (s < 60) return `≈ ${Math.round(s)} s restantes`;
    const m = Math.floor(s / 60);
    const r = Math.round(s % 60);
    return `≈ ${m} min ${r ? r + ' s ' : ''}restantes`;
  }
  const fmtPct = (r, digits = 0) => (isNum(r) ? `${(r * 100).toFixed(digits).replace('.', ',')} %` : '–');
  function fmtDate(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString('fr-FR', { dateStyle: 'short', timeStyle: 'short' });
  }
  function debounce(fn, ms) {
    let t;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  }

  async function api(url, opts = {}) {
    const res = await fetch(url, opts);
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`;
      try {
        const j = await res.json();
        if (j && j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
      } catch (_) { /* corps non JSON */ }
      const err = new Error(msg);
      err.status = res.status;
      throw err;
    }
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) return res.json();
    return res.arrayBuffer();
  }

  // ====================================================================== état
  const state = {
    jobId: null,
    summary: null,
    timeline: [],
    duration: 0,
    charts: [],
    sec: -1,
    raf: 0,
    pollToken: 0,
    reels: [],
    shots: [],
    ocr: null,
    cmpCharts: [],
    coachPlan: [],
  };

  // ====================================================================== onglets
  const tabs = [$('#tab-btn-before'), $('#tab-btn-after')];
  function selectTab(btn, focus = false) {
    for (const t of tabs) {
      const on = t === btn;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    }
    if (focus) btn.focus();
    if (btn.id === 'tab-btn-after') After.open();
    else resizeAll();
  }
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => selectTab(t));
    t.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        selectTab(tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length], true);
      }
    });
  });

  // ====================================================================== santé
  async function refreshHealth() {
    const box = $('#health');
    try {
      const h = await api('/api/health');
      const chips = [];
      if (h.fake_mode) chips.push(['Mode factice', 'warn', 'REEL_PREDICTOR_FAKE=1 : les analyses sont simulées']);
      const st = h.status || {};
      if (!h.pipeline_available && !h.fake_mode) {
        chips.push(['Pipeline indisponible', 'bad', h.pipeline_error || '']);
      } else if (st.brain_expected === 'tribe' && (st.hf_token === false || st.llama_access === false)) {
        chips.push(['TRIBE v2 (sans texte)', 'warn', st.reason || 'Accès Llama 3.2 absent : voir scripts/check_hf_access.py']);
      } else if (st.brain_expected === 'tribe') {
        chips.push(['TRIBE v2 prêt', 'good', st.tribe_model || '']);
      } else if (st.brain_expected === 'heuristic') {
        chips.push(['Mode heuristique', 'warn', st.reason || 'TRIBE indisponible : indices estimés sans modèle cérébral']);
      }
      if (st.assets && typeof st.assets === 'object') {
        const missing = Object.entries(st.assets).filter(([, ok]) => ok === false).map(([k]) => k);
        if (missing.length) chips.push([`Modèles manquants (${missing.length})`, 'warn', `${missing.join(', ')} : lancez make setup`]);
      }
      if (h.assets && !h.assets.uplot) chips.push(['uPlot manquant', 'bad', 'Lancez make setup']);
      if (h.assets && !h.assets.brain_mesh) chips.push(['Maillage cérébral manquant', 'bad', 'Lancez make setup']);
      box.replaceChildren(...chips.map(([t, cls, title]) => el('span', { class: `chip ${cls}`, title }, t)));
    } catch (e) {
      box.replaceChildren(el('span', { class: 'chip bad', title: e.message }, 'Serveur injoignable'));
    }
  }

  // ====================================================================== dépôt & progression
  const dropzone = $('#dropzone');
  const fileInput = $('#file-input');
  function wireDropzone(zone, input, onFiles) {
    zone.addEventListener('click', (e) => { if (e.target.tagName !== 'INPUT') input.click(); });
    zone.addEventListener('keydown', (e) => {
      if ((e.key === 'Enter' || e.key === ' ') && e.target === zone) { e.preventDefault(); input.click(); }
    });
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('over'));
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('over');
      if (e.dataTransfer && e.dataTransfer.files.length) onFiles([...e.dataTransfer.files]);
    });
    input.addEventListener('change', () => { if (input.files.length) onFiles([...input.files]); input.value = ''; });
  }
  wireDropzone(dropzone, fileInput, (files) => uploadVideo(files[0]));
  $('#browse-btn').addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });

  function showUploadError(msg) {
    const b = $('#upload-error');
    b.textContent = msg || '';
    b.hidden = !msg;
  }

  function setScanning(on) {
    document.body.classList.toggle('scanning', on);
    $('#hero-scan').hidden = !on;
    if (on) {
      document.body.classList.remove('has-result');
      $('#result').hidden = true;
      $('#empty-state').hidden = false;
    }
    if (heroBrain) heroBrain.setMode(on ? 'scan' : 'idle');
  }

  function renderProgress(st) {
    const box = $('#progress');
    box.hidden = false;
    const pct = clamp(Number(st.pct) || 0, 0, 1);
    if (!document.body.classList.contains('scanning')) setScanning(true);
    $('#hero-scan-stage').textContent = st.stage_label || 'Analyse';
    $('#hero-scan-pct').textContent = `${Math.round(pct * 100)} %`;
    $('#progress-stage').textContent = st.stage_label || (st.status === 'queued' ? 'En attente' : 'Analyse');
    $('#progress-pct').textContent = `${Math.round(pct * 100)} %`;
    $('#progress-fill').style.width = `${(pct * 100).toFixed(1)}%`;
    $('#progress-bar').setAttribute('aria-valuenow', String(Math.round(pct * 100)));
    $('#progress-msg').textContent = st.message || '';
    $('#progress-eta').textContent = st.status === 'running' ? fmtEta(st.eta_s) : '';
  }

  function uploadVideo(file) {
    if (!file) return;
    showUploadError('');
    if (!/^video\//.test(file.type) && !/\.(mp4|mov|m4v)$/i.test(file.name)) {
      showUploadError(`« ${file.name} » ne semble pas être une vidéo : déposez un fichier .mp4.`);
      return;
    }
    renderProgress({ stage_label: 'Envoi de la vidéo', pct: 0, message: file.name, status: 'uploading' });
    window.scrollTo({ top: 0, behavior: 'smooth' });
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/analyze');
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        renderProgress({ stage_label: 'Envoi de la vidéo', pct: 0, message: `${file.name} — ${Math.round((e.loaded / e.total) * 100)} % envoyé`, status: 'uploading' });
      }
    });
    xhr.addEventListener('load', () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300 && body.job_id) {
        pollJob(body.job_id);
      } else {
        { $('#progress').hidden = true; setScanning(false); }
        showUploadError(`Envoi refusé : ${body.detail || xhr.statusText || xhr.status}`);
      }
    });
    xhr.addEventListener('error', () => {
      { $('#progress').hidden = true; setScanning(false); }
      showUploadError("Envoi impossible : le serveur local ne répond pas.");
    });
    const fd = new FormData();
    fd.append('file', file, file.name);
    xhr.send(fd);
  }

  async function pollJob(jobId) {
    const token = ++state.pollToken;
    for (;;) {
      if (token !== state.pollToken) return;
      let st;
      try {
        st = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      } catch (e) {
        { $('#progress').hidden = true; setScanning(false); }
        showUploadError(`Suivi de l'analyse impossible : ${e.message}`);
        return;
      }
      renderProgress(st);
      if (st.status === 'done') {
        setTimeout(() => { if (token === state.pollToken) $('#progress').hidden = true; }, 800);
        setScanning(false);
        await refreshHistory();
        await loadResult(jobId);
        if (state.jobId === jobId) Reveal.open();
        return;
      }
      if (st.status === 'error') {
        { $('#progress').hidden = true; setScanning(false); }
        showUploadError(st.error || st.message || "L'analyse a échoué.");
        refreshHistory();
        return;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  // ====================================================================== historique
  async function refreshHistory() {
    const list = $('#history');
    try {
      const data = await api('/api/reels');
      state.reels = data.reels || [];
      state.calibration = data.calibration || null;
    } catch (e) {
      list.replaceChildren(el('li', { class: 'empty' }, `Liste indisponible : ${e.message}`));
      return;
    }
    const reels = state.reels.filter((r) => r.result_available);
    if (!reels.length) {
      list.replaceChildren(el('li', { class: 'empty' }, 'Aucune analyse pour le moment.'));
      return;
    }
    list.replaceChildren(...reels.map((r) => {
      const flags = [r.fake ? 'factice' : null, r.degraded ? 'dégradé' : null].filter(Boolean).join(' · ');
      return el('li', {},
        el('button', {
          type: 'button',
          'aria-current': r.job_id === state.jobId ? 'true' : 'false',
          title: `Ouvrir l'analyse ${r.job_id}`,
          onclick: () => loadResult(r.job_id),
        },
        el('span', { class: 'h-name' }, r.filename || r.job_id),
        el('span', { class: 'h-score' }, isNum(r.score) ? Math.round(r.score) : '–'),
        el('span', { class: 'h-date' }, `${fmtDate(r.analyzed_at)}${flags ? ' · ' + flags : ''}`)));
    }));
  }

  // ====================================================================== carte cérébrale
  const PALETTE_STOPS = [
    [0.0, [49, 54, 149]], [0.25, [116, 173, 209]], [0.5, [247, 247, 247]], [0.75, [244, 109, 67]], [1.0, [165, 0, 38]],
  ];
  const LUT = (() => {
    const lut = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      const x = i / 255;
      let k = 0;
      while (k < PALETTE_STOPS.length - 2 && x > PALETTE_STOPS[k + 1][0]) k++;
      const [x0, c0] = PALETTE_STOPS[k];
      const [x1, c1] = PALETTE_STOPS[k + 1];
      const f = clamp((x - x0) / (x1 - x0), 0, 1);
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(c0[c] + (c1[c] - c0[c]) * f);
    }
    return lut;
  })();

  class BrainMap {
    constructor(canvas, emptyEl) {
      this.canvas = canvas;
      this.emptyEl = emptyEl;
      this.ctx = canvas.getContext('2d');
      this.mesh = null;
      this.meshError = null;
      this.data = null;
      this.meta = null;
      this.t = -1;
      this.raster = null;
    }

    async loadMesh() {
      if (this.mesh || this.meshError) return this.mesh;
      if (!this.meshPromise) {
        this.meshPromise = fetch('/brain_mesh.json').then((r) => {
          if (!r.ok) throw new Error('maillage absent (lancez « make setup »)');
          return r.json();
        });
      }
      try {
        this.mesh = await this.meshPromise;
        this.prepareMesh();
      } catch (e) {
        this.meshError = e.message;
      }
      return this.mesh;
    }

    prepareMesh() {
      const m = this.mesh;
      this.views = {};
      let nFaces = 0;
      for (const v of m.views) { this.views[v.id] = v; nFaces += v.faces.length / 3; }
      // Tableaux globaux : sommets (index global) et ombrage de chaque face
      this.fa = new Uint16Array(nFaces);
      this.fb = new Uint16Array(nFaces);
      this.fc = new Uint16Array(nFaces);
      this.fshade = new Float32Array(nFaces);
      let f = 0;
      for (const v of m.views) {
        v.faceBase = f;
        const off = v.offset;
        for (let i = 0; i < v.faces.length; i += 3, f++) {
          this.fa[f] = v.faces[i] + off;
          this.fb[f] = v.faces[i + 1] + off;
          this.fc[f] = v.faces[i + 2] + off;
          this.fshade[f] = v.shade[i / 3] / 255;
        }
      }
      this.faceColor = new Uint32Array(nFaces);
      this.maxW = Math.max(...m.views.map((v) => v.w));
      this.maxH = Math.max(...m.views.map((v) => v.h));
    }

    showEmpty(msg) {
      this.data = null;
      this.canvas.hidden = true;
      this.emptyEl.hidden = false;
      this.emptyEl.textContent = msg;
      $('#brain-legend').hidden = true;
      $('#brain-units').textContent = '';
      $('#brain-time').textContent = '';
    }

    async setData(meta, u8) {
      await this.loadMesh();
      if (!this.mesh) { this.showEmpty(`Carte cérébrale indisponible : ${this.meshError}`); return; }
      const nv = meta.n_vertices || 20484;
      if (u8.length < nv) { this.showEmpty('Carte cérébrale vide.'); return; }
      this.meta = meta;
      this.data = u8;
      this.nv = nv;
      this.T = Math.floor(u8.length / nv);
      this.canvas.hidden = false;
      this.emptyEl.hidden = true;
      $('#brain-legend').hidden = false;
      $('#brain-vmin').textContent = isNum(meta.vmin) ? nf2.format(meta.vmin) : '';
      $('#brain-vmax').textContent = isNum(meta.vmax) ? `+${nf2.format(meta.vmax)}` : '';
      $('#brain-units').textContent = `${meta.units || ''} · bleu = sous la moyenne, rouge = au-dessus · ${meta.space || 'fsaverage5'}`;
      this.layout();
      this.draw(Math.max(0, state.sec), true);
    }

    // Rasterisation (une fois par taille) : pour chaque pixel, la face visible la plus proche.
    layout() {
      if (!this.mesh || this.canvas.hidden) return;
      const cssW = Math.max(200, this.canvas.parentElement.clientWidth);
      const dpr = clamp(window.devicePixelRatio || 1, 1, 2);
      const gap = 10;
      const labelH = 16;
      const cellW = (cssW - gap) / 2;
      const s = cellW / this.maxW;
      const cellH = this.maxH * s + labelH;
      const cssH = Math.ceil(cellH * 2 + gap);
      this.canvas.style.height = `${cssH}px`;
      const W = Math.round(cssW * dpr);
      const H = Math.round(cssH * dpr);
      this.canvas.width = W;
      this.canvas.height = H;
      this.dpr = dpr;
      this.cells = [];
      const faceOf = new Int32Array(W * H).fill(-1);
      const layout = this.mesh.layout || [['left_lateral', 'right_lateral'], ['left_medial', 'right_medial']];
      layout.forEach((row, ri) => row.forEach((vid, ci) => {
        const v = this.views[vid];
        if (!v) return;
        const ox = (ci * (cellW + gap) + (cellW - v.w * s) / 2) * dpr;
        const oy = (ri * (cellH + gap)) * dpr;
        const k = s * dpr;
        this.cells.push({ label: v.label, x: (ci * (cellW + gap) + cellW / 2) * dpr, y: oy + (v.h * s + labelH - 3) * dpr });
        const xy = v.xy;
        const faces = v.faces;
        for (let i = 0, f = v.faceBase; i < faces.length; i += 3, f++) {
          const a = faces[i] * 2, b = faces[i + 1] * 2, c = faces[i + 2] * 2;
          rasterTri(ox + xy[a] * k, oy + xy[a + 1] * k, ox + xy[b] * k, oy + xy[b + 1] * k,
            ox + xy[c] * k, oy + xy[c + 1] * k, f, faceOf, W, H);
        }
      }));
      // Bouche-trous : pixel vide entouré de pixels pleins (faces sous-pixel)
      for (let y = 1; y < H - 1; y++) {
        for (let x = 1; x < W - 1; x++) {
          const p = y * W + x;
          if (faceOf[p] !== -1) continue;
          const l = faceOf[p - 1], r = faceOf[p + 1], u = faceOf[p - W], d = faceOf[p + W];
          if ((l !== -1) + (r !== -1) + (u !== -1) + (d !== -1) >= 3) faceOf[p] = l !== -1 ? l : (r !== -1 ? r : u);
        }
      }
      let n = 0;
      for (let p = 0; p < faceOf.length; p++) if (faceOf[p] !== -1) n++;
      const pix = new Uint32Array(n);
      const pf = new Uint32Array(n);
      for (let p = 0, k = 0; p < faceOf.length; p++) {
        if (faceOf[p] !== -1) { pix[k] = p; pf[k] = faceOf[p]; k++; }
      }
      this.raster = { W, H, pix, pf };
      this.img = this.ctx.createImageData(W, H);
      this.buf = new Uint32Array(this.img.data.buffer);
      this.t = -1;
    }

    draw(t, force = false) {
      if (!this.data || !this.raster) return;
      t = clamp(Math.floor(t), 0, this.T - 1);
      if (!force && t === this.t) return;
      this.t = t;
      const nv = this.nv;
      const row = this.data.subarray(t * nv, (t + 1) * nv);
      const { fa, fb, fc, fshade, faceColor } = this;
      for (let f = 0; f < faceColor.length; f++) {
        const ci = ((row[fa[f]] + row[fb[f]] + row[fc[f]]) / 3) | 0;
        const sh = fshade[f];
        const r = (LUT[ci * 3] * sh) | 0;
        const g = (LUT[ci * 3 + 1] * sh) | 0;
        const b = (LUT[ci * 3 + 2] * sh) | 0;
        faceColor[f] = (255 << 24) | (b << 16) | (g << 8) | r; // RGBA petit-boutiste
      }
      const { pix, pf } = this.raster;
      const buf = this.buf;
      buf.fill(0);
      for (let k = 0; k < pix.length; k++) buf[pix[k]] = faceColor[pf[k]];
      this.ctx.putImageData(this.img, 0, 0);
      const ctx = this.ctx;
      ctx.save();
      ctx.fillStyle = '#a3acbb';
      ctx.font = `${12 * this.dpr}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
      ctx.textAlign = 'center';
      for (const c of this.cells) ctx.fillText(c.label, c.x, c.y);
      ctx.restore();
      $('#brain-time').textContent = `${fmtTime(t)} → ${fmtTime(t + 1)}`;
    }
  }

  function rasterTri(x0, y0, x1, y1, x2, y2, fid, out, W, H) {
    const area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (area === 0) return;
    const minX = Math.max(0, Math.floor(Math.min(x0, x1, x2)));
    const maxX = Math.min(W - 1, Math.ceil(Math.max(x0, x1, x2)));
    const minY = Math.max(0, Math.floor(Math.min(y0, y1, y2)));
    const maxY = Math.min(H - 1, Math.ceil(Math.max(y0, y1, y2)));
    const inv = 1 / area;
    const eps = -1e-6;
    for (let y = minY; y <= maxY; y++) {
      const py = y + 0.5;
      for (let x = minX; x <= maxX; x++) {
        const px = x + 0.5;
        const w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * inv;
        if (w0 < eps) continue;
        const w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * inv;
        if (w1 < eps || 1 - w0 - w1 < eps) continue;
        out[y * W + x] = fid;
      }
    }
  }

  const brain = new BrainMap($('#brain-canvas'), $('#brain-empty'));

  // ====================================================================== résultat
  const video = $('#video');

  async function loadResult(jobId) {
    showUploadError('');
    let summary, timeline;
    try {
      [summary, timeline] = await Promise.all([
        api(`/api/results/${encodeURIComponent(jobId)}/summary`),
        api(`/api/results/${encodeURIComponent(jobId)}/timeline`),
      ]);
    } catch (e) {
      showUploadError(`Résultat illisible (${jobId}) : ${e.message}`);
      return;
    }
    state.jobId = jobId;
    state.summary = summary;
    state.timeline = Array.isArray(timeline) ? timeline : [];
    state.duration = (summary.video && isNum(summary.video.duration_s) && summary.video.duration_s > 0)
      ? summary.video.duration_s : Math.max(1, state.timeline.length);
    state.sec = -1;
    try { history.replaceState(null, '', `#job=${encodeURIComponent(jobId)}`); } catch (_) { /* ignore */ }

    $('#empty-state').hidden = true;
    $('#result').hidden = false;
    document.body.classList.add('has-result');
    window.scrollTo({ top: 0 });
    renderHeader();
    renderBanners();
    renderScore();
    renderRecommendations();
    renderTranscript();
    video.src = `/media/${encodeURIComponent(jobId)}.mp4`;
    video.load();
    buildCharts();
    loadBrain();
    Coach.load(jobId);
    onTime(true);
    for (const b of document.querySelectorAll('#history button')) {
      b.setAttribute('aria-current', b.title.endsWith(jobId) ? 'true' : 'false');
    }
  }

  function renderHeader() {
    const s = state.summary;
    const app = s.app || {};
    const v = s.video || {};
    $('#r-title').textContent = app.filename || v.filename || state.jobId;
    const bits = [
      fmtDuration(state.duration),
      v.width && v.height ? `${v.width}×${v.height}` : null,
      isNum(v.fps) ? `${nf1.format(v.fps)} i/s` : null,
      v.has_audio === false ? 'sans audio' : null,
      `analysé le ${fmtDate(s.created_at)}`,
      `réf. ${state.jobId}`,
    ].filter(Boolean);
    $('#r-sub').textContent = bits.join(' · ');
  }

  function banner(kind, title, text) {
    return el('div', { class: `banner ${kind}`, role: kind === 'warn' ? 'status' : null }, el('strong', {}, title), text);
  }

  function renderBanners() {
    const s = state.summary;
    const b = s.brain || {};
    const out = [];
    if (s.fake) out.push(banner('warn', 'Résultat factice.', 'Mode développement (REEL_PREDICTOR_FAKE=1) : valeurs synthétiques, sans rapport avec la vidéo.'));
    if (b.degraded || !b.available) {
      out.push(banner('warn', 'Mode dégradé.',
        `Le modèle cérébral TRIBE v2 n'a pas été utilisé${b.reason ? ` : ${b.reason}` : ''}. ` +
        'Les indices reposent sur des signaux vidéo (heuristique) et sont moins fiables.'));
    }
    if (b.available) {
      out.push(banner('info', '5 premières secondes : contexte cérébral limité.',
        "TRIBE s'appuie sur les secondes précédentes : le début de la vidéo est estimé avec moins de contexte (zone hachurée)."));
    }
    const conf = b.confidence || {};
    const low = INDEX_KEYS.filter((k) => conf[k] === 'low').map((k) => INDEX_LABELS[k]);
    if (low.length) {
      out.push(banner('info', 'Confiance par indice.',
        `Lien cerveau → indice spéculatif (confiance faible) pour : ${low.join(', ')}. Voir les pastilles de la légende.`));
    }
    const warns = Array.isArray(s.warnings) ? s.warnings.filter(Boolean) : [];
    if (warns.length) {
      out.push(el('details', { class: 'banner info' },
        el('summary', {}, el('strong', {}, `Avertissements de l'analyse (${warns.length})`), ' — cliquer pour afficher'),
        el('ul', { class: 'small' }, ...warns.map((w) => el('li', {}, String(w))))));
    }
    $('#banners').replaceChildren(...out);
  }

  function scoreClass(v) { return v >= 65 ? 'good' : v >= 40 ? 'mid' : 'low'; }

  function renderScore() {
    const s = state.summary;
    const score = isNum(s.score) ? Math.round(s.score) : null;
    const sv = $('#score-value');
    sv.className = `score-value ${score === null ? '' : scoreClass(score)}`;
    sv.tabIndex = score === null ? -1 : 0;
    sv.setAttribute('role', 'button');
    sv.title = 'Rejouer la révélation du score';
    cancelAnimationFrame(sv._raf);
    if (score === null || Reveal.reduce) sv.textContent = score === null ? '–' : String(score);
    else {
      const t0 = performance.now();
      const step = (now) => {
        const k = Math.min(1, (now - t0) / 1100);
        sv.textContent = String(Math.round(score * (1 - Math.pow(1 - k, 4))));
        if (k < 1) sv._raf = requestAnimationFrame(step);
      };
      sv._raf = requestAnimationFrame(step);
    }
    sv.setAttribute('aria-label', score === null ? 'Score indisponible' : `Score prédit ${score} sur 100`);

    const bd = s.score_breakdown || {};
    const means = bd.index_means || {};
    const conf = (s.brain && s.brain.confidence) || {};
    $('#index-means').replaceChildren(...INDEX_KEYS.map((k) => {
      const v = means[k];
      return el('div', { class: 'im-row' },
        el('span', {}, INDEX_LABELS[k], conf[k] ? el('span', { class: `conf ${conf[k]}`, title: CONF_LABELS[conf[k]] || '' }, conf[k] === 'low' ? 'faible' : conf[k] === 'medium' ? 'moyenne' : conf[k]) : null),
        el('span', { class: 'muted' }, isNum(v) ? nf2.format(v) : '–'),
        el('div', { class: 'im-bar' }, el('span', { style: { width: `${isNum(v) ? clamp(v, 0, 1) * 100 : 0}%`, background: INDEX_COLORS[k] } })));
    }));

    const r = s.retention || {};
    const rows = [
      ['Bonus accroche', isNum(bd.hook_bonus) ? `${bd.hook_bonus >= 0 ? '+' : ''}${nf1.format(bd.hook_bonus)}` : '–'],
      ['Pénalité creux', isNum(bd.dip_penalty) ? nf1.format(bd.dip_penalty || 0) : '–'],
      ['Visionnage moyen prédit', isNum(r.avg_watch_s) ? `${nf1.format(r.avg_watch_s)} s (${fmtPct(r.avg_watch_ratio)})` : fmtPct(r.avg_watch_ratio)],
      ['Vont jusqu\'au bout', fmtPct(r.completion)],
      ['Modèle de rétention', r.calibrated === true ? 'recalibré' : r.calibrated === false ? 'par défaut' : '–'],
      ['Cuts', String((s.cuts || []).length)],
      ['Recommandations', String((s.recommendations || []).length)],
    ];
    $('#score-detail').replaceChildren(...rows.flatMap(([k, v]) => [el('dt', {}, k), el('dd', {}, v)]));
  }

  function sortedRecs() {
    return [...(state.summary.recommendations || [])].sort((a, b) =>
      (a.t - b.t) || ((SEV_ORDER[a.severity] ?? 3) - (SEV_ORDER[b.severity] ?? 3)));
  }

  function renderRecommendations() {
    const recs = sortedRecs();
    const list = $('#recs');
    if (!recs.length) {
      list.replaceChildren(el('li', { class: 'muted small' }, 'Aucune recommandation : rien de bloquant détecté.'));
      return;
    }
    list.replaceChildren(...recs.map((r) => {
      const sev = SEV_LABELS[r.severity] ? r.severity : 'low';
      const range = isNum(r.t_end) && r.t_end > r.t ? `${fmtTime(r.t)}–${fmtTime(r.t_end)}` : fmtTime(r.t);
      return el('li', {},
        el('button', {
          type: 'button', class: sev, dataset: { t: String(r.t), tEnd: String(isNum(r.t_end) ? r.t_end : r.t + 1) },
          title: `Aller à ${fmtTime(r.t)}`, 'aria-label': `${range}, sévérité ${SEV_LABELS[sev]} : ${r.message}`,
          onclick: () => seek(r.t),
        },
        el('span', { class: 'rec-time' }, range),
        el('span', { class: `sev ${sev}` }, SEV_LABELS[sev]),
        el('span', {}, r.message || r.rule || '')));
    }));
  }

  function renderTranscript() {
    const box = $('#transcript');
    const tr = state.summary.transcript || {};
    const groups = state.timeline.filter((row) => Array.isArray(row.words) && row.words.length);
    const lang = tr.language ? el('p', { class: 'lang' }, `Langue détectée : ${tr.language}`) : null;
    if (!groups.length) {
      box.replaceChildren(lang || '', el('p', { class: tr.text ? '' : 'muted small' }, tr.text || 'Aucune parole détectée.'));
      return;
    }
    box.replaceChildren(lang || '', el('p', {}, ...groups.map((row) => el('button', {
      type: 'button', dataset: { t: String(row.t) }, title: `Aller à ${fmtTime(row.t)}`, onclick: () => seek(row.t),
    }, el('span', { class: 'ts' }, fmtTime(row.t)), ' ', row.words.join(' ')))));
  }

  // ---------------------------------------------------------------- cerveau 3D
  const view3d = { on: true, ok: typeof window.Brain3D === 'function', inst: null, hl: null };
  let heroBrain = null;
  if (view3d.ok) {
    try {
      heroBrain = new window.Brain3D($('#hero-brain'), { zoom: false, hover: false, dist: 3.7 });
      heroBrain.ready.catch(() => { $('#hero-brain').hidden = true; });
    } catch (_) { heroBrain = null; }
  }

  function brain3d() {
    if (!view3d.ok) return null;
    if (!view3d.inst) {
      try {
        view3d.inst = new window.Brain3D($('#brain3d'), { dist: 3.8 });
        view3d.inst.ready.catch(() => { view3d.ok = false; applyBrainView(); });
        view3d.inst.on('hover', showBrainTip);
        view3d.inst.on('select', (info) => {
          const r = info && info.roi;
          const k = r ? Object.keys(r.indices).sort((a, b) => Math.abs(r.indices[b]) - Math.abs(r.indices[a]))[0] : null;
          setBrainHighlight(k && k !== view3d.hl ? k : null);
        });
      } catch (_) { view3d.ok = false; }
    }
    return view3d.inst;
  }

  function showBrainTip(info) {
    const tip = $('#brain-tip');
    if (!info) { tip.hidden = true; return; }
    const r = info.roi;
    const z = isNum(info.value) && brain.meta ? brainValue(info.value) : null;
    const idx = r ? Object.entries(r.indices).filter(([, w]) => w) : [];
    tip.replaceChildren(
      el('div', { class: 'tip-name' }, r ? r.label : 'Cortex (hors régions suivies)'),
      el('div', { class: 'muted' }, `Hémisphère ${info.hemi}`),
      z !== null ? el('div', { class: 'tip-val grad' }, `${z >= 0 ? '+' : ''}${nf2.format(z)}`) : null,
      z !== null ? el('div', { class: 'muted small' }, z >= 0 ? 'au-dessus de la moyenne' : 'sous la moyenne') : null,
      idx.length ? el('div', { class: 'tip-idx' }, ...idx.map(([k, w]) => el('span', { style: { color: INDEX_COLORS[k] } },
        `${w < 0 ? '−' : ''}${INDEX_LABELS[k]}`))) : null);
    const wrap = $('#brain3d-wrap');
    const x = Math.min(info.x, wrap.clientWidth - 200);
    tip.style.left = `${Math.max(0, x)}px`;
    tip.style.top = `${Math.min(info.y, wrap.clientHeight - 120)}px`;
    tip.hidden = false;
  }

  function brainValue(u8) {
    const m = brain.meta || {};
    const lo = isNum(m.vmin) ? m.vmin : -1, hi = isNum(m.vmax) ? m.vmax : 1;
    return lo + (u8 / 255) * (hi - lo);
  }

  function hexRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255]; }

  function setBrainHighlight(k) {
    view3d.hl = k;
    const inst = brain3d();
    if (inst) inst.ready.then(() => inst.highlightIndex(k, k ? hexRgb(INDEX_COLORS[k]) : null)).catch(() => {});
    for (const b of document.querySelectorAll('#brain-hl button')) b.setAttribute('aria-pressed', String(b.dataset.key === k));
  }

  function renderBrainHl() {
    const box = $('#brain-hl');
    if (!view3d.ok || !view3d.on) { box.replaceChildren(); return; }
    box.replaceChildren(el('span', { class: 'lbl' }, 'Zones qui alimentent'),
      ...INDEX_KEYS.map((k) => el('button', {
        type: 'button', dataset: { key: k }, 'aria-pressed': String(view3d.hl === k), style: { color: INDEX_COLORS[k] },
        onclick: () => setBrainHighlight(view3d.hl === k ? null : k),
      }, el('span', { class: 'sw', style: { background: INDEX_COLORS[k] } }), el('span', { style: { color: 'var(--text)' } }, INDEX_LABELS[k]))));
  }

  function applyBrainView() {
    const has = !!brain.data;
    const use3d = view3d.ok && view3d.on;
    $('#brain3d-wrap').hidden = !(has && use3d);
    $('#brain-canvas').hidden = !(has && !use3d);
    $('#brain-views').hidden = !use3d;
    $('#bv-3d').setAttribute('aria-pressed', String(use3d));
    $('#bv-flat').setAttribute('aria-pressed', String(!use3d));
    $('#bv-3d').disabled = !view3d.ok;
    $('#brain-legend').classList.toggle('g3d', use3d);
    if (has && brain.meta) {
      $('#brain-units').textContent = `${brain.meta.units || ''} · ${use3d ? 'sombre / bleu = sous la moyenne, lavande → braise = au-dessus · glisser pour tourner, survoler une zone' : 'bleu = sous la moyenne, rouge = au-dessus'} · ${brain.meta.space || 'fsaverage5'}`;
    }
    if (has && !use3d) { brain.layout(); brain.draw(Math.max(0, state.sec), true); }
    renderBrainHl();
  }
  $('#bv-3d').addEventListener('click', () => { view3d.on = true; applyBrainView(); });
  $('#bv-flat').addEventListener('click', () => { view3d.on = false; applyBrainView(); });
  for (const b of document.querySelectorAll('#brain-views button')) {
    b.addEventListener('click', () => { const i = brain3d(); if (i) i.view(b.dataset.view); });
  }

  async function loadBrain() {
    const b = state.summary.brain || {};
    const jobId = state.jobId;
    $('#brain3d-wrap').hidden = true;
    $('#brain-tip').hidden = true;
    if (!b.available) {
      brain.showEmpty(`Carte cérébrale indisponible${b.reason ? ` : ${b.reason}` : ' (mode dégradé)'}.`);
      renderBrainHl();
      return;
    }
    brain.showEmpty('Chargement de la carte cérébrale…');
    try {
      const [meta, buf] = await Promise.all([
        api(`/api/results/${encodeURIComponent(jobId)}/brain_meta`),
        api(`/api/results/${encodeURIComponent(jobId)}/brain`),
      ]);
      if (jobId !== state.jobId) return;
      const u8 = new Uint8Array(buf);
      await brain.setData(meta, u8);
      const inst = brain3d();
      if (inst) {
        try {
          await inst.ready;
          inst.setData(u8, meta.n_vertices || 20484);
          inst.setTime(video.currentTime || 0);
          if (view3d.hl) inst.highlightIndex(view3d.hl, hexRgb(INDEX_COLORS[view3d.hl]));
        } catch (_) { view3d.ok = false; }
      }
      applyBrainView();
    } catch (e) {
      if (jobId === state.jobId) brain.showEmpty(`Carte cérébrale illisible : ${e.message}`);
    }
  }

  // ====================================================================== graphiques
  function chartWidth(node) { return Math.max(260, node.clientWidth || node.parentElement.clientWidth || 600); }

  function overlayPlugin({ zones = true, recs = false, cuts = true } = {}) {
    return {
      hooks: {
        drawClear: [(u) => {
          const { ctx } = u;
          const { top, height } = u.bbox;
          ctx.save();
          if (zones && state.summary && state.summary.brain && state.summary.brain.available) {
            const x0 = u.valToPos(0, 'x', true);
            const x1 = u.valToPos(Math.min(LOW_CONTEXT_S, state.duration), 'x', true);
            ctx.fillStyle = 'rgba(255,255,255,0.045)';
            ctx.fillRect(x0, top, x1 - x0, height);
            ctx.beginPath();
            ctx.rect(x0, top, x1 - x0, height);
            ctx.clip();
            ctx.strokeStyle = 'rgba(255,255,255,0.09)';
            ctx.lineWidth = 1;
            const step = 8 * devicePixelRatio;
            for (let x = x0 - height; x < x1; x += step) {
              ctx.beginPath();
              ctx.moveTo(x, top + height);
              ctx.lineTo(x + height, top);
              ctx.stroke();
            }
          }
          ctx.restore();
          if (recs) {
            ctx.save();
            for (const r of sortedRecs()) {
              const a = u.valToPos(r.t, 'x', true);
              const b = u.valToPos(isNum(r.t_end) && r.t_end > r.t ? r.t_end : r.t + 1, 'x', true);
              ctx.fillStyle = SEV_FILL[r.severity] || SEV_FILL.low;
              ctx.fillRect(a, top, Math.max(2, b - a), height);
            }
            ctx.restore();
          }
        }],
        draw: [(u) => {
          if (!cuts || !state.summary) return;
          const { ctx } = u;
          const { top, height } = u.bbox;
          ctx.save();
          ctx.strokeStyle = 'rgba(255,255,255,0.45)';
          ctx.lineWidth = 1.2 * devicePixelRatio;
          ctx.setLineDash([4 * devicePixelRatio, 4 * devicePixelRatio]);
          for (const c of state.summary.cuts || []) {
            const x = Math.round(u.valToPos(c, 'x', true));
            ctx.beginPath();
            ctx.moveTo(x, top);
            ctx.lineTo(x, top + height);
            ctx.stroke();
          }
          ctx.restore();
          // Repères du plan de montage proposé par le coach (bande + triangle en haut du graphique)
          if (state.coachPlan && state.coachPlan.length) {
            ctx.save();
            const dpr = devicePixelRatio;
            for (const a of state.coachPlan) {
              const x0 = u.valToPos(clamp(a.t, 0, state.duration), 'x', true);
              const x1 = u.valToPos(clamp(Math.max(a.t_end, a.t + 0.5), 0, state.duration), 'x', true);
              ctx.fillStyle = 'rgba(199,125,255,0.30)';
              ctx.fillRect(x0, top, Math.max(2 * dpr, x1 - x0), 5 * dpr);
              ctx.fillStyle = COACH_COLOR;
              ctx.beginPath();
              ctx.moveTo(x0 - 5 * dpr, top);
              ctx.lineTo(x0 + 5 * dpr, top);
              ctx.lineTo(x0, top + 8 * dpr);
              ctx.closePath();
              ctx.fill();
            }
            ctx.restore();
          }
        }],
      },
    };
  }

  function addPlayhead(u) {
    const ph = el('div', { class: 'playhead', 'aria-hidden': 'true' });
    u.over.appendChild(ph);
    u._playhead = ph;
    u.over.addEventListener('click', (e) => {
      const rect = u.over.getBoundingClientRect();
      seek(u.posToVal(e.clientX - rect.left, 'x'));
    });
  }

  function baseOpts(node, height, yRange, yFmt) {
    return {
      width: chartWidth(node),
      height,
      legend: { show: false },
      cursor: { drag: { x: false, y: false }, sync: { key: 'reel-sync' }, points: { size: 7 } },
      scales: { x: { time: false, range: () => [0, state.duration] }, y: { range: yRange } },
      axes: [
        { ...AXIS_STYLE, values: (u, vals) => vals.map((v) => fmtTime(v)) },
        { ...AXIS_STYLE, size: 44, values: (u, vals) => vals.map(yFmt) },
      ],
    };
  }

  function buildCharts() {
    for (const c of state.charts) c.destroy();
    state.charts = [];
    const nodeI = $('#chart-indices');
    const nodeR = $('#chart-retention');
    nodeI.replaceChildren();
    nodeR.replaceChildren();
    renderLegend();
    if (!hasUPlot) {
      nodeI.append(el('p', { class: 'banner error' }, 'Bibliothèque de graphiques uPlot absente : lancez « make setup ».'));
      return;
    }
    const tl = state.timeline;
    const xs = tl.map((r) => Math.min(r.t + 0.5, state.duration));
    const dataI = [xs, ...INDEX_KEYS.map((k) => tl.map((r) => (r.indices && isNum(r.indices[k]) ? r.indices[k] : null)))];
    const optsI = baseOpts(nodeI, 230, [0, 1], (v) => nf1.format(v));
    optsI.series = [{}, ...INDEX_KEYS.map((k) => ({ label: INDEX_LABELS[k], stroke: INDEX_COLORS[k], width: 2, points: { show: false } }))];
    optsI.plugins = [overlayPlugin({ zones: true, recs: false, cuts: true })];
    const uI = new uPlot(optsI, dataI, nodeI);
    addPlayhead(uI);
    state.charts.push(uI);

    const xr = [0, ...tl.map((r) => Math.min(r.t + 1, state.duration))];
    const yr = [1, ...tl.map((r) => (isNum(r.retention) ? r.retention : null))];
    if (yr.slice(1).every((v) => v === null)) {
      const curve = (state.summary.retention && state.summary.retention.curve) || [];
      xr.length = 0; yr.length = 0;
      curve.forEach((v, i) => { xr.push(Math.min(i, state.duration)); yr.push(isNum(v) ? v : null); });
    }
    const optsR = baseOpts(nodeR, 170, [0, 1], (v) => `${Math.round(v * 100)} %`);
    optsR.series = [{}, { label: 'Rétention prédite', stroke: RETENTION_COLOR, width: 2, fill: 'rgba(233,236,242,0.08)', points: { show: false } }];
    optsR.plugins = [overlayPlugin({ zones: false, recs: true, cuts: true })];
    const uR = new uPlot(optsR, [xr, yr], nodeR);
    addPlayhead(uR);
    state.charts.push(uR);
    const r = state.summary.retention || {};
    $('#retention-note').textContent = isNum(r.avg_watch_ratio)
      ? `visionnage moyen prédit : ${fmtPct(r.avg_watch_ratio)} de la vidéo` : '';
  }

  function renderLegend() {
    const conf = (state.summary.brain && state.summary.brain.confidence) || {};
    $('#index-legend').replaceChildren(...INDEX_KEYS.map((k, i) => el('button', {
      type: 'button', 'aria-pressed': 'true', dataset: { key: k },
      title: `${INDEX_LABELS[k]} — ${CONF_LABELS[conf[k]] || 'confiance non précisée'} (cliquer pour masquer/afficher)`,
      onclick: (e) => {
        const btn = e.currentTarget;
        const on = btn.getAttribute('aria-pressed') !== 'true';
        btn.setAttribute('aria-pressed', String(on));
        if (state.charts[0]) state.charts[0].setSeries(i + 1, { show: on });
      },
    },
    el('span', { class: 'sw', style: { background: INDEX_COLORS[k] } }),
    INDEX_LABELS[k],
    conf[k] ? el('span', { class: `conf ${conf[k]}` }, conf[k] === 'low' ? 'faible' : conf[k] === 'medium' ? 'moyenne' : conf[k]) : null,
    el('span', { class: 'val' }, '–'))));
  }

  // ====================================================================== synchronisation vidéo
  function seek(t) {
    if (!isNum(t)) return;
    const max = isNum(video.duration) && video.duration > 0 ? video.duration - 0.05 : state.duration;
    video.currentTime = clamp(t, 0, Math.max(0, max));
    onTime(true);
  }

  function setPlayheads(t) {
    for (const u of state.charts) {
      if (!u._playhead) continue;
      const x = u.valToPos(clamp(t, 0, state.duration), 'x');
      u._playhead.style.transform = `translateX(${x}px)`;
    }
  }

  function onTime(force = false) {
    if (!state.summary) return;
    const t = video.currentTime || 0;
    setPlayheads(t);
    if (view3d.inst && brain.data) view3d.inst.setTime(t);
    const sec = clamp(Math.floor(t), 0, Math.max(0, state.timeline.length - 1));
    if (force === true || sec !== state.sec) {
      state.sec = sec;
      brain.draw(sec);
      updateNow(sec, t);
    }
  }

  function updateNow(sec, t) {
    const row = state.timeline[sec] || {};
    const ind = row.indices || {};
    const parts = [el('span', { class: 't' }, `${fmtTime(t)}`)];
    if (isNum(row.retention)) parts.push(el('span', {}, `rétention ${fmtPct(row.retention)}`));
    if (row.cut) parts.push(el('span', {}, '✂ cut'));
    if (row.text) parts.push(el('span', { class: 'muted' }, `texte : « ${row.text} »`));
    $('#now-readout').replaceChildren(...parts);
    for (const btn of document.querySelectorAll('#index-legend button')) {
      const v = ind[btn.dataset.key];
      btn.querySelector('.val').textContent = isNum(v) ? nf2.format(v) : '–';
    }
    for (const b of document.querySelectorAll('#recs button')) {
      const a = Number(b.dataset.t);
      const e = Number(b.dataset.tEnd);
      b.classList.toggle('active', t >= a && t < Math.max(e, a + 1));
    }
    for (const b of document.querySelectorAll('#transcript button')) {
      b.classList.toggle('active', Number(b.dataset.t) === sec);
    }
    for (const b of document.querySelectorAll('#coach-plan button')) {
      const a = Number(b.dataset.t);
      const e = Number(b.dataset.tEnd);
      b.classList.toggle('active', t >= a && t < Math.max(e, a + 0.5));
    }
  }

  video.addEventListener('timeupdate', () => onTime());
  video.addEventListener('seeked', () => onTime(true));
  video.addEventListener('loadedmetadata', () => onTime(true));
  video.addEventListener('play', () => {
    cancelAnimationFrame(state.raf);
    const loop = () => {
      onTime();
      if (!video.paused && !video.ended) state.raf = requestAnimationFrame(loop);
    };
    state.raf = requestAnimationFrame(loop);
  });
  video.addEventListener('pause', () => cancelAnimationFrame(state.raf));

  function resizeAll() {
    for (const u of state.charts) {
      const node = u.root.parentElement;
      u.setSize({ width: chartWidth(node), height: u.height });
    }
    for (const u of state.cmpCharts) {
      const node = u.root.parentElement;
      u.setSize({ width: chartWidth(node), height: u.height });
    }
    if (brain.data && !$('#brain-canvas').hidden) { brain.layout(); brain.draw(Math.max(0, state.sec), true); }
    if (state.summary) setPlayheads(video.currentTime || 0);
  }
  window.addEventListener('resize', debounce(resizeAll, 150));

  // ====================================================================== coach script & montage
  const ROLE_LABELS = { hook: 'Accroche', fact: 'Chiffre clé', body: 'Développement', cta: "Appel à l'action" };
  const SHOT_LABELS = { face_cam: 'Face caméra', close_up: 'Plan serré', screen: 'Écran', broll: 'B-roll', text_card: 'Carton texte' };
  const ACTION_LABELS = { cut: 'Cut', trim: 'Raccourcir', zoom: 'Zoom', text: 'Texte', broll: 'B-roll', sfx: 'Effet sonore', music: 'Musique', reorder: 'Réordonner' };
  const MOMENT_LABELS = { hook: 'Accroche', fact: 'Chiffre clé', cta: "Appel à l'action" };
  const SIM_MODES = { tribe_no_video: 'modèle cérébral TRIBE (texte + audio, sans image)', heuristic: 'heuristique (sans modèle cérébral)' };
  const COACH_COLOR = '#c77dff';
  const fmtUsd = (v) => (isNum(v) ? `${nf2.format(v)} $` : '–');

  const Coach = {
    jobId: null,
    info: null,
    current: null,
    pollToken: 0,
    busy: false,

    reset() {
      this.current = null;
      this.info = null;
      this.pollToken++;
      state.coachPlan = [];
      $('#coach-result').hidden = true;
      $('#coach-result').replaceChildren();
      $('#coach-progress').hidden = true;
      $('#coach-error').hidden = true;
      $('#coach-history-wrap').hidden = true;
      $('#coach-cost').textContent = '';
    },

    async load(jobId) {
      this.reset();
      this.jobId = jobId;
      await this.refresh(true);
    },

    async refresh(openLatest = false) {
      const jobId = this.jobId;
      let info;
      try {
        info = await api(`/api/results/${encodeURIComponent(jobId)}/coach`);
      } catch (e) {
        if (jobId === this.jobId) this.showError(`Coach indisponible : ${e.message}`);
        return;
      }
      if (jobId !== this.jobId) return;
      this.info = info;
      this.renderControls();
      this.renderHistory();
      const active = (info.pending || []).find((p) => p.status === 'running' || p.status === 'queued');
      if (active) { this.poll(active.coach_id); return; }
      if (openLatest && info.items && info.items.length) this.open(info.items[0].coach_id);
    },

    estimateUsd() {
      const i = this.info || {};
      if (i.estimate && isNum(i.estimate.usd)) return i.estimate.usd;
      return i.config && isNum(i.config.estimated_cost_usd) ? i.config.estimated_cost_usd : null;
    },

    renderControls() {
      const i = this.info || {};
      const btn = $('#coach-btn');
      const usd = this.estimateUsd();
      btn.textContent = `Générer des suggestions (API Claude${isNum(usd) ? `, ≈ ${fmtUsd(usd)}` : ''})`;
      btn.disabled = !i.claude_api || this.busy;
      $('#coach-nokey').hidden = !!i.claude_api;
      const cfg = i.config || {};
      const bits = [];
      if (cfg.model) bits.push(`${cfg.model}${cfg.effort ? ` (effort ${cfg.effort})` : ''}`);
      if (isNum(cfg.n_variants)) bits.push(`${cfg.n_variants} variantes`);
      if (isNum(cfg.max_cost_usd)) bits.push(`plafond ${fmtUsd(cfg.max_cost_usd)}`);
      if (i.estimate && i.estimate.method === 'approx') bits.push('estimation approximative (tokens comptés à l\'envoi)');
      $('#coach-cost').textContent = bits.join(' · ');
    },

    renderHistory() {
      const items = (this.info && this.info.items) || [];
      $('#coach-history-wrap').hidden = !items.length;
      $('#coach-history').replaceChildren(...items.map((it) => {
        const best = it.best;
        const att = best && isNum(best.attainment) ? `moments clés ${fmtPct(best.attainment)}` : 'sans simulation';
        return el('li', {}, el('button', {
          type: 'button', dataset: { id: it.coach_id },
          'aria-current': this.current && this.current.coach_id === it.coach_id ? 'true' : 'false',
          title: `Ouvrir la génération du ${fmtDate(it.created_at)}`,
          onclick: () => this.open(it.coach_id),
        },
        el('span', { class: 'h-name' }, `${fmtDate(it.created_at)} · ${it.n_variants} variante(s)`),
        el('span', { class: 'h-score' }, fmtUsd(it.cost_usd)),
        el('span', { class: 'h-date' }, `${it.model || ''} · ${att}${it.n_warnings ? ` · ${it.n_warnings} avertissement(s)` : ''}`)));
      }));
    },

    showError(msg) {
      const b = $('#coach-error');
      b.textContent = msg || '';
      b.hidden = !msg;
    },

    async start() {
      const usd = this.estimateUsd();
      const cfg = (this.info && this.info.config) || {};
      const dlg = $('#coach-dialog');
      $('#coach-dialog-cost').textContent = `Coût estimé : ≈ ${fmtUsd(usd)} (modèle ${cfg.model || '?'}` +
        `${isNum(cfg.max_cost_usd) ? `, plafond ${fmtUsd(cfg.max_cost_usd)}` : ''}).`;
      $('#coach-confirm').textContent = `Lancer (≈ ${fmtUsd(usd)})`;
      const ok = await new Promise((resolve) => {
        if (typeof dlg.showModal !== 'function') {
          resolve(window.confirm(`Générer des suggestions avec l'API Claude (≈ ${fmtUsd(usd)}) ? Seul du texte est envoyé, jamais la vidéo.`));
          return;
        }
        dlg.returnValue = '';
        dlg.addEventListener('close', () => resolve(dlg.returnValue === 'ok'), { once: true });
        dlg.showModal();
      });
      if (!ok) return;
      this.showError('');
      try {
        const res = await api(`/api/results/${encodeURIComponent(this.jobId)}/coach`, { method: 'POST' });
        this.poll(res.coach_id);
      } catch (e) {
        this.showError(`Génération refusée : ${e.message}`);
      }
    },

    renderProgress(st) {
      $('#coach-progress').hidden = false;
      const pct = clamp(Number(st.pct) || 0, 0, 1);
      $('#coach-stage').textContent = st.stage_label || (st.status === 'queued' ? 'En attente' : 'Génération');
      $('#coach-pct').textContent = `${Math.round(pct * 100)} %`;
      $('#coach-fill').style.width = `${(pct * 100).toFixed(1)}%`;
      $('#coach-bar').setAttribute('aria-valuenow', String(Math.round(pct * 100)));
      $('#coach-msg').textContent = st.message || '';
      $('#coach-eta').textContent = st.status === 'running' ? fmtEta(st.eta_s) : '';
    },

    async poll(coachId) {
      const token = ++this.pollToken;
      this.busy = true;
      this.renderControls();
      try {
        for (;;) {
          if (token !== this.pollToken) return;
          let st;
          try {
            st = await api(`/api/coach/${encodeURIComponent(coachId)}`);
          } catch (e) {
            this.showError(`Suivi de la génération impossible : ${e.message}`);
            return;
          }
          if (token !== this.pollToken) return;
          if (st.status === 'done') {
            $('#coach-progress').hidden = true;
            if (st.result) this.render(st.result);
            this.busy = false;
            await this.refresh(false);
            return;
          }
          if (st.status === 'error') {
            $('#coach-progress').hidden = true;
            this.showError(st.error || st.message || 'La génération a échoué.');
            return;
          }
          this.renderProgress(st);
          await new Promise((r) => setTimeout(r, 1500));
        }
      } finally {
        if (token === this.pollToken) { this.busy = false; this.renderControls(); }
      }
    },

    async open(coachId) {
      try {
        const st = await api(`/api/coach/${encodeURIComponent(coachId)}`);
        if (st.result) this.render(st.result);
        else if (st.status === 'running' || st.status === 'queued') this.poll(coachId);
      } catch (e) {
        this.showError(`Génération illisible : ${e.message}`);
      }
    },

    render(res) {
      if (!res || res.job_id !== this.jobId) return;
      this.current = res;
      const out = res.output || {};
      const sim = res.simulation || null;
      const results = (sim && sim.results) || [];
      const rk = res.ranking || {};
      const box = $('#coach-result');
      const parts = [];

      // --- méta
      const meta = [
        `Générée le ${fmtDate(res.created_at)}`,
        `modèle ${res.model || '?'}`,
        `coût réel ${fmtUsd(res.cost_usd)}${res.estimate && isNum(res.estimate.usd) ? ` (estimé ${fmtUsd(res.estimate.usd)})` : ''}`,
        sim ? `simulation : ${SIM_MODES[sim.mode] || sim.mode}` : 'sans simulation',
      ];
      parts.push(el('p', { class: 'small muted coach-meta' }, meta.join(' · ')));
      if (sim && sim.mode === 'heuristic') {
        parts.push(banner('warn', 'Simulation dégradée.', `Modèle cérébral indisponible pour la simulation${sim.reason ? ` (${sim.reason})` : ''} : indices estimés par heuristique.`));
      }
      const warns = (res.warnings || []).filter(Boolean);
      if (warns.length) {
        parts.push(el('details', { class: 'banner warn' },
          el('summary', {}, el('strong', {}, `Avertissements (${warns.length})`)),
          el('ul', { class: 'small' }, ...warns.map((w) => el('li', {}, w)))));
      }

      // --- diagnostic
      parts.push(el('h4', {}, 'Diagnostic'));
      parts.push(el('p', {}, out.diagnosis_summary || '–'));
      const ev = (res.brief && res.brief.evaluation) || {};
      const tgr = (res.config && res.config.targets) || {};
      const basis = tgr.basis || {};
      if (basis.mode === 'relative') {
        const pbi = tgr.peak_by_index || {};
        parts.push(el('p', { class: 'small muted' },
          `Cibles relatives à vos ${basis.n_reels} Reels analysés (${basis.n_seconds} s) : un moment clé est atteint quand l'indice égale votre ${Math.round(basis.peak_percentile)}ᵉ percentile — `,
          Object.entries(pbi).map(([k, v]) => `${INDEX_LABELS[k] || k} ${nf2.format(v)}`).join(' · '),
          ` ; plancher ${INDEX_LABELS[tgr.floor_index] || 'attention'} ${nf2.format(tgr.floor)}. Dans la simulation, ces cibles sont converties à son échelle.`));
      } else if (basis.reason) {
        parts.push(el('p', { class: 'small muted' }, `Cibles fixes : ${basis.reason}.`));
      }
      if (ev.moments && ev.moments.length) {
        parts.push(el('div', { class: 'moment-chips' }, ...ev.moments.map((m) => el('span', {
          class: `chip ${m.met ? 'good' : 'warn'}`,
          title: `Vidéo actuelle, ${fmtTime(m.t)}–${fmtTime(m.t_end)}`,
        }, `${MOMENT_LABELS[m.name] || m.name} ${fmtTime(m.t)}–${fmtTime(m.t_end)} : `,
        Object.entries(m.values).map(([k, v]) => `${INDEX_LABELS[k] || k} ${nf2.format(v)}`).join(' · '))),
        el('span', { class: 'chip' }, `${(ev.seconds_below_floor || []).length} s sous le plancher`)));
      }

      // --- plan de montage
      const plan = (out.montage_plan || []).slice().sort((a, b) => a.t - b.t);
      state.coachPlan = plan;
      parts.push(el('h4', {}, 'Plan de montage de la vidéo actuelle ',
        el('span', { class: 'muted small' }, '(sans retourner · clic = aller au moment · repères violets sur les courbes)')));
      if (!plan.length) parts.push(el('p', { class: 'muted small' }, 'Aucune action proposée.'));
      else {
        parts.push(el('ol', { id: 'coach-plan', class: 'recs coach-plan' }, ...plan.map((a) => el('li', {}, el('button', {
          type: 'button', dataset: { t: String(a.t), tEnd: String(a.t_end) },
          title: `Aller à ${fmtTime(a.t)}`,
          'aria-label': `${fmtTime(a.t)} à ${fmtTime(a.t_end)}, ${ACTION_LABELS[a.action] || a.action} : ${a.instruction}`,
          onclick: () => seek(a.t),
        },
        el('span', { class: 'rec-time' }, `${fmtTime(a.t)}–${fmtTime(a.t_end)}`),
        el('span', { class: 'sev act' }, ACTION_LABELS[a.action] || a.action),
        el('span', { class: 'plan-body' },
          el('span', {}, a.instruction),
          a.on_screen_text ? el('span', { class: 'plan-text' }, `Texte : « ${a.on_screen_text} »`) : null,
          el('span', { class: 'muted small' }, a.expected_effect || ''),
          el('span', { class: 'idx-tags' }, ...(a.target_indices || []).map((k) => el('span', { class: 'idx-tag', style: { borderColor: INDEX_COLORS[k], color: INDEX_COLORS[k] } }, INDEX_LABELS[k] || k)))))))));
      }

      // --- variantes
      const nV = (out.variants || []).length;
      const order = (rk.order || []).filter((i) => i > 0).map((i) => i - 1).filter((i) => i < nV);
      for (let i = 0; i < nV; i++) if (!order.includes(i)) order.push(i);
      parts.push(el('h4', {}, `Variantes de script${results.length ? ' (meilleure simulation d\'abord)' : ''}`));
      if (rk.original_rank === 1) {
        parts.push(banner('info', "L'original reste en tête.", "Dans la simulation, aucune variante ne dépasse l'original re-voicé : privilégiez le plan de montage."));
      }
      parts.push(el('div', { class: 'variants' }, ...order.map((vi) => this.variantCard(res, vi, results[vi + 1] || null, results[0] || null, vi === rk.best_variant))));

      if (out.notes && out.notes.length) {
        parts.push(el('details', { class: 'coach-notes' }, el('summary', {}, `Notes et limites (${out.notes.length})`),
          el('ul', { class: 'small' }, ...out.notes.map((n) => el('li', {}, n)))));
      }
      parts.push(el('p', {}, el('a', { class: 'btn', href: `/api/coach/${encodeURIComponent(res.coach_id)}/markdown`, download: '' }, 'Télécharger le rapport complet (.md)')));
      box.replaceChildren(...parts);
      box.hidden = false;
      for (const b of document.querySelectorAll('#coach-history button')) b.setAttribute('aria-current', b.dataset.id === res.coach_id ? 'true' : 'false');
      for (const u of state.charts) u.redraw(false, false);
      onTime(true);
    },

    variantCard(res, vi, r, base, isBest) {
      const v = res.output.variants[vi];
      // cibles à l'échelle de la simulation (converties via l'original re-voicé), sinon cibles de config
      const tg = (r && r.evaluation) || (res.config && (res.config.targets_sim || res.config.targets)) || {};
      const peak = isNum(tg.peak) ? tg.peak : 0.9;
      const floor = isNum(tg.floor) ? tg.floor : 0.5;
      const head = el('div', { class: 'variant-head' },
        el('h5', {}, `Variante ${vi + 1} — ${v.angle}`),
        isBest ? el('span', { class: 'badge best' }, 'recommandée') : null,
        r && isNum(r.score) ? el('span', { class: 'chip', title: 'Score simulé (original re-voicé entre parenthèses)' },
          `score simulé ${Math.round(r.score)}${base && isNum(base.score) ? ` (original ${Math.round(base.score)})` : ''}`) : null,
        r && isNum(r.duration_s) ? el('span', { class: 'chip' }, `${nf1.format(r.duration_s)} s`) : null);
      const body = [head, el('p', { class: 'small' }, el('strong', {}, 'Accroche : '), v.hook_summary)];

      if (r && r.evaluation) {
        const e = r.evaluation;
        const be = (base && base.evaluation) || {};
        const bm = Object.fromEntries((be.moments || []).map((m) => [m.name, m]));
        const rows = [];
        for (const m of e.moments || []) {
          const ob = bm[m.name];
          rows.push(el('div', { class: 'mbar-group' },
            el('div', { class: 'mbar-title' }, `${MOMENT_LABELS[m.name] || m.name} `, el('span', { class: 'muted' }, `${fmtTime(m.t)}–${fmtTime(m.t_end)}`),
              m.met ? el('span', { class: 'badge ok' }, 'cible atteinte') : null),
            ...Object.entries(m.values).map(([k, val]) => {
              const ov = ob && ob.values ? ob.values[k] : null;
              const met = m.met_by_index && m.met_by_index[k];
              const pk = m.targets && isNum(m.targets[k]) ? m.targets[k] : peak;
              const d = isNum(ov) ? val - ov : 0;
              const trend = !isNum(ov) ? null : d > 0.02 ? el('span', { class: 'ok-text', title: 'en hausse vs original' }, ' ▲')
                : d < -0.02 ? el('span', { class: 'err-text', title: 'en baisse vs original' }, ' ▼') : null;
              return el('div', { class: 'mbar', role: 'img', 'aria-label': `${INDEX_LABELS[k] || k} : variante ${nf2.format(val)}${isNum(ov) ? `, original ${nf2.format(ov)}${d < -0.02 ? ', en baisse' : d > 0.02 ? ', en hausse' : ''}` : ''}, cible ${nf2.format(pk)}${met ? ', atteinte' : ''}` },
                el('span', { class: 'mbar-k' }, INDEX_LABELS[k] || k),
                el('span', { class: 'mbar-track', 'aria-hidden': 'true' },
                  isNum(ov) ? el('span', { class: 'mbar-orig', style: { width: `${clamp(ov, 0, 1) * 100}%` } }) : null,
                  el('span', { class: 'mbar-var', style: { width: `${clamp(val, 0, 1) * 100}%`, background: INDEX_COLORS[k] } }),
                  el('span', { class: 'mbar-line peak', style: { left: `${clamp(pk, 0, 1) * 100}%` }, title: `cible ${nf2.format(pk)}` }),
                  el('span', { class: 'mbar-line floor', style: { left: `${clamp(floor, 0, 1) * 100}%` } })),
                el('span', { class: `mbar-v ${met ? 'ok-text' : ''}` }, `${nf2.format(val)}${isNum(ov) ? ` (orig. ${nf2.format(ov)})` : ''}`, trend,
                  el('span', { class: 'muted' }, ` · cible ${nf2.format(pk)}`)));
            })));
        }
        const regs = (r.vs_original && r.vs_original.regressions) || [];
        if (regs.length) {
          rows.push(el('p', { class: 'small err-text regressions' }, el('strong', {}, '▼ En baisse vs original : '),
            regs.map((g) => `${INDEX_LABELS[g.index] || g.index} (${(MOMENT_LABELS[g.moment] || g.moment).toLowerCase()}) ${nf2.format(g.original)} → ${nf2.format(g.variant)}`).join(' · ')));
        } else if (r.vs_original) {
          rows.push(el('p', { class: 'small ok-text' }, 'Aucun indice clé en baisse par rapport à l\'original.'));
        }
        const below = (e.seconds_below_floor || []).length;
        const belowO = (be.seconds_below_floor || []).length;
        body.push(el('div', { class: 'mbars' }, ...rows,
          el('p', { class: 'small mbar-legend' },
            el('span', { class: 'key-var' }), ' variante · ', el('span', { class: 'key-orig' }), ' original re-voicé · ',
            el('span', { class: 'key-peak' }), ' cible (par indice) · ', el('span', { class: 'key-floor' }), ` plancher ${nf2.format(floor)}`),
          el('p', { class: `small ${below <= belowO ? 'ok-text' : 'err-text'}` },
            `Secondes sous le plancher (${INDEX_LABELS[e.floor_index] || e.floor_index || 'attention'}) : ${below}${base ? ` (original : ${belowO})` : ''}`)));
      } else {
        body.push(el('p', { class: 'small muted' }, 'Pas de simulation pour cette variante.'));
      }

      const spans = (r && r.segments) || [];
      body.push(el('div', { class: 'table-wrap' }, el('table', { class: 'script-table' },
        el('caption', { class: 'sr-only' }, `Script de la variante ${vi + 1}`),
        el('thead', {}, el('tr', {}, ...['Temps', 'Voix off', "Texte à l'écran", 'Plan', 'Montage'].map((h) => el('th', { scope: 'col' }, h)))),
        el('tbody', {}, ...v.segments.map((s, i) => {
          const sp = spans[i];
          return el('tr', { class: `role-${s.role}` },
            el('td', { class: 'nowrap' }, sp ? `${fmtTime(sp.t)}–${fmtTime(sp.t_end)}` : `#${i + 1}`, el('br'), el('span', { class: `role-tag ${s.role}` }, ROLE_LABELS[s.role] || s.role)),
            el('td', {}, s.voice_over),
            el('td', {}, s.on_screen_text ? `« ${s.on_screen_text} »` : el('span', { class: 'muted' }, '—')),
            el('td', { class: 'nowrap' }, SHOT_LABELS[s.shot] || s.shot, s.cut_at_start ? el('span', { class: 'muted', title: 'Cut au début du segment' }, ' ✂') : null),
            el('td', {}, s.edit || el('span', { class: 'muted' }, '—')));
        })))));
      body.push(el('details', { class: 'small' }, el('summary', {}, 'Pourquoi cette version'), el('p', {}, v.rationale)));
      const status = el('span', { class: 'small', 'aria-live': 'polite' });
      body.push(el('div', { class: 'variant-actions' },
        el('button', {
          type: 'button', class: 'btn',
          onclick: async () => {
            const text = scriptText(v, spans, `Variante ${vi + 1} — ${v.angle}`);
            try { await navigator.clipboard.writeText(text); status.textContent = 'Script copié.'; } catch (_) {
              const ta = el('textarea', { class: 'sr-only' }); ta.value = text; document.body.append(ta); ta.select();
              try { document.execCommand('copy'); status.textContent = 'Script copié.'; } catch (__) { status.textContent = 'Copie impossible.'; }
              ta.remove();
            }
            setTimeout(() => { status.textContent = ''; }, 2500);
          },
        }, 'Copier le script'),
        el('a', { class: 'btn', href: `/api/coach/${encodeURIComponent(res.coach_id)}/markdown`, download: '' }, 'Télécharger .md'),
        status));
      return el('article', { class: `variant${isBest ? ' best' : ''}`, 'aria-label': `Variante ${vi + 1}${isBest ? ' (recommandée)' : ''}` }, ...body);
    },
  };

  function scriptText(v, spans, title) {
    const lines = [title, ''];
    v.segments.forEach((s, i) => {
      const sp = spans && spans[i];
      lines.push(`${sp ? `[${fmtTime(sp.t)}–${fmtTime(sp.t_end)}] ` : ''}${(ROLE_LABELS[s.role] || s.role).toUpperCase()}`);
      lines.push(`Voix off : ${s.voice_over}`);
      if (s.on_screen_text) lines.push(`Texte à l'écran : ${s.on_screen_text}`);
      lines.push(`Plan : ${SHOT_LABELS[s.shot] || s.shot}${s.cut_at_start ? ' (cut)' : ''}`);
      if (s.edit) lines.push(`Montage : ${s.edit}`);
      lines.push('');
    });
    return `${lines.join('\n').trim()}\n`;
  }

  $('#coach-btn').addEventListener('click', () => Coach.start());

  // ====================================================================== après publication
  function parseCountText(s) {
    s = String(s || '').trim().toLowerCase().replace(/[\s  ]+/g, ' ');
    if (!s) return null;
    const m = s.match(/^(\d[\d .,']*?)\s*(k|m|md|b)?\.?$/);
    if (!m) return NaN;
    let num = m[1].replace(/['\s]/g, '');
    const suf = m[2];
    if (suf) {
      num = num.replace(',', '.');
      const mult = suf === 'k' ? 1e3 : suf === 'm' ? 1e6 : 1e9;
      const v = parseFloat(num);
      return Number.isFinite(v) ? Math.round(v * mult) : NaN;
    }
    if (/^\d{1,3}([.,]\d{3})+$/.test(num)) return Number(num.replace(/[.,]/g, ''));
    const v = Number(num.replace(',', '.'));
    return Number.isFinite(v) ? Math.round(v) : NaN;
  }
  function parseDurationText(s) {
    s = String(s || '').trim().toLowerCase();
    if (!s) return null;
    let m = s.match(/^(\d+):(\d{2})(?::(\d{2}))?$/);
    if (m) return m[3] !== undefined ? (+m[1]) * 3600 + (+m[2]) * 60 + (+m[3]) : (+m[1]) * 60 + (+m[2]);
    if (/^\d+([.,]\d+)?$/.test(s)) return Number(s.replace(',', '.'));
    const compact = s.replace(/\s+/g, '');
    const re = /(\d+(?:[.,]\d+)?)(jours?|j|d|heures?|h|minutes?|min|mn|m|secondes?|sec|s)\.?/gy;
    let total = 0; let pos = 0;
    while ((m = re.exec(compact))) {
      const v = Number(m[1].replace(',', '.'));
      const u = m[2];
      const mult = /^(j|d|jour)/.test(u) ? 86400 : /^h/.test(u) ? 3600 : /^(m|mn|min)/.test(u) ? 60 : 1;
      total += v * mult;
      pos = re.lastIndex;
    }
    return pos === compact.length && pos > 0 ? total : NaN;
  }
  function parsePercentText(s) {
    s = String(s || '').trim().replace('%', '').replace(',', '.').trim();
    if (!s) return null;
    const v = Number(s);
    return Number.isFinite(v) && v >= 0 && v <= 100 ? v : NaN;
  }
  const PARSERS = { count: parseCountText, duration: parseDurationText, percent: parsePercentText };
  function formatFieldValue(kind, v) {
    if (!isNum(v)) return '';
    if (kind === 'duration') return v >= 60 ? fmtDuration(v) : `${nf1.format(v)} s`;
    if (kind === 'percent') return `${nf1.format(v)}`;
    return String(Math.round(v));
  }

  const After = {
    built: false,

    async open() {
      if (!this.built) this.build();
      await refreshHistory();
      this.fillSelect();
      await this.refreshComparison();
    },

    build() {
      this.built = true;
      const fields = $('#fields');
      fields.replaceChildren(...INSIGHT_FIELDS.map((f) => {
        const id = `f-${f.id}`;
        const input = el('input', {
          type: 'text', id, name: f.id, autocomplete: 'off',
          inputmode: f.kind === 'count' ? 'numeric' : 'text',
          placeholder: f.kind === 'duration' ? 'ex. 2:35 ou 1 h 20 min' : f.kind === 'percent' ? 'ex. 45,2' : 'ex. 12 300 ou 12,3 k',
          'aria-describedby': `${id}-hint`,
        });
        input.addEventListener('input', () => { this.updateHint(f); input.dataset.source = 'manuel'; this.setBadge(f.id, null); });
        return el('div', { class: 'field' },
          el('label', { for: id }, el('span', {}, f.label), el('span', { class: 'ocr-badge manuel', id: `${id}-badge` }, 'manuel')),
          input,
          el('div', { class: 'hint', id: `${id}-hint` }));
      }));
      wireDropzone($('#shot-drop'), $('#shot-input'), (files) => this.addShots(files));
      $('#shot-browse').addEventListener('click', (e) => { e.stopPropagation(); $('#shot-input').click(); });
      $('#ocr-btn').addEventListener('click', () => this.runOcr());
      $('#reel-select').addEventListener('change', () => { this.prefill(); this.refreshComparison(); });
      for (const r of document.querySelectorAll('input[name="window"]')) r.addEventListener('change', () => this.prefill());
      $('#validated').addEventListener('change', () => this.updateSaveState());
      $('#insights-form').addEventListener('submit', (e) => { e.preventDefault(); this.save(); });
      $('#scatter-metric').addEventListener('change', () => this.refreshComparison());
      $('#calib-btn').addEventListener('click', () => this.calibrate());
    },

    selectedJob() { return $('#reel-select').value || null; },
    selectedWindow() { return (document.querySelector('input[name="window"]:checked') || {}).value || '48h'; },

    fillSelect() {
      const sel = $('#reel-select');
      const prev = sel.value || state.jobId;
      const reels = state.reels;
      if (!reels.length) {
        sel.replaceChildren(el('option', { value: '' }, 'Aucun reel analysé'));
        sel.disabled = true;
      } else {
        sel.disabled = false;
        sel.replaceChildren(...reels.map((r) => {
          const has = ['48h', '7d'].filter((w) => r.insights && r.insights[w]).map((w) => (w === '7d' ? '7 j' : '48 h'));
          return el('option', { value: r.job_id },
            `${r.filename || r.job_id} — ${fmtDate(r.analyzed_at)} — score ${isNum(r.score) ? Math.round(r.score) : '–'}` +
            `${has.length ? ` — stats ${has.join(' + ')}` : ''}${r.result_available ? '' : ' (résultat supprimé)'}`);
        }));
        if (prev && reels.some((r) => r.job_id === prev)) sel.value = prev;
      }
      this.prefill();
      this.updateCalib(state.calibration);
    },

    prefill() {
      const job = this.selectedJob();
      const reel = state.reels.find((r) => r.job_id === job);
      const ins = reel && reel.insights ? reel.insights[this.selectedWindow()] : null;
      for (const f of INSIGHT_FIELDS) {
        const input = $(`#f-${f.id}`);
        input.value = ins && isNum(ins[f.id]) ? formatFieldValue(f.kind, ins[f.id]) : '';
        input.dataset.source = ins ? 'enregistré' : 'manuel';
        this.setBadge(f.id, ins && isNum(ins[f.id]) ? { level: 'enregistré' } : null);
        this.updateHint(f);
      }
      $('#validated').checked = false;
      $('#save-status').textContent = ins ? `Statistiques ${this.selectedWindow() === '7d' ? '7 jours' : '48 h'} déjà enregistrées le ${fmtDate(ins.validated_at)} (modifiables).` : '';
      this.updateSaveState();
    },

    setBadge(id, info) {
      const b = $(`#f-${id}-badge`);
      if (!b) return;
      if (!info) { b.className = 'ocr-badge manuel'; b.textContent = 'manuel'; b.title = 'Saisie manuelle'; return; }
      if (info.level === 'enregistré') { b.className = 'ocr-badge manuel'; b.textContent = 'enregistré'; b.title = 'Valeur déjà validée'; return; }
      b.className = `ocr-badge ${info.level}`;
      b.textContent = `OCR ${info.level} (${nf2.format(info.confidence)})`;
      b.title = `Lu : « ${info.value_text || ''} » près de « ${info.label_text || ''} »` +
        (info.alternatives && info.alternatives.length ? ` — autres valeurs lues : ${info.alternatives.join(', ')}` : '');
    },

    updateHint(f) {
      const input = $(`#f-${f.id}`);
      const hint = $(`#f-${f.id}-hint`);
      const v = PARSERS[f.kind](input.value);
      if (v === null) { hint.textContent = ''; hint.className = 'hint'; input.removeAttribute('aria-invalid'); return; }
      if (Number.isNaN(v)) {
        hint.textContent = f.kind === 'duration' ? 'Format non reconnu (ex. 2:35, 45 s, 1 h 20 min)' : 'Nombre non reconnu';
        hint.className = 'hint err';
        input.setAttribute('aria-invalid', 'true');
        return;
      }
      input.removeAttribute('aria-invalid');
      hint.className = 'hint';
      hint.textContent = f.kind === 'duration' ? `= ${nf1.format(v)} s` : f.kind === 'percent' ? `= ${nf1.format(v)} %` : `= ${nf0.format(v)}`;
    },

    readValues() {
      const out = {};
      let bad = 0;
      for (const f of INSIGHT_FIELDS) {
        const v = PARSERS[f.kind]($(`#f-${f.id}`).value);
        if (v === null) out[f.id] = null;
        else if (Number.isNaN(v)) bad++;
        else out[f.id] = v;
      }
      return { values: out, bad };
    },

    updateSaveState() {
      $('#save-btn').disabled = !($('#validated').checked && this.selectedJob());
    },

    addShots(files) {
      const imgs = files.filter((f) => /^image\//.test(f.type));
      if (imgs.length < files.length) $('#ocr-status').textContent = 'Seules les images sont acceptées.';
      for (const f of imgs) {
        if (state.shots.length >= 3) { $('#ocr-status').textContent = '3 captures maximum.'; break; }
        state.shots.push({ file: f, url: URL.createObjectURL(f) });
      }
      this.renderShots();
    },

    renderShots() {
      $('#shots').replaceChildren(...state.shots.map((s, i) => el('li', {},
        el('img', { src: s.url, alt: `Capture ${i + 1} : ${s.file.name}` }),
        el('button', {
          type: 'button', 'aria-label': `Retirer la capture ${i + 1}`, title: 'Retirer',
          onclick: () => { URL.revokeObjectURL(s.url); state.shots.splice(i, 1); this.renderShots(); },
        }, '×'))));
      $('#ocr-btn').disabled = !state.shots.length;
    },

    async runOcr() {
      if (!state.shots.length) return;
      const status = $('#ocr-status');
      status.textContent = 'Lecture des captures en cours…';
      $('#ocr-btn').disabled = true;
      const fd = new FormData();
      for (const s of state.shots) fd.append('file', s.file, s.file.name);
      fd.append('window', this.selectedWindow());
      try {
        const res = await api('/api/insights/ocr', { method: 'POST', body: fd });
        state.ocr = res;
        const fields = res.fields || {};
        let n = 0;
        for (const f of INSIGHT_FIELDS) {
          const got = fields[f.id];
          if (!got || !isNum(got.value)) continue;
          n++;
          const input = $(`#f-${f.id}`);
          input.value = formatFieldValue(f.kind, got.value);
          input.dataset.source = 'ocr';
          this.setBadge(f.id, got);
          this.updateHint(f);
        }
        $('#ocr-raw-text').textContent = res.raw_text || '(aucun texte reconnu)';
        let msg = `${n} champ(s) pré-rempli(s) sur ${INSIGHT_FIELDS.length}. Vérifiez chaque valeur avant de valider.`;
        if (fields.interactions && isNum(fields.interactions.value)) msg += ` (Interactions lues : ${nf0.format(fields.interactions.value)}, non enregistrées.)`;
        if (res.window_guess && res.window_guess !== this.selectedWindow()) {
          msg += ` Attention : les captures semblent concerner la fenêtre ${res.window_guess === '7d' ? '7 jours' : '48 h'}.`;
        }
        status.textContent = msg;
        $('#validated').checked = false;
        this.updateSaveState();
      } catch (e) {
        status.textContent = `OCR impossible : ${e.message}`;
      } finally {
        $('#ocr-btn').disabled = !state.shots.length;
      }
    },

    async save() {
      const job = this.selectedJob();
      const st = $('#save-status');
      if (!job) { st.textContent = 'Choisissez un reel.'; return; }
      if (!$('#validated').checked) { st.textContent = 'Cochez la case de vérification.'; return; }
      const { values, bad } = this.readValues();
      if (bad) { st.className = 'small err-text'; st.textContent = `${bad} champ(s) mal formé(s) : corrigez-les.`; return; }
      const body = {
        job_id: job, window: this.selectedWindow(), validated: true, ...values,
        raw_ocr: state.ocr ? { fields: state.ocr.fields, raw_text: state.ocr.raw_text, window_guess: state.ocr.window_guess } : null,
      };
      try {
        await api('/api/insights', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        st.className = 'small ok-text';
        st.textContent = 'Statistiques enregistrées.';
        $('#validated').checked = false;
        this.updateSaveState();
        await refreshHistory();
        const keep = $('#reel-select').value;
        this.fillSelect();
        $('#reel-select').value = keep;
        st.className = 'small ok-text';
        st.textContent = 'Statistiques enregistrées.';
        await this.refreshComparison();
      } catch (e) {
        st.className = 'small err-text';
        st.textContent = `Enregistrement refusé : ${e.message}`;
      }
    },

    updateCalib(cal) {
      const btn = $('#calib-btn');
      const info = $('#calib-info');
      if (!cal) { btn.disabled = true; info.textContent = ''; return; }
      btn.disabled = !cal.can_calibrate;
      info.textContent = cal.can_calibrate
        ? `${cal.n_eligible} reels avec durée moyenne de visionnage réelle.`
        : `Disponible à partir de ${cal.min_required} reels renseignés (durée moyenne de visionnage) — actuellement ${cal.n_eligible}.`;
      if (cal.current && cal.current.params) {
        info.textContent += ` Calibration active (n = ${cal.current.n ?? '?'}).`;
      }
    },

    async refreshComparison() {
      for (const c of state.cmpCharts) c.destroy();
      state.cmpCharts = [];
      const job = this.selectedJob();
      const sum = $('#cmp-summary');
      $('#chart-cmp').replaceChildren();
      $('#chart-scatter').replaceChildren();
      if (!job) { sum.replaceChildren(el('p', { class: 'muted' }, 'Analysez un reel puis renseignez ses statistiques.')); return; }
      let cmp;
      try {
        cmp = await api(`/api/reels/${encodeURIComponent(job)}/comparison`);
      } catch (e) {
        sum.replaceChildren(el('p', { class: 'err-text' }, `Comparaison indisponible : ${e.message}`));
        return;
      }
      this.updateCalib(cmp.calibration);
      const p = cmp.predicted || {};
      const real = cmp.actual_avg_watch_ratio;
      const w = cmp.best_window;
      const act = w ? cmp.actual[w] : null;
      const stat = (k, v, title) => el('div', { class: 'stat', title: title || '' }, el('div', { class: 'k' }, k), el('div', { class: 'v' }, v));
      const stats = [
        stat('Visionnage moyen prédit', isNum(p.avg_watch_s) ? `${nf1.format(p.avg_watch_s)} s · ${fmtPct(p.avg_watch_ratio)}` : fmtPct(p.avg_watch_ratio)),
        stat(`Visionnage moyen réel${w ? ` (${w === '7d' ? '7 j' : '48 h'})` : ''}`,
          act && isNum(act.avg_watch_s) ? `${nf1.format(act.avg_watch_s)} s · ${fmtPct(real)}` : 'non renseigné'),
        stat('Ratio prédit / réel', isNum(cmp.ratio_pred_vs_real) ? `× ${nf2.format(cmp.ratio_pred_vs_real)}` : '–',
          '> 1 : la prédiction surestime la rétention'),
        stat('Écart', isNum(cmp.error_pts) ? `${cmp.error_pts >= 0 ? '+' : ''}${nf1.format(cmp.error_pts)} pts` : '–'),
        stat('Score prédit', isNum(p.score) ? String(Math.round(p.score)) : '–'),
      ];
      for (const ww of ['48h', '7d']) {
        const a = cmp.actual[ww];
        if (a && isNum(a.views)) stats.push(stat(`Vues ${ww === '7d' ? '7 j' : '48 h'}`, nf0.format(a.views)));
      }
      sum.replaceChildren(...stats);
      if (!hasUPlot) return;

      // (a) rétention prédite vs durée moyenne réelle
      const curve = Array.isArray(p.curve) ? p.curve : [];
      const dur = isNum(cmp.duration_s) ? cmp.duration_s : Math.max(1, curve.length - 1);
      if (curve.length) {
        const xs = curve.map((_, i) => Math.min(i, dur));
        const node = $('#chart-cmp');
        const realS = act && isNum(act.avg_watch_s) ? act.avg_watch_s : null;
        const predS = isNum(p.avg_watch_s) ? p.avg_watch_s : null;
        const u = new uPlot({
          width: chartWidth(node), height: 220, legend: { show: true },
          cursor: { drag: { x: false, y: false } },
          scales: { x: { time: false, range: [0, dur] }, y: { range: [0, 1] } },
          axes: [{ ...AXIS_STYLE, values: (uu, v) => v.map(fmtTime) }, { ...AXIS_STYLE, size: 44, values: (uu, v) => v.map((x) => `${Math.round(x * 100)} %`) }],
          series: [{ label: 't', value: (uu, v) => fmtTime(v) },
            { label: 'Rétention prédite', stroke: RETENTION_COLOR, width: 2, fill: 'rgba(233,236,242,0.07)', points: { show: false },
              value: (uu, v) => (isNum(v) ? fmtPct(v) : '–') }],
          hooks: {
            draw: [(uu) => {
              const { ctx } = uu;
              const { top, height } = uu.bbox;
              const line = (s, color, dash, label) => {
                if (!isNum(s)) return;
                const x = Math.round(uu.valToPos(Math.min(s, dur), 'x', true));
                ctx.save();
                ctx.strokeStyle = color;
                ctx.lineWidth = 2 * devicePixelRatio;
                ctx.setLineDash(dash.map((d) => d * devicePixelRatio));
                ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke();
                ctx.fillStyle = color;
                ctx.font = `${12 * devicePixelRatio}px -apple-system, sans-serif`;
                ctx.fillText(label, x + 5 * devicePixelRatio, top + 14 * devicePixelRatio + (dash.length ? 16 * devicePixelRatio : 0));
                ctx.restore();
              };
              line(predS, '#a3acbb', [6, 4], `prédit ${nf1.format(predS || 0)} s`);
              line(realS, '#e8400d', [], `réel ${nf1.format(realS || 0)} s`);
            }],
          },
        }, [xs, curve.map((v) => (isNum(v) ? v : null))], node);
        state.cmpCharts.push(u);
      }

      // (b) nuage score prédit vs vues / portée
      const metric = $('#scatter-metric').value;
      const pts = (cmp.scatter || []).filter((s) => isNum(s.score) && isNum(s[metric])).sort((a, b) => a.score - b.score);
      const note = $('#scatter-note');
      if (!pts.length) {
        note.textContent = 'Aucun reel avec des statistiques enregistrées pour le moment.';
      } else {
        note.textContent = `${pts.length} reel(s) renseigné(s) (fenêtre 7 j si disponible, sinon 48 h). Point orange = reel sélectionné.`;
        const node = $('#chart-scatter');
        const xs = pts.map((s) => s.score);
        const others = pts.map((s) => (s.job_id === job ? null : s[metric]));
        const cur = pts.map((s) => (s.job_id === job ? s[metric] : null));
        const ymax = Math.max(...pts.map((s) => s[metric])) * 1.15 || 1;
        const u = new uPlot({
          width: chartWidth(node), height: 240, legend: { show: false },
          cursor: { drag: { x: false, y: false } },
          scales: { x: { time: false, range: [0, 100] }, y: { range: [0, ymax] } },
          axes: [{ ...AXIS_STYLE, label: 'Score prédit', labelSize: 18 }, { ...AXIS_STYLE, size: 60, values: (uu, v) => v.map((x) => nf0.format(x)) }],
          series: [{}, { label: 'Reels', stroke: '#d0b2ff', fill: '#d0b2ff', paths: () => null, points: { show: true, size: 9, fill: '#d0b2ff' } },
            { label: 'Sélectionné', stroke: '#e8400d', paths: () => null, points: { show: true, size: 12, fill: '#e8400d' } }],
        }, [xs, others, cur], node);
        state.cmpCharts.push(u);
      }
    },

    async calibrate() {
      const out = $('#calib-result');
      out.className = 'small';
      out.textContent = 'Calibration en cours…';
      $('#calib-btn').disabled = true;
      try {
        const res = await api('/api/calibrate', { method: 'POST' });
        const r = res.result || {};
        const params = r.params ? Object.entries(r.params).map(([k, v]) => `${k} = ${isNum(v) ? nf2.format(v) : v}`).join(', ') : '';
        out.className = 'small ok-text';
        out.textContent = `Rétention recalibrée sur ${r.n ?? res.pairs.length} reels. Erreur moyenne ${isNum(r.mae_before) ? fmtPct(r.mae_before, 1) : '?'} → ${isNum(r.mae_after) ? fmtPct(r.mae_after, 1) : '?'}` +
          `${params ? ` (${params})` : ''}. S'applique aux prochaines analyses.`;
      } catch (e) {
        out.className = 'small err-text';
        out.textContent = `Calibration impossible : ${e.message}`;
      } finally {
        await refreshHistory();
        this.updateCalib(state.calibration);
      }
    },
  };

  // ====================================================================== révélation du score
  const Reveal = {
    sparks: [],
    rings: [],
    raf: 0,
    reduce: window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches,

    verdict(v) {
      if (v >= 75) return 'Fort potentiel';
      if (v >= 60) return 'Bon, encore à affiner';
      if (v >= 40) return 'Moyen : des creux à corriger';
      return 'Faible : à retravailler';
    },

    open() {
      const s = state.summary;
      if (!s || !isNum(s.score)) return Promise.resolve();
      const score = Math.round(s.score);
      const box = $('#reveal');
      const num = $('#reveal-num');
      const r = s.retention || {};
      $('#reveal-name').textContent = $('#r-title').textContent || '';
      $('#reveal-verdict').textContent = this.verdict(score);
      $('#reveal-verdict').className = `reveal-verdict ${scoreClass(score)}`;
      const stat = (k, v) => el('div', { class: 'stat' }, el('div', { class: 'k' }, k), el('div', { class: 'v' }, v));
      $('#reveal-stats').replaceChildren(
        stat('Visionnage moyen', isNum(r.avg_watch_s) ? `${nf1.format(r.avg_watch_s)} s` : fmtPct(r.avg_watch_ratio)),
        stat("Jusqu'au bout", fmtPct(r.completion)),
        stat('Corrections', String((s.recommendations || []).length)));
      num.className = `reveal-num ${scoreClass(score)}`;
      num.textContent = '0';
      box.classList.remove('done', 'out');
      box.hidden = false;
      // une seule scène 3D animée à la fois pendant la révélation
      for (const b of [heroBrain, view3d.inst]) if (b) b.paused = true;
      this.prevFocus = document.activeElement;
      $('#reveal-go').focus({ preventScroll: true });
      this.fx = $('#reveal-fx');
      this.ctx = this.fx.getContext('2d');
      this.sizeFx();
      this.sparks = []; this.rings = [];
      cancelAnimationFrame(this.raf);
      const t0 = performance.now();
      const dur = this.reduce ? 0 : 2200;
      let landed = false;
      const loop = (now) => {
        const k = dur ? Math.min(1, (now - t0) / dur) : 1;
        const e = k === 1 ? 1 : 1 - Math.pow(2, -10 * k);
        num.textContent = String(Math.round(score * e));
        if (!landed && k < 1 && !this.reduce && Math.random() < 0.5) this.ember();
        if (k === 1 && !landed) { landed = true; this.land(score); }
        this.drawFx();
        this.raf = requestAnimationFrame(loop);
      };
      this.raf = requestAnimationFrame(loop);
      return new Promise((res) => { this.resolve = res; });
    },

    land(score) {
      const num = $('#reveal-num');
      num.classList.add('landed');
      $('#reveal').classList.add('done');
      if (this.reduce) return;
      if (view3d.inst) setTimeout(() => view3d.inst.pulse(), 600);
      const rc = num.getBoundingClientRect();
      const cx = rc.left + rc.width / 2, cy = rc.top + rc.height / 2;
      const cols = ['#d0b2ff', '#ffeed8', '#ffc9a8', '#e8400d', '#ffffff'];
      const n = 140 + Math.round(score * 1.6);
      for (let i = 0; i < n; i++) {
        const a = Math.random() * Math.PI * 2, v = 2 + Math.random() * 11;
        this.sparks.push({ x: cx, y: cy, vx: Math.cos(a) * v, vy: Math.sin(a) * v * 0.8, life: 1, decay: 0.008 + Math.random() * 0.018,
          r: 0.8 + Math.random() * 2.6, c: cols[(Math.random() * cols.length) | 0] });
      }
      this.rings.push({ x: cx, y: cy, r: 20, a: 1, w: 3 }, { x: cx, y: cy, r: 0, a: 0.7, w: 1.5, delay: 8 });
    },

    ember() {
      const w = this.fx.clientWidth, h = this.fx.clientHeight;
      this.sparks.push({ x: w * (0.3 + Math.random() * 0.4), y: h * (0.65 + Math.random() * 0.3), vx: (Math.random() - 0.5) * 0.6,
        vy: -1 - Math.random() * 2.2, life: 1, decay: 0.012 + Math.random() * 0.012, r: 0.6 + Math.random() * 1.6,
        c: Math.random() < 0.5 ? '#d0b2ff' : '#ffeed8' });
    },

    sizeFx() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.dpr = dpr;
      this.fx.width = Math.round(this.fx.clientWidth * dpr);
      this.fx.height = Math.round(this.fx.clientHeight * dpr);
    },

    drawFx() {
      const ctx = this.ctx, dpr = this.dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, this.fx.width, this.fx.height);
      ctx.globalCompositeOperation = 'lighter';
      for (const g of this.rings) {
        if (g.delay) { g.delay--; continue; }
        g.r += 14 + g.r * 0.02; g.a *= 0.955;
        ctx.strokeStyle = `rgba(255, 228, 205, ${g.a})`;
        ctx.lineWidth = g.w;
        ctx.beginPath(); ctx.arc(g.x, g.y, g.r, 0, Math.PI * 2); ctx.stroke();
      }
      this.rings = this.rings.filter((g) => g.a > 0.02);
      for (const p of this.sparks) {
        p.x += p.vx; p.y += p.vy; p.vx *= 0.975; p.vy = p.vy * 0.975 + 0.02; p.life -= p.decay;
        ctx.globalAlpha = Math.max(0, p.life);
        ctx.fillStyle = p.c;
        ctx.shadowColor = p.c; ctx.shadowBlur = 12;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
      }
      ctx.shadowBlur = 0; ctx.globalAlpha = 1;
      this.sparks = this.sparks.filter((p) => p.life > 0);
    },

    close() {
      const box = $('#reveal');
      if (box.hidden || box.classList.contains('out')) return;
      box.classList.add('out');
      setTimeout(() => {
        box.hidden = true;
        box.classList.remove('out', 'done');
        for (const b of [heroBrain, view3d.inst]) if (b) b.paused = false;
        cancelAnimationFrame(this.raf);
        if (this.prevFocus && this.prevFocus.focus) this.prevFocus.focus({ preventScroll: true });
        if (this.resolve) { this.resolve(); this.resolve = null; }
      }, this.reduce ? 0 : 500);
    },
  };
  $('#reveal-go').addEventListener('click', () => Reveal.close());
  $('#reveal').addEventListener('click', (e) => { if (e.target === e.currentTarget && $('#reveal').classList.contains('done')) Reveal.close(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('#reveal').hidden) Reveal.close(); });
  window.addEventListener('resize', () => { if (!$('#reveal').hidden) Reveal.sizeFx(); });
  $('#score-value').addEventListener('click', () => Reveal.open());
  $('#score-value').addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); Reveal.open(); } });

  function backToStart() {
    document.body.classList.remove('has-result');
    $('#result').hidden = true;
    $('#empty-state').hidden = false;
    state.jobId = null;
    try { history.replaceState(null, '', location.pathname); } catch (_) { /* ignore */ }
    for (const b of document.querySelectorAll('#history button')) b.setAttribute('aria-current', 'false');
    video.pause();
    selectTab(tabs[0]);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
  $('#new-analysis-btn').addEventListener('click', () => { backToStart(); setTimeout(() => $('#dropzone').focus({ preventScroll: true }), 400); });
  $('#brand-home').addEventListener('click', (e) => { e.preventDefault(); backToStart(); });

  // ====================================================================== démarrage
  async function init() {
    refreshHealth();
    setInterval(refreshHealth, 30000);
    await refreshHistory();
    // Analyse en cours (rechargement de la page) ?
    try {
      const jobs = await api('/api/jobs');
      const active = jobs.find((j) => j.status === 'running') || jobs.find((j) => j.status === 'queued');
      if (active) pollJob(active.job_id);
    } catch (_) { /* ignore */ }
    const m = location.hash.match(/job=([^&]+)/);
    if (m) loadResult(decodeURIComponent(m[1]));
    brain.loadMesh();
    window.addEventListener('hashchange', () => {
      const h = location.hash.match(/job=([^&]+)/);
      if (h && decodeURIComponent(h[1]) !== state.jobId) loadResult(decodeURIComponent(h[1]));
    });
  }
  init();
})();
