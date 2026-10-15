/* Reel Predictor — cerveau 3D interactif (WebGL 1, aucune dépendance).
   Surface piale fsaverage5 (20 484 sommets, même ordre que la sortie TRIBE v2), régions de l'atlas par sommet.
   Modes : « idle » (ondes neuronales décoratives), « scan » (balayage pendant l'analyse), « data » (activation
   prédite, interpolée entre deux secondes). Glisser = tourner, molette/pincer = zoom, survol = nom de la région. */
'use strict';

(() => {
  const VS = `
attribute vec3 aPos; attribute vec3 aNrm; attribute float aSulc; attribute float aActA; attribute float aActB;
attribute float aHl; attribute float aRoi;
uniform mat4 uMVP; uniform mat4 uModel; uniform float uMix; uniform float uTime; uniform float uMode;
uniform float uHover; uniform float uScan;
varying vec3 vN; varying vec3 vP; varying float vSulc; varying float vAct; varying float vHl; varying float vHover;
varying float vWave; varying float vScan;
float hash(float n) { return fract(sin(n) * 43758.5453); }
void main() {
  vec3 p = aPos;
  vAct = mix(aActA, aActB, uMix) * 2.0 - 1.0;
  // ondes (idle) : 3 fronts qui traversent le cortex + étincelles
  float w = 0.0;
  w += smoothstep(0.985, 1.0, sin(dot(p, vec3(2.6, 1.1, 0.4)) * 4.0 - uTime * 1.3)) * 0.9;
  w += smoothstep(0.988, 1.0, sin(dot(p, vec3(-0.7, 2.2, 1.6)) * 3.4 - uTime * 0.9 + 1.7)) * 0.8;
  w += smoothstep(0.975, 1.0, sin(length(p - vec3(0.3, -0.4, 0.2)) * 7.0 - uTime * 1.7)) * 1.1;
  float id = aRoi * 13.1 + p.x * 91.7 + p.y * 37.3 + p.z * 11.9;
  w += step(0.992, hash(floor(uTime * 4.0) + id)) * 1.6;
  vWave = w;
  // scan : plan horizontal qui monte et descend
  float sy = mix(-1.0, 1.0, 0.5 + 0.5 * sin(uTime * 1.6));
  vScan = uScan * (smoothstep(0.08, 0.0, abs(p.z - sy)) + 0.08 * step(p.z, sy));
  vHover = (uHover > 0.5 && abs(aRoi - uHover) < 0.5) ? 1.0 : 0.0;
  // léger gonflement des zones très actives (data)
  float bump = uMode > 1.5 ? max(vAct, 0.0) * 0.012 : 0.0;
  p += aNrm * (bump + vHover * 0.008);
  vec4 wp = uModel * vec4(p, 1.0);
  vP = wp.xyz; vN = normalize((uModel * vec4(aNrm, 0.0)).xyz);
  vSulc = aSulc; vHl = aHl;
  gl_Position = uMVP * vec4(p, 1.0);
}`;

  const FS = `
precision highp float;
varying vec3 vN; varying vec3 vP; varying float vSulc; varying float vAct; varying float vHl; varying float vHover;
varying float vWave; varying float vScan;
uniform float uMode; uniform float uTime; uniform float uHlOn; uniform vec3 uHlColor; uniform float uData;
vec3 diverge(float a) {
  // sous la moyenne : bleu nuit -> indigo ; au-dessus : lavande -> pêche -> braise
  vec3 neg = mix(vec3(0.12, 0.11, 0.17), vec3(0.16, 0.22, 0.70), pow(clamp(-a, 0.0, 1.0), 1.6));
  vec3 pos = a < 0.45 ? mix(vec3(0.16, 0.14, 0.20), vec3(0.82, 0.70, 1.0), a / 0.45)
                      : a < 0.75 ? mix(vec3(0.82, 0.70, 1.0), vec3(1.0, 0.93, 0.85), (a - 0.45) / 0.3)
                                 : mix(vec3(1.0, 0.93, 0.85), vec3(0.91, 0.25, 0.05), (a - 0.75) / 0.25);
  return a < 0.0 ? neg : pos;
}
void main() {
  vec3 N = normalize(vN);
  vec3 V = normalize(vec3(0.0, 0.0, 3.2) - vP);
  if (dot(N, V) < 0.0) N = -N;
  vec3 L = normalize(vec3(-0.4, 0.8, 1.0));
  float lam = 0.35 + 0.65 * max(dot(N, L), 0.0);
  float spec = pow(max(dot(reflect(-L, N), V), 0.0), 24.0) * 0.25;
  float fres = pow(1.0 - max(dot(N, V), 0.0), 2.4);
  float gyri = 1.0 - 0.45 * vSulc;  // sillons plus sombres
  vec3 base = mix(vec3(0.09, 0.08, 0.12), vec3(0.24, 0.20, 0.30), gyri) ;
  vec3 col; vec3 emit = vec3(0.0);
  if (uMode > 1.5) {            // data
    vec3 c = diverge(vAct);
    col = c * (0.55 + 0.45 * gyri);
    emit = c * smoothstep(0.3, 1.0, vAct) * 0.9;
  } else {                      // idle / scan
    col = base;
    emit = mix(vec3(0.82, 0.70, 1.0), vec3(1.0, 0.75, 0.55), clamp(vP.z * 0.5 + 0.5, 0.0, 1.0)) * vWave * 0.9 * gyri;
    emit += vec3(1.0, 0.85, 0.7) * vScan * 1.2;
  }
  if (uHlOn > 0.5) {
    float h = vHl;
    col = mix(col * 0.35, col, step(0.01, h));
    emit += uHlColor * h * (0.55 + 0.25 * sin(uTime * 4.0));
  }
  emit += vec3(1.0, 0.95, 0.9) * vHover * 0.45;
  vec3 rim = mix(vec3(0.82, 0.70, 1.0), vec3(0.91, 0.25, 0.05), 0.25) * fres * (uMode > 1.5 ? 0.55 : 0.9);
  gl_FragColor = vec4(col * lam + spec + emit + rim, 1.0);
}`;

  // particules : nuage ellipsoïdal autour du cortex (orbite lente, aspirées vers le cerveau pendant le scan,
  // plus vives quand l'activité prédite monte, explosion à la révélation du score)
  const PVS = `
attribute vec4 aSeed; attribute vec2 aExt;
uniform mat4 uMVP; uniform float uTime; uniform float uScan; uniform float uEnergy; uniform float uBurst;
uniform float uDpr; uniform float uH;
varying float vCol; varying float vA;
void main() {
  float r = aSeed.x;
  float th = aSeed.y + uTime * aSeed.w;
  float ph = aSeed.z + sin(uTime * 0.3 + aSeed.y) * 0.15;
  float k = fract(uTime * 0.5 * (0.4 + abs(aSeed.w) * 3.0) + aSeed.y * 0.159);
  r = mix(r, mix(r + 0.35, 0.92, k), uScan);
  r *= 1.0 + uBurst * (0.5 + 0.9 * fract(aSeed.y * 7.13));
  vec3 p = vec3(sin(ph) * cos(th) * 0.80, sin(ph) * sin(th) * 1.02, cos(ph) * 0.78) * r;
  vec4 cp = uMVP * vec4(p, 1.0);
  gl_Position = cp;
  float tw = 0.55 + 0.45 * sin(uTime * (1.5 + abs(aSeed.w) * 9.0) + aSeed.y * 20.0);
  float fade = smoothstep(2.25, 1.35, r) * (uScan > 0.5 ? smoothstep(0.9, 1.05, r) : 1.0);
  vA = tw * fade * (0.6 + 0.9 * uEnergy + 0.5 * uScan + 1.6 * uBurst);
  vCol = aExt.x;
  gl_PointSize = min(aExt.y * uDpr * (1.0 + uBurst * 1.4 + uEnergy * 0.5) * (uH / 900.0) * 5.2 / cp.w, 18.0 * uDpr);
}`;
  const PFS = `
precision highp float;
varying float vCol; varying float vA;
void main() {
  float d = length(gl_PointCoord - 0.5);
  float a = smoothstep(0.5, 0.0, d);
  a = (a * a + smoothstep(0.12, 0.0, d)) * vA;
  vec3 c = mix(vec3(0.82, 0.70, 1.0), vec3(1.0, 0.93, 0.85), clamp(vCol * 1.15, 0.0, 1.0));
  c = mix(c, vec3(0.95, 0.38, 0.12), step(0.9, vCol));
  gl_FragColor = vec4(c * a, a);
}`;

  // ------------------------------------------------------------------ maths
  const m4 = {
    persp(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
      return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
    },
    mul(a, b) {
      const o = new Float32Array(16);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
        o[c * 4 + r] = s;
      }
      return o;
    },
    rx(t) { const c = Math.cos(t), s = Math.sin(t); return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]); },
    ry(t) { const c = Math.cos(t), s = Math.sin(t); return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]); },
    trans(x, y, z) { return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]); },
    // RAS (x droite, y avant, z haut) -> GL (x droite, y haut, z vers l'écran = arrière)
    ras: new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]),
  };
  const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer;

  let meshPromise = null;
  function loadMesh(url = '/brain3d.json') {
    if (!meshPromise) {
      meshPromise = fetch(url).then((r) => {
        if (!r.ok) throw new Error('cerveau 3D absent (lancez « make setup »)');
        return r.json();
      }).then((j) => {
        const pos = new Float32Array(b64(j.positions));
        const faces = new Uint16Array(b64(j.faces));
        const sulc = new Uint8Array(b64(j.sulc));
        const roi = new Uint8Array(b64(j.roi));
        const nrm = new Float32Array(pos.length);
        for (let i = 0; i < faces.length; i += 3) {
          const a = faces[i] * 3, b = faces[i + 1] * 3, c = faces[i + 2] * 3;
          const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
          const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
          const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
          for (const k of [a, b, c]) { nrm[k] += nx; nrm[k + 1] += ny; nrm[k + 2] += nz; }
        }
        for (let i = 0; i < nrm.length; i += 3) {
          const l = Math.hypot(nrm[i], nrm[i + 1], nrm[i + 2]) || 1;
          nrm[i] /= l; nrm[i + 1] /= l; nrm[i + 2] /= l;
        }
        return { ...j, pos, faces, sulc, roi, nrm, nv: j.n_vertices };
      });
    }
    return meshPromise;
  }

  // ------------------------------------------------------------------ composant
  class Brain3D {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.opts = { autoRotate: true, zoom: true, hover: true, dist: 3.9, ...opts };
      this.yaw = opts.yaw ?? -1.15;
      this.pitch = opts.pitch ?? 0.18;
      this.vyaw = 0; this.vpitch = 0;
      this.dist = this.opts.dist;
      this.mode = 'idle';
      this.scan = 0;
      this.mix = 0;
      this.hover = 0;
      this.hlOn = false;
      this.hlColor = [0.82, 0.7, 1.0];
      this.lastInteract = -1e9;
      this.listeners = {};
      this.ready = this.init();
    }

    on(ev, fn) { (this.listeners[ev] ||= []).push(fn); return this; }
    emit(ev, d) { for (const fn of this.listeners[ev] || []) fn(d); }

    async init() {
      const gl = this.canvas.getContext('webgl', { antialias: true, alpha: true, premultipliedAlpha: true });
      if (!gl) throw new Error('WebGL indisponible');
      this.gl = gl;
      this.mesh = await loadMesh();
      this.setupGL();
      // perte de contexte GPU (veille, pilote, trop de contextes) : on attend sa restauration, sans planter
      this.canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); this.lost = true; });
      this.canvas.addEventListener('webglcontextrestored', () => {
        this.lost = false;
        this.setupGL();
        if (this.data) { this.ta = this.tb = -1; this.setTime(this.lastT || 0); }
        if (this.hlKey !== undefined) this.highlightIndex(this.hlKey, this.hlColor);
      });
      this.visible = true;
      if ('IntersectionObserver' in window) {
        this.io = new IntersectionObserver((es) => { this.visible = es[es.length - 1].isIntersecting; });
        this.io.observe(this.canvas);
      }
      this.bindInput();
      this.ro = new ResizeObserver(() => this.resize());
      this.ro.observe(this.canvas);
      this.resize();
      this.t0 = performance.now();
      const loop = (now) => { this.frame(now); this.raf = requestAnimationFrame(loop); };
      this.raf = requestAnimationFrame(loop);
      return this;
    }

    setupGL() {
      const gl = this.gl;
      const mesh = this.mesh;
      const sh = (type, src) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        return s;
      };
      const link = (vs, fs) => {
        const p = gl.createProgram();
        gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
        gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
        return p;
      };
      const prog = link(VS, FS);
      this.prog = prog;
      const nv = mesh.nv;
      this.buf = {};
      this.brainAttrs = [];
      this.partAttrs = [];
      this.enabled = new Set();
      const attr = (name, data, size, type = gl.FLOAT, norm = false, usage = gl.STATIC_DRAW, p = prog, list = this.brainAttrs) => {
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, data, usage);
        const loc = gl.getAttribLocation(p, name);
        if (loc >= 0) list.push({ loc, b, size, type, norm });
        this.buf[name] = b;
      };
      attr('aPos', mesh.pos, 3);
      attr('aNrm', mesh.nrm, 3);
      attr('aSulc', mesh.sulc, 1, gl.UNSIGNED_BYTE, true);
      attr('aRoi', new Float32Array(mesh.roi), 1);
      const mid = new Uint8Array(nv).fill(128);
      attr('aActA', mid, 1, gl.UNSIGNED_BYTE, true, gl.DYNAMIC_DRAW);
      attr('aActB', mid, 1, gl.UNSIGNED_BYTE, true, gl.DYNAMIC_DRAW);
      attr('aHl', new Uint8Array(nv), 1, gl.UNSIGNED_BYTE, true, gl.DYNAMIC_DRAW);
      const ib = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.faces, gl.STATIC_DRAW);
      this.u = {};
      for (const n of ['uMVP', 'uModel', 'uMix', 'uTime', 'uMode', 'uHover', 'uScan', 'uHlOn', 'uHlColor', 'uData']) {
        this.u[n] = gl.getUniformLocation(prog, n);
      }
      // particules
      const pprog = link(PVS, PFS);
      this.pprog = pprog;
      const N = this.opts.particles ?? 1600;
      const seed = new Float32Array(N * 4), ext = new Float32Array(N * 2);
      let h = 1234567;
      const rnd = () => { h = (h * 16807) % 2147483647; return h / 2147483647; };
      for (let i = 0; i < N; i++) {
        seed[i * 4] = 1.06 + Math.pow(rnd(), 1.8) * 0.95;            // rayon (dense près du cortex)
        seed[i * 4 + 1] = rnd() * Math.PI * 2;                         // azimut
        seed[i * 4 + 2] = Math.acos(2 * rnd() - 1);                    // élévation
        seed[i * 4 + 3] = (0.03 + rnd() * 0.16) * (rnd() < 0.5 ? -1 : 1); // vitesse orbitale
        ext[i * 2] = rnd();                                            // couleur
        ext[i * 2 + 1] = 1.4 + Math.pow(rnd(), 2.5) * 6.5;               // taille
      }
      this.nPart = N;
      attr('aSeed', seed, 4, gl.FLOAT, false, gl.STATIC_DRAW, pprog, this.partAttrs);
      attr('aExt', ext, 2, gl.FLOAT, false, gl.STATIC_DRAW, pprog, this.partAttrs);
      this.pu = {};
      for (const n of ['uMVP', 'uTime', 'uScan', 'uEnergy', 'uBurst', 'uDpr', 'uH']) this.pu[n] = gl.getUniformLocation(pprog, n);
      this.energy = this.energy ?? 0.25;
      this.energyTarget = this.energyTarget ?? 0.25;
      this.burstT = this.burstT ?? -1e9;
      this.enabled = new Set();
      gl.enable(gl.DEPTH_TEST);
      gl.clearColor(0, 0, 0, 0);
    }

    resize() {
      // résolution plafonnée (1 600 px de côté) : au-delà, coût GPU sans gain visible
      const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
      const dpr = Math.min(window.devicePixelRatio || 1, 2, 1600 / Math.max(cw, ch, 1));
      const w = Math.max(1, Math.round(cw * dpr));
      const h = Math.max(1, Math.round(ch * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    }

    model() {
      return m4.mul(m4.rx(this.pitch), m4.mul(m4.ry(this.yaw), m4.ras));
    }

    frame(now) {
      const gl = this.gl;
      if (!gl || this.lost || this.paused || !this.visible || !this.canvas.isConnected || !this.canvas.getClientRects().length) { this.prev = now; return; }
      const t = (now - this.t0) / 1000;
      const dt = Math.min(0.05, (now - (this.prev || now)) / 1000);
      this.prev = now;
      if (!this.drag) {
        this.yaw += this.vyaw; this.pitch += this.vpitch;
        this.vyaw *= 0.94; this.vpitch *= 0.94;
        if (this.opts.autoRotate && now - this.lastInteract > 3500) this.yaw += dt * (this.mode === 'scan' ? 0.6 : 0.18);
      }
      this.pitch = Math.max(-1.3, Math.min(1.3, this.pitch));
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      const model = this.model();
      this.modelM = model;
      const proj = m4.persp(0.62, this.canvas.width / this.canvas.height, 0.1, 20);
      const mvp = m4.mul(proj, m4.mul(m4.trans(0, 0, -this.dist), model));
      gl.useProgram(this.prog);
      this.bindAttrs(this.brainAttrs);
      gl.disable(gl.BLEND);
      gl.depthMask(true);
      gl.uniformMatrix4fv(this.u.uMVP, false, mvp);
      gl.uniformMatrix4fv(this.u.uModel, false, m4.mul(m4.trans(0, 0, 0), model));
      gl.uniform1f(this.u.uMix, this.mix);
      gl.uniform1f(this.u.uTime, t);
      gl.uniform1f(this.u.uMode, this.mode === 'data' ? 2 : this.mode === 'scan' ? 1 : 0);
      gl.uniform1f(this.u.uHover, this.hover);
      gl.uniform1f(this.u.uScan, this.mode === 'scan' ? 1 : 0);
      gl.uniform1f(this.u.uHlOn, this.hlOn ? 1 : 0);
      gl.uniform3fv(this.u.uHlColor, this.hlColor);
      gl.drawElements(gl.TRIANGLES, this.mesh.faces.length, gl.UNSIGNED_SHORT, 0);
      // particules (additives, masquées par le cerveau grâce au test de profondeur)
      const target = this.mode === 'scan' ? 0.8 : this.mode === 'data' ? this.energyTarget : 0.4;
      this.energy += (target - this.energy) * Math.min(1, dt * 3);
      const bt = (now - this.burstT) / 1000;
      const burst = bt < 0 ? 0 : bt < 0.18 ? bt / 0.18 : Math.exp(-(bt - 0.18) * 1.6);
      gl.useProgram(this.pprog);
      this.bindAttrs(this.partAttrs);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.depthMask(false);
      gl.uniformMatrix4fv(this.pu.uMVP, false, mvp);
      gl.uniform1f(this.pu.uTime, t);
      gl.uniform1f(this.pu.uScan, this.mode === 'scan' ? 1 : 0);
      gl.uniform1f(this.pu.uEnergy, this.energy);
      gl.uniform1f(this.pu.uBurst, burst);
      gl.uniform1f(this.pu.uDpr, Math.min(window.devicePixelRatio || 1, 2));
      gl.uniform1f(this.pu.uH, this.canvas.height);
      gl.drawArrays(gl.POINTS, 0, this.nPart);
      gl.depthMask(true);
    }

    bindAttrs(list) {
      const gl = this.gl;
      const want = new Set(list.map((a) => a.loc));
      for (const loc of this.enabled) if (!want.has(loc)) gl.disableVertexAttribArray(loc);
      for (const a of list) {
        gl.bindBuffer(gl.ARRAY_BUFFER, a.b);
        gl.enableVertexAttribArray(a.loc);
        gl.vertexAttribPointer(a.loc, a.size, a.type, a.norm, 0, 0);
      }
      this.enabled = want;
    }

    /** Onde de choc : les particules jaillissent puis reviennent (révélation du score). */
    pulse() { this.burstT = performance.now(); }

    // ---------------------------------------------------------------- données
    setMode(mode) { this.mode = mode; }

    /** u8 : (T × 20 484) octets, 128 ≈ moyenne. */
    setData(u8, nv) {
      this.data = u8;
      this.nv = nv || this.mesh?.nv || 20484;
      this.T = Math.floor(u8.length / this.nv);
      this.ta = -1; this.tb = -1;
      this.mode = 'data';
      this.setTime(0);
    }

    clearData() { this.data = null; this.mode = 'idle'; }

    /** t en secondes (fractionnaire) : interpole la seconde courante et la suivante. */
    setTime(t) {
      this.lastT = t;
      if (!this.data || !this.gl || this.lost) return;
      const T = this.T;
      const a = Math.max(0, Math.min(T - 1, Math.floor(t)));
      const b = Math.min(T - 1, a + 1);
      const gl = this.gl;
      if (a !== this.ta) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.aActA);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.data.subarray(a * this.nv, (a + 1) * this.nv));
        this.ta = a;
      }
      if (b !== this.tb) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.aActB);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.data.subarray(b * this.nv, (b + 1) * this.nv));
        this.tb = b;
      }
      const f = t - a;
      this.mix = f * f * (3 - 2 * f);
      this.cur = this.data.subarray(a * this.nv, (a + 1) * this.nv);
      if (a !== this.ea) {
        this.ea = a;
        let sum = 0;
        const row = this.cur;
        for (let i = 0; i < row.length; i += 4) sum += Math.max(0, row[i] - 128);
        this.energyTarget = Math.max(0.12, Math.min(1, (sum / (row.length / 4) / 127) * 5));
      }
    }

    /** Met en lumière les régions qui alimentent un indice (poids |w| normalisés), ou rien (null). */
    highlightIndex(key, color) {
      this.hlKey = key;
      if (!this.mesh || !this.gl || this.lost) return;
      const gl = this.gl;
      const nv = this.mesh.nv;
      const hl = new Uint8Array(nv);
      if (key) {
        const ws = this.mesh.rois.map((r) => Math.abs(r.indices[key] || 0));
        const mx = Math.max(...ws, 1e-6);
        for (let i = 0; i < nv; i++) {
          const r = this.mesh.roi[i];
          if (r) hl[i] = Math.round((ws[r - 1] / mx) * 255);
        }
        if (color) this.hlColor = color;
      }
      this.hlOn = !!key;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.aHl);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, hl);
    }

    view(name) {
      const v = { left: [-1.57, 0.1], right: [1.57, 0.1], top: [0, 1.25], front: [Math.PI, 0.1], back: [0, 0.1], three: [-1.15, 0.18] }[name];
      if (!v) return;
      this.lastInteract = performance.now();
      const [ty, tp] = v;
      const y0 = this.yaw, p0 = this.pitch, t0 = performance.now();
      let dy = ((ty - y0) % (2 * Math.PI) + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
      const step = (now) => {
        const k = Math.min(1, (now - t0) / 700);
        const e = 1 - Math.pow(1 - k, 3);
        this.yaw = y0 + dy * e; this.pitch = p0 + (tp - p0) * e;
        this.lastInteract = now;
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }

    // ---------------------------------------------------------------- interaction
    bindInput() {
      const c = this.canvas;
      c.style.touchAction = 'none';
      const pts = new Map();
      let pinch0 = 0, dist0 = 0;
      c.addEventListener('pointerdown', (e) => {
        c.setPointerCapture(e.pointerId);
        pts.set(e.pointerId, [e.clientX, e.clientY]);
        this.drag = true; this.moved = 0;
        this.vyaw = this.vpitch = 0;
        if (pts.size === 2) { const [a, b] = [...pts.values()]; pinch0 = Math.hypot(a[0] - b[0], a[1] - b[1]); dist0 = this.dist; }
      });
      c.addEventListener('pointermove', (e) => {
        if (pts.has(e.pointerId)) {
          const [px, py] = pts.get(e.pointerId);
          pts.set(e.pointerId, [e.clientX, e.clientY]);
          if (pts.size === 2 && this.opts.zoom) {
            const [a, b] = [...pts.values()];
            const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
            this.dist = Math.max(2.2, Math.min(6, dist0 * pinch0 / Math.max(d, 1)));
          } else {
            const dx = e.clientX - px, dy = e.clientY - py;
            this.moved += Math.abs(dx) + Math.abs(dy);
            this.vyaw = dx * 0.008; this.vpitch = dy * 0.008;
            this.yaw += this.vyaw; this.pitch += this.vpitch;
          }
          this.lastInteract = performance.now();
          return;
        }
        if (this.opts.hover) this.queuePick(e);
      });
      const end = (e) => {
        pts.delete(e.pointerId);
        if (!pts.size) {
          this.drag = false;
          if (this.moved < 4 && this.opts.hover) { this.pick(e); this.emit('select', this.hoverInfo); }
        }
      };
      c.addEventListener('pointerup', end);
      c.addEventListener('pointercancel', end);
      c.addEventListener('pointerleave', () => { if (!this.drag) { this.hover = 0; this.hoverInfo = null; this.emit('hover', null); } });
      c.addEventListener('dblclick', () => this.view('three'));
      if (this.opts.zoom) {
        c.addEventListener('wheel', (e) => {
          e.preventDefault();
          this.dist = Math.max(2.2, Math.min(6, this.dist * Math.exp(e.deltaY * 0.0012)));
          this.lastInteract = performance.now();
        }, { passive: false });
      }
    }

    queuePick(e) {
      this.pendingPick = e;
      if (this.pickRaf) return;
      this.pickRaf = requestAnimationFrame(() => { this.pickRaf = 0; this.pick(this.pendingPick); });
    }

    /** Lancer de rayon CPU (repère du modèle) -> région survolée. */
    pick(e) {
      if (!this.mesh || !this.modelM) return;
      const r = this.canvas.getBoundingClientRect();
      const nx = ((e.clientX - r.left) / r.width) * 2 - 1;
      const ny = -(((e.clientY - r.top) / r.height) * 2 - 1);
      const f = 1 / Math.tan(0.31), aspect = r.width / r.height;
      // rayon en espace caméra, puis en espace modèle (modèle = rotation pure : inverse = transposée)
      const dir = [nx * aspect / f, ny / f, -1];
      const org = [0, 0, this.dist];
      const M = this.modelM;
      const inv = (v) => [M[0] * v[0] + M[1] * v[1] + M[2] * v[2], M[4] * v[0] + M[5] * v[1] + M[6] * v[2], M[8] * v[0] + M[9] * v[1] + M[10] * v[2]];
      const o = inv(org), d = inv(dir);
      const { pos, faces, roi } = this.mesh;
      let best = Infinity, bestV = -1;
      for (let i = 0; i < faces.length; i += 3) {
        const a = faces[i] * 3, b = faces[i + 1] * 3, c = faces[i + 2] * 3;
        const e1x = pos[b] - pos[a], e1y = pos[b + 1] - pos[a + 1], e1z = pos[b + 2] - pos[a + 2];
        const e2x = pos[c] - pos[a], e2y = pos[c + 1] - pos[a + 1], e2z = pos[c + 2] - pos[a + 2];
        const px = d[1] * e2z - d[2] * e2y, py = d[2] * e2x - d[0] * e2z, pz = d[0] * e2y - d[1] * e2x;
        const det = e1x * px + e1y * py + e1z * pz;
        if (Math.abs(det) < 1e-9) continue;
        const id = 1 / det;
        const tx = o[0] - pos[a], ty = o[1] - pos[a + 1], tz = o[2] - pos[a + 2];
        const u = (tx * px + ty * py + tz * pz) * id;
        if (u < 0 || u > 1) continue;
        const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
        const v = (d[0] * qx + d[1] * qy + d[2] * qz) * id;
        if (v < 0 || u + v > 1) continue;
        const t = (e2x * qx + e2y * qy + e2z * qz) * id;
        if (t > 0 && t < best) { best = t; bestV = faces[i + (u > v ? 1 : 2)]; }
      }
      const rid = bestV >= 0 ? roi[bestV] : 0;
      this.hover = rid;
      this.canvas.style.cursor = bestV >= 0 ? (rid ? 'pointer' : 'grab') : 'grab';
      const info = bestV < 0 ? null : {
        vertex: bestV,
        hemi: bestV < this.mesh.hemi_split ? 'gauche' : 'droit',
        roi: rid ? this.mesh.rois[rid - 1] : null,
        value: this.cur && this.mode === 'data' ? this.cur[bestV] : null,
        x: e.clientX - r.left, y: e.clientY - r.top,
      };
      this.hoverInfo = info;
      this.emit('hover', info);
    }
  }

  window.Brain3D = Brain3D;
  window.Brain3D.loadMesh = loadMesh;
})();
