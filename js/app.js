/**
 * app.js — orquestração da interface.
 */

import { MODELS, loadModel, isLoaded, humanSize, ort } from './models.js';
import { createDrawPad, predictDigit } from './draw.js';
import { detect, render, loadImageFromFile, className } from './vision.js';
import { labelPt, colorFor } from './coco.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

/* ---------------- toast ---------------- */
let toastTimer;
function toast(msg, kind = '') {
  const el = $('#toast');
  el.textContent = msg;
  el.className = 'toast ' + kind;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
}

/* ---------------- boot ---------------- */
function setBoot(p, text) {
  $('#boot-bar').style.width = Math.round(p * 100) + '%';
  if (text) $('#boot-text').textContent = text;
}

async function boot() {
  setBoot(0.15, 'preparando runtime wasm…');
  try {
    await loadModel('mnist-cnn', (p, t) => setBoot(0.15 + p * 0.8, t));
  } catch (e) {
    setBoot(1, 'falha: ' + e.message);
    toast('Não foi possível carregar o modelo inicial: ' + e.message, 'err');
  }
  setBoot(1, 'pronto');

  const thr = ort.env.wasm.numThreads;
  $('#hw-badge').textContent = `ONNX Runtime ${ort.env.versions?.web || '1.19.2'} · wasm · ${thr} thread${thr > 1 ? 's' : ''}`;

  setTimeout(() => {
    $('#splash').classList.add('gone');
    $('#app').classList.remove('hidden');
    renderModelList();
  }, 260);

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register(new URL('sw.js', location.href).href).catch(() => {});
  }
}

/* ---------------- tabs ---------------- */
$$('.tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    $$('.tab').forEach((b) => b.classList.toggle('active', b === btn));
    $$('.view').forEach((v) => v.classList.toggle('active', v.id === 'view-' + btn.dataset.view));
  });
});

$('#btn-info')?.addEventListener('click', () => {
  $$('.tab').forEach((b) => b.classList.toggle('active', b.dataset.view === 'models'));
  $$('.view').forEach((v) => v.classList.toggle('active', v.id === 'view-models'));
});

/* ---------------- desenho ---------------- */
const pad = createDrawPad($('#draw-canvas'), $('#mini-canvas'), () => {
  $('#canvas-hint').classList.add('gone');
});

$('#btn-clear').addEventListener('click', () => {
  pad.clear();
  $('#draw-results').hidden = true;
  $('#canvas-hint').classList.remove('gone');
});

function predCard(r) {
  const top = r.probs
    .map((p, i) => ({ i, p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 3);
  const max = Math.max(...r.probs, 1e-6);
  const rows = r.probs.map((p, i) => `
    <div class="bar-row ${i === r.label ? 'top' : ''}">
      <span>${i}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(p / max) * 100}%"></div></div>
    </div>`).join('');

  const verdict = r.exact
    ? `Dígito ${r.label} · ${(r.probs[r.label] * 100).toFixed(1)}% de confiança`
    : `Dígito ${r.label} · rótulo direto (modelo não expõe distribuição)`;
  const detail = r.exact
    ? `${r.ms.toFixed(0)} ms · top-3: ${top.map((t) => `${t.i} (${(t.p * 100).toFixed(0)}%)`).join(', ')}`
    : `${r.ms.toFixed(0)} ms · saída seq(map) não legível no ONNX Runtime Web`;

  return `
    <div class="pred-card">
      <div class="pred-digit">${r.label}</div>
      <div class="pred-body">
        <div class="pred-name">${r.name}</div>
        <div class="pred-verdict">${verdict}</div>
        <div class="pred-conf">${detail}</div>
        <div class="bars">${rows}</div>
      </div>
    </div>`;
}

$('#btn-predict').addEventListener('click', async () => {
  if (!pad.hasInk()) { toast('Desenhe um dígito primeiro.'); return; }
  const btn = $('#btn-predict');
  const spin = btn.querySelector('.spinner');
  const label = btn.querySelector('.btn-label');
  btn.disabled = true; spin.hidden = false; label.textContent = 'Reconhecendo…';

  const box = $('#draw-results');
  box.hidden = false;
  box.innerHTML = '';

  try {
    const raw = pad.grab();
    for (const id of ['mnist-cnn', 'mnist-lr']) {
      const r = await predictDigit(id, raw, (p, t) => { label.textContent = `${MODELS[id].name}: ${t}`; });
      box.insertAdjacentHTML('beforeend', predCard(r));
    }
  } catch (e) {
    toast('Falha na inferência: ' + e.message, 'err');
  } finally {
    btn.disabled = false; spin.hidden = true; label.textContent = 'Reconhecer';
  }
});

/* ---------------- visão ---------------- */
let srcCanvas = null;
let visionModel = 'yolov8n';

$$('#vision-seg .seg-btn').forEach((b) => {
  b.addEventListener('click', () => {
    $$('#vision-seg .seg-btn').forEach((x) => x.classList.toggle('active', x === b));
    visionModel = b.dataset.model;
    $('#vision-tag').textContent = MODELS[visionModel].name.replace(/ — .*/, '');
    $('#vision-results').innerHTML = '';
  });
});

async function useFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) { toast('Selecione uma imagem.', 'err'); return; }
  try {
    srcCanvas = await loadImageFromFile(file);
    $('#dz-empty').classList.add('hidden');
    $('#preview-shell').classList.remove('hidden');
    render($('#preview-canvas'), srcCanvas, []);
    $('#btn-detect').disabled = false;
    $('#vision-results').innerHTML = '';
  } catch (e) {
    toast('Não foi possível ler a imagem: ' + e.message, 'err');
  }
}

$$('[data-pick]').forEach((b) => b.addEventListener('click', (e) => {
  e.stopPropagation();
  $(b.dataset.pick === 'cam' ? '#cam-input' : '#file-input').click();
}));
$('#dropzone').addEventListener('click', () => $('#file-input').click());
$('#file-input').addEventListener('change', (e) => useFile(e.target.files[0]));
$('#cam-input').addEventListener('change', (e) => useFile(e.target.files[0]));

$('#btn-detect').addEventListener('click', async () => {
  if (!srcCanvas) return;
  const btn = $('#btn-detect');
  const spin = btn.querySelector('.spinner');
  const label = btn.querySelector('.btn-label');
  btn.disabled = true; spin.hidden = false; label.textContent = 'Detectando…';

  try {
    const { dets, ms, outShape } = await detect(visionModel, srcCanvas,
      (p, t) => { label.textContent = t; });
    render($('#preview-canvas'), srcCanvas, dets);

    const counts = new Map();
    for (const d of dets) counts.set(d.cls, (counts.get(d.cls) || 0) + 1);
    const chips = [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([c, n]) => `<span class="chip"><i style="background:${colorFor(c)}"></i>${labelPt(c)} <b>×${n}</b></span>`)
      .join('');

    $('#vision-results').innerHTML = dets.length
      ? `<div class="chips">${chips}</div>
         <div class="pred-conf">${dets.length} detecção(ões) em ${ms.toFixed(0)} ms · saída [${outShape.join('×')}] · confiança mínima 25%</div>`
      : `<div class="empty-note">Nada encontrado com confiança acima de 25%.</div>`;
    toast(`${dets.length} detecção(ões) em ${ms.toFixed(0)} ms`);
  } catch (e) {
    toast('Falha: ' + e.message, 'err');
  } finally {
    btn.disabled = false; spin.hidden = true; label.textContent = 'Detectar';
  }
});

/* ---------------- lista de modelos ---------------- */
function renderModelList() {
  const host = $('#model-list');
  host.innerHTML = Object.values(MODELS).map((m) => `
    <div class="mrow" data-id="${m.id}">
      <div class="micon">${m.emoji}</div>
      <div class="mbody">
        <div class="mtitle">${m.name}</div>
        <div class="mdesc">${m.desc}</div>
        <div class="mmeta">
          <span class="pill">${humanSize(m.size)}</span>
          <span class="pill">${m.io}</span>
          <span class="pill">${m.producer}</span>
          <span class="pill">github.com/${m.repo}</span>
        </div>
        <div class="mprog"><i></i></div>
      </div>
      <button class="mload ${isLoaded(m.id) ? 'done' : ''}" data-load="${m.id}">
        ${isLoaded(m.id) ? 'pronto' : 'carregar'}
      </button>
    </div>`).join('');

  $$('[data-load]').forEach((b) => b.addEventListener('click', async () => {
    const id = b.dataset.load;
    const row = b.closest('.mrow');
    const bar = row.querySelector('.mprog i');
    b.disabled = true;
    try {
      await loadModel(id, (p, t) => {
        bar.style.width = (p * 100) + '%';
        b.textContent = p < 1 ? Math.round(p * 100) + '%' : 'pronto';
      });
      b.classList.add('done');
      b.textContent = 'pronto';
      toast(MODELS[id].name + ' carregado.');
    } catch (e) {
      b.textContent = 'erro';
      toast('Falha ao carregar: ' + e.message, 'err');
    } finally {
      b.disabled = false;
    }
  }));
}

boot();
