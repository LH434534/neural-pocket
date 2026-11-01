/**
 * draw.js — bloco de desenho + reconhecimento de dígito (MNIST).
 *
 * Pré-processamento calibrado empiricamente com onnxruntime:
 *   • mnist-cnn  → entrada [1,1,28,28] normalizada para [0,1]
 *   • mnist-lr   → entrada [1,784] com pixels crus em [0,255] (qualquer outra
 *                  escala faz o modelo errar — verificado)
 */

import { ort, loadModel, MODELS } from './models.js';

export function createDrawPad(canvas, mini, onFirstStroke) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const mctx = mini.getContext('2d');
  let drawing = false, last = null, everDrawn = false;

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 17;

  const pos = (e) => {
    const r = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) * (canvas.width / r.width),
      y: (e.clientY - r.top) * (canvas.height / r.height),
    };
  };

  function start(e) {
    e.preventDefault();
    canvas.setPointerCapture?.(e.pointerId);
    drawing = true; last = pos(e);
    if (!everDrawn) { everDrawn = true; onFirstStroke?.(); }
  }
  function move(e) {
    if (!drawing) return;
    e.preventDefault();
    const p = pos(e);
    ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(p.x, p.y); ctx.stroke();
    last = p;
    paintMini();
  }
  function end() { drawing = false; }

  canvas.addEventListener('pointerdown', start);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('pointerleave', end);

  function paintMini() {
    mctx.clearRect(0, 0, 28, 28);
    mctx.imageSmoothingEnabled = true;
    mctx.drawImage(canvas, 0, 0, 28, 28);
  }

  return {
    clear() {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      mctx.clearRect(0, 0, 28, 28);
      everDrawn = false;
    },
    hasInk() {
      const d = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      for (let i = 0; i < d.length; i += 4) if (d[i] > 12) return true;
      return false;
    },
    /** 28×28 em cinza, como o modelo recebe */
    grab() {
      const off = document.createElement('canvas');
      off.width = 28; off.height = 28;
      const octx = off.getContext('2d', { willReadFrequently: true });
      octx.imageSmoothingEnabled = true;
      octx.drawImage(canvas, 0, 0, 28, 28);
      const px = octx.getImageData(0, 0, 28, 28).data;
      const raw = new Float32Array(784);
      for (let i = 0; i < 784; i++) raw[i] = px[i * 4]; // cinza: canal R basta
      return raw;
    },
  };
}

function softmax(a) {
  const m = Math.max(...a);
  const e = a.map((v) => Math.exp(v - m));
  const s = e.reduce((x, y) => x + y, 0);
  return e.map((v) => v / s);
}

/**
 * O modelo sklearn expõe as probabilidades como seq(map(int64,tensor(float))).
 * O ONNX Runtime Web não consegue ler esse tipo ("Reading data from non-tensor
 * typed value is not supported"), então pedimos SOMENTE output_label na chamada
 * (evita materializar o índice 1) e devolvemos um one-hot.
 */

/**
 * Roda um modelo de dígito.
 * @returns {Promise<{label:number, probs:number[], ms:number}>}
 */
export async function predictDigit(modelId, raw, onProgress) {
  const meta = MODELS[modelId];
  const sess = await loadModel(modelId, onProgress);
  const t0 = performance.now();
  let feeds, probs;

  if (modelId === 'mnist-cnn') {
    const x = new Float32Array(784);
    for (let i = 0; i < 784; i++) x[i] = raw[i] / 255;
    feeds = { Input3: new ort.Tensor('float32', x, [1, 1, 28, 28]) };
    const res = await sess.run(feeds);
    probs = softmax(Array.from(res.Plus214_Output_0.data));
  } else {
    feeds = { input: new ort.Tensor('float32', raw, [1, 784]) }; // já em 0–255
    // outputNames explícito: impede o runtime de tocar na saída seq(map)
    const res = await sess.run(feeds, ['output_label']);
    const label = Number(res.output_label.data[0]);
    probs = new Array(10).fill(0);
    if (Number.isFinite(label) && label >= 0 && label < 10) probs[label] = 1;
  }

  const ms = performance.now() - t0;
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  // exact = o modelo devolveu uma distribuição real de probabilidades
  return { label: best, probs, ms, name: meta.name, exact: modelId === 'mnist-cnn' };
}
