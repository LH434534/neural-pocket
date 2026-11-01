/**
 * vision.js — detecção de objetos com YOLOv8 (ONNX) no navegador.
 *
 * Pipeline validado com onnxruntime + imagem real (bus.jpg → person 0.90, bus 0.84):
 *   letterbox 640 → RGB /255 → NCHW → scores JÁ vêm como probabilidade
 *   (aplicar sigmoid gera centenas de falsos positivos) → NMS por classe (IoU 0.45).
 */

import { ort, loadModel, MODELS } from './models.js';
import { COCO, colorFor } from './coco.js';

const SIZE = 640;
const CONF_T = 0.25;
const IOU_T = 0.45;

export function className(modelId, idx) {
  const m = MODELS[modelId];
  if (m.singleClass) return m.singleClass;
  return COCO[idx] || `cls${idx}`;
}

/** Letterbox 640 mantendo proporção; devolve canvas + escala + padding. */
function letterbox(img) {
  const w = img.width, h = img.height;
  const s = Math.min(SIZE / w, SIZE / h);
  const nw = Math.round(w * s), nh = Math.round(h * s);
  const c = document.createElement('canvas');
  c.width = SIZE; c.height = SIZE;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(img, Math.round((SIZE - nw) / 2), Math.round((SIZE - nh) / 2), nw, nh);
  return { canvas: c, scale: s, padX: Math.round((SIZE - nw) / 2), padY: Math.round((SIZE - nh) / 2) };
}

function iou(a, b) {
  const xx1 = Math.max(a[0], b[0]), yy1 = Math.max(a[1], b[1]);
  const xx2 = Math.min(a[2], b[2]), yy2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, xx2 - xx1) * Math.max(0, yy2 - yy1);
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

function nms(dets) {
  dets.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of dets) {
    let dup = false;
    for (const k of keep) if (k.cls === d.cls && iou(k.box, d.box) > IOU_T) { dup = true; break; }
    if (!dup) keep.push(d);
  }
  return keep;
}

export async function loadImageFromFile(file) {
  const bmp = await createImageBitmap(file);
  const c = document.createElement('canvas');
  c.width = bmp.width; c.height = bmp.height;
  c.getContext('2d').drawImage(bmp, 0, 0);
  return c;
}

/**
 * @param {string} modelId
 * @param {HTMLCanvasElement} srcCanvas  imagem original
 * @returns {Promise<{dets:Array, ms:number, outShape:number[]}>}
 */
export async function detect(modelId, srcCanvas, onProgress) {
  const sess = await loadModel(modelId, onProgress);

  const lb = letterbox(srcCanvas);
  const px = lb.canvas.getContext('2d', { willReadFrequently: true })
    .getImageData(0, 0, SIZE, SIZE).data;

  const x = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const o = i * 4;
    x[i] = px[o] / 255;                       // R
    x[SIZE * SIZE + i] = px[o + 1] / 255;     // G
    x[2 * SIZE * SIZE + i] = px[o + 2] / 255; // B
  }

  const t0 = performance.now();
  const res = await sess.run({ images: new ort.Tensor('float32', x, [1, 3, SIZE, SIZE]) });
  const ms = performance.now() - t0;

  const out = res.output0;                 // [1, C, 8400]
  const dims = out.dims;
  const C = dims[1], N = dims[2];
  const d = out.data;

  const dets = [];
  for (let i = 0; i < N; i++) {
    let bestScore = 0, bestCls = 0;
    for (let k = 4; k < C; k++) {
      const s = d[k * N + i];
      if (s > bestScore) { bestScore = s; bestCls = k - 4; }
    }
    if (bestScore < CONF_T) continue;

    const cx = d[0 * N + i], cy = d[1 * N + i];
    const bw = d[2 * N + i], bh = d[3 * N + i];

    // volta para as coordenadas da imagem original
    const ox1 = (cx - bw / 2 - lb.padX) / lb.scale;
    const oy1 = (cy - bh / 2 - lb.padY) / lb.scale;
    const ox2 = (cx + bw / 2 - lb.padX) / lb.scale;
    const oy2 = (cy + bh / 2 - lb.padY) / lb.scale;

    dets.push({
      box: [
        Math.max(0, ox1), Math.max(0, oy1),
        Math.min(srcCanvas.width, ox2), Math.min(srcCanvas.height, oy2),
      ],
      score: bestScore,
      cls: className(modelId, bestCls),
    });
  }

  const kept = nms(dets).slice(0, 60);
  for (const k of kept) k.color = colorFor(k.cls);
  return { dets: kept, ms, outShape: dims };
}

/** Desenha a imagem + caixas em um canvas de exibição. */
export function render(canvas, srcCanvas, dets) {
  const maxW = Math.min(520, canvas.parentElement?.clientWidth || 360);
  const scale = Math.min(1, maxW / srcCanvas.width);
  const w = Math.round(srcCanvas.width * scale), h = Math.round(srcCanvas.height * scale);
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(srcCanvas, 0, 0, w, h);

  for (const d of dets) {
    const [x1, y1, x2, y2] = d.box.map((v) => v * scale);
    const bw = x2 - x1, bh = y2 - y1;
    ctx.strokeStyle = d.color;
    ctx.lineWidth = Math.max(2, Math.round(w / 260) + 1);
    ctx.strokeRect(x1, y1, bw, bh);

    const label = `${d.cls} ${(d.score * 100).toFixed(0)}%`;
    ctx.font = `600 ${Math.max(10, Math.round(w / 32))}px -apple-system,sans-serif`;
    const tw = ctx.measureText(label).width + 10;
    const th = Math.max(16, Math.round(w / 26));
    const ty = y1 > th + 2 ? y1 - th : y1;
    ctx.fillStyle = d.color;
    ctx.fillRect(x1, ty, tw, th);
    ctx.fillStyle = '#0a0a14';
    ctx.fillText(label, x1 + 5, ty + th - 5);
  }
}
