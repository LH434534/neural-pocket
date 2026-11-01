/**
 * models.js — registro dos modelos e carregamento com progresso real.
 *
 * Todos os pesos são arquivos ONNX públicos obtidos de repositórios do GitHub.
 * As entradas/saídas abaixo foram verificadas com onnxruntime (CPU) antes de
 * este arquivo ser escrito — ver README.md, seção "Contratos verificados".
 */

import * as ort from '../vendor/ort/ort.min.mjs';

// Resolve o caminho dos binários WASM independentemente de onde a página é servida
ort.env.wasm.wasmPaths = new URL('../vendor/ort/', location.href).href;
ort.env.wasm.numThreads = 1; // evita exigir COOP/COEP (SharedArrayBuffer)
ort.env.logLevel = 'error';

export const MODELS = {
  'mnist-cnn': {
    id: 'mnist-cnn',
    name: 'MNIST CNN',
    emoji: '✍️',
    task: 'mnist',
    file: 'models/mnist-cnn.onnx',
    repo: 'gespriella/Mnist-ONNX',
    path: 'MnistONNX/MnistCNN.onnx',
    producer: 'CNTK 2.5.1',
    size: 26454,
    desc: 'Convolucional treinada no MNIST. Recebe 28×28 em escala de cinza [0–1].',
    io: 'Input3 [1,1,28,28] → [1,10]',
    classes: ['0','1','2','3','4','5','6','7','8','9'],
  },
  'mnist-lr': {
    id: 'mnist-lr',
    name: 'MNIST Regressão Logística',
    emoji: '📈',
    task: 'mnist',
    file: 'models/mnist-lr.onnx',
    repo: 'gespriella/Mnist-ONNX',
    path: 'MnistONNX/MnistLR.onnx',
    producer: 'skl2onnx 1.7.1',
    size: 967702,
    desc: 'Regressão logística do scikit-learn exportada para ONNX. Espera pixels crus 0–255 achatados em 784.',
    io: 'input [1,784] → rótulo + probabilidades',
    classes: ['0','1','2','3','4','5','6','7','8','9'],
  },
  'yolov8n': {
    id: 'yolov8n',
    name: 'YOLOv8n — 80 objetos',
    emoji: '🔍',
    task: 'detect',
    file: 'models/yolov8n.onnx',
    repo: 'senko-sleep/detection',
    path: 'hidden/yolov8n.onnx',
    producer: 'Ultralytics YOLOv8n',
    size: 12851094,
    desc: 'Detector COCO com 80 classes: pessoas, veículos, animais, objetos do dia a dia.',
    io: 'images [1,3,640,640] → [1,84,8400]',
  },
  'yolov8-face': {
    id: 'yolov8-face',
    name: 'YOLOv8 — rostos',
    emoji: '🙂',
    task: 'detect',
    file: 'models/yolov8-face.onnx',
    repo: 'senko-sleep/detection',
    path: 'hidden/yolov8_face.onnx',
    producer: 'YOLOv8 (1 classe)',
    size: 12265666,
    desc: 'Detector especializado em rostos humanos.',
    io: 'images [1,3,640,640] → [1,5,8400]',
    singleClass: 'face',
  },
  'yolov8-head': {
    id: 'yolov8-head',
    name: 'YOLOv8 — cabeças',
    emoji: '🧠',
    task: 'detect',
    file: 'models/yolov8-head.onnx',
    repo: 'senko-sleep/detection',
    path: 'hidden/yolov8n_head.onnx',
    producer: 'YOLOv8 (1 classe)',
    size: 12265669,
    desc: 'Detector de cabeças — útil em cenas com pessoas parcialmente visíveis.',
    io: 'images [1,3,640,640] → [1,5,8400]',
    singleClass: 'head',
  },
};

const sessions = new Map();   // id -> InferenceSession
const inflight = new Map();   // id -> Promise

export function isLoaded(id) { return sessions.has(id); }
export function getSession(id) { return sessions.get(id); }

export function humanSize(b) {
  if (b >= 1073741824) return (b / 1073741824).toFixed(1) + ' GB';
  if (b >= 1048576) return (b / 1048576).toFixed(1) + ' MB';
  if (b >= 1024) return (b / 1024).toFixed(0) + ' KB';
  return b + ' B';
}

/**
 * Baixa o .onnx acompanhando bytes recebidos (barra de progresso real) e
 * cria a sessão de inferência.
 */
export async function loadModel(id, onProgress) {
  if (sessions.has(id)) { onProgress?.(1, 'já carregado'); return sessions.get(id); }
  if (inflight.has(id)) return inflight.get(id);

  const meta = MODELS[id];
  if (!meta) throw new Error('modelo desconhecido: ' + id);

  const p = (async () => {
    const url = new URL(meta.file, location.href).href;
    onProgress?.(0, 'baixando ' + humanSize(meta.size));

    const res = await fetch(url, { cache: 'force-cache' });
    if (!res.ok) throw new Error('HTTP ' + res.status + ' ao baixar ' + meta.file);

    const total = Number(res.headers.get('content-length')) || meta.size;
    let buf;
    if (res.body && typeof res.body.getReader === 'function') {
      const reader = res.body.getReader();
      const chunks = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.length;
        onProgress?.(Math.min(0.95, got / total), 'baixando ' + humanSize(got));
      }
      buf = new Uint8Array(got);
      let off = 0;
      for (const c of chunks) { buf.set(c, off); off += c.length; }
    } else {
      buf = new Uint8Array(await res.arrayBuffer());
    }

    onProgress?.(0.96, 'compilando grafo…');
    const sess = await ort.InferenceSession.create(buf, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    sessions.set(id, sess);
    onProgress?.(1, 'pronto');
    return sess;
  })();

  inflight.set(id, p);
  try { return await p; } finally { inflight.delete(id); }
}

export { ort };
