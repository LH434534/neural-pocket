# Neural Pocket — IA on-device no navegador

App mobile (PWA) que roda **modelos ONNX reais obtidos de repositórios do GitHub**
diretamente no seu celular, via WebAssembly. Não há servidor de inferência:
nenhuma imagem ou desenho sai do aparelho.

**Publicado em:** https://lh434534.github.io/neural-pocket/

## O que faz

| Tela | Modelos | Entrada |
|---|---|---|
| **Desenhar** | MNIST CNN + MNIST Regressão Logística | desenho de um dígito a dedo |
| **Visão** | YOLOv8n (80 classes), YOLOv8-face, YOLOv8-head | foto da câmera ou galeria |
| **Modelos** | todos os acima | carregamento sob demanda com barra de progresso |

Funciona offline depois do primeiro acesso (service worker + cache do app shell).

## Modelos — origem real

| Modelo | Repositório de origem | Arquivo | Tamanho |
|---|---|---|---|
| MNIST CNN | `gespriella/Mnist-ONNX` | `MnistONNX/MnistCNN.onnx` | 26 KB |
| MNIST Regressão Logística | `gespriella/Mnist-ONNX` | `MnistONNX/MnistLR.onnx` | 946 KB |
| YOLOv8n (80 classes COCO) | `senko-sleep/detection` | `hidden/yolov8n.onnx` | 12,3 MB |
| YOLOv8-face | `senko-sleep/detection` | `hidden/yolov8_face.onnx` | 11,7 MB |
| YOLOv8-head | `senko-sleep/detection` | `hidden/yolov8n_head.onnx` | 11,7 MB |

Runtime: ONNX Runtime Web 1.19.2 (backend WASM, 1 thread).

## Contratos verificados

Cada modelo foi inspecionado e executado com `onnxruntime` (CPU) **antes** de a
interface ser escrita — as entradas/saídas abaixo são as reais, não presumidas:

```
mnist-cnn   Input3   [1,1,28,28] float  → Plus214_Output_0 [1,10]     (valores em [0,1])
mnist-lr    input    [1,784]     float  → output_label int64 + seq(map) (pixels crus [0,255])
yolov8n     images   [1,3,640,640]      → output0 [1,84,8400]
yolov8-face images   [1,3,640,640]      → output0 [1,5,8400]
yolov8-head images   [1,3,640,640]      → output0 [1,5,8400]
```

Dois detalhes que só apareceram nos testes e que mudam o resultado:

1. **MNIST-LR exige pixels em 0–255.** Com 0–1 ou 0–16 o modelo erra sistematicamente.
2. **Os scores do YOLOv8 já vêm como probabilidade.** Aplicar sigmoid gera
   centenas de falsos positivos (1.560 em vez de 5 com a mesma imagem).

Também: o ONNX Runtime Web **não consegue ler** a segunda saída do modelo sklearn
(`seq(map)`). Por isso a chamada pede explicitamente só `output_label`.

## Verificação executada

Testes em Chromium headless (Playwright), viewport 390×844, antes de publicar:

| Teste | Resultado |
|---|---|
| App inicializa | ONNX Runtime 1.19.2 · wasm · 1 thread |
| MNIST CNN sobre dígito "1" desenhado | **1** com 91,1% de confiança |
| MNIST Regressão Logística, mesmo desenho | **1** (rótulo direto) |
| YOLOv8n sobre `bus.jpg` (imagem real) | **pessoa ×4, ônibus ×1** em 1.064 ms |
| YOLOv8-face sobre a mesma imagem | **rosto ×2** em 632 ms |
| 5 modelos listados na aba Modelos | 5 de 5 |
| Manifest + service worker | HTTP 200 · 1 registro |
| Erros de console | **nenhum** |
| Layout em 375 / 390 / 412 px | sem rolagem horizontal, sem estouro |
| Integridade do publicado | 20/20 blobs idênticos aos testados |

## Rodando localmente

```bash
cd ai-mobile-app
python3 -m http.server 8080
# abra http://127.0.0.1:8080/
```

Exige HTTP (não `file://`) porque usa ES modules e WASM.

## Limitações conhecidas

- O primeiro uso de cada detector baixa ~12 MB; depois fica em cache.
- A detecção leva ~0,6–1,1 s por imagem em WASM de 1 thread.
- O modelo de regressão logística devolve apenas o rótulo, sem distribuição de
  probabilidades (limitação do tipo `seq(map)` no ONNX Runtime Web).
- Neste ambiente de build, `github.io` e `huggingface.co` estão bloqueados por
  política de saída, portanto o acesso ao site não pôde ser verificado de fora
  do sandbox. O GitHub Pages reporta `status: built`.
