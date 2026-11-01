/** classes COCO (80) usadas pelo YOLOv8n */
export const COCO = [
  'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
  'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
  'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
  'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
  'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
  'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
  'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
  'remote','keyboard','cell phone','microwave','oven','toaster','sink',
  'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush',
];

const PT_BR = {
  person:'pessoa', bicycle:'bicicleta', car:'carro', motorcycle:'moto', airplane:'avião',
  bus:'ônibus', train:'trem', truck:'caminhão', boat:'barco', 'traffic light':'semáforo',
  'fire hydrant':'hidrante', 'stop sign':'placa de pare', 'parking meter':'parquímetro',
  bench:'banco', bird:'pássaro', cat:'gato', dog:'cachorro', horse:'cavalo', sheep:'ovelha',
  cow:'vaca', elephant:'elefante', bear:'urso', zebra:'zebra', giraffe:'girafa',
  backpack:'mochila', umbrella:'guarda-chuva', handbag:'bolsa', tie:'gravata',
  suitcase:'mala', frisbee:'frisbee', skis:'esquis', snowboard:'snowboard',
  'sports ball':'bola', kite:'pipa', 'baseball bat':'taco', 'baseball glove':'luva',
  skateboard:'skate', surfboard:'prancha', 'tennis racket':'raquete', bottle:'garrafa',
  'wine glass':'taça', cup:'copo', fork:'garfo', knife:'faca', spoon:'colher', bowl:'tigela',
  banana:'banana', apple:'maçã', sandwich:'sanduíche', orange:'laranja', broccoli:'brócolis',
  carrot:'cenoura', 'hot dog':'cachorro-quente', pizza:'pizza', donut:'rosquinha', cake:'bolo',
  chair:'cadeira', couch:'sofá', 'potted plant':'planta', bed:'cama', 'dining table':'mesa',
  toilet:'vaso', tv:'TV', laptop:'notebook', mouse:'mouse', remote:'controle',
  keyboard:'teclado', 'cell phone':'celular', microwave:'micro-ondas', oven:'forno',
  toaster:'torradeira', sink:'pia', refrigerator:'geladeira', book:'livro', clock:'relógio',
  vase:'vaso', scissors:'tesoura', 'teddy bear':'ursinho', 'hair drier':'secador',
  toothbrush:'escova de dente', face:'rosto', head:'cabeça',
};

export function labelPt(en) { return PT_BR[en] || en; }

/** cor estável por classe */
export function colorFor(name) {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return `hsl(${h % 360} 85% 62%)`;
}
