import { NeuralNetwork } from "./TFNeuralNetwork";


function fun() {
  const nn = new NeuralNetwork([2, 2, 1]);
  const batchSize = 100;
  for(let i = 0; i < 100000; i++) {
    let er = 0;
    let acc = 0;
    Array.from({ length: batchSize })
      .map((_) => ([Math.random(), Math.random()]))
      .forEach(([x1, x2]) => {
        const y = x1 ** 2 * x2 ** (1 / 2);
        const { output, error } = nn.train([x1, x2], [y]);
        er += error;
        acc += 1 - Math.abs(y - output.dataSync()[0]);
      });
    console.log({ averr: (er / batchSize).toFixed(4), avacc: (acc / batchSize).toFixed(4) })
  }

}
fun();
function xor() {
  const nn = new NeuralNetwork([2, 2, 1]);

  const data = [
    { in: [0, 0], out: [0] },
    { in: [1, 0], out: [1] },
    { in: [0, 1], out: [1] },
    { in: [1, 1], out: [0] }
  ];

  for(let i = 0; i < 100000; i++) {
    let er = 0;
    let acc = 0;
    data.forEach((example) => {
      const { output, error } = nn.train(example.in, example.out);
      er += error;
      acc += 1 - Math.abs(example.out[0] - output.dataSync()[0]);
    });
    console.log({ averr: (er / data.length).toFixed(4), avacc: (acc / data.length).toFixed(4) })
  }
}