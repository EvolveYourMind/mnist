import { add, avg, hmul, Matrix, sub, sum, zip } from "./Matrix";
const phi = (x: number) => 1 / (1 + Math.exp(-x))
const dphi = (y: number) => y * (1 - y)

export class NeuralNetwork {
	weights: Matrix[]
	biases: number[][]

	constructor(shape: number[]) {
		this.weights = zip(shape, shape.slice(1)).map(Matrix.random);
		this.biases = shape.slice(1).map(x => Array.from({ length: x }, _ => 2 * (Math.random() - 1)))
	}
	forward(x: number[], d: number[], layer = 0): { errSignal: number[], deltaW: Matrix, deltaB: number[], y: number[] }[] {
		const ws = this.weights[layer];
		const y = add(Matrix.vmv(x, ws.T()), this.biases[layer]).map(phi);
		const dy = y.map(dphi);
		if(layer === this.weights.length - 1) {
			const errSignal = hmul(sub(d, y), dy);
			const deltaW = ws.map((_, i, j) => errSignal[j] * x[i]);
			const deltaB = errSignal.map(er => er);
			return [{ errSignal, deltaW, deltaB, y }];
		} else {
			const next = this.forward(y, d, layer + 1);
			const errSignal = dy.map((dyj, j) => dyj * sum(next[0].errSignal.map((dk, k) => dk * this.weights[layer + 1].data[j][k])))
			const deltaW = ws.map((_, i, j) => errSignal[j] * x[i]);
			const deltaB = errSignal.map(er => er);
			return [{ errSignal, deltaW, deltaB, y: next[0].y }, ...next];
		}
	}
	train(xs: number[][], ys: number[][], lr: number): number[][] {
		const forwards = zip(xs, ys).map(([x, y]) => this.forward(x, y));
		for(let l = 0; l < this.weights.length; l++) {
			const layerDeltas = forwards.map(delta => delta[l]);
			this.weights[l] = this.weights[l].add(Matrix.avg(layerDeltas.map(d => d.deltaW)).map(v => v * lr));
			this.biases[l] = add(this.biases[l], layerDeltas.map(d => d.deltaB).reduce((acc, v) => add(acc, v)).map(v => v / xs.length * lr));
		}
		return forwards.map(f => f[0].y);
	}
	fit(xs: number[][], ys: number[][], epochs: number, lr: number, batchSize: number) {
		for(let i = 0; i < epochs; i++) {
			for(let j = 0; j < Math.floor(xs.length / batchSize); j++) {
				const batch_xs = xs.slice(j * batchSize, j * batchSize + batchSize);
				const batch_ys = ys.slice(j * batchSize, j * batchSize + batchSize);
				const preds = this.train(batch_xs, batch_ys, lr);
				const res = avg(zip(batch_ys, preds).map(([y, _y]) => avg(sub(y, _y).map(v => v ** 2))));
				console.log("Epoch ", i + "/" + epochs, "Loss", res);
			}
		}
	}
	guess(x: number[], layer = 0): number[] {
		const ws = this.weights[layer];
		const y = add(Matrix.vmv(x, ws.T()), this.biases[layer]).map(phi);
		return layer === this.weights.length - 1 ? y : this.guess(y, layer + 1);
	}
}