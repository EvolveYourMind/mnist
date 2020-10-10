import { Matrix, rand, zip } from "./util";

const logistic: (x: number) => number = x => 1 / (1 + Math.E ** (-x));
const derlogistic = (y: number) => y * (1 - y);

const tanh: (x: number) => number = Math.tanh
const dtanh = (y: number) => 1 - y ** 2



export class NeuralNetwork {
	weights: Matrix[];
	biases: Matrix[];
	shape: number[];
	act: typeof logistic;
	dact: typeof derlogistic;
	lr = 0.1;
	constructor(shape: number[]) {
		this.shape = shape;
		this.act = logistic;
		this.dact = derlogistic;
		this.weights = shape.slice(1).map((sz, i) => new Matrix(sz, shape[i]).map(rand));
		this.biases = shape.slice(1).map(sz => new Matrix(sz, 1).map(rand));
	}
	serialize() {
		return {
			weights: this.weights.map(x => x.serialize())
			, biases: this.biases.map(x => x.serialize())
			, shape: this.shape
		};
	}
	static load(data: ReturnType<NeuralNetwork["serialize"]>) {
		const res = new NeuralNetwork(data.shape);
		res.weights = data.weights.map(Matrix.load);
		res.biases = data.biases.map(Matrix.load);
		return res;
	}
	guess(input: number[]) {
		return zip(this.weights, this.biases)
			.reduce((r, [ws, bs]) => ws.mul(r).add(bs).map(this.act)
				, Matrix.fromArray(input)
			).data;
	}
	train(input: number[], target: number[]) {
		const allOutputs = zip(this.weights, this.biases)
			.reduce((r, [weights, biases]) => r.concat(
				weights
					.mul(r[r.length - 1])
					.add(biases)
					.map(this.act)
			), [Matrix.fromArray(input)])
		let errors = Matrix.fromArray(target).sub(allOutputs[allOutputs.length - 1]);
		for(let i = allOutputs.length - 1; i >= 1; i--) {
			const grad = allOutputs[i].map(this.dact).hamMul(errors).mul(this.lr);
			const deltas = grad.mul(allOutputs[i - 1].T());
			this.weights[i - 1] = this.weights[i - 1].add(deltas);
			this.biases[i - 1] = this.biases[i - 1].add(grad);
			errors = this.weights[i - 1].T().mul(errors);
		}
		return {
			output: allOutputs[allOutputs.length - 1],
			error: Matrix.fromArray(target).sub(allOutputs[allOutputs.length - 1]).map(v => v ** 2).sum()
		};
	}
}