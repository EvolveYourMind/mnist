import * as tf from "@tensorflow/tfjs-node";
import { Tensor } from "@tensorflow/tfjs-node";
import { SSL_OP_SSLEAY_080_CLIENT_DH_BUG } from "constants";

const zip = <T, U>(xs: T[], ys: U[]): [T, U][] => xs.map((x, i) => [x, ys[i]]);
type Activation = { act: (x: tf.Tensor) => tf.Tensor, der: (y: tf.Tensor) => tf.Tensor }

export class NeuralNetwork {
	weights: tf.Tensor[];
	biases: tf.Tensor[];
	shape: number[];
	activation: Activation;
	lr = 0.1;
	constructor(shape: number[]) {
		this.shape = shape;

		this.activation = {
			act: xs => xs.sigmoid()
			, der: ys => ys.mul(tf.ones(ys.shape).sub(ys))
		}
		this.weights = shape.slice(1).map((sz, i) => tf.randomUniform([sz, shape[i]]));
		this.biases = shape.slice(1).map(sz => tf.randomUniform([sz, 1]));
	}
	fit(inputs: number[][], targets: number[][], epochs: number, batchSize: number) {
		const data = zip(
			inputs.map(xs => tf.tensor(xs, [xs.length, 1]))
			, targets.map(xs => tf.tensor(xs, [xs.length, 1]))
		);
		for(let ep = 0; ep < epochs; ep++) {
			let cumulativeErr = 0;
			let acceptable = 0;
			let lowestError = Infinity;
			let epochAcceptable = 0;
			let epochError = 0;
			data.forEach(([xs, ys], i) => {
				const t = this.trainTens(xs, ys);
				cumulativeErr += t.error;
				epochError += t.error;
				lowestError = Math.min(lowestError, t.error);
				if(t.error < 0.1) {
					acceptable++;
					epochAcceptable++;
				}
				if((i + 1) % batchSize === 0) {
					console.log({
						ep
						, progress: (i / data.length).toFixed(2)
						, avgErr: cumulativeErr / batchSize
						, accuracy: acceptable / batchSize
						, lowestError
						, epochAcc: epochAcceptable / i
						, epochError: epochError / i
					});

					lowestError = Infinity;
					cumulativeErr = 0;
					acceptable = 0;
				}
			});
		}
	}
	private trainTens(inputTensor: Tensor, targetTensor: Tensor) {
		const allOutputs = zip(this.weights, this.biases)
			.reduce((r, [weights, biases]) => r.concat(
				weights
					.matMul(r[r.length - 1])
					.add(biases)
					.sigmoid()
			), [inputTensor])
		const output = allOutputs[allOutputs.length - 1];
		let errors = targetTensor.sub(output);
		for(let i = allOutputs.length - 1; i >= 1; i--) {
			const grad = this.activation.der(allOutputs[i]).mul(errors);
			const deltas = grad.matMul(allOutputs[i - 1].transpose());
			errors = this.weights[i - 1].transpose().matMul(errors);
			this.weights[i - 1] = this.weights[i - 1].add(deltas.mul(this.lr));
			this.biases[i - 1] = this.biases[i - 1].add(grad.mul(this.lr));
		}
		return {
			output: allOutputs[allOutputs.length - 1],
			error: targetTensor.sub(allOutputs[allOutputs.length - 1]).square().sum().dataSync()[0]
		};
	}
	train(input: number[], target: number[]) {
		return this.trainTens(
			tf.tensor(input, [input.length, 1])
			, tf.tensor(target, [target.length, 1])
		)
	}
}