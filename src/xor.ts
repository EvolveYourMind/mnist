import { sum } from "./Matrix";
import { NeuralNetwork } from "./NeuralNetwork";

const mlp = new NeuralNetwork([2, 2, 1]);
const lr = 0.1;
// xor
mlp.fit([
	[0, 0]
	, [0,1]
	, [1, 0]
	, [1,1]
], [
	[0]
	, [1]
	, [1]
	, [0]
], 1000000, 0.1, 4);
