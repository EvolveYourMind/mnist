import { getData } from "./mnist_data";
import { NeuralNetwork } from "../NeuralNetwork";

const data = getData("train");

const nn = new NeuralNetwork([data.nofRows * data.nofRows, 16, 16, 10]);

nn.fit(
	data.images
	, data.labels.map(l => Array.from({ length: 10 }, (_, i) => i === l ? 1 : 0))
	, 1000
	, 0.1
	, 256);
