import { getData } from "./mnist";
import * as tf from "@tensorflow/tfjs-node-gpu"
import { resolve } from "path";
import { existsSync } from "fs";

(async () => {
	const data = getData("training");
	const tfDataset = tf.data.zip({
		xs: tf.data.array(data.images.map(i => tf.tensor1d(i)))
		, ys: tf.data.array(data.labels.map(l => tf.tensor1d(Array.from({ length: 10 }, (_, i) => i === l ? 1 : 0))))
	}).batch(4)
		.shuffle(4);

	const modelPath = resolve(__dirname, "..", "model2");

	const nn = existsSync(modelPath) ? await tf.loadLayersModel("file://" + resolve(modelPath, "model.json")) : tf.sequential({
		layers: [
			tf.layers.dense({ inputShape: [data.images[0].length], units: 20, activation: "sigmoid" })
			, tf.layers.dense({ units: 10, activation: "softmax" })
		]
	});
	nn.compile({
		optimizer: "sgd"
		, loss: "meanSquaredError"
		, metrics: ["accuracy"]
	});
	await nn.fitDataset(tfDataset, {
		epochs: 1000
		, verbose: 1
		, callbacks: {
			onEpochEnd: async ep => {
				try {
					await nn.save("file://" + modelPath);
					console.log("Saved model for epoch ", ep)
					const testData = getData("test");
					const zz = nn.evaluate(
						tf.tensor2d(testData.images)
						, tf.tensor2d(testData.labels.map(l => (Array.from({ length: 10 }, (_, i) => i === l ? 1 : 0))))
					)
					console.log(zz.toString());
				} catch(err) {
					console.error(err);
				}
			}
		}
	})

})();