import { NeuralNetwork } from "./NeuralNetwork";

const obMap = <K extends Object>(v: K, fs: { [k in keyof K]: (v: K[k]) => any }) => {
	return Object.fromEntries(Object.entries(v).map(([k, z]) => [k, (fs as any)[k](z)]));
}

const nn = new NeuralNetwork([2, 2, 1]);

for(let i = 0; i < 1000000; i++) {
	[
		{ inputs: [0, 0], outputs: [0] }
		, { inputs: [0, 1], outputs: [1] }
		, { inputs: [1, 0], outputs: [1] }
		, { inputs: [1, 1], outputs: [0] }

	].forEach(({ inputs, outputs }) => {
		const res = nn.train(inputs, outputs);
		console.table([{
			...obMap(res, {
				output: x => x.data.map(z => z.map(y => y.toFixed(2)).join(",")).join("\n")
				, error: e => e.toFixed(2)
			})
			, inputs: inputs.map(v => v.toFixed(2))
		}]);
	})
}