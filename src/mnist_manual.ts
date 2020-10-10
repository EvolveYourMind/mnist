import { getData } from "./mnist";
import { NeuralNetwork } from "./TFNeuralNetwork";

const data = getData("training");

const nn = new NeuralNetwork([data.nofRows * data.nofRows, 16, 10]);

nn.fit(
	data.images
	, data.labels.map(l => Array.from({ length: 10 }, (_, i) => i === l ? 1 : 0))
	, 1000
	, 500);

// for(let ep = 0; ep < 1000; ep++) {
// 	let ok = 0;
// 	let cnt = 0;
// 	data.images.forEach((img, i) => {
// 		const target = Array.from({ length: 10 }).map(_ => 0);
// 		target[data.labels[i]] = 1;
// 		const { output, error } = nn.train(img, target);
// 		if(output.dataSync()[data.labels[i]] > 0.9) {
// 			ok++;
// 		}
// 		// er += error;
// 		cnt++;
// 		if(i % 500 === 0) {
// 			console.table([{
// 				epoch: ep,
// 				outputs: [...output.dataSync()].map(x => x.toFixed(2)).join(","),//.map(z => z.map(y => y.toFixed(1)).join(",")).join(","),
// 				target: data.labels[i],
// 				avgerr: error.toFixed(2),
// 				curscore: (ok / cnt).toFixed(2),
// 				progress: (i / data.nofImages).toFixed(2)
// 			}]);
// 			ok = 0;
// 			cnt = 0;
// 		}
// 	});
// }
