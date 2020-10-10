import fs from "fs";
import { resolve } from "path";

function readImgs(imgsPath: string, labelsPath: string) {
	const fbuff = fs.readFileSync(imgsPath);
	const lbuff = fs.readFileSync(labelsPath);
	const magic = fbuff.readInt32BE(0);
	const nofImages = fbuff.readInt32BE(4);
	const nofRows = fbuff.readInt32BE(8);
	const nofCols = fbuff.readInt32BE(12);
	const sz = nofRows * nofCols;
	return {
		magic,
		nofImages,
		nofRows,
		nofCols,
		images: Array.from({ length: nofImages })
			.map((_, row) => 16 + row * sz)
			.map(begin => Array.from({ length: sz }).map((_, i) => fbuff[begin + i] / 255)),
		labels: Array.from({ length: nofImages }).map((_, i) => lbuff[8 + i])
	}
}

export function getData(type: "test" | "training"): ReturnType<typeof readImgs> {
	return JSON.parse(fs.readFileSync(resolve("data", type, "data.json")).toString());
}
const WEIGHTS_PATH = "dist/weights2.json";