const canvas = document.createElement("canvas");
const canvas2 = document.createElement("canvas");
const resetButton = document.createElement("button");
const guessButton = document.createElement("button");
const loadRandomButton = document.createElement("button");
const guessDiv = document.createElement("div");
const labelPostProcs = document.createElement("p");
labelPostProcs.innerText = "Post processing:"
resetButton.textContent = "Reset"
guessButton.textContent = "Guess"
loadRandomButton.textContent = "Load random mnist image"
const pixelMultiplier = 20;
canvas.width = 28 * pixelMultiplier;
canvas.height = 28 * pixelMultiplier;
canvas2.width = 28 * pixelMultiplier;
canvas2.height = 28 * pixelMultiplier;
document.body.appendChild(canvas);
document.body.appendChild(document.createElement("br"))
document.body.appendChild(resetButton);
document.body.appendChild(guessButton);
document.body.appendChild(loadRandomButton);
document.body.appendChild(guessDiv);
document.body.appendChild(labelPostProcs);
document.body.appendChild(canvas2)
const ctx2 = canvas2.getContext("2d");
const ctx = canvas.getContext("2d");
let points = [];
resetButton.addEventListener("click", () =>{
	ctx.fillStyle = "#000"
	ctx.fillRect(0, 0, canvas.width, canvas.height)
})
guessButton.addEventListener("click", drawGuess);
loadRandomButton.addEventListener("click", loadGuess	);
ctx.fillStyle = "#000"
ctx.fillRect(0, 0, canvas.width, canvas.height)
let mousedown = false;
canvas.addEventListener("mousedown", e => {
	mousedown = true
	ctx.fillStyle = "#fff"
	ctx.strokeStyle = "#fff"
	ctx.beginPath();
	ctx.moveTo(e.offsetX, e.offsetY);
})

document.body.addEventListener("mouseup", e => {
	mousedown = false
	ctx.closePath();
})

canvas.addEventListener("mousemove", e => {
	if(mousedown) {
		ctx.lineWidth = 40;
		ctx.lineCap = "round"
		ctx.lineJoin = "round"
		ctx.lineTo(e.offsetX, e.offsetY)
		ctx.stroke();
		ctx.moveTo(e.offsetX, e.offsetY);
	}
});


function drawGuess() {
	const img = tf.image.resizeBilinear(tf.browser.fromPixels(canvas, 1, true), [28, 28]).toFloat().div(255);
	const imgData = img.dataSync();
	for(let x = 0; x < 28; x++) {
		for(let y = 0; y < 28; y++) {
			ctx2.fillStyle = `#${Array.from({ length: 3 }, _ => Math.floor(imgData[y * 28 + x] * 15).toString(16)).join("")}`
			ctx2.fillRect(x * pixelMultiplier, y * pixelMultiplier, pixelMultiplier, pixelMultiplier);
		}
	}

	const outputs = [...model.predict(img.expandDims()).dataSync()];
	output(outputs);
}
let examples = null;
async function loadGuess() {
	if(examples === null) examples = await fetch("./examples.json").then(r => r.json());
	const { images } = examples;
	const randomImg = images[Math.floor(Math.random() * images.length)];
	for(let x = 0; x < 28; x++) {
		for(let y = 0; y < 28; y++) {
			ctx2.fillStyle = `#${Array.from({ length: 3 }, _ => Math.floor(randomImg[y * 28 + x] * 15).toString(16)).join("")}`
			ctx2.fillRect(x * pixelMultiplier, y * pixelMultiplier, pixelMultiplier, pixelMultiplier);
		}
	}
	const outputs = [...model.predict(tf.tensor1d(randomImg).reshape([28, 28, 1]).expandDims()).dataSync()];
	output(outputs)
}

function output(outputs) {
	const g = outputs.map((p, num) => ({ p, num }));
	g.sort((a, b) => b.p - a.p);
	guessDiv.innerText = `Guesses: \n${g.slice(0, 3).map(({ p, num }) => `${num}: ${(p * 100).toFixed(2)}%`).join("\n")}`
}

let model
(async () => {
	model = await tf.loadLayersModel(location.href + "/../model.json");
})();