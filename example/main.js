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
resetButton.addEventListener("click", () => points = [])
guessButton.addEventListener("click", drawGuess);
loadRandomButton.addEventListener("click", loadGuess	);
function draw() {
	ctx.fillStyle = "#000"
	ctx.fillRect(0, 0, canvas.width, canvas.height)
	points.forEach(p => {
		ctx.beginPath();
		ctx.fillStyle = "#fff"
		ctx.arc(p.x, p.y, pixelMultiplier / 2, 0, Math.PI * 2);
		ctx.fill();
		ctx.closePath();
	});
}
setInterval(draw, 33);
let mousedown = false;
canvas.addEventListener("mousedown", e => {
	mousedown = true
})

document.body.addEventListener("mouseup", e => mousedown = false)

canvas.addEventListener("mousemove", e => {
	if(mousedown) {
		points.push({ x: e.offsetX, y: e.offsetY });
	}
});


function drawGuess() {
	const img = Array.from({ length: 28 }, () => Array.from({ length: 28 }, () => 0))
	points.map(p => [Math.floor(p.x / pixelMultiplier), Math.floor(p.y / pixelMultiplier)]).forEach(([x, y]) => {
		img[y][x] = 0.8;
		[
				[x - 1, y]
			, [x + 1, y]
			, [x, y + 1]
			, [x, y - 1]
			, [x + 1, y + 1]
			, [x - 1, y + 1]
			, [x + 1, y - 1]
			, [x - 1, y - 1]
		].filter((coords) => coords.every(p => 0 <= p && p < 28))
			.forEach(([nx, ny]) => {
				img[ny][nx] = Math.min(1, img[ny][nx] + 0.05)
			})
	})

	for(let x = 0; x < 28; x++) {
		for(let y = 0; y < 28; y++) {
			ctx2.fillStyle = `#${Array.from({ length: 3 }, _ => Math.floor(img[y][x] * 15).toString(16)).join("")}`
			ctx2.fillRect(x * pixelMultiplier, y * pixelMultiplier, pixelMultiplier, pixelMultiplier);
		}
	}

	const outputs = [...model.predict(tf.tensor1d(img.reduce((a, v) => a.concat(v))).expandDims()).dataSync()];
	const g = outputs.map((p, num) => ({ p, num }));
	g.sort((a, b) => b.p - a.p);
	guessDiv.innerText = `Guesses: \n${g.slice(0, 3).map(({ p, num }) => `${num}: ${(p * 100).toFixed(2)}%`).join("\n")}`
}

async function loadGuess() {
	const { images } = await fetch("./examples.json").then(r => r.json());
	const randomImg = images[Math.floor(Math.random() * images.length)];
	for(let x = 0; x < 28; x++) {
		for(let y = 0; y < 28; y++) {
			ctx2.fillStyle = `#${Array.from({ length: 3 }, _ => Math.floor(randomImg[y * 28 + x] * 15).toString(16)).join("")}`
			ctx2.fillRect(x * pixelMultiplier, y * pixelMultiplier, pixelMultiplier, pixelMultiplier);
		}
	}
	const outputs = [...model.predict(tf.tensor1d(randomImg).expandDims()).dataSync()];
	const g = outputs.map((p, num) => ({ p, num }));
	g.sort((a, b) => b.p - a.p);
	guessDiv.innerText = `Guesses: \n${g.slice(0, 3).map(({ p, num }) => `${num}: ${(p * 100).toFixed(2)}%`).join("\n")}`
}

let model
(async () => {
	model = await tf.loadLayersModel(location.href + "/../../model2/model.json");
})();