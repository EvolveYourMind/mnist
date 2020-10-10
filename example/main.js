const canvas = document.createElement("canvas");
const canvas2 = document.createElement("canvas");
const resetButton = document.createElement("button");
const guessButton = document.createElement("button");
const guessDiv = document.createElement("div");
const labelPostProcs = document.createElement("p");
labelPostProcs.innerText = "Post processing:"
resetButton.textContent = "Reset"
guessButton.textContent = "Guess"
const pixelMultiplier = 20;
canvas.width = 28 * pixelMultiplier;
canvas.height = 28 * pixelMultiplier;
canvas2.width = 28 * pixelMultiplier;
canvas2.height = 28 * pixelMultiplier;
document.body.appendChild(canvas);
document.body.appendChild(resetButton);
document.body.appendChild(guessButton);
document.body.appendChild(guessDiv);
document.body.appendChild(labelPostProcs);
document.body.appendChild(canvas2)
const ctx2 = canvas2.getContext("2d");
const ctx = canvas.getContext("2d");
let points = [];
resetButton.addEventListener("click", () => points = [])
guessButton.addEventListener("click", drawGuess);
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
		setTimeout(() => points.push({ x: e.offsetX, y: e.offsetY }), 0)
	}
});


function drawGuess() {
	const img = Array.from({ length: 28 }, () => Array.from({ length: 28 }, () => 0))
	points.map(p => [Math.floor(p.x / pixelMultiplier), Math.floor(p.y / pixelMultiplier)]).forEach(([x, y]) => {
		img[x][y] = 1;
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
				img[nx][ny] = Math.min(1, img[nx][ny] + 0.01)
			})
	})

	for(let x = 0; x < 28; x++) {
		for(let y = 0; y < 28; y++) {
			ctx2.fillStyle = `#${Array.from({ length: 3 }, _ => Math.floor(img[x][y] * 15).toString(16)).join("")}`
			ctx2.fillRect(x * pixelMultiplier, y * pixelMultiplier, pixelMultiplier, pixelMultiplier);
		}
	}

	const outputs = [...model.predict(tf.tensor(img.reduce((a, v) => a.concat(v), []), [1, 784])).dataSync()];
	const g = outputs.map((p, num) => ({ p, num }));
	g.sort((a, b) => b.p - a.p);
	guessDiv.innerText = `Guesses: \n${g.slice(0, 3).map(({ p, num }) => `(${num}, ${p})`).join("\n")}`
}

async function loadGuess() {
	const { images } = await fetch("/data/test/data.json").then(r => r.json());
	const randomImg = images[Math.floor(Math.random() * images.length)];
	for(let x = 0; x < 28; x++) {
		for(let y = 0; y < 28; y++) {
			ctx2.fillStyle = `#${Array.from({ length: 3 }, _ => Math.floor(randomImg[x * 28 + y] * 15).toString(16)).join("")}`
			ctx2.fillRect(x * pixelMultiplier, y * pixelMultiplier, pixelMultiplier, pixelMultiplier);
		}
	}
	const outputs = [...model.predict(tf.tensor(randomImg, [1, 784])).dataSync()];
	const g = outputs.map((p, num) => ({ p, num }));
	g.sort((a, b) => b.p - a.p);
	guessDiv.innerText = `Guesses: \n${g.slice(0, 3).map(({ p, num }) => `(${num}, ${p})`).join("\n")}`
}

let model
(async () => {
	model = await tf.loadLayersModel('http://localhost:5000/model/model.json');
	// let prev = 0;
	// setInterval(() => {
		// prev++;
		// if(prev % 2 === 0) {
		// drawGuess(model);
		// } else {
		// loadGuess(model);
		// }
	// }, 5000);
})();