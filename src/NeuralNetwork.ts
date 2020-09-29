const logistic: (x: number) => number = x => 1 / (1 + Math.E ** (-x));
const derlogistic = (y: number) => y * (1 - y);

const tanh: (x: number) => number = Math.tanh
const dtanh = (y: number) => 1 - Math.tanh(y) ** 2


const rand = () => Math.random() * 2 - 1;
const sum = (xs: number[]) => xs.reduce((a, v) => a + v);
const zip = <A, B>(a: A[], b: B[]) => a.map((x, i) => [x, b[i]]) as [A, B][];

class Matrix {
	data: number[][];
	rows: number;
	cols: number;
	constructor(rows: number, cols: number) {
		this.data = Array.from({ length: rows }, _ => Array.from({ length: cols }));
		this.rows = rows;
		this.cols = cols;
	}
	shape() {
		return [this.rows, this.cols];
	}
	static fromArray(ar: number[]) {
		const m = new Matrix(ar.length, 1);
		m.data = ar.map(x => [x]);
		return m;
	}
	static fromMat(m: number[][]) {
		return new Matrix(m.length, m[0].length).map((_, i, j) => m[i][j]);
	}
	serialize() {
		return { data: this.data, rows: this.rows, cols: this.cols };
	}
	map(f: (x: number, row: number, col: number) => number) {
		const res = new Matrix(this.rows, this.cols);
		res.data = this.data.map((xs, i) => xs.map((x, j) => f(x, i, j)))
		return res;
	}
	reduce<T>(f: (acc: T, row: number[], i: number) => T, initialValue: T) {
		return this.data.reduce(f, initialValue);
	}
	sum() {
		return this.reduce((acc, row) => acc + sum(row), 0);
	}
	assertEqualSize(other: Matrix) {
		if(this.cols !== other.cols || this.rows !== other.rows) {
			throw new Error("SIZE MISMATCH: " + this.shape() + ", " + other.shape())
		}
	}
	zipWith(other: Matrix, f: (x: number, y: number) => number) {
		this.assertEqualSize(other);
		return this.map((x, i, j) => f(x, other.data[i][j]));
	}
	add(other: Matrix) {
		return this.zipWith(other, (x, y) => x + y);
	}
	sub(other: Matrix) {
		return this.zipWith(other, (x, y) => x - y);
	}
	hamMul(other: Matrix) {
		return this.zipWith(other, (x, y) => x * y);
	}
	mul(other: number | Matrix) {
		if(typeof other === "number") {
			return this.map(x => x * other);
		} else {
			if(this.cols !== other.rows) {
				throw new Error("SIZE MISMATCH: " + this.shape() + ", " + other.shape())
			}
			const res = new Matrix(this.rows, other.cols);
			res.data = this.data.map(xs => other.data[0].map((_, j) => sum(xs.map((x, p) => x * other.data[p][j]))));
			return res;
		}
	}
	T() {
		const res = new Matrix(this.cols, this.rows);
		res.data = res.data.map((rw, i) => rw.map((_, j) => this.data[j][i]));
		return res;
	}
	print() {
		console.table(this.data);
	}
	eq(m: Matrix) {
		return this.data.every((row, i) => row.every((v, j) => v === m.data[i][j]))
	}
}

export class NeuralNetwork {
	weights: Matrix[];
	biases: Matrix[];
	shape: number[];
	act: typeof logistic;
	dact: typeof derlogistic;
	lr = 0.2;
	constructor(shape: number[]) {
		this.shape = shape;
		this.act = logistic;
		this.dact = derlogistic;
		this.weights = shape.slice(1).map((sz, i) => new Matrix(sz, shape[i]).map(rand));
		this.biases = shape.slice(1).map(sz => new Matrix(sz, 1).map(rand));
	}
	serialize() {
		return {
			weights: this.weights.map(x => x.serialize())
			, biases: this.biases.map(x => x.serialize())
			, shape: this.shape
		};
	}
	train(input: number[], target: number[]) {
		const allOutputs = zip(this.weights, this.biases)
			.reduce((r, [weights, biases]) => r.concat(
				weights
					.mul(r[r.length - 1])
					.add(biases)
					.map(this.act)
			), [Matrix.fromArray(input)])
		let errors = Matrix.fromArray(target).sub(allOutputs[allOutputs.length - 1]);
		for(let i = allOutputs.length - 1; i >= 1; i--) {
			const grad = allOutputs[i].map(this.dact).hamMul(errors).mul(this.lr);
			const deltas = grad.mul(allOutputs[i - 1].T());
			this.weights[i - 1] = this.weights[i - 1].add(deltas);
			this.biases[i - 1] = this.biases[i - 1].add(grad);
			errors = this.weights[i - 1].T().mul(errors);
		}
		return {
			output: allOutputs[allOutputs.length - 1],
			error: Matrix.fromArray(target).sub(allOutputs[allOutputs.length - 1]).map(v => v ** 2).sum()
		};
	}
}