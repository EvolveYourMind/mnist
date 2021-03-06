export const sum = (xs: number[]) => xs.reduce((a, v) => a + v);
export const zip = <A, B>(a: A[], b: B[]): [A, B][] => a.slice(0, Math.min(a.length, b.length)).map((x, i) => [x, b[i]]);
export const zipWith = <A, B, V>(a: A[], b: B[], f: (ab: [A, B]) => V): V[] => zip(a, b).map(f);
export const dot = (a: number[], b: number[]): number => sum(zip(a, b).map(([x, y]) => x * y));
export const sub = (a: number[], b: number[]) => zipWith(a, b, ([x, y]) => x - y);
export const add = (a: number[], b: number[]) => zipWith(a, b, ([x, y]) => x + y);
export const mul = (a: number[], b: number) => a.map(x => x * b)
export const hmul = (a: number[], b: number[]) => zipWith(a, b, ([x, y]) => x * y);
export const avg = (a: number[]) => sum(a) / a.length
export class Matrix {
	data: number[][];
	rows: number;
	cols: number;
	constructor(rows: number, cols: number) {
		this.data = Array.from({ length: rows }, _ => Array.from({ length: cols }));
		this.rows = rows;
		this.cols = cols;
	}
	static random(shape: [number, number]) {
		return new Matrix(shape[0], shape[1]).map(_ => (Math.random() - 1) * 2);
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
	static load(struct: ReturnType<Matrix["serialize"]>) {
		const mat = new Matrix(struct.rows, struct.cols);
		mat.data = struct.data;
		return mat;
	}
	map(f: (x: number, row: number, col: number) => number) {
		const res = new Matrix(this.rows, this.cols);
		res.data = this.data.map((xs, i) => xs.map((x, j) => f(x, i, j)))
		return res;
	}
	mapRows<T>(f: (xs: number[]) => T): T[] {
		return this.data.map(f);
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
	dot(other: Matrix) {
		return this.hamMul(other).sum();
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
	static vmv(x: number[], ws: Matrix): number[] {
		return ws.data.map(w => dot(x, w));
	}
	static avg(ms: Matrix[]): Matrix {
		return ms.reduce((acc, v) => acc.add(v)).map(v => v / ms.length);
	}
}