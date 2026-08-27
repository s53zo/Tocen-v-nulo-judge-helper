//#region node_modules/exifreader/src/dataview.js
var e = class {
	constructor(e) {
		if (t(e)) throw Error("DataView: Passed buffer type is unsupported.");
		this.buffer = e, this.byteLength = this.buffer.length;
	}
	getUint8(e) {
		return this.buffer.readUInt8(e);
	}
	getUint16(e, t) {
		return t ? this.buffer.readUInt16LE(e) : this.buffer.readUInt16BE(e);
	}
	getUint32(e, t) {
		return t ? this.buffer.readUInt32LE(e) : this.buffer.readUInt32BE(e);
	}
	getInt32(e, t) {
		return t ? this.buffer.readInt32LE(e) : this.buffer.readInt32BE(e);
	}
};
function t(e) {
	return typeof e != "object" || e.length === void 0 || e.readUInt8 === void 0 || e.readUInt16LE === void 0 || e.readUInt16BE === void 0 || e.readUInt32LE === void 0 || e.readUInt32BE === void 0 || e.readInt32LE === void 0 || e.readInt32BE === void 0;
}
//#endregion
//#region node_modules/exifreader/src/utils.js
function n(t, n, r) {
	try {
		return new DataView(t, n, r);
	} catch {
		return new e(t, n, r);
	}
}
function r(e, t, n) {
	let r = [];
	for (let i = 0; i < n && t + i < e.byteLength; i++) r.push(e.getUint8(t + i));
	return s(r);
}
function i(e, t) {
	let n = [], r = 0;
	for (; t + r < e.byteLength;) {
		let i = e.getUint8(t + r);
		if (i === 0) break;
		n.push(i), r++;
	}
	return s(n);
}
function a(e, t, n) {
	let r = [];
	for (let i = 0; i + 2 <= n && t + i + 2 <= e.byteLength; i += 2) r.push(e.getUint16(t + i));
	return r[r.length - 1] === 0 && r.pop(), s(r);
}
function o(e, t) {
	let n = e.getUint8(t);
	return [n, r(e, t + 1, n)];
}
function s(e) {
	return e.map((e) => String.fromCharCode(e)).join("");
}
function c(e, t, n, r) {
	e && t && e.push({
		type: t,
		start: n,
		end: r
	});
}
function l() {
	if (typeof Promise > "u") throw Error("Promise is required when async mode is enabled.");
}
function u() {
	for (let e = 1; e < arguments.length; e++) for (let t in arguments[e]) d(arguments[0], t, arguments[e][t]);
	return arguments[0];
}
function d(e, t, n) {
	if (t === "__proto__") {
		Object.defineProperty(e, t, {
			value: n,
			enumerable: !0,
			writable: !0,
			configurable: !0
		});
		return;
	}
	e[t] = n;
}
function f(e, t, n) {
	let r = !1;
	Object.defineProperty(e, t, {
		get() {
			return r || (r = !0, Object.defineProperty(e, t, {
				configurable: !0,
				enumerable: !0,
				value: n.apply(e),
				writable: !0
			})), e[t];
		},
		configurable: !0,
		enumerable: !0
	});
}
function p(e) {
	if (typeof btoa < "u") return btoa(typeof e == "string" ? e : Array.prototype.reduce.call(new Uint8Array(e), (e, t) => e + String.fromCharCode(t), ""));
	if (!(typeof Buffer > "u")) return Buffer.from === void 0 ? new Buffer(e).toString("base64") : Buffer.from(e).toString("base64");
}
function m(e) {
	let t = e.substring(e.indexOf(",") + 1);
	if (e.indexOf(";base64") !== -1) return typeof atob < "u" ? Uint8Array.from(atob(t), (e) => e.charCodeAt(0)).buffer : typeof Buffer > "u" ? void 0 : Buffer.from === void 0 ? new Buffer(t, "base64") : Buffer.from(t, "base64");
	let n = decodeURIComponent(t);
	return typeof Buffer < "u" ? Buffer.from === void 0 ? new Buffer(n) : Buffer.from(n) : Uint8Array.from(n, (e) => e.charCodeAt(0)).buffer;
}
function h(e, t, n) {
	return _(n, Math.max(0, t - e.length)) + e;
}
function g(e, t) {
	return parseInt(e.replace(".", ""), t) / t ** +(e.split(".")[1] || "").length;
}
function _(e, t) {
	return Array(t + 1).join(e);
}
var v = "brotli", ee = 134217728;
function te(e, t, n, r = "string", i) {
	let a = ne(i);
	if (i && t !== void 0) {
		let o = i[t === 0 ? "deflate" : "brotli"];
		if (typeof o == "function") {
			let t = new Uint8Array(e.buffer, e.byteOffset, e.byteLength);
			return Promise.resolve(o(t)).then((e) => ie(e) > a ? x(a) : r === "dataview" ? e instanceof DataView ? e : e instanceof ArrayBuffer ? new DataView(e) : new DataView(e.buffer, e.byteOffset, e.byteLength) : new TextDecoder(n).decode(e));
		}
	}
	if (t === 0 && typeof DecompressionStream == "function") return y(e, "deflate", a).then((e) => b(e, r, n));
	if (t === "brotli") {
		if (typeof DecompressionStream == "function") try {
			return y(e, "brotli", a).then((e) => b(e, r, n));
		} catch {}
		return Promise.reject("Brotli decompression is not supported in this environment. Pass in a brotli decompression function via the decompress option.");
	}
	if (t !== void 0) return Promise.reject(`Unknown compression method ${t}.`);
	if (r === "string") try {
		return new TextDecoder(n).decode(e);
	} catch {
		let t = new Uint8Array(e.buffer, e.byteOffset, e.byteLength);
		return Array.from(t, (e) => String.fromCharCode(e)).join("");
	}
	return e;
}
function ne(e) {
	return e && typeof e.maxDecompressedSize == "number" ? e.maxDecompressedSize : ee;
}
function y(e, t, n) {
	let r = new DecompressionStream(t), i = new Blob([e]).stream().pipeThrough(r).getReader(), a = [], o = 0;
	return s();
	function s() {
		return i.read().then(({ done: e, value: t }) => e ? re(a, o) : (o += t.byteLength, o > n ? i.cancel().then(() => x(n)) : (a.push(t), s())));
	}
}
function re(e, t) {
	let n = new Uint8Array(t), r = 0;
	for (let t = 0; t < e.length; t++) n.set(e[t], r), r += e[t].byteLength;
	return n.buffer;
}
function b(e, t, n) {
	return t === "dataview" ? new DataView(e) : new TextDecoder(n).decode(e);
}
function ie(e) {
	return e && typeof e.byteLength == "number" ? e.byteLength : 0;
}
function x(e) {
	return typeof console < "u" && typeof console.warn == "function" && console.warn(`ExifReader: skipped a compressed metadata block that would exceed the maximum decompressed size of ${e} bytes.`), Promise.reject(`Decompressed metadata exceeded the maximum allowed size of ${e} bytes.`);
}
function S(e) {
	return typeof e == "string";
}
function C(e) {
	return typeof File < "u" && e instanceof File;
}
function w(e) {
	return /^data:[^;,]*(;base64)?,/.test(e);
}
function ae(e, t) {
	return /^\w+:\/\//.test(e) ? typeof fetch < "u" ? E(e, T(t)).then((e) => e.buffer) : le(e, T(t)).then((e) => e.buffer) : w(e) ? Promise.resolve(m(e)) : O(e, T(t)).then((e) => e.buffer);
}
function oe(e, t) {
	return ge(e, T(t)).then((e) => e.buffer);
}
function T(e) {
	return e && Number.isInteger(e.length) && e.length >= 0 ? {
		start: 0,
		end: e.length
	} : { start: 0 };
}
function E(e, { start: t = 0, end: n } = {}) {
	let r = { method: "GET" };
	return (t > 0 || n !== void 0 && n !== Infinity) && (r.headers = { range: D(t, n) }), fetch(e, r).then((e) => {
		let t = e && typeof e.status == "number" ? e.status : void 0;
		if (t !== void 0 && !ce(t)) {
			let n = e.statusText || "";
			return Promise.reject(Error(`Could not fetch file: ${t} ${n}`.trim()));
		}
		let n = se(e);
		return Promise.resolve(e.arrayBuffer()).then((e) => ({
			buffer: e,
			totalSize: n,
			status: t
		}));
	});
}
function D(e, t) {
	return t === void 0 || t === Infinity ? `bytes=${e}-` : `bytes=${e}-${t - 1}`;
}
function se(e) {
	if (!(!e || !e.headers || typeof e.headers.get != "function")) return de(e.headers.get("Content-Range"), e.headers.get("Content-Length"));
}
function ce(e) {
	return e >= 200 && e <= 299 || e === 416;
}
function le(e, { start: t = 0, end: n } = {}) {
	return new Promise((r, i) => {
		let a = {};
		(t > 0 || n !== void 0 && n !== Infinity) && (a.headers = { range: D(t, n) }), fe(e)(e, a, (e) => {
			if (e.statusCode >= 200 && e.statusCode <= 299) {
				let t = ue(e), n = [];
				e.on("data", (e) => n.push(Buffer.from(e))), e.on("error", (e) => i(e)), e.on("end", () => r({
					buffer: Buffer.concat(n),
					totalSize: t,
					status: e.statusCode
				}));
			} else e.statusCode === 416 ? (e.resume(), r({
				buffer: Buffer.alloc(0),
				totalSize: ue(e),
				status: e.statusCode
			})) : (i(/* @__PURE__ */ Error(`Could not fetch file: ${e.statusCode} ${e.statusMessage}`)), e.resume());
		}).on("error", (e) => i(e));
	});
}
function ue(e) {
	if (!(!e || !e.headers)) return de(e.headers["content-range"], e.headers["content-length"]);
}
function de(e, t) {
	if (e) {
		let t = /\/(\d+|\*)$/.exec(e);
		if (t && t[1] !== "*") return parseInt(t[1], 10);
	}
	if (t) {
		let e = parseInt(t, 10);
		if (Number.isFinite(e)) return e;
	}
}
function fe(e) {
	return /^https:\/\//.test(e) ? __non_webpack_require__("https").get : __non_webpack_require__("http").get;
}
function O(e, { start: t = 0, end: n, totalSize: r } = {}) {
	return new Promise((i, a) => {
		let o = he();
		o.open(e, (s, c) => {
			if (s) return a(s);
			pe(o, e, r, (r, s) => {
				if (r) return o.close(c, () => a(r));
				let l = n === void 0 || n === Infinity || n > s ? s : n, u = Math.min(Math.max(0, t), l), d = l - u, f = Buffer.alloc(d);
				if (d === 0) return o.close(c, (t) => me(e, t, f, s, i));
				o.read(c, {
					buffer: f,
					length: d,
					position: u
				}, (t) => {
					if (t) return o.close(c, () => a(t));
					o.close(c, (t) => me(e, t, f, s, i));
				});
			});
		});
	});
}
function pe(e, t, n, r) {
	if (n !== void 0) return r(null, n);
	e.stat(t, (e, t) => r(e, e ? void 0 : t.size));
}
function me(e, t, n, r, i) {
	t && console.warn(`Could not close file ${e}:`, t), i({
		buffer: n,
		totalSize: r
	});
}
function he() {
	try {
		return __non_webpack_require__("fs");
	} catch {
		return;
	}
}
function ge(e, { start: t = 0, end: n } = {}) {
	return new Promise((r, i) => {
		let a = new FileReader(), o = typeof e.size == "number" ? e.size : void 0;
		if (a.onload = (e) => r({
			buffer: e.target.result,
			totalSize: o
		}), a.onerror = () => i(a.error), e && typeof e.slice == "function" && (t > 0 || n !== void 0 && n !== Infinity)) {
			let r = n === void 0 || n === Infinity ? o : n;
			a.readAsArrayBuffer(e.slice(t, r));
		} else a.readAsArrayBuffer(e);
	});
}
function _e(e, t) {
	if (!e) return t;
	if (!t) return e;
	if (typeof Buffer < "u" && Buffer.isBuffer(e) && Buffer.isBuffer(t)) return Buffer.concat([e, t]);
	let n = ve(e), r = ve(t), i = new Uint8Array(n.byteLength + r.byteLength);
	return i.set(n, 0), i.set(r, n.byteLength), i.buffer;
}
function ve(e) {
	return e instanceof ArrayBuffer ? new Uint8Array(e) : typeof Buffer < "u" && Buffer.isBuffer(e) || ArrayBuffer.isView(e) ? new Uint8Array(e.buffer, e.byteOffset, e.byteLength) : new Uint8Array(e);
}
//#endregion
//#region node_modules/exifreader/src/load-auto.js
var ye = 131072, be = 4, xe = "length: \"auto\" could not locate metadata in this file (no metadata blocks found after reading the file — e.g. plain TIFF, bare JPEG XL codestream, or a file with no recognizable metadata).", Se = "length: \"auto\" is not supported for this file type (no leading metadata container — e.g. plain TIFF or bare JPEG XL codestream).";
function Ce(e) {
	if (e.expanded !== !0 || e.includeOffsets !== !0) throw Error("length: \"auto\" requires both expanded: true and includeOffsets: true.");
}
function we(e) {
	return function(e, i) {
		l();
		let a = u({}, i, { async: !0 });
		return S(e) ? t(e, a) : C(e) ? n(e, a) : r(e, a);
	};
	function t(e, t) {
		return /^\w+:\/\//.test(e) ? i(typeof fetch < "u" ? {
			readRange: (t, n) => E(e, {
				start: t,
				end: n
			}),
			options: t
		} : {
			readRange: (t, n) => le(e, {
				start: t,
				end: n
			}),
			options: t
		}) : w(e) ? r(m(e), t) : i({
			readRange: (t, n, r) => O(e, {
				start: t,
				end: n,
				totalSize: r ? r.totalSize : void 0
			}),
			options: t
		});
	}
	function n(e, t) {
		return i({
			readRange: (t, n) => ge(e, {
				start: t,
				end: n
			}),
			options: t
		});
	}
	function r(t, n) {
		return Promise.resolve().then(() => e(t, n)).then((e) => {
			if (!e || !e.metadataRange) throw Error(Se);
			let n = e.metadataRange.end;
			return e.metadataRange.buffer = ke(t, n), e.metadataRange.fetched = Ae(e.metadataRange.buffer), e.metadataRange.requests = 0, e;
		});
	}
	function i({ readRange: e, options: t }) {
		return a({
			readRange: e,
			options: t,
			buffer: null,
			fetched: 0,
			requests: 0,
			totalSize: void 0,
			need: ye,
			iter: 0
		});
	}
	function a(t) {
		return t.iter >= be || (t.totalSize !== void 0 && (t.need = Math.min(t.need, t.totalSize)), t.need <= t.fetched) ? c(t) : t.readRange(t.fetched, t.need, { totalSize: t.totalSize }).then((n) => (t.requests++, n.status === 416 ? s(t) : (Te(t, n), Promise.resolve(e(t.buffer, t.options)).then((e) => o(t, e)))));
	}
	function o(e, t) {
		let n = t && t.metadataRange;
		if (n && n.complete) return Ee(t, e.buffer, e.fetched, e.requests);
		let r = e.totalSize !== void 0 && e.fetched >= e.totalSize;
		if (!n && r) throw Error(xe);
		let i = De({
			range: n,
			totalSize: e.totalSize,
			fetched: e.fetched
		});
		if (i <= e.fetched) {
			if (n) return Ee(t, e.buffer, e.fetched, e.requests);
			throw Error(xe);
		}
		return e.need = i, e.iter++, a(e);
	}
	function s(t) {
		return t.readRange(0, Infinity, { totalSize: void 0 }).then((n) => (t.requests++, t.buffer = n.buffer, t.fetched = Ae(t.buffer), n.totalSize !== void 0 && (t.totalSize = n.totalSize), Promise.resolve(e(t.buffer, t.options)))).then((e) => {
			if (!e || !e.metadataRange) throw Error(xe);
			return Ee(e, t.buffer, t.fetched, t.requests);
		});
	}
	function c(t) {
		console.warn(`ExifReader: length:"auto" did not converge in ${be} iterations; falling back to full read.`);
		let n;
		return n = t.totalSize !== void 0 && t.fetched < t.totalSize ? t.readRange(t.fetched, t.totalSize, { totalSize: t.totalSize }).then((e) => {
			t.requests++, Te(t, e);
		}) : t.totalSize === void 0 ? t.readRange(t.fetched, Infinity, { totalSize: t.totalSize }).then((e) => {
			t.requests++, Te(t, e);
		}) : Promise.resolve(), n.then(() => Promise.resolve(e(t.buffer, t.options))).then((e) => {
			if (!e || !e.metadataRange) throw Error(xe);
			return Ee(e, t.buffer, t.fetched, t.requests);
		});
	}
}
function Te(e, t) {
	if (t.status === 200 && e.fetched > 0) {
		e.buffer = t.buffer, e.fetched = Ae(e.buffer), t.totalSize !== void 0 && (e.totalSize = t.totalSize);
		return;
	}
	t.totalSize !== void 0 && e.totalSize === void 0 && (e.totalSize = t.totalSize), e.buffer = _e(e.buffer, t.buffer), e.fetched = Ae(e.buffer);
}
function Ee(e, t, n, r) {
	let i = e.metadataRange.end;
	return e.metadataRange.buffer = Oe(t, i), e.metadataRange.fetched = n, e.metadataRange.requests = r, e;
}
function De({ range: e, totalSize: t, fetched: n }) {
	return e ? Math.min(Math.max(e.end, n * 2), t === void 0 ? Infinity : t) : t === void 0 ? n * 2 : t;
}
function Oe(e, t) {
	return !e || typeof t != "number" || t < 0 ? e : typeof Buffer < "u" && Buffer.isBuffer(e) ? Buffer.from(e.subarray(0, Math.min(t, e.length))) : e instanceof ArrayBuffer || typeof SharedArrayBuffer < "u" && e instanceof SharedArrayBuffer ? e.slice(0, Math.min(t, e.byteLength)) : e;
}
function ke(e, t) {
	return typeof Buffer < "u" && Buffer.isBuffer(e) ? Buffer.from(e.subarray(0, t)) : e instanceof ArrayBuffer || typeof SharedArrayBuffer < "u" && e instanceof SharedArrayBuffer ? e.slice(0, t) : ArrayBuffer.isView(e) ? e.buffer.slice(e.byteOffset, e.byteOffset + t) : e;
}
function Ae(e) {
	return e ? typeof e.byteLength == "number" ? e.byteLength : typeof e.length == "number" ? e.length : 0 : 0;
}
//#endregion
//#region node_modules/exifreader/src/constants.js
var k = {
	USE_FILE: !0,
	USE_JFIF: !0,
	USE_PNG_FILE: !0,
	USE_EXIF: !0,
	USE_IPTC: !0,
	USE_XMP: !0,
	USE_ICC: !0,
	USE_MPF: !0,
	USE_PHOTOSHOP: !0,
	USE_THUMBNAIL: !0,
	USE_TIFF: !0,
	USE_JPEG: !0,
	USE_PNG: !0,
	USE_HEIC: !0,
	USE_AVIF: !0,
	USE_JXL: !0,
	USE_WEBP: !0,
	USE_GIF: !0,
	USE_MAKER_NOTES: !0
};
//#endregion
//#region node_modules/exifreader/src/tag-names-utils.js
function je(e) {
	if (e.length >= 8) {
		let t = s(e.slice(0, 8));
		if (t === "ASCII\0\0\0") return s(e.slice(8));
		if (t === "JIS\0\0\0\0\0") return "[JIS encoded text]";
		if (t === "UNICODE\0") return "[Unicode encoded text]";
		if (t === "\0\0\0\0\0\0\0\0") {
			let t = s(e.slice(8));
			return /[\x20-\x7e]/.test(t) ? t : "[Undefined encoding]";
		}
	}
	return "Undefined";
}
function Me(e) {
	return e[0][0] / e[0][1] + e[1][0] / e[1][1] / 60 + e[2][0] / e[2][1] / 3600;
}
//#endregion
//#region node_modules/exifreader/src/byte-order.js
var Ne = 18761, Pe = 19789, A = {
	BIG_ENDIAN: Pe,
	LITTLE_ENDIAN: Ne,
	getByteOrder: Fe
};
function Fe(e, t) {
	if (e.getUint16(t) === Ne) return Ne;
	if (e.getUint16(t) === Pe) return Pe;
	throw Error("Illegal byte order value. Faulty image.");
}
//#endregion
//#region node_modules/exifreader/src/image-header-iso-bmff-utils.js
function Ie(e, t) {
	return e.getUint32(t + 4);
}
function Le(e, t, n) {
	return t + n <= e.byteLength;
}
//#endregion
//#region node_modules/exifreader/src/image-header-iso-bmff-iloc.js
var Re = 1048576;
function ze(e, t, n, r, i, a) {
	let { offsets: o, sizes: s } = Be(n, r + 3), c = e.getUint8(o.offsetSize) >> 4;
	s.item.extent.extentOffset = c;
	let l = e.getUint8(o.lengthSize) & 15;
	s.item.extent.extentLength = l;
	let u = e.getUint8(o.baseOffsetSize) >> 4;
	s.item.baseOffset = u;
	let d = Ve(e, o.indexSize, n);
	s.item.extent.extentIndex = d === void 0 ? 0 : d;
	let f = He(e, o.itemCount, n);
	return {
		type: "iloc",
		items: Ue(e, Math.min(t + i, a), n, o, s, c, l, d, f),
		length: i
	};
}
function Be(e, t) {
	let n = { item: {
		dataReferenceIndex: 2,
		extentCount: 2,
		extent: {}
	} };
	e < 2 ? (n.itemCount = 2, n.item.itemId = 2) : e === 2 && (n.itemCount = 4, n.item.itemId = 4), e === 1 || e === 2 ? n.item.constructionMethod = 2 : n.item.constructionMethod = 0;
	let r = {
		offsetSize: t,
		lengthSize: t,
		baseOffsetSize: t + 1,
		indexSize: t + 1
	};
	return r.itemCount = t + 2, r.items = r.itemCount + n.itemCount, r.item = { itemId: 0 }, r.item.constructionMethod = r.item.itemId + n.item.itemId, r.item.dataReferenceIndex = r.item.constructionMethod + n.item.constructionMethod, {
		offsets: r,
		sizes: n
	};
}
function Ve(e, t, n) {
	if (n === 1 || n === 2) return e.getUint8(t) & 15;
}
function He(e, t, n) {
	if (n < 2) return e.getUint16(t);
	if (n === 2) return e.getUint32(t);
}
function Ue(e, t, n, r, i, a, o, s, c) {
	if (c === void 0) return [];
	let l = [], u = r.items, d = i.item.extent.extentIndex + i.item.extent.extentOffset + i.item.extent.extentLength, f = i.item.itemId + i.item.constructionMethod + i.item.dataReferenceIndex + i.item.baseOffset + i.item.extentCount, p = 0;
	for (let r = 0; r < c && !(u + f > t); r++) {
		let r = { extents: [] };
		r.itemId = Ge(e, u, n), u += i.item.itemId, r.constructionMethod = n === 1 || n === 2 ? e.getUint16(u) & 15 : void 0, u += i.item.constructionMethod, r.dataReferenceIndex = e.getUint16(u), u += i.item.dataReferenceIndex, r.baseOffset = qe(e, u, i.item.baseOffset), u += i.item.baseOffset, r.extentCount = e.getUint16(u), u += i.item.extentCount;
		let c = We(r.extentCount, d, u, t, Re - p);
		for (let t = 0; t < c; t++) {
			let t = {};
			t.extentIndex = Ke(e, n, u, s), u += i.item.extent.extentIndex, t.extentOffset = qe(e, u, a), u += i.item.extent.extentOffset, t.extentLength = qe(e, u, o), u += i.item.extent.extentLength, r.extents.push(t);
		}
		if (p += c, l.push(r), (r.extentCount - c) * d > 0) break;
	}
	return l;
}
function We(e, t, n, r, i) {
	if (t === 0) return 0;
	let a = Math.floor((r - n) / t);
	return Math.max(0, Math.min(e, a, i));
}
function Ge(e, t, n) {
	if (n < 2) return e.getUint16(t);
	if (n === 2) return e.getUint32(t);
}
function Ke(e, t, n, r) {
	if ((t === 1 || t === 2) && r > 0) return qe(e, n, r);
}
function qe(e, t, n) {
	return n === 4 ? e.getUint32(t) : n === 8 ? (console.warn("This file uses an 8-bit offset which is currently not supported by ExifReader. Contact the maintainer to get it fixed."), Ie(e, t)) : 0;
}
//#endregion
//#region node_modules/exifreader/src/image-header-iso-bmff.js
var Je = 1718909296, Ye = 1768977008, Xe = 1835365473, Ze = 1768714083, Qe = 1768186228, $e = 1768517222, et = 1768842853, tt = 1768973167, nt = 1668246642, rt = 1165519206, it = 1835625829, at = 1970432288;
function ot(e, t, n = e.byteLength) {
	if (Le(e, t, 8)) try {
		let { length: r, contentOffset: i } = ct(e, t);
		if (r < 8) return;
		let a = e.getUint32(t + 4);
		if (a === Je) return Et(e, i, r);
		if (a === Ye) return Dt(e, t, i, r, n);
		if (a === tt) return Ot(e, t, i, r, n);
		if (a === nt) return kt(e, i, r);
		if (a === Qe) return Mt(i, r);
		if (!Le(e, i, 1)) return;
		let o = e.getUint8(i);
		return a === Xe ? jt(e, t, i + 1, r, n) : a === Ze ? ze(e, t, o, i + 1, r, n) : a === $e ? Pt(e, t, o, i + 1, r, n) : a === et ? Ft(e, t, o, i + 1, r) : {
			type: void 0,
			length: r
		};
	} catch {
		return;
	}
}
function st(e, t) {
	if (!e) return !1;
	try {
		let n = ot(e, 0);
		return n !== void 0 && t.indexOf(n.majorBrand) !== -1;
	} catch {
		return !1;
	}
}
function ct(e, t) {
	let n = e.getUint32(t);
	if (lt(n)) return {
		length: e.byteLength - t,
		contentOffset: t + 4 + 4
	};
	if (ut(n)) {
		if (!Le(e, t, 16)) return {
			length: 0,
			contentOffset: t + 4 + 4
		};
		if (dt(e, t)) return {
			length: e.getUint32(t + 12),
			contentOffset: t + 4 + 4 + 8
		};
	}
	return {
		length: n,
		contentOffset: t + 4 + 4
	};
}
function lt(e) {
	return e === 0;
}
function ut(e) {
	return e === 1;
}
function dt(e, t) {
	return e.getUint32(t + 8) === 0;
}
function ft(e, t) {
	if (k.USE_EXIF || k.USE_XMP || k.USE_ICC) {
		let n = {}, r = pt(e);
		if (!r) return { hasAppMarkers: !1 };
		if (k.USE_EXIF) {
			let i = mt(e, r, t);
			i !== void 0 && (n.tiffHeaderOffset = i.tiffHeaderOffset, i.dataView !== void 0 && (n.exifDataView = i.dataView));
		}
		if (k.USE_XMP) {
			let i = Ct(e, r, t);
			i !== void 0 && (n.xmpChunks = i.chunks, i.dataView !== void 0 && (n.xmpDataView = i.dataView));
		}
		return k.USE_ICC && (n.iccChunks = Tt(r, t)), n.hasAppMarkers = n.tiffHeaderOffset !== void 0 || n.xmpChunks !== void 0 || n.iccChunks !== void 0, n;
	}
	return {};
}
function pt(e) {
	let t = 0;
	for (; t + 4 + 4 <= e.byteLength;) {
		let n = ot(e, t);
		if (n === void 0) break;
		if (n.type === "meta") return n;
		t += n.length;
	}
}
function mt(e, t, n) {
	try {
		let r = bt(t).itemId, i = xt(t, r);
		vt(i);
		let a = _t(t);
		if (ht(n, i, "exif", a), i.extents.length > 1) {
			let t = yt(e, i, a);
			return t === void 0 ? void 0 : {
				tiffHeaderOffset: St(t, 0),
				dataView: t
			};
		}
		let o = gt(i, i.extents[0], a);
		return o === void 0 ? void 0 : { tiffHeaderOffset: St(e, o) };
	} catch {
		return;
	}
}
function ht(e, t, n, r) {
	if (e) for (let i = 0; i < t.extents.length; i++) {
		let a = t.extents[i], o = gt(t, a, r);
		o !== void 0 && c(e, n, o, o + a.extentLength);
	}
}
function gt(e, t, n) {
	let r = e.constructionMethod;
	if (r === void 0 || r === 0) return e.baseOffset + t.extentOffset;
	if (r === 1 && n !== void 0) return n + e.baseOffset + t.extentOffset;
}
function _t(e) {
	let t = e.subBoxes.find((e) => e.type === "idat");
	return t ? t.contentOffset : void 0;
}
function vt(e) {
	e.constructionMethod === 2 && console.warn("This file uses iloc constructionMethod 2 (item_offset) which is currently not supported by ExifReader. Contact the maintainer to get it fixed.");
}
function yt(e, t, n) {
	let r = [], i = 0;
	for (let a = 0; a < t.extents.length; a++) {
		let o = t.extents[a], s = gt(t, o, n);
		if (s === void 0 || s + o.extentLength > e.byteLength || (r.push({
			start: s,
			length: o.extentLength
		}), i += o.extentLength, i > e.byteLength)) return;
	}
	let a = new Uint8Array(i), o = 0;
	for (let t = 0; t < r.length; t++) {
		let { start: n, length: i } = r[t];
		for (let t = 0; t < i; t++) a[o + t] = e.getUint8(n + t);
		o += i;
	}
	return new DataView(a.buffer);
}
function bt(e) {
	return e.subBoxes.find((e) => e.type === "iinf").itemInfos.find((e) => e.itemType === rt);
}
function xt(e, t) {
	return e.subBoxes.find((e) => e.type === "iloc").items.find((e) => e.itemId === t);
}
function St(e, t) {
	return t + 4 + e.getUint32(t);
}
function Ct(e, t, n) {
	try {
		let r = wt(t).itemId, i = xt(t, r);
		vt(i);
		let a = _t(t);
		if (ht(n, i, "xmp", a), i.extents.length > 1) {
			let t = yt(e, i, a);
			return t === void 0 ? void 0 : {
				chunks: [{
					dataOffset: 0,
					length: t.byteLength
				}],
				dataView: t
			};
		}
		let o = i.extents[0], s = gt(i, o, a);
		return s === void 0 ? void 0 : { chunks: [{
			dataOffset: s,
			length: o.extentLength
		}] };
	} catch {
		return;
	}
}
function wt(e) {
	return e.subBoxes.find((e) => e.type === "iinf").itemInfos.find((e) => e.itemType === 1835625829 && e.contentType === "application/rdf+xml");
}
function Tt(e, t) {
	try {
		let n = e.subBoxes.find((e) => e.type === "iprp").subBoxes.find((e) => e.type === "ipco").properties.find((e) => e.type === "colr").icc;
		if (n) return c(t, "icc", n.offset, n.offset + n.length), [n];
	} catch {}
}
function Et(e, t, n) {
	return {
		type: "ftyp",
		majorBrand: r(e, t, 4),
		length: n
	};
}
function Dt(e, t, n, r, i) {
	return {
		type: "iprp",
		subBoxes: Nt(e, n, r - (n - t), i),
		length: r
	};
}
function Ot(e, t, n, r, i) {
	return {
		type: "ipco",
		properties: Nt(e, n, r - (n - t), i),
		length: r
	};
}
function kt(e, t, n) {
	return {
		type: "colr",
		icc: At(e, t),
		length: n
	};
}
function At(e, t) {
	let n = r(e, t, 4);
	if (n === "prof" || n === "rICC") return {
		offset: t + 4,
		length: e.getUint32(t + 4),
		chunkNumber: 1,
		chunksTotal: 1
	};
}
function jt(e, t, n, r, i) {
	return {
		type: "meta",
		subBoxes: Nt(e, n + 3, r - (n + 3 - t), i),
		length: r
	};
}
function Mt(e, t) {
	return {
		type: "idat",
		contentOffset: e,
		length: t
	};
}
function Nt(e, t, n, r) {
	let i = [rt, it], a = Math.min(t + n, r), o = [], s = t;
	for (; s + 8 <= a;) {
		let t = ot(e, s, a);
		if (t === void 0) break;
		t.type !== void 0 && (t.itemType === void 0 || i.indexOf(t.itemType) !== -1) && o.push(t), s += t.length;
	}
	return o;
}
function Pt(e, t, n, r, i, a) {
	let o = n === 0 ? 2 : 4, s = r + 3 + o;
	return {
		type: "iinf",
		itemInfos: Nt(e, s, i - (s - t), a),
		length: i
	};
}
function Ft(e, t, n, r, a) {
	r += 3;
	let o = {
		type: "infe",
		length: a
	};
	return (n === 0 || n === 1) && (o.itemId = e.getUint16(r), r += 2, o.itemProtectionIndex = e.getUint16(r), r += 2, o.itemName = i(e, r), r += o.itemName.length + 1), n >= 2 && (n === 2 ? (o.itemId = e.getUint16(r), r += 2) : n === 3 && (o.itemId = e.getUint32(r), r += 4), o.itemProtectionIndex = e.getUint16(r), r += 2, o.itemType = e.getUint32(r), r += 4, o.itemName = i(e, r), r += o.itemName.length + 1, o.itemType === 1835625829 ? (o.contentType = i(e, r), r += o.contentType.length + 1, t + a > r && (o.contentEncoding = i(e, r), r += o.contentEncoding.length + 1)) : o.itemType === at && (o.itemUri = i(e, r), r += o.itemUri.length + 1)), o;
}
//#endregion
//#region node_modules/exifreader/src/image-header-tiff.js
var It = {
	isTiffFile: Lt,
	findTiffOffsets: zt
};
function Lt(e) {
	return !!e && e.byteLength >= 4 && Rt(e);
}
function Rt(e) {
	let t = e.getUint16(0) === A.LITTLE_ENDIAN;
	return e.getUint16(2, t) === 42;
}
function zt() {
	return k.USE_EXIF ? {
		hasAppMarkers: !0,
		tiffHeaderOffset: 0
	} : {};
}
//#endregion
//#region node_modules/exifreader/src/image-header-jpeg.js
var Bt = {
	isJpegFile: xn,
	findJpegOffsets: Sn
}, Vt = 2, Ht = 65496, Ut = 2, j = 4, M = 2, Wt = 2, Gt = 10, Kt = 18, qt = 33, Jt = 79, Yt = 18, Xt = 8, Zt = "ICC_PROFILE\0", Qt = 16, $t = 17, en = "MPF\0", tn = 65472, nn = 65474, rn = 65476, an = 65499, on = 65501, sn = 65498, cn = 65504, ln = 65505, un = 65506, dn = 65517, fn = 65519, pn = 65534, mn = 65535, hn = "JFIF", gn = "Exif", _n = "http://ns.adobe.com/xap/1.0/\0", vn = "http://ns.adobe.com/xmp/extension/\0", yn = "Photoshop 3.0", bn = "8BIM";
function xn(e) {
	return !!e && e.byteLength >= Vt && e.getUint16(0) === Ht;
}
function Sn(e, t) {
	let n = Ut, r, i, a, o, s, l, u, d, f, p = 0, m, h = !1;
	for (; n + j + 5 <= e.byteLength;) {
		let g;
		if (e.getUint16(n) === sn && (h = !0), k.USE_FILE && Cn(e, n)) r = e.getUint16(n + M), i = n + M, g = "file";
		else if (k.USE_FILE && wn(e, n)) r = e.getUint16(n + M), a = n + M, g = "file";
		else if (k.USE_JFIF && Dn(e, n)) r = e.getUint16(n + M), o = n + Wt, g = "jfif";
		else if (k.USE_EXIF && On(e, n)) {
			r = e.getUint16(n + M), p++;
			let t = n + Gt;
			if (p === 1) m = {
				appMarkerPosition: n,
				fieldLength: r,
				tiffHeaderOffset: t,
				score: void 0
			};
			else {
				m.score === void 0 && (m.score = Rn(e, m.appMarkerPosition, m.fieldLength, m.tiffHeaderOffset));
				let i = Rn(e, n, r, t);
				i > m.score && (m = {
					appMarkerPosition: n,
					fieldLength: r,
					tiffHeaderOffset: t,
					score: i
				});
			}
			s = m.tiffHeaderOffset, g = "exif";
		} else if (k.USE_XMP && kn(e, n)) u ||= [], r = e.getUint16(n + M), u.push(Nn(n, r)), g = "xmp";
		else if (k.USE_XMP && jn(e, n)) u ||= [], r = e.getUint16(n + M), u.push(Pn(n, r)), g = "xmp";
		else if (k.USE_IPTC && Fn(e, n)) r = e.getUint16(n + M), l = n + Kt, g = "iptc";
		else if (k.USE_ICC && Tn(e, n)) {
			r = e.getUint16(n + M);
			let t = n + Yt, i = r - 16, a = e.getUint8(n + Qt), o = e.getUint8(n + $t);
			d ||= [], d.push({
				offset: t,
				length: i,
				chunkNumber: a,
				chunksTotal: o
			}), g = "icc";
		} else if (k.USE_MPF && En(e, n)) r = e.getUint16(n + M), f = n + Xt, g = "mpf";
		else if (In(e, n)) r = e.getUint16(n + M);
		else if (Ln(e, n)) {
			n++;
			continue;
		} else break;
		c(t, g, n, n + M + r), n += M + r;
	}
	return p > 1 && Bn(p), t && (!h && n + M <= e.byteLength && e.getUint16(n) === sn && (h = !0), t.truncated = !h), {
		hasAppMarkers: n > Ut,
		fileDataOffset: i || a,
		jfifDataOffset: o,
		tiffHeaderOffset: s,
		iptcDataOffset: l,
		xmpChunks: u,
		iccChunks: d,
		mpfDataOffset: f
	};
}
function Cn(e, t) {
	return e.getUint16(t) === tn;
}
function wn(e, t) {
	return e.getUint16(t) === nn;
}
function Tn(e, t) {
	return e.getUint16(t) === un && r(e, t + j, 12) === Zt;
}
function En(e, t) {
	return e.getUint16(t) === un && r(e, t + j, 4) === en;
}
function Dn(e, t) {
	return e.getUint16(t) === cn && r(e, t + j, 4) === hn && e.getUint8(t + j + 4) === 0;
}
function On(e, t) {
	return e.getUint16(t) === ln && r(e, t + j, 4) === gn && e.getUint8(t + j + 4) === 0;
}
function kn(e, t) {
	return e.getUint16(t) === ln && An(e, t);
}
function An(e, t) {
	return r(e, t + j, 29) === _n;
}
function jn(e, t) {
	return e.getUint16(t) === ln && Mn(e, t);
}
function Mn(e, t) {
	return r(e, t + j, 35) === vn;
}
function Nn(e, t) {
	return {
		dataOffset: e + qt,
		length: t - 31
	};
}
function Pn(e, t) {
	return {
		dataOffset: e + Jt,
		length: t - 77
	};
}
function Fn(e, t) {
	let n = t + j + 13 + 1;
	return e.getUint16(t) === dn && r(e, t + j, 13) === yn && e.getUint8(t + j + 13) === 0 && r(e, n, 4) === bn;
}
function In(e, t) {
	let n = e.getUint16(t);
	return n >= cn && n <= fn || n === pn || n === tn || n === nn || n === rn || n === an || n === on || n === sn;
}
function Ln(e, t) {
	return e.getUint16(t) === mn;
}
function Rn(e, t, n, r) {
	let i = zn(e, r, t + M + n);
	if (!i) return 0;
	let a = Number.isInteger(i.ifdEntries) ? i.ifdEntries : 0;
	return (i.isValid ? 1e9 : 0) + a * 1e6 + n;
}
function zn(e, t, n) {
	try {
		if (t + 8 > n) return;
		let r = e.getUint16(t + 0), i = r === A.LITTLE_ENDIAN;
		if (!i && r !== A.BIG_ENDIAN || e.getUint16(t + 2, i) !== 42) return;
		let a = t + e.getUint32(t + 4, i);
		if (a + 2 > n) return;
		let o = e.getUint16(a, i);
		return {
			ifdEntries: o,
			isValid: a + (2 + o * 12 + 4) <= n && o > 0
		};
	} catch {
		return;
	}
}
function Bn(e) {
	typeof console > "u" || typeof console.warn != "function" || console.warn(`ExifReader: Found ${e} Exif segments (APP1). Will pick the best candidate segment.`);
}
//#endregion
//#region node_modules/exifreader/src/image-header-png.js
var Vn = {
	isPngFile: Yn,
	findPngOffsets: Xn
}, Hn = "PNG\r\n\n", Un = 4, Wn = "XML:com.adobe.xmp\0", Gn = "pHYs", Kn = "tIME", qn = "eXIf", Jn = "iCCP";
function Yn(e) {
	return !!e && r(e, 0, 8) === Hn;
}
function Xn(e, t, n) {
	let i = { hasAppMarkers: !1 }, a = 8, o = !1;
	for (; a + Un + 4 <= e.byteLength;) {
		let s, l = e.getUint32(a + 0);
		if (n && r(e, a + 4, 4) === "IEND" && (o = !0), k.USE_PNG_FILE && Zn(e, a)) i.hasAppMarkers = !0, i.pngHeaderOffset = a + 8, s = "file";
		else if (k.USE_XMP && Qn(e, a)) {
			let t = rr(e, a);
			t !== void 0 && (i.hasAppMarkers = !0, i.xmpChunks = [{
				dataOffset: t,
				length: l - (t - (a + 8))
			}], s = "xmp");
		} else if ($n(e, a, t)) {
			i.hasAppMarkers = !0;
			let t = r(e, a + 4, 4);
			i.pngTextChunks ||= [], i.pngTextChunks.push({
				length: l,
				type: t,
				offset: a + 8
			}), s = "png";
		} else if (er(e, a)) i.hasAppMarkers = !0, i.tiffHeaderOffset = a + 8, s = "exif";
		else if (k.USE_ICC && t && tr(e, a)) {
			let t = a + 8, n = ir(e, t);
			n !== void 0 && (i.hasAppMarkers = !0, i.iccChunks ||= [], i.iccChunks.push({
				offset: n.compressedProfileOffset,
				length: l - (n.compressedProfileOffset - t),
				chunkNumber: 1,
				chunksTotal: 1,
				profileName: n.profileName,
				compressionMethod: n.compressionMethod
			}), s = "icc");
		} else nr(e, a) && (i.hasAppMarkers = !0, i.pngChunkOffsets ||= [], i.pngChunkOffsets.push(a + 0), s = "png");
		let u = l + Un + 4 + 4;
		c(n, s, a, a + u), a += u;
	}
	return n && (n.truncated = !o), i;
}
function Zn(e, t) {
	return r(e, t + 4, 4) === "IHDR";
}
function Qn(e, t) {
	return r(e, t + 4, 4) === "iTXt" && r(e, t + 8, 18) === Wn;
}
function $n(e, t, n) {
	let i = r(e, t + 4, 4);
	return i === "tEXt" || i === "iTXt" || i === "zTXt" && n;
}
function er(e, t) {
	return r(e, t + 4, 4) === qn;
}
function tr(e, t) {
	return r(e, t + 4, 4) === Jn;
}
function nr(e, t) {
	let n = [Gn, Kn], i = r(e, t + 4, 4);
	return n.includes(i);
}
function rr(e, t) {
	t += 28;
	let n = 0;
	for (; n < 2 && t < e.byteLength;) e.getUint8(t) === 0 && n++, t++;
	if (!(n < 2)) return t;
}
function ir(e, t) {
	let n = i(e, t);
	if (t += n.length + 1, t + 1 > e.byteLength) return;
	let r = e.getUint8(t);
	return t += 1, {
		profileName: n,
		compressionMethod: r,
		compressedProfileOffset: t
	};
}
//#endregion
//#region node_modules/exifreader/src/image-header-heic.js
var ar = {
	isHeicFile: sr,
	findHeicOffsets: cr
}, or = [
	"heic",
	"heix",
	"hevc",
	"hevx",
	"heim",
	"heis",
	"hevm",
	"hevs",
	"mif1"
];
function sr(e) {
	return st(e, or);
}
function cr(e, t) {
	return ft(e, t);
}
//#endregion
//#region node_modules/exifreader/src/image-header-avif.js
var lr = {
	isAvifFile: dr,
	findAvifOffsets: fr
}, ur = ["avif"];
function dr(e) {
	return st(e, ur);
}
function fr(e, t) {
	return ft(e, t);
}
//#endregion
//#region node_modules/exifreader/src/image-header-jxl.js
var pr = {
	isJxlFile: Cr,
	findJxlOffsets: Er
}, mr = [
	0,
	0,
	0,
	12,
	74,
	88,
	76,
	32,
	13,
	10,
	135,
	10
], hr = [255, 10], gr = 1165519206, _r = 2020437024, vr = 1651666786, yr = 1786276963, br = 1786276976, xr = 4, Sr = 4;
function Cr(e) {
	try {
		return wr(e) || Tr(e);
	} catch {
		return !1;
	}
}
function wr(e) {
	if (!e || e.byteLength < mr.length) return !1;
	for (let t = 0; t < mr.length; t++) if (e.getUint8(t) !== mr[t]) return !1;
	return !0;
}
function Tr(e) {
	return !e || e.byteLength < hr.length ? !1 : e.getUint8(0) === hr[0] && e.getUint8(1) === hr[1];
}
function Er(e, t) {
	let n = 0, r, i, a, o, s;
	if (Tr(e)) return {
		hasAppMarkers: !0,
		tiffHeaderOffset: void 0,
		xmpChunks: void 0,
		brobExifChunk: void 0,
		brobXmpChunk: void 0,
		jxlCodestreamOffset: 0
	};
	for (; n + 8 <= e.byteLength;) {
		let { length: l, contentOffset: u } = ct(e, n);
		if (l < 8) break;
		let d = e.getUint32(n + 4), f;
		if (k.USE_EXIF && d === gr) {
			try {
				r = St(e, u);
			} catch {}
			f = "exif";
		}
		if (k.USE_XMP && d === _r && (i = [{
			dataOffset: u,
			length: l - (u - n)
		}], f = "xmp"), d === yr && s === void 0 && (s = u), d === br && s === void 0 && u + xr <= e.byteLength && (e.getUint32(u) & 2147483647 || (s = u + xr)), d === vr && u + Sr <= e.byteLength) {
			let t = e.getUint32(u), s = u + Sr, c = l - (s - n);
			k.USE_EXIF && t === gr && (f = "exif", r === void 0 && !a && (a = {
				dataOffset: s,
				length: c
			})), k.USE_XMP && t === _r && (f = "xmp", !i && !o && (o = {
				dataOffset: s,
				length: c
			}));
		}
		c(t, f, n, n + l), n += l;
	}
	return {
		hasAppMarkers: r !== void 0 || i !== void 0 || a !== void 0 || o !== void 0 || s !== void 0,
		tiffHeaderOffset: r,
		xmpChunks: i,
		brobExifChunk: a,
		brobXmpChunk: o,
		jxlCodestreamOffset: s
	};
}
//#endregion
//#region node_modules/exifreader/src/image-header-webp.js
var Dr = {
	isWebpFile: Or,
	findOffsets: kr
};
function Or(e) {
	return !!e && r(e, 0, 4) === "RIFF" && r(e, 8, 4) === "WEBP";
}
function kr(e, t) {
	let n = 12, i = !1, a, o, s, l;
	for (; n + 8 < e.byteLength;) {
		let u = r(e, n, 4), d = e.getUint32(n + 4, !0), f;
		k.USE_EXIF && u === "EXIF" ? (i = !0, a = r(e, n + 8, 6) === "Exif\0\0" ? n + 8 + 6 : n + 8, f = "exif") : k.USE_XMP && u === "XMP " ? (i = !0, o = [{
			dataOffset: n + 8,
			length: d
		}], f = "xmp") : k.USE_ICC && u === "ICCP" ? (i = !0, s = [{
			offset: n + 8,
			length: d,
			chunkNumber: 1,
			chunksTotal: 1
		}], f = "icc") : u === "VP8X" && (i = !0, l = n + 8, f = "riff");
		let p = 8 + (d % 2 == 0 ? d : d + 1);
		c(t, f, n, n + p), n += p;
	}
	return {
		hasAppMarkers: i,
		tiffHeaderOffset: a,
		xmpChunks: o,
		iccChunks: s,
		vp8xChunkOffset: l
	};
}
//#endregion
//#region node_modules/exifreader/src/image-header-gif.js
var Ar = {
	isGifFile: Pr,
	findOffsets: Fr
}, jr = 6, Mr = ["GIF87a", "GIF89a"], Nr = 13;
function Pr(e) {
	return !!e && Mr.includes(r(e, 0, jr));
}
function Fr(e, t) {
	return c(t, "gif", 0, Nr), { gifHeaderOffset: 0 };
}
//#endregion
//#region node_modules/exifreader/src/xml.js
var Ir = {
	isXMLFile: Br,
	findOffsets: Vr
}, Lr = 0, Rr = "<?xpacket begin", zr = ["<?xpacket end=", "</x:xmpmeta>"];
function Br(e) {
	return !!e && r(e, Lr, 15) === Rr;
}
function Vr(e, t) {
	let n = [];
	return n.push({
		dataOffset: Lr,
		length: e.byteLength
	}), c(t, "xmp", 0, e.byteLength), t && (t.truncated = !Hr(e)), { xmpChunks: n };
}
function Hr(e) {
	let t = r(e, 0, e.byteLength);
	for (let e = 0; e < zr.length; e++) if (t.indexOf(zr[e]) !== -1) return !0;
	return !1;
}
//#endregion
//#region node_modules/exifreader/src/image-header.js
var Ur = { parseAppMarkers: Wr };
function Wr(e, t, n) {
	let r = n ? [] : void 0;
	if (k.USE_TIFF && It.isTiffFile(e)) return N(It.findTiffOffsets(), "tiff", "TIFF", r);
	if (k.USE_JPEG && Bt.isJpegFile(e)) return N(Bt.findJpegOffsets(e, r), "jpeg", "JPEG", r);
	if (k.USE_PNG && Vn.isPngFile(e)) return N(Vn.findPngOffsets(e, t, r), "png", "PNG", r);
	if (k.USE_HEIC && ar.isHeicFile(e)) return N(ar.findHeicOffsets(e, r), "heic", "HEIC", r);
	if (k.USE_AVIF && lr.isAvifFile(e)) return N(lr.findAvifOffsets(e, r), "avif", "AVIF", r);
	if (k.USE_JXL && pr.isJxlFile(e)) return N(pr.findJxlOffsets(e, r), "jxl", "JPEG XL", r);
	if (k.USE_WEBP && Dr.isWebpFile(e)) return N(Dr.findOffsets(e, r), "webp", "WebP", r);
	if (k.USE_GIF && Ar.isGifFile(e)) return N(Ar.findOffsets(e, r), "gif", "GIF", r);
	if (k.USE_XMP && Ir.isXMLFile(e)) return N(Ir.findOffsets(e, r), "xml", "XML", r);
	throw Error("Invalid image format");
}
function N(e, t, n, r) {
	let i = u({}, e, { fileType: {
		value: t,
		description: n
	} });
	return r !== void 0 && (i.metadataBlocks = r, i.metadataTruncated = !!r.truncated), i;
}
//#endregion
//#region node_modules/exifreader/src/tag-names-common.js
var P = {
	ApertureValue: (e) => (Math.sqrt(2) ** +(e[0] / e[1])).toFixed(2),
	ColorSpace(e) {
		return e === 1 ? "sRGB" : e === 65535 ? "Uncalibrated" : "Unknown";
	},
	ComponentsConfiguration(e) {
		return e.map((e) => {
			if (e === 49) return "Y";
			if (e === 50) return "Cb";
			if (e === 51) return "Cr";
			if (e === 52) return "R";
			if (e === 53) return "G";
			if (e === 54) return "B";
		}).join("");
	},
	Contrast(e) {
		return e === 0 ? "Normal" : e === 1 ? "Soft" : e === 2 ? "Hard" : "Unknown";
	},
	CustomRendered(e) {
		return e === 0 ? "Normal process" : e === 1 ? "Custom process" : "Unknown";
	},
	ExposureMode(e) {
		return e === 0 ? "Auto exposure" : e === 1 ? "Manual exposure" : e === 2 ? "Auto bracket" : "Unknown";
	},
	ExposureProgram(e) {
		return e === 0 ? "Undefined" : e === 1 ? "Manual" : e === 2 ? "Normal program" : e === 3 ? "Aperture priority" : e === 4 ? "Shutter priority" : e === 5 ? "Creative program" : e === 6 ? "Action program" : e === 7 ? "Portrait mode" : e === 8 ? "Landscape mode" : e === 9 ? "Bulb" : "Unknown";
	},
	ExposureTime(e) {
		if (e[0] / e[1] > .25) {
			let t = e[0] / e[1];
			return Number.isInteger(t) ? "" + t : t.toFixed(1);
		}
		return e[0] === 0 ? `0/${e[1]}` : `1/${Math.round(e[1] / e[0])}`;
	},
	FNumber: (e) => `f/${Number(e[0] / e[1]).toFixed(1)}`,
	FocalLength: (e) => e[0] / e[1] + " mm",
	FocalPlaneResolutionUnit(e) {
		return e === 2 ? "inches" : e === 3 ? "centimeters" : e === 4 ? "millimeters" : "Unknown";
	},
	LightSource: (e) => e === 1 ? "Daylight" : e === 2 ? "Fluorescent" : e === 3 ? "Tungsten (incandescent light)" : e === 4 ? "Flash" : e === 9 ? "Fine weather" : e === 10 ? "Cloudy weather" : e === 11 ? "Shade" : e === 12 ? "Daylight fluorescent (D 5700 – 7100K)" : e === 13 ? "Day white fluorescent (N 4600 – 5400K)" : e === 14 ? "Cool white fluorescent (W 3900 – 4500K)" : e === 15 ? "White fluorescent (WW 3200 – 3700K)" : e === 17 ? "Standard light A" : e === 18 ? "Standard light B" : e === 19 ? "Standard light C" : e === 20 ? "D55" : e === 21 ? "D65" : e === 22 ? "D75" : e === 23 ? "D50" : e === 24 ? "ISO studio tungsten" : e === 255 ? "Other light source" : "Unknown",
	MeteringMode(e) {
		return e === 1 ? "Average" : e === 2 ? "CenterWeightedAverage" : e === 3 ? "Spot" : e === 4 ? "MultiSpot" : e === 5 ? "Pattern" : e === 6 ? "Partial" : e === 255 ? "Other" : "Unknown";
	},
	ResolutionUnit(e) {
		return e === 2 ? "inches" : e === 3 ? "centimeters" : "Unknown";
	},
	Saturation(e) {
		return e === 0 ? "Normal" : e === 1 ? "Low saturation" : e === 2 ? "High saturation" : "Unknown";
	},
	FocalLengthIn35mmFilm(e) {
		return e === 0 ? "Unknown" : e + " mm";
	},
	SceneCaptureType(e) {
		return e === 0 ? "Standard" : e === 1 ? "Landscape" : e === 2 ? "Portrait" : e === 3 ? "Night scene" : "Unknown";
	},
	Sharpness(e) {
		return e === 0 ? "Normal" : e === 1 ? "Soft" : e === 2 ? "Hard" : "Unknown";
	},
	ShutterSpeedValue(e) {
		let t = 2 ** (e[0] / e[1]);
		return t <= 1 ? `${Math.round(1 / t)}` : `1/${Math.round(t)}`;
	},
	WhiteBalance(e) {
		return e === 0 ? "Auto white balance" : e === 1 ? "Manual white balance" : "Unknown";
	},
	XResolution: (e) => "" + Math.round(e[0] / e[1]),
	YResolution: (e) => "" + Math.round(e[0] / e[1])
}, Gr = {
	11: "ProcessingSoftware",
	254: {
		name: "SubfileType",
		description: (e) => ({
			0: "Full-resolution image",
			1: "Reduced-resolution image",
			2: "Single page of multi-page image",
			3: "Single page of multi-page reduced-resolution image",
			4: "Transparency mask",
			5: "Transparency mask of reduced-resolution image",
			6: "Transparency mask of multi-page image",
			7: "Transparency mask of reduced-resolution multi-page image",
			65537: "Alternate reduced-resolution image",
			4294967295: "Invalid"
		})[e] || "Unknown"
	},
	255: {
		name: "OldSubfileType",
		description: (e) => ({
			0: "Full-resolution image",
			1: "Reduced-resolution image",
			2: "Single page of multi-page image"
		})[e] || "Unknown"
	},
	256: "ImageWidth",
	257: "ImageLength",
	258: "BitsPerSample",
	259: "Compression",
	262: "PhotometricInterpretation",
	263: {
		name: "Thresholding",
		description: (e) => ({
			1: "No dithering or halftoning",
			2: "Ordered dither or halfton",
			3: "Randomized dither"
		})[e] || "Unknown"
	},
	264: "CellWidth",
	265: "CellLength",
	266: {
		name: "FillOrder",
		description: (e) => ({
			1: "Normal",
			2: "Reversed"
		})[e] || "Unknown"
	},
	269: "DocumentName",
	270: "ImageDescription",
	271: "Make",
	272: "Model",
	273: "StripOffsets",
	274: {
		name: "Orientation",
		description: (e) => e === 1 ? "top-left" : e === 2 ? "top-right" : e === 3 ? "bottom-right" : e === 4 ? "bottom-left" : e === 5 ? "left-top" : e === 6 ? "right-top" : e === 7 ? "right-bottom" : e === 8 ? "left-bottom" : "Undefined"
	},
	277: "SamplesPerPixel",
	278: "RowsPerStrip",
	279: "StripByteCounts",
	280: "MinSampleValue",
	281: "MaxSampleValue",
	282: {
		name: "XResolution",
		description: P.XResolution
	},
	283: {
		name: "YResolution",
		description: P.YResolution
	},
	284: "PlanarConfiguration",
	285: "PageName",
	286: {
		name: "XPosition",
		description: (e) => "" + Math.round(e[0] / e[1])
	},
	287: {
		name: "YPosition",
		description: (e) => "" + Math.round(e[0] / e[1])
	},
	290: {
		name: "GrayResponseUnit",
		description: (e) => ({
			1: "0.1",
			2: "0.001",
			3: "0.0001",
			4: "1e-05",
			5: "1e-06"
		})[e] || "Unknown"
	},
	296: {
		name: "ResolutionUnit",
		description: P.ResolutionUnit
	},
	297: "PageNumber",
	301: "TransferFunction",
	305: "Software",
	306: "DateTime",
	315: "Artist",
	316: "HostComputer",
	317: "Predictor",
	318: {
		name: "WhitePoint",
		description: (e) => e.map((e) => `${e[0]}/${e[1]}`).join(", ")
	},
	319: {
		name: "PrimaryChromaticities",
		description: (e) => e.map((e) => `${e[0]}/${e[1]}`).join(", ")
	},
	321: "HalftoneHints",
	322: "TileWidth",
	323: "TileLength",
	330: "A100DataOffset",
	332: {
		name: "InkSet",
		description: (e) => ({
			1: "CMYK",
			2: "Not CMYK"
		})[e] || "Unknown"
	},
	337: "TargetPrinter",
	338: {
		name: "ExtraSamples",
		description: (e) => ({
			0: "Unspecified",
			1: "Associated Alpha",
			2: "Unassociated Alpha"
		})[e] || "Unknown"
	},
	339: {
		name: "SampleFormat",
		description: (e) => {
			let t = {
				1: "Unsigned",
				2: "Signed",
				3: "Float",
				4: "Undefined",
				5: "Complex int",
				6: "Complex float"
			};
			return Array.isArray(e) ? e.map((e) => t[e] || "Unknown").join(", ") : "Unknown";
		}
	},
	513: "JPEGInterchangeFormat",
	514: "JPEGInterchangeFormatLength",
	529: {
		name: "YCbCrCoefficients",
		description: (e) => e.map((e) => "" + e[0] / e[1]).join("/")
	},
	530: "YCbCrSubSampling",
	531: {
		name: "YCbCrPositioning",
		description: (e) => e === 1 ? "centered" : e === 2 ? "co-sited" : "undefined " + e
	},
	532: {
		name: "ReferenceBlackWhite",
		description: (e) => e.map((e) => "" + e[0] / e[1]).join(", ")
	},
	700: "ApplicationNotes",
	18246: "Rating",
	18249: "RatingPercent",
	33432: {
		name: "Copyright",
		description: (e) => e.join("; ")
	},
	33550: "PixelScale",
	33723: "IPTC-NAA",
	33920: "IntergraphMatrix",
	33922: "ModelTiePoint",
	34118: "SEMInfo",
	34264: "ModelTransform",
	34377: "PhotoshopSettings",
	34665: "Exif IFD Pointer",
	34675: "ICC_Profile",
	34735: "GeoTiffDirectory",
	34736: "GeoTiffDoubleParams",
	34737: "GeoTiffAsciiParams",
	34853: "GPS Info IFD Pointer",
	40091: {
		name: "XPTitle",
		description: F
	},
	40092: {
		name: "XPComment",
		description: F
	},
	40093: {
		name: "XPAuthor",
		description: F
	},
	40094: {
		name: "XPKeywords",
		description: F
	},
	40095: {
		name: "XPSubject",
		description: F
	},
	42112: "GDALMetadata",
	42113: "GDALNoData",
	50341: "PrintIM",
	50707: "DNGBackwardVersion",
	50708: "UniqueCameraModel",
	50709: "LocalizedCameraModel",
	50721: "ColorMatrix1",
	50722: "ColorMatrix2",
	50723: "CameraCalibration1",
	50724: "CameraCalibration2",
	50725: "ReductionMatrix1",
	50726: "ReductionMatrix2",
	50727: "AnalogBalance",
	50728: "AsShotNeutral",
	50729: "AsShotWhiteXY",
	50730: "BaselineExposure",
	50731: "BaselineNoise",
	50732: "BaselineSharpness",
	50734: "LinearResponseLimit",
	50735: "CameraSerialNumber",
	50736: "DNGLensInfo",
	50739: "ShadowScale",
	50741: {
		name: "MakerNoteSafety",
		description: (e) => ({
			0: "Unsafe",
			1: "Safe"
		})[e] || "Unknown"
	},
	50778: {
		name: "CalibrationIlluminant1",
		description: P.LightSource
	},
	50779: {
		name: "CalibrationIlluminant2",
		description: P.LightSource
	},
	50781: "RawDataUniqueID",
	50827: "OriginalRawFileName",
	50828: "OriginalRawFileData",
	50831: "AsShotICCProfile",
	50832: "AsShotPreProfileMatrix",
	50833: "CurrentICCProfile",
	50834: "CurrentPreProfileMatrix",
	50879: "ColorimetricReference",
	50885: "SRawType",
	50898: "PanasonicTitle",
	50899: "PanasonicTitle2",
	50931: "CameraCalibrationSig",
	50932: "ProfileCalibrationSig",
	50933: "ProfileIFD",
	50934: "AsShotProfileName",
	50936: "ProfileName",
	50937: "ProfileHueSatMapDims",
	50938: "ProfileHueSatMapData1",
	50939: "ProfileHueSatMapData2",
	50940: "ProfileToneCurve",
	50941: {
		name: "ProfileEmbedPolicy",
		description: (e) => ({
			0: "Allow Copying",
			1: "Embed if Used",
			2: "Never Embed",
			3: "No Restrictions"
		})[e] || "Unknown"
	},
	50942: "ProfileCopyright",
	50964: "ForwardMatrix1",
	50965: "ForwardMatrix2",
	50966: "PreviewApplicationName",
	50967: "PreviewApplicationVersion",
	50968: "PreviewSettingsName",
	50969: "PreviewSettingsDigest",
	50970: {
		name: "PreviewColorSpace",
		description: (e) => ({
			1: "Gray Gamma 2.2",
			2: "sRGB",
			3: "Adobe RGB",
			4: "ProPhoto RGB"
		})[e] || "Unknown"
	},
	50971: "PreviewDateTime",
	50972: "RawImageDigest",
	50973: "OriginalRawFileDigest",
	50981: "ProfileLookTableDims",
	50982: "ProfileLookTableData",
	51043: "TimeCodes",
	51044: "FrameRate",
	51058: "TStop",
	51081: "ReelName",
	51089: "OriginalDefaultFinalSize",
	51090: "OriginalBestQualitySize",
	51091: "OriginalDefaultCropSize",
	51105: "CameraLabel",
	51107: {
		name: "ProfileHueSatMapEncoding",
		description: (e) => ({
			0: "Linear",
			1: "sRGB"
		})[e] || "Unknown"
	},
	51108: {
		name: "ProfileLookTableEncoding",
		description: (e) => ({
			0: "Linear",
			1: "sRGB"
		})[e] || "Unknown"
	},
	51109: "BaselineExposureOffset",
	51110: {
		name: "DefaultBlackRender",
		description: (e) => ({
			0: "Auto",
			1: "None"
		})[e] || "Unknown"
	},
	51111: "NewRawImageDigest",
	51112: "RawToPreviewGain"
};
function F(e) {
	return new TextDecoder("utf-16").decode(new Uint8Array(e)).replace(/\u0000+$/, "");
}
//#endregion
//#region node_modules/exifreader/src/tag-names-exif-ifd.js
var Kr = {
	33434: {
		name: "ExposureTime",
		description: P.ExposureTime
	},
	33437: {
		name: "FNumber",
		description: P.FNumber
	},
	34850: {
		name: "ExposureProgram",
		description: P.ExposureProgram
	},
	34852: "SpectralSensitivity",
	34855: "ISOSpeedRatings",
	34856: {
		name: "OECF",
		description: () => "[Raw OECF table data]"
	},
	34858: "TimeZoneOffset",
	34859: "SelfTimerMode",
	34864: {
		name: "SensitivityType",
		description: (e) => ({
			1: "Standard Output Sensitivity",
			2: "Recommended Exposure Index",
			3: "ISO Speed",
			4: "Standard Output Sensitivity and Recommended Exposure Index",
			5: "Standard Output Sensitivity and ISO Speed",
			6: "Recommended Exposure Index and ISO Speed",
			7: "Standard Output Sensitivity, Recommended Exposure Index and ISO Speed"
		})[e] || "Unknown"
	},
	34865: "StandardOutputSensitivity",
	34866: "RecommendedExposureIndex",
	34867: "ISOSpeed",
	34868: "ISOSpeedLatitudeyyy",
	34869: "ISOSpeedLatitudezzz",
	36864: {
		name: "ExifVersion",
		description: (e) => s(e)
	},
	36867: "DateTimeOriginal",
	36868: "DateTimeDigitized",
	36873: "GooglePlusUploadCode",
	36880: "OffsetTime",
	36881: "OffsetTimeOriginal",
	36882: "OffsetTimeDigitized",
	37121: {
		name: "ComponentsConfiguration",
		description: P.ComponentsConfiguration
	},
	37122: "CompressedBitsPerPixel",
	37377: {
		name: "ShutterSpeedValue",
		description: P.ShutterSpeedValue
	},
	37378: {
		name: "ApertureValue",
		description: P.ApertureValue
	},
	37379: "BrightnessValue",
	37380: "ExposureBiasValue",
	37381: {
		name: "MaxApertureValue",
		description: (e) => (Math.sqrt(2) ** +(e[0] / e[1])).toFixed(2)
	},
	37382: {
		name: "SubjectDistance",
		description: (e) => e[0] / e[1] + " m"
	},
	37383: {
		name: "MeteringMode",
		description: P.MeteringMode
	},
	37384: {
		name: "LightSource",
		description: P.LightSource
	},
	37385: {
		name: "Flash",
		description: (e) => e === 0 ? "Flash did not fire" : e === 1 ? "Flash fired" : e === 5 ? "Strobe return light not detected" : e === 7 ? "Strobe return light detected" : e === 9 ? "Flash fired, compulsory flash mode" : e === 13 ? "Flash fired, compulsory flash mode, return light not detected" : e === 15 ? "Flash fired, compulsory flash mode, return light detected" : e === 16 ? "Flash did not fire, compulsory flash mode" : e === 24 ? "Flash did not fire, auto mode" : e === 25 ? "Flash fired, auto mode" : e === 29 ? "Flash fired, auto mode, return light not detected" : e === 31 ? "Flash fired, auto mode, return light detected" : e === 32 ? "No flash function" : e === 65 ? "Flash fired, red-eye reduction mode" : e === 69 ? "Flash fired, red-eye reduction mode, return light not detected" : e === 71 ? "Flash fired, red-eye reduction mode, return light detected" : e === 73 ? "Flash fired, compulsory flash mode, red-eye reduction mode" : e === 77 ? "Flash fired, compulsory flash mode, red-eye reduction mode, return light not detected" : e === 79 ? "Flash fired, compulsory flash mode, red-eye reduction mode, return light detected" : e === 89 ? "Flash fired, auto mode, red-eye reduction mode" : e === 93 ? "Flash fired, auto mode, return light not detected, red-eye reduction mode" : e === 95 ? "Flash fired, auto mode, return light detected, red-eye reduction mode" : "Unknown"
	},
	37386: {
		name: "FocalLength",
		description: P.FocalLength
	},
	37393: "ImageNumber",
	37394: {
		name: "SecurityClassification",
		description: (e) => ({
			C: "Confidential",
			R: "Restricted",
			S: "Secret",
			T: "Top Secret",
			U: "Unclassified"
		})[e] || "Unknown"
	},
	37395: "ImageHistory",
	37396: {
		name: "SubjectArea",
		description: (e) => e.length === 2 ? `Location; X: ${e[0]}, Y: ${e[1]}` : e.length === 3 ? `Circle; X: ${e[0]}, Y: ${e[1]}, diameter: ${e[2]}` : e.length === 4 ? `Rectangle; X: ${e[0]}, Y: ${e[1]}, width: ${e[2]}, height: ${e[3]}` : "Unknown"
	},
	37500: {
		name: "MakerNote",
		description: () => "[Raw maker note data]"
	},
	37510: {
		name: "UserComment",
		description: je
	},
	37520: "SubSecTime",
	37521: "SubSecTimeOriginal",
	37522: "SubSecTimeDigitized",
	37724: "ImageSourceData",
	37888: {
		name: "AmbientTemperature",
		description: (e) => e[0] / e[1] + " °C"
	},
	37889: {
		name: "Humidity",
		description: (e) => e[0] / e[1] + " %"
	},
	37890: {
		name: "Pressure",
		description: (e) => e[0] / e[1] + " hPa"
	},
	37891: {
		name: "WaterDepth",
		description: (e) => e[0] / e[1] + " m"
	},
	37892: {
		name: "Acceleration",
		description: (e) => e[0] / e[1] + " mGal"
	},
	37893: {
		name: "CameraElevationAngle",
		description: (e) => e[0] / e[1] + " °"
	},
	40960: {
		name: "FlashpixVersion",
		description: (e) => e.map((e) => String.fromCharCode(e)).join("")
	},
	40961: {
		name: "ColorSpace",
		description: P.ColorSpace
	},
	40962: "PixelXDimension",
	40963: "PixelYDimension",
	40964: "RelatedSoundFile",
	40965: "Interoperability IFD Pointer",
	41483: "FlashEnergy",
	41484: {
		name: "SpatialFrequencyResponse",
		description: () => "[Raw SFR table data]"
	},
	41486: "FocalPlaneXResolution",
	41487: "FocalPlaneYResolution",
	41488: {
		name: "FocalPlaneResolutionUnit",
		description: P.FocalPlaneResolutionUnit
	},
	41492: {
		name: "SubjectLocation",
		description: ([e, t]) => `X: ${e}, Y: ${t}`
	},
	41493: "ExposureIndex",
	41495: {
		name: "SensingMethod",
		description: (e) => e === 1 ? "Undefined" : e === 2 ? "One-chip color area sensor" : e === 3 ? "Two-chip color area sensor" : e === 4 ? "Three-chip color area sensor" : e === 5 ? "Color sequential area sensor" : e === 7 ? "Trilinear sensor" : e === 8 ? "Color sequential linear sensor" : "Unknown"
	},
	41728: {
		name: "FileSource",
		description: (e) => e === 3 ? "DSC" : "Unknown"
	},
	41729: {
		name: "SceneType",
		description: (e) => e === 1 ? "A directly photographed image" : "Unknown"
	},
	41730: {
		name: "CFAPattern",
		description: () => "[Raw CFA pattern table data]"
	},
	41985: {
		name: "CustomRendered",
		description: P.CustomRendered
	},
	41986: {
		name: "ExposureMode",
		description: P.ExposureMode
	},
	41987: {
		name: "WhiteBalance",
		description: P.WhiteBalance
	},
	41988: {
		name: "DigitalZoomRatio",
		description: (e) => e[0] === 0 ? "Digital zoom was not used" : "" + e[0] / e[1]
	},
	41989: {
		name: "FocalLengthIn35mmFilm",
		description: P.FocalLengthIn35mmFilm
	},
	41990: {
		name: "SceneCaptureType",
		description: P.SceneCaptureType
	},
	41991: {
		name: "GainControl",
		description: (e) => e === 0 ? "None" : e === 1 ? "Low gain up" : e === 2 ? "High gain up" : e === 3 ? "Low gain down" : e === 4 ? "High gain down" : "Unknown"
	},
	41992: {
		name: "Contrast",
		description: P.Contrast
	},
	41993: {
		name: "Saturation",
		description: P.Saturation
	},
	41994: {
		name: "Sharpness",
		description: P.Sharpness
	},
	41995: {
		name: "DeviceSettingDescription",
		description: () => "[Raw device settings table data]"
	},
	41996: {
		name: "SubjectDistanceRange",
		description: (e) => e === 1 ? "Macro" : e === 2 ? "Close view" : e === 3 ? "Distant view" : "Unknown"
	},
	42016: "ImageUniqueID",
	42032: "CameraOwnerName",
	42033: "BodySerialNumber",
	42034: {
		name: "LensSpecification",
		description: (e) => {
			let t = `${parseFloat((e[0][0] / e[0][1]).toFixed(5))}-${parseFloat((e[1][0] / e[1][1]).toFixed(5))} mm`;
			if (e[3][1] === 0) return `${t} f/?`;
			let n = 1 / (e[2][1] / e[2][1] / (e[3][0] / e[3][1]));
			return `${t} f/${parseFloat(n.toFixed(5))}`;
		}
	},
	42035: "LensMake",
	42036: "LensModel",
	42037: "LensSerialNumber",
	42080: {
		name: "CompositeImage",
		description: (e) => ({
			1: "Not a Composite Image",
			2: "General Composite Image",
			3: "Composite Image Captured While Shooting"
		})[e] || "Unknown"
	},
	42081: "SourceImageNumberOfCompositeImage",
	42082: "SourceExposureTimesOfCompositeImage",
	42240: "Gamma",
	59932: "Padding",
	59933: "OffsetSchema",
	65e3: "OwnerName",
	65001: "SerialNumber",
	65002: "Lens",
	65100: "RawFile",
	65101: "Converter",
	65102: "WhiteBalance",
	65105: "Exposure",
	65106: "Shadows",
	65107: "Brightness",
	65108: "Contrast",
	65109: "Saturation",
	65110: "Sharpness",
	65111: "Smoothness",
	65112: "MoireFilter"
}, qr = {
	0: {
		name: "GPSVersionID",
		description: (e) => e[0] === 2 && e[1] === 2 && e[2] === 0 && e[3] === 0 ? "Version 2.2" : "Unknown"
	},
	1: {
		name: "GPSLatitudeRef",
		description: (e) => {
			let t = e.join("");
			return t === "N" ? "North latitude" : t === "S" ? "South latitude" : "Unknown";
		}
	},
	2: {
		name: "GPSLatitude",
		description: Me
	},
	3: {
		name: "GPSLongitudeRef",
		description: (e) => {
			let t = e.join("");
			return t === "E" ? "East longitude" : t === "W" ? "West longitude" : "Unknown";
		}
	},
	4: {
		name: "GPSLongitude",
		description: Me
	},
	5: {
		name: "GPSAltitudeRef",
		description: (e) => e === 0 ? "Sea level" : e === 1 ? "Sea level reference (negative value)" : "Unknown"
	},
	6: {
		name: "GPSAltitude",
		description: (e) => e[0] / e[1] + " m"
	},
	7: {
		name: "GPSTimeStamp",
		description: (e) => e.map(([e, t]) => {
			let n = e / t;
			return /^\d(\.|$)/.test(`${n}`) ? `0${n}` : n;
		}).join(":")
	},
	8: "GPSSatellites",
	9: {
		name: "GPSStatus",
		description: (e) => {
			let t = e.join("");
			return t === "A" ? "Measurement in progress" : t === "V" ? "Measurement Interoperability" : "Unknown";
		}
	},
	10: {
		name: "GPSMeasureMode",
		description: (e) => {
			let t = e.join("");
			return t === "2" ? "2-dimensional measurement" : t === "3" ? "3-dimensional measurement" : "Unknown";
		}
	},
	11: "GPSDOP",
	12: {
		name: "GPSSpeedRef",
		description: (e) => {
			let t = e.join("");
			return t === "K" ? "Kilometers per hour" : t === "M" ? "Miles per hour" : t === "N" ? "Knots" : "Unknown";
		}
	},
	13: "GPSSpeed",
	14: {
		name: "GPSTrackRef",
		description: (e) => {
			let t = e.join("");
			return t === "T" ? "True direction" : t === "M" ? "Magnetic direction" : "Unknown";
		}
	},
	15: "GPSTrack",
	16: {
		name: "GPSImgDirectionRef",
		description: (e) => {
			let t = e.join("");
			return t === "T" ? "True direction" : t === "M" ? "Magnetic direction" : "Unknown";
		}
	},
	17: "GPSImgDirection",
	18: "GPSMapDatum",
	19: {
		name: "GPSDestLatitudeRef",
		description: (e) => {
			let t = e.join("");
			return t === "N" ? "North latitude" : t === "S" ? "South latitude" : "Unknown";
		}
	},
	20: {
		name: "GPSDestLatitude",
		description: (e) => e[0][0] / e[0][1] + e[1][0] / e[1][1] / 60 + e[2][0] / e[2][1] / 3600
	},
	21: {
		name: "GPSDestLongitudeRef",
		description: (e) => {
			let t = e.join("");
			return t === "E" ? "East longitude" : t === "W" ? "West longitude" : "Unknown";
		}
	},
	22: {
		name: "GPSDestLongitude",
		description: (e) => e[0][0] / e[0][1] + e[1][0] / e[1][1] / 60 + e[2][0] / e[2][1] / 3600
	},
	23: {
		name: "GPSDestBearingRef",
		description: (e) => {
			let t = e.join("");
			return t === "T" ? "True direction" : t === "M" ? "Magnetic direction" : "Unknown";
		}
	},
	24: "GPSDestBearing",
	25: {
		name: "GPSDestDistanceRef",
		description: (e) => {
			let t = e.join("");
			return t === "K" ? "Kilometers" : t === "M" ? "Miles" : t === "N" ? "Knots" : "Unknown";
		}
	},
	26: "GPSDestDistance",
	27: {
		name: "GPSProcessingMethod",
		description: je
	},
	28: {
		name: "GPSAreaInformation",
		description: je
	},
	29: "GPSDateStamp",
	30: {
		name: "GPSDifferential",
		description: (e) => e === 0 ? "Measurement without differential correction" : e === 1 ? "Differential correction applied" : "Unknown"
	},
	31: "GPSHPositioningError"
}, Jr = {
	1: "InteroperabilityIndex",
	2: {
		name: "InteroperabilityVersion",
		description: (e) => s(e)
	},
	4096: "RelatedImageFileFormat",
	4097: "RelatedImageWidth",
	4098: "RelatedImageHeight"
}, Yr = {
	45056: {
		name: "MPFVersion",
		description: (e) => s(e)
	},
	45057: "NumberOfImages",
	45058: "MPEntry",
	45059: "ImageUIDList",
	45060: "TotalFrames"
}, Xr = {
	1: {
		name: "CameraSettings",
		description: (e) => e
	},
	4: {
		name: "ShotInfo",
		description: (e) => e
	},
	149: "LensModel"
}, Zr = {
	0: {
		name: "PentaxVersion",
		description: (e) => e.join(".")
	},
	5: "PentaxModelID",
	555: "LevelInfo"
}, Qr = u({}, Gr, Kr), $r = "exif", ei = "interoperability", ti = "canon", ni = "pentax", I = {
	"0th": Qr,
	"1st": Gr,
	[$r]: Qr,
	gps: qr,
	[ei]: Jr,
	mpf: k.USE_MPF ? Yr : {},
	[ti]: k.USE_MAKER_NOTES ? Xr : {},
	[ni]: k.USE_MAKER_NOTES ? Zr : {}
}, ri = {
	1: 1,
	2: 1,
	3: 2,
	4: 4,
	5: 8,
	7: 1,
	9: 4,
	10: 8,
	13: 4
}, ii = {
	BYTE: 1,
	ASCII: 2,
	SHORT: 3,
	LONG: 4,
	RATIONAL: 5,
	UNDEFINED: 7,
	SLONG: 9,
	SRATIONAL: 10,
	IFD: 13
}, L = {
	getAsciiValue: ai,
	getByteAt: oi,
	getAsciiAt: si,
	getShortAt: ci,
	getLongAt: li,
	getRationalAt: ui,
	getUndefinedAt: di,
	getSlongAt: fi,
	getSrationalAt: pi,
	getIfdPointerAt: mi,
	typeSizes: ri,
	tagTypes: ii,
	getTypeSize: hi
};
function ai(e) {
	return e.map((e) => String.fromCharCode(e));
}
function oi(e, t) {
	return e.getUint8(t);
}
function si(e, t) {
	return e.getUint8(t);
}
function ci(e, t, n) {
	return e.getUint16(t, n === A.LITTLE_ENDIAN);
}
function li(e, t, n) {
	return e.getUint32(t, n === A.LITTLE_ENDIAN);
}
function ui(e, t, n) {
	return [li(e, t, n), li(e, t + 4, n)];
}
function di(e, t) {
	return oi(e, t);
}
function fi(e, t, n) {
	return e.getInt32(t, n === A.LITTLE_ENDIAN);
}
function pi(e, t, n) {
	return [fi(e, t, n), fi(e, t + 4, n)];
}
function mi(e, t, n) {
	return li(e, t, n);
}
function hi(e) {
	if (ii[e] === void 0) throw Error("No such type found.");
	return ri[ii[e]];
}
//#endregion
//#region node_modules/exifreader/src/tag-filter-config.js
var gi = {
	exif: !0,
	iptc: !0,
	photoshop: !0,
	mpf: !0,
	makerNotes: !0
}, _i = {
	exif: !0,
	iptc: !0,
	xmp: !0,
	icc: !0,
	photoshop: !0,
	makerNotes: !0,
	mpf: !0,
	file: !0,
	jfif: !0,
	png: !0,
	riff: !0,
	gif: !0,
	gps: !0,
	composite: !0,
	thumbnail: !0
}, R = {
	exifIfdPointer: "Exif IFD Pointer",
	gpsInfoIfdPointer: "GPS Info IFD Pointer",
	interoperabilityIfdPointer: "Interoperability IFD Pointer"
}, vi = {
	thumbnail: ["JPEGInterchangeFormat", "JPEGInterchangeFormatLength"],
	iptc: ["IPTC-NAA"],
	xmp: ["ApplicationNotes"],
	icc: ["ICC_Profile"],
	photoshop: ["ImageSourceData", "PhotoshopSettings"],
	makerNotes: ["MakerNote", "Make"],
	gps: [
		"GPSLatitude",
		"GPSLatitudeRef",
		"GPSLongitude",
		"GPSLongitudeRef",
		"GPSAltitude",
		"GPSAltitudeRef"
	]
}, yi = {
	file: ["Image Width", "Image Height"],
	exif: [
		"FocalLength",
		"FocalPlaneXResolution",
		"FocalPlaneYResolution",
		"FocalPlaneResolutionUnit",
		"FocalLengthIn35mmFilm"
	]
};
function bi(e) {
	let t = Object.create(null);
	if (!e) return t;
	Array.isArray(e.exif) && e.exif.length > 0 && (t[R.exifIfdPointer] = !0, n(t, e.exif));
	for (let n in vi) if (xi(e[n])) {
		let e = vi[n];
		for (let n = 0; n < e.length; n++) t[e[n]] = !0;
		t[R.exifIfdPointer] = !0;
	}
	return xi(e.gps) && (t[R.gpsInfoIfdPointer] = !0, t[R.exifIfdPointer] = !0), t;
	function n(e, t) {
		for (let n = 0; n < t.length; n++) {
			let r = t[n];
			if (typeof r != "string") continue;
			let i = r.toLowerCase();
			i.indexOf("gps") === 0 && (e[R.gpsInfoIfdPointer] = !0), (i.indexOf("interoperability") === 0 || i.indexOf("relatedimage") === 0) && (e[R.interoperabilityIfdPointer] = !0);
		}
	}
}
function xi(e) {
	return e === !0 || Array.isArray(e) && e.length > 0;
}
//#endregion
//#region node_modules/exifreader/src/tag-filter.js
function Si({ includeTags: e, excludeTags: t } = {}) {
	let n = !!e, r = !!t, i = n || r, a = Object.create(null), o = Object.create(null), s = Object.create(null);
	if (!i) return Ci();
	let c = n && (B(e, "iptc") || B(e, "xmp") || B(e, "icc") || B(e, "photoshop") || B(e, "makerNotes") || B(e, "thumbnail") || B(e, "gps") || B(e, "composite")), l = n && B(e, "composite"), u = Object.create(null);
	n && (u = wi(e));
	let d = Object.create(null);
	n && l && (d = Mi(yi.file));
	for (let i in _i) {
		let f = n && Object.prototype.hasOwnProperty.call(e, i), p = r && Object.prototype.hasOwnProperty.call(t, i), m;
		p && !f && (m = t[i]);
		let h = Oi({
			groupKey: i,
			includeValue: f ? e[i] : void 0,
			excludeValue: m,
			extraIncludeNames: Object.create(null)
		});
		a[i] = !n || f, f && Di(e[i]) && (a[i] = !1), h.excludeAll && (a[i] = !1), o[i] = h, s[i] = Oi({
			groupKey: i,
			includeValue: Ti({
				groupKey: i,
				hasIncludeTags: n,
				hasIncludeEntry: f,
				includeTags: e,
				shouldParseExif: c,
				shouldParseFile: l
			}),
			excludeValue: m,
			extraIncludeNames: Ei({
				groupKey: i,
				hasIncludeTags: n,
				hasIncludeEntry: f,
				shouldParseExif: c,
				shouldParseFile: l,
				exifIncludeDependencies: u,
				fileIncludeDependencies: d
			})
		});
	}
	return {
		isActive: i,
		shouldReturnGroup: f,
		shouldParseGroup: p,
		shouldReturnTag: m,
		shouldParseTag: h
	};
	function f(e) {
		return !_i[e] || !!a[e];
	}
	function p(e) {
		return !_i[e] || f(e) ? !0 : e === "exif" ? c : e === "file" && l;
	}
	function m(e, t, n) {
		return _i[e] ? f(e) ? ki(o[e], t, n) : !1 : !0;
	}
	function h(e, t, n) {
		return _i[e] ? p(e) ? ki(s[e], t, n) : !1 : !0;
	}
}
function Ci() {
	return {
		isActive: !1,
		shouldReturnGroup: e,
		shouldParseGroup: e,
		shouldReturnTag: e,
		shouldParseTag: e
	};
	function e() {
		return !0;
	}
}
var z = Ci();
function wi(e) {
	let t = bi(e);
	return B(e, "composite") && Ni(t, yi.exif), t;
}
function Ti({ groupKey: e, hasIncludeTags: t, hasIncludeEntry: n, includeTags: r, shouldParseExif: i, shouldParseFile: a }) {
	if (!t) return !0;
	if (n) return e === "thumbnail" && Array.isArray(r[e]) ? !0 : r[e];
	if (e === "exif" && i || e === "file" && a) return [];
}
function Ei({ groupKey: e, hasIncludeTags: t, hasIncludeEntry: n, shouldParseExif: r, shouldParseFile: i, exifIncludeDependencies: a, fileIncludeDependencies: o }) {
	return t ? e === "exif" && (r || n && Object.keys(a).length > 0) ? a : e === "file" && i ? o : Object.create(null) : Object.create(null);
}
function B(e, t) {
	if (!e || !Object.prototype.hasOwnProperty.call(e, t)) return !1;
	let n = e[t];
	return n === !0 || Array.isArray(n) && n.length > 0;
}
function Di(e) {
	return Array.isArray(e) && e.length === 0;
}
function Oi({ groupKey: e, includeValue: t, excludeValue: n, extraIncludeNames: r }) {
	let i = !!gi[e], a = {
		includeAll: !1,
		includeNames: void 0,
		includeIds: void 0,
		excludeAll: !1,
		excludeNames: void 0,
		excludeIds: void 0
	};
	if (t === !0) a.includeAll = !0;
	else if (Array.isArray(t)) {
		a.includeNames = Object.create(null), i && (a.includeIds = Object.create(null));
		for (let e = 0; e < t.length; e++) o(a, t[e]);
		for (let e in r) a.includeNames[e.toLowerCase()] = !0;
	} else if (r && Object.keys(r).length > 0) {
		a.includeNames = Object.create(null);
		for (let e in r) a.includeNames[e.toLowerCase()] = !0;
	}
	if (n === !0) return a.excludeAll = !0, a;
	if (Array.isArray(n)) {
		a.excludeNames = Object.create(null), i && (a.excludeIds = Object.create(null));
		for (let e = 0; e < n.length; e++) s(a, n[e]);
	}
	return a;
	function o(e, t) {
		if (typeof t == "number" && e.includeIds) {
			e.includeIds[String(t)] = !0;
			return;
		}
		typeof t == "string" && (e.includeNames[t.toLowerCase()] = !0);
	}
	function s(e, t) {
		if (typeof t == "number" && e.excludeIds) {
			e.excludeIds[String(t)] = !0;
			return;
		}
		typeof t == "string" && (e.excludeNames[t.toLowerCase()] = !0);
	}
}
function ki(e, t, n) {
	return e.excludeAll ? !1 : e.includeAll || !e.includeNames && !e.includeIds || Ai(e, t, n) ? !ji(e, t, n) : !1;
}
function Ai(e, t, n) {
	return !!(n !== void 0 && e.includeIds && e.includeIds[String(n)] || t && e.includeNames && e.includeNames[String(t).toLowerCase()]);
}
function ji(e, t, n) {
	return !!(n !== void 0 && e.excludeIds && e.excludeIds[String(n)] || t && e.excludeNames && e.excludeNames[String(t).toLowerCase()]);
}
function Mi(e) {
	let t = Object.create(null);
	for (let n = 0; n < e.length; n++) t[e[n].toLowerCase()] = !0;
	return t;
}
function Ni(e, t) {
	for (let n = 0; n < t.length; n++) e[t[n]] = !0;
}
//#endregion
//#region node_modules/exifreader/src/tags-helpers.js
var Pi = 4, Fi = {
	1: L.getByteAt,
	2: L.getAsciiAt,
	3: L.getShortAt,
	4: L.getLongAt,
	5: L.getRationalAt,
	7: L.getUndefinedAt,
	9: L.getSlongAt,
	10: L.getSrationalAt,
	13: L.getIfdPointerAt
};
function Ii(e, t, n) {
	let r = t + 4;
	if (!(r + L.getTypeSize("LONG") > e.byteLength)) return t + L.getLongAt(e, r, n);
}
function V(e, t, n, r, i, a, o = !1, s = z, c = "exif", l = Li(e)) {
	let u = L.getTypeSize("SHORT"), d = {}, f = Ri(e, r, i);
	r += u;
	for (let u = 0; u < f && !(r + 12 > e.byteLength); u++) {
		let u = zi(e, t, n, r, i, a, s, c, l);
		u !== void 0 && (d[u.name] = {
			id: u.id,
			value: u.value,
			description: u.description
		}, o && (d[u.name].computed = Ji(u.tagType, u.value)), (u.name === "MakerNote" || t === "pentax" && u.name === "LevelInfo") && (d[u.name].__offset = u.__offset)), r += 12;
	}
	if (k.USE_THUMBNAIL && r < e.byteLength - L.getTypeSize("LONG")) {
		let c = L.getLongAt(e, r, i);
		c !== 0 && t === "0th" && s.shouldParseGroup("thumbnail") && (d.Thumbnail = V(e, "1st", n, n + c, i, a, o, s, "thumbnail", l));
	}
	return d;
}
function Li(e) {
	return { remaining: e.byteLength * Pi };
}
function Ri(e, t, n) {
	return t + L.getTypeSize("SHORT") <= e.byteLength ? L.getShortAt(e, t, n) : 0;
}
function zi(e, t, n, r, i, a = !1, o = z, s = "exif", c) {
	let l = L.getTypeSize("SHORT"), u = l + L.getTypeSize("SHORT"), d = u + L.getTypeSize("LONG"), f = L.getShortAt(e, r, i), p = L.getShortAt(e, r + l, i), m = L.getLongAt(e, r + u, i), h, g;
	if (L.typeSizes[p] === void 0 || !a && I[t][f] === void 0) return;
	let _ = Bi(t, f);
	if (!o.shouldParseTag(s, _, f)) return;
	if (Vi(p, m)) g = r + d, h = Hi(e, g, p, m, i);
	else if (g = L.getLongAt(e, r + d, i), Ui(e, n, g, p, m)) {
		let t = f === 33723, r = Wi(c.remaining, p, m);
		c.remaining -= r * L.typeSizes[p], h = Hi(e, n + g, p, r, i, t);
	} else h = "<faulty value>";
	p === L.tagTypes.ASCII && (h = Gi(h), h = Ki(h));
	let v = h;
	if (I[t][f] !== void 0) {
		if (I[t][f].name !== void 0 && I[t][f].description !== void 0) try {
			v = I[t][f].description(h);
		} catch {
			v = qi(h);
		}
		else v = p === L.tagTypes.RATIONAL || p === L.tagTypes.SRATIONAL ? "" + h[0] / h[1] : qi(h);
	}
	return {
		id: f,
		name: _,
		value: h,
		description: v,
		tagType: p,
		__offset: g
	};
}
function Bi(e, t) {
	if (I[e][t] !== void 0) {
		if (typeof I[e][t] == "string") return I[e][t];
		if (I[e][t].name) return I[e][t].name;
	}
	return `undefined-${t}`;
}
function Vi(e, t) {
	return L.typeSizes[e] * t <= L.getTypeSize("LONG");
}
function Hi(e, t, n, r, i, a = !1) {
	let o = [];
	a && (r *= L.typeSizes[n], n = L.tagTypes.BYTE);
	for (let a = 0; a < r; a++) o.push(Fi[n](e, t, i)), t += L.typeSizes[n];
	return n === L.tagTypes.ASCII ? o = L.getAsciiValue(o) : o.length === 1 && (o = o[0]), o;
}
function Ui(e, t, n, r, i) {
	return t + n + L.typeSizes[r] * i <= e.byteLength;
}
function Wi(e, t, n) {
	let r = Math.min(n, Math.floor(e / L.typeSizes[t]));
	return r === 1 && n > 1 ? 0 : r;
}
function Gi(e) {
	let t = [], n = 0;
	for (let r = 0; r < e.length; r++) {
		if (e[r] === "\0") {
			n++;
			continue;
		}
		t[n] === void 0 && (t[n] = ""), t[n] += e[r];
	}
	return t;
}
function Ki(e) {
	try {
		return e.map((e) => decodeURIComponent(escape(e)));
	} catch {
		return e;
	}
}
function qi(e) {
	return e instanceof Array ? e.join(", ") : e;
}
function Ji(e, t) {
	return e === L.tagTypes.ASCII ? Array.isArray(t) && t.length === 1 ? t[0] : t : e === L.tagTypes.RATIONAL || e === L.tagTypes.SRATIONAL ? Yi(t) ? Xi(t) : Array.isArray(t) ? t.map((e) => Xi(e)) : t : t;
}
function Yi(e) {
	return !Array.isArray(e) || e.length !== 2 ? !1 : typeof e[0] == "number" && typeof e[1] == "number";
}
function Xi(e) {
	if (!Array.isArray(e) || e.length !== 2) return e;
	let t = e[0], n = e[1];
	return !Number.isFinite(t) || !Number.isFinite(n) ? e : n === 0 ? null : t / n;
}
//#endregion
//#region node_modules/exifreader/src/tags.js
var Zi = [
	{
		pointerKey: "Exif IFD Pointer",
		ifdType: $r
	},
	{
		pointerKey: "GPS Info IFD Pointer",
		ifdType: "gps"
	},
	{
		pointerKey: "Interoperability IFD Pointer",
		ifdType: ei
	}
], Qi = { read: $i };
function $i(e, t, n, r = !1, i = void 0) {
	let a = A.getByteOrder(e, t), o = Li(e), s = ea(e, t, a, n, r, i, o);
	for (let c = 0; c < Zi.length; c++) s = ta(Zi[c], s, e, t, a, n, r, i, o);
	return {
		tags: s,
		byteOrder: a
	};
}
function ea(e, t, n, r, i, a, o) {
	let s = Ii(e, t, n);
	return s === void 0 ? {} : V(e, "0th", t, s, n, r, i, a, "exif", o);
}
function ta(e, t, n, r, i, a, o, s, c) {
	let l = t[e.pointerKey];
	return l === void 0 ? t : u(t, V(n, e.ifdType, r, r + l.value, i, a, o, s, "exif", c));
}
//#endregion
//#region node_modules/exifreader/src/mpf-tags.js
var na = { read: ia }, H = 16, ra = 8;
function ia(e, t, n, r = !1, i = void 0) {
	try {
		let a = A.getByteOrder(e, t), o = Ii(e, t, a);
		return o === void 0 ? {} : aa(e, t, V(e, "mpf", t, o, a, n, r, i, "mpf"), a);
	} catch {
		return {};
	}
}
function aa(e, t, n, r) {
	if (!n.MPEntry) return n;
	let i = e.buffer.byteLength, a = i * ra, o = [];
	for (let s = 0; s < Math.ceil(n.MPEntry.value.length / H); s++) {
		o[s] = {};
		let c = oa(n.MPEntry.value, s * H, L.getTypeSize("LONG"), r);
		o[s].ImageFlags = sa(c), o[s].ImageFormat = ca(c), o[s].ImageType = la(c);
		let l = oa(n.MPEntry.value, s * H + 4, L.getTypeSize("LONG"), r);
		o[s].ImageSize = {
			value: l,
			description: "" + l
		};
		let u = ua(s, n.MPEntry, r, t);
		o[s].ImageOffset = {
			value: u,
			description: "" + u
		};
		let d = oa(n.MPEntry.value, s * H + 12, L.getTypeSize("SHORT"), r);
		o[s].DependentImage1EntryNumber = {
			value: d,
			description: "" + d
		};
		let m = oa(n.MPEntry.value, s * H + 14, L.getTypeSize("SHORT"), r);
		o[s].DependentImage2EntryNumber = {
			value: m,
			description: "" + m
		};
		let h = Math.min(Math.max(u, 0), i), g = Math.min(h + Math.max(l, 0), i, h + a);
		o[s].image = e.buffer.slice(h, g), a -= g - h, f(o[s], "base64", function() {
			return p(this.image);
		});
	}
	return n.Images = o, n;
}
function oa(e, t, n, r) {
	if (r === A.LITTLE_ENDIAN) {
		let r = 0;
		for (let i = 0; i < n; i++) r += e[t + i] << 8 * i;
		return r;
	}
	let i = 0;
	for (let r = 0; r < n; r++) i += e[t + r] << 8 * (n - 1 - r);
	return i;
}
function sa(e) {
	let t = [
		e >> 31 & 1,
		e >> 30 & 1,
		e >> 29 & 1
	], n = [];
	return t[0] && n.push("Dependent Parent Image"), t[1] && n.push("Dependent Child Image"), t[2] && n.push("Representative Image"), {
		value: t,
		description: n.join(", ") || "None"
	};
}
function ca(e) {
	let t = e >> 24 & 7;
	return {
		value: t,
		description: t === 0 ? "JPEG" : "Unknown"
	};
}
function la(e) {
	let t = e & 16777215;
	return {
		value: t,
		description: {
			196608: "Baseline MP Primary Image",
			65537: "Large Thumbnail (VGA equivalent)",
			65538: "Large Thumbnail (Full HD equivalent)",
			131073: "Multi-Frame Image (Panorama)",
			131074: "Multi-Frame Image (Disparity)",
			131075: "Multi-Frame Image (Multi-Angle)",
			0: "Undefined"
		}[t] || "Unknown"
	};
}
function ua(e, t, n, r) {
	return da(e) ? 0 : oa(t.value, e * H + 8, L.getTypeSize("LONG"), n) + r;
}
function da(e) {
	return e === 0;
}
//#endregion
//#region node_modules/exifreader/src/file-tags.js
var fa = { read: pa };
function pa(e, t) {
	let n = ma(e, t), r = va(e, t, n);
	return {
		"Bits Per Sample": ha(e, t, n),
		"Image Height": ga(e, t, n),
		"Image Width": _a(e, t, n),
		"Color Components": r,
		Subsampling: r && ya(e, t, r.value, n)
	};
}
function ma(e, t) {
	return L.getShortAt(e, t);
}
function ha(e, t, n) {
	if (3 > n) return;
	let r = L.getByteAt(e, t + 2);
	return {
		value: r,
		description: "" + r
	};
}
function ga(e, t, n) {
	if (5 > n) return;
	let r = L.getShortAt(e, t + 3);
	return {
		value: r,
		description: `${r}px`
	};
}
function _a(e, t, n) {
	if (7 > n) return;
	let r = L.getShortAt(e, t + 5);
	return {
		value: r,
		description: `${r}px`
	};
}
function va(e, t, n) {
	if (8 > n) return;
	let r = L.getByteAt(e, t + 7);
	return {
		value: r,
		description: "" + r
	};
}
function ya(e, t, n, r) {
	if (8 + 3 * n > r) return;
	let i = [];
	for (let r = 0; r < n; r++) {
		let n = t + 8 + r * 3;
		i.push([
			L.getByteAt(e, n),
			L.getByteAt(e, n + 1),
			L.getByteAt(e, n + 2)
		]);
	}
	return {
		value: i,
		description: i.length > 1 ? ba(i) + xa(i) : ""
	};
}
function ba(e) {
	let t = {
		1: "Y",
		2: "Cb",
		3: "Cr",
		4: "I",
		5: "Q"
	};
	return e.map((e) => t[e[0]]).join("");
}
function xa(e) {
	let t = {
		17: "4:4:4 (1 1)",
		18: "4:4:0 (1 2)",
		20: "4:4:1 (1 4)",
		33: "4:2:2 (2 1)",
		34: "4:2:0 (2 2)",
		36: "4:2:1 (2 4)",
		65: "4:1:1 (4 1)",
		66: "4:1:0 (4 2)"
	};
	return e.length === 0 || e[0][1] === void 0 || t[e[0][1]] === void 0 ? "" : t[e[0][1]];
}
//#endregion
//#region node_modules/exifreader/src/jxl-file-tags.js
var Sa = { read: Ea }, Ca = 2, wa = [
	9,
	13,
	18,
	30
], Ta = [
	0,
	1,
	12 / 10,
	4 / 3,
	3 / 2,
	16 / 9,
	5 / 4,
	2
];
function Ea(e, t) {
	let n = {};
	try {
		let { height: r, width: i } = Da(ka(e, t + Ca));
		n["Image Height"] = {
			value: r,
			description: `${r}px`
		}, n["Image Width"] = {
			value: i,
			description: `${i}px`
		};
	} catch {}
	return n;
}
function Da(e) {
	let t = e.readBits(1), n, r;
	if (t) {
		n = (e.readBits(5) + 1) * 8;
		let t = e.readBits(3);
		r = t === 0 ? (e.readBits(5) + 1) * 8 : Math.ceil(n * Ta[t]);
	} else {
		n = Oa(e);
		let t = e.readBits(3);
		r = t === 0 ? Oa(e) : Math.ceil(n * Ta[t]);
	}
	return {
		height: n,
		width: r
	};
}
function Oa(e) {
	let t = e.readBits(2);
	return 1 + e.readBits(wa[t]);
}
function ka(e, t) {
	let n = t, r = 0;
	return { readBits(t) {
		let i = 0;
		for (let a = 0; a < t; a++) {
			if (n >= e.byteLength) throw Error("Unexpected end of data");
			let t = e.getUint8(n) >> r & 1;
			i |= t << a, r++, r >= 8 && (r = 0, n++);
		}
		return i;
	} };
}
//#endregion
//#region node_modules/exifreader/src/jfif-tags.js
var Aa = { read: ja };
function ja(e, t) {
	let n = Ma(e, t), r = Ra(e, t, n), i = za(e, t, n), a = {
		"JFIF Version": Na(e, t, n),
		"Resolution Unit": Pa(e, t, n),
		XResolution: Ia(e, t, n),
		YResolution: La(e, t, n),
		"JFIF Thumbnail Width": r,
		"JFIF Thumbnail Height": i
	};
	if (r !== void 0 && i !== void 0) {
		let o = Ba(e, t, 3 * r.value * i.value, n);
		o && (a["JFIF Thumbnail"] = o);
	}
	for (let e in a) a[e] === void 0 && delete a[e];
	return a;
}
function Ma(e, t) {
	return L.getShortAt(e, t);
}
function Na(e, t, n) {
	if (9 > n) return;
	let r = L.getByteAt(e, t + 7), i = L.getByteAt(e, t + 7 + 1);
	return {
		value: r * 256 + i,
		description: r + "." + i
	};
}
function Pa(e, t, n) {
	if (10 > n) return;
	let r = L.getByteAt(e, t + 9);
	return {
		value: r,
		description: Fa(r)
	};
}
function Fa(e) {
	return e === 0 ? "None" : e === 1 ? "inches" : e === 2 ? "cm" : "Unknown";
}
function Ia(e, t, n) {
	if (12 > n) return;
	let r = L.getShortAt(e, t + 10);
	return {
		value: r,
		description: "" + r
	};
}
function La(e, t, n) {
	if (14 > n) return;
	let r = L.getShortAt(e, t + 12);
	return {
		value: r,
		description: "" + r
	};
}
function Ra(e, t, n) {
	if (15 > n) return;
	let r = L.getByteAt(e, t + 14);
	return {
		value: r,
		description: `${r}px`
	};
}
function za(e, t, n) {
	if (16 > n) return;
	let r = L.getByteAt(e, t + 15);
	return {
		value: r,
		description: `${r}px`
	};
}
function Ba(e, t, n, r) {
	if (!(n === 0 || 16 + n > r)) return {
		value: e.buffer.slice(t + 16, t + 16 + n),
		description: "<24-bit RGB pixel data>"
	};
}
//#endregion
//#region node_modules/exifreader/src/iptc-tag-names.js
var U = { iptc: {
	256: {
		name: "Model Version",
		description: (e) => ((e[0] << 8) + e[1]).toString()
	},
	261: {
		name: "Destination",
		repeatable: !0
	},
	276: {
		name: "File Format",
		description: (e) => ((e[0] << 8) + e[1]).toString()
	},
	278: {
		name: "File Format Version",
		description: (e) => ((e[0] << 8) + e[1]).toString()
	},
	286: "Service Identifier",
	296: "Envelope Number",
	306: "Product ID",
	316: "Envelope Priority",
	326: {
		name: "Date Sent",
		description: Va
	},
	336: {
		name: "Time Sent",
		description: Ha
	},
	346: {
		name: "Coded Character Set",
		description: Ua,
		encoding_name: Ua
	},
	356: "UNO",
	376: {
		name: "ARM Identifier",
		description: (e) => ((e[0] << 8) + e[1]).toString()
	},
	378: {
		name: "ARM Version",
		description: (e) => ((e[0] << 8) + e[1]).toString()
	},
	512: {
		name: "Record Version",
		description: (e) => ((e[0] << 8) + e[1]).toString()
	},
	515: "Object Type Reference",
	516: "Object Attribute Reference",
	517: "Object Name",
	519: "Edit Status",
	520: {
		name: "Editorial Update",
		description: (e) => s(e) === "01" ? "Additional Language" : "Unknown"
	},
	522: "Urgency",
	524: {
		name: "Subject Reference",
		repeatable: !0,
		description: (e) => {
			let t = s(e).split(":");
			return t[2] + (t[3] ? "/" + t[3] : "") + (t[4] ? "/" + t[4] : "");
		}
	},
	527: "Category",
	532: {
		name: "Supplemental Category",
		repeatable: !0
	},
	534: "Fixture Identifier",
	537: {
		name: "Keywords",
		repeatable: !0
	},
	538: {
		name: "Content Location Code",
		repeatable: !0
	},
	539: {
		name: "Content Location Name",
		repeatable: !0
	},
	542: "Release Date",
	547: "Release Time",
	549: "Expiration Date",
	550: "Expiration Time",
	552: "Special Instructions",
	554: {
		name: "Action Advised",
		description: (e) => {
			let t = s(e);
			return t === "01" ? "Object Kill" : t === "02" ? "Object Replace" : t === "03" ? "Object Append" : t === "04" ? "Object Reference" : "Unknown";
		}
	},
	557: {
		name: "Reference Service",
		repeatable: !0
	},
	559: {
		name: "Reference Date",
		repeatable: !0
	},
	562: {
		name: "Reference Number",
		repeatable: !0
	},
	567: {
		name: "Date Created",
		description: Va
	},
	572: {
		name: "Time Created",
		description: Ha
	},
	574: {
		name: "Digital Creation Date",
		description: Va
	},
	575: {
		name: "Digital Creation Time",
		description: Ha
	},
	577: "Originating Program",
	582: "Program Version",
	587: {
		name: "Object Cycle",
		description: (e) => {
			let t = s(e);
			return t === "a" ? "morning" : t === "p" ? "evening" : t === "b" ? "both" : "Unknown";
		}
	},
	592: {
		name: "By-line",
		repeatable: !0
	},
	597: {
		name: "By-line Title",
		repeatable: !0
	},
	602: "City",
	604: "Sub-location",
	607: "Province/State",
	612: "Country/Primary Location Code",
	613: "Country/Primary Location Name",
	615: "Original Transmission Reference",
	617: "Headline",
	622: "Credit",
	627: "Source",
	628: "Copyright Notice",
	630: {
		name: "Contact",
		repeatable: !0
	},
	632: "Caption/Abstract",
	634: {
		name: "Writer/Editor",
		repeatable: !0
	},
	637: {
		name: "Rasterized Caption",
		description: (e) => e
	},
	642: "Image Type",
	643: {
		name: "Image Orientation",
		description: (e) => {
			let t = s(e);
			return t === "P" ? "Portrait" : t === "L" ? "Landscape" : t === "S" ? "Square" : "Unknown";
		}
	},
	647: "Language Identifier",
	662: {
		name: "Audio Type",
		description: (e) => {
			let t = s(e), n = t.charAt(0), r = t.charAt(1), i = "";
			return n === "1" ? i += "Mono" : n === "2" && (i += "Stereo"), r === "A" ? i += ", actuality" : r === "C" ? i += ", question and answer session" : r === "M" ? i += ", music, transmitted by itself" : r === "Q" ? i += ", response to a question" : r === "R" ? i += ", raw sound" : r === "S" ? i += ", scener" : r === "V" ? i += ", voicer" : r === "W" && (i += ", wrap"), i === "" ? t : i;
		}
	},
	663: {
		name: "Audio Sampling Rate",
		description: (e) => parseInt(s(e), 10) + " Hz"
	},
	664: {
		name: "Audio Sampling Resolution",
		description: (e) => {
			let t = parseInt(s(e), 10);
			return t + (t === 1 ? " bit" : " bits");
		}
	},
	665: {
		name: "Audio Duration",
		description: (e) => {
			let t = s(e);
			return t.length >= 6 ? t.substr(0, 2) + ":" + t.substr(2, 2) + ":" + t.substr(4, 2) : t;
		}
	},
	666: "Audio Outcue",
	698: "Short Document ID",
	699: "Unique Document ID",
	700: "Owner ID",
	712: {
		name: (e) => e.length === 2 ? "ObjectData Preview File Format" : "Record 2 destination",
		description: (e) => {
			if (e.length === 2) {
				let t = (e[0] << 8) + e[1];
				return t === 0 ? "No ObjectData" : t === 1 ? "IPTC-NAA Digital Newsphoto Parameter Record" : t === 2 ? "IPTC7901 Recommended Message Format" : t === 3 ? "Tagged Image File Format (Adobe/Aldus Image data)" : t === 4 ? "Illustrator (Adobe Graphics data)" : t === 5 ? "AppleSingle (Apple Computer Inc)" : t === 6 ? "NAA 89-3 (ANPA 1312)" : t === 7 ? "MacBinary II" : t === 8 ? "IPTC Unstructured Character Oriented File Format (UCOFF)" : t === 9 ? "United Press International ANPA 1312 variant" : t === 10 ? "United Press International Down-Load Message" : t === 11 ? "JPEG File Interchange (JFIF)" : t === 12 ? "Photo-CD Image-Pac (Eastman Kodak)" : t === 13 ? "Microsoft Bit Mapped Graphics File [*.BMP]" : t === 14 ? "Digital Audio File [*.WAV] (Microsoft & Creative Labs)" : t === 15 ? "Audio plus Moving Video [*.AVI] (Microsoft)" : t === 16 ? "PC DOS/Windows Executable Files [*.COM][*.EXE]" : t === 17 ? "Compressed Binary File [*.ZIP] (PKWare Inc)" : t === 18 ? "Audio Interchange File Format AIFF (Apple Computer Inc)" : t === 19 ? "RIFF Wave (Microsoft Corporation)" : t === 20 ? "Freehand (Macromedia/Aldus)" : t === 21 ? "Hypertext Markup Language \"HTML\" (The Internet Society)" : t === 22 ? "MPEG 2 Audio Layer 2 (Musicom), ISO/IEC" : t === 23 ? "MPEG 2 Audio Layer 3, ISO/IEC" : t === 24 ? "Portable Document File (*.PDF) Adobe" : t === 25 ? "News Industry Text Format (NITF)" : t === 26 ? "Tape Archive (*.TAR)" : t === 27 ? "Tidningarnas Telegrambyrå NITF version (TTNITF DTD)" : t === 28 ? "Ritzaus Bureau NITF version (RBNITF DTD)" : t === 29 ? "Corel Draw [*.CDR]" : `Unknown format ${t}`;
			}
			return s(e);
		}
	},
	713: {
		name: "ObjectData Preview File Format Version",
		description: (e, t) => {
			let n = {
				"00": { "00": "1" },
				"01": {
					"01": "1",
					"02": "2",
					"03": "3",
					"04": "4"
				},
				"02": { "04": "4" },
				"03": {
					"01": "5.0",
					"02": "6.0"
				},
				"04": { "01": "1.40" },
				"05": { "01": "2" },
				"06": { "01": "1" },
				11: { "01": "1.02" },
				20: {
					"01": "3.1",
					"02": "4.0",
					"03": "5.0",
					"04": "5.5"
				},
				21: { "02": "2.0" }
			}, r = s(e);
			if (t["ObjectData Preview File Format"]) {
				let e = s(t["ObjectData Preview File Format"].value);
				if (n[e] && n[e][r]) return n[e][r];
			}
			return r;
		}
	},
	714: "ObjectData Preview Data",
	1802: {
		name: "Size Mode",
		description: (e) => e[0].toString()
	},
	1812: {
		name: "Max Subfile Size",
		description: (e) => {
			let t = 0;
			for (let n = 0; n < e.length; n++) t = (t << 8) + e[n];
			return t.toString();
		}
	},
	1882: {
		name: "ObjectData Size Announced",
		description: (e) => {
			let t = 0;
			for (let n = 0; n < e.length; n++) t = (t << 8) + e[n];
			return t.toString();
		}
	},
	1887: {
		name: "Maximum ObjectData Size",
		description: (e) => {
			let t = 0;
			for (let n = 0; n < e.length; n++) t = (t << 8) + e[n];
			return t.toString();
		}
	}
} };
function Va(e) {
	let t = s(e);
	return t.length >= 8 ? t.substr(0, 4) + "-" + t.substr(4, 2) + "-" + t.substr(6, 2) : t;
}
function Ha(e) {
	let t = s(e), n = t;
	return t.length >= 6 && (n = t.substr(0, 2) + ":" + t.substr(2, 2) + ":" + t.substr(4, 2), t.length === 11 && (n += t.substr(6, 1) + t.substr(7, 2) + ":" + t.substr(9, 2))), n;
}
function Ua(e) {
	let t = s(e);
	return t === "\x1B%G" ? "UTF-8" : t === "\x1B%5" ? "Windows-1252" : t === "\x1B%/G" ? "UTF-8 Level 1" : t === "\x1B%/H" ? "UTF-8 Level 2" : t === "\x1B%/I" ? "UTF-8 Level 3" : t === "\x1B/A" ? "ISO-8859-1" : t === "\x1B/B" ? "ISO-8859-2" : t === "\x1B/C" ? "ISO-8859-3" : t === "\x1B/D" ? "ISO-8859-4" : t === "\x1B/@" ? "ISO-8859-5" : t === "\x1B/G" ? "ISO-8859-6" : t === "\x1B/F" ? "ISO-8859-7" : t === "\x1B/H" ? "ISO-8859-8" : "Unknown";
}
//#endregion
//#region node_modules/exifreader/src/text-decoder.js
var Wa = { get: Ga };
function Ga() {
	if (typeof TextDecoder < "u") return TextDecoder;
}
//#endregion
//#region node_modules/exifreader/src/tag-decoder.js
var Ka = 5, qa = {
	128: 8364,
	130: 8218,
	131: 402,
	132: 8222,
	133: 8230,
	134: 8224,
	135: 8225,
	136: 710,
	137: 8240,
	138: 352,
	139: 8249,
	140: 338,
	142: 381,
	145: 8216,
	146: 8217,
	147: 8220,
	148: 8221,
	149: 8226,
	150: 8211,
	151: 8212,
	152: 732,
	153: 8482,
	154: 353,
	155: 8250,
	156: 339,
	158: 382,
	159: 376
}, Ja = {
	decode: Ya,
	TAG_HEADER_SIZE: Ka
};
function Ya(e, t) {
	if (typeof t == "string") return Za(t);
	let n = Wa.get();
	if (n !== void 0 && e !== void 0) try {
		return new n(e).decode(t instanceof DataView ? t.buffer : Uint8Array.from(t));
	} catch {}
	let r = t.map((e) => String.fromCharCode(e)).join("");
	try {
		return decodeURIComponent(escape(r));
	} catch {
		return Xa(t);
	}
}
function Xa(e) {
	let t = e instanceof DataView, n = t ? e.byteLength : e.length, r = t ? (t) => e.getUint8(t) : (t) => e[t], i = Array(n);
	for (let e = 0; e < n; e++) {
		let t = r(e), n = qa[t];
		i[e] = n === void 0 ? t : n;
	}
	return String.fromCharCode.apply(null, i);
}
function Za(e) {
	try {
		return decodeURIComponent(escape(e));
	} catch {
		return e;
	}
}
//#endregion
//#region node_modules/exifreader/src/iptc-tags.js
var Qa = 943868237, $a = 4, eo = 2, to = 1, no = 4, ro = 12, io = 1028, ao = 5, oo = { read: so };
function so(e, t, n, r = z) {
	try {
		if (Array.isArray(e)) return po(new DataView(Uint8Array.from(e).buffer), { size: e.length }, 0, n, r);
		let { naaBlock: i, dataOffset: a } = co(e, t);
		return po(e, i, a, n, r);
	} catch {
		return {};
	}
}
function co(e, t) {
	for (; t + ro <= e.byteLength;) {
		let n = lo(e, t);
		if (uo(n)) return {
			naaBlock: n,
			dataOffset: t + n.headerSize
		};
		t += n.headerSize + n.size + fo(n);
	}
	throw Error("No IPTC NAA resource block.");
}
function lo(e, t) {
	if (e.getUint32(t, !1) !== Qa) throw Error("Not an IPTC resource block.");
	let n = e.getUint8(t + $a + eo), r = (n % 2 == 0 ? n + 1 : n) + to;
	return {
		headerSize: 6 + r + no,
		type: e.getUint16(t + $a),
		size: e.getUint32(t + $a + eo + r)
	};
}
function uo(e) {
	return e.type === io;
}
function fo(e) {
	return e.size % 2 == 0 ? 0 : 1;
}
function po(e, t, n, r, i) {
	let a = {}, o, s = n + t.size;
	for (; n < s && n < e.byteLength;) {
		let { tag: t, tagSize: s } = mo(e, n, a, o, r, i);
		if (t === null) break;
		t && ("encoding" in t && (o = t.encoding), a[t.name] === void 0 || t.repeatable === void 0 ? a[t.name] = {
			id: t.id,
			value: t.value,
			description: t.description
		} : (a[t.name] instanceof Array || (a[t.name] = [{
			id: a[t.name].id,
			value: a[t.name].value,
			description: a[t.name].description
		}]), a[t.name].push({
			id: t.id,
			value: t.value,
			description: t.description
		}))), n += ao + s;
	}
	return a;
}
function mo(e, t, n = {}, r = void 0, i = !1, a = z) {
	if (go(e, t)) return {
		tag: null,
		tagSize: 0
	};
	let o = e.getUint16(t + 1), s = e.getUint16(t + 3);
	if (!i && !U.iptc[o] || !a.shouldParseTag("iptc", ho(o, i), o)) return {
		tag: void 0,
		tagSize: s
	};
	let c = _o(e, t + ao, s), l = {
		id: o,
		name: vo(U.iptc[o], o, c),
		value: c,
		description: xo(U.iptc[o], c, n, r)
	};
	return wo(o) && (l.repeatable = !0), To(o) && (l.encoding = U.iptc[o].encoding_name(c)), {
		tag: l,
		tagSize: s
	};
}
function ho(e, t) {
	let n = U.iptc[e];
	if (!n) return t ? `undefined-${e}` : void 0;
	if (typeof n == "string") return n;
	if (n && typeof n.name == "string") return n.name;
}
function go(e, t) {
	return e.getUint8(t) !== 28;
}
function _o(e, t, n) {
	let r = [];
	for (let i = 0; i < n; i++) r.push(e.getUint8(t + i));
	return r;
}
function vo(e, t, n) {
	return e ? yo(e) ? e : bo(e) ? e.name(n) : e.name : `undefined-${t}`;
}
function yo(e) {
	return typeof e == "string";
}
function bo(e) {
	return typeof e.name == "function";
}
function xo(e, t, n, r) {
	if (Co(e)) try {
		return e.description(t, n);
	} catch {}
	return So(e, t) ? Ja.decode(r, t) : t;
}
function So(e, t) {
	return e && t instanceof Array;
}
function Co(e) {
	return e && e.description !== void 0;
}
function wo(e) {
	return U.iptc[e] && U.iptc[e].repeatable;
}
function To(e) {
	return U.iptc[e] && U.iptc[e].encoding_name !== void 0;
}
//#endregion
//#region node_modules/exifreader/src/xmp-tag-names.js
var Eo = {
	"tiff:Orientation"(e) {
		return e === "1" ? "Horizontal (normal)" : e === "2" ? "Mirror horizontal" : e === "3" ? "Rotate 180" : e === "4" ? "Mirror vertical" : e === "5" ? "Mirror horizontal and rotate 270 CW" : e === "6" ? "Rotate 90 CW" : e === "7" ? "Mirror horizontal and rotate 90 CW" : e === "8" ? "Rotate 270 CW" : e;
	},
	"tiff:ResolutionUnit": (e) => P.ResolutionUnit(parseInt(e, 10)),
	"tiff:XResolution": (e) => W(P.XResolution, e),
	"tiff:YResolution": (e) => W(P.YResolution, e),
	"exif:ApertureValue": (e) => W(P.ApertureValue, e),
	"exif:GPSLatitude": ko,
	"exif:GPSLongitude": ko,
	"exif:FNumber": (e) => W(P.FNumber, e),
	"exif:FocalLength": (e) => W(P.FocalLength, e),
	"exif:FocalPlaneResolutionUnit": (e) => P.FocalPlaneResolutionUnit(parseInt(e, 10)),
	"exif:ColorSpace": (e) => P.ColorSpace(Do(e)),
	"exif:ComponentsConfiguration"(e, t) {
		if (/^\d, \d, \d, \d$/.test(t)) {
			let e = t.split(", ").map((e) => e.charCodeAt(0));
			return P.ComponentsConfiguration(e);
		}
		return t;
	},
	"exif:Contrast": (e) => P.Contrast(parseInt(e, 10)),
	"exif:CustomRendered": (e) => P.CustomRendered(parseInt(e, 10)),
	"exif:ExposureMode": (e) => P.ExposureMode(parseInt(e, 10)),
	"exif:ExposureProgram": (e) => P.ExposureProgram(parseInt(e, 10)),
	"exif:ExposureTime"(e) {
		return Oo(e) ? P.ExposureTime(e.split("/").map((e) => parseInt(e, 10))) : e;
	},
	"exif:MeteringMode": (e) => P.MeteringMode(parseInt(e, 10)),
	"exif:Saturation": (e) => P.Saturation(parseInt(e, 10)),
	"exif:SceneCaptureType": (e) => P.SceneCaptureType(parseInt(e, 10)),
	"exif:Sharpness": (e) => P.Sharpness(parseInt(e, 10)),
	"exif:ShutterSpeedValue": (e) => W(P.ShutterSpeedValue, e),
	"exif:WhiteBalance": (e) => P.WhiteBalance(parseInt(e, 10))
};
function W(e, t) {
	return Oo(t) ? e(t.split("/")) : t;
}
function Do(e) {
	return e.substring(0, 2) === "0x" ? parseInt(e.substring(2), 16) : parseInt(e, 10);
}
function Oo(e) {
	return /^-?\d+\/-?\d+$/.test(e);
}
function ko(e) {
	let [t, n] = e.split(",");
	if (t !== void 0 && n !== void 0) {
		let e = parseFloat(t), r = parseFloat(n), i = n.charAt(n.length - 1);
		if (!Number.isNaN(e) && !Number.isNaN(r)) return "" + (e + r / 60) + i;
	}
	return e;
}
//#endregion
//#region node_modules/exifreader/src/dom-parser.js
var Ao = { get: jo };
function jo(e) {
	if (e) return e;
	if (typeof DOMParser < "u") return new DOMParser();
	try {
		let { DOMParser: e, onErrorStopParsing: t } = __non_webpack_require__("@xmldom/xmldom");
		return new e({ onError: t });
	} catch {
		return;
	}
}
//#endregion
//#region node_modules/exifreader/src/xmp-namespaces.js
function Mo(e) {
	let t = [
		"prefix is non-null and namespace is null",
		"prefix not bound to a namespace",
		"prefix inte bundet till en namnrymd",
		/Namespace prefix .+ is not defined/
	];
	for (let n = 0; n < t.length; n++) if (new RegExp(t[n]).test(e.message)) return !0;
	return !1;
}
function No(e) {
	let t = Po(e);
	if (t === -1) return e;
	let { insertionIndex: n, attributeValueSpans: r } = Fo(e, t + 1);
	if (n === -1) return e;
	let i = Ro(e, t, n, r), a = Bo(e).filter((e) => i[e] === void 0);
	if (a.length === 0) return e;
	let o = Ho(a);
	return e.slice(0, n) + o + e.slice(n);
}
function Po(e) {
	let t = 0;
	for (; t < e.length;) {
		if (t = e.indexOf("<", t), t === -1) return -1;
		if (/[A-Za-z_]/.test(e.charAt(t + 1))) return t;
		Io(e, "<!--", t) ? t = Lo(e, t + 4, "-->") : Io(e, "<![CDATA[", t) ? t = Lo(e, t + 9, "]]>") : e.charAt(t + 1) === "?" ? t = Lo(e, t + 2, "?>") : t++;
	}
	return -1;
}
function Fo(e, t) {
	let n = [], r, i;
	for (let a = t; a < e.length; a++) {
		let t = e.charAt(a);
		if (r !== void 0) t === r && (n.push({
			start: i,
			end: a
		}), r = void 0);
		else if (t === "\"" || t === "'") r = t, i = a + 1;
		else if (t === ">") return e.charAt(a - 1) === "/" ? {
			insertionIndex: a - 1,
			attributeValueSpans: n
		} : {
			insertionIndex: a,
			attributeValueSpans: n
		};
	}
	return {
		insertionIndex: -1,
		attributeValueSpans: n
	};
}
function Io(e, t, n) {
	return e.slice(n, n + t.length) === t;
}
function Lo(e, t, n) {
	let r = e.indexOf(n, t);
	return r === -1 ? e.length : r + n.length;
}
function Ro(e, t, n, r) {
	let i = Object.create(null), a = /xmlns:([A-Za-z_][A-Za-z0-9._-]*)\s*=\s*["']([^"']*)["']/g, o, s = 0;
	for (; (o = a.exec(e)) !== null;) {
		for (; s < r.length && r[s].end <= o.index;) s++;
		if (s < r.length && r[s].start <= o.index) {
			a.lastIndex = r[s].end;
			continue;
		}
		(o[2] !== "" || zo(o.index, t, n)) && (i[o[1]] = !0);
	}
	return i;
}
function zo(e, t, n) {
	return e > t && e < n;
}
function Bo(e) {
	let t = [], n = Object.create(null), r = /(?:^|[^A-Za-z0-9._-])([A-Za-z_][A-Za-z0-9._-]*):[A-Za-z_][A-Za-z0-9._-]*/g, i;
	for (; (i = r.exec(e)) !== null;) {
		let e = i[1];
		e !== "xmlns" && e !== "xml" && n[e] === void 0 && (n[e] = !0, t.push(e));
	}
	return t;
}
var Vo = {
	xmp: "http://ns.adobe.com/xap/1.0/",
	tiff: "http://ns.adobe.com/tiff/1.0/",
	exif: "http://ns.adobe.com/exif/1.0/",
	dc: "http://purl.org/dc/elements/1.1/",
	xmpMM: "http://ns.adobe.com/xap/1.0/mm/",
	stEvt: "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#",
	stRef: "http://ns.adobe.com/xap/1.0/sType/ResourceRef#",
	photoshop: "http://ns.adobe.com/photoshop/1.0/"
};
function Ho(e) {
	let t = [];
	for (let n = 0; n < e.length; n++) {
		let r = e[n];
		t.push(" xmlns:" + r + "=\"" + Uo(r) + "\"");
	}
	return t.join("");
}
function Uo(e) {
	return Object.prototype.hasOwnProperty.call(Vo, e) ? Vo[e] : "http://fallback.namespace/" + e;
}
//#endregion
//#region node_modules/exifreader/src/xmp-tags.js
var Wo = { read: Ko }, Go = class extends Error {
	constructor(e) {
		super(e), this.name = "ParseError";
	}
};
function Ko(e, t, n) {
	let r = {};
	if (typeof e == "string") return Xo(r, e, n), r;
	let [i, a] = qo(e, t), o = Xo(r, i, n);
	if (a) {
		let i = Xo(r, a, n);
		!o && !i && (delete r._raw, Xo(r, Jo(e, t), n));
	}
	return r;
}
function qo(e, t) {
	if (t.length === 0) return [];
	let n = [Jo(e, t.slice(0, 1))];
	return t.length > 1 && n.push(Jo(e, t.slice(1))), n;
}
function Jo(e, t) {
	let n = t.map((t) => Yo(e, t)), r = n.reduce((e, t) => e + t.length, 0), i = new Uint8Array(r), a = 0;
	for (let e = 0; e < n.length; e++) i.set(n[e], a), a += n[e].length;
	return new DataView(i.buffer);
}
function Yo(e, t) {
	let n = e.buffer.byteLength, r = Math.min(Math.max(t.dataOffset, 0), n), i = Math.min(r + Math.max(t.length, 0), n);
	return new Uint8Array(e.buffer.slice(r, i));
}
function Xo(e, t, n) {
	try {
		let { doc: r, raw: i } = Zo(t, n);
		return e._raw = (e._raw || "") + i, u(e, ls(ts(es(r), !0))), !0;
	} catch {
		return !1;
	}
}
function Zo(e, t) {
	let n = Ao.get(t);
	if (!n) throw console.warn("Warning: DOMParser is not available. It is needed to be able to parse XMP tags."), Error();
	let i = typeof e == "string" ? e : r(e, 0, e.byteLength);
	return {
		doc: $o(n, Qo(i)),
		raw: i
	};
}
function Qo(e) {
	return e.replace(/^.+(<\?xpacket begin)/, "$1").replace(/(<\?xpacket end=".*"\?>).+$/, "$1");
}
function $o(e, t, n = !1) {
	try {
		let n = e.parseFromString(t, "application/xml"), r = n.getElementsByTagName("parsererror");
		if (r.length > 0) throw new Go(r[0].textContent);
		return n;
	} catch (r) {
		if (r.name === "ParseError" && Mo(r) && !n) return $o(e, No(t), !0);
		throw r;
	}
}
function es(e) {
	for (let t = 0; t < e.childNodes.length; t++) {
		if (e.childNodes[t].tagName === "x:xmpmeta") return es(e.childNodes[t]);
		if (e.childNodes[t].tagName === "rdf:RDF") return e.childNodes[t];
	}
	throw Error();
}
function ts(e, t = !1) {
	let n = ns(e);
	return rs(n) ? t ? {} : is(n[0]) : as(n);
}
function ns(e) {
	let t = [];
	for (let n = 0; n < e.childNodes.length; n++) t.push(e.childNodes[n]);
	return t;
}
function rs(e) {
	return e.length === 1 && e[0].nodeName === "#text";
}
function is(e) {
	return e.nodeValue;
}
function as(e) {
	let t = Object.create(null);
	return e.forEach((e) => {
		if (os(e)) {
			let n = ss(e);
			t[e.nodeName] === void 0 ? t[e.nodeName] = n : (Array.isArray(t[e.nodeName]) || (t[e.nodeName] = [t[e.nodeName]]), t[e.nodeName].push(n));
		}
	}), t;
}
function os(e) {
	return e.nodeName && e.nodeName !== "#text";
}
function ss(e) {
	return {
		attributes: cs(e),
		value: ts(e)
	};
}
function cs(e) {
	let t = {};
	for (let n = 0; n < e.attributes.length; n++) d(t, e.attributes[n].nodeName, decodeURIComponent(escape(e.attributes[n].value)));
	return t;
}
function ls(e) {
	let t = {};
	if (typeof e == "string") return e;
	for (let n in e) {
		let r = e[n];
		Array.isArray(r) || (r = [r]), r.forEach((e) => {
			u(t, us(e.attributes)), typeof e.value == "object" && u(t, vs(e.value));
		});
	}
	return t;
}
function us(e) {
	let t = {};
	for (let n in e) try {
		ds(n) && d(t, ps(n), {
			value: e[n],
			attributes: {},
			description: G(e[n], n)
		});
	} catch {}
	return t;
}
function ds(e) {
	return e !== "rdf:parseType" && !fs(e);
}
function fs(e) {
	return e.split(":")[0] === "xmlns";
}
function ps(e) {
	return /^MicrosoftPhoto(_\d+_)?:Rating$/i.test(e) ? "RatingPercent" : e.split(":")[1];
}
function G(e, t = void 0) {
	if (Array.isArray(e)) {
		let n = ms(e);
		if (hs(t)) try {
			let r = Eo[t](e, n);
			if (typeof r == "string") return r;
		} catch {}
		return n;
	}
	if (typeof e == "object") return gs(e);
	try {
		return hs(t) ? Eo[t](e) : decodeURIComponent(escape(e));
	} catch {
		return e;
	}
}
function ms(e) {
	return e.map((e) => e.value === void 0 ? G(e) : G(e.value)).join(", ");
}
function hs(e) {
	return Object.prototype.hasOwnProperty.call(Eo, e) && typeof Eo[e] == "function";
}
function gs(e) {
	let t = [];
	for (let n in e) t.push(`${_s(n)}: ${G(e[n].value)}`);
	return t.join("; ");
}
function _s(e) {
	return e === "CiAdrCity" ? "CreatorCity" : e === "CiAdrCtry" ? "CreatorCountry" : e === "CiAdrExtadr" ? "CreatorAddress" : e === "CiAdrPcode" ? "CreatorPostalCode" : e === "CiAdrRegion" ? "CreatorRegion" : e === "CiEmailWork" ? "CreatorWorkEmail" : e === "CiTelWork" ? "CreatorWorkPhone" : e === "CiUrlWork" ? "CreatorWorkUrl" : e;
}
function vs(e, t = {}) {
	for (let n in e) try {
		fs(n) || d(t, ps(n), ys(e[n], n));
	} catch {}
	return t;
}
function ys(e, t) {
	return xs(e) ? Ss(e, t) : bs(e) ? {
		value: "",
		attributes: {},
		description: ""
	} : Cs(e) ? ws(e, t) : As(e) ? js(e, t) : Ms(e) ? Ns(e, t) : Ps(e) ? Is(e, t) : Rs(e, t);
}
function bs(e) {
	return e.attributes["rdf:parseType"] === "Resource" && typeof e.value == "string" && e.value.trim() === "";
}
function xs(e) {
	return Array.isArray(e);
}
function Ss(e, t) {
	return Rs(e[e.length - 1], t);
}
function Cs(e) {
	return e.attributes["rdf:parseType"] === "Resource" && e.value["rdf:value"] !== void 0 || e.value["rdf:Description"] !== void 0 && e.value["rdf:Description"].value["rdf:value"] !== void 0;
}
function ws(e, t) {
	let n = Ts(e);
	e.value["rdf:Description"] !== void 0 && (e = e.value["rdf:Description"]), u(n, Ts(e), Es(e));
	let r = Ds(e);
	return {
		value: r,
		attributes: n,
		description: G(r, t)
	};
}
function Ts(e) {
	let t = {};
	for (let n in e.attributes) n !== "rdf:parseType" && n !== "rdf:resource" && !fs(n) && d(t, ps(n), e.attributes[n]);
	return t;
}
function Es(e) {
	let t = {};
	for (let n in e.value) n !== "rdf:value" && !fs(n) && d(t, ps(n), e.value[n].value);
	return t;
}
function Ds(e) {
	let t = Os(e.value["rdf:value"]);
	return zs(t) || ks(t);
}
function Os(e) {
	return xs(e) ? e[e.length - 1] : e;
}
function ks(e) {
	return Ps(e) ? Is(e).value : typeof e.value == "object" ? vs(e.value, Object.create(null)) : e.value;
}
function As(e) {
	return e.attributes["rdf:parseType"] === "Resource" || e.value["rdf:Description"] !== void 0 && e.value["rdf:Description"].value["rdf:value"] === void 0;
}
function js(e, t) {
	let n = {
		value: {},
		attributes: {}
	};
	return e.value["rdf:Description"] !== void 0 && (u(n.value, us(e.value["rdf:Description"].attributes)), u(n.attributes, Ts(e)), e = e.value["rdf:Description"]), u(n.value, vs(e.value)), n.description = G(n.value, t), n;
}
function Ms(e) {
	return Object.keys(e.value).length === 0 && e.attributes["xml:lang"] === void 0 && e.attributes["rdf:resource"] === void 0;
}
function Ns(e, t) {
	let n = us(e.attributes);
	return {
		value: n,
		attributes: {},
		description: G(n, t)
	};
}
function Ps(e) {
	return Fs(e.value) !== void 0;
}
function Fs(e) {
	return Os(e["rdf:Bag"] || e["rdf:Seq"] || e["rdf:Alt"]);
}
function Is(e, t) {
	let n = Fs(e.value).value["rdf:li"], r = Ts(e), i = [];
	return n === void 0 ? n = [] : Array.isArray(n) || (n = [n]), n.forEach((e) => {
		i.push(Ls(e));
	}), {
		value: i,
		attributes: r,
		description: G(i, t)
	};
}
function Ls(e) {
	return Cs(e) ? ws(e) : As(e) ? js(e).value : Ms(e) ? Ns(e).value : Rs(e);
}
function Rs(e, t) {
	let n = zs(e) || ls(e.value);
	return {
		value: n,
		attributes: Ts(e),
		description: G(n, t)
	};
}
function zs(e) {
	return e.attributes && e.attributes["rdf:resource"];
}
//#endregion
//#region node_modules/exifreader/src/photoshop-tag-names.js
var K = {
	CLOSED_SUBPATH_LENGTH: 0,
	CLOSED_SUBPATH_BEZIER_LINKED: 1,
	CLOSED_SUBPATH_BEZIER_UNLINKED: 2,
	OPEN_SUBPATH_LENGTH: 3,
	OPEN_SUBPATH_BEZIER_LINKED: 4,
	OPEN_SUBPATH_BEZIER_UNLINKED: 5,
	FILL_RULE: 6,
	CLIPBOARD: 7,
	INITIAL_FILL_RULE: 8
}, Bs = 24, q = {
	2e3: {
		name: "PathInformation",
		description: Vs
	},
	2999: {
		name: "ClippingPathName",
		description(e) {
			let [, t] = o(e, 0);
			return t;
		}
	}
};
function Vs(e) {
	let t = {}, n = [];
	for (let r = 0; r < e.byteLength; r += 26) {
		let i = L.getShortAt(e, r);
		Hs[i] && (t[i] || (t[i] = Hs[i].description), n.push({
			type: i,
			path: Hs[i].path(e, r + 2)
		}));
	}
	return JSON.stringify({
		types: t,
		paths: n
	});
}
var Hs = {
	[K.CLOSED_SUBPATH_LENGTH]: {
		description: "Closed subpath length",
		path: (e, t) => [L.getShortAt(e, t)]
	},
	[K.CLOSED_SUBPATH_BEZIER_LINKED]: {
		description: "Closed subpath Bezier knot, linked",
		path: Us
	},
	[K.CLOSED_SUBPATH_BEZIER_UNLINKED]: {
		description: "Closed subpath Bezier knot, unlinked",
		path: Us
	},
	[K.OPEN_SUBPATH_LENGTH]: {
		description: "Open subpath length",
		path: (e, t) => [L.getShortAt(e, t)]
	},
	[K.OPEN_SUBPATH_BEZIER_LINKED]: {
		description: "Open subpath Bezier knot, linked",
		path: Us
	},
	[K.OPEN_SUBPATH_BEZIER_UNLINKED]: {
		description: "Open subpath Bezier knot, unlinked",
		path: Us
	},
	[K.FILL_RULE]: {
		description: "Path fill rule",
		path: () => []
	},
	[K.INITIAL_FILL_RULE]: {
		description: "Initial fill rule",
		path: (e, t) => [L.getShortAt(e, t)]
	},
	[K.CLIPBOARD]: {
		description: "Clipboard",
		path: Gs
	}
};
function Us(e, t) {
	let n = [];
	for (let r = 0; r < Bs; r += 8) n.push(Ws(e, t + r));
	return n;
}
function Ws(e, t) {
	let n = J(e, t, 8);
	return [J(e, t + 4, 8), n];
}
function Gs(e, t) {
	return [[
		J(e, t, 8),
		J(e, t + 4, 8),
		J(e, t + 8, 8),
		J(e, t + 12, 8)
	], J(e, t + 16, 8)];
}
function J(e, t, n) {
	let r = L.getLongAt(e, t), i = r >>> 31 ? -1 : 1, a = (r & 2130706432) >>> 32 - n, o = r & parseInt(_("1", 32 - n), 2);
	return i * g(a.toString(2) + "." + h(o.toString(2), 32 - n, "0"), 2);
}
//#endregion
//#region node_modules/exifreader/src/photoshop-tags.js
var Ks = { read: Qs }, qs = "8BIM", Js = 2, Ys = 4, Xs = 4, Zs = 12;
function Qs(e, t, i = z) {
	let a = n(new Uint8Array(e).buffer), o = {}, s = 0;
	for (; s + Zs <= e.length;) {
		let c = r(a, s, Xs);
		s += Xs;
		let l = L.getShortAt(a, s);
		s += Js;
		let { tagName: u, tagNameSize: d } = ec(a, s);
		if (s += d, s + Ys > e.length) break;
		let f = L.getLongAt(a, s);
		s += Ys;
		let p = Math.min(f, e.length - s);
		if (c === qs) {
			let e = $s(l, u, t);
			if (!i.shouldParseTag("photoshop", e, l)) {
				s += p + p % 2;
				continue;
			}
			let c = n(a.buffer, s, p), d = {
				id: l,
				value: r(c, 0, p)
			};
			if (q[l]) {
				try {
					d.description = q[l].description(c);
				} catch {
					d.description = "<no description formatter>";
				}
				o[u || q[l].name] = d;
			} else t && (o[`undefined-${l}`] = d);
		}
		s += p + p % 2;
	}
	return o;
}
function $s(e, t, n) {
	if (t) return t;
	if (q[e] && q[e].name) return q[e].name;
	if (n) return `undefined-${e}`;
}
function ec(e, t) {
	let [n, r] = o(e, t);
	return {
		tagName: r,
		tagNameSize: 1 + n + +(n % 2 == 0)
	};
}
//#endregion
//#region node_modules/exifreader/src/icc-tag-names.js
var tc = {
	desc: { name: "ICC Description" },
	cprt: { name: "ICC Copyright" },
	dmdd: { name: "ICC Device Model Description" },
	vued: { name: "ICC Viewing Conditions Description" },
	dmnd: { name: "ICC Device Manufacturer for Display" },
	tech: { name: "Technology" }
}, nc = {
	4: {
		name: "Preferred CMM type",
		value: (e, t) => r(e, t, 4),
		description: (e) => e === null ? "" : ic(e)
	},
	8: {
		name: "Profile Version",
		value: (e, t) => e.getUint8(t).toString(10) + "." + (e.getUint8(t + 1) >> 4).toString(10) + "." + (e.getUint8(t + 1) % 16).toString(10)
	},
	12: {
		name: "Profile/Device class",
		value: (e, t) => r(e, t, 4),
		description: (e) => {
			switch (e.toLowerCase()) {
				case "scnr": return "Input Device profile";
				case "mntr": return "Display Device profile";
				case "prtr": return "Output Device profile";
				case "link": return "DeviceLink profile";
				case "abst": return "Abstract profile";
				case "spac": return "ColorSpace profile";
				case "nmcl": return "NamedColor profile";
				case "cenc": return "ColorEncodingSpace profile";
				case "mid ": return "MultiplexIdentification profile";
				case "mlnk": return "MultiplexLink profile";
				case "mvis": return "MultiplexVisualization profile";
				default: return e;
			}
		}
	},
	16: {
		name: "Color Space",
		value: (e, t) => r(e, t, 4)
	},
	20: {
		name: "Connection Space",
		value: (e, t) => r(e, t, 4)
	},
	24: {
		name: "ICC Profile Date",
		value: (e, t) => rc(e, t).toISOString()
	},
	36: {
		name: "ICC Signature",
		value: (e, t) => Y(e.buffer.slice(t, t + 4))
	},
	40: {
		name: "Primary Platform",
		value: (e, t) => r(e, t, 4),
		description: (e) => ic(e)
	},
	48: {
		name: "Device Manufacturer",
		value: (e, t) => r(e, t, 4),
		description: (e) => ic(e)
	},
	52: {
		name: "Device Model Number",
		value: (e, t) => r(e, t, 4)
	},
	64: {
		name: "Rendering Intent",
		value: (e, t) => e.getUint32(t),
		description: (e) => {
			switch (e) {
				case 0: return "Perceptual";
				case 1: return "Relative Colorimetric";
				case 2: return "Saturation";
				case 3: return "Absolute Colorimetric";
				default: return e;
			}
		}
	},
	80: {
		name: "Profile Creator",
		value: (e, t) => r(e, t, 4)
	}
};
function rc(e, t) {
	let n = e.getUint16(t), r = e.getUint16(t + 2) - 1, i = e.getUint16(t + 4), a = e.getUint16(t + 6), o = e.getUint16(t + 8), s = e.getUint16(t + 10);
	return new Date(Date.UTC(n, r, i, a, o, s));
}
function Y(e) {
	return String.fromCharCode.apply(null, new Uint8Array(e));
}
function ic(e) {
	switch (e.toLowerCase()) {
		case "appl": return "Apple";
		case "adbe": return "Adobe";
		case "msft": return "Microsoft";
		case "sunw": return "Sun Microsystems";
		case "sgi": return "Silicon Graphics";
		case "tgnt": return "Taligent";
		default: return e;
	}
}
//#endregion
//#region node_modules/exifreader/src/icc-tags.js
var ac = { read: hc }, oc = 84, sc = "acsp", cc = "desc", lc = "mluc", uc = "text", dc = "sig ", fc = 12, pc = 12, mc = 16;
function hc(e, t, n, r) {
	return n && t[0].compressionMethod !== void 0 ? gc(e, t, r) : vc(e, t);
}
function gc(e, t, n) {
	return _c(t[0].compressionMethod) ? te(new DataView(e.buffer.slice(t[0].offset, t[0].offset + t[0].length)), t[0].compressionMethod, "utf-8", "dataview", n).then(Sc).catch(() => ({})) : {};
}
function _c(e) {
	return e === 0;
}
function vc(e, t) {
	try {
		let n = yc(e), r = t.reduce((e, t) => e + t.length, 0), i = Math.min(r, n.byteLength), a = new Uint8Array(i), o = 0;
		for (let e = 1; e <= t.length; e++) {
			let r = t.find((t) => t.chunkNumber === e);
			if (!r) throw Error(`ICC chunk ${e} not found`);
			let i = n.slice(r.offset, r.offset + r.length), s = new Uint8Array(i);
			a.set(s, o), o += s.length;
		}
		return Sc(new DataView(a.buffer));
	} catch {
		return {};
	}
}
function yc(e) {
	return Array.isArray(e) ? new DataView(Uint8Array.from(e).buffer).buffer : e.buffer;
}
function bc(e) {
	return e.byteLength < 132;
}
function xc(e, t) {
	return e.byteLength < t + fc;
}
function Sc(e) {
	let t = e.buffer, n = e.getUint32();
	if (e.byteLength !== n) throw Error("ICC profile length not matching");
	if (e.byteLength < oc) throw Error("ICC profile too short");
	let i = {}, o = Object.keys(nc);
	for (let t = 0; t < o.length; t++) {
		let n = o[t], r = nc[n], a = r.value(e, parseInt(n, 10)), s = a;
		r.description && (s = r.description(a)), i[r.name] = {
			value: a,
			description: s
		};
	}
	if (Y(t.slice(36, 40)) !== sc) throw Error("ICC profile: missing signature");
	if (bc(e)) return i;
	let s = e.getUint32(128), c = 132, l = e.byteLength;
	for (let n = 0; n < s; n++) {
		if (xc(e, c)) return i;
		let n = r(e, c, 4), o = e.getUint32(c + 4), s = e.getUint32(c + 8);
		if (o > e.byteLength) return i;
		let u = r(e, o, 4);
		if (u === cc) {
			let r = e.getUint32(o + 8);
			if (r > s) return i;
			X(i, n, Y(t.slice(o + 12, o + r + 11)));
		} else if (u === lc) {
			let t = e.getUint32(o + 8), c = e.getUint32(o + 12);
			if (c < pc || t > 1e3 || t * c > e.byteLength - o - mc) return i;
			let u = Math.min(s, e.byteLength - o), d = o + mc, f = [];
			for (let n = 0; n < t; n++) {
				let t = r(e, d + 0, 2), n = r(e, d + 2, 2), i = e.getUint32(d + 4), s = e.getUint32(d + 8), p = Math.max(0, u - s), m = Math.min(i, p, l);
				l -= m;
				let h = a(e, o + s, m);
				f.push({
					languageCode: t,
					countryCode: n,
					text: h
				}), d += c;
			}
			if (t === 1) X(i, n, f[0].text);
			else {
				let e = {};
				for (let t = 0; t < f.length; t++) e[`${f[t].languageCode}-${f[t].countryCode}`] = f[t].text;
				X(i, n, e);
			}
		} else u === uc ? X(i, n, Y(t.slice(o + 8, o + s - 7))) : u === dc && X(i, n, Y(t.slice(o + 8, o + 12)));
		c += 12;
	}
	return i;
}
function X(e, t, n) {
	tc[t] ? e[tc[t].name] = {
		value: n,
		description: n
	} : e[t] = {
		value: n,
		description: n
	};
}
//#endregion
//#region node_modules/exifreader/src/canon-tags.js
var Cc = 27, wc = 22, Tc = {
	read: Ec,
	SHOT_INFO_AUTO_ROTATE: Cc,
	CAMERA_SETTINGS_LENS_TYPE: wc
};
function Ec(e, t, n, r, i, a = !1, o = void 0) {
	let s = V(e, ti, t, t + n, r, i, a, o, "makerNotes");
	return s.ShotInfo && (s = u({}, s, Dc(s.ShotInfo.value)), delete s.ShotInfo), s.CameraSettings && (s = u({}, s, Oc(s.CameraSettings.value)), delete s.CameraSettings), s;
}
function Dc(e) {
	let t = {};
	return e[Cc] !== void 0 && (t.AutoRotate = {
		value: e[Cc],
		description: kc(e[Cc])
	}), t;
}
function Oc(e) {
	let t = {};
	if (!Array.isArray(e)) return t;
	let n = e[wc];
	return Number.isFinite(n) && (t.LensType = {
		value: n,
		description: `${n}`
	}), t;
}
function kc(e) {
	return e === 0 ? "None" : e === 1 ? "Rotate 90 CW" : e === 2 ? "Rotate 180" : e === 3 ? "Rotate 270 CW" : "Unknown";
}
//#endregion
//#region node_modules/exifreader/src/pentax-tags.js
var Ac = 8, jc = 10, Mc = { K3_III: 78420 }, Nc = {
	CAMERA_ORIENTATION: 1,
	ROLL_ANGLE: 3,
	PITCH_ANGLE: 5
}, Pc = {
	read: Fc,
	PENTAX_IFD_OFFSET: jc,
	MODEL_ID: Mc,
	LIK3III: Nc
};
function Fc(e, t, n, r, i = !1, a = void 0) {
	try {
		let o = A.getByteOrder(e, t + n + Ac), s = t + n, c = V(e, ni, s, s + jc, o, r, i, a, "makerNotes");
		return Ic(c) && (c = u({}, c, Lc(e, s + c.LevelInfo.__offset, o)), delete c.LevelInfo), c;
	} catch {
		return {};
	}
}
function Ic(e) {
	return e.PentaxModelID && e.PentaxModelID.value === Mc.K3_III && e.LevelInfo;
}
function Lc(e, t, n) {
	let r = {};
	if (t + 7 > e.byteLength) return r;
	let i = e.getInt8(t + Nc.CAMERA_ORIENTATION);
	r.CameraOrientation = {
		value: i,
		description: Rc(i)
	};
	let a = e.getInt16(t + Nc.ROLL_ANGLE, n === A.LITTLE_ENDIAN);
	r.RollAngle = {
		value: a,
		description: zc(a)
	};
	let o = e.getInt16(t + Nc.PITCH_ANGLE, n === A.LITTLE_ENDIAN);
	return r.PitchAngle = {
		value: o,
		description: Bc(o)
	}, r;
}
function Rc(e) {
	return e === 0 ? "Horizontal (normal)" : e === 1 ? "Rotate 270 CW" : e === 2 ? "Rotate 180" : e === 3 ? "Rotate 90 CW" : e === 4 ? "Upwards" : e === 5 ? "Downwards" : "Unknown";
}
function zc(e) {
	return "" + e * -.5;
}
function Bc(e) {
	return "" + e * -.5;
}
//#endregion
//#region node_modules/exifreader/src/png-file-tags.js
var Vc = { read: Gc }, Hc = {
	0: "Grayscale",
	2: "RGB",
	3: "Palette",
	4: "Grayscale with Alpha",
	6: "RGB with Alpha"
}, Uc = {
	0: "Noninterlaced",
	1: "Adam7 Interlace"
}, Wc = [
	{
		name: "Image Width",
		offset: 0,
		size: 4,
		read: L.getLongAt,
		description: (e) => `${e}px`
	},
	{
		name: "Image Height",
		offset: 4,
		size: 4,
		read: L.getLongAt,
		description: (e) => `${e}px`
	},
	{
		name: "Bit Depth",
		offset: 8,
		size: 1,
		read: L.getByteAt,
		description: (e) => `${e}`
	},
	{
		name: "Color Type",
		offset: 9,
		size: 1,
		read: L.getByteAt,
		description: (e) => Hc[e] || "Unknown"
	},
	{
		name: "Compression",
		offset: 10,
		size: 1,
		read: L.getByteAt,
		description: (e) => e === 0 ? "Deflate/Inflate" : "Unknown"
	},
	{
		name: "Filter",
		offset: 11,
		size: 1,
		read: L.getByteAt,
		description: (e) => e === 0 ? "Adaptive" : "Unknown"
	},
	{
		name: "Interlace",
		offset: 12,
		size: 1,
		read: L.getByteAt,
		description: (e) => Uc[e] || "Unknown"
	}
];
function Gc(e, t) {
	let n = {};
	for (let r = 0; r < Wc.length; r++) n[Wc[r].name] = Kc(e, t, Wc[r]);
	return n;
}
function Kc(e, t, n) {
	if (t + n.offset + n.size > e.byteLength) return;
	let r = n.read(e, t + n.offset);
	return {
		value: r,
		description: n.description(r)
	};
}
//#endregion
//#region node_modules/exifreader/src/png-text-tags.js
var qc = { read: nl }, Jc = "STATE_KEYWORD", Yc = "STATE_COMPRESSION", Xc = "STATE_LANG", Zc = "STATE_TRANSLATED_KEYWORD", Qc = "STATE_TEXT", $c = 1, el = 1, tl = 6;
function nl(e, t, n, r, i = !1, a = z, o) {
	let s = {}, c = [];
	for (let l = 0; l < t.length; l++) {
		let { offset: u, length: d, type: f } = t[l], p = rl(e, u, d, f, n, o);
		if (p instanceof Promise) c.push(p.then(({ name: e, value: t, description: n }) => {
			try {
				if (k.USE_EXIF && dl(e, t)) return a.shouldParseGroup("exif") ? { __exif: Qi.read(pl(t), tl, r, i, a).tags } : {};
				if (k.USE_IPTC && fl(e, t)) return a.shouldParseGroup("iptc") ? { __iptc: oo.read(pl(t), 0, r, a) } : {};
				if (e && !dl(e, t) && !fl(e, t)) return a.shouldParseGroup("png") ? { [e]: {
					value: t,
					description: n
				} } : {};
			} catch {}
			return {};
		}));
		else {
			let { name: e, value: t, description: n } = p;
			e && a.shouldParseGroup("png") && (s[e] = {
				value: t,
				description: n
			});
		}
	}
	return {
		readTags: s,
		readTagsPromise: c.length > 0 ? Promise.all(c) : void 0
	};
}
function rl(e, t, n, r, i, a) {
	let o = [], s = [], c = [], l, u = Jc, d;
	for (let i = 0; i < n && t + i < e.byteLength; i++) {
		if (u === Yc) {
			d = il({
				type: r,
				dataView: e,
				offset: t + i
			}), r === "iTXt" && (i += $c), u = al(r, u);
			continue;
		}
		if (u === Qc) {
			l = new DataView(e.buffer.slice(t + i, t + n));
			break;
		}
		let a = e.getUint8(t + i);
		a === 0 ? u = al(r, u) : u === Jc ? o.push(a) : u === Xc ? s.push(a) : u === Zc && c.push(a);
	}
	if (d !== void 0 && !i) return {};
	let f = te(l, d, ol(r), "string", a);
	return f instanceof Promise ? f.then((e) => sl(e, r, s, o)).catch(() => sl("<text using unknown compression>".split(""), r, s, o)) : sl(f, r, s, o);
}
function il({ type: e, dataView: t, offset: n }) {
	if (e === "iTXt") {
		if (t.getUint8(n) === el) return t.getUint8(n + 1);
	} else if (e === "zTXt") return t.getUint8(n);
}
function al(e, t) {
	return t === Jc && ["iTXt", "zTXt"].includes(e) ? Yc : t === Yc ? e === "iTXt" ? Xc : Qc : t === Xc ? Zc : Qc;
}
function ol(e) {
	return e === "tEXt" || e === "zTXt" ? "latin1" : "utf-8";
}
function sl(e, t, n, r) {
	let i = ll(e);
	return {
		name: cl(t, n, r),
		value: i,
		description: t === "iTXt" ? ul(e) : i
	};
}
function cl(e, t, n) {
	let r = s(n);
	return e === "tEXt" || t.length === 0 ? r : `${r} (${s(t)})`;
}
function ll(e) {
	return e instanceof DataView ? r(e, 0, e.byteLength) : e;
}
function ul(e) {
	return Ja.decode("UTF-8", e);
}
function dl(e, t) {
	return e.toLowerCase() === "raw profile type exif" && t.substring(1, 5) === "exif";
}
function fl(e, t) {
	return e.toLowerCase() === "raw profile type iptc" && t.substring(1, 5) === "iptc";
}
function pl(e) {
	return ml(e.match(/\n(exif|iptc)\n\s*\d+\n([\s\S]*)$/)[2].replace(/\n/g, ""));
}
function ml(e) {
	let t = /* @__PURE__ */ new DataView(/* @__PURE__ */ new ArrayBuffer(e.length / 2));
	for (let n = 0; n < e.length; n += 2) t.setUint8(n / 2, parseInt(e.substring(n, n + 2), 16));
	return t;
}
//#endregion
//#region node_modules/exifreader/src/png-tags.js
var hl = { read: gl };
function gl(e, t) {
	let n = {};
	for (let i = 0; i < t.length; i++) {
		let a = L.getLongAt(e, t[i] + 0), o = r(e, t[i] + 4, 4);
		o === "pHYs" ? (n["Pixels Per Unit X"] = _l(e, t[i], a), n["Pixels Per Unit Y"] = vl(e, t[i], a), n["Pixel Units"] = yl(e, t[i], a)) : o === "tIME" && (n["Modify Date"] = bl(e, t[i], a));
	}
	return n;
}
function _l(e, t, n) {
	if (!xl(e, t, n, 0, 4)) return;
	let r = L.getLongAt(e, t + 8 + 0);
	return {
		value: r,
		description: "" + r
	};
}
function vl(e, t, n) {
	if (!xl(e, t, n, 4, 4)) return;
	let r = L.getLongAt(e, t + 8 + 4);
	return {
		value: r,
		description: "" + r
	};
}
function yl(e, t, n) {
	if (!xl(e, t, n, 8, 1)) return;
	let r = L.getByteAt(e, t + 8 + 8);
	return {
		value: r,
		description: r === 1 ? "meters" : "Unknown"
	};
}
function bl(e, t, n) {
	if (!xl(e, t, n, 0, 7)) return;
	let r = L.getShortAt(e, t + 8), i = L.getByteAt(e, t + 8 + 2), a = L.getByteAt(e, t + 8 + 3), o = L.getByteAt(e, t + 8 + 4), s = L.getByteAt(e, t + 8 + 5), c = L.getByteAt(e, t + 8 + 6);
	return {
		value: [
			r,
			i,
			a,
			o,
			s,
			c
		],
		description: `${Z(r, 4)}-${Z(i, 2)}-${Z(a, 2)} ${Z(o, 2)}:${Z(s, 2)}:${Z(c, 2)}`
	};
}
function xl(e, t, n, r, i) {
	return r + i <= n && t + 8 + r + i <= e.byteLength;
}
function Z(e, t) {
	return h("" + e, t, "0");
}
//#endregion
//#region node_modules/exifreader/src/vp8x-tags.js
var Sl = { read: El }, Cl = 4, wl = 7, Tl = 10;
function El(e, t) {
	let n = {};
	if (t + Tl > e.byteLength) return n;
	let r = L.getByteAt(e, t);
	return n.Alpha = Dl(r), n.Animation = Ol(r), n.ImageWidth = kl(e, t + Cl), n.ImageHeight = kl(e, t + wl), n;
}
function Dl(e) {
	let t = e & 16;
	return {
		value: +!!t,
		description: t ? "Yes" : "No"
	};
}
function Ol(e) {
	let t = e & 2;
	return {
		value: +!!t,
		description: t ? "Yes" : "No"
	};
}
function kl(e, t) {
	let n = L.getByteAt(e, t) + 256 * L.getByteAt(e, t + 1) + 65536 * L.getByteAt(e, t + 2) + 1;
	return {
		value: n,
		description: n + "px"
	};
}
//#endregion
//#region node_modules/exifreader/src/gif-file-tags.js
var Al = { read: Nl }, jl = (e) => `${e} ${e === 1 ? "bit" : "bits"}`, Ml = [
	{
		name: "GIF Version",
		offset: 3,
		size: 3,
		getValue: (e, t) => r(e, t, 3),
		description: (e) => e
	},
	{
		name: "Image Width",
		offset: 6,
		size: 2,
		getValue: (e, t) => e.getUint16(t, !0),
		description: (e) => `${e}px`
	},
	{
		name: "Image Height",
		offset: 8,
		size: 2,
		getValue: (e, t) => e.getUint16(t, !0),
		description: (e) => `${e}px`
	},
	{
		name: "Global Color Map",
		offset: 10,
		size: 1,
		getValue: (e, t) => (e.getUint8(t) & 128) >>> 7,
		description: (e) => e === 1 ? "Yes" : "No"
	},
	{
		name: "Bits Per Pixel",
		offset: 10,
		size: 1,
		getValue: (e, t) => (e.getUint8(t) & 7) + 1,
		description: jl
	},
	{
		name: "Color Resolution Depth",
		offset: 10,
		size: 1,
		getValue: (e, t) => ((e.getUint8(t) & 112) >>> 4) + 1,
		description: jl
	}
];
function Nl(e) {
	let t = {};
	for (let n = 0; n < Ml.length; n++) t[Ml[n].name] = Pl(e, Ml[n]);
	return t;
}
function Pl(e, t) {
	if (t.offset + t.size > e.byteLength) return;
	let n = t.getValue(e, t.offset);
	return {
		value: n,
		description: t.description(n)
	};
}
//#endregion
//#region node_modules/exifreader/src/thumbnail.js
var Fl = [
	6,
	7,
	99
], Il = { get: Ll };
function Ll(e, t, n) {
	if (Rl(t)) {
		t.type = "image/jpeg";
		let r = n + t.JPEGInterchangeFormat.value;
		t.image = e.buffer.slice(r, r + t.JPEGInterchangeFormatLength.value), f(t, "base64", function() {
			return p(this.image);
		});
	}
	return t;
}
function Rl(e) {
	return e && (e.Compression === void 0 || Fl.includes(e.Compression.value)) && e.JPEGInterchangeFormat && e.JPEGInterchangeFormat.value && e.JPEGInterchangeFormatLength && e.JPEGInterchangeFormatLength.value;
}
//#endregion
//#region node_modules/exifreader/src/composite.js
var zl = {
	INCHES: 2,
	CENTIMETERS: 3,
	MILLIMETERS: 4
}, Bl = {
	INCHES_TO_MM: 25.4,
	CM_TO_MM: 10,
	MM_TO_MM: 1
}, Vl = { get: Hl };
function Hl(e, t) {
	let n = {}, r = !1, i = Q(e, "exif", "FocalLength", t), a = Q(e, "exif", "FocalPlaneXResolution", t), o = Q(e, "exif", "FocalPlaneYResolution", t), s = Q(e, "exif", "FocalPlaneResolutionUnit", t), c = Q(e, "file", "Image Width", t), l = Q(e, "file", "Image Height", t), u = Q(e, "exif", "FocalLengthIn35mmFilm", t) || Ul(a, o, s, c, l, i);
	u && (n.FocalLength35efl = {
		value: u,
		description: P.FocalLengthIn35mmFilm(u)
	}, r = !0);
	let d = Wl(i, u);
	d && (n.ScaleFactorTo35mmEquivalent = d, r = !0);
	let f = Gl(u);
	if (f && (n.FieldOfView = f, r = !0), r) return n;
}
function Q(e, t, n, r) {
	if (r && e[t] && e[t][n]) return e[t][n].value;
	if (!r && e[n]) return e[n].value;
}
function Ul(e, t, n, r, i, a) {
	if (e && t && n && r && i && a) try {
		let o;
		switch (n) {
			case zl.INCHES:
				o = Bl.INCHES_TO_MM;
				break;
			case zl.CENTIMETERS:
				o = Bl.CM_TO_MM;
				break;
			case zl.MILLIMETERS:
				o = Bl.MM_TO_MM;
				break;
			default: return;
		}
		let s = e[0] / e[1] * o, c = t[0] / t[1] * o, l = r / s, u = i / c, d = Math.sqrt(l ** 2 + u ** 2);
		return a[0] / a[1] * (43.27 / d);
	} catch {}
}
function Wl(e, t) {
	if (e && t) try {
		let n = t / (e[0] / e[1]);
		return {
			value: n,
			description: n.toFixed(1)
		};
	} catch {}
}
function Gl(e) {
	if (e) try {
		let t = 2 * Math.atan(36 / (2 * e)) * (180 / Math.PI);
		return {
			value: t,
			description: t.toFixed(1) + " deg"
		};
	} catch {}
}
//#endregion
//#region node_modules/exifreader/src/loadview-pipeline.js
function Kl({ mergeSteps: e, deferredResults: t, parsedGroups: n, expanded: r, tagFilter: i, dataView: a, tiffHeaderOffset: o, exifDataView: s, fileType: c, pngTextChunks: l, pngTextIsAsync: u, thumbnailIfdTags: d, deps: f }) {
	let p = {};
	for (let l = 0; l < e.length; l++) p = ql({
		step: e[l],
		deferredResults: t,
		parsedGroups: n,
		expanded: r,
		tagFilter: i,
		dataView: a,
		tiffHeaderOffset: o,
		exifDataView: s,
		fileType: c,
		thumbnailIfdTags: d,
		tags: p,
		deps: f
	});
	return k.USE_PNG && r && u && i.shouldReturnGroup("png") && p.png && (p.pngText = f.objectAssign({}, p.png)), k.USE_PNG && r && i.shouldReturnGroup("png") && f.hasPngTextData(l) && p.png && !p.pngText && (p.pngText = f.objectAssign({}, p.png)), p;
}
function ql({ step: e, deferredResults: t, parsedGroups: n, expanded: r, tagFilter: i, dataView: a, tiffHeaderOffset: o, exifDataView: s, fileType: c, thumbnailIfdTags: l, tags: u, deps: d }) {
	if (e.type === "mergeGroupAssign") {
		let t = d.filterTagsForReturn(e.groupKey, e.parsedTags, i);
		return Jl(u, e.groupKey, t, r, d);
	}
	if (e.type === "mergeGroupMerge") {
		let t = d.filterTagsForReturn(e.groupKey, e.parsedTags, i);
		return Xl(u, e.groupKey, t, r, d);
	}
	if (k.USE_XMP && e.type === "mergeXmpGroupAssign") {
		let t = d.filterTagsForReturn("xmp", e.parsedTags, i);
		return Yl(u, t, r, d);
	}
	if (k.USE_ICC && e.type === "mergeIccDeferred") {
		let a = t[e.deferredKey], o = d.filterTagsForParse("icc", a, i);
		if (n.icc = o, !i.shouldReturnGroup("icc")) return u;
		let s = d.filterTagsForReturn("icc", o, i);
		return Jl(u, "icc", s, r, d);
	}
	if (k.USE_JXL && e.type === "mergeBrobExifDeferred") {
		let a = t[e.deferredKey];
		if (!a || Object.keys(a).length === 0) return u;
		let o = d.filterTagsForParse("exif", a, i);
		if (n.exif = n.exif ? d.objectAssign({}, n.exif, o) : o, !i.shouldReturnGroup("exif")) return u;
		let s = d.filterTagsForReturn("exif", o, i);
		return Jl(u, "exif", s, r, d);
	}
	if (k.USE_JXL && e.type === "mergeBrobXmpDeferred") {
		let a = t[e.deferredKey];
		if (!a || Object.keys(a).length === 0) return u;
		let o = d.filterTagsForParse("xmp", a, i);
		if (n.xmp = o, !i.shouldReturnGroup("xmp")) return u;
		let s = d.filterTagsForReturn("xmp", o, i);
		return Yl(u, s, r, d);
	}
	if (k.USE_PNG && e.type === "mergePngFile") {
		let t = d.filterTagsForReturn("png", e.parsedTags, i);
		return i.shouldReturnGroup("png") ? r ? (u.png = u.png ? d.objectAssign({}, u.png, t) : t, u.pngFile = t, u) : d.objectAssign({}, u, t) : u;
	}
	if (k.USE_PNG && e.type === "mergePngChunk") {
		let t = d.filterTagsForReturn("png", e.parsedTags, i);
		return i.shouldReturnGroup("png") ? r ? (u.png = u.png ? d.objectAssign({}, u.png, t) : t, u) : d.objectAssign({}, u, t) : u;
	}
	if (k.USE_PNG && e.type === "processPngTextReadTags") return Zl({
		readTags: e.readTags,
		parsedGroups: n,
		expanded: r,
		tagFilter: i,
		tags: u,
		deps: d
	});
	if (k.USE_PNG && e.type === "processPngTextReadTagsDeferredList") {
		let a = t[e.deferredKey] || [];
		for (let e = 0; e < a.length; e++) u = Zl({
			readTags: a[e],
			parsedGroups: n,
			expanded: r,
			tagFilter: i,
			tags: u,
			deps: d
		});
		return u;
	}
	if (e.type === "gps") {
		if (r && i.shouldReturnGroup("gps") && n.exif) {
			let e = d.getGpsGroupFromExifTags(n.exif);
			if (e) {
				let t = d.filterTagsForReturn("gps", e, i);
				u.gps = t;
			}
		}
		return u;
	}
	if (e.type === "composite") {
		if (!i.shouldReturnGroup("composite")) return u;
		let e = u, t = r;
		i.isActive && (e = {
			exif: n.exif,
			file: n.file
		}, t = !0);
		let a = d.Composite.get(e, t);
		if (!a) return u;
		let o = d.filterTagsForReturn("composite", a, i);
		return Jl(u, "composite", o, r, d);
	}
	if (e.type === "thumbnail") {
		if (!i.shouldReturnGroup("thumbnail") || !i.shouldReturnTag("thumbnail", "Thumbnail")) return delete u.Thumbnail, u;
		if (!l) return u;
		let e = l ? d.filterTagsForParse("thumbnail", l, i) : void 0;
		e && (n.thumbnail = e);
		let t = (k.USE_JPEG || k.USE_WEBP) && k.USE_EXIF && k.USE_THUMBNAIL && d.Thumbnail.get(s || a, e, o);
		return t ? u.Thumbnail = t : delete u.Thumbnail, u;
	}
	if (e.type === "metadataRange") {
		if (!r) return u;
		let t = $l(e.metadataBlocks, e.metadataTruncated, a, n);
		return t && (u.metadataRange = t), u;
	}
	return e.type === "fileType" && c && i.shouldReturnGroup("file") && i.shouldReturnTag("file", "FileType") && (r ? (u.file || (u.file = {}), u.file.FileType = c) : u.FileType = c), u;
}
function Jl(e, t, n, r, i) {
	return r ? (e[t] = n, e) : i.objectAssign({}, e, n);
}
function Yl(e, t, n, r) {
	if (n) return e.xmp = t, e;
	let i = r.objectAssign({}, t);
	return delete i._raw, r.objectAssign({}, e, i);
}
function Xl(e, t, n, r, i) {
	return r ? (e[t] = e[t] ? i.objectAssign({}, e[t], n) : n, e) : i.objectAssign({}, e, n);
}
function Zl({ readTags: e, parsedGroups: t, expanded: n, tagFilter: r, tags: i, deps: a }) {
	let o = e.__exif, s = e.__iptc;
	if (delete e.__exif, delete e.__iptc, o) {
		let e = a.filterTagsForParse("exif", o, r);
		if (t.exif = t.exif ? a.objectAssign({}, t.exif, e) : e, r.shouldReturnGroup("exif")) {
			let t = a.filterTagsForReturn("exif", e, r);
			n ? i.exif = i.exif ? a.objectAssign({}, i.exif, t) : t : i = a.objectAssign({}, i, t);
		}
	}
	if (s) {
		let e = a.filterTagsForParse("iptc", s, r);
		if (t.iptc = t.iptc ? a.objectAssign({}, t.iptc, e) : e, r.shouldReturnGroup("iptc")) {
			let t = a.filterTagsForReturn("iptc", e, r);
			n ? i.iptc = i.iptc ? a.objectAssign({}, i.iptc, t) : t : i = a.objectAssign({}, i, t);
		}
	}
	if (r.shouldReturnGroup("png")) {
		let o = a.filterTagsForParse("png", e, r), s = a.filterTagsForReturn("png", o, r);
		t.pngText = o, n ? (i.png = i.png ? a.objectAssign({}, i.png, s) : s, s && Object.keys(s).length > 0 && (i.pngText = i.pngText ? a.objectAssign({}, i.pngText, s) : s)) : i = a.objectAssign({}, i, s);
	}
	return i;
}
function Ql(e) {
	return !!e && typeof e.then == "function";
}
function $l(e, t, n, r) {
	let i = (e || []).slice(), a = !!t;
	if (r && r.mpf && Array.isArray(r.mpf.Images)) for (let e = 0; e < r.mpf.Images.length; e++) {
		let t = r.mpf.Images[e];
		if (!t || !t.ImageOffset || !t.ImageSize) continue;
		let n = t.ImageOffset.value, a = t.ImageSize.value;
		typeof n != "number" || n <= 0 || typeof a != "number" || a <= 0 || i.push({
			type: "mpfImage",
			start: n,
			end: n + a
		});
	}
	if (i.length === 0) return;
	i.sort((e, t) => e.start - t.start);
	let o = i[0].start, s = i[0].end;
	for (let e = 1; e < i.length; e++) i[e].end > s && (s = i[e].end);
	let c = n && typeof n.byteLength == "number" ? n.byteLength : 0;
	return {
		start: o,
		end: s,
		complete: !a && s <= c,
		blocks: i
	};
}
//#endregion
//#region node_modules/exifreader/src/errors.js
function eu(e) {
	this.name = "MetadataMissingError", this.message = e || "No Exif data", this.stack = (/* @__PURE__ */ Error()).stack;
}
eu.prototype = /* @__PURE__ */ Error();
var tu = { MetadataMissingError: eu }, nu = {
	load: ru,
	loadView: ou,
	errors: tu
};
function ru(e, t = {}) {
	if (t.length === "auto") return Ce(t), we(iu)(e, t);
	if (S(e) || C(e)) {
		l();
		let n = S(e) ? ae : oe, r = u({}, t, { async: !0 });
		return n(e, r).then((e) => iu(e, r));
	}
	return iu(e, t);
}
function iu(e, t) {
	return au(e) && (e = new Uint8Array(e).buffer), ou(n(e), t);
}
function au(e) {
	try {
		return Buffer.isBuffer(e);
	} catch {
		return !1;
	}
}
function ou(e, { expanded: t = !1, async: n = !1, computed: r = !1, includeUnknown: i = !1, includeOffsets: a = !1, domParser: o = void 0, includeTags: c = void 0, excludeTags: d = void 0, decompress: f = void 0 } = {}) {
	let p = Si({
		includeTags: c,
		excludeTags: d
	}), m = Object.create(null), h = [], g = Object.create(null), _ = [], ee = !1, ne, y, { fileType: re, fileDataOffset: b, jfifDataOffset: ie, tiffHeaderOffset: x, iptcDataOffset: S, xmpChunks: C, iccChunks: w, mpfDataOffset: ae, pngHeaderOffset: oe, pngTextChunks: T, pngChunkOffsets: E, vp8xChunkOffset: D, gifHeaderOffset: se, brobExifChunk: ce, brobXmpChunk: le, jxlCodestreamOffset: ue, metadataBlocks: de, metadataTruncated: fe, exifDataView: O, xmpDataView: pe } = Ur.parseAppMarkers(e, n, t && a), me = uu({
		fileType: re,
		fileDataOffset: b,
		jfifDataOffset: ie,
		tiffHeaderOffset: x,
		iptcDataOffset: S,
		xmpChunks: C,
		iccChunks: w,
		mpfDataOffset: ae,
		pngHeaderOffset: oe,
		pngTextChunks: T,
		pngChunkOffsets: E,
		vp8xChunkOffset: D,
		gifHeaderOffset: se,
		jxlCodestreamOffset: ue
	});
	if (k.USE_JPEG && k.USE_FILE && b !== void 0 && p.shouldParseGroup("file") && ge(c)) {
		let t = $("file", fa.read(e, b), p);
		m.file = t, p.shouldReturnGroup("file") && h.push({
			type: "mergeGroupAssign",
			groupKey: "file",
			parsedTags: t
		});
	}
	if (k.USE_JPEG && k.USE_JFIF && ie !== void 0 && p.shouldParseGroup("jfif")) {
		let t = $("jfif", Aa.read(e, ie), p);
		m.jfif = t, p.shouldReturnGroup("jfif") && h.push({
			type: "mergeGroupAssign",
			groupKey: "jfif",
			parsedTags: t
		});
	}
	if (k.USE_EXIF && x !== void 0 && p.shouldParseGroup("exif")) {
		let { tags: n, byteOrder: a } = su(O || e, x, i, r, p);
		n.Thumbnail && (ne = n.Thumbnail, delete n.Thumbnail);
		let c = $("exif", n, p);
		if (m.exif = c, k.USE_TIFF && k.USE_IPTC && c["IPTC-NAA"] && S === void 0 && p.shouldParseGroup("iptc")) {
			let e = $("iptc", oo.read(c["IPTC-NAA"].value, 0, i, p), p);
			m.iptc = e, p.shouldReturnGroup("iptc") && h.push({
				type: "mergeGroupAssign",
				groupKey: "iptc",
				parsedTags: e
			});
		}
		if (k.USE_TIFF && k.USE_XMP && c.ApplicationNotes && !fu(C) && p.shouldParseGroup("xmp")) {
			let e = $("xmp", Wo.read(s(c.ApplicationNotes.value), void 0, o), p);
			if (m.xmp = e, p.shouldReturnGroup("xmp")) {
				let n = {
					type: "mergeXmpGroupAssign",
					parsedTags: e
				};
				t ? h.push(n) : y = n;
			}
		}
		if (k.USE_PHOTOSHOP && c.ImageSourceData && c.PhotoshopSettings && p.shouldParseGroup("photoshop")) {
			let e = $("photoshop", Ks.read(c.PhotoshopSettings.value, i, p), p);
			m.photoshop = e, p.shouldReturnGroup("photoshop") && h.push({
				type: "mergeGroupAssign",
				groupKey: "photoshop",
				parsedTags: e
			});
		}
		if (k.USE_TIFF && k.USE_ICC && c.ICC_Profile && !pu(w) && p.shouldParseGroup("icc")) {
			let e = $("icc", ac.read(c.ICC_Profile.value, [{
				offset: 0,
				length: c.ICC_Profile.value.length,
				chunkNumber: 1,
				chunksTotal: 1
			}]), p);
			m.icc = e, p.shouldReturnGroup("icc") && h.push({
				type: "mergeGroupAssign",
				groupKey: "icc",
				parsedTags: e
			});
		}
		if (k.USE_MAKER_NOTES && c.MakerNote && p.shouldParseGroup("makerNotes")) {
			if (mu(c)) {
				let t = Tc.read(O || e, x, c.MakerNote.__offset, a, i, r, p);
				m.makerNotes = t, p.shouldReturnGroup("makerNotes") && h.push({
					type: "mergeGroupAssign",
					groupKey: "makerNotes",
					parsedTags: t
				});
			} else if (hu(c)) {
				let t = Pc.read(O || e, x, c.MakerNote.__offset, i, r, p);
				m.makerNotes = t, p.shouldReturnGroup("makerNotes") && h.push({
					type: "mergeGroupAssign",
					groupKey: "makerNotes",
					parsedTags: t
				});
			}
		}
		c.MakerNote && delete c.MakerNote.__offset, p.shouldReturnGroup("exif") && h.push({
			type: "mergeGroupAssign",
			groupKey: "exif",
			parsedTags: c
		}), !t && y && (h.push(y), y = void 0);
	}
	if (k.USE_JPEG && k.USE_IPTC && S !== void 0 && p.shouldParseGroup("iptc")) {
		let t = $("iptc", oo.read(e, S, i, p), p);
		m.iptc = t, p.shouldReturnGroup("iptc") && h.push({
			type: "mergeGroupAssign",
			groupKey: "iptc",
			parsedTags: t
		});
	}
	if (k.USE_XMP && fu(C) && p.shouldParseGroup("xmp")) {
		let t = $("xmp", Wo.read(pe || e, C, o), p);
		m.xmp = t, p.shouldReturnGroup("xmp") && h.push({
			type: "mergeXmpGroupAssign",
			parsedTags: t
		});
	}
	if (k.USE_JXL && k.USE_EXIF && ce && x === void 0 && p.shouldParseGroup("exif") && n) {
		let t = new DataView(e.buffer, e.byteOffset + ce.dataOffset, ce.length);
		_.push(te(t, v, void 0, "dataview", f).then((e) => {
			let t = St(e, 0), { tags: n } = Qi.read(e, t, i, r, p);
			n.Thumbnail && delete n.Thumbnail, g.brobExif = n;
		}).catch(() => {
			g.brobExif = {};
		})), h.push({
			type: "mergeBrobExifDeferred",
			deferredKey: "brobExif"
		});
	}
	if (k.USE_JXL && k.USE_XMP && le && !fu(C) && p.shouldParseGroup("xmp") && n) {
		let t = new DataView(e.buffer, e.byteOffset + le.dataOffset, le.length);
		_.push(te(t, v, void 0, "dataview", f).then((e) => {
			g.brobXmp = Wo.read(e, [{
				dataOffset: 0,
				length: e.byteLength
			}], o);
		}).catch(() => {
			g.brobXmp = {};
		})), h.push({
			type: "mergeBrobXmpDeferred",
			deferredKey: "brobXmp"
		});
	}
	if ((k.USE_JPEG || k.USE_WEBP) && k.USE_ICC && pu(w) && p.shouldParseGroup("icc")) {
		let t = ac.read(e, w, n, f);
		if (Ql(t)) {
			if (!n) throw Error("Promise is required when async mode is enabled.");
			_.push(t.then((e) => {
				g.iccApp = e;
			})), h.push({
				type: "mergeIccDeferred",
				deferredKey: "iccApp"
			});
		} else {
			let e = $("icc", t, p);
			m.icc = e, p.shouldReturnGroup("icc") && h.push({
				type: "mergeGroupAssign",
				groupKey: "icc",
				parsedTags: e
			});
		}
	}
	if (k.USE_MPF && ae !== void 0 && p.shouldParseGroup("mpf")) {
		let t = $("mpf", na.read(e, ae, i, r, p), p);
		m.mpf = t, p.shouldReturnGroup("mpf") && h.push({
			type: "mergeGroupAssign",
			groupKey: "mpf",
			parsedTags: t
		});
	}
	if (k.USE_PNG && k.USE_PNG_FILE && oe !== void 0 && p.shouldParseGroup("png")) {
		let t = $("png", Vc.read(e, oe), p);
		m.pngFile = t, p.shouldReturnGroup("png") && h.push({
			type: "mergePngFile",
			parsedTags: t
		});
	}
	if (k.USE_PNG && gu(T) && (p.shouldParseGroup("png") || p.shouldParseGroup("exif") || p.shouldParseGroup("iptc"))) {
		let { readTags: t, readTagsPromise: a } = qc.read(e, T, n, i, r, p, f);
		ee = !!a, h.push({
			type: "processPngTextReadTags",
			readTags: t
		}), a && (_.push(a.then((e) => {
			g.pngTextTagList = e;
		})), h.push({
			type: "processPngTextReadTagsDeferredList",
			deferredKey: "pngTextTagList"
		}));
	}
	if (k.USE_PNG && E !== void 0 && p.shouldParseGroup("png")) {
		let t = $("png", hl.read(e, E), p);
		m.pngChunk = t, p.shouldReturnGroup("png") && h.push({
			type: "mergePngChunk",
			parsedTags: t
		});
	}
	if (k.USE_WEBP && D !== void 0 && p.shouldParseGroup("riff")) {
		let t = $("riff", Sl.read(e, D), p);
		m.riff = t, p.shouldReturnGroup("riff") && h.push({
			type: "mergeGroupMerge",
			groupKey: "riff",
			parsedTags: t
		});
	}
	if (k.USE_GIF && se !== void 0 && p.shouldParseGroup("gif")) {
		let t = $("gif", Al.read(e, se), p);
		m.gif = t, p.shouldReturnGroup("gif") && h.push({
			type: "mergeGroupMerge",
			groupKey: "gif",
			parsedTags: t
		});
	}
	if (k.USE_JXL && ue !== void 0 && p.shouldParseGroup("file")) {
		let t = $("file", Sa.read(e, ue), p);
		m.file = t, p.shouldReturnGroup("file") && h.push({
			type: "mergeGroupAssign",
			groupKey: "file",
			parsedTags: t
		});
	}
	if (h.push({ type: "gps" }), h.push({ type: "composite" }), h.push({ type: "thumbnail" }), h.push({ type: "fileType" }), t && a && h.push({
		type: "metadataRange",
		metadataBlocks: de,
		metadataTruncated: !!fe
	}), !me) throw new tu.MetadataMissingError();
	let he = {
		objectAssign: u,
		hasPngTextData: gu,
		filterTagsForParse: $,
		filterTagsForReturn: cu,
		getGpsGroupFromExifTags: du,
		Composite: Vl,
		Thumbnail: Il
	};
	if (n) return l(), Promise.all(_).then(() => Kl({
		mergeSteps: h,
		deferredResults: g,
		parsedGroups: m,
		expanded: t,
		tagFilter: p,
		dataView: e,
		tiffHeaderOffset: x,
		exifDataView: O,
		fileType: re,
		pngTextChunks: T,
		pngTextIsAsync: ee,
		thumbnailIfdTags: ne,
		deps: he
	}));
	return Kl({
		mergeSteps: h,
		deferredResults: g,
		parsedGroups: m,
		expanded: t,
		tagFilter: p,
		dataView: e,
		tiffHeaderOffset: x,
		exifDataView: O,
		fileType: re,
		pngTextChunks: T,
		pngTextIsAsync: ee,
		thumbnailIfdTags: ne,
		deps: he
	});
	function ge(e) {
		if (!e) return !0;
		let { composite: t, file: n } = e;
		return t === !0 || Array.isArray(t) && t.length > 0 ? !0 : !(Array.isArray(n) && n.length === 1 && n[0] === "FileType");
	}
}
function su(e, t, n, r, i) {
	try {
		return Qi.read(e, t, n, r, i);
	} catch {
		return {
			tags: {},
			byteOrder: A.BIG_ENDIAN
		};
	}
}
function $(e, t, n) {
	return n.isActive ? lu(e, t, n.shouldParseTag) : t;
}
function cu(e, t, n) {
	return n.isActive ? lu(e, t, n.shouldReturnTag) : t;
}
function lu(e, t, n) {
	if (!t) return t;
	let r = {};
	for (let a in t) {
		let o = t[a];
		n(e, a, i(o)) && (r[a] = o);
	}
	return r;
	function i(e) {
		if (e) return Array.isArray(e) ? e.length === 0 ? void 0 : e[0].id : e.id;
	}
}
function uu({ fileType: e, fileDataOffset: t, jfifDataOffset: n, tiffHeaderOffset: r, iptcDataOffset: i, xmpChunks: a, iccChunks: o, mpfDataOffset: s, pngHeaderOffset: c, pngTextChunks: l, pngChunkOffsets: u, vp8xChunkOffset: d, gifHeaderOffset: f, jxlCodestreamOffset: p }) {
	return !!e || k.USE_JPEG && k.USE_FILE && t !== void 0 || k.USE_JPEG && k.USE_JFIF && n !== void 0 || k.USE_EXIF && r !== void 0 || k.USE_JPEG && k.USE_IPTC && i !== void 0 || k.USE_XMP && fu(a) || (k.USE_JPEG || k.USE_WEBP) && k.USE_ICC && pu(o) || k.USE_MPF && s !== void 0 || k.USE_PNG && k.USE_PNG_FILE && c !== void 0 || k.USE_PNG && gu(l) || k.USE_PNG && u !== void 0 || k.USE_WEBP && d !== void 0 || k.USE_GIF && f !== void 0 || k.USE_JXL && p !== void 0;
}
function du(e) {
	let t;
	if (e.GPSLatitude && e.GPSLatitudeRef) {
		t ||= {};
		try {
			t.Latitude = Me(e.GPSLatitude.value), e.GPSLatitudeRef.value.join("") === "S" && (t.Latitude = -t.Latitude);
		} catch {}
	}
	if (e.GPSLongitude && e.GPSLongitudeRef) {
		t ||= {};
		try {
			t.Longitude = Me(e.GPSLongitude.value), e.GPSLongitudeRef.value.join("") === "W" && (t.Longitude = -t.Longitude);
		} catch {}
	}
	if (e.GPSAltitude && e.GPSAltitudeRef) {
		t ||= {};
		try {
			t.Altitude = e.GPSAltitude.value[0] / e.GPSAltitude.value[1], e.GPSAltitudeRef.value === 1 && (t.Altitude = -t.Altitude);
		} catch {}
	}
	if (t) return t;
}
function fu(e) {
	return Array.isArray(e) && e.length > 0;
}
function pu(e) {
	return Array.isArray(e) && e.length > 0;
}
function mu(e) {
	return e.Make && e.Make.value && Array.isArray(e.Make.value) && e.Make.value[0] === "Canon" && e.MakerNote && e.MakerNote.__offset;
}
function hu(e) {
	return e.MakerNote.value.length > 7 && s(e.MakerNote.value.slice(0, 7)) === "PENTAX " && e.MakerNote.__offset;
}
function gu(e) {
	return Array.isArray(e) && e.length > 0;
}
//#endregion
export { nu as default };
