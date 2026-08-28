import e from "./fontkit.es-B5FHmNtr.js";
import { PDFDocument as t, rgb as n } from "./es-DLUcyPiH.js";
//#region src/photo-handout.ts
var r = [595.28, 841.89];
function i(e) {
	return Array.from(e.normalize("NFC"), (e) => e.charCodeAt(0) < 32 ? " " : e).join("");
}
function a(e, t) {
	if (t === null) return 0;
	let n = e.taskAnalysis?.alongRouteM ?? e.analysis?.alongRouteM;
	return +(n !== void 0 && n > t);
}
function o(e, t, r, a, o, s, c, l = n(.12, .15, .18), u = Infinity) {
	let d = i(r).split(/\s+/), f = "", p = 0;
	for (let n of d) {
		let r = f ? `${f} ${n}` : n;
		if (t.widthOfTextAtSize(r, c) > s && f) {
			if (p >= u - 1) {
				let n = `${f}…`;
				for (; n.length > 1 && t.widthOfTextAtSize(n, c) > s;) n = `${n.slice(0, -2)}…`;
				return e.drawText(n, {
					x: a,
					y: o - p * (c + 2),
					font: t,
					size: c,
					color: l
				}), p + 1;
			}
			e.drawText(f, {
				x: a,
				y: o - p * (c + 2),
				font: t,
				size: c,
				color: l
			}), p += 1, f = n;
		} else f = r;
	}
	return f && e.drawText(f, {
		x: a,
		y: o - p * (c + 2),
		font: t,
		size: c,
		color: l
	}), f ? p + 1 : p;
}
async function s(s, c, l, u) {
	let d = await t.create();
	d.registerFontkit(e);
	let f = await d.embedFont(l, { subset: !0 }), p = [{
		title: `Photos before ${u.splitWaypoint}`,
		items: s.filter(({ record: e }) => a(e, u.splitAfterM) === 0)
	}, {
		title: `Photos after ${u.splitWaypoint}`,
		items: s.filter(({ record: e }) => a(e, u.splitAfterM) === 1)
	}].filter((e) => e.items.length > 0), m = (r[1] - 40 - 38 - 14) / 2;
	for (let e of p) for (let t = 0; t < e.items.length; t += 2) {
		let a = d.addPage(r);
		a.drawText(e.title, {
			x: 20,
			y: r[1] - 20 - 15,
			size: 15,
			font: f,
			color: n(.08, .26, .2)
		});
		for (let s = 0; s < 2; s += 1) {
			let c = e.items[t + s];
			if (!c) continue;
			let l = r[1] - 20 - 38 - s * (m + 14), u = m - 34, p = await d.embedJpg(c.jpeg), h = Math.min((r[0] - 40) / p.width, u / p.height), g = p.width * h, _ = p.height * h, v = (r[0] - g) / 2, y = l - _;
			a.drawImage(p, {
				x: v,
				y,
				width: g,
				height: _
			}), a.drawRectangle({
				x: v,
				y,
				width: g,
				height: _,
				borderWidth: .8,
				borderColor: n(.72, .75, .76)
			});
			let b = c.record, x = i(b.identifier || "?"), S = Math.max(34, f.widthOfTextAtSize(x, 20) + 14);
			a.drawRectangle({
				x: v + 8,
				y: y + _ - 37,
				width: S,
				height: 29,
				color: n(1, 1, 1),
				opacity: .9,
				borderColor: n(.08, .26, .2),
				borderWidth: 1
			}), a.drawText(x, {
				x: v + 15,
				y: y + _ - 31,
				size: 20,
				font: f,
				color: n(.08, .26, .2)
			});
			let C = b.findings.some((e) => e.severity === "violation") ? b.exceptionAccepted ? "ACCEPTED EXCEPTION — VIOLATIONS RETAINED" : "AGAINST THE RULES" : b.findings.some((e) => e.severity === "warning") ? "MANUAL REVIEW" : "OK";
			o(a, f, `${b.identifier || "?"} · ${b.classification}${b.linkedWaypoint ? ` · ${b.linkedWaypoint}` : ""} · ${C} · ${b.fileName}`, 20, y - 13, r[0] - 40, 8, n(.12, .15, .18), 2);
		}
		a.drawText(`Page ${d.getPageCount()}`, {
			x: r[0] - 70,
			y: 8,
			size: 7,
			font: f,
			color: n(.4, .4, .4)
		});
	}
	if (u.includeSummary) {
		let e = () => {
			let e = d.addPage(r);
			return e.drawText("Photo compliance summary", {
				x: 20,
				y: r[1] - 42,
				size: 18,
				font: f,
				color: n(.08, .26, .2)
			}), e.drawText(`${c.status.toUpperCase()} · ${s.length} photos · ${c.violationCount} violations · ${c.warningCount} manual checks`, {
				x: 20,
				y: r[1] - 68,
				size: 10,
				font: f
			}), e;
		}, t = e(), i = r[1] - 94;
		for (let a of c.findings.filter((e) => e.severity !== "pass")) {
			i < 58 && (t = e(), i = r[1] - 94);
			let c = s.some(({ record: e }) => e.id === a.photoId && e.exceptionAccepted && a.severity === "violation"), l = o(t, f, `${a.rule} · ${a.affected}: ${a.measured}; permitted ${a.permitted}.${c ? " ACCEPTED JUDGE EXCEPTION; VIOLATION RETAINED." : ""}`, 20, i, r[0] - 40, 8, n(.12, .15, .18), 3);
			i -= Math.max(30, l * 11 + 8);
		}
	}
	return d.getPageCount() === 0 && d.addPage(r).drawText("No photos were selected.", {
		x: 20,
		y: r[1] - 42,
		size: 14,
		font: f
	}), d.save();
}
//#endregion
export { s as buildPhotoHandout };
