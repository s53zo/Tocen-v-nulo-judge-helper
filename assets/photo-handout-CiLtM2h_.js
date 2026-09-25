import { i as e, n as t } from "./photo-compliance-B0dXBoRF.js";
import { i as n, n as r } from "./es-DomlnFlq.js";
import i from "./fontkit.es-DHpaJX9V.js";
//#region src/photo-handout.ts
var a = [595.28, 841.89];
function o(e) {
	return Array.from(e.normalize("NFC"), (e) => e.charCodeAt(0) < 32 ? " " : e).join("");
}
function s(e, t) {
	if (t === null) return 0;
	let n = e.taskAnalysis?.alongRouteM ?? e.analysis?.alongRouteM;
	return +(n !== void 0 && n > t);
}
function c(e, t) {
	let n = e.record.classification === "enroute", r = t.record.classification === "enroute";
	return n && r ? e.record.identifier.localeCompare(t.record.identifier) : n === r ? (e.record.taskAnalysis?.alongRouteM ?? e.record.analysis?.alongRouteM ?? 0) - (t.record.taskAnalysis?.alongRouteM ?? t.record.analysis?.alongRouteM ?? 0) : n ? -1 : 1;
}
function l(e, t, r, i, a, s, c, l = n(.12, .15, .18), u = Infinity) {
	let d = o(r).split(/\s+/), f = "", p = 0;
	for (let n of d) {
		let r = f ? `${f} ${n}` : n;
		if (t.widthOfTextAtSize(r, c) > s && f) {
			if (p >= u - 1) {
				let n = `${f}…`;
				for (; n.length > 1 && t.widthOfTextAtSize(n, c) > s;) n = `${n.slice(0, -2)}…`;
				return e.drawText(n, {
					x: i,
					y: a - p * (c + 2),
					font: t,
					size: c,
					color: l
				}), p + 1;
			}
			e.drawText(f, {
				x: i,
				y: a - p * (c + 2),
				font: t,
				size: c,
				color: l
			}), p += 1, f = n;
		} else f = r;
	}
	return f && e.drawText(f, {
		x: i,
		y: a - p * (c + 2),
		font: t,
		size: c,
		color: l
	}), f ? p + 1 : p;
}
async function u(u, d, f, p) {
	let m = await r.create();
	m.registerFontkit(i);
	let h = await m.embedFont(f, { subset: !0 }), g = [
		{
			title: "Judge control-point solutions",
			items: u.filter(({ record: e }) => ["control-correct", "control-false"].includes(e.classification)).sort(c)
		},
		{
			title: `Judge solutions before ${p.splitWaypoint}`,
			items: u.filter(({ record: e }) => e.classification === "enroute" && s(e, p.splitAfterM) === 0).sort(c)
		},
		{
			title: `Judge solutions after ${p.splitWaypoint}`,
			items: u.filter(({ record: e }) => e.classification === "enroute" && s(e, p.splitAfterM) === 1).sort(c)
		}
	].filter((e) => e.items.length > 0), _ = (a[1] - 40 - 38 - 14) / 2;
	for (let t of g) for (let r = 0; r < t.items.length; r += 2) {
		let i = m.addPage(a);
		i.drawText(t.title, {
			x: 20,
			y: a[1] - 20 - 15,
			size: 15,
			font: h,
			color: n(.08, .26, .2)
		});
		for (let s = 0; s < 2; s += 1) {
			let c = t.items[r + s];
			if (!c) continue;
			let u = a[1] - 20 - 38 - s * (_ + 14), d = _ - 34, f = await m.embedJpg(c.jpeg), p = Math.min((a[0] - 40) / f.width, d / f.height), g = f.width * p, v = f.height * p, y = (a[0] - g) / 2, b = u - v;
			i.drawImage(f, {
				x: y,
				y: b,
				width: g,
				height: v
			}), i.drawRectangle({
				x: y,
				y: b,
				width: g,
				height: v,
				borderWidth: .8,
				borderColor: n(.72, .75, .76)
			});
			let x = c.record, S = o(x.identifier || "?"), C = Math.max(34, h.widthOfTextAtSize(S, 20) + 14);
			i.drawRectangle({
				x: y + 8,
				y: b + v - 37,
				width: C,
				height: 29,
				color: n(1, 1, 1),
				opacity: .9,
				borderColor: n(.08, .26, .2),
				borderWidth: 1
			}), i.drawText(S, {
				x: y + 15,
				y: b + v - 31,
				size: 20,
				font: h,
				color: n(.08, .26, .2)
			});
			let w = x.findings.filter(e), T = w.some((e) => e.severity === "violation") ? x.exceptionAccepted ? "ACCEPTED EXCEPTION — VIOLATIONS RETAINED" : "AGAINST THE RULES" : w.some((e) => e.severity === "warning") ? "MANUAL REVIEW" : "OK";
			l(i, h, `${x.identifier || "?"} · ${x.classification}${x.linkedWaypoint ? ` · ${x.linkedWaypoint}` : ""} · ${T} · ${x.fileName}`, 20, b - 13, a[0] - 40, 8, n(.12, .15, .18), 2), x.generatedOrthophoto && l(i, h, `${x.generatedOrthophoto.targetSource ? `${x.generatedOrthophoto.targetSource.name ?? x.generatedOrthophoto.targetSource.featureType} · ${x.generatedOrthophoto.targetSource.attribution} · ` : ""}${x.generatedOrthophoto.attribution} · modeled coverage ${x.generatedOrthophoto.coverageWidthM.toFixed(0)} × ${x.generatedOrthophoto.coverageHeightM.toFixed(0)} m`, 20, b - 33, a[0] - 40, 6, n(.32, .35, .38), 2);
		}
		i.drawText(`Page ${m.getPageCount()}`, {
			x: a[0] - 70,
			y: 8,
			size: 7,
			font: h,
			color: n(.4, .4, .4)
		});
	}
	if (p.includeSummary) {
		let e = () => {
			let e = m.addPage(a);
			return e.drawText("Judge photo solution summary", {
				x: 20,
				y: a[1] - 42,
				size: 18,
				font: h,
				color: n(.08, .26, .2)
			}), e.drawText(`${d.status.toUpperCase()} · ${u.length} photos · ${d.violationCount} violations · ${d.warningCount} manual checks`, {
				x: 20,
				y: a[1] - 68,
				size: 10,
				font: h
			}), e;
		}, r = e(), i = a[1] - 94;
		for (let o of d.findings.filter((e) => e.severity !== "pass")) {
			i < 58 && (r = e(), i = a[1] - 94);
			let s = u.some(({ record: e }) => e.id === o.photoId && e.exceptionAccepted && o.severity === "violation" && t(o) === "primary"), c = l(r, h, `${o.rule} · ${o.affected}: ${o.measured}; permitted ${o.permitted}.${s ? " ACCEPTED JUDGE EXCEPTION; VIOLATION RETAINED." : ""}`, 20, i, a[0] - 40, 8, n(.12, .15, .18), 3);
			i -= Math.max(30, c * 11 + 8);
		}
	}
	return m.getPageCount() === 0 && m.addPage(a).drawText("No photos were selected.", {
		x: 20,
		y: a[1] - 42,
		size: 14,
		font: h
	}), m.save();
}
async function d(e, t, l) {
	let u = await r.create();
	u.registerFontkit(i);
	let d = await u.embedFont(t, { subset: !0 }), f = [
		{
			title: "Control-point photos",
			items: e.filter(({ record: e }) => ["control-correct", "control-false"].includes(e.classification)).sort(c)
		},
		{
			title: `Before ${l.splitWaypoint}`,
			items: e.filter(({ record: e }) => e.classification === "enroute" && s(e, l.splitAfterM) === 0).sort(c)
		},
		{
			title: `After ${l.splitWaypoint}`,
			items: e.filter(({ record: e }) => e.classification === "enroute" && s(e, l.splitAfterM) === 1).sort(c)
		}
	].filter((e) => e.items.length > 0), p = (a[1] - 40 - 38 - 14) / 2;
	for (let e of f) for (let t = 0; t < e.items.length; t += 2) {
		let r = u.addPage(a);
		r.drawText(e.title, {
			x: 20,
			y: a[1] - 20 - 15,
			size: 15,
			font: d,
			color: n(.08, .26, .2)
		});
		for (let i = 0; i < 2; i += 1) {
			let s = e.items[t + i];
			if (!s) continue;
			let c = a[1] - 20 - 38 - i * (p + 14), l = await u.embedJpg(s.jpeg), f = Math.min((a[0] - 40) / l.width, p / l.height), m = l.width * f, h = l.height * f, g = (a[0] - m) / 2, _ = c - h;
			r.drawImage(l, {
				x: g,
				y: _,
				width: m,
				height: h
			}), r.drawRectangle({
				x: g,
				y: _,
				width: m,
				height: h,
				borderWidth: .8,
				borderColor: n(.72, .75, .76)
			});
			let v = o(s.record.identifier || "?"), y = Math.max(34, d.widthOfTextAtSize(v, 20) + 14);
			r.drawRectangle({
				x: g + 8,
				y: _ + h - 37,
				width: y,
				height: 29,
				color: n(1, 1, 1),
				opacity: .9,
				borderColor: n(.08, .26, .2),
				borderWidth: 1
			}), r.drawText(v, {
				x: g + 15,
				y: _ + h - 31,
				size: 20,
				font: d,
				color: n(.08, .26, .2)
			});
		}
	}
	return u.getPageCount() === 0 && u.addPage(a).drawText("No photos were selected.", {
		x: 20,
		y: a[1] - 42,
		size: 14,
		font: d
	}), u.save();
}
//#endregion
export { d as buildCompetitorPhotoHandout, u as buildPhotoHandout };
