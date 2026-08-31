//#region src/domain.ts
var e = .514444, t = [
	70,
	75,
	80,
	85,
	90
], n = .05, r = {
	mapScaleDenominator: 25e4,
	minimumDistanceNm: 70,
	maximumDistanceNm: 120,
	minimumLegDistanceNm: 5,
	allowedSpeedsKt: t
};
function i(t) {
	let n = (t.trim().toLowerCase().replace(/\s+/g, "") || "75").match(/^(\d+(?:[.,]\d+)?)(kt|kts|mph|km\/h|kmh|kph)?$/);
	if (!n) throw Error("Groundspeed must be a positive number followed by kt, mph, or km/h.");
	let r = Number.parseFloat(n[1].replace(",", "."));
	if (!Number.isFinite(r) || r <= 0) throw Error("Groundspeed must be greater than zero.");
	let i = n[2] ?? "kt", a = i === "mph" ? "mph" : i.startsWith("km") || i === "kph" ? "kmh" : "kt", o = a === "kt" ? e : a === "mph" ? .44704 : .2777777777777778, s = Number.isInteger(r) ? r.toFixed(0) : r.toFixed(1), c = a === "kmh" ? "km/h" : a;
	return {
		value: r,
		unit: a,
		metersPerSecond: r * o,
		knots: r * o / e,
		label: `${s} ${c}`
	};
}
function a(e) {
	return e / 1852;
}
function o(e, t, i, o) {
	let s = a(e.totalDistance), c = Math.ceil(s / 10), l = e.legs.filter((e) => a(e.length) < r.minimumLegDistanceNm - 1e-9), u = r.allowedSpeedsKt.find((e) => Math.abs(i.knots - e) <= n), d = t.at(0)?.[0].trim().toUpperCase(), f = t.at(-1)?.[0].trim().toUpperCase(), p = [
		{
			rule: "A1.4",
			title: "Official chart scale",
			passed: o === r.mapScaleDenominator,
			message: o === r.mapScaleDenominator ? "Selected map is 1:250,000." : o ? `Selected map is 1:${o.toLocaleString("en-US")}; 1:250,000 is required.` : "Selected map has no fixed 1:250,000 competition scale."
		},
		{
			rule: "A1.5",
			title: "Competition groundspeed",
			passed: u !== void 0,
			message: u === void 0 ? `${i.label} equals ${i.knots.toFixed(2)} kt; allowed values are 70, 75, 80, 85, or 90 kt.` : `${i.label} equals an approved ${u} kt groundspeed.`
		},
		{
			rule: "A2.1.1",
			title: "Route distance",
			passed: s >= r.minimumDistanceNm - 1e-9 && s <= r.maximumDistanceNm + 1e-9,
			message: `${s.toFixed(2)} NM total; required range is 70-120 NM.`
		},
		{
			rule: "A2.1.2",
			title: "Minimum leg distance",
			passed: l.length === 0,
			message: l.length === 0 ? `Every leg is at least ${r.minimumLegDistanceNm} NM.` : `Legs below 5 NM: ${l.map((e) => `${e.fromName}-${e.toName} (${a(e.length).toFixed(2)} NM)`).join(", ")}.`
		},
		{
			rule: "A2.1.2",
			title: "Control-point limit",
			passed: t.length <= c,
			message: `${t.length} control points; a ${s.toFixed(2)} NM route allows at most ${c}.`
		},
		{
			rule: "A2.1.2",
			title: "Start and finish identifiers",
			passed: d === "SP" && f === "FP",
			message: d === "SP" && f === "FP" ? "Route starts at SP and finishes at FP." : `First and last control points must be named SP and FP (currently ${d || "missing"} and ${f || "missing"}).`
		}
	], m = p.filter((e) => !e.passed);
	return {
		status: m.length === 0 ? "ok" : "against-rules",
		checks: p,
		violations: m,
		totalDistanceNm: s,
		maximumControlPoints: c,
		manualChecks: [
			{
				rule: "A2.1.1",
				message: "Confirm that the route finishes with a precision landing."
			},
			{
				rule: "A2.2.1-A2.2.3",
				message: "Confirm every control point has an unambiguous navigation-plan description and coordinates or course/distance."
			},
			{
				rule: "A2.3.1-A2.3.5",
				message: "Designate 3-5 timed control points and confirm direction, 500 ft AGL minimum, 0.5 NM approach tolerance, and no prohibited circling."
			},
			{
				rule: "A2.4.1-A2.4.7",
				message: "Check observation tasks: at most 12 route photos, at most 15 tasks, required photo/sign placement, and no task in the 1 NM segment after a control point."
			},
			{
				rule: "A2.6; A3.9",
				message: "Confirm the approved WGS 84 IGC logger setup, chart currency/event approval, weather, airspace, and all VFR operational requirements."
			}
		]
	};
}
function s(e, t, n, r) {
	let [i, a, o, s] = [
		e,
		n,
		n - e,
		r - t
	].map((e) => e * Math.PI / 180), c = Math.sin(o / 2) ** 2 + Math.cos(i) * Math.cos(a) * Math.sin(s / 2) ** 2;
	return 12742e3 * Math.atan2(Math.sqrt(c), Math.sqrt(1 - c));
}
function c(e) {
	if (e.length < 2) throw Error("At least two waypoints are required.");
	let t = [], n = 0;
	for (let r = 0; r < e.length - 1; r += 1) {
		let [i, a, o] = e[r], [c, l, u] = e[r + 1], d = s(a, o, l, u);
		if (d < .01) throw Error(`Waypoints ${i} and ${c} have the same position.`);
		t.push({
			fromName: i,
			toName: c,
			fromLat: a,
			fromLon: o,
			toLat: l,
			toLon: u,
			length: d,
			cumulativeStart: n
		}), n += d;
	}
	return {
		legs: t,
		totalDistance: n
	};
}
function l(e, t, n, r) {
	let [i, a, o] = [
		e,
		n,
		r - t
	].map((e) => e * Math.PI / 180), s = Math.sin(o) * Math.cos(a), c = Math.cos(i) * Math.sin(a) - Math.sin(i) * Math.cos(a) * Math.cos(o);
	return (Math.atan2(s, c) * 180 / Math.PI + 360) % 360;
}
function u(e) {
	return (Math.round(e) % 360 + 360) % 360;
}
function d(e, t, n, r) {
	if (!Number.isFinite(r) || r <= 0) throw Error("Groundspeed must be greater than zero.");
	return new Map(t.map((t, i) => {
		let a = i === 0 ? 0 : e.legs[i - 1].cumulativeStart + e.legs[i - 1].length;
		return [t[0], n + a / r];
	}));
}
function f(e, t, n, r) {
	if (n <= 0 || r <= 0) throw Error("Speed and minute-marker interval must be greater than zero.");
	let i = [], a = t + e.totalDistance / n;
	for (let o = 0; o <= a + 1e-6; o += r) {
		let r = (o - t) * n;
		if (r < -1e-6 || r > e.totalDistance + 1e-6) continue;
		let a = r;
		for (let t of e.legs) {
			if (a <= t.length + 1e-6) {
				let e = a / t.length;
				i.push({
					minute: o,
					leg: t,
					ratio: e,
					lat: t.fromLat + e * (t.toLat - t.fromLat),
					lon: t.fromLon + e * (t.toLon - t.fromLon)
				});
				break;
			}
			a -= t.length;
		}
	}
	return i;
}
//#endregion
//#region src/photo-compliance.ts
var p = 1852, m = 500, h = /* @__PURE__ */ new Set(["task-position-missing"]), g = /* @__PURE__ */ new Set([
	"camera-angle-missing",
	"capture-altitude-missing",
	"focal-length",
	"focal-length-missing",
	"judge-content-review"
]);
function _(e) {
	return h.has(e.code) ? "action" : g.has(e.code) ? "audit" : "primary";
}
function v(e) {
	return _(e) !== "audit";
}
function y(e) {
	let t = e.filter(v);
	return {
		violationCount: t.filter((e) => e.severity === "violation").length,
		warningCount: t.filter((e) => e.severity === "warning").length
	};
}
function b(e) {
	return e.exceptionAccepted || !e.findings.some((e) => v(e) && e.severity === "violation");
}
function x(e, t, n) {
	let r = e[0].trim().toUpperCase();
	return t === 0 && r === "SP" ? "start" : t === n.length - 1 && r === "FP" ? "finish" : /^TP\d+$/.test(r) ? "turning-point" : "control-point";
}
function S(e, t) {
	if (!e.linkedWaypoint) return null;
	let n = t.findIndex(([t]) => t.trim().toUpperCase() === e.linkedWaypoint?.trim().toUpperCase());
	return n < 0 ? null : {
		point: t[n],
		index: n
	};
}
function C(e) {
	return e.classification !== "reference";
}
function w(e, t) {
	if (!C(e)) return !1;
	let n = S(e, t);
	return !n || x(n.point, n.index, t) !== "turning-point";
}
function T(e, t, n, r, i, a, o, s = null) {
	return {
		photoId: s,
		severity: e,
		code: t,
		rule: n,
		affected: r,
		message: i,
		measured: a,
		permitted: o
	};
}
function E(e, t) {
	if (e.taskLatitude.reliable && e.taskLongitude.reliable && e.taskLatitude.value !== null && e.taskLongitude.value !== null) return [e.taskLatitude.value, e.taskLongitude.value];
	let n = S(e, t);
	return n ? [n.point[1], n.point[2]] : null;
}
function D(e, t) {
	let n = [], r = e.filter((e) => e.classification === "enroute"), i = e.filter((e) => w(e, t));
	n.push(T(r.length <= 12 ? "pass" : "violation", "enroute-photo-count", "A2.4.5", "route", r.length <= 12 ? "En-route photo count is within the limit." : "Too many en-route photos.", `${r.length} en-route photos`, "maximum 12"), T(i.length <= 15 ? "pass" : "violation", "route-task-count", "A2.4.6", "route", i.length <= 15 ? "Route task count is within the limit." : "Too many route tasks.", `${i.length} route tasks`, "maximum 15, excluding verified photos/signs at turning points"));
	let a = r.map((e) => e.identifier.trim().toUpperCase()), o = a.every((e) => /^[A-Z]$/.test(e)) && new Set(a).size === a.length;
	n.push(T(o ? "pass" : "violation", "enroute-identifiers", "A2.4.5", "route", o ? "En-route photo identifiers are unique single letters." : "En-route photo identifiers must be unique single letters.", a.join(", ") || "no en-route photos", "unique letters A-Z"));
	let c = new Map(t.map(([e, t, n]) => [e.toUpperCase(), {
		latitude: t,
		longitude: n
	}]));
	for (let r of e) {
		let e = `${r.identifier || "?"} (${r.fileName})`, i = (t, n, i, a, o, s) => T(t, n, i, e, a, o, s, r.id), a = r.metadata.latitude.reliable && r.metadata.longitude.reliable, o = S(r, t);
		if (r.importError && n.push(i("warning", "photo-import-error", "A2.4.1-A2.4.8", "The image or metadata import failed and requires manual review.", r.importError, "readable image and reviewable metadata")), r.linkedWaypoint && !o && n.push(i("warning", "stale-waypoint-link", "A2.4.2-A2.4.7", "The linked waypoint does not exist in the current route.", r.linkedWaypoint, "an existing route waypoint")), w(r, t)) {
			if (r.taskAnalysis) {
				if (r.taskAnalysis.routePosition !== "on-route") n.push(i("violation", "task-outside-route", "A2.4.6-A2.4.7", "The task lies before SP or after FP.", r.taskAnalysis.routePosition, "a task on the competition route"));
				else {
					let e = r.taskAnalysis.distanceAfterPreviousControlPointM;
					n.push(i(e >= p ? "pass" : "violation", "post-control-spacing", "A2.4.6", e >= p ? "Task is outside the prohibited segment after the previous control point." : `Task is too close after ${r.taskAnalysis.previousControlPoint}.`, `${(e / p).toFixed(2)} NM after ${r.taskAnalysis.previousControlPoint}`, "at least 1.00 NM after a control point"));
				}
				r.taskAnalysis.ambiguousLegIndices.length > 1 && !r.taskAnalysis.manuallySelectedLeg && n.push(i("warning", "task-leg-ambiguous", "A2.4.6-A2.4.7", "Task position matches multiple route legs; confirm its intended sequence.", `candidate legs ${r.taskAnalysis.ambiguousLegIndices.map((e) => e + 1).join(", ")}`, "one confirmed route leg"));
			} else n.push(i("warning", "task-position-missing", "A2.4.6-A2.4.7", "Task spacing and map placement require a separate task/object position.", "reliable task position unavailable", "task/object coordinates or a valid linked waypoint"));
		}
		if (r.classification === "sign-task" && n.push(r.taskAnalysis ? i(r.taskAnalysis.lateralDistanceM <= 100 ? "pass" : "violation", "sign-route-axis-distance", "A2.4.4", r.taskAnalysis.lateralDistanceM <= 100 ? "Sign task is within the route-axis limit." : "Sign task is too far from the route axis.", `${r.taskAnalysis.lateralDistanceM.toFixed(1)} m`, "maximum 100 m") : i("warning", "sign-route-axis-distance-missing", "A2.4.4", "Sign distance from the route requires manual review.", "task position unavailable", "maximum 100 m")), r.classification === "enroute") {
			!r.analysis || !a ? n.push(i("warning", "photo-position-missing", "A2.4.5", "Camera position cannot be checked.", r.analysis ? "GPS is marked unreliable" : "GPS unavailable", "reliable camera position")) : n.push(i(r.analysis.lateralDistanceM <= m ? "pass" : "violation", "route-axis-distance", "A2.4.5", r.analysis.lateralDistanceM <= m ? "Camera is within the configured route-axis screening limit." : "Camera exceeds the configured route-axis screening limit.", `${r.analysis.lateralDistanceM.toFixed(0)} m`, `configured maximum ${m} m`)), r.analysis?.headingDifferenceDeg !== null && r.analysis?.headingDifferenceDeg !== void 0 && r.metadata.headingDeg.reliable && r.metadata.headingReference.value === "true" && r.metadata.headingReference.reliable && a ? n.push(i(r.analysis.headingDifferenceDeg <= 45 ? "pass" : "violation", "camera-angle", "A2.4.5", r.analysis.headingDifferenceDeg <= 45 ? "True camera direction is within the route-leg limit." : "True camera direction exceeds the route-leg limit.", `${r.analysis.headingDifferenceDeg.toFixed(1)} degrees`, "maximum 45 degrees from true leg direction")) : n.push(i("warning", "camera-angle-missing", "A2.4.5", "Camera angle requires manual review.", `heading reference ${r.metadata.headingReference.value ?? "unavailable"}`, "reliable true heading within 45 degrees of leg direction"));
			let e = r.metadata.altitudeAglFt;
			n.push(e.value !== null && e.reliable ? i(e.value >= 500 && e.value <= 1e3 ? "pass" : "violation", "capture-altitude", "A2.4.5", e.value >= 500 && e.value <= 1e3 ? "Capture altitude is within range." : "Capture altitude is outside the permitted range.", `${e.value.toFixed(0)} ft AGL`, "500-1,000 ft AGL") : i("warning", "capture-altitude-missing", "A2.4.5", "Capture altitude requires manual review.", "reliable AGL unavailable", "500-1,000 ft AGL"));
			let t = r.metadata.focalLength35Mm;
			n.push(t.value !== null && t.reliable ? i(t.value >= 50 && t.value <= 70 ? "pass" : "violation", "focal-length", "A2.4.5", t.value >= 50 && t.value <= 70 ? "Equivalent focal length is within range." : "Equivalent focal length is outside the permitted range.", `${t.value.toFixed(1)} mm equivalent`, "50-70 mm equivalent") : i("warning", "focal-length-missing", "A2.4.5", "Focal length requires manual review.", "reliable 35 mm equivalent unavailable", "50-70 mm equivalent"));
		}
		if (r.classification === "control-false") {
			let e = r.linkedWaypoint ? c.get(r.linkedWaypoint.toUpperCase()) : void 0, a = E(r, t);
			if (e && a) {
				let t = s(a[0], a[1], e.latitude, e.longitude);
				n.push(i(t >= p ? "pass" : "violation", "false-control-distance", "A2.4.2", t >= p ? "False object is sufficiently separated from the correct object." : "False object is too close to the correct object.", `${(t / p).toFixed(2)} NM`, "at least 1.00 NM"));
			} else n.push(i("warning", "false-control-distance-missing", "A2.4.2", "False-object separation requires manual review.", "task coordinate or linked waypoint missing", "at least 1.00 NM from correct object"));
		}
		r.classification !== "reference" && n.push(i("warning", "judge-content-review", "A2.4.1-A2.4.8", "A judge must confirm content, quality, identification, map marking, and presentation requirements.", "not safely automatable from metadata", "manual judge confirmation"));
	}
	let { violationCount: l, warningCount: u } = y(n);
	return {
		status: l > 0 ? "against-rules" : u > 0 ? "manual-review" : "ok",
		findings: n,
		violationCount: l,
		warningCount: u,
		enroutePhotoCount: r.length,
		routeTaskCount: i.length
	};
}
//#endregion
export { b as a, c, o as d, s as f, u as h, v as i, f as l, i as m, _ as n, y as o, a as p, w as r, l as s, D as t, d as u };
