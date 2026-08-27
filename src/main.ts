import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import fontkit from '@pdf-lib/fontkit';
import notoSansBoldUrl from 'notosans-fontface/fonts/NotoSans-Bold.ttf?url';
import { degrees, PDFDocument, rgb } from 'pdf-lib';
import { chooseTrueScaleCropPage } from './crop';
import { parseCsv } from './csv';
import { decodeDataUrl } from './data-url';
import {
  bearingDegrees,
  buildRoute,
  computeMinuteMarkers,
  computeWaypointTimes,
  evaluateRouteCompliance,
  metersToNauticalMiles,
  parseSpeed,
  type RouteCompliance,
  roundedBearing,
} from './domain';
import rawMapPresets from './map-presets.json';
import { renderBoundedMapPreview } from './map-preview';
import { loadMapPresets } from './maps';
import { buildPhotoHandout } from './photo-handout';
import { preparePhotoJpeg } from './photo-image';
import { photoAnalysisCsv, photoOverlayKeyCsv, photoSummaryJson } from './photo-output';
import { PhotoWorkflow } from './photo-workflow';
import './styles.css';

const APP_BASE_URL = new URL('./', document.baseURI);
const assetUrl = (path) => new URL(path, APP_BASE_URL).href;

const MAP_PRESETS = loadMapPresets(rawMapPresets, APP_BASE_URL);
const APP_VERSION = '2.0.0';
const DEFAULT_MAP_KEY = 'vfr';
let selectedMapKey = DEFAULT_MAP_KEY;
const ROUTE_WIDTH_SCALE = 2.5;
const TP_RADIUS_SCALE = 20.0;
const TP_FONT_SCALE = 12.0;
const HEADING_FONT_SCALE = 18.0;
const HEADING_OFFSET_SCALE = 60.0;
const MINUTE_TICK_HALF_SCALE = 7.0;
const MINUTE_LINE_WIDTH_SCALE = 1.2;
const MINUTE_LABEL_FONT_SCALE = 10.0;
const MINUTE_LABEL_OFFSET_MULTIPLIER = 4.0;
const TP_LABEL_OFFSET_FACTOR = 0.35;
const DEFAULT_TAKEOFF_TO_SP_MIN = 4.0;
const MM_TO_PT = 72 / 25.4;
const PREVIEW_STEP_TIMEOUT_MS = 15000;
const LABEL_COLLISION_MARGIN = 3.0;
const LABEL_DISTANCE_STEP = 5.0;
const MAX_LABEL_ADJUST_STEPS = 12;

function requiredElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Required page element is missing: #${id}`);
  return element as T;
}

const statusEl = requiredElement<HTMLElement>('status');
const generateBtn = requiredElement<HTMLButtonElement>('generate');
const mapPresetGrid = requiredElement<HTMLElement>('mapPresetGrid');
const mapPresetButtons = Array.from(mapPresetGrid.querySelectorAll<HTMLButtonElement>('[data-map-key]'));
const outputsSection = requiredElement<HTMLElement>('outputs');
const summarySection = requiredElement<HTMLElement>('summary');
const legsTable = requiredElement<HTMLTableSectionElement>('legsTable');
const waypointTable = requiredElement<HTMLTableSectionElement>('waypointTable');
const summaryText = requiredElement<HTMLElement>('summaryText');
const routeComplianceSection = requiredElement<HTMLElement>('routeCompliance');
const complianceBadge = requiredElement<HTMLElement>('complianceBadge');
const complianceTitle = requiredElement<HTMLElement>('complianceTitle');
const complianceSummary = requiredElement<HTMLElement>('complianceSummary');
const complianceChecks = requiredElement<HTMLUListElement>('complianceChecks');
const manualComplianceChecks = requiredElement<HTMLUListElement>('manualComplianceChecks');
const speedInput = requiredElement<HTMLInputElement>('speed');
const minuteIntervalInput = requiredElement<HTMLInputElement>('minuteInterval');
const takeoffBufferInput = requiredElement<HTMLInputElement>('takeoffBuffer');
const styleRouteWidthInput = requiredElement<HTMLInputElement>('styleRouteWidth');
const styleWaypointFontInput = requiredElement<HTMLInputElement>('styleWaypointFont');
const styleHeadingFontInput = requiredElement<HTMLInputElement>('styleHeadingFont');
const styleMinuteLabelFontInput = requiredElement<HTMLInputElement>('styleMinuteLabelFont');
const styleMinuteMarkerSizeInput = requiredElement<HTMLInputElement>('styleMinuteMarkerSize');
const styleMinuteLineWidthInput = requiredElement<HTMLInputElement>('styleMinuteLineWidth');
const waypointTextarea = requiredElement<HTMLTextAreaElement>('waypoints');
const waypointLibrary = requiredElement<HTMLElement>('waypointLibrary');
const waypointLibraryToggle = requiredElement<HTMLButtonElement>('toggleWaypointLibrary');
const waypointLibraryStatus = requiredElement<HTMLElement>('waypointLibraryStatus');
const locationTypeFilter = requiredElement<HTMLSelectElement>('locationTypeFilter');
const locationCountryFilter = requiredElement<HTMLSelectElement>('locationCountryFilter');
const libraryControls = requiredElement<HTMLElement>('libraryControls');
const locationsList = requiredElement<HTMLElement>('locationsList');
const addAllFilteredBtn = requiredElement<HTMLButtonElement>('addAllFiltered');

const downloadPdfLink = requiredElement<HTMLAnchorElement>('downloadPdf');
const downloadOverlayLink = requiredElement<HTMLAnchorElement>('downloadOverlay');
const downloadCroppedLink = requiredElement<HTMLAnchorElement>('downloadCropped');
const downloadSummaryLink = requiredElement<HTMLAnchorElement>('downloadSummary');
const downloadPhotoAnalysisLink = requiredElement<HTMLAnchorElement>('downloadPhotoAnalysis');
const downloadPhotoKeyLink = requiredElement<HTMLAnchorElement>('downloadPhotoKey');
const downloadPhotoHandoutLink = requiredElement<HTMLAnchorElement>('downloadPhotoHandout');
downloadCroppedLink.style.display = 'none';
const downloadUrls = {
  pdf: null,
  overlay: null,
  cropped: null,
  summary: null,
  photoAnalysis: null,
  photoKey: null,
  photoHandout: null,
};
let previewObjectUrl = null;
let croppedPreviewController: AbortController | null = null;

function openPreviewInNewTab(url, descriptor) {
  if (!url) {
    setStatus(`Generate the overlay before opening ${descriptor}.`);
    return;
  }
  const newTab = window.open('', '_blank');
  if (!newTab) {
    setStatus(`Pop-up blocked: allow pop-ups to open ${descriptor} in a new tab.`);
    return;
  }
  newTab.opener = null;
  newTab.location = url;
}

function registerDownloadPreview(linkEl, key) {
  if (!linkEl) {
    return;
  }
  const descriptor = linkEl.textContent.trim() || key;
  const openFromEvent = (event) => {
    event.preventDefault();
    openPreviewInNewTab(downloadUrls[key], descriptor);
  };
  linkEl.addEventListener('click', (event) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
      openFromEvent(event);
    }
  });
  linkEl.addEventListener('auxclick', (event) => {
    if (event.button === 1) {
      openFromEvent(event);
    }
  });
  linkEl.addEventListener('keydown', (event) => {
    if ((event.key === 'Enter' || event.key === ' ') && (event.metaKey || event.ctrlKey)) {
      openFromEvent(event);
    }
  });
  linkEl.setAttribute('title', `${descriptor} — click to download, Cmd/Ctrl-click to open in a new tab`);
}

registerDownloadPreview(downloadPdfLink, 'pdf');
registerDownloadPreview(downloadOverlayLink, 'overlay');
registerDownloadPreview(downloadCroppedLink, 'cropped');
registerDownloadPreview(downloadSummaryLink, 'summary');
registerDownloadPreview(downloadPhotoAnalysisLink, 'photoAnalysis');
registerDownloadPreview(downloadPhotoKeyLink, 'photoKey');
registerDownloadPreview(downloadPhotoHandoutLink, 'photoHandout');

const croppedPreviewContainer = requiredElement<HTMLElement>('croppedPreviewContainer');
const croppedPreviewLink = requiredElement<HTMLAnchorElement>('croppedPreviewLink');
const croppedPreviewImage = requiredElement<HTMLImageElement>('croppedPreviewImage');
const croppedPreviewMessage = requiredElement<HTMLElement>('croppedPreviewMessage');

const resultsPlaceholder = requiredElement<HTMLElement>('resultsPlaceholder');
const resultsContent = requiredElement<HTMLElement>('resultsContent');
let hasGeneratedOnce = false;
let cachedPresetKey: string | null = null;
let cachedPresetBuffer: ArrayBuffer | null = null;
const mapStatus = requiredElement<HTMLElement>('mapStatus');
const chartWarning = requiredElement<HTMLElement>('chartWarning');
const osmMapContainer = requiredElement<HTMLElement>('osmMapContainer');
const osmMapEl = requiredElement<HTMLElement>('osmMap');
const DEFAULT_MINUTE_INTERVAL = 1;
const DEFAULT_TAKEOFF_BUFFER = DEFAULT_TAKEOFF_TO_SP_MIN;
const LOCATION_FILE = assetUrl('locations.txt');
let locationLibrary = null;
const locationFilters = { search: '', type: 'all', country: 'all' };
let osmMap = null;
let osmLayers = [];
const OSM_TILE_URL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';

const D96_TM = {
  a: 6378137.0,
  invF: 298.257222101,
  k0: 0.9999,
  lon0Rad: (15 * Math.PI) / 180,
  falseEasting: 500000.0,
  falseNorthing: -5000000.0,
};

const HTML_ESCAPE_RE = /[&<>"']/g;
const HTML_ESCAPE_LOOKUP = new Map([
  ['&', '&amp;'],
  ['<', '&lt;'],
  ['>', '&gt;'],
  ['"', '&quot;'],
  ["'", '&#39;'],
]);

function escapeHtml(text) {
  if (!text) {
    return '';
  }
  return String(text).replace(HTML_ESCAPE_RE, (ch) => HTML_ESCAPE_LOOKUP.get(ch) || ch);
}

function replaceTableRows(tableBody, rows) {
  const fragment = document.createDocumentFragment();
  for (const values of rows) {
    const row = document.createElement('tr');
    for (const value of values) {
      const cell = document.createElement('td');
      cell.textContent = String(value);
      row.appendChild(cell);
    }
    fragment.appendChild(row);
  }
  tableBody.replaceChildren(fragment);
}

function createPdfObjectUrl(bytes: Uint8Array): string {
  const copy = Uint8Array.from(bytes);
  return URL.createObjectURL(new Blob([copy.buffer], { type: 'application/pdf' }));
}

async function loadAssetBytes(url: string, label: string): Promise<Uint8Array> {
  if (url.startsWith('data:')) {
    return decodeDataUrl(url).bytes;
  }
  let response: Response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new Error(`Could not load ${label}. Check the connection and reload the page.`, { cause: error });
  }
  if (!response.ok) {
    throw new Error(`Could not load ${label} (${response.status} ${response.statusText}).`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

function setDownloadUrl(key, url, filename, linkEl) {
  if (downloadUrls[key]) {
    URL.revokeObjectURL(downloadUrls[key]);
  }
  downloadUrls[key] = url;
  linkEl.onclick = null;
  linkEl.href = url;
  linkEl.download = filename;
  if (key === 'pdf') {
    linkEl.textContent = 'Marked Map (PDF)';
  }
  syncResultsVisibility();
}

function clearDownloadUrl(key, linkEl) {
  if (downloadUrls[key]) {
    URL.revokeObjectURL(downloadUrls[key]);
    downloadUrls[key] = null;
  }
  linkEl.removeAttribute('href');
  linkEl.removeAttribute('download');
  linkEl.onclick = null;
  syncResultsVisibility();
}

function syncResultsVisibility() {
  if (!resultsContent) {
    return;
  }
  const hasVisibleOutputs =
    (outputsSection && !outputsSection.hidden) ||
    (summarySection && !summarySection.hidden) ||
    (croppedPreviewContainer && !croppedPreviewContainer.hidden) ||
    !routeComplianceSection.hidden;
  if (resultsPlaceholder && !hasGeneratedOnce) {
    resultsPlaceholder.hidden = hasVisibleOutputs;
  }
  resultsContent.hidden = !hasVisibleOutputs;
}

function setStatus(message, tone = message.startsWith('Error:') ? 'error' : 'neutral') {
  statusEl.textContent = message;
  statusEl.classList.toggle('is-error', tone === 'error');
  statusEl.classList.toggle('is-success', tone === 'success');
  statusEl.classList.toggle('is-warning', tone === 'warning');
}

const photoWorkflow = new PhotoWorkflow(setStatus);

function renderRuleList(
  target: HTMLUListElement,
  entries: Array<{ rule: string; message: string; passed?: boolean }>
) {
  const fragment = document.createDocumentFragment();
  for (const entry of entries) {
    const item = document.createElement('li');
    if (typeof entry.passed === 'boolean') {
      item.className = entry.passed ? 'compliance-check-pass' : 'compliance-check-fail';
      item.append(`${entry.passed ? 'Pass' : 'Fail'} - `);
    }
    const rule = document.createElement('strong');
    rule.textContent = entry.rule;
    item.append(rule, `: ${entry.message}`);
    fragment.appendChild(item);
  }
  target.replaceChildren(fragment);
}

function renderCompliance(compliance: RouteCompliance) {
  const isOk = compliance.status === 'ok';
  routeComplianceSection.classList.toggle('is-ok', isOk);
  routeComplianceSection.classList.toggle('is-fail', !isOk);
  complianceBadge.textContent = isOk ? 'OK' : 'FAIL';
  complianceTitle.textContent = isOk ? 'OK for automated route rules' : 'Route is against the rules';
  complianceSummary.textContent = isOk
    ? 'No violations were detected in the rules this app can calculate. Manual judge checks below are still required.'
    : `${compliance.violations.length} automated rule ${compliance.violations.length === 1 ? 'violation was' : 'violations were'} detected. Generated files are available for correction and review.`;
  renderRuleList(complianceChecks, compliance.checks);
  renderRuleList(manualComplianceChecks, compliance.manualChecks);
  routeComplianceSection.hidden = false;
}

function clearCroppedPreview() {
  croppedPreviewController?.abort('A newer operation replaced this preview.');
  croppedPreviewController = null;
  croppedPreviewContainer.hidden = true;
  croppedPreviewImage.removeAttribute('src');
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
  if (croppedPreviewLink) {
    croppedPreviewLink.setAttribute('aria-disabled', 'true');
    croppedPreviewLink.removeAttribute('href');
  }
  croppedPreviewMessage.textContent = 'JPEG preview generated from the cropped PDF.';
  syncResultsVisibility();
}

if (typeof MutationObserver !== 'undefined') {
  const resultsObserver = new MutationObserver(syncResultsVisibility);
  [outputsSection, summarySection, croppedPreviewContainer, routeComplianceSection].forEach((el) => {
    if (el) {
      resultsObserver.observe(el, { attributes: true, attributeFilter: ['hidden'] });
    }
  });
}
syncResultsVisibility();

function updateMapStatus(message, isError = false) {
  if (!mapStatus) {
    return;
  }
  mapStatus.textContent = message;
  mapStatus.style.color = isError ? '#c62828' : '#555';
}

function updateChartWarning(preset) {
  if (!chartWarning) return;
  if (!preset?.requiresValidityReview) {
    chartWarning.hidden = true;
    chartWarning.textContent = '';
    return;
  }
  chartWarning.textContent = `${preset.label} (${preset.edition} edition): confirm the chart is current and approved for this event before use.`;
  chartWarning.hidden = false;
}

function getPreset(mapKey = selectedMapKey) {
  return MAP_PRESETS[mapKey] || MAP_PRESETS[DEFAULT_MAP_KEY];
}

function isPdfPreset(mapKey = selectedMapKey) {
  const preset = getPreset(mapKey);
  return (preset?.type ?? 'pdf') !== 'osm';
}

async function ensureOsmMap() {
  if (!osmMapEl) {
    return null;
  }
  if (!osmMap) {
    osmMap = L.map(osmMapEl, { center: [46.05, 14.5], zoom: 7, preferCanvas: true });
    L.tileLayer(OSM_TILE_URL, {
      maxZoom: 19,
      attribution: '© OpenStreetMap contributors',
    }).addTo(osmMap);
  }
  setTimeout(() => {
    if (osmMap) {
      osmMap.invalidateSize();
    }
  }, 0);
  return osmMap;
}

function updateMapInputsVisibility() {
  const isPdf = isPdfPreset();
  const preset = getPreset();
  updateChartWarning(preset);
  if (osmMapContainer) {
    osmMapContainer.hidden = isPdf;
  }
  if (!isPdf) {
    updateMapStatus('Interactive OpenStreetMap view active.');
    ensureOsmMap().catch((err) => {
      console.warn('Leaflet map unavailable', err);
    });
  } else {
    updateMapStatus(`Using preset map: ${preset.label}`);
  }
}

async function ensurePresetBuffer(mapKey, options: { forceReload?: boolean } = {}) {
  const preset = getPreset(mapKey);
  if (!preset) {
    throw new Error(`Unknown map preset: ${mapKey}`);
  }
  if (preset.type !== 'pdf') {
    return null;
  }
  if (!options.forceReload && cachedPresetKey === mapKey && cachedPresetBuffer) {
    if (mapKey === selectedMapKey) {
      updateMapStatus(`Using preset map: ${preset.label}`);
    }
    return cachedPresetBuffer;
  }
  if (!preset.url) {
    throw new Error(`Preset '${mapKey}' does not include a bundled PDF URL.`);
  }
  if (mapKey === selectedMapKey) {
    updateMapStatus(`Loading preset map: ${preset.label}...`);
  }
  const response = await fetch(encodeURI(preset.url));
  if (!response.ok) {
    throw new Error(`Failed to load preset PDF (${response.status} ${response.statusText})`);
  }
  const buffer = await response.arrayBuffer();
  cachedPresetKey = mapKey;
  cachedPresetBuffer = buffer;
  if (mapKey === selectedMapKey) {
    updateMapStatus(`Using preset map: ${preset.label}`);
  }
  return buffer;
}

function selectMapPresetButton(mapKey) {
  selectedMapKey = MAP_PRESETS[mapKey] ? mapKey : DEFAULT_MAP_KEY;
  mapPresetButtons.forEach((btn) => {
    const isSelected = btn.dataset.mapKey === selectedMapKey;
    btn.classList.toggle('is-selected', isSelected);
    btn.setAttribute('aria-pressed', String(isSelected));
  });
}

function handleMapPresetChange(newKey = selectedMapKey) {
  selectMapPresetButton(newKey);
  updateMapInputsVisibility();
}

async function renderCroppedPreview(options) {
  croppedPreviewController?.abort('A newer preview was requested.');
  const controller = new AbortController();
  croppedPreviewController = controller;
  croppedPreviewContainer.hidden = false;
  croppedPreviewMessage.classList.remove('danger-text');
  croppedPreviewMessage.textContent = 'Rendering cropped map preview...';
  const timeout = setTimeout(
    () => controller.abort(`Preview exceeded ${PREVIEW_STEP_TIMEOUT_MS / 1000} seconds.`),
    PREVIEW_STEP_TIMEOUT_MS
  );
  try {
    const blob = await renderBoundedMapPreview({ ...options, signal: controller.signal });
    if (controller.signal.aborted || croppedPreviewController !== controller) return false;
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
      previewObjectUrl = null;
    }
    previewObjectUrl = URL.createObjectURL(blob);
    const previewUrl = previewObjectUrl;
    croppedPreviewImage.src = previewUrl;
    croppedPreviewImage.alt = 'Preview of the cropped true-scale route map';
    croppedPreviewLink.href = previewUrl;
    croppedPreviewLink.setAttribute('aria-disabled', 'false');
    croppedPreviewMessage.textContent =
      'Bounded preview generated from the calibrated low-resolution map. Download the cropped PDF for print detail.';
    return true;
  } catch (error) {
    console.error('Could not render cropped map preview', error);
    if (croppedPreviewController !== controller) return false;
    croppedPreviewImage.removeAttribute('src');
    croppedPreviewLink.removeAttribute('href');
    croppedPreviewLink.setAttribute('aria-disabled', 'true');
    croppedPreviewMessage.classList.add('danger-text');
    croppedPreviewMessage.textContent = `Preview could not be rendered: ${error instanceof Error ? error.message : String(error)}. The cropped PDF is still available.`;
    return false;
  } finally {
    clearTimeout(timeout);
    if (croppedPreviewController === controller) croppedPreviewController = null;
  }
}

function parseNumericInput(input, fallback) {
  const rawValue = input?.value?.trim();
  if (!rawValue) {
    input.value = String(fallback);
    return fallback;
  }
  const value = Number.parseFloat(rawValue);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${input.id} must be a positive number.`);
  }
  const rounded = Math.round(value * 100) / 100;
  input.value = rounded % 1 === 0 ? String(Math.round(rounded)) : String(rounded);
  return rounded;
}

function getOverlayStyleConfig() {
  return {
    routeWidth: parseNumericInput(styleRouteWidthInput, ROUTE_WIDTH_SCALE),
    waypointFont: parseNumericInput(styleWaypointFontInput, TP_FONT_SCALE),
    headingFont: parseNumericInput(styleHeadingFontInput, HEADING_FONT_SCALE),
    minuteLabelFont: parseNumericInput(styleMinuteLabelFontInput, MINUTE_LABEL_FONT_SCALE),
    minuteMarkerHalf: parseNumericInput(styleMinuteMarkerSizeInput, MINUTE_TICK_HALF_SCALE),
    minuteLineWidth: parseNumericInput(styleMinuteLineWidthInput, MINUTE_LINE_WIDTH_SCALE),
  };
}

function parseCoordinateToken(token) {
  if (typeof token !== 'string') {
    return Number.NaN;
  }
  const match = token.trim().match(/^([NSEW])?\s*([+-]?\d+(?:[.,]\d+)?)$/i);
  if (!match) {
    return Number.NaN;
  }
  const direction = match[1]?.toUpperCase();
  let value = Number.parseFloat(match[2].replace(',', '.'));
  if ((direction === 'S' || direction === 'W') && value > 0) {
    value *= -1;
  }
  return value;
}

function maybeLatLon(lat, lon) {
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return null;
  }
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) {
    return null;
  }
  return { lat, lon };
}

function isLikelyD96Pair(easting, northing) {
  if (!Number.isFinite(easting) || !Number.isFinite(northing)) {
    return false;
  }
  if (Math.abs(easting) < 1000 || Math.abs(northing) < 1000) {
    return false;
  }
  return easting >= 200000 && easting <= 800000 && northing >= -600000 && northing <= 400000;
}

function convertLatLonToD96Tm(lat, lon) {
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return null;
  }
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;
  const f = 1 / D96_TM.invF;
  const e2 = 2 * f - f * f;
  const ePrime2 = e2 / (1 - e2);
  let deltaLon = lonRad - D96_TM.lon0Rad;
  while (deltaLon > Math.PI) {
    deltaLon -= 2 * Math.PI;
  }
  while (deltaLon < -Math.PI) {
    deltaLon += 2 * Math.PI;
  }
  const sinLat = Math.sin(latRad);
  const cosLat = Math.cos(latRad);
  const tanLat = Math.tan(latRad);
  const n = D96_TM.a / Math.sqrt(1 - e2 * sinLat * sinLat);
  const t = tanLat * tanLat;
  const c = ePrime2 * cosLat * cosLat;
  const a = deltaLon * cosLat;
  const a2 = a * a;
  const a3 = a2 * a;
  const a4 = a2 * a2;
  const a5 = a4 * a;
  const a6 = a4 * a2;
  const m =
    D96_TM.a *
    ((1 - e2 / 4 - (3 * e2 * e2) / 64 - (5 * e2 ** 3) / 256) * latRad -
      ((3 * e2) / 8 + (3 * e2 * e2) / 32 + (45 * e2 ** 3) / 1024) * Math.sin(2 * latRad) +
      ((15 * e2 * e2) / 256 + (45 * e2 ** 3) / 1024) * Math.sin(4 * latRad) -
      ((35 * e2 ** 3) / 3072) * Math.sin(6 * latRad));
  const easting =
    D96_TM.falseEasting +
    D96_TM.k0 * n * (a + ((1 - t + c) * a3) / 6 + ((5 - 18 * t + t * t + 72 * c - 58 * ePrime2) * a5) / 120);
  const northing =
    D96_TM.falseNorthing +
    D96_TM.k0 *
      (m +
        n *
          tanLat *
          (a2 / 2 +
            ((5 - t + 9 * c + 4 * c * c) * a4) / 24 +
            ((61 - 58 * t + t * t + 600 * c - 330 * ePrime2) * a6) / 720));
  return { easting, northing };
}

function convertD96TmToLatLon(easting, northing) {
  if (!Number.isFinite(easting) || !Number.isFinite(northing)) {
    return null;
  }
  const f = 1 / D96_TM.invF;
  const e2 = 2 * f - f * f;
  const ePrime2 = e2 / (1 - e2);
  const x = easting - D96_TM.falseEasting;
  const y = northing - D96_TM.falseNorthing;
  const m = y / D96_TM.k0;
  const mu = m / (D96_TM.a * (1 - e2 / 4 - (3 * e2 * e2) / 64 - (5 * e2 ** 3) / 256));
  const e1 = (1 - Math.sqrt(1 - e2)) / (1 + Math.sqrt(1 - e2));
  const e1Sq = e1 * e1;
  const e1Cu = e1Sq * e1;
  const e1Qu = e1Sq * e1Sq;
  const sin2Mu = Math.sin(2 * mu);
  const sin4Mu = Math.sin(4 * mu);
  const sin6Mu = Math.sin(6 * mu);
  const sin8Mu = Math.sin(8 * mu);
  const phi1 =
    mu +
    ((3 * e1) / 2 - (27 * e1Cu) / 32) * sin2Mu +
    ((21 * e1Sq) / 16 - (55 * e1Qu) / 32) * sin4Mu +
    ((151 * e1Cu) / 96) * sin6Mu +
    ((1097 * e1Qu) / 512) * sin8Mu;
  const sinPhi1 = Math.sin(phi1);
  const cosPhi1 = Math.cos(phi1);
  const tanPhi1 = Math.tan(phi1);
  const c1 = ePrime2 * cosPhi1 * cosPhi1;
  const t1 = tanPhi1 * tanPhi1;
  const n1 = D96_TM.a / Math.sqrt(1 - e2 * sinPhi1 * sinPhi1);
  const r1 = (D96_TM.a * (1 - e2)) / (1 - e2 * sinPhi1 * sinPhi1) ** 1.5;
  const d = x / (n1 * D96_TM.k0);
  const d2 = d * d;
  const d4 = d2 * d2;
  const d6 = d4 * d2;
  const lat =
    phi1 -
    ((n1 * tanPhi1) / r1) *
      (d2 / 2 -
        ((5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * ePrime2) * d4) / 24 +
        ((61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * ePrime2 - 3 * c1 * c1) * d6) / 720);
  const lon =
    D96_TM.lon0Rad +
    (d -
      ((1 + 2 * t1 + c1) * d2 * d) / 6 +
      ((5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * ePrime2 + 24 * t1 * t1) * d4 * d) / 120) /
      cosPhi1;
  return { lat: (lat * 180) / Math.PI, lon: (lon * 180) / Math.PI };
}

function tryInterpretAsD96(first, second) {
  if (isLikelyD96Pair(first, second)) {
    const converted = convertD96TmToLatLon(first, second);
    if (converted && maybeLatLon(converted.lat, converted.lon)) {
      return converted;
    }
  }
  if (isLikelyD96Pair(second, first)) {
    const converted = convertD96TmToLatLon(second, first);
    if (converted && maybeLatLon(converted.lat, converted.lon)) {
      return converted;
    }
  }
  return null;
}

function parseWaypoints(raw: string): Array<[string, number, number]> {
  const points = raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'))
    .map((line) => {
      const parts = line.split(',').map((part) => part.trim());
      if (parts.length !== 3) {
        throw new Error(`Invalid waypoint line: ${line}`);
      }
      const [name, firstCoord, secondCoord] = parts;
      if (!name) {
        throw new Error(`Invalid waypoint line: ${line}`);
      }
      const first = parseCoordinateToken(firstCoord);
      const second = parseCoordinateToken(secondCoord);
      const latLon = maybeLatLon(first, second) || tryInterpretAsD96(first, second);
      if (!latLon) {
        throw new Error(`Invalid waypoint line: ${line}`);
      }
      return [name, Number(latLon.lat.toFixed(6)), Number(latLon.lon.toFixed(6))] as [string, number, number];
    });
  if (points.length < 2) {
    throw new Error('At least two waypoints are required');
  }
  const seenNames = new Set();
  for (const [name] of points) {
    const normalizedName = name.toLocaleUpperCase();
    const hasControlCharacter = Array.from(name).some((character) => {
      const code = character.charCodeAt(0);
      return code < 32 || code === 127;
    });
    if (name.length > 40 || hasControlCharacter) {
      throw new Error(`Invalid waypoint name: ${name}`);
    }
    if (seenNames.has(normalizedName)) {
      throw new Error(`Waypoint names must be unique: ${name}`);
    }
    seenNames.add(normalizedName);
  }
  return points;
}

function dmsToDecimal(raw) {
  const cleaned = (raw || '').trim().replace(/\s+/g, ' ');
  if (!cleaned) {
    return null;
  }
  const parts = cleaned.split(' ');
  const [dir, degPart] = [parts[0][0].toUpperCase(), parts[0].slice(1)];
  if (!['N', 'S', 'E', 'W'].includes(dir)) {
    return null;
  }
  const [degrees, minutes, seconds] = [degPart, parts[1] || '0', parts[2] || '0'].map((p) =>
    Number.parseFloat(p)
  );
  if (![degrees, minutes, seconds].every(Number.isFinite)) {
    return null;
  }
  if (minutes < 0 || minutes >= 60 || seconds < 0 || seconds >= 60) {
    return null;
  }
  let decimal = degrees + minutes / 60 + seconds / 3600;
  if (dir === 'S' || dir === 'W') {
    decimal *= -1;
  }
  return Number(decimal.toFixed(6));
}

function parseLocationsCsv(csvText) {
  const rows = parseCsv(csvText);
  if (rows.length <= 1) {
    return { records: [], types: [], countries: [] };
  }
  const records = [],
    types = new Set(),
    countries = new Set(),
    seen = new Set();
  for (let i = 1; i < rows.length; i++) {
    const [name, id, type, country, latText, lonText] = rows[i];
    const lat = dmsToDecimal(latText);
    const lon = dmsToDecimal(lonText);
    if (!name || lat === null || lon === null) {
      continue;
    }
    const key = `${name}|${lat}|${lon}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    const entry = { name, id, type: type || 'Unknown', country: country || 'Unknown', lat, lon };
    records.push(entry);
    types.add(entry.type);
    countries.add(entry.country);
  }
  return { records, types: Array.from(types).sort(), countries: Array.from(countries).sort() };
}

async function ensureLocationLibrary() {
  if (locationLibrary) {
    return locationLibrary;
  }
  try {
    updateWaypointLibraryStatus('Loading locations...');
    const response = await fetch(LOCATION_FILE);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    const parsed = parseLocationsCsv(await response.text());
    locationLibrary = parsed;
    updateSelectOptions(locationTypeFilter, parsed.types, 'All types');
    updateSelectOptions(locationCountryFilter, parsed.countries, 'All countries');
    renderLocationList();
    updateWaypointLibraryStatus(
      parsed.records.length ? `Loaded ${parsed.records.length} locations.` : 'No locations found.'
    );
  } catch (err) {
    console.error('Failed to load locations', err);
    updateWaypointLibraryStatus('Could not load saved locations.', true);
    locationLibrary = { records: [], types: [], countries: [] };
  }
  return locationLibrary;
}

function updateSelectOptions(selectEl, entries, allLabel) {
  selectEl.innerHTML = `<option value="all">${allLabel}</option>`;
  for (const value of entries) {
    selectEl.innerHTML += `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`;
  }
}

function updateWaypointLibraryStatus(message, isError = false) {
  if (!waypointLibraryStatus) {
    return;
  }
  waypointLibraryStatus.textContent = message;
  waypointLibraryStatus.style.color = isError ? 'var(--color-danger)' : 'var(--color-text-muted)';
}

const matchesLocationFilters = (loc) => {
  const search = (locationFilters.search || '').toLowerCase();
  return (
    (!search || `${loc.name} ${loc.id}`.toLowerCase().includes(search)) &&
    (locationFilters.type === 'all' || loc.type === locationFilters.type) &&
    (locationFilters.country === 'all' || loc.country === locationFilters.country)
  );
};

function renderLocationList() {
  if (!locationsList || !locationLibrary) {
    return;
  }
  const filtered = locationLibrary.records.filter(matchesLocationFilters);
  if (filtered.length === 0) {
    locationsList.innerHTML = `<div style="padding: 0.5rem; color: #6c757d;">No locations match filters.</div>`;
    return;
  }
  locationsList.innerHTML = filtered
    .map(
      (loc) => `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid #eee; gap: 0.75rem;">
                <div>
                    <div>${escapeHtml(loc.name)}</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">${escapeHtml(loc.id || '—')} &bull; ${escapeHtml(loc.type)}</div>
                </div>
                <button class="btn btn-secondary" data-lat="${loc.lat}" data-lon="${loc.lon}" data-name="${escapeHtml(loc.name)}" data-id="${escapeHtml(loc.id || '')}" data-type="${escapeHtml(loc.type)}">Add</button>
            </div>
        `
    )
    .join('');
}

function appendLocationToWaypoints(name, lat, lon) {
  const line = `${name},${lat.toFixed(6)},${lon.toFixed(6)}`;
  const current = waypointTextarea.value.trim();
  waypointTextarea.value = current ? `${current}\n${line}` : line;
  waypointTextarea.dispatchEvent(new Event('change'));
}

async function toggleWaypointLibraryVisibility() {
  const shouldOpen = waypointLibrary.hidden;
  waypointLibraryToggle.disabled = true;
  try {
    if (shouldOpen) {
      waypointLibrary.hidden = false;
      await ensureLocationLibrary();
    } else {
      waypointLibrary.hidden = true;
    }
  } finally {
    waypointLibraryToggle.disabled = false;
  }
}

function solveAffine(cp) {
  if (!Array.isArray(cp) || cp.length !== 3) {
    throw new Error('Affine map calibration requires exactly three control points.');
  }
  const [lonLat, xT, yT] = [cp.map(([, lat, lon]) => [lon, lat]), cp.map((p) => p[3]), cp.map((p) => p[4])];
  const A = lonLat.map((p) => [...p, 1]);
  const det = (m) =>
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const detA = det(A);
  if (Math.abs(detA) < 1e-9) {
    throw new Error('Control points are colinear.');
  }
  const rep = (c, v) => A.map((r, i) => r.map((val, j) => (j === c ? v[i] : val)));
  const [ax, bx, cx] = [0, 1, 2].map((c) => det(rep(c, xT)) / detA);
  const [ay, by, cy] = [0, 1, 2].map((c) => det(rep(c, yT)) / detA);
  return {
    project(lat, lon) {
      return [ax * lon + bx * lat + cx, ay * lon + by * lat + cy];
    },
  };
}

function buildTfwProjector(mapConfig) {
  const tfw = mapConfig.tfw;
  if (!tfw) {
    throw new Error('TFW parameters are missing.');
  }
  const a = tfw.pixelSizeX;
  const b = tfw.rotationX;
  const d = tfw.rotationY;
  const e = tfw.pixelSizeY;
  const c = tfw.originX;
  const originY = tfw.originY;
  const scaleX = Number.isFinite(tfw.scaleX) ? tfw.scaleX : 1;
  const scaleY = Number.isFinite(tfw.scaleY) ? tfw.scaleY : 1;
  const offsetX = Number.isFinite(tfw.offsetX) ? tfw.offsetX : 0;
  const offsetY = Number.isFinite(tfw.offsetY) ? tfw.offsetY : 0;
  if (![a, b, d, e, c, originY, scaleX, scaleY, offsetX, offsetY].every((val) => Number.isFinite(val))) {
    throw new Error('Invalid TFW parameters.');
  }
  const det = a * e - b * d;
  if (Math.abs(det) < 1e-9) {
    throw new Error('TFW transform is not invertible.');
  }
  return {
    project(lat, lon) {
      const tm = convertLatLonToD96Tm(lat, lon);
      if (!tm) {
        throw new Error('Lat/Lon could not be projected into D96/TM.');
      }
      const dx = tm.easting - c;
      const dy = tm.northing - originY;
      const col = (e * dx - b * dy) / det;
      const row = (-d * dx + a * dy) / det;
      const adjCol = scaleX * col + offsetX;
      const adjRow = scaleY * row + offsetY;
      return [adjCol, adjRow];
    },
  };
}

function buildProjector(mapConfig) {
  const transform = (mapConfig.transform || (mapConfig.tfw ? 'tfw' : 'affine')).toLowerCase();
  if (transform === 'tfw') {
    return buildTfwProjector(mapConfig);
  }
  if (transform !== 'affine') {
    throw new Error(`Unsupported map transform: ${mapConfig.transform}`);
  }
  return solveAffine(mapConfig.controlPoints);
}

function previewToPdf(x, y, pw, ph, bw, bh) {
  return [(x / bw) * pw, ph - (y / bh) * ph];
}
function normalize(dx, dy) {
  const l = Math.hypot(dx, dy);
  return l === 0 ? [0, 0] : [dx / l, dy / l];
}

function rotatedBoundingBox(x, y, width, height, angleRad) {
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  const xs = [];
  const ys = [];
  [
    [0, 0],
    [width, 0],
    [width, height],
    [0, height],
  ].forEach(([cx, cy]) => {
    const px = x + cx * cos - cy * sin;
    const py = y + cx * sin + cy * cos;
    xs.push(px);
    ys.push(py);
  });
  return [
    Math.min(...xs) - LABEL_COLLISION_MARGIN,
    Math.min(...ys) - LABEL_COLLISION_MARGIN,
    Math.max(...xs) + LABEL_COLLISION_MARGIN,
    Math.max(...ys) + LABEL_COLLISION_MARGIN,
  ];
}

function rotatedCorners(x, y, width, height, angleRad) {
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  return [
    [0, 0],
    [width, 0],
    [width, height],
    [0, height],
  ].map(([cx, cy]) => [x + cx * cos - cy * sin, y + cx * sin + cy * cos]);
}

function boxesOverlap(a, b) {
  return !(a[2] <= b[0] || b[2] <= a[0] || a[3] <= b[1] || b[3] <= a[1]);
}

function formatElapsedMinutesLabel(totalMinutes, useHourFormat) {
  if (!Number.isFinite(totalMinutes)) {
    return '';
  }
  const normalized = Math.max(0, totalMinutes);
  const totalSeconds = Math.round(normalized * 60);
  if (normalized >= 60 - 1e-6 || (useHourFormat && normalized >= 60)) {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    return `${hours}:${String(minutes).padStart(2, '0')}`;
  }
  return String(Math.round(normalized)).padStart(2, '0');
}

function formatWaypointTimeLabel(totalMinutes) {
  if (!Number.isFinite(totalMinutes)) {
    return '';
  }
  const normalized = Math.max(0, totalMinutes);
  const totalSeconds = Math.round(normalized * 60);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (normalized >= 60 - 1e-6) {
    const hours = Math.floor(totalSeconds / 3600);
    return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  const wholeMinutes = Math.floor(totalSeconds / 60);
  return `${String(wholeMinutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function adjustLabelPosition(
  x,
  y,
  dirX,
  dirY,
  width,
  height,
  angleRad,
  placedBoxes,
  options: { step?: number; maxSteps?: number; fallbackAngleRad?: number; allowNegative?: boolean } = {}
) {
  const step = options.step ?? LABEL_DISTANCE_STEP;
  const maxSteps = options.maxSteps ?? MAX_LABEL_ADJUST_STEPS;
  const fallbackAngleRad = options.fallbackAngleRad ?? null;
  const allowNegative = options.allowNegative ?? true;

  const length = Math.hypot(dirX, dirY);
  const primaryDir = length === 0 ? [1, 0] : [dirX / length, dirY / length];
  const fallbackDir =
    fallbackAngleRad != null
      ? [Math.cos(fallbackAngleRad), Math.sin(fallbackAngleRad)]
      : [primaryDir[1], -primaryDir[0]];

  const directions = [primaryDir, fallbackDir];

  const testPositions = (direction) => {
    const [dx, dy] = direction;
    for (let i = 0; i <= maxSteps; i += 1) {
      const multipliers = i === 0 ? [0] : allowNegative ? [i, -i] : [i];
      for (const mult of multipliers) {
        const candX = x + dx * mult * step;
        const candY = y + dy * mult * step;
        const bbox = rotatedBoundingBox(candX, candY, width, height, angleRad);
        if (!placedBoxes.some((existing) => boxesOverlap(bbox, existing))) {
          return { x: candX, y: candY, box: bbox };
        }
      }
    }
    return null;
  };

  for (const direction of directions) {
    const result = testPositions(direction);
    if (result) {
      return result;
    }
  }

  return { x, y, box: rotatedBoundingBox(x, y, width, height, angleRad) };
}
async function generate() {
  const originalButtonHtml = generateBtn.innerHTML;
  try {
    generateBtn.disabled = true;
    generateBtn.classList.add('is-loading');
    setStatus('Processing...');
    resultsContent.hidden = true;
    if (resultsPlaceholder && !hasGeneratedOnce) {
      resultsPlaceholder.hidden = false;
    }

    clearDownloadUrl('pdf', downloadPdfLink);
    clearDownloadUrl('overlay', downloadOverlayLink);
    clearDownloadUrl('cropped', downloadCroppedLink);
    clearDownloadUrl('summary', downloadSummaryLink);
    clearDownloadUrl('photoAnalysis', downloadPhotoAnalysisLink);
    clearDownloadUrl('photoKey', downloadPhotoKeyLink);
    clearDownloadUrl('photoHandout', downloadPhotoHandoutLink);
    downloadOverlayLink.style.display = 'none';
    downloadCroppedLink.style.display = 'none';
    downloadSummaryLink.style.display = 'none';
    downloadPhotoAnalysisLink.style.display = 'none';
    downloadPhotoKeyLink.style.display = 'none';
    downloadPhotoHandoutLink.style.display = 'none';
    clearCroppedPreview();
    if (outputsSection) {
      outputsSection.hidden = true;
    }
    if (summarySection) {
      summarySection.hidden = true;
    }
    routeComplianceSection.hidden = true;
    if (osmMapContainer) {
      osmMapContainer.hidden = true;
    }
    syncResultsVisibility();

    const mapConfig = getPreset();
    if (!mapConfig) {
      throw new Error(`Unknown map preset: ${selectedMapKey}`);
    }

    const speed = parseSpeed(speedInput.value);
    const minuteInterval = parseNumericInput(minuteIntervalInput, DEFAULT_MINUTE_INTERVAL);
    const takeoffToSp = parseNumericInput(takeoffBufferInput, DEFAULT_TAKEOFF_BUFFER);
    const points = parseWaypoints(waypointTextarea.value);

    const route = buildRoute(points);
    const compliance = evaluateRouteCompliance(route, points, speed, mapConfig.scaleDenominator);
    const photoCompliance = photoWorkflow.analyze(points);
    const metersPerMinute = speed.metersPerSecond * 60;
    const waypointTimes = computeWaypointTimes(route, points, takeoffToSp, metersPerMinute);
    const legsSummary = route.legs.map((leg) => ({
      id: `${leg.fromName}-${leg.toName}`,
      distanceKm: leg.length / 1000,
      distanceNm: metersToNauticalMiles(leg.length),
      bearingDeg: roundedBearing(bearingDegrees(leg.fromLat, leg.fromLon, leg.toLat, leg.toLon)),
    }));
    const routeDurationMinutes = takeoffToSp + route.totalDistance / metersPerMinute;
    const useHourFormat = routeDurationMinutes >= 60 - 1e-6;
    const minuteMarkersRaw = computeMinuteMarkers(route, takeoffToSp, metersPerMinute, minuteInterval);

    const summaryMinuteMarkersOutput = minuteMarkersRaw.map((m) => ({
      minute: m.minute,
      timeLabel: formatElapsedMinutesLabel(m.minute, useHourFormat),
      leg: `${m.leg.fromName}-${m.leg.toName}`,
      fraction: m.ratio,
    }));
    let summaryCropped = null;
    const mapFilename = mapConfig.type === 'pdf' ? mapConfig.fileName : null;

    if (mapConfig.type === 'pdf') {
      const mapBytes = await ensurePresetBuffer(selectedMapKey);
      if (!mapBytes) {
        throw new Error('Map PDF unavailable.');
      }
      updateMapStatus(`Using ${mapConfig.label}`);

      const projector = buildProjector(mapConfig);
      const pdfDoc = await PDFDocument.load(mapBytes);
      const overlayDoc = await PDFDocument.create();
      pdfDoc.registerFontkit(fontkit);
      overlayDoc.registerFontkit(fontkit);
      const [page] = pdfDoc.getPages();
      const [pageWidth, pageHeight] = [page.getWidth(), page.getHeight()];
      const overlayPage = overlayDoc.addPage([pageWidth, pageHeight]);
      const styleScale = Number.isFinite(mapConfig.styleScale) ? mapConfig.styleScale : 1;
      const scaleAvg =
        ((pageWidth / mapConfig.baseWidth + pageHeight / mapConfig.baseHeight) / 2) * styleScale;
      const projectToPdf = (lat, lon) => {
        const [bx, by] = projector.project(lat, lon);
        return previewToPdf(bx, by, pageWidth, pageHeight, mapConfig.baseWidth, mapConfig.baseHeight);
      };

      const styleConfig = getOverlayStyleConfig();
      const ds = {
        routeLineWidth: Math.max(0.4, styleConfig.routeWidth * scaleAvg),
        tpRadius: TP_RADIUS_SCALE * scaleAvg,
        tpFontSize: Math.max(4, styleConfig.waypointFont * scaleAvg),
        minuteCrossHalf: styleConfig.minuteMarkerHalf * scaleAvg,
        minuteLineWidth: Math.max(0.4, styleConfig.minuteLineWidth * scaleAvg),
        minuteFontSize: Math.max(3, styleConfig.minuteLabelFont * scaleAvg),
        headingFontSize: Math.max(4, styleConfig.headingFont * scaleAvg),
        headingOffset: HEADING_OFFSET_SCALE * scaleAvg,
      };

      const fontBytes = await loadAssetBytes(notoSansBoldUrl, 'the PDF label font');
      const overlayFontBold = await overlayDoc.embedFont(fontBytes, { subset: true });
      const baseFontBold = await pdfDoc.embedFont(fontBytes, { subset: true });
      const drawTargets = [
        { page: overlayPage, fontBold: overlayFontBold },
        { page, fontBold: baseFontBold },
      ];

      const colors = { route: rgb(0.82, 0, 0), heading: rgb(1, 0, 0), minute: rgb(0.05, 0.15, 0.4) };
      const bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
      const expandBounds = (x, y) => {
        bounds.minX = Math.min(bounds.minX, x);
        bounds.minY = Math.min(bounds.minY, y);
        bounds.maxX = Math.max(bounds.maxX, x);
        bounds.maxY = Math.max(bounds.maxY, y);
      };
      const placedLabelBoxes = [];
      const registerLabelBox = (box) => {
        placedLabelBoxes.push(box);
        const [minX, minY, maxX, maxY] = box;
        expandBounds(minX, minY);
        expandBounds(minX, maxY);
        expandBounds(maxX, minY);
        expandBounds(maxX, maxY);
      };

      const projected = points.map(([name, lat, lon]) => ({ name, pdf: projectToPdf(lat, lon) }));
      const outsideMap = projected.filter(
        ({ pdf: [x, y] }) =>
          !Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x > pageWidth || y > pageHeight
      );
      if (outsideMap.length > 0) {
        throw new Error(
          `Route falls outside the calibrated map at: ${outsideMap.map(({ name }) => name).join(', ')}`
        );
      }
      const pointLookup = new Map(projected.map((p) => [p.name, p.pdf]));

      for (let i = 0; i < projected.length - 1; i++) {
        const [x1, y1] = projected[i].pdf;
        const [x2, y2] = projected[i + 1].pdf;
        const [dx, dy] = [x2 - x1, y2 - y1];
        const dist = Math.hypot(dx, dy);
        if (dist === 0) continue;
        const [ux, uy] = normalize(dx, dy);
        const trim = Math.min(ds.tpRadius, Math.max(0, dist / 2 - ds.routeLineWidth));
        const [sx, sy, ex, ey] = [x1 + ux * trim, y1 + uy * trim, x2 - ux * trim, y2 - uy * trim];
        if (Math.hypot(ex - sx, ey - sy) < 0.2) continue;
        expandBounds(sx, sy);
        expandBounds(ex, ey);
        drawTargets.forEach((t) => {
          t.page.drawLine({
            start: { x: sx, y: sy },
            end: { x: ex, y: ey },
            thickness: ds.routeLineWidth,
            color: colors.route,
          });
        });
      }

      for (let i = 0; i < projected.length; i++) {
        const { name, pdf } = projected[i];
        const [x, y] = pdf;
        expandBounds(x - ds.tpRadius, y - ds.tpRadius);
        expandBounds(x + ds.tpRadius, y + ds.tpRadius);
        drawTargets.forEach((t) => {
          t.page.drawCircle({
            x,
            y,
            size: ds.tpRadius,
            borderWidth: ds.routeLineWidth,
            borderColor: colors.route,
          });
        });

        const nextVec =
          i < projected.length - 1 ? [projected[i + 1].pdf[0] - x, projected[i + 1].pdf[1] - y] : [0, 0];
        const prevVec = i > 0 ? [x - projected[i - 1].pdf[0], y - projected[i - 1].pdf[1]] : [0, 0];
        let vOut = nextVec;
        const vIn = prevVec;
        if (Math.hypot(vOut[0], vOut[1]) < 1e-6 && Math.hypot(vIn[0], vIn[1]) < 1e-6) {
          vOut = [1, 0];
        }
        const vOutUnit = normalize(vOut[0], vOut[1]);
        const vInUnit =
          Math.hypot(vIn[0], vIn[1]) < 1e-6 ? [-vOutUnit[0], -vOutUnit[1]] : normalize(vIn[0], vIn[1]);
        const sumVec = [vInUnit[0] + vOutUnit[0], vInUnit[1] + vOutUnit[1]];
        const sumLen = Math.hypot(sumVec[0], sumVec[1]);
        let exterior =
          sumLen > 1e-6 ? [-sumVec[0] / sumLen, -sumVec[1] / sumLen] : normalize(-vOutUnit[1], vOutUnit[0]);
        if (Math.hypot(exterior[0], exterior[1]) < 1e-6) {
          exterior = normalize(vOutUnit[1], -vOutUnit[0]);
        }

        let headingVec = vInUnit;
        if (i === 0 && projected.length > 1) {
          headingVec = normalize(projected[1].pdf[0] - x, projected[1].pdf[1] - y);
        }
        if (Math.hypot(headingVec[0], headingVec[1]) < 1e-6) {
          headingVec = vOutUnit;
        }
        const headingAngleRad = Math.atan2(headingVec[1], headingVec[0]) - Math.PI / 2;
        const headingAngleDeg = (headingAngleRad * 180) / Math.PI;

        let radialDir = normalize(exterior[0], exterior[1]);
        if (Math.hypot(radialDir[0], radialDir[1]) < 1e-6) {
          radialDir = [1, 0];
        }
        let tangentialDir = normalize(-radialDir[1], radialDir[0]);
        if (Math.hypot(tangentialDir[0], tangentialDir[1]) < 1e-6) {
          tangentialDir = [0, 1];
        }

        const labelMargin = Math.max(
          ds.tpFontSize * TP_LABEL_OFFSET_FACTOR,
          ds.routeLineWidth * 2.5,
          ds.minuteCrossHalf * 2.2
        );
        const baseRadius = ds.tpRadius + labelMargin;

        const candidateConfigs = [
          [1.0, 0.0],
          [0.95, 0.25],
          [0.95, -0.25],
          [0.85, 0.45],
          [0.85, -0.45],
        ];

        const pdfName = name;
        const nameWidth = overlayFontBold.widthOfTextAtSize(pdfName, ds.tpFontSize);
        const nameHeight = ds.tpFontSize;

        let bestPlacement = null;
        let bestVec = radialDir;
        for (const [radialWeight, tangentialWeight] of candidateConfigs) {
          let vec = [
            radialDir[0] * radialWeight + tangentialDir[0] * tangentialWeight,
            radialDir[1] * radialWeight + tangentialDir[1] * tangentialWeight,
          ];
          if (Math.hypot(vec[0], vec[1]) < 1e-6) {
            continue;
          }
          vec = normalize(vec[0], vec[1]);
          if (vec[0] * radialDir[0] + vec[1] * radialDir[1] <= 0.25) {
            continue;
          }
          const anchorX = x + vec[0] * baseRadius;
          const anchorY = y + vec[1] * baseRadius;
          const fallbackAngleRad = Math.atan2(vec[1], vec[0]);
          const adjusted = adjustLabelPosition(
            anchorX,
            anchorY,
            vec[0],
            vec[1],
            nameWidth,
            nameHeight,
            headingAngleRad,
            placedLabelBoxes,
            { fallbackAngleRad, allowNegative: false }
          );
          const distance = Math.hypot(adjusted.x - x, adjusted.y - y);
          if (!bestPlacement || distance < bestPlacement.distance) {
            bestPlacement = { distance, position: adjusted };
            bestVec = vec;
          }
        }

        if (!bestPlacement) {
          const adjusted = adjustLabelPosition(
            x + radialDir[0] * baseRadius,
            y + radialDir[1] * baseRadius,
            radialDir[0],
            radialDir[1],
            nameWidth,
            nameHeight,
            headingAngleRad,
            placedLabelBoxes,
            { allowNegative: false }
          );
          bestPlacement = { distance: Math.hypot(adjusted.x - x, adjusted.y - y), position: adjusted };
          bestVec = radialDir;
        }

        if (bestPlacement.distance < baseRadius - 0.5) {
          let direction = [bestPlacement.position.x - x, bestPlacement.position.y - y];
          if (Math.hypot(direction[0], direction[1]) < 1e-6) {
            direction = bestVec;
          }
          direction = normalize(direction[0], direction[1]);
          const newX = x + direction[0] * baseRadius;
          const newY = y + direction[1] * baseRadius;
          const newBox = rotatedBoundingBox(newX, newY, nameWidth, nameHeight, headingAngleRad);
          if (!placedLabelBoxes.some((existing) => boxesOverlap(newBox, existing))) {
            bestPlacement = {
              distance: baseRadius,
              position: { x: newX, y: newY, box: newBox },
            };
          }
        }

        const ensureOutside = () => {
          const corners = rotatedCorners(
            bestPlacement.position.x,
            bestPlacement.position.y,
            nameWidth,
            nameHeight,
            headingAngleRad
          );
          const minCornerDistance = Math.min(...corners.map(([cx, cy]) => Math.hypot(cx - x, cy - y)));
          const requiredDistance = ds.tpRadius + ds.routeLineWidth * 1.5;
          if (minCornerDistance < requiredDistance) {
            let direction = normalize(bestVec[0], bestVec[1]);
            if (Math.hypot(direction[0], direction[1]) < 1e-6) {
              direction = [...radialDir];
            }
            const push = requiredDistance - minCornerDistance + 1.5;
            const newX = bestPlacement.position.x + direction[0] * push;
            const newY = bestPlacement.position.y + direction[1] * push;
            const newBox = rotatedBoundingBox(newX, newY, nameWidth, nameHeight, headingAngleRad);
            bestPlacement = {
              distance: Math.hypot(newX - x, newY - y),
              position: { x: newX, y: newY, box: newBox },
            };
          }
        };
        ensureOutside();

        registerLabelBox(bestPlacement.position.box);
        drawTargets.forEach((t) => {
          t.page.drawText(pdfName, {
            x: bestPlacement.position.x,
            y: bestPlacement.position.y,
            size: ds.tpFontSize,
            font: t.fontBold,
            color: colors.route,
            rotate: degrees(headingAngleDeg),
          });
        });

        const timeMinutes = waypointTimes.get(name);
        const timeLabel = typeof timeMinutes === 'number' ? formatWaypointTimeLabel(timeMinutes) : null;
        if (timeLabel) {
          let timeDir = normalize(-bestVec[1], bestVec[0]);
          if (Math.hypot(timeDir[0], timeDir[1]) < 1e-6) {
            timeDir = [0, 1];
          }
          const timeFontSize = Math.max(4, ds.tpFontSize * 0.75);
          const timeMargin = Math.max(timeFontSize * 0.6, ds.routeLineWidth * 1.4, ds.minuteCrossHalf);
          const timeRadius = baseRadius + timeMargin;
          const timeAnchorX = x + timeDir[0] * timeRadius;
          const timeAnchorY = y + timeDir[1] * timeRadius;
          const fallbackAngleRad = Math.atan2(bestVec[1], bestVec[0]);
          const timeWidth = overlayFontBold.widthOfTextAtSize(timeLabel, timeFontSize);
          const timeHeight = timeFontSize;
          const adjustedTime = adjustLabelPosition(
            timeAnchorX,
            timeAnchorY,
            timeDir[0],
            timeDir[1],
            timeWidth,
            timeHeight,
            headingAngleRad,
            placedLabelBoxes,
            { fallbackAngleRad }
          );
          registerLabelBox(adjustedTime.box);
          drawTargets.forEach((t) => {
            t.page.drawText(timeLabel, {
              x: adjustedTime.x,
              y: adjustedTime.y,
              size: timeFontSize,
              font: t.fontBold,
              color: colors.minute,
              rotate: degrees(headingAngleDeg),
            });
          });
        }
      }

      minuteMarkersRaw.forEach((m) => {
        const legStart = pointLookup.get(m.leg.fromName);
        const legEnd = pointLookup.get(m.leg.toName);
        if (!legStart || !legEnd) {
          return;
        }
        const [dx, dy] = [legEnd[0] - legStart[0], legEnd[1] - legStart[1]];
        const segLength = Math.hypot(dx, dy);
        if (segLength === 0) {
          return;
        }
        const [ux, uy] = [dx / segLength, dy / segLength];
        const [pxVec, pyVec] = [-uy, ux];
        const [xp, yp] = projectToPdf(m.lat, m.lon);
        if (
          Math.hypot(xp - legStart[0], yp - legStart[1]) < ds.tpRadius ||
          Math.hypot(xp - legEnd[0], yp - legEnd[1]) < ds.tpRadius
        ) {
          return;
        }
        const markerOffset = ds.minuteCrossHalf;
        const [x1, y1, x2, y2] = [
          xp - pxVec * markerOffset,
          yp - pyVec * markerOffset,
          xp + pxVec * markerOffset,
          yp + pyVec * markerOffset,
        ];
        expandBounds(x1, y1);
        expandBounds(x2, y2);
        drawTargets.forEach((t) => {
          t.page.drawLine({
            start: { x: x1, y: y1 },
            end: { x: x2, y: y2 },
            thickness: ds.minuteLineWidth,
            color: colors.minute,
          });
        });

        const minuteLabel = formatElapsedMinutesLabel(m.minute, useHourFormat);
        const labelOffset = markerOffset * MINUTE_LABEL_OFFSET_MULTIPLIER;
        const baseX = xp + pxVec * labelOffset;
        const baseY = yp + pyVec * labelOffset;
        const angleRad = Math.atan2(dy, dx);
        const angleDeg = (angleRad * 180) / Math.PI - 90;
        const textAngleRad = (angleDeg * Math.PI) / 180;
        const textWidth = overlayFontBold.widthOfTextAtSize(minuteLabel, ds.minuteFontSize);
        const textHeight = ds.minuteFontSize;
        const fallbackAngleRad = Math.atan2(pyVec, pxVec) + Math.PI / 2;
        const adjusted = adjustLabelPosition(
          baseX,
          baseY,
          pxVec,
          pyVec,
          textWidth,
          textHeight,
          textAngleRad,
          placedLabelBoxes,
          { fallbackAngleRad }
        );
        registerLabelBox(adjusted.box);

        drawTargets.forEach((t) => {
          t.page.drawText(minuteLabel, {
            x: adjusted.x,
            y: adjusted.y,
            size: ds.minuteFontSize,
            font: t.fontBold,
            color: colors.minute,
            rotate: degrees(angleDeg),
          });
        });
      });

      const photoLayerOptions = photoWorkflow.layerOptions;
      const photoColors = {
        enroute: rgb(0.45, 0.12, 0.66),
        'control-correct': rgb(0.05, 0.48, 0.25),
        'control-false': rgb(0.92, 0.42, 0.04),
        'sign-task': rgb(0.05, 0.35, 0.72),
        reference: rgb(0.35, 0.38, 0.4),
      };
      const photoMarkerSize = Math.max(5, 7 * scaleAvg);
      const drawPhotoSymbol = (target, x, y, classification, color, warning) => {
        const borderColor = warning ? rgb(0.8, 0.02, 0.02) : color;
        if (classification === 'enroute') {
          target.page.drawLine({
            start: { x: x - photoMarkerSize, y },
            end: { x: x + photoMarkerSize, y },
            thickness: 2,
            color: borderColor,
          });
          target.page.drawLine({
            start: { x, y: y - photoMarkerSize },
            end: { x, y: y + photoMarkerSize },
            thickness: 2,
            color: borderColor,
          });
        } else if (classification === 'control-correct') {
          target.page.drawRectangle({
            x: x - photoMarkerSize,
            y: y - photoMarkerSize,
            width: photoMarkerSize * 2,
            height: photoMarkerSize * 2,
            borderWidth: warning ? 2.5 : 1.8,
            borderColor,
            color: rgb(1, 1, 1),
            opacity: 0.88,
          });
        } else if (classification === 'control-false') {
          const vertices = [
            [x, y + photoMarkerSize],
            [x + photoMarkerSize, y],
            [x, y - photoMarkerSize],
            [x - photoMarkerSize, y],
          ];
          vertices.forEach((start, index) => {
            target.page.drawLine({
              start: { x: start[0], y: start[1] },
              end: { x: vertices[(index + 1) % 4][0], y: vertices[(index + 1) % 4][1] },
              thickness: warning ? 2.5 : 1.8,
              color: borderColor,
            });
          });
        } else if (classification === 'sign-task') {
          const vertices = [
            [x, y + photoMarkerSize],
            [x + photoMarkerSize, y - photoMarkerSize],
            [x - photoMarkerSize, y - photoMarkerSize],
          ];
          vertices.forEach((start, index) => {
            target.page.drawLine({
              start: { x: start[0], y: start[1] },
              end: { x: vertices[(index + 1) % 3][0], y: vertices[(index + 1) % 3][1] },
              thickness: warning ? 2.5 : 1.8,
              color: borderColor,
            });
          });
        } else {
          target.page.drawCircle({
            x,
            y,
            size: photoMarkerSize,
            borderWidth: warning ? 2.5 : 1.8,
            borderColor,
          });
        }
      };
      for (const photo of photoWorkflow.records) {
        const latitude = photo.metadata.latitude.value;
        const longitude = photo.metadata.longitude.value;
        if (latitude === null || longitude === null || !photo.analysis) continue;
        const exact = projectToPdf(latitude, longitude);
        const projectedPhoto = projectToPdf(photo.analysis.closestLatitude, photo.analysis.closestLongitude);
        if (![...exact, ...projectedPhoto].every(Number.isFinite)) continue;
        const color = photoColors[photo.classification];
        const hasViolation = photo.findings.some((finding) => finding.severity === 'violation');
        if (
          photoLayerOptions.connectors &&
          Math.hypot(exact[0] - projectedPhoto[0], exact[1] - projectedPhoto[1]) > 1
        ) {
          drawTargets.forEach((target) => {
            target.page.drawLine({
              start: { x: exact[0], y: exact[1] },
              end: { x: projectedPhoto[0], y: projectedPhoto[1] },
              thickness: Math.max(0.6, scaleAvg),
              color,
              opacity: 0.65,
              dashArray: [3, 3],
            });
          });
        }
        if (photoLayerOptions.exactDots) {
          drawTargets.forEach((target) => {
            target.page.drawCircle({
              x: exact[0],
              y: exact[1],
              size: Math.max(2.3, 3 * scaleAvg),
              color,
              borderColor: hasViolation ? rgb(0.8, 0.02, 0.02) : rgb(1, 1, 1),
              borderWidth: 0.8,
            });
          });
        }
        if (photoLayerOptions.projectedMarkers) {
          drawTargets.forEach((target) => {
            drawPhotoSymbol(
              target,
              projectedPhoto[0],
              projectedPhoto[1],
              photo.classification,
              color,
              hasViolation
            );
          });
          const label = photo.identifier || '?';
          const fontSize = Math.max(5, 9 * scaleAvg);
          const width = overlayFontBold.widthOfTextAtSize(label, fontSize);
          const adjusted = adjustLabelPosition(
            projectedPhoto[0] + photoMarkerSize * 1.5,
            projectedPhoto[1] + photoMarkerSize,
            1,
            1,
            width,
            fontSize,
            0,
            placedLabelBoxes,
            { allowNegative: false }
          );
          registerLabelBox(adjusted.box);
          drawTargets.forEach((target) => {
            target.page.drawText(label, {
              x: adjusted.x,
              y: adjusted.y,
              size: fontSize,
              font: target.fontBold,
              color,
            });
          });
        }
        if (photoLayerOptions.headingArrows && photo.metadata.headingDeg.value !== null) {
          const radians = (photo.metadata.headingDeg.value * Math.PI) / 180;
          const distanceM = 300;
          const headingLat = latitude + ((distanceM * Math.cos(radians)) / 6371008.8) * (180 / Math.PI);
          const headingLon =
            longitude +
            ((distanceM * Math.sin(radians)) / (6371008.8 * Math.cos((latitude * Math.PI) / 180))) *
              (180 / Math.PI);
          const endpoint = projectToPdf(headingLat, headingLon);
          drawTargets.forEach((target) => {
            target.page.drawLine({
              start: { x: exact[0], y: exact[1] },
              end: { x: endpoint[0], y: endpoint[1] },
              thickness: Math.max(0.8, 1.2 * scaleAvg),
              color,
            });
          });
        }
        if (photoLayerOptions.includeInCrop) {
          [exact, projectedPhoto].forEach(([x, y]) => {
            expandBounds(x - photoMarkerSize * 2, y - photoMarkerSize * 2);
            expandBounds(x + photoMarkerSize * 2, y + photoMarkerSize * 2);
          });
        }
      }
      if (photoLayerOptions.legend && photoWorkflow.records.length > 0) {
        const legendItems = [
          ['enroute', 'En-route'],
          ['control-correct', 'Correct CP'],
          ['control-false', 'False CP'],
          ['sign-task', 'Sign task'],
          ['reference', 'Reference'],
        ];
        const lineHeight = 12;
        const legendWidth = 105;
        const legendHeight = legendItems.length * lineHeight + 14;
        const legendLeft = Math.min(Math.max(0, bounds.minX), Math.max(0, pageWidth - legendWidth));
        const belowRoute = bounds.minY - legendHeight - 6;
        const legendBottom =
          belowRoute >= 0
            ? belowRoute
            : Math.min(Math.max(0, pageHeight - legendHeight), Math.max(0, bounds.maxY + 6));
        const legendX = legendLeft + 8;
        const legendY = legendBottom + 7;
        drawTargets.forEach((target) => {
          target.page.drawRectangle({
            x: legendLeft,
            y: legendBottom,
            width: legendWidth,
            height: legendHeight,
            color: rgb(1, 1, 1),
            opacity: 0.88,
            borderColor: rgb(0.55, 0.58, 0.6),
            borderWidth: 0.6,
          });
          legendItems.forEach(([classification, label], index) => {
            const y = legendY + (legendItems.length - 1 - index) * lineHeight;
            drawPhotoSymbol(target, legendX, y + 2, classification, photoColors[classification], false);
            target.page.drawText(label, {
              x: legendX + 12,
              y,
              size: 7,
              font: target.fontBold,
              color: photoColors[classification],
            });
          });
        });
        if (photoLayerOptions.includeInCrop) {
          expandBounds(legendLeft, legendBottom);
          expandBounds(legendLeft + legendWidth, legendBottom + legendHeight);
        }
      }

      for (let i = 0; i < route.legs.length; i++) {
        const start = projected[i].pdf;
        const end = projected[i + 1].pdf;
        const [dx, dy] = [end[0] - start[0], end[1] - start[1]];
        const segLength = Math.hypot(dx, dy);
        if (segLength === 0) {
          continue;
        }
        const [ux, uy] = [dx / segLength, dy / segLength];
        const [pxVec, pyVec] = [-uy, ux];
        const midX = (start[0] + end[0]) / 2;
        const midY = (start[1] + end[1]) / 2;
        const anchorX = midX + pxVec * ds.headingOffset;
        const anchorY = midY + pyVec * ds.headingOffset;
        const text = `${legsSummary[i].bearingDeg}°`.padStart(4, '0');
        const angleRad = Math.atan2(dy, dx);
        const angleDeg = (angleRad * 180) / Math.PI - 90;
        const textAngleRad = (angleDeg * Math.PI) / 180;
        const textWidth = overlayFontBold.widthOfTextAtSize(text, ds.headingFontSize);
        const textHeight = ds.headingFontSize;
        const fallbackAngleRad = Math.atan2(pyVec, pxVec) + Math.PI / 2;
        const adjusted = adjustLabelPosition(
          anchorX,
          anchorY,
          pxVec,
          pyVec,
          textWidth,
          textHeight,
          textAngleRad,
          placedLabelBoxes,
          { fallbackAngleRad }
        );
        registerLabelBox(adjusted.box);

        drawTargets.forEach((t) => {
          t.page.drawText(text, {
            x: adjusted.x,
            y: adjusted.y,
            size: ds.headingFontSize,
            font: t.fontBold,
            color: colors.heading,
            rotate: degrees(angleDeg),
          });
        });
      }

      const overlayOnlyBytes = await overlayDoc.save();
      const markedBytes = await pdfDoc.save();
      let croppedBytes = null;
      let previewOptions = null;
      if (Object.values(bounds).every(Number.isFinite)) {
        const m = 10 * MM_TO_PT;
        const [minX, minY, maxX, maxY] = [
          Math.max(0, bounds.minX - m),
          Math.max(0, bounds.minY - m),
          Math.min(pageWidth, bounds.maxX + m),
          Math.min(pageHeight, bounds.maxY + m),
        ];
        if (maxX > minX + 1 && maxY > minY + 1) {
          const contentWidth = maxX - minX;
          const contentHeight = maxY - minY;
          const targetPage = chooseTrueScaleCropPage(contentWidth, contentHeight, [
            210 * MM_TO_PT,
            297 * MM_TO_PT,
          ]);
          const cropDoc = await PDFDocument.create();
          const [embedded] = await cropDoc.embedPdf(markedBytes, [0]);
          const cropPage = cropDoc.addPage([targetPage.width, targetPage.height]);
          cropPage.drawPage(embedded, {
            x: -minX + (targetPage.width - contentWidth) / 2,
            y: -minY + (targetPage.height - contentHeight) / 2,
          });
          croppedBytes = await cropDoc.save();
          const previewPhotoColors = {
            enroute: '#7331a5',
            'control-correct': '#0d7a40',
            'control-false': '#eb6b0a',
            'sign-task': '#0d59b8',
            reference: '#596166',
          };
          previewOptions = {
            imageUrl: mapConfig.previewUrl,
            pageWidth,
            pageHeight,
            crop: { minX, minY, maxX, maxY },
            route: projected.map(({ name, pdf: [x, y] }) => ({ x, y, label: name })),
            photos: photoWorkflow.records.flatMap((photo) => {
              if (!photo.analysis) return [];
              const [x, y] = projectToPdf(photo.analysis.closestLatitude, photo.analysis.closestLongitude);
              if (![x, y].every(Number.isFinite)) return [];
              return [
                {
                  x,
                  y,
                  label: photo.identifier || '?',
                  color: previewPhotoColors[photo.classification],
                  warning: photo.findings.some((finding) => finding.severity === 'violation'),
                },
              ];
            }),
          };
          summaryCropped = {
            available: true,
            widthMm: targetPage.width / MM_TO_PT,
            heightMm: targetPage.height / MM_TO_PT,
            format: targetPage.format,
            scale: '100%',
          };
        }
      }
      setDownloadUrl('pdf', createPdfObjectUrl(markedBytes), 'route_marked.pdf', downloadPdfLink);
      setDownloadUrl(
        'overlay',
        createPdfObjectUrl(overlayOnlyBytes),
        'route_overlay.pdf',
        downloadOverlayLink
      );
      downloadOverlayLink.style.display = 'inline-flex';
      if (croppedBytes && previewOptions) {
        setDownloadUrl('cropped', createPdfObjectUrl(croppedBytes), 'route_cropped.pdf', downloadCroppedLink);
        downloadCroppedLink.style.display = 'inline-flex';
        void renderCroppedPreview(previewOptions);
      }
    } else {
      if (osmMapContainer) {
        osmMapContainer.hidden = false;
      }
      await ensureOsmMap();
      if (!osmMap) {
        throw new Error('Interactive map unavailable.');
      }
      osmLayers.forEach((layer) => {
        layer.remove();
      });
      osmLayers = [];
      const latLngs: L.LatLngExpression[] = points.map(([, lat, lon]) => [lat, lon] as [number, number]);
      if (latLngs.length === 0) {
        throw new Error('No waypoints.');
      }
      const routeLayer = L.polyline(latLngs, { color: '#D21F26', weight: 3.5, opacity: 0.85 }).addTo(osmMap);
      osmLayers.push(routeLayer);
      const routeBounds = routeLayer.getBounds();
      points.forEach(([name, lat, lon]) => {
        osmLayers.push(
          L.circleMarker([lat, lon], {
            radius: 6,
            color: '#D21F26',
            fillColor: '#FFFFFF',
            fillOpacity: 0.9,
            weight: 2,
          }).addTo(osmMap)
        );
        const timeMinutes = waypointTimes.get(name);
        const timeLabel = typeof timeMinutes === 'number' ? formatWaypointTimeLabel(timeMinutes) : '';
        const labelHtml = `<div class="osm-label"><span>${escapeHtml(name)}</span>${timeLabel ? `<span class="osm-label-time">${escapeHtml(timeLabel)}</span>` : ''}</div>`;
        osmLayers.push(
          L.marker([lat, lon], {
            icon: L.divIcon({ className: 'leaflet-marker-icon osm-label-icon', html: labelHtml }),
            interactive: false,
          }).addTo(osmMap)
        );
      });
      minuteMarkersRaw.forEach((m) => {
        const label = formatElapsedMinutesLabel(m.minute, useHourFormat);
        osmLayers.push(
          L.circleMarker([m.lat, m.lon], {
            radius: 4,
            color: '#153E73',
            fillColor: '#153E73',
            fillOpacity: 0.85,
            weight: 1,
          }).addTo(osmMap)
        );
        osmLayers.push(
          L.marker([m.lat, m.lon], {
            icon: L.divIcon({
              className: 'leaflet-marker-icon osm-minute-label-icon',
              html: `<div class="osm-minute-label">${escapeHtml(label)}</div>`,
            }),
            interactive: false,
          }).addTo(osmMap)
        );
      });
      const photoLayerOptions = photoWorkflow.layerOptions;
      const photoColors = {
        enroute: '#731fa8',
        'control-correct': '#087a40',
        'control-false': '#eb6b0b',
        'sign-task': '#0d59b8',
        reference: '#5b6166',
      };
      for (const photo of photoWorkflow.records) {
        const latitude = photo.metadata.latitude.value;
        const longitude = photo.metadata.longitude.value;
        if (latitude === null || longitude === null || !photo.analysis) continue;
        const exact: [number, number] = [latitude, longitude];
        const projectedPhoto: [number, number] = [
          photo.analysis.closestLatitude,
          photo.analysis.closestLongitude,
        ];
        const color = photoColors[photo.classification];
        const hasViolation = photo.findings.some((finding) => finding.severity === 'violation');
        if (photoLayerOptions.connectors) {
          osmLayers.push(
            L.polyline([exact, projectedPhoto], { color, weight: 1.5, opacity: 0.7, dashArray: '4 4' }).addTo(
              osmMap
            )
          );
        }
        if (photoLayerOptions.exactDots) {
          osmLayers.push(
            L.circleMarker(exact, {
              radius: 4,
              color: hasViolation ? '#c90000' : '#fff',
              fillColor: color,
              fillOpacity: 1,
              weight: 2,
            })
              .bindTooltip(`${photo.identifier}: exact photo position`)
              .addTo(osmMap)
          );
        }
        if (photoLayerOptions.projectedMarkers) {
          osmLayers.push(
            L.circleMarker(projectedPhoto, {
              radius: photo.classification === 'enroute' ? 7 : 9,
              color: hasViolation ? '#c90000' : color,
              fillColor: '#fff',
              fillOpacity: 0.9,
              weight: hasViolation ? 3 : 2,
            })
              .bindTooltip(`${photo.identifier} · ${photo.classification}`)
              .addTo(osmMap)
          );
          osmLayers.push(
            L.marker(projectedPhoto, {
              icon: L.divIcon({
                className: 'leaflet-marker-icon osm-photo-label-icon',
                html: `<div class="osm-photo-label" style="border-color:${color};color:${color}">${escapeHtml(photo.identifier || '?')}</div>`,
              }),
              interactive: false,
            }).addTo(osmMap)
          );
        }
        if (photoLayerOptions.headingArrows && photo.metadata.headingDeg.value !== null) {
          const radians = (photo.metadata.headingDeg.value * Math.PI) / 180;
          const headingLat = latitude + ((300 * Math.cos(radians)) / 6371008.8) * (180 / Math.PI);
          const headingLon =
            longitude +
            ((300 * Math.sin(radians)) / (6371008.8 * Math.cos((latitude * Math.PI) / 180))) *
              (180 / Math.PI);
          osmLayers.push(
            L.polyline([exact, [headingLat, headingLon]], { color, weight: 2, opacity: 0.9 }).addTo(osmMap)
          );
        }
        routeBounds.extend(exact);
      }
      const refreshOsmViewport = () => {
        if (!osmMap) {
          return;
        }
        if (latLngs.length === 1) {
          osmMap.setView(latLngs[0], 11);
          return;
        }
        if (!routeBounds || typeof routeBounds.isValid !== 'function' || routeBounds.isValid()) {
          osmMap.fitBounds(routeBounds, { padding: [20, 20] });
        } else if (latLngs.length > 0) {
          osmMap.setView(latLngs[0], 11);
        }
      };
      refreshOsmViewport();
      const scheduleViewportRefresh = (fn) => {
        if (typeof requestAnimationFrame === 'function') {
          requestAnimationFrame(fn);
        } else {
          setTimeout(fn, 16);
        }
      };
      scheduleViewportRefresh(() => {
        if (!osmMap) {
          return;
        }
        osmMap.invalidateSize();
        refreshOsmViewport();
      });
      downloadPdfLink.textContent = 'Print Map to PDF';
      downloadPdfLink.removeAttribute('download');
      downloadPdfLink.href = '#';
      downloadPdfLink.onclick = (e) => {
        e.preventDefault();
        window.print();
      };
    }

    const analysisCsv = photoAnalysisCsv(photoWorkflow.records);
    const overlayKeyCsv = photoOverlayKeyCsv(photoWorkflow.records);
    setDownloadUrl(
      'photoAnalysis',
      URL.createObjectURL(new Blob([analysisCsv], { type: 'text/csv;charset=utf-8' })),
      'photo_analysis.csv',
      downloadPhotoAnalysisLink
    );
    setDownloadUrl(
      'photoKey',
      URL.createObjectURL(new Blob([overlayKeyCsv], { type: 'text/csv;charset=utf-8' })),
      'photo_overlay_key.csv',
      downloadPhotoKeyLink
    );
    downloadPhotoAnalysisLink.style.display = 'inline-flex';
    downloadPhotoKeyLink.style.display = 'inline-flex';

    setStatus(
      `Preparing ${photoWorkflow.records.length} photo${photoWorkflow.records.length === 1 ? '' : 's'} for the handout...`
    );
    const handoutPhotos = [];
    for (const record of photoWorkflow.records) {
      handoutPhotos.push({
        record,
        jpeg: await preparePhotoJpeg(record.file, record.metadata.orientation.value ?? 1),
      });
    }
    const handoutFontBytes = await loadAssetBytes(notoSansBoldUrl, 'the photo handout font');
    const handoutBytes = await buildPhotoHandout(
      handoutPhotos,
      photoCompliance,
      handoutFontBytes,
      photoWorkflow.handoutOptions
    );
    setDownloadUrl(
      'photoHandout',
      createPdfObjectUrl(handoutBytes),
      'photo_handout.pdf',
      downloadPhotoHandoutLink
    );
    downloadPhotoHandoutLink.style.display = 'inline-flex';

    const summary = {
      schemaVersion: 3,
      appVersion: APP_VERSION,
      generatedAt: new Date().toISOString(),
      speedLabel: speed.label,
      speedKnots: speed.knots,
      totalDistanceKm: route.totalDistance / 1000,
      totalDistanceNm: compliance.totalDistanceNm,
      map: {
        key: selectedMapKey,
        label: mapConfig.label,
        edition: mapConfig.edition,
        file: mapFilename || null,
        transform: mapConfig.type === 'pdf' ? mapConfig.transform : null,
        validityReviewRequired: mapConfig.requiresValidityReview,
        scaleDenominator: mapConfig.scaleDenominator,
      },
      waypoints: points.map(([name, latitude, longitude]) => ({ name, latitude, longitude })),
      legs: legsSummary,
      waypointTimes: Object.fromEntries(
        Array.from(waypointTimes.entries()).map(([name, minutes]) => [name, formatWaypointTimeLabel(minutes)])
      ),
      minuteMarkers: summaryMinuteMarkersOutput,
      minuteInterval,
      takeoffToSp,
      totalMinutes: routeDurationMinutes,
      cropped: summaryCropped,
      compliance: {
        rulebook: 'Pravilnik Rally Letenja za Prvenstvo Slovenije, corrected August 2023',
        automatedStatus: compliance.status,
        maximumControlPoints: compliance.maximumControlPoints,
        checks: compliance.checks,
        manualChecks: compliance.manualChecks,
      },
      photos: photoSummaryJson(photoWorkflow.records, photoCompliance),
      warnings: mapConfig.requiresValidityReview
        ? [`Confirm that ${mapConfig.label} (${mapConfig.edition} edition) is current and event-approved.`]
        : [],
    };

    setDownloadUrl(
      'summary',
      URL.createObjectURL(new Blob([JSON.stringify(summary, null, 2)], { type: 'application/json' })),
      'route_summary.json',
      downloadSummaryLink
    );
    downloadSummaryLink.style.display = 'inline-flex';

    if (outputsSection) {
      outputsSection.hidden = false;
    }

    replaceTableRows(
      legsTable,
      legsSummary.map((leg) => [
        leg.id,
        leg.distanceNm.toFixed(2),
        leg.distanceKm.toFixed(2),
        leg.bearingDeg.toString().padStart(3, '0'),
      ])
    );
    replaceTableRows(
      waypointTable,
      Array.from(waypointTimes.entries()).map(([name, minutes]) => [name, formatWaypointTimeLabel(minutes)])
    );
    summaryText.textContent = `${summary.speedLabel} - ${summary.totalDistanceNm.toFixed(2)} NM (${summary.totalDistanceKm.toFixed(2)} km) course - ${mapConfig.label}`;

    renderCompliance(compliance);

    if (summarySection) {
      summarySection.hidden = false;
    }
    if (osmMapContainer) {
      osmMapContainer.hidden = mapConfig.type === 'pdf';
    }

    if (resultsPlaceholder) {
      if (!hasGeneratedOnce) {
        resultsPlaceholder.remove();
        hasGeneratedOnce = true;
      } else {
        resultsPlaceholder.hidden = true;
      }
    }
    resultsContent.hidden = false;
    syncResultsVisibility();
    const generatedStatus =
      compliance.status !== 'ok' || photoCompliance.status === 'against-rules'
        ? 'against-rules'
        : photoCompliance.status === 'manual-review'
          ? 'manual-review'
          : 'ok';
    setStatus(
      generatedStatus === 'against-rules'
        ? `Generated: Against the rules · ${compliance.violations.length + photoCompliance.violationCount} automated violation(s). Review the affected route and photo findings.`
        : generatedStatus === 'manual-review'
          ? `Generated: Manual review required · automated checks passed, but ${photoCompliance.warningCount} photo item(s) lack reliable metadata.`
          : 'Generated: OK for automated checks. Complete the listed manual judge checks.',
      generatedStatus === 'ok' ? 'success' : 'warning'
    );
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    resultsContent.hidden = true;
    if (resultsPlaceholder && !hasGeneratedOnce) {
      resultsPlaceholder.hidden = false;
    }
    syncResultsVisibility();
  } finally {
    generateBtn.disabled = false;
    generateBtn.classList.remove('is-loading');
    generateBtn.innerHTML = originalButtonHtml;
  }
}

generateBtn.addEventListener('click', generate);
waypointLibraryToggle.addEventListener('click', toggleWaypointLibraryVisibility);

if (libraryControls) {
  libraryControls.addEventListener('input', (e) => {
    const target = e.target as HTMLInputElement | HTMLSelectElement;
    if (target.id === 'locationSearch') {
      locationFilters.search = target.value.trim();
    }
    if (target.id === 'locationTypeFilter') {
      locationFilters.type = target.value;
    }
    if (target.id === 'locationCountryFilter') {
      locationFilters.country = target.value;
    }
    renderLocationList();
  });
}

addAllFilteredBtn.addEventListener('click', () => {
  locationLibrary?.records.filter(matchesLocationFilters).forEach((location) => {
    appendLocationToWaypoints(location.name, location.lat, location.lon);
  });
});

locationsList.addEventListener('click', (e) => {
  const target = e.target as HTMLElement;
  if (target.tagName === 'BUTTON' && target.dataset.name) {
    const { name, lat, lon, id, type } = target.dataset;
    const typeKey = (type || '').toLowerCase();
    const useShort = !!id && (typeKey.includes('aerodrome') || typeKey.includes('heliport'));
    const waypointName = useShort ? id.toUpperCase() : name;
    appendLocationToWaypoints(waypointName, parseFloat(lat), parseFloat(lon));
  }
});

mapPresetButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    handleMapPresetChange(btn.dataset.mapKey);
  });
});

window.addEventListener('beforeunload', () => {
  Object.values(downloadUrls).forEach((url) => {
    if (url) {
      URL.revokeObjectURL(url);
    }
  });
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
  }
});

// Initial UI state setup
setStatus('');
handleMapPresetChange(selectedMapKey);
