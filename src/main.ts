import { strToU8, Zip, ZipPassThrough } from 'fflate';
import type { LatLngExpression } from 'leaflet';
import notoSansBoldUrl from 'notosans-fontface/fonts/NotoSans-Bold.ttf?url';
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
  type Waypoint,
} from './domain';
import rawMapPresets from './map-presets.json';
import { renderBoundedMapPreview, waitForAbortSignal } from './map-preview';
import { loadMapPresets } from './maps';
import {
  type OrthophotoCaptureModel,
  type OrthophotoTarget,
  orthophotoCoverage,
  orthophotoRequestUrl,
} from './orthophoto';
import {
  type ControlPhotoProposal,
  createSeededRandom,
  discoverOsmPhotoCandidates,
  fetchOsmControlPhotoProposals,
  fetchOverpassPhotoData,
  type OsmDiscoveryProgress,
  type OsmFailedRequest,
  type OsmPhotoCandidate,
  type OverpassResponse,
  retryOverpassPhotoRequests,
  selectOsmPhotoCandidates,
  targetSelectionIssue,
} from './osm-photo-candidates';
import {
  evaluatePhotoCompliance,
  findingPresentation,
  isCountedRouteTask,
  isPhotoAcceptedForJudge,
} from './photo-compliance';
import { preparePhotoJpeg } from './photo-image';
import { photoAnalysisCsv, photoOverlayKeyCsv, photoSummaryJson } from './photo-output';
import { PHOTO_IMPORT_LIMITS, PhotoWorkflow } from './photo-workflow';
import {
  DEFAULT_CUSTOM_SPEEDS,
  PROJECT_FILE_FORMAT,
  PROJECT_FILE_SCHEMA_VERSION,
  parseSavedRouteProject,
  type SavedCustomSpeed,
  type SavedProjectSettings,
  type SavedRouteProject,
  SPEED_EDITION_KNOTS,
  STANDARD_SPEED_PRESETS,
} from './project-file';
import { GuidedWorkflow, type WorkflowStage } from './workflow';
import './styles.css';

const APP_BASE_URL = new URL('./', document.baseURI);
const assetUrl = (path) => new URL(path, APP_BASE_URL).href;

const MAP_PRESETS = loadMapPresets(rawMapPresets, APP_BASE_URL);
const APP_VERSION = '2.4.0';
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
const PREVIEW_STEP_TIMEOUT_MS = 30000;
const LABEL_COLLISION_MARGIN = 3.0;
const LABEL_DISTANCE_STEP = 5.0;
const MAX_LABEL_ADJUST_STEPS = 12;

function requiredElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Required page element is missing: #${id}`);
  return element as T;
}

const statusEl = requiredElement<HTMLElement>('status');
const saveProjectBtn = requiredElement<HTMLButtonElement>('saveProject');
const loadProjectBtn = requiredElement<HTMLButtonElement>('loadProject');
const projectFileInput = requiredElement<HTMLInputElement>('projectFile');
const generateBtn = requiredElement<HTMLButtonElement>('generate');
const cancelGenerationBtn = requiredElement<HTMLButtonElement>('cancelGeneration');
const workflowShell = requiredElement<HTMLElement>('workflowShell');
const artifactStatuses = requiredElement<HTMLElement>('artifactStatuses');
const mapPresetGrid = requiredElement<HTMLElement>('mapPresetGrid');
const osmConsentRow = requiredElement<HTMLElement>('osmConsentRow');
const osmThirdPartyConsent = requiredElement<HTMLInputElement>('osmThirdPartyConsent');
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
const speedInput = requiredElement<HTMLSelectElement>('speed');
const customSpeedPanel = document.querySelector<HTMLDetailsElement>('.custom-speed-panel');
const customSpeedValueInputs = Array.from(
  document.querySelectorAll<HTMLInputElement>('[data-custom-speed-value]')
);
const customSpeedUnitInputs = Array.from(
  document.querySelectorAll<HTMLSelectElement>('[data-custom-speed-unit]')
);
if (customSpeedValueInputs.length !== 4 || customSpeedUnitInputs.length !== 4) {
  throw new Error('Exactly four custom speed controls are required.');
}
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
const orthophotoRandomCount = requiredElement<HTMLInputElement>('orthophotoRandomCount');
const orthophotoAltitude = requiredElement<HTMLInputElement>('orthophotoAltitude');
const orthophotoFocalLength = requiredElement<HTMLInputElement>('orthophotoFocalLength');
const orthophotoDepression = requiredElement<HTMLInputElement>('orthophotoDepression');
const orthophotoCoveragePreview = requiredElement<HTMLElement>('orthophotoCoveragePreview');
const handoutSplit = requiredElement<HTMLSelectElement>('handoutSplit');
const addRandomOrthophotos = requiredElement<HTMLButtonElement>('addRandomOrthophotos');
const loadWaypointOrthophotos = requiredElement<HTMLButtonElement>('loadWaypointOrthophotos');
const osmCandidateReview = requiredElement<HTMLElement>('osmCandidateReview');
const osmCandidateSummary = requiredElement<HTMLElement>('osmCandidateSummary');
const osmCandidateList = requiredElement<HTMLElement>('osmCandidateList');
const osmCandidateSelectionStatus = requiredElement<HTMLElement>('osmCandidateSelectionStatus');
const refreshOsmCandidates = requiredElement<HTMLButtonElement>('refreshOsmCandidates');
const selectAllOsmCandidates = requiredElement<HTMLButtonElement>('selectAllOsmCandidates');
const clearOsmCandidateSelection = requiredElement<HTMLButtonElement>('clearOsmCandidateSelection');
const importOsmCandidates = requiredElement<HTMLButtonElement>('importOsmCandidates');
const osmDiscoveryProgress = requiredElement<HTMLElement>('osmDiscoveryProgress');
const osmDiscoveryProgressText = requiredElement<HTMLElement>('osmDiscoveryProgressText');
const osmDiscoveryProgressPercent = requiredElement<HTMLElement>('osmDiscoveryProgressPercent');
const osmDiscoveryProgressBar = requiredElement<HTMLProgressElement>('osmDiscoveryProgressBar');
const osmDiscoveryLegs = requiredElement<HTMLElement>('osmDiscoveryLegs');
const osmDiscoveryWarning = requiredElement<HTMLElement>('osmDiscoveryWarning');
const retryFailedOsmRequests = requiredElement<HTMLButtonElement>('retryFailedOsmRequests');
const cancelOsmDiscovery = requiredElement<HTMLButtonElement>('cancelOsmDiscovery');
const findControlPhotoOptions = requiredElement<HTMLButtonElement>('findControlPhotoOptions');
const controlPhotoReview = requiredElement<HTMLElement>('controlPhotoReview');
const controlPhotoStatus = requiredElement<HTMLElement>('controlPhotoStatus');
const controlPhotoProgress = requiredElement<HTMLElement>('controlPhotoProgress');
const controlPhotoProgressCount = requiredElement<HTMLElement>('controlPhotoProgressCount');
const controlPhotoProgressPhase = requiredElement<HTMLElement>('controlPhotoProgressPhase');
const controlPhotoProgressBar = requiredElement<HTMLProgressElement>('controlPhotoProgressBar');
const controlPhotoList = requiredElement<HTMLElement>('controlPhotoList');
const importControlPhotos = requiredElement<HTMLButtonElement>('importControlPhotos');
const generationProgress = requiredElement<HTMLProgressElement>('generationProgress');
const generateSpeedSetBtn = requiredElement<HTMLButtonElement>('generateSpeedSet');
const speedSetProgress = requiredElement<HTMLElement>('speedSetProgress');
const speedSetProgressText = requiredElement<HTMLElement>('speedSetProgressText');
const speedSetProgressCount = requiredElement<HTMLElement>('speedSetProgressCount');
const speedSetProgressBar = requiredElement<HTMLProgressElement>('speedSetProgressBar');
const downloadSpeedSetLink = requiredElement<HTMLAnchorElement>('downloadSpeedSet');
const routeSetupError = requiredElement<HTMLElement>('routeSetupError');
const speedError = requiredElement<HTMLElement>('speedError');
const takeoffBufferError = requiredElement<HTMLElement>('takeoffBufferError');
const minuteIntervalError = requiredElement<HTMLElement>('minuteIntervalError');
const routeControlSummary = requiredElement<HTMLElement>('routeControlSummary');
const routeLegSummary = requiredElement<HTMLElement>('routeLegSummary');
const routeDistanceSummary = requiredElement<HTMLElement>('routeDistanceSummary');
const routeDurationSummary = requiredElement<HTMLElement>('routeDurationSummary');
const controlPreparationBadge = requiredElement<HTMLElement>('controlPreparationBadge');
const competitionPreparationBadge = requiredElement<HTMLElement>('competitionPreparationBadge');
const selectedPhotoCount = requiredElement<HTMLElement>('selectedPhotoCount');
const photoLegCoverage = requiredElement<HTMLElement>('photoLegCoverage');
const photoSplitBalance = requiredElement<HTMLElement>('photoSplitBalance');
const photoCoverageWarning = requiredElement<HTMLElement>('photoCoverageWarning');
const readinessRoute = requiredElement<HTMLElement>('readinessRoute');
const readinessCount = requiredElement<HTMLElement>('readinessCount');
const readinessCoverage = requiredElement<HTMLElement>('readinessCoverage');
const readinessSplit = requiredElement<HTMLElement>('readinessSplit');
const readinessControls = requiredElement<HTMLElement>('readinessControls');
const readinessRules = requiredElement<HTMLElement>('readinessRules');
const readinessManual = requiredElement<HTMLElement>('readinessManual');
const readinessExceptions = requiredElement<HTMLElement>('readinessExceptions');
const routeOnlyNote = requiredElement<HTMLElement>('routeOnlyNote');
const workflowFindingGroups = requiredElement<HTMLElement>('workflowFindingGroups');

const downloadPdfLink = requiredElement<HTMLAnchorElement>('downloadPdf');
const downloadOverlayLink = requiredElement<HTMLAnchorElement>('downloadOverlay');
const downloadCroppedLink = requiredElement<HTMLAnchorElement>('downloadCropped');
const downloadSummaryLink = requiredElement<HTMLAnchorElement>('downloadSummary');
const downloadPhotoAnalysisLink = requiredElement<HTMLAnchorElement>('downloadPhotoAnalysis');
const downloadPhotoKeyLink = requiredElement<HTMLAnchorElement>('downloadPhotoKey');
const downloadPhotoHandoutLink = requiredElement<HTMLAnchorElement>('downloadPhotoHandout');
const downloadCompetitorPhotoHandoutLink = requiredElement<HTMLAnchorElement>(
  'downloadCompetitorPhotoHandout'
);
downloadCroppedLink.style.display = 'none';
const downloadUrls = {
  pdf: null,
  overlay: null,
  cropped: null,
  summary: null,
  photoAnalysis: null,
  photoKey: null,
  photoHandout: null,
  competitorPhotoHandout: null,
};
type SharedArtifactKey = 'cropped' | 'photoAnalysis' | 'photoKey' | 'photoHandout' | 'competitorPhotoHandout';
const sharedArtifactBytes: Record<SharedArtifactKey, Uint8Array | null> = {
  cropped: null,
  photoAnalysis: null,
  photoKey: null,
  photoHandout: null,
  competitorPhotoHandout: null,
};
let previewObjectUrl = null;
let speedSetObjectUrl: string | null = null;
let croppedPreviewController: AbortController | null = null;
let generationController: AbortController | null = null;
let speedSetGenerationActive = false;
const disabledControlState = new Map<
  HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement | HTMLButtonElement,
  boolean
>();

type ArtifactState = 'not-started' | 'processing' | 'ok' | 'manual-review' | 'failed' | 'cancelled';

interface GeneratedMapPair {
  judge: Uint8Array;
  competitor: Uint8Array;
}

interface GenerateOptions {
  speedText?: string;
  mapsOnly?: boolean;
}

interface SpeedEdition {
  speedText: string;
  slug: string;
  label: string;
}

function setArtifactState(artifact: string, state: ArtifactState, detail: string): void {
  const item = artifactStatuses.querySelector<HTMLElement>(`[data-artifact="${artifact}"]`);
  if (!item) return;
  item.dataset.state = state;
  const labels: Record<string, string> = {
    map: 'Judge map',
    overlay: 'Competitor route',
    crop: 'Empty map',
    preview: 'Preview',
    data: 'CSV/JSON',
    handout: 'Judge photo handout',
    competitorHandout: 'Competitor photo handout',
  };
  const label = labels[artifact] ?? artifact;
  item.textContent = `${label}: ${detail}`;
  generationProgress.value = artifactStatuses.querySelectorAll(
    '[data-state="ok"], [data-state="manual-review"], [data-state="failed"], [data-state="cancelled"]'
  ).length;
}

function resetArtifactStates(): void {
  generationProgress.value = 0;
  for (const artifact of ['map', 'overlay', 'crop', 'preview', 'data', 'handout', 'competitorHandout']) {
    setArtifactState(artifact, 'not-started', 'not started');
  }
}

function setGenerationBusy(busy: boolean): void {
  const controls = [
    ...workflowShell.querySelectorAll<
      HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement | HTMLButtonElement
    >('input,textarea,select,button'),
    ...document.querySelectorAll<HTMLButtonElement>('.workflow-stepper button'),
    saveProjectBtn,
    loadProjectBtn,
  ];
  if (busy) {
    disabledControlState.clear();
    controls.forEach((control) => {
      disabledControlState.set(control, control.disabled);
      control.disabled = true;
    });
  } else {
    disabledControlState.forEach((wasDisabled, control) => {
      if (control.isConnected) control.disabled = wasDisabled;
    });
    disabledControlState.clear();
  }
  cancelGenerationBtn.hidden = !busy;
  cancelGenerationBtn.disabled = !busy;
  statusEl.setAttribute('aria-busy', String(busy));
  workflowShell.setAttribute('aria-busy', String(busy));
  photoWorkflow.setExternalBusy(busy);
}

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
registerDownloadPreview(downloadCompetitorPhotoHandoutLink, 'competitorPhotoHandout');

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
let L: typeof import('leaflet') | null = null;
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

async function loadAssetBytes(url: string, label: string, signal?: AbortSignal): Promise<Uint8Array> {
  if (url.startsWith('data:')) {
    return decodeDataUrl(url).bytes;
  }
  let response: Response;
  try {
    response = await fetch(url, { signal });
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

function clearSpeedSetDownload(): void {
  if (speedSetGenerationActive) return;
  if (speedSetObjectUrl) {
    URL.revokeObjectURL(speedSetObjectUrl);
    speedSetObjectUrl = null;
  }
  downloadSpeedSetLink.removeAttribute('href');
  downloadSpeedSetLink.removeAttribute('download');
  downloadSpeedSetLink.hidden = true;
  speedSetProgress.hidden = true;
  speedSetProgress.removeAttribute('data-state');
}

function clearSharedArtifactBytes(): void {
  for (const key of Object.keys(sharedArtifactBytes) as SharedArtifactKey[]) {
    sharedArtifactBytes[key] = null;
  }
}

function invalidateGeneratedPackage(): void {
  clearSpeedSetDownload();
  clearSharedArtifactBytes();
  const downloads = [
    ['pdf', downloadPdfLink],
    ['overlay', downloadOverlayLink],
    ['cropped', downloadCroppedLink],
    ['summary', downloadSummaryLink],
    ['photoAnalysis', downloadPhotoAnalysisLink],
    ['photoKey', downloadPhotoKeyLink],
    ['photoHandout', downloadPhotoHandoutLink],
    ['competitorPhotoHandout', downloadCompetitorPhotoHandoutLink],
  ] as const;
  downloads.forEach(([key, link]) => {
    clearDownloadUrl(key, link);
    link.style.display = 'none';
  });
  clearCroppedPreview();
  outputsSection.hidden = true;
  summarySection.hidden = true;
  routeComplianceSection.hidden = true;
  resultsContent.hidden = true;
}

function setStatus(message, tone = message.startsWith('Error:') ? 'error' : 'neutral') {
  statusEl.textContent = message;
  statusEl.classList.toggle('is-error', tone === 'error');
  statusEl.classList.toggle('is-success', tone === 'success');
  statusEl.classList.toggle('is-warning', tone === 'warning');
}

function customSpeedIndex(preset = speedInput.value): number | null {
  const match = preset.match(/^custom-([1-4])$/);
  return match ? Number(match[1]) - 1 : null;
}

function customSpeedSettings(): SavedCustomSpeed[] {
  return customSpeedValueInputs.map((input, index) => ({
    value: input.value.trim(),
    unit: customSpeedUnitInputs[index].value === 'kmh' ? 'kmh' : 'kt',
  }));
}

function speedTextForPreset(preset: string, customSpeeds: SavedCustomSpeed[]): string {
  if (STANDARD_SPEED_PRESETS.includes(preset)) return preset;
  const index = customSpeedIndex(preset);
  if (index === null) throw new Error('Choose a valid groundspeed preset.');
  const custom = customSpeeds[index];
  if (!custom?.value.trim()) throw new Error(`Custom speed ${index + 1} is not configured.`);
  return `${custom.value}${custom.unit}`;
}

function selectedSpeedText(): string {
  return speedTextForPreset(speedInput.value, customSpeedSettings());
}

function updateCustomSpeedOptionLabels(): void {
  const customSpeeds = customSpeedSettings();
  customSpeeds.forEach((custom, index) => {
    const option = speedInput.querySelector<HTMLOptionElement>(`option[value="custom-${index + 1}"]`);
    if (!option) return;
    const unit = custom.unit === 'kmh' ? 'km/h' : 'kt';
    option.textContent = custom.value
      ? `Custom ${index + 1} — ${custom.value} ${unit}`
      : `Custom ${index + 1} — not set`;
  });
}

function applyCustomSpeedSettings(customSpeeds: SavedCustomSpeed[]): void {
  customSpeeds.forEach((custom, index) => {
    customSpeedValueInputs[index].value = custom.value;
    customSpeedUnitInputs[index].value = custom.unit;
  });
  updateCustomSpeedOptionLabels();
}

function currentProjectSettings(): SavedProjectSettings {
  return {
    waypoints: waypointTextarea.value,
    speed: selectedSpeedText(),
    speedPreset: speedInput.value,
    customSpeeds: customSpeedSettings(),
    takeoffBuffer: takeoffBufferInput.value,
    minuteInterval: minuteIntervalInput.value,
    mapKey: selectedMapKey,
    handoutSplit: handoutSplit.value,
    orthophotoRandomCount: orthophotoRandomCount.value,
    orthophotoAltitude: orthophotoAltitude.value,
    orthophotoFocalLength: orthophotoFocalLength.value,
    orthophotoDepression: orthophotoDepression.value,
    styleRouteWidth: styleRouteWidthInput.value,
    styleWaypointFont: styleWaypointFontInput.value,
    styleHeadingFont: styleHeadingFontInput.value,
    styleMinuteLabelFont: styleMinuteLabelFontInput.value,
    styleMinuteMarkerSize: styleMinuteMarkerSizeInput.value,
    styleMinuteLineWidth: styleMinuteLineWidthInput.value,
    photoExactDots: requiredElement<HTMLInputElement>('photoExactDots').checked,
    photoProjectedMarkers: requiredElement<HTMLInputElement>('photoProjectedMarkers').checked,
    photoHeadingArrows: requiredElement<HTMLInputElement>('photoHeadingArrows').checked,
    photoConnectors: requiredElement<HTMLInputElement>('photoConnectors').checked,
    photoLegend: requiredElement<HTMLInputElement>('photoLegend').checked,
    photoCropBounds: requiredElement<HTMLInputElement>('photoCropBounds').checked,
    handoutSummary: requiredElement<HTMLInputElement>('handoutSummary').checked,
  };
}

function validateProjectSettings(settings: SavedProjectSettings): void {
  parseWaypoints(settings.waypoints);
  parseSpeed(speedTextForPreset(settings.speedPreset, settings.customSpeeds));
  for (const [label, value] of [
    ['Takeoff-to-SP time', settings.takeoffBuffer],
    ['Minute-marker interval', settings.minuteInterval],
    ['DOF025 target count', settings.orthophotoRandomCount],
    ['DOF025 altitude', settings.orthophotoAltitude],
    ['DOF025 focal length', settings.orthophotoFocalLength],
    ['DOF025 depression angle', settings.orthophotoDepression],
    ['Route width', settings.styleRouteWidth],
    ['Waypoint font size', settings.styleWaypointFont],
    ['Heading font size', settings.styleHeadingFont],
    ['Minute label font size', settings.styleMinuteLabelFont],
    ['Minute marker size', settings.styleMinuteMarkerSize],
    ['Minute line width', settings.styleMinuteLineWidth],
  ]) {
    if (!Number.isFinite(Number(value)) || Number(value) <= 0) {
      throw new Error(`${label} in the project must be a positive number.`);
    }
  }
  if (!MAP_PRESETS[settings.mapKey]) throw new Error(`Unknown saved map preset: ${settings.mapKey}.`);
}

function applyProjectSettings(settings: SavedProjectSettings): void {
  waypointTextarea.value = settings.waypoints;
  applyCustomSpeedSettings(settings.customSpeeds);
  speedInput.value = settings.speedPreset;
  customSpeedPanel?.toggleAttribute('open', customSpeedIndex(settings.speedPreset) !== null);
  takeoffBufferInput.value = settings.takeoffBuffer;
  minuteIntervalInput.value = settings.minuteInterval;
  orthophotoRandomCount.value = settings.orthophotoRandomCount;
  orthophotoAltitude.value = settings.orthophotoAltitude;
  orthophotoFocalLength.value = settings.orthophotoFocalLength;
  orthophotoDepression.value = settings.orthophotoDepression;
  styleRouteWidthInput.value = settings.styleRouteWidth;
  styleWaypointFontInput.value = settings.styleWaypointFont;
  styleHeadingFontInput.value = settings.styleHeadingFont;
  styleMinuteLabelFontInput.value = settings.styleMinuteLabelFont;
  styleMinuteMarkerSizeInput.value = settings.styleMinuteMarkerSize;
  styleMinuteLineWidthInput.value = settings.styleMinuteLineWidth;
  requiredElement<HTMLInputElement>('photoExactDots').checked = settings.photoExactDots;
  requiredElement<HTMLInputElement>('photoProjectedMarkers').checked = settings.photoProjectedMarkers;
  requiredElement<HTMLInputElement>('photoHeadingArrows').checked = settings.photoHeadingArrows;
  requiredElement<HTMLInputElement>('photoConnectors').checked = settings.photoConnectors;
  requiredElement<HTMLInputElement>('photoLegend').checked = settings.photoLegend;
  requiredElement<HTMLInputElement>('photoCropBounds').checked = settings.photoCropBounds;
  requiredElement<HTMLInputElement>('handoutSummary').checked = settings.handoutSummary;
  handleMapPresetChange(settings.mapKey);
  osmThirdPartyConsent.checked = false;
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.hidden = true;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

async function createRouteProjectArtifact(): Promise<{
  project: SavedRouteProject;
  filename: string;
  bytes: Uint8Array;
}> {
  const project: SavedRouteProject = {
    format: PROJECT_FILE_FORMAT,
    schemaVersion: PROJECT_FILE_SCHEMA_VERSION,
    appVersion: APP_VERSION,
    savedAt: new Date().toISOString(),
    settings: currentProjectSettings(),
    photos: await photoWorkflow.saveProjectPhotos(),
  };
  return {
    project,
    filename: `route_project_v${PROJECT_FILE_SCHEMA_VERSION}_${project.savedAt.slice(0, 10)}.tvn-project`,
    bytes: strToU8(JSON.stringify(project)),
  };
}

async function saveRouteProject(): Promise<void> {
  if (generationController || photoWorkflow.isBusy || speedSetGenerationActive) {
    setStatus('Wait for the active operation to finish before saving the project.', 'warning');
    return;
  }
  saveProjectBtn.disabled = true;
  loadProjectBtn.disabled = true;
  photoWorkflow.setExternalBusy(true);
  try {
    const artifact = await createRouteProjectArtifact();
    downloadBlob(
      new Blob([Uint8Array.from(artifact.bytes).buffer], { type: 'application/json' }),
      artifact.filename
    );
    setStatus(`Saved route project with ${artifact.project.photos.length} embedded photo(s).`, 'success');
  } catch (error) {
    setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`, 'error');
  } finally {
    photoWorkflow.setExternalBusy(false);
    saveProjectBtn.disabled = false;
    loadProjectBtn.disabled = false;
  }
}

async function loadRouteProject(file: File): Promise<void> {
  if (generationController || photoWorkflow.isBusy || speedSetGenerationActive) {
    throw new Error('Wait for the active operation to finish before loading a project.');
  }
  const maximumProjectBytes = Math.ceil((PHOTO_IMPORT_LIMITS.maximumTotalBytes * 4) / 3) + 5 * 1024 * 1024;
  if (file.size > maximumProjectBytes)
    throw new Error('The selected project exceeds the safe file-size limit.');
  saveProjectBtn.disabled = true;
  loadProjectBtn.disabled = true;
  photoWorkflow.setExternalBusy(true);
  try {
    const project = parseSavedRouteProject(await file.text());
    validateProjectSettings(project.settings);
    const points = parseWaypoints(project.settings.waypoints);
    await photoWorkflow.restoreProjectPhotos(project.photos, points);
    applyProjectSettings(project.settings);
    photoWorkflow.analyze(points);
    if (Array.from(handoutSplit.options).some((option) => option.value === project.settings.handoutSplit)) {
      handoutSplit.value = project.settings.handoutSplit;
      photoWorkflow.analyze(points);
    }
    invalidateGeneratedPackage();
    clearOsmCandidateReview();
    clearControlPhotoReview();
    updateOrthophotoCoveragePreview();
    updateWorkflowReadiness();
    guidedWorkflow?.activate(1);
    setStatus(
      `Loaded project saved ${new Date(project.savedAt).toLocaleString()} with ${project.photos.length} photo(s). Review it, then generate again.`,
      'success'
    );
  } finally {
    photoWorkflow.setExternalBusy(false);
    saveProjectBtn.disabled = false;
    loadProjectBtn.disabled = false;
  }
}

function effectiveTaskCoordinates(photo, points): [number, number] | null {
  if (
    photo.taskLatitude.reliable &&
    photo.taskLongitude.reliable &&
    photo.taskLatitude.value !== null &&
    photo.taskLongitude.value !== null
  ) {
    return [photo.taskLatitude.value, photo.taskLongitude.value];
  }
  if (!photo.linkedWaypoint) return null;
  const linked = points.find(
    ([name]) => name.trim().toUpperCase() === photo.linkedWaypoint.trim().toUpperCase()
  );
  return linked ? [linked[1], linked[2]] : null;
}

function isRouteControlPhoto(photo): boolean {
  const waypoint = (photo.linkedWaypoint || photo.identifier).trim().toUpperCase();
  return waypoint === 'SP' || waypoint === 'FP' || Boolean(waypoint && /^TP\d+$/.test(waypoint));
}

function textNodeElement(text: string): HTMLElement {
  const element = document.createElement('span');
  element.textContent = text;
  return element;
}

const photoWorkflow = new PhotoWorkflow(setStatus);
let guidedWorkflow: GuidedWorkflow | null = null;
let discoveredOsmCandidates: OsmPhotoCandidate[] = [];
let selectedOsmCandidateIds = new Set<string>();
let osmCandidateRouteKey = '';
let osmDiscoveryBusy = false;
let osmSelectionSalt = '';
const osmDiscoveryWarningLegs = new Set<number>();
const osmDiscoveryCache = new Map<string, { data: OverpassResponse; cachedAt: number }>();
const OSM_CACHE_MAXIMUM_ENTRIES = 5;
const OSM_CACHE_TTL_MS = 10 * 60_000;
let osmDiscoveryStartedAt = 0;
let osmDiscoveryElapsedTimer: ReturnType<typeof setInterval> | null = null;
let osmDiscoveryController: AbortController | null = null;
let failedOsmRequests: OsmFailedRequest[] = [];
let controlPhotoProposals: ControlPhotoProposal[] = [];
let selectedControlPhotoRoles = new Map<string, 'true' | 'false'>();
let controlPhotoRouteKey = '';
let controlPhotoImportHadFailures = false;
let waypointAnalysisTimer: ReturnType<typeof setTimeout> | null = null;
waypointTextarea.addEventListener('input', () => {
  if (waypointAnalysisTimer !== null) clearTimeout(waypointAnalysisTimer);
  waypointAnalysisTimer = setTimeout(() => {
    waypointAnalysisTimer = null;
    if (generationController || photoWorkflow.isBusy) return;
    try {
      const points = parseWaypoints(waypointTextarea.value);
      photoWorkflow.analyze(points);
      if (osmCandidateRouteKey && osmCandidateRouteKey !== JSON.stringify(points)) {
        clearOsmCandidateReview();
      }
      if (controlPhotoRouteKey && controlPhotoRouteKey !== JSON.stringify(points)) {
        clearControlPhotoReview();
      }
      setStatus('Route changed: photo analysis refreshed. Generate again to rebuild outputs.');
    } catch {
      setStatus('Route changed: photo analysis is waiting for a valid route.', 'warning');
    }
    updateWorkflowReadiness();
  }, 300);
});

function readOrthophotoModel(): OrthophotoCaptureModel {
  const model = {
    altitudeM: Number(orthophotoAltitude.value),
    focalLength35Mm: Number(orthophotoFocalLength.value),
    depressionDeg: Number(orthophotoDepression.value),
    aspectRatio: 16 / 9,
  };
  if (!Number.isFinite(model.altitudeM) || model.altitudeM < 10 || model.altitudeM > 1000) {
    throw new Error('DOF025 height must be between 10 and 1,000 m AGL.');
  }
  if (!Number.isFinite(model.focalLength35Mm) || model.focalLength35Mm < 10 || model.focalLength35Mm > 300) {
    throw new Error('DOF025 focal length must be between 10 and 300 mm.');
  }
  if (!Number.isFinite(model.depressionDeg) || model.depressionDeg < 10 || model.depressionDeg > 90) {
    throw new Error('DOF025 depression angle must be between 10° and 90°.');
  }
  orthophotoCoverage(model);
  return model;
}

function updateOrthophotoCoveragePreview(): void {
  try {
    const coverage = orthophotoCoverage(readOrthophotoModel());
    orthophotoCoveragePreview.textContent = `≈ ${coverage.widthM.toFixed(0)} × ${coverage.heightM.toFixed(0)} m`;
    orthophotoCoveragePreview.removeAttribute('data-invalid');
  } catch {
    orthophotoCoveragePreview.textContent = 'Invalid capture model';
    orthophotoCoveragePreview.dataset.invalid = 'true';
  }
}

function requestedOsmTargetCount(): number {
  const count = Number(orthophotoRandomCount.value);
  if (!Number.isInteger(count) || count < 1 || count > 12) {
    throw new Error('Target count must be an integer from 1 to 12.');
  }
  return count;
}

function osmCacheKey(
  points: Array<[string, number, number]>,
  requestedCount: number,
  splitAfterM: number | null
): string {
  return JSON.stringify({ version: 2, points, requestedCount, splitAfterM });
}

function readCachedOsmData(key: string): OverpassResponse | null {
  const entry = osmDiscoveryCache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.cachedAt > OSM_CACHE_TTL_MS) {
    osmDiscoveryCache.delete(key);
    return null;
  }
  osmDiscoveryCache.delete(key);
  osmDiscoveryCache.set(key, entry);
  return entry.data;
}

function cacheOsmData(key: string, data: OverpassResponse): void {
  if (data.warnings?.length) return;
  osmDiscoveryCache.set(key, { data, cachedAt: Date.now() });
  while (osmDiscoveryCache.size > OSM_CACHE_MAXIMUM_ENTRIES) {
    const oldestKey = osmDiscoveryCache.keys().next().value;
    if (typeof oldestKey !== 'string') break;
    osmDiscoveryCache.delete(oldestKey);
  }
}

function setOsmDiscoveryBusy(busy: boolean): void {
  osmDiscoveryBusy = busy;
  addRandomOrthophotos.disabled = busy;
  orthophotoRandomCount.disabled = busy;
  waypointTextarea.disabled = busy;
  refreshOsmCandidates.disabled = busy;
  selectAllOsmCandidates.disabled = busy;
  clearOsmCandidateSelection.disabled = busy;
  importOsmCandidates.disabled = busy;
  retryFailedOsmRequests.disabled = busy || failedOsmRequests.length === 0;
  loadWaypointOrthophotos.disabled = busy;
  findControlPhotoOptions.disabled = busy;
  importControlPhotos.disabled = busy;
  handoutSplit.disabled = busy || handoutSplit.options.length === 0;
  cancelOsmDiscovery.hidden = !busy || osmDiscoveryController === null;
  cancelOsmDiscovery.disabled = !busy || osmDiscoveryController === null;
  osmCandidateReview.setAttribute('aria-busy', String(busy));
  controlPhotoReview.setAttribute('aria-busy', String(busy));
  photoWorkflow.setExternalBusy(busy);
  if (!busy && discoveredOsmCandidates.length > 0) renderOsmCandidateReview();
  if (!busy && controlPhotoProposals.length > 0) renderControlPhotoReview();
}

function stopOsmDiscoveryElapsedTimer(): void {
  if (osmDiscoveryElapsedTimer !== null) clearInterval(osmDiscoveryElapsedTimer);
  osmDiscoveryElapsedTimer = null;
}

function renderOsmDiscoveryPercent(): void {
  const percent = Math.round((osmDiscoveryProgressBar.value / osmDiscoveryProgressBar.max) * 100);
  const elapsedSeconds = Math.max(0, Math.floor((Date.now() - osmDiscoveryStartedAt) / 1000));
  osmDiscoveryProgressPercent.textContent = `${percent}% · ${elapsedSeconds}s`;
}

function beginOsmDiscoveryProgress(points: Array<[string, number, number]>): void {
  stopOsmDiscoveryElapsedTimer();
  osmDiscoveryStartedAt = Date.now();
  osmDiscoveryWarningLegs.clear();
  osmDiscoveryProgress.hidden = false;
  osmDiscoveryProgress.dataset.state = 'working';
  osmDiscoveryProgressText.textContent = 'Starting per-leg OpenStreetMap discovery…';
  osmDiscoveryProgressPercent.textContent = '0%';
  osmDiscoveryProgressBar.max = Math.max(1, (points.length - 1) * 3);
  osmDiscoveryProgressBar.value = 0;
  osmDiscoveryElapsedTimer = setInterval(renderOsmDiscoveryPercent, 1000);
  osmDiscoveryWarning.textContent =
    'Each route leg is checked separately. Building searches run only if stronger targets are insufficient.';
  osmDiscoveryLegs.replaceChildren(
    ...points.slice(0, -1).map(([name], index) => {
      const chip = document.createElement('span');
      chip.className = 'osm-leg-progress';
      chip.dataset.legIndex = String(index);
      chip.dataset.state = 'pending';
      chip.textContent = `${name}–${points[index + 1][0]}`;
      chip.setAttribute('aria-label', `${chip.textContent}: pending`);
      return chip;
    })
  );
}

function setOsmLegProgressState(chip: HTMLElement | null, state: string): void {
  if (!chip) return;
  chip.dataset.state = state;
  chip.setAttribute('aria-label', `${chip.textContent}: ${state}`);
}

function updateOsmDiscoveryProgress(progress: OsmDiscoveryProgress): void {
  const stageLabels = {
    features: 'identifiable features',
    roads: 'road junctions',
    buildings: 'building fallback',
  } as const;
  osmDiscoveryProgressBar.max = progress.totalSteps;
  osmDiscoveryProgressBar.value = progress.completedSteps;
  renderOsmDiscoveryPercent();
  const chip = osmDiscoveryLegs.querySelector<HTMLElement>(`[data-leg-index="${progress.legIndex}"]`);
  if (progress.state === 'started' || progress.state === 'retrying') {
    const retryPrefix = progress.state === 'retrying' ? 'Retrying after the first leg pass · ' : '';
    if (progress.detail?.includes('shared route feature index')) {
      osmDiscoveryProgressText.textContent = `Downloading the shared OpenStreetMap feature index for all ${progress.legCount} legs…`;
      setStatus(`Downloading OpenStreetMap features for all ${progress.legCount} route legs…`);
      osmDiscoveryLegs.querySelectorAll<HTMLElement>('.osm-leg-progress').forEach((legChip) => {
        setOsmLegProgressState(legChip, 'active');
      });
    } else {
      osmDiscoveryProgressText.textContent = `${retryPrefix}Leg ${progress.legIndex + 1} of ${progress.legCount} · ${progress.legName}: checking ${stageLabels[progress.stage]}…`;
      setStatus(
        `Searching OpenStreetMap · leg ${progress.legIndex + 1} of ${progress.legCount} · ${stageLabels[progress.stage]}…`
      );
      setOsmLegProgressState(chip, 'active');
    }
  } else if (progress.state === 'warning') {
    osmDiscoveryWarningLegs.add(progress.legIndex);
    osmDiscoveryWarning.textContent = `${progress.legName} ${stageLabels[progress.stage]} could not be loaded (${progress.detail ?? 'request failed'}). Continuing with completed results.`;
    setOsmLegProgressState(chip, 'warning');
  } else if (progress.state === 'recovered') {
    osmDiscoveryWarningLegs.delete(progress.legIndex);
    osmDiscoveryWarning.textContent = `${progress.legName} ${stageLabels[progress.stage]} recovered on the deferred retry pass.`;
    setOsmLegProgressState(chip, 'recovered');
  } else if (progress.stage === 'features' || progress.stage === 'roads') {
    setOsmLegProgressState(chip, osmDiscoveryWarningLegs.has(progress.legIndex) ? 'warning' : 'scanned');
  } else if (progress.stage === 'buildings') {
    setOsmLegProgressState(chip, osmDiscoveryWarningLegs.has(progress.legIndex) ? 'warning' : 'done');
  }
}

function clearControlPhotoReview(): void {
  controlPhotoProposals = [];
  selectedControlPhotoRoles.clear();
  controlPhotoRouteKey = '';
  controlPhotoImportHadFailures = false;
  controlPhotoList.replaceChildren();
  controlPhotoProgress.hidden = true;
  controlPhotoProgress.removeAttribute('data-state');
  controlPhotoProgressBar.value = 0;
  controlPhotoProgressBar.max = 1;
  controlPhotoReview.hidden = true;
}

function importedControlPhoto(waypoint: string) {
  return photoWorkflow.records.find(
    (record) => record.generatedOrthophoto?.targetSource?.controlWaypoint === waypoint
  );
}

function updateControlPhotoImportAction(): void {
  let routeControlCount = controlPhotoProposals.length;
  try {
    routeControlCount = parseWaypoints(waypointTextarea.value).length;
  } catch {
    // Route validation owns the detailed error; keep the review action usable.
  }
  const pendingCount = controlPhotoProposals.filter(
    ({ waypoint }) => !importedControlPhoto(waypoint[0])
  ).length;
  const missingCount = Math.max(
    0,
    routeControlCount -
      new Set([
        ...controlPhotoProposals.map(({ waypoint }) => waypoint[0]),
        ...photoWorkflow.records
          .map((record) => record.generatedOrthophoto?.targetSource?.controlWaypoint)
          .filter((waypoint): waypoint is string => Boolean(waypoint)),
      ]).size
  );
  importControlPhotos.disabled = osmDiscoveryBusy || pendingCount === 0;
  importControlPhotos.textContent =
    pendingCount === 0
      ? missingCount > 0
        ? `${missingCount} control photo${missingCount === 1 ? '' : 's'} still missing`
        : 'Control photos imported'
      : controlPhotoImportHadFailures
        ? `Retry ${pendingCount} failed control photo${pendingCount === 1 ? '' : 's'}`
        : controlPhotoProposals.length < routeControlCount
          ? `Import ${pendingCount} available control photo${pendingCount === 1 ? '' : 's'}`
          : 'Import chosen control photos';
}

function renderControlPhotoReview(): void {
  const coverage = orthophotoCoverage(readOrthophotoModel());
  const points = parseWaypoints(waypointTextarea.value);
  const proposalsByWaypoint = new Map(
    controlPhotoProposals.map((proposal) => [proposal.waypoint[0], proposal])
  );
  const fragment = document.createDocumentFragment();
  for (const [waypointName] of points) {
    const proposal = proposalsByWaypoint.get(waypointName);
    const imported = importedControlPhoto(waypointName);
    const row = document.createElement('section');
    row.className = 'control-photo-row';
    row.dataset.controlWaypoint = waypointName;
    const heading = document.createElement('h5');
    heading.className = 'control-photo-waypoint';
    heading.textContent = waypointName;
    row.appendChild(heading);
    if (!proposal) {
      if (imported) {
        const importedStatus = document.createElement('p');
        importedStatus.className = 'note';
        importedStatus.textContent = `Already imported as ${imported.classification === 'control-false' ? 'false' : 'true'}.`;
        row.appendChild(importedStatus);
      } else {
        row.dataset.state = 'missing';
        const missing = document.createElement('div');
        missing.className = 'control-photo-missing';
        const missingCopy = document.createElement('div');
        const missingTitle = document.createElement('strong');
        missingTitle.textContent = `No ${waypointName} photo proposal`;
        const missingDetail = document.createElement('p');
        missingDetail.className = 'note';
        missingDetail.textContent =
          'The previous OpenStreetMap lookup did not complete or found no usable object.';
        missingCopy.append(missingTitle, missingDetail);
        const retry = document.createElement('button');
        retry.type = 'button';
        retry.className = 'btn btn-secondary btn-small';
        retry.dataset.retryControlWaypoint = waypointName;
        retry.textContent = `Retry ${waypointName}`;
        retry.disabled = osmDiscoveryBusy;
        missing.append(missingCopy, retry);
        row.appendChild(missing);
      }
      fragment.appendChild(row);
      continue;
    }
    const options: Array<['true' | 'false', typeof proposal.trueTarget | null]> = [
      ['true', proposal.trueTarget],
      ...(waypointName === 'SP' || waypointName === 'FP'
        ? []
        : ([['false', proposal.falseTarget]] as Array<
            ['true' | 'false', typeof proposal.trueTarget | null]
          >)),
    ];
    for (const [role, target] of options) {
      if (!target) continue;
      const option = document.createElement('div');
      option.className = 'control-photo-option';
      const label = document.createElement('label');
      label.className = 'control-photo-choice';
      const choice = document.createElement('input');
      choice.type = 'radio';
      choice.name = `control-photo-${waypointName}`;
      choice.value = role;
      choice.dataset.controlWaypoint = waypointName;
      choice.checked = selectedControlPhotoRoles.get(waypointName) === role;
      choice.disabled = Boolean(imported);
      const title = document.createElement('strong');
      title.textContent =
        role === 'true' ? `True · exact ${waypointName} position` : `False · ${target.name}`;
      const metrics = document.createElement('span');
      metrics.className = 'note';
      metrics.textContent =
        role === 'true'
          ? `OSM reference: ${target.name} (${target.featureType}), ${target.distanceFromWaypointM.toFixed(0)} m away`
          : `${target.featureType} · ${(target.distanceFromCorrectM / 1852).toFixed(2)} NM from the true object`;
      const preview = document.createElement('img');
      preview.alt = `${waypointName} ${role} orthophoto preview`;
      preview.loading = 'lazy';
      preview.src = orthophotoRequestUrl(target.latitude, target.longitude, coverage, 800);
      label.append(choice, title, metrics, preview);
      const technical = document.createElement('details');
      technical.className = 'candidate-technical-details';
      const technicalSummary = document.createElement('summary');
      technicalSummary.textContent = 'OSM reference details';
      const technicalCopy = document.createElement('p');
      technicalCopy.textContent = `${target.source.provider} ${target.source.elementId} · ${target.source.category} / ${target.source.featureType} · score ${target.source.score} · ${target.source.attribution}`;
      technical.append(technicalSummary, technicalCopy);
      option.append(label, technical);
      row.appendChild(option);
    }
    if (imported) {
      const importedStatus = document.createElement('p');
      importedStatus.className = 'note';
      importedStatus.textContent = `Already imported as ${imported.classification === 'control-false' ? 'false' : 'true'}.`;
      row.appendChild(importedStatus);
    }
    fragment.appendChild(row);
  }
  controlPhotoList.replaceChildren(fragment);
  controlPhotoReview.hidden = false;
  updateControlPhotoImportAction();
  updateWorkflowReadiness();
}

function finishOsmDiscoveryProgress(candidateCount: number, warnings: string[]): void {
  stopOsmDiscoveryElapsedTimer();
  osmDiscoveryProgressBar.value = osmDiscoveryProgressBar.max;
  osmDiscoveryProgressPercent.textContent = '100%';
  osmDiscoveryProgress.dataset.state = warnings.length ? 'warning' : 'complete';
  osmDiscoveryProgressText.textContent = `Discovery complete · ${candidateCount} eligible feature${candidateCount === 1 ? '' : 's'} found`;
  osmDiscoveryLegs.querySelectorAll<HTMLElement>('.osm-leg-progress').forEach((chip) => {
    if (chip.dataset.state !== 'warning') setOsmLegProgressState(chip, 'done');
  });
  osmDiscoveryWarning.textContent = warnings.length
    ? `${warnings.length} OpenStreetMap request${warnings.length === 1 ? '' : 's'} could not be loaded. Candidates from completed stages were retained. ${warnings.slice(0, 3).join(' | ')}${warnings.length > 3 ? ` | ${warnings.length - 3} more` : ''}`
    : 'Every required leg stage completed. Unnecessary building searches were skipped.';
  retryFailedOsmRequests.hidden = failedOsmRequests.length === 0;
  retryFailedOsmRequests.disabled = false;
  retryFailedOsmRequests.textContent = `Retry ${failedOsmRequests.length} failed OSM request${failedOsmRequests.length === 1 ? '' : 's'}`;
}

function failOsmDiscoveryProgress(message: string): void {
  stopOsmDiscoveryElapsedTimer();
  osmDiscoveryProgress.hidden = false;
  osmDiscoveryProgress.dataset.state = 'error';
  osmDiscoveryProgressText.textContent = 'OpenStreetMap discovery stopped';
  osmDiscoveryWarning.textContent = message;
}

function clearOsmCandidateReview(): void {
  stopOsmDiscoveryElapsedTimer();
  discoveredOsmCandidates = [];
  selectedOsmCandidateIds.clear();
  osmCandidateRouteKey = '';
  osmSelectionSalt = '';
  failedOsmRequests = [];
  retryFailedOsmRequests.hidden = true;
  osmCandidateList.replaceChildren();
  osmCandidateReview.hidden = true;
  osmDiscoveryProgress.hidden = true;
  osmDiscoveryWarningLegs.clear();
}

function selectedOsmCandidates(): OsmPhotoCandidate[] {
  return discoveredOsmCandidates.filter((candidate) => selectedOsmCandidateIds.has(candidate.id));
}

function osmSelectionWarnings(selected: OsmPhotoCandidate[]): string[] {
  if (selected.length === 0) return [];
  const warnings: string[] = [];
  const selectionIssue = targetSelectionIssue(
    selected,
    requestedOsmTargetCount(),
    photoWorkflow.handoutOptions.splitAfterM
  );
  if (selectionIssue) warnings.push(selectionIssue);
  const existingEnroute = photoWorkflow.records.filter(
    (record) => record.classification === 'enroute'
  ).length;
  if (existingEnroute + selected.length > 12) {
    warnings.push(
      `Rule A2.4.5 warning: importing all selected targets would create ${existingEnroute + selected.length} en-route photos; permitted maximum 12.`
    );
  }
  try {
    const points = parseWaypoints(waypointTextarea.value);
    const existingTasks = photoWorkflow.records.filter((record) => isCountedRouteTask(record, points)).length;
    if (existingTasks + selected.length > 15) {
      warnings.push(
        `Rule A2.4.6 warning: importing all selected targets would create ${existingTasks + selected.length} route tasks; permitted maximum 15.`
      );
    }
  } catch {
    // Route parsing already has its own validation; keep target review usable.
  }
  return warnings;
}

function renderOsmCandidateReview(): void {
  const selected = selectedOsmCandidates();
  const warnings = osmSelectionWarnings(selected);
  const exceedsImportCapacity =
    photoWorkflow.records.length + selected.length > PHOTO_IMPORT_LIMITS.maximumCount;
  osmCandidateSummary.textContent = `All ${discoveredOsmCandidates.length} eligible OSM feature${discoveredOsmCandidates.length === 1 ? ' is' : 's are'} shown. ${selected.length} selected. Selection mix ${osmSelectionSalt.slice(0, 8)}. The initial selection is only a suggestion; select any number of targets.`;
  osmCandidateSelectionStatus.textContent =
    selected.length === 0
      ? 'Select at least one target to import.'
      : exceedsImportCapacity
        ? `${selected.length} selected · app safety limit: a maximum of ${PHOTO_IMPORT_LIMITS.maximumCount} total photos can be loaded.`
        : warnings.length
          ? `${selected.length} selected · ${warnings.join(' ')}`
          : `${selected.length} selected · ready for DOF025 import`;
  osmCandidateSelectionStatus.dataset.tone =
    selected.length === 0 || exceedsImportCapacity || warnings.length ? 'warning' : 'ok';
  importOsmCandidates.disabled = osmDiscoveryBusy || selected.length === 0 || exceedsImportCapacity;
  const fragment = document.createDocumentFragment();
  const legCounts = new Map<number, number>();
  for (const candidate of discoveredOsmCandidates) {
    legCounts.set(candidate.routeLegIndex, (legCounts.get(candidate.routeLegIndex) ?? 0) + 1);
  }
  let currentLegIndex = -1;
  let currentLegGrid: HTMLElement | null = null;
  for (const candidate of discoveredOsmCandidates) {
    if (candidate.routeLegIndex !== currentLegIndex) {
      currentLegIndex = candidate.routeLegIndex;
      const group = document.createElement('section');
      group.className = 'osm-candidate-leg-group';
      group.dataset.legIndex = String(currentLegIndex);
      const heading = document.createElement('h5');
      heading.className = 'osm-candidate-leg-heading';
      heading.textContent = `${candidate.routeLegName} · ${legCounts.get(currentLegIndex) ?? 0} candidate${legCounts.get(currentLegIndex) === 1 ? '' : 's'}`;
      currentLegGrid = document.createElement('div');
      currentLegGrid.className = 'osm-candidate-leg-grid';
      group.append(heading, currentLegGrid);
      fragment.appendChild(group);
    }
    const row = document.createElement('article');
    row.className = 'osm-candidate';
    const choice = document.createElement('label');
    choice.className = 'osm-candidate-choice';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = selectedOsmCandidateIds.has(candidate.id);
    checkbox.dataset.osmCandidateId = candidate.id;
    const copy = document.createElement('span');
    copy.className = 'osm-candidate-copy';
    const title = document.createElement('strong');
    title.textContent = candidate.name;
    const context = document.createElement('span');
    context.textContent = `${candidate.featureType} · ${candidate.routeLegName}`;
    copy.append(title, context);
    const confidence = document.createElement('span');
    confidence.className = 'osm-confidence';
    confidence.dataset.confidence = candidate.confidence;
    confidence.textContent = `heuristic ${candidate.confidence} ${candidate.score}`;
    choice.append(checkbox, copy, confidence);
    const technical = document.createElement('details');
    technical.className = 'candidate-technical-details';
    const technicalSummary = document.createElement('summary');
    technicalSummary.textContent = 'Route and OSM details';
    const technicalCopy = document.createElement('p');
    technicalCopy.textContent = `${((candidate.alongRouteM ?? 0) / 1852).toFixed(1)} NM along route · ${candidate.lateralDistanceM.toFixed(0)} m lateral · ${candidate.distanceAfterControlM.toFixed(0)} m after ${candidate.previousControlPoint} · ${candidate.source.elementId} · ${candidate.source.attribution}`;
    technical.append(technicalSummary, technicalCopy);
    row.append(choice, technical);
    currentLegGrid?.appendChild(row);
  }
  osmCandidateList.replaceChildren(fragment);
  osmCandidateReview.hidden = false;
  updateWorkflowReadiness();
}

function chooseOsmCandidateProposal(): void {
  const selected = selectOsmPhotoCandidates(
    discoveredOsmCandidates,
    requestedOsmTargetCount(),
    photoWorkflow.handoutOptions.splitAfterM,
    createSeededRandom(osmSelectionSalt)
  );
  selectedOsmCandidateIds = new Set(selected.map((candidate) => candidate.id));
  renderOsmCandidateReview();
}

function newOsmSelectionSalt(): string {
  const values = crypto.getRandomValues(new Uint32Array(2));
  return Array.from(values, (value) => value.toString(16).padStart(8, '0')).join('');
}

function appendOrthophotoTargetRows(targets: OrthophotoTarget[]): void {
  const rows = targets.map(
    (target) => `PHOTO_${target.label},${target.latitude.toFixed(6)},${target.longitude.toFixed(6)}`
  );
  const current = waypointTextarea.value.trimEnd();
  waypointTextarea.value = `${current}${current ? '\n' : ''}${rows.join('\n')}`;
  waypointTextarea.dispatchEvent(new Event('input'));
}

for (const input of [orthophotoAltitude, orthophotoFocalLength, orthophotoDepression]) {
  input.addEventListener('input', updateOrthophotoCoveragePreview);
}
orthophotoRandomCount.addEventListener('input', () => {
  if (discoveredOsmCandidates.length === 0) return;
  try {
    chooseOsmCandidateProposal();
  } catch {
    osmCandidateSelectionStatus.textContent = 'Enter a target count from 1 to 12.';
    importOsmCandidates.disabled = true;
  }
});
handoutSplit.addEventListener('change', () => {
  if (discoveredOsmCandidates.length === 0 || osmDiscoveryBusy) return;
  try {
    osmSelectionSalt = newOsmSelectionSalt();
    chooseOsmCandidateProposal();
    setStatus('Handout split changed: the proposed OSM targets were reselected and revalidated.', 'warning');
  } catch (error) {
    setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`, 'error');
  }
});
updateOrthophotoCoveragePreview();

addRandomOrthophotos.addEventListener('click', async () => {
  if (photoWorkflow.isBusy || osmDiscoveryBusy) return;
  try {
    const count = requestedOsmTargetCount();
    const points = parseWaypoints(waypointTextarea.value);
    photoWorkflow.analyze(points);
    const route = buildRoute(points);
    const routeKey = JSON.stringify(points);
    const retainedCandidates = osmCandidateRouteKey === routeKey ? [...discoveredOsmCandidates] : [];
    const splitAfterM = photoWorkflow.handoutOptions.splitAfterM;
    const cacheKey = osmCacheKey(points, count, splitAfterM);
    const cachedData = readCachedOsmData(cacheKey);
    clearOsmCandidateReview();
    osmSelectionSalt = newOsmSelectionSalt();
    beginOsmDiscoveryProgress(points);
    if (!cachedData) {
      osmDiscoveryController = new AbortController();
    }
    setOsmDiscoveryBusy(true);
    setStatus(
      cachedData
        ? 'Reusing the complete cached OpenStreetMap discovery with a new selection mix…'
        : `Searching OpenStreetMap leg by leg (1 of ${points.length - 1})…`
    );
    const data =
      cachedData ??
      (await fetchOverpassPhotoData(points, {
        route,
        requestedCount: count,
        splitAfterM,
        signal: osmDiscoveryController?.signal,
        onProgress: updateOsmDiscoveryProgress,
      }));
    const newlyDiscoveredCandidates = discoverOsmPhotoCandidates(data, route, points);
    discoveredOsmCandidates = [
      ...new Map(
        [...retainedCandidates, ...newlyDiscoveredCandidates].map((candidate) => [candidate.id, candidate])
      ).values(),
    ].sort((left, right) => (left.alongRouteM ?? 0) - (right.alongRouteM ?? 0) || right.score - left.score);
    osmCandidateRouteKey = routeKey;
    if (discoveredOsmCandidates.length === 0) {
      throw new Error('No identifiable OSM targets met the route, clearance, and distance requirements.');
    }
    if (!cachedData) cacheOsmData(cacheKey, data);
    failedOsmRequests = data.failedRequests ?? [];
    finishOsmDiscoveryProgress(discoveredOsmCandidates.length, data.warnings ?? []);
    chooseOsmCandidateProposal();
    const selectedCount = selectedOsmCandidateIds.size;
    const partialWarning = data.warnings?.length
      ? ` ${data.warnings.length} OpenStreetMap request${data.warnings.length === 1 ? '' : 's'} could not be loaded; completed results were retained.`
      : '';
    const accumulatedNotice = retainedCandidates.length
      ? ` Results were merged with ${retainedCandidates.length} candidates retained from the previous attempt.`
      : '';
    setStatus(
      `${`Found ${discoveredOsmCandidates.length} eligible OSM features and displayed all of them. ${selectedCount} are initially selected; choose any number before importing.`}${partialWarning}${accumulatedNotice}`,
      data.warnings?.length ? 'warning' : 'success'
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    failOsmDiscoveryProgress(message);
    setStatus(`Error: ${message}`, 'error');
  } finally {
    osmDiscoveryController = null;
    if (osmDiscoveryBusy) setOsmDiscoveryBusy(false);
  }
});

retryFailedOsmRequests.addEventListener('click', async () => {
  if (photoWorkflow.isBusy || osmDiscoveryBusy || failedOsmRequests.length === 0) return;
  try {
    const points = parseWaypoints(waypointTextarea.value);
    const route = buildRoute(points);
    const retryCount = failedOsmRequests.length;
    osmDiscoveryController = new AbortController();
    beginOsmDiscoveryProgress(points);
    osmDiscoveryProgressText.textContent = `Retrying ${retryCount} failed OpenStreetMap request${retryCount === 1 ? '' : 's'}…`;
    retryFailedOsmRequests.textContent = 'Retrying failed OSM requests…';
    setOsmDiscoveryBusy(true);
    setStatus(`Retrying only the ${retryCount} failed OpenStreetMap request${retryCount === 1 ? '' : 's'}…`);
    const data = await retryOverpassPhotoRequests(points, failedOsmRequests, {
      route,
      signal: osmDiscoveryController.signal,
      onProgress: updateOsmDiscoveryProgress,
    });
    const recoveredCandidates = discoverOsmPhotoCandidates(data, route, points);
    discoveredOsmCandidates = [
      ...new Map(
        [...discoveredOsmCandidates, ...recoveredCandidates].map((candidate) => [candidate.id, candidate])
      ).values(),
    ].sort((left, right) => (left.alongRouteM ?? 0) - (right.alongRouteM ?? 0) || right.score - left.score);
    failedOsmRequests = data.failedRequests ?? [];
    finishOsmDiscoveryProgress(discoveredOsmCandidates.length, data.warnings ?? []);
    renderOsmCandidateReview();
    const recoveredCount = retryCount - failedOsmRequests.length;
    setStatus(
      failedOsmRequests.length
        ? `Recovered ${recoveredCount} of ${retryCount} failed OpenStreetMap requests; ${failedOsmRequests.length} can be retried again.`
        : `Recovered all ${retryCount} failed OpenStreetMap request${retryCount === 1 ? '' : 's'} and merged the new candidates.`,
      failedOsmRequests.length ? 'warning' : 'success'
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    failOsmDiscoveryProgress(message);
    setStatus(`Error: ${message}`, 'error');
  } finally {
    osmDiscoveryController = null;
    if (osmDiscoveryBusy) setOsmDiscoveryBusy(false);
  }
});

cancelOsmDiscovery.addEventListener('click', () => {
  osmDiscoveryController?.abort(new Error('OpenStreetMap discovery cancelled by the user.'));
});

refreshOsmCandidates.addEventListener('click', () => {
  if (discoveredOsmCandidates.length === 0 || osmDiscoveryBusy) return;
  try {
    const previousIds = [...selectedOsmCandidateIds].sort().join('|');
    let changed = false;
    for (let attempt = 0; attempt < 12; attempt += 1) {
      osmSelectionSalt = newOsmSelectionSalt();
      chooseOsmCandidateProposal();
      if ([...selectedOsmCandidateIds].sort().join('|') !== previousIds) {
        changed = true;
        break;
      }
    }
    setStatus(
      changed
        ? 'Replaced the proposed OSM target selection. Review it before importing.'
        : 'No alternative valid target set was available; the selection remains unchanged.',
      changed ? 'neutral' : 'warning'
    );
  } catch (error) {
    setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`, 'error');
  }
});

selectAllOsmCandidates.addEventListener('click', () => {
  selectedOsmCandidateIds = new Set(discoveredOsmCandidates.map((candidate) => candidate.id));
  renderOsmCandidateReview();
  setStatus(`Selected all ${selectedOsmCandidateIds.size} discovered OSM targets.`, 'warning');
});

clearOsmCandidateSelection.addEventListener('click', () => {
  selectedOsmCandidateIds.clear();
  renderOsmCandidateReview();
  setStatus('Cleared the OSM target selection.');
});

osmCandidateList.addEventListener('change', (event) => {
  const checkbox = (event.target as HTMLElement).closest<HTMLInputElement>('input[data-osm-candidate-id]');
  if (!checkbox) return;
  const id = checkbox.dataset.osmCandidateId;
  if (!id) return;
  if (checkbox.checked) selectedOsmCandidateIds.add(id);
  else selectedOsmCandidateIds.delete(id);
  const scrollTop = osmCandidateList.scrollTop;
  renderOsmCandidateReview();
  osmCandidateList.scrollTop = scrollTop;
  osmCandidateList.querySelector<HTMLInputElement>(`[data-osm-candidate-id="${CSS.escape(id)}"]`)?.focus();
});

importOsmCandidates.addEventListener('click', async () => {
  if (photoWorkflow.isBusy || osmDiscoveryBusy) return;
  try {
    const selected = selectedOsmCandidates();
    if (selected.length === 0) throw new Error('Select at least one proposed target.');
    const existingLabels = new Set(
      parseOrthophotoTargets(waypointTextarea.value).map((target) => target.label.toLocaleUpperCase())
    );
    const targets: OrthophotoTarget[] = selected.map((candidate, index) => {
      const category = candidate.category
        .replace(/[^A-Z0-9]+/gi, '_')
        .toUpperCase()
        .slice(0, 14);
      let suffix = index + 1;
      let label = `OSM_${category}_${String(suffix).padStart(2, '0')}`;
      while (existingLabels.has(label)) {
        suffix += 1;
        label = `OSM_${category}_${String(suffix).padStart(2, '0')}`;
      }
      existingLabels.add(label);
      return {
        ...candidate,
        label,
        source: { ...candidate.source, selectionSalt: osmSelectionSalt },
      };
    });
    const model = readOrthophotoModel();
    const result = await photoWorkflow.importOrthophotoTargets(targets, model);
    if (result.importedTargets.length > 0) appendOrthophotoTargetRows(result.importedTargets);
    if (result.failedTargets.length === 0 && !result.cancelled) {
      clearOsmCandidateReview();
    } else {
      const failedElementIds = new Set(
        result.failedTargets
          .map(({ target }) => target.source?.elementId)
          .filter((id): id is string => Boolean(id))
      );
      selectedOsmCandidateIds = new Set(
        discoveredOsmCandidates
          .filter((candidate) => failedElementIds.has(candidate.source.elementId))
          .map((candidate) => candidate.id)
      );
      renderOsmCandidateReview();
    }
  } catch (error) {
    setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`, 'error');
  }
});

async function discoverControlPhotoOptions(
  points: Waypoint[],
  requestedIndices: number[],
  replaceExisting: boolean
): Promise<void> {
  if (photoWorkflow.isBusy || osmDiscoveryBusy) return;
  try {
    photoWorkflow.analyze(points);
    const previousSelections = new Map(selectedControlPhotoRoles);
    if (replaceExisting) clearControlPhotoReview();
    osmDiscoveryController = new AbortController();
    setOsmDiscoveryBusy(true);
    controlPhotoReview.hidden = false;
    const controlLookupStepCount = Math.max(
      1,
      requestedIndices.reduce(
        (total, index) => total + (index === 0 || index === points.length - 1 ? 1 : 2),
        0
      )
    );
    const requestedPosition = new Map(requestedIndices.map((index, position) => [index, position]));
    const completedLookups = new Set<string>();
    const requestedNames = requestedIndices.map((index) => points[index][0]);
    controlPhotoProgress.hidden = false;
    controlPhotoProgress.dataset.state = 'working';
    controlPhotoProgressBar.max = controlLookupStepCount;
    controlPhotoProgressBar.value = 0;
    controlPhotoProgressCount.textContent = `Preparing ${requestedNames.join(', ')}`;
    controlPhotoProgressPhase.textContent = `0 of ${controlLookupStepCount} lookup steps`;
    controlPhotoStatus.textContent = `Finding photo options for ${requestedNames.join(', ')}…`;
    setStatus(`Searching OpenStreetMap for ${requestedNames.join(', ')} photo options…`);
    const result = await fetchOsmControlPhotoProposals(points, {
      signal: osmDiscoveryController.signal,
      waypointIndices: requestedIndices,
      onProgress: ({ waypointIndex, waypointName, stage, state }) => {
        if (['completed', 'recovered', 'warning'].includes(state)) {
          completedLookups.add(`${waypointIndex}:${stage}`);
        }
        controlPhotoProgressBar.value = Math.min(controlPhotoProgressBar.max, completedLookups.size);
        const action =
          state === 'retrying'
            ? 'retrying after the first pass'
            : stage === 'true'
              ? 'finding the true object'
              : 'finding a similar false object 1-10 NM away';
        controlPhotoProgressCount.textContent = `Preparing control ${(requestedPosition.get(waypointIndex) ?? 0) + 1} of ${requestedIndices.length} · ${waypointName}`;
        controlPhotoProgressPhase.textContent = `${action} · ${controlPhotoProgressBar.value} of ${controlPhotoProgressBar.max} lookup steps`;
        controlPhotoStatus.textContent = `${waypointName} · ${action}`;
      },
    });
    const merged = new Map(
      (replaceExisting ? [] : controlPhotoProposals).map((proposal) => [proposal.waypoint[0], proposal])
    );
    for (const proposal of result.proposals) merged.set(proposal.waypoint[0], proposal);
    controlPhotoProposals = points.flatMap(([waypointName]) => {
      const proposal = merged.get(waypointName);
      return proposal ? [proposal] : [];
    });
    controlPhotoRouteKey = JSON.stringify(points);
    selectedControlPhotoRoles = new Map([
      ...(replaceExisting ? [] : previousSelections.entries()),
      ...result.proposals.map(({ waypoint }) => {
        const waypointName = waypoint[0];
        const importedRole =
          importedControlPhoto(waypointName)?.generatedOrthophoto?.targetSource?.controlRole;
        return [
          waypointName,
          importedRole ?? previousSelections.get(waypointName) ?? ('true' as const),
        ] as const;
      }),
    ]);
    controlPhotoImportHadFailures = false;
    renderControlPhotoReview();
    controlPhotoProgressBar.value = controlPhotoProgressBar.max;
    controlPhotoProgress.dataset.state = result.warnings.length ? 'warning' : 'complete';
    controlPhotoProgressCount.textContent = `${requestedIndices.length} control${requestedIndices.length === 1 ? '' : 's'} processed`;
    controlPhotoProgressPhase.textContent = `${result.proposals.length} new photo proposal${result.proposals.length === 1 ? '' : 's'} ready`;
    const preparedCount = new Set([
      ...controlPhotoProposals.map(({ waypoint }) => waypoint[0]),
      ...photoWorkflow.records
        .map((record) => record.generatedOrthophoto?.targetSource?.controlWaypoint)
        .filter((waypoint): waypoint is string => Boolean(waypoint)),
    ]).size;
    controlPhotoStatus.textContent = `${preparedCount} of ${points.length} controls have a photo proposal or imported photo.${result.warnings.length ? ` ${result.warnings.length} warning${result.warnings.length === 1 ? '' : 's'}: ${result.warnings.join(' | ')}` : ' Choose true or false for each available TP.'}`;
    setStatus(
      result.warnings.length
        ? `Control-photo lookup completed with ${result.warnings.length} warning${result.warnings.length === 1 ? '' : 's'}. Missing controls can be retried individually.`
        : `${requestedNames.join(', ')} photo option${requestedNames.length === 1 ? ' is' : 's are'} ready.`,
      result.warnings.length ? 'warning' : 'success'
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    controlPhotoProgress.dataset.state = 'error';
    controlPhotoProgressPhase.textContent = 'Discovery stopped';
    controlPhotoStatus.textContent = message;
    setStatus(`Error: ${message}`, 'error');
  } finally {
    osmDiscoveryController = null;
    if (osmDiscoveryBusy) setOsmDiscoveryBusy(false);
    if (!controlPhotoReview.hidden) renderControlPhotoReview();
  }
}

findControlPhotoOptions.addEventListener('click', async () => {
  try {
    const points = parseWaypoints(waypointTextarea.value);
    const routeKey = JSON.stringify(points);
    const availableWaypoints = new Set([
      ...controlPhotoProposals.map(({ waypoint }) => waypoint[0]),
      ...photoWorkflow.records
        .map((record) => record.generatedOrthophoto?.targetSource?.controlWaypoint)
        .filter((waypoint): waypoint is string => Boolean(waypoint)),
    ]);
    const missingIndices = points.flatMap(([waypointName], index) =>
      availableWaypoints.has(waypointName) ? [] : [index]
    );
    const retryMissingOnly = controlPhotoRouteKey === routeKey && missingIndices.length > 0;
    await discoverControlPhotoOptions(
      points,
      retryMissingOnly ? missingIndices : points.map((_, index) => index),
      !retryMissingOnly
    );
  } catch (error) {
    setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`, 'error');
  }
});

controlPhotoList.addEventListener('click', async (event) => {
  const retry = (event.target as HTMLElement).closest<HTMLButtonElement>(
    'button[data-retry-control-waypoint]'
  );
  if (!retry || photoWorkflow.isBusy || osmDiscoveryBusy) return;
  const waypoint = retry.dataset.retryControlWaypoint;
  if (!waypoint) return;
  try {
    const points = parseWaypoints(waypointTextarea.value);
    const waypointIndex = points.findIndex(([name]) => name === waypoint);
    if (waypointIndex < 0) throw new Error(`${waypoint} is no longer part of the route.`);
    retry.disabled = true;
    retry.textContent = `Retrying ${waypoint}…`;
    await discoverControlPhotoOptions(points, [waypointIndex], false);
  } catch (error) {
    setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`, 'error');
  }
});

controlPhotoList.addEventListener('change', (event) => {
  const choice = (event.target as HTMLElement).closest<HTMLInputElement>('input[data-control-waypoint]');
  if (!choice) return;
  const waypoint = choice.dataset.controlWaypoint;
  if (!waypoint || (choice.value !== 'true' && choice.value !== 'false')) return;
  selectedControlPhotoRoles.set(waypoint, choice.value);
  updateControlPhotoImportAction();
  updateWorkflowReadiness();
});

importControlPhotos.addEventListener('click', async () => {
  if (photoWorkflow.isBusy || osmDiscoveryBusy) return;
  try {
    const pendingProposals = controlPhotoProposals.filter(
      ({ waypoint }) => !importedControlPhoto(waypoint[0])
    );
    const targets: OrthophotoTarget[] = pendingProposals.flatMap((proposal) => {
      const waypoint = proposal.waypoint[0];
      const role = waypoint === 'SP' || waypoint === 'FP' ? 'true' : selectedControlPhotoRoles.get(waypoint);
      const selected = role === 'false' ? proposal.falseTarget : proposal.trueTarget;
      if (!role || !selected) return [];
      return [
        {
          ...selected,
          label: `CONTROL_${waypoint}_${role.toUpperCase()}`,
          source: {
            ...selected.source,
            controlRole: role,
            controlWaypoint: waypoint,
            ...(role === 'false'
              ? {
                  correctObjectLatitude: proposal.trueTarget.latitude,
                  correctObjectLongitude: proposal.trueTarget.longitude,
                }
              : {}),
          },
          control: {
            classification: role === 'false' ? 'control-false' : 'control-correct',
            waypoint,
            identifier: waypoint,
          },
        },
      ];
    });
    if (targets.length !== pendingProposals.length) {
      throw new Error('Choose an available true or false photo for every proposed control.');
    }
    if (targets.length === 0) {
      updateControlPhotoImportAction();
      return;
    }
    const result = await photoWorkflow.importOrthophotoTargets(targets, readOrthophotoModel());
    controlPhotoImportHadFailures = result.failedTargets.length > 0 || result.cancelled;
    const points = parseWaypoints(waypointTextarea.value);
    const allControlsImported = points.every(([waypoint]) => Boolean(importedControlPhoto(waypoint)));
    if (allControlsImported && !result.cancelled) {
      clearControlPhotoReview();
    } else {
      renderControlPhotoReview();
      const preparedCount = points.filter(([waypoint]) => Boolean(importedControlPhoto(waypoint))).length;
      controlPhotoStatus.textContent = `${preparedCount} of ${points.length} control photos imported. ${result.failedTargets.length ? `${result.failedTargets.length} failed crop${result.failedTargets.length === 1 ? '' : 's'} can be retried without reimporting successful photos.` : 'Run discovery again for the missing controls.'}`;
    }
  } catch (error) {
    setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`, 'error');
  }
});

loadWaypointOrthophotos.addEventListener('click', async () => {
  if (photoWorkflow.isBusy) return;
  try {
    const points = parseWaypoints(waypointTextarea.value);
    photoWorkflow.analyze(points);
    const loadedLabels = new Set(
      photoWorkflow.records
        .map((record) => record.generatedOrthophoto?.targetLabel.toLocaleUpperCase())
        .filter((label): label is string => Boolean(label))
    );
    const targets = parseOrthophotoTargets(waypointTextarea.value).filter(
      (target) => !loadedLabels.has(target.label.toLocaleUpperCase())
    );
    if (targets.length === 0) {
      throw new Error('No unloaded PHOTO_name,latitude,longitude rows were found.');
    }
    await photoWorkflow.importOrthophotoTargets(targets, readOrthophotoModel());
  } catch (error) {
    setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`, 'error');
  }
});

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
  croppedPreviewImage.hidden = true;
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
    L ??= await import('leaflet');
    await import('leaflet/dist/leaflet.css');
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
  osmConsentRow.hidden = isPdf;
  if (!isPdf) {
    updateMapStatus(
      osmThirdPartyConsent.checked
        ? 'Interactive OpenStreetMap view active; third-party tiles receive the displayed route area.'
        : 'OpenStreetMap requires consent to third-party tile requests. Choose a bundled map for local-only processing.',
      !osmThirdPartyConsent.checked
    );
  } else {
    updateMapStatus(`Using preset map: ${preset.label}`);
  }
}

async function ensurePresetBuffer(mapKey, options: { forceReload?: boolean; signal?: AbortSignal } = {}) {
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
  const response = await fetch(encodeURI(preset.url), { signal: options.signal });
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
  croppedPreviewImage.hidden = true;
  setArtifactState('preview', 'processing', 'rendering bounded preview');
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
    await waitForAbortSignal(croppedPreviewImage.decode(), controller.signal);
    croppedPreviewImage.hidden = false;
    croppedPreviewImage.alt = 'Preview of the true-scale judge solution map';
    croppedPreviewLink.href = previewUrl;
    croppedPreviewLink.setAttribute('aria-disabled', 'false');
    croppedPreviewMessage.textContent =
      'High-resolution judge-solution preview generated from native-detail map tiles. Use the three PDFs for print.';
    setArtifactState('preview', 'ok', 'ready');
    return true;
  } catch (error) {
    console.error('Could not render cropped map preview', error);
    if (croppedPreviewController !== controller) return false;
    croppedPreviewImage.removeAttribute('src');
    croppedPreviewImage.hidden = true;
    croppedPreviewLink.removeAttribute('href');
    croppedPreviewLink.setAttribute('aria-disabled', 'true');
    croppedPreviewMessage.classList.add('danger-text');
    croppedPreviewMessage.textContent = `Preview could not be rendered: ${error instanceof Error ? error.message : String(error)}. The cropped PDF is still available.`;
    setArtifactState(
      'preview',
      controller.signal.aborted ? 'cancelled' : 'failed',
      controller.signal.aborted ? 'cancelled' : 'failed; cropped PDF remains available'
    );
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

function isPhotoCoordinateName(name: string): boolean {
  return /^PHOTO_[A-Z0-9][A-Z0-9_-]{0,30}$/i.test(name);
}

function parseCoordinateRows(raw: string): Array<[string, number, number]> {
  return raw
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
}

function parseWaypoints(raw: string): Array<[string, number, number]> {
  const points = parseCoordinateRows(raw).filter(([name]) => !isPhotoCoordinateName(name));
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

function positiveInputValue(input: HTMLInputElement, label: string): number {
  const value = Number(input.value);
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${label} must be a positive number.`);
  return value;
}

function routeSetupProblem(): string | null {
  try {
    parseWaypoints(waypointTextarea.value);
    parseSpeed(selectedSpeedText());
    positiveInputValue(takeoffBufferInput, 'Takeoff-to-SP time');
    positiveInputValue(minuteIntervalInput, 'Minute-marker interval');
    return null;
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
}

function setFieldValidation(
  input: HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement,
  message: HTMLElement,
  validate: () => void
): string | null {
  try {
    validate();
    message.textContent = '';
    input.removeAttribute('aria-invalid');
    return null;
  } catch (error) {
    const problem = error instanceof Error ? error.message : String(error);
    message.textContent = problem;
    input.setAttribute('aria-invalid', 'true');
    return problem;
  }
}

function readinessTone(target: HTMLElement, tone: 'ok' | 'warning' | 'fail'): void {
  target.parentElement?.setAttribute('data-tone', tone);
}

function appendFindingGroup(
  parent: DocumentFragment,
  title: string,
  tone: 'blocking' | 'violation' | 'warning' | 'manual' | 'info',
  findings: string[]
): void {
  const section = document.createElement('details');
  section.className = 'finding-group';
  section.dataset.tone = tone;
  section.open = tone === 'blocking' || tone === 'violation';
  const summary = document.createElement('summary');
  summary.textContent = `${title} (${findings.length})`;
  const list = document.createElement('ul');
  const entries = findings.length ? findings : ['None.'];
  for (const finding of entries) {
    const item = document.createElement('li');
    item.textContent = finding;
    list.appendChild(item);
  }
  section.append(summary, list);
  parent.appendChild(section);
}

function updateWorkflowReadiness(): void {
  const blocking: string[] = [];
  const violations: string[] = [];
  const warnings: string[] = [];
  const manual: string[] = [];
  const information: string[] = [];
  let points: Array<[string, number, number]> = [];
  let routeLegCount = 0;
  let coveredLegCount = 0;
  let beforeCount = 0;
  let afterCount = 0;
  let routeViolationCount = 0;
  let routeManualCount = 0;
  const fieldProblems = [
    setFieldValidation(waypointTextarea, routeSetupError, () => {
      parseWaypoints(waypointTextarea.value);
    }),
    setFieldValidation(speedInput, speedError, () => {
      parseSpeed(selectedSpeedText());
    }),
    setFieldValidation(takeoffBufferInput, takeoffBufferError, () => {
      positiveInputValue(takeoffBufferInput, 'Takeoff-to-SP time');
    }),
    setFieldValidation(minuteIntervalInput, minuteIntervalError, () => {
      positiveInputValue(minuteIntervalInput, 'Minute-marker interval');
    }),
  ];
  const setupProblem = fieldProblems.find((problem) => problem !== null) ?? null;

  if (setupProblem) {
    blocking.push(setupProblem);
    routeControlSummary.textContent = 'Invalid route';
    routeLegSummary.textContent = '—';
    routeDistanceSummary.textContent = '—';
    routeDurationSummary.textContent = '—';
    readinessRoute.textContent = 'Needs correction';
    readinessTone(readinessRoute, 'fail');
  } else {
    points = parseWaypoints(waypointTextarea.value);
    const route = buildRoute(points);
    const speed = parseSpeed(selectedSpeedText());
    const takeoffMinutes = positiveInputValue(takeoffBufferInput, 'Takeoff-to-SP time');
    const routeDuration = takeoffMinutes + route.totalDistance / (speed.metersPerSecond * 60);
    const tpCount = points.filter(([name]) => /^TP\d+$/i.test(name.trim())).length;
    routeLegCount = route.legs.length;
    routeControlSummary.textContent = `SP · ${tpCount} TP · FP`;
    routeLegSummary.textContent = String(routeLegCount);
    routeDistanceSummary.textContent = `${metersToNauticalMiles(route.totalDistance).toFixed(1)} NM`;
    routeDurationSummary.textContent = `${routeDuration.toFixed(1)} min`;
    readinessRoute.textContent = `${points.length} controls · ${routeLegCount} legs`;
    readinessTone(readinessRoute, 'ok');
    const preset = getPreset();
    const compliance = evaluateRouteCompliance(route, points, speed, preset?.scaleDenominator ?? null);
    routeViolationCount = compliance.violations.length;
    routeManualCount = compliance.manualChecks.length;
    violations.push(
      ...compliance.violations.map((finding) => `${finding.rule} · ${finding.title}: ${finding.message}`)
    );
    manual.push(...compliance.manualChecks.map((finding) => `${finding.rule}: ${finding.message}`));
    information.push(
      `${compliance.checks.length - compliance.violations.length} automated route checks passed.`
    );
  }

  const enroutePhotos = photoWorkflow.records.filter((record) => record.classification === 'enroute');
  const pendingOsmTargets = selectedOsmCandidates();
  const plannedEnrouteCount = enroutePhotos.length + pendingOsmTargets.length;
  const controlPhotos = photoWorkflow.records.filter(
    (record) => record.classification === 'control-correct' || record.classification === 'control-false'
  );
  const coveredLegs = new Set(
    enroutePhotos
      .map((record) => record.analysis?.legIndex)
      .filter((legIndex): legIndex is number => legIndex !== undefined)
  );
  coveredLegCount = coveredLegs.size;
  const plannedCoveredLegs = new Set([
    ...coveredLegs,
    ...pendingOsmTargets.map((candidate) => candidate.routeLegIndex),
  ]);
  const splitAfterM = (() => {
    try {
      return photoWorkflow.handoutOptions.splitAfterM;
    } catch {
      return null;
    }
  })();
  const positionedEnroute = enroutePhotos.filter((record) => record.analysis !== null);
  beforeCount = positionedEnroute.filter(
    (record) => splitAfterM !== null && (record.analysis?.alongRouteM ?? 0) <= splitAfterM
  ).length;
  afterCount = positionedEnroute.filter(
    (record) => splitAfterM !== null && (record.analysis?.alongRouteM ?? 0) > splitAfterM
  ).length;
  const pendingBeforeCount = pendingOsmTargets.filter(
    (candidate) => splitAfterM !== null && (candidate.alongRouteM ?? 0) <= splitAfterM
  ).length;
  const pendingAfterCount = pendingOsmTargets.filter(
    (candidate) => splitAfterM !== null && (candidate.alongRouteM ?? 0) > splitAfterM
  ).length;
  const expectedControlNames = new Set(points.map(([name]) => name.toUpperCase()));
  const importedControlNames = new Set(
    controlPhotos.map((record) => record.linkedWaypoint?.toUpperCase()).filter(Boolean)
  );
  const importedControlCount = [...expectedControlNames].filter((name) =>
    importedControlNames.has(name)
  ).length;
  const proposedControlCount = [...expectedControlNames].filter((name) =>
    selectedControlPhotoRoles.has(name)
  ).length;

  selectedPhotoCount.textContent = String(plannedEnrouteCount);
  photoLegCoverage.textContent = `${plannedCoveredLegs.size} / ${routeLegCount}`;
  photoSplitBalance.textContent = `${beforeCount + pendingBeforeCount} / ${afterCount + pendingAfterCount}`;
  competitionPreparationBadge.textContent = `${enroutePhotos.length} competition photo${enroutePhotos.length === 1 ? '' : 's'}`;
  controlPreparationBadge.textContent =
    importedControlCount === expectedControlNames.size && expectedControlNames.size > 0
      ? `${importedControlCount} imported`
      : proposedControlCount > 0
        ? `${proposedControlCount} of ${expectedControlNames.size} choices ready`
        : 'Not prepared';
  const coverageMessages: string[] = [];
  if (enroutePhotos.length > 12) coverageMessages.push(`${enroutePhotos.length} exceeds the 12-photo limit.`);
  if (enroutePhotos.length > 0 && coveredLegCount < routeLegCount) {
    coverageMessages.push(`${routeLegCount - coveredLegCount} route leg(s) have no competition photo.`);
  }
  if (enroutePhotos.length > 0 && (beforeCount === 0 || afterCount === 0)) {
    coverageMessages.push('The selected split must leave photos in both route parts.');
  }
  photoCoverageWarning.textContent = coverageMessages.join(' ');

  readinessCount.textContent = `${enroutePhotos.length} / 12`;
  readinessTone(readinessCount, enroutePhotos.length > 12 ? 'fail' : 'ok');
  readinessCoverage.textContent = `${coveredLegCount} / ${routeLegCount} legs`;
  readinessTone(
    readinessCoverage,
    enroutePhotos.length > 0 && coveredLegCount < routeLegCount ? 'warning' : 'ok'
  );
  readinessSplit.textContent = `${beforeCount} before / ${afterCount} after`;
  readinessTone(readinessSplit, enroutePhotos.length > 0 && (!beforeCount || !afterCount) ? 'warning' : 'ok');
  readinessControls.textContent = `${importedControlCount} / ${expectedControlNames.size}`;
  readinessTone(
    readinessControls,
    importedControlCount === expectedControlNames.size && expectedControlNames.size > 0 ? 'ok' : 'warning'
  );

  for (const record of photoWorkflow.records) {
    if (record.importError) blocking.push(`${record.fileName}: ${record.importError}`);
  }
  for (const finding of photoWorkflow.compliance.findings) {
    if (finding.severity === 'pass') continue;
    const text = `${finding.rule} · ${finding.affected}: ${finding.measured}; permitted ${finding.permitted}.`;
    if (finding.severity === 'violation') violations.push(text);
    else if (findingPresentation(finding) === 'primary') warnings.push(text);
    else manual.push(text);
  }
  const acceptedExceptions = photoWorkflow.records.filter((record) => record.exceptionAccepted).length;
  const automatedViolationCount = routeViolationCount + photoWorkflow.compliance.violationCount;
  const manualCount = routeManualCount + manual.length;
  readinessRules.textContent = automatedViolationCount
    ? `${automatedViolationCount} violation(s)`
    : 'No violations';
  readinessTone(readinessRules, automatedViolationCount ? 'fail' : 'ok');
  readinessManual.textContent = `${manualCount} to review`;
  readinessTone(readinessManual, manualCount ? 'warning' : 'ok');
  readinessExceptions.textContent = String(acceptedExceptions);
  readinessTone(readinessExceptions, acceptedExceptions ? 'warning' : 'ok');
  routeOnlyNote.hidden = photoWorkflow.records.length > 0;

  const fragment = document.createDocumentFragment();
  appendFindingGroup(fragment, 'Blocking problems', 'blocking', blocking);
  appendFindingGroup(fragment, 'Against-rules findings', 'violation', violations);
  appendFindingGroup(fragment, 'Warnings', 'warning', warnings);
  appendFindingGroup(fragment, 'Manual review', 'manual', manual);
  appendFindingGroup(fragment, 'Information', 'info', information);
  workflowFindingGroups.replaceChildren(fragment);

  guidedWorkflow?.setStepSummary(
    1,
    setupProblem ? 'Route needs correction' : `${routeLegCount} legs · ${routeDistanceSummary.textContent}`
  );
  guidedWorkflow?.setStepSummary(
    2,
    `${importedControlCount}/${expectedControlNames.size} controls · ${enroutePhotos.length} competition photos`
  );
  guidedWorkflow?.setStepSummary(
    3,
    blocking.length || violations.length
      ? `${blocking.length + violations.length} issue(s) need attention`
      : `${manualCount} manual check(s) remain`
  );
}

function parseOrthophotoTargets(raw: string): OrthophotoTarget[] {
  const targets = parseCoordinateRows(raw)
    .filter(([name]) => isPhotoCoordinateName(name))
    .map(([name, latitude, longitude]) => ({
      label: name.slice('PHOTO_'.length),
      latitude,
      longitude,
    }));
  const seen = new Set<string>();
  for (const target of targets) {
    const normalized = target.label.toLocaleUpperCase();
    if (seen.has(normalized)) throw new Error(`PHOTO_ names must be unique: ${target.label}`);
    seen.add(normalized);
  }
  return targets;
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
async function generate(options: GenerateOptions = {}): Promise<GeneratedMapPair | null> {
  const mapsOnly = options.mapsOnly === true;
  if (generationController) {
    setStatus('Generation is already running. Cancel it before starting another run.', 'warning');
    return null;
  }
  if (photoWorkflow.isBusy) {
    setStatus('Wait for the current photo import to finish before generating.', 'warning');
    return null;
  }
  const setupProblem = routeSetupProblem();
  if (setupProblem) {
    updateWorkflowReadiness();
    guidedWorkflow?.activate(1);
    setStatus(`Error: ${setupProblem}`, 'error');
    return null;
  }
  if (!mapsOnly) clearSpeedSetDownload();
  guidedWorkflow?.markComplete(3);
  guidedWorkflow?.activate(4);
  const controller = new AbortController();
  generationController = controller;
  const runMapKey = selectedMapKey;
  const artifactFailures: string[] = [];
  let generatedMaps: GeneratedMapPair | null = null;
  const originalButtonHtml = generateBtn.innerHTML;
  try {
    setGenerationBusy(true);
    generateBtn.classList.add('is-loading');
    setStatus('Processing...');
    resetArtifactStates();
    setArtifactState('map', 'processing', 'preparing route map');
    setArtifactState('overlay', 'processing', 'preparing overlay');
    if (!mapsOnly) {
      clearSharedArtifactBytes();
      resultsContent.hidden = true;
      if (resultsPlaceholder && !hasGeneratedOnce) resultsPlaceholder.hidden = false;
      clearDownloadUrl('pdf', downloadPdfLink);
      clearDownloadUrl('overlay', downloadOverlayLink);
      clearDownloadUrl('cropped', downloadCroppedLink);
      clearDownloadUrl('summary', downloadSummaryLink);
      clearDownloadUrl('photoAnalysis', downloadPhotoAnalysisLink);
      clearDownloadUrl('photoKey', downloadPhotoKeyLink);
      clearDownloadUrl('photoHandout', downloadPhotoHandoutLink);
      clearDownloadUrl('competitorPhotoHandout', downloadCompetitorPhotoHandoutLink);
      downloadOverlayLink.style.display = 'none';
      downloadCroppedLink.style.display = 'none';
      downloadSummaryLink.style.display = 'none';
      downloadPhotoAnalysisLink.style.display = 'none';
      downloadPhotoKeyLink.style.display = 'none';
      downloadPhotoHandoutLink.style.display = 'none';
      downloadCompetitorPhotoHandoutLink.style.display = 'none';
      clearCroppedPreview();
      if (outputsSection) outputsSection.hidden = true;
      if (summarySection) summarySection.hidden = true;
      routeComplianceSection.hidden = true;
      if (osmMapContainer) osmMapContainer.hidden = true;
      syncResultsVisibility();
    }

    const mapConfig = getPreset(runMapKey);
    if (!mapConfig) {
      throw new Error(`Unknown map preset: ${runMapKey}`);
    }

    const speed = parseSpeed(options.speedText ?? selectedSpeedText());
    const minuteInterval = parseNumericInput(minuteIntervalInput, DEFAULT_MINUTE_INTERVAL);
    const takeoffToSp = parseNumericInput(takeoffBufferInput, DEFAULT_TAKEOFF_BUFFER);
    const points = parseWaypoints(waypointTextarea.value);

    const route = buildRoute(points);
    const compliance = evaluateRouteCompliance(route, points, speed, mapConfig.scaleDenominator);
    const photoCompliance = photoWorkflow.analyze(points);
    const styleConfig = getOverlayStyleConfig();
    const photoLayerOptions = { ...photoWorkflow.layerOptions };
    const handoutOptions = { ...photoWorkflow.handoutOptions };
    const generationPhotos = [...photoWorkflow.records];
    const judgePhotos = generationPhotos.filter(isPhotoAcceptedForJudge);
    const positionedJudgePhotos = judgePhotos.filter((photo) => photo.analysis !== null);
    const judgePhotoCompliance = evaluatePhotoCompliance(judgePhotos, points);
    controller.signal.throwIfAborted();
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
      const [{ degrees, PDFDocument, PrintScaling, rgb, StandardFonts }, { default: fontkit }] =
        await Promise.all([import('pdf-lib'), import('@pdf-lib/fontkit')]);
      const mapBytes = await ensurePresetBuffer(runMapKey, { signal: controller.signal });
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

      const fontBytes = await loadAssetBytes(notoSansBoldUrl, 'the PDF label font', controller.signal);
      const overlayFontBold = await overlayDoc.embedFont(fontBytes, { subset: true });
      const baseFontBold = await pdfDoc.embedFont(fontBytes, { subset: true });
      const drawTargets = [
        { page: overlayPage, fontBold: overlayFontBold },
        { page, fontBold: baseFontBold },
      ];
      const solutionTargets = [{ page, fontBold: baseFontBold }];

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

      // Place route headings before judge-only photo labels so both map editions
      // share identical route geometry while photo labels avoid route text.
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

      // Match the historical Python handout: every accepted photo solution is a
      // single violet tick perpendicular to its assigned route leg, with a label.
      const photoColor = rgb(0.45, 0.12, 0.66);
      const photoTickHalf = Math.max(5, 7 * scaleAvg);
      const photoLineWidth = Math.max(1.2, 2.6 * scaleAvg);
      for (const photo of judgePhotos) {
        const routeControlPhoto = isRouteControlPhoto(photo);
        const latitude = photo.metadata.latitude.value;
        const longitude = photo.metadata.longitude.value;
        const cameraExact =
          latitude === null || longitude === null ? null : projectToPdf(latitude, longitude);
        const projectedCamera = photo.analysis
          ? projectToPdf(photo.analysis.closestLatitude, photo.analysis.closestLongitude)
          : null;
        if (
          !routeControlPhoto &&
          photoLayerOptions.connectors &&
          cameraExact &&
          projectedCamera &&
          Math.hypot(cameraExact[0] - projectedCamera[0], cameraExact[1] - projectedCamera[1]) > 1
        ) {
          solutionTargets.forEach((target) => {
            target.page.drawLine({
              start: { x: cameraExact[0], y: cameraExact[1] },
              end: { x: projectedCamera[0], y: projectedCamera[1] },
              thickness: Math.max(0.6, scaleAvg),
              color: photoColor,
              opacity: 0.65,
              dashArray: [3, 3],
            });
          });
        }
        if (photoLayerOptions.exactDots && cameraExact) {
          solutionTargets.forEach((target) => {
            target.page.drawCircle({
              x: cameraExact[0],
              y: cameraExact[1],
              size: Math.max(2.3, 3 * scaleAvg),
              color: photoColor,
              borderColor: rgb(1, 1, 1),
              borderWidth: 0.8,
            });
          });
        }
        if (photoLayerOptions.projectedMarkers && projectedCamera && photo.analysis) {
          const legStart = projected[photo.analysis.legIndex]?.pdf;
          const legEnd = projected[photo.analysis.legIndex + 1]?.pdf;
          if (!legStart || !legEnd) continue;
          const legLength = Math.hypot(legEnd[0] - legStart[0], legEnd[1] - legStart[1]);
          if (legLength <= 1e-6) continue;
          const perpendicular = [
            -(legEnd[1] - legStart[1]) / legLength,
            (legEnd[0] - legStart[0]) / legLength,
          ];
          const tickStart = [
            projectedCamera[0] - perpendicular[0] * photoTickHalf,
            projectedCamera[1] - perpendicular[1] * photoTickHalf,
          ];
          const tickEnd = [
            projectedCamera[0] + perpendicular[0] * photoTickHalf,
            projectedCamera[1] + perpendicular[1] * photoTickHalf,
          ];
          if (!routeControlPhoto) {
            solutionTargets.forEach((target) => {
              target.page.drawLine({
                start: { x: tickStart[0], y: tickStart[1] },
                end: { x: tickEnd[0], y: tickEnd[1] },
                thickness: photoLineWidth,
                color: photoColor,
              });
            });
          }
          const label = photo.identifier || '?';
          const fontSize = Math.max(5, 12 * scaleAvg);
          const width = overlayFontBold.widthOfTextAtSize(label, fontSize);
          const labelOffset = routeControlPhoto
            ? Math.max(photoTickHalf * 4, ds.tpRadius + fontSize)
            : photoTickHalf * 4;
          const adjusted = adjustLabelPosition(
            projectedCamera[0] + perpendicular[0] * labelOffset,
            projectedCamera[1] + perpendicular[1] * labelOffset,
            perpendicular[0],
            perpendicular[1],
            width,
            fontSize,
            0,
            placedLabelBoxes,
            { allowNegative: false }
          );
          registerLabelBox(adjusted.box);
          solutionTargets.forEach((target) => {
            target.page.drawText(label, {
              x: adjusted.x,
              y: adjusted.y,
              size: fontSize,
              font: target.fontBold,
              color: photoColor,
            });
          });
        }
        if (
          !routeControlPhoto &&
          photoLayerOptions.headingArrows &&
          cameraExact &&
          latitude !== null &&
          longitude !== null &&
          photo.metadata.headingDeg.value !== null &&
          photo.metadata.headingReference.value === 'true' &&
          photo.metadata.headingReference.reliable
        ) {
          const radians = (photo.metadata.headingDeg.value * Math.PI) / 180;
          const distanceM = 300;
          const headingLat = latitude + ((distanceM * Math.cos(radians)) / 6371008.8) * (180 / Math.PI);
          const headingLon =
            longitude +
            ((distanceM * Math.sin(radians)) / (6371008.8 * Math.cos((latitude * Math.PI) / 180))) *
              (180 / Math.PI);
          const endpoint = projectToPdf(headingLat, headingLon);
          solutionTargets.forEach((target) => {
            target.page.drawLine({
              start: { x: cameraExact[0], y: cameraExact[1] },
              end: { x: endpoint[0], y: endpoint[1] },
              thickness: Math.max(0.8, 1.2 * scaleAvg),
              color: photoColor,
            });
          });
        }
        if (photoLayerOptions.includeInCrop) {
          const cropPoints = [projectedCamera];
          if (photoLayerOptions.exactDots) cropPoints.push(cameraExact);
          cropPoints.filter(Boolean).forEach(([x, y]) => {
            expandBounds(x - photoTickHalf * 2, y - photoTickHalf * 2);
            expandBounds(x + photoTickHalf * 2, y + photoTickHalf * 2);
          });
        }
      }
      if (photoLayerOptions.legend && judgePhotos.length > 0) {
        const legendWidth = 120;
        const legendHeight = 26;
        const legendLeft = Math.min(Math.max(0, bounds.minX), Math.max(0, pageWidth - legendWidth));
        const belowRoute = bounds.minY - legendHeight - 6;
        const legendBottom =
          belowRoute >= 0
            ? belowRoute
            : Math.min(Math.max(0, pageHeight - legendHeight), Math.max(0, bounds.maxY + 6));
        const legendX = legendLeft + 8;
        const legendY = legendBottom + legendHeight / 2;
        solutionTargets.forEach((target) => {
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
          target.page.drawLine({
            start: { x: legendX, y: legendY - 6 },
            end: { x: legendX, y: legendY + 6 },
            thickness: 2,
            color: photoColor,
          });
          target.page.drawText('Accepted photo solution', {
            x: legendX + 10,
            y: legendY - 3,
            size: 7,
            font: target.fontBold,
            color: photoColor,
          });
        });
        if (photoLayerOptions.includeInCrop) {
          expandBounds(legendLeft, legendBottom);
          expandBounds(legendLeft + legendWidth, legendBottom + legendHeight);
        }
      }

      const routeOverlayBytes = await overlayDoc.save();
      const markedBytes = await pdfDoc.save();
      const competitorDoc = await PDFDocument.load(mapBytes);
      const [competitorPage] = competitorDoc.getPages();
      const [embeddedRouteOverlay] = await competitorDoc.embedPdf(routeOverlayBytes, [0]);
      competitorPage.drawPage(embeddedRouteOverlay, {
        x: 0,
        y: 0,
        width: pageWidth,
        height: pageHeight,
      });
      const competitorBytes = await competitorDoc.save();
      controller.signal.throwIfAborted();
      let judgeCroppedBytes = null;
      let competitorCroppedBytes = null;
      let emptyCroppedBytes = null;
      let previewOptions = null;
      if (Object.values(bounds).every(Number.isFinite)) {
        const printScale = mapConfig.printScale ?? 1;
        const m = (10 * MM_TO_PT) / printScale;
        const [minX, minY, maxX, maxY] = [
          Math.max(0, bounds.minX - m),
          Math.max(0, bounds.minY - m),
          Math.min(pageWidth, bounds.maxX + m),
          Math.min(pageHeight, bounds.maxY + m),
        ];
        if (maxX > minX + 1 && maxY > minY + 1) {
          const sourceContentWidth = maxX - minX;
          const sourceContentHeight = maxY - minY;
          const contentWidth = sourceContentWidth * printScale;
          const contentHeight = sourceContentHeight * printScale;
          const targetPage = chooseTrueScaleCropPage(
            contentWidth,
            contentHeight,
            [210 * MM_TO_PT, 297 * MM_TO_PT],
            [297 * MM_TO_PT, 420 * MM_TO_PT]
          );
          if (!targetPage) {
            const requiredWidthMm = contentWidth / MM_TO_PT;
            const requiredHeightMm = contentHeight / MM_TO_PT;
            throw new Error(
              `The true-scale route requires ${requiredWidthMm.toFixed(0)} × ${requiredHeightMm.toFixed(0)} mm, which exceeds A3. Shorten or reshape the route; the map will not be shrunk or tiled.`
            );
          }
          const cropPdf = async (sourceBytes) => {
            const cropDoc = await PDFDocument.create();
            const [embedded] = await cropDoc.embedPdf(sourceBytes, [0]);
            const cropPage = cropDoc.addPage([targetPage.width, targetPage.height]);
            cropPage.drawPage(embedded, {
              x: -minX * printScale + (targetPage.width - contentWidth) / 2,
              y: -minY * printScale + (targetPage.height - contentHeight) / 2,
              width: pageWidth * printScale,
              height: pageHeight * printScale,
            });
            const verificationFont = await cropDoc.embedFont(StandardFonts.Helvetica);
            const checkStartX = 10 * MM_TO_PT;
            const checkEndX = checkStartX + 100 * MM_TO_PT;
            const checkY = 4 * MM_TO_PT;
            cropPage.drawRectangle({
              x: 8 * MM_TO_PT,
              y: 2 * MM_TO_PT,
              width: 106 * MM_TO_PT,
              height: 8 * MM_TO_PT,
              color: rgb(1, 1, 1),
              opacity: 0.92,
            });
            cropPage.drawLine({
              start: { x: checkStartX, y: checkY },
              end: { x: checkEndX, y: checkY },
              thickness: 0.8,
              color: rgb(0, 0, 0),
            });
            for (const x of [checkStartX, checkEndX]) {
              cropPage.drawLine({
                start: { x, y: checkY - 1.5 * MM_TO_PT },
                end: { x, y: checkY + 1.5 * MM_TO_PT },
                thickness: 0.8,
                color: rgb(0, 0, 0),
              });
            }
            cropPage.drawText(
              `100 mm print check - Actual size / 100% - 1:${mapConfig.scaleDenominator.toLocaleString('en-US')}`,
              {
                x: checkStartX,
                y: 6.2 * MM_TO_PT,
                size: 5.5,
                font: verificationFont,
                color: rgb(0, 0, 0),
              }
            );
            const viewerPreferences = cropDoc.catalog.getOrCreateViewerPreferences();
            viewerPreferences.setPrintScaling(PrintScaling.None);
            viewerPreferences.setPickTrayByPDFSize(true);
            return cropDoc.save();
          };
          [judgeCroppedBytes, competitorCroppedBytes, emptyCroppedBytes] = await Promise.all([
            cropPdf(markedBytes),
            cropPdf(competitorBytes),
            cropPdf(mapBytes),
          ]);
          previewOptions = {
            imageUrl: mapConfig.previewUrl,
            tileSet: mapConfig.previewTiles,
            pageWidth,
            pageHeight,
            crop: { minX, minY, maxX, maxY },
            route: projected.map(({ name, pdf: [x, y] }) => ({ x, y, label: name })),
            photos: judgePhotos.flatMap((photo) => {
              if (!photo.analysis) return [];
              const [x, y] = projectToPdf(photo.analysis.closestLatitude, photo.analysis.closestLongitude);
              if (![x, y].every(Number.isFinite)) return [];
              const legStart = projected[photo.analysis.legIndex]?.pdf;
              const legEnd = projected[photo.analysis.legIndex + 1]?.pdf;
              if (!legStart || !legEnd) return [];
              const length = Math.hypot(legEnd[0] - legStart[0], legEnd[1] - legStart[1]);
              if (length <= 1e-6) return [];
              return [
                {
                  x,
                  y,
                  label: photo.identifier || '?',
                  color: '#7331a5',
                  tickVector: [-(legEnd[1] - legStart[1]) / length, (legEnd[0] - legStart[0]) / length],
                  showTick: !isRouteControlPhoto(photo),
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
            scaleDenominator: mapConfig.scaleDenominator,
            sourcePrintScalePercent: printScale * 100,
          };
        }
      }
      if (judgeCroppedBytes && competitorCroppedBytes && emptyCroppedBytes && previewOptions) {
        generatedMaps = { judge: judgeCroppedBytes, competitor: competitorCroppedBytes };
        setArtifactState(
          'map',
          'ok',
          `${judgePhotos.length} accepted photo(s); ${positionedJudgePhotos.length} positioned on map`
        );
        setArtifactState('overlay', 'ok', 'route-only PDF ready');
        setArtifactState(
          'crop',
          'ok',
          `${summaryCropped.format}; print at Actual size / 100%; verify the 100 mm line`
        );
        if (!mapsOnly) {
          downloadPdfLink.textContent = 'Judge Solutions (PDF)';
          downloadPdfLink.onclick = null;
          setDownloadUrl(
            'pdf',
            createPdfObjectUrl(judgeCroppedBytes),
            'judge_solution_map.pdf',
            downloadPdfLink
          );
          setDownloadUrl(
            'overlay',
            createPdfObjectUrl(competitorCroppedBytes),
            'competitor_route_map.pdf',
            downloadOverlayLink
          );
          downloadOverlayLink.style.display = 'inline-flex';
          setDownloadUrl(
            'cropped',
            createPdfObjectUrl(emptyCroppedBytes),
            'empty_map.pdf',
            downloadCroppedLink
          );
          sharedArtifactBytes.cropped = Uint8Array.from(emptyCroppedBytes);
          downloadCroppedLink.style.display = 'inline-flex';
          void renderCroppedPreview(previewOptions);
        }
      } else {
        setArtifactState('map', 'manual-review', 'no valid crop bounds');
        setArtifactState('overlay', 'manual-review', 'no valid crop bounds');
        setArtifactState('crop', 'manual-review', 'no valid crop bounds');
        setArtifactState('preview', 'manual-review', 'not available without crop bounds');
      }
    } else {
      if (!osmThirdPartyConsent.checked) {
        throw new Error('Consent to third-party OpenStreetMap tile requests or choose a bundled PDF map.');
      }
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
      const latLngs: LatLngExpression[] = points.map(([, lat, lon]) => [lat, lon] as [number, number]);
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
      const photoColors = {
        enroute: '#731fa8',
        'control-correct': '#087a40',
        'control-false': '#eb6b0b',
        'sign-task': '#0d59b8',
        reference: '#5b6166',
      };
      for (const photo of generationPhotos) {
        const latitude = photo.metadata.latitude.value;
        const longitude = photo.metadata.longitude.value;
        const cameraExact: [number, number] | null =
          latitude === null || longitude === null ? null : [latitude, longitude];
        const taskExact = effectiveTaskCoordinates(photo, points);
        const projectedTask: [number, number] | null = photo.taskAnalysis
          ? [photo.taskAnalysis.closestLatitude, photo.taskAnalysis.closestLongitude]
          : null;
        const color = photoColors[photo.classification];
        const hasViolation = photo.findings.some((finding) => finding.severity === 'violation');
        if (photoLayerOptions.connectors && taskExact && projectedTask) {
          osmLayers.push(
            L.polyline([taskExact, projectedTask], {
              color,
              weight: 1.5,
              opacity: 0.7,
              dashArray: '4 4',
            }).addTo(osmMap)
          );
        }
        if (photoLayerOptions.exactDots && cameraExact) {
          osmLayers.push(
            L.circleMarker(cameraExact, {
              radius: 4,
              color: hasViolation ? '#c90000' : '#fff',
              fillColor: color,
              fillOpacity: 1,
              weight: 2,
            })
              .bindTooltip(textNodeElement(`${photo.identifier}: exact camera position`))
              .addTo(osmMap)
          );
        }
        if (photoLayerOptions.projectedMarkers && taskExact) {
          osmLayers.push(
            L.circleMarker(taskExact, {
              radius: photo.classification === 'enroute' ? 7 : 9,
              color: hasViolation ? '#c90000' : color,
              fillColor: '#fff',
              fillOpacity: 0.9,
              weight: hasViolation ? 3 : 2,
            })
              .bindTooltip(textNodeElement(`${photo.identifier} · ${photo.classification} task position`))
              .addTo(osmMap)
          );
          osmLayers.push(
            L.marker(taskExact, {
              icon: L.divIcon({
                className: 'leaflet-marker-icon osm-photo-label-icon',
                html: `<div class="osm-photo-label" style="border-color:${color};color:${color}">${escapeHtml(photo.identifier || '?')}</div>`,
              }),
              interactive: false,
            }).addTo(osmMap)
          );
        }
        if (
          photoLayerOptions.headingArrows &&
          cameraExact &&
          latitude !== null &&
          longitude !== null &&
          photo.metadata.headingDeg.value !== null &&
          photo.metadata.headingReference.value === 'true' &&
          photo.metadata.headingReference.reliable
        ) {
          const radians = (photo.metadata.headingDeg.value * Math.PI) / 180;
          const headingLat = latitude + ((300 * Math.cos(radians)) / 6371008.8) * (180 / Math.PI);
          const headingLon =
            longitude +
            ((300 * Math.sin(radians)) / (6371008.8 * Math.cos((latitude * Math.PI) / 180))) *
              (180 / Math.PI);
          osmLayers.push(
            L.polyline([cameraExact, [headingLat, headingLon]], { color, weight: 2, opacity: 0.9 }).addTo(
              osmMap
            )
          );
        }
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
      setArtifactState('map', 'manual-review', 'interactive OSM print view');
      setArtifactState('overlay', 'manual-review', 'included in interactive view');
      setArtifactState('crop', 'manual-review', 'not available for OSM');
      setArtifactState('preview', 'manual-review', 'interactive map is the preview');
    }

    if (mapsOnly) {
      if (!generatedMaps) throw new Error('This map preset did not produce downloadable PDF maps.');
      return generatedMaps;
    }

    if (outputsSection) outputsSection.hidden = false;
    resultsContent.hidden = false;
    syncResultsVisibility();
    controller.signal.throwIfAborted();

    if (generationPhotos.length === 0) {
      setArtifactState('data', 'processing', 'summary pending; photo CSV files omitted');
      setArtifactState('handout', 'manual-review', 'omitted for route-only package');
      setArtifactState('competitorHandout', 'manual-review', 'omitted for route-only package');
    } else {
      const analysisCsv = photoAnalysisCsv(generationPhotos);
      const overlayKeyCsv = photoOverlayKeyCsv(generationPhotos);
      sharedArtifactBytes.photoAnalysis = strToU8(analysisCsv);
      sharedArtifactBytes.photoKey = strToU8(overlayKeyCsv);
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
      setArtifactState('data', 'ok', 'CSV files ready; summary pending');

      setArtifactState('handout', 'processing', 'preparing photos');
      setArtifactState('competitorHandout', 'processing', 'preparing photos');
      try {
        setStatus(
          `Preparing ${judgePhotos.length} accepted photo${judgePhotos.length === 1 ? '' : 's'} for the judge and competitor handouts...`
        );
        const handoutPhotos = [];
        for (const record of judgePhotos) {
          controller.signal.throwIfAborted();
          handoutPhotos.push({
            record,
            jpeg: await preparePhotoJpeg(record.file, record.metadata.orientation.value ?? 1, 1600),
          });
          await new Promise<void>((resolve) => setTimeout(resolve, 0));
        }
        const handoutFontBytes = await loadAssetBytes(
          notoSansBoldUrl,
          'the photo handout font',
          controller.signal
        );
        const { buildCompetitorPhotoHandout, buildPhotoHandout } = await import('./photo-handout');
        try {
          const handoutBytes = await buildPhotoHandout(
            handoutPhotos,
            judgePhotoCompliance,
            handoutFontBytes,
            handoutOptions
          );
          setDownloadUrl(
            'photoHandout',
            createPdfObjectUrl(handoutBytes),
            'judge_photo_handout.pdf',
            downloadPhotoHandoutLink
          );
          sharedArtifactBytes.photoHandout = Uint8Array.from(handoutBytes);
          downloadPhotoHandoutLink.style.display = 'inline-flex';
          setArtifactState('handout', 'ok', 'PDF ready');
        } catch (error) {
          if (controller.signal.aborted) throw error;
          const message = error instanceof Error ? error.message : String(error);
          artifactFailures.push(`judge photo handout: ${message}`);
          setArtifactState('handout', 'failed', message);
        }
        try {
          const competitorHandoutBytes = await buildCompetitorPhotoHandout(
            handoutPhotos,
            handoutFontBytes,
            handoutOptions
          );
          setDownloadUrl(
            'competitorPhotoHandout',
            createPdfObjectUrl(competitorHandoutBytes),
            'competitor_photo_handout.pdf',
            downloadCompetitorPhotoHandoutLink
          );
          sharedArtifactBytes.competitorPhotoHandout = Uint8Array.from(competitorHandoutBytes);
          downloadCompetitorPhotoHandoutLink.style.display = 'inline-flex';
          setArtifactState('competitorHandout', 'ok', 'PDF ready');
        } catch (error) {
          if (controller.signal.aborted) throw error;
          const message = error instanceof Error ? error.message : String(error);
          artifactFailures.push(`competitor photo handout: ${message}`);
          setArtifactState('competitorHandout', 'failed', message);
        }
      } catch (error) {
        if (controller.signal.aborted) throw error;
        const message = error instanceof Error ? error.message : String(error);
        artifactFailures.push(`photo handouts: ${message}`);
        setArtifactState('handout', 'failed', message);
        setArtifactState('competitorHandout', 'failed', message);
      }
    }

    const summary = {
      schemaVersion: 4,
      appVersion: APP_VERSION,
      generatedAt: new Date().toISOString(),
      speedLabel: speed.label,
      speedKnots: speed.knots,
      totalDistanceKm: route.totalDistance / 1000,
      totalDistanceNm: compliance.totalDistanceNm,
      map: {
        key: runMapKey,
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
      photos: photoSummaryJson(generationPhotos, photoCompliance),
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
    setArtifactState(
      'data',
      'ok',
      generationPhotos.length ? 'CSV and JSON ready' : 'summary JSON ready; photo CSV files omitted'
    );

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
    summaryText.textContent = `${summary.speedLabel} - ${summary.totalDistanceNm.toFixed(2)} NM (${summary.totalDistanceKm.toFixed(2)} km) course - ${mapConfig.label}${summaryCropped ? ` - ${summaryCropped.format}, print at Actual size / 100%` : ''}`;

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
    const acceptedExceptionCount = generationPhotos.filter((photo) => photo.exceptionAccepted).length;
    setStatus(
      artifactFailures.length
        ? `Generated with partial failures: ${artifactFailures.join(' | ')}. Completed downloads remain available.`
        : generatedStatus === 'against-rules'
          ? `Generated: Against the rules · ${compliance.violations.length + photoCompliance.violationCount} automated violation(s)${acceptedExceptionCount ? `; ${acceptedExceptionCount} photo exception(s) accepted by the judge` : ''}. Review the retained findings.`
          : generatedStatus === 'manual-review'
            ? `Generated: Manual review required · no automated violation was found, but ${photoCompliance.warningCount} photo/judge review item(s) remain.`
            : 'Generated: OK for automated checks. Complete the listed manual judge checks.',
      generatedStatus === 'ok' && artifactFailures.length === 0 ? 'success' : 'warning'
    );
    guidedWorkflow?.setStepSummary(
      4,
      artifactFailures.length
        ? `Generated with ${artifactFailures.length} partial failure(s)`
        : 'Package ready'
    );
    return generatedMaps;
  } catch (err) {
    console.error(err);
    const cancelled = controller.signal.aborted;
    setStatus(
      cancelled
        ? 'Generation cancelled. Any completed downloads remain available.'
        : `Error: ${err instanceof Error ? err.message : String(err)}`
    );
    guidedWorkflow?.setStepSummary(4, cancelled ? 'Generation cancelled' : 'Generation failed');
    if (!outputsSection || outputsSection.hidden) resultsContent.hidden = true;
    if (resultsPlaceholder && !hasGeneratedOnce) {
      resultsPlaceholder.hidden = false;
    }
    syncResultsVisibility();
    return null;
  } finally {
    if (generationController === controller) generationController = null;
    setGenerationBusy(false);
    generateBtn.classList.remove('is-loading');
    generateBtn.innerHTML = originalButtonHtml;
  }
}

function createSpeedEditionArchive(): {
  add: (name: string, data: Uint8Array) => void;
  finish: () => Promise<Blob>;
} {
  const chunks: ArrayBuffer[] = [];
  let resolveArchive: ((blob: Blob) => void) | null = null;
  let rejectArchive: ((error: Error) => void) | null = null;
  const completed = new Promise<Blob>((resolve, reject) => {
    resolveArchive = resolve;
    rejectArchive = reject;
  });
  const archive = new Zip((error, data, final) => {
    if (error) {
      rejectArchive?.(error);
      return;
    }
    chunks.push(Uint8Array.from(data).buffer);
    if (final) resolveArchive?.(new Blob(chunks, { type: 'application/zip' }));
  });
  return {
    add(name, data) {
      const entry = new ZipPassThrough(name);
      archive.add(entry);
      entry.push(data, true);
    },
    finish() {
      archive.end();
      return completed;
    },
  };
}

function requestedSpeedEditions(): SpeedEdition[] {
  const editions: SpeedEdition[] = SPEED_EDITION_KNOTS.map((speed) => ({
    speedText: `${speed}kt`,
    slug: `${speed}kt`,
    label: `${speed} kt`,
  }));
  const knownKnots = editions.map((edition) => parseSpeed(edition.speedText).knots);
  customSpeedSettings().forEach((custom, index) => {
    if (!custom.value) return;
    const speedText = `${custom.value}${custom.unit}`;
    const parsed = parseSpeed(speedText);
    if (knownKnots.some((knots) => Math.abs(knots - parsed.knots) < 0.01)) return;
    knownKnots.push(parsed.knots);
    const safeValue = custom.value.replace(',', '.').replace('.', '_');
    const unit = custom.unit === 'kmh' ? 'kmh' : 'kt';
    editions.push({
      speedText,
      slug: `custom${index + 1}_${safeValue}${unit}`,
      label: `Custom ${index + 1}: ${parsed.label}`,
    });
  });
  return editions;
}

function generatedArtifactBytes(key: SharedArtifactKey, label: string): Uint8Array {
  const bytes = sharedArtifactBytes[key];
  if (!bytes) throw new Error(`${label} was not generated, so the complete speed archive cannot be built.`);
  return bytes;
}

async function addSharedCompetitionFiles(
  archive: ReturnType<typeof createSpeedEditionArchive>
): Promise<void> {
  const entries: Array<[SharedArtifactKey, string, string]> = [
    ['cropped', 'empty map', 'shared/empty_map.pdf'],
  ];
  if (photoWorkflow.records.length > 0) {
    entries.push(
      ['photoHandout', 'judge photo handout', 'shared/judge_photo_handout.pdf'],
      ['competitorPhotoHandout', 'competitor photo handout', 'shared/competitor_photo_handout.pdf'],
      ['photoAnalysis', 'photo analysis', 'shared/photo_analysis.csv'],
      ['photoKey', 'photo overlay key', 'shared/photo_overlay_key.csv']
    );
  }
  for (const [key, label, path] of entries) {
    archive.add(path, generatedArtifactBytes(key, label));
  }
  const project = await createRouteProjectArtifact();
  archive.add(`shared/${project.filename}`, project.bytes);
}

async function generateSpeedEditionSet(): Promise<void> {
  if (generationController || photoWorkflow.isBusy || speedSetGenerationActive) return;
  if (!isPdfPreset()) {
    setStatus('Error: speed-edition PDFs require one of the calibrated PDF map presets.', 'error');
    return;
  }
  speedSetGenerationActive = true;
  let editions: SpeedEdition[];
  try {
    editions = requestedSpeedEditions();
  } catch (error) {
    speedSetGenerationActive = false;
    setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`, 'error');
    return;
  }
  generateSpeedSetBtn.disabled = true;
  generateBtn.disabled = true;
  saveProjectBtn.disabled = true;
  loadProjectBtn.disabled = true;
  speedSetProgress.hidden = false;
  speedSetProgressBar.max = editions.length;
  speedSetProgressBar.value = 0;
  speedSetProgress.dataset.state = 'working';
  downloadSpeedSetLink.hidden = true;
  if (speedSetObjectUrl) {
    URL.revokeObjectURL(speedSetObjectUrl);
    speedSetObjectUrl = null;
  }
  const archive = createSpeedEditionArchive();
  archive.add(
    'README.txt',
    strToU8(
      'Speed editions generated by Route Overlay Generator.\nEach speed folder contains a judge solution map and a competitor route map for the stated groundspeed.\nThe archive contains the default 50–100 kt set and configured non-duplicate custom speeds.\nThe shared folder contains all speed-independent competition files and a versioned route project that can recreate the package.\n'
    )
  );
  try {
    speedSetProgressText.textContent = 'Refreshing speed-independent competition files…';
    speedSetProgressCount.textContent = `0 of ${editions.length}`;
    setStatus('Speed editions: preparing shared competition files and route project…');
    if (!(await generate())) throw new Error('The shared competition package could not be generated.');
    await addSharedCompetitionFiles(archive);
    for (const [index, edition] of editions.entries()) {
      speedSetProgressText.textContent = `Preparing ${edition.label} judge and competitor maps…`;
      speedSetProgressCount.textContent = `${index} of ${editions.length}`;
      setStatus(`Speed editions: preparing ${edition.label} (${index + 1} of ${editions.length})…`);
      const maps = await generate({ speedText: edition.speedText, mapsOnly: true });
      if (!maps) throw new Error(`The ${edition.label} edition could not be generated.`);
      archive.add(`${edition.slug}/judge_solution_map_${edition.slug}.pdf`, maps.judge);
      archive.add(`${edition.slug}/competitor_route_map_${edition.slug}.pdf`, maps.competitor);
      speedSetProgressBar.value = index + 1;
      speedSetProgressCount.textContent = `${index + 1} of ${editions.length}`;
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    speedSetProgressText.textContent = 'Finalizing ZIP archive…';
    const blob = await archive.finish();
    speedSetObjectUrl = URL.createObjectURL(blob);
    downloadSpeedSetLink.href = speedSetObjectUrl;
    downloadSpeedSetLink.download =
      editions.length > SPEED_EDITION_KNOTS.length
        ? 'route_speed_editions_50-100kt_plus_custom.zip'
        : 'route_speed_editions_50-100kt.zip';
    downloadSpeedSetLink.hidden = false;
    speedSetProgress.dataset.state = 'complete';
    speedSetProgressText.textContent = 'All speed editions are ready';
    speedSetProgressCount.textContent = `${editions.length} of ${editions.length}`;
    setStatus(
      `Prepared judge and competitor route maps for ${editions.length} speed edition${editions.length === 1 ? '' : 's'}.`,
      'success'
    );
  } catch (error) {
    speedSetProgress.dataset.state = 'error';
    speedSetProgressText.textContent = 'Speed-edition generation stopped';
    setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`, 'error');
  } finally {
    speedSetGenerationActive = false;
    generateSpeedSetBtn.disabled = false;
    generateBtn.disabled = false;
    saveProjectBtn.disabled = false;
    loadProjectBtn.disabled = false;
  }
}

generateBtn.addEventListener('click', () => void generate());
generateSpeedSetBtn.addEventListener('click', () => void generateSpeedEditionSet());
saveProjectBtn.addEventListener('click', () => void saveRouteProject());
loadProjectBtn.addEventListener('click', () => projectFileInput.click());
projectFileInput.addEventListener('change', () => {
  const file = projectFileInput.files?.[0];
  projectFileInput.value = '';
  if (!file) return;
  void loadRouteProject(file).catch((error) => {
    setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`, 'error');
  });
});
cancelGenerationBtn.addEventListener('click', () => {
  generationController?.abort('Cancelled by the user.');
  croppedPreviewController?.abort('Cancelled by the user.');
});
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
    clearSpeedSetDownload();
    handleMapPresetChange(btn.dataset.mapKey);
    updateWorkflowReadiness();
  });
});
osmThirdPartyConsent.addEventListener('change', () => {
  updateMapInputsVisibility();
  updateWorkflowReadiness();
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
  if (speedSetObjectUrl) URL.revokeObjectURL(speedSetObjectUrl);
});

// Initial UI state setup
guidedWorkflow = new GuidedWorkflow({
  canEnter: (stage: WorkflowStage) => (stage > 1 ? routeSetupProblem() : null),
  onBlocked: (message) => {
    updateWorkflowReadiness();
    setStatus(`Error: ${message}`, 'error');
    waypointTextarea.focus();
  },
  onStageChange: (stage) => {
    if (stage === 3 && !routeSetupProblem()) {
      photoWorkflow.analyze(parseWaypoints(waypointTextarea.value));
    }
    updateWorkflowReadiness();
  },
});
for (const input of [speedInput, takeoffBufferInput, minuteIntervalInput]) {
  input.addEventListener('input', () => {
    clearSpeedSetDownload();
    updateWorkflowReadiness();
  });
}
speedInput.addEventListener('change', () => {
  const index = customSpeedIndex();
  if (index === null) return;
  if (customSpeedPanel) customSpeedPanel.open = true;
  if (!customSpeedValueInputs[index].value) customSpeedValueInputs[index].focus();
});
customSpeedValueInputs.forEach((input) => {
  input.addEventListener('input', () => {
    updateCustomSpeedOptionLabels();
    clearSpeedSetDownload();
    updateWorkflowReadiness();
  });
});
customSpeedUnitInputs.forEach((input) => {
  input.addEventListener('change', () => {
    updateCustomSpeedOptionLabels();
    clearSpeedSetDownload();
    updateWorkflowReadiness();
  });
});
handoutSplit.addEventListener('change', () => {
  clearSpeedSetDownload();
  updateWorkflowReadiness();
});
workflowShell.addEventListener('change', clearSpeedSetDownload);
document.addEventListener('photo-workflow-change', () => {
  clearSpeedSetDownload();
  updateWorkflowReadiness();
});
setStatus('');
applyCustomSpeedSettings(DEFAULT_CUSTOM_SPEEDS);
handleMapPresetChange(selectedMapKey);
photoWorkflow.analyze(parseWaypoints(waypointTextarea.value));
updateWorkflowReadiness();
