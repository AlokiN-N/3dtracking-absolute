const byId = id => document.getElementById(id);
const video = byId("video");
const chart = byId("chart");
const map = byId("map");
const controls = byId("lag-controls");
const lagInput = byId("lag-k");
const historyInput = byId("lag-history");
const methodInput = byId("lag-method");
const lockInput = byId("lag-lock");
const number = new Intl.NumberFormat("ru-RU", {maximumFractionDigits: 2});
const integer = new Intl.NumberFormat("ru-RU");
const green = "#86f0b4";
const orange = "#ff8f53";
const muted = "#a5b3c2";
const names = {multi: "Триангуляция", mono: "Одна камера", motion: "Движение и рёбра", flow: "Оптический поток", forecast: "Прогноз", lost: "Потеря"};
const pad = {left: 51, right: 16, top: 13, bottom: 23};
let data;
let frames;
let lag;
let lagRequest;
let lagTimer;
let index = 0;
let axis = 0;

function canvasContext(canvas) {
  const {width, height} = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * scale) || canvas.height !== Math.round(height * scale)) {
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = "11px ui-monospace, monospace";
  return {ctx, width, height};
}

function line(ctx, x1, y1, x2, y2, color) {
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function bounds(values) {
  if (!values.length) return [0, 1];
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  const span = Math.max(max - min, 0.1);
  return [min - .06 * span, max + .06 * span];
}

function trace(ctx, count, point, color) {
  ctx.strokeStyle = color;
  ctx.beginPath();
  let connecting = false;
  for (let i = 0; i < count; i++) {
    const coordinate = point(i);
    if (!coordinate) {
      connecting = false;
      continue;
    }
    if (connecting) ctx.lineTo(...coordinate);
    else ctx.moveTo(...coordinate);
    connecting = true;
  }
  ctx.stroke();
}

function delayedTrace(ctx, count, point) {
  ctx.save();
  ctx.setLineDash([5, 3]);
  ctx.lineWidth = 1.8;
  trace(ctx, count, point, orange);
  ctx.restore();
}

function dot(ctx, point, color, radius) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(...point, radius, 0, 2 * Math.PI);
  ctx.fill();
}

function drawMap() {
  const {ctx, width, height} = canvasContext(map);
  const valid = frames.concat(lag?.frames ?? []).filter(frame => frame.xyz[0] !== null);
  if (!valid.length) return;
  const [xmin, xmax] = bounds(valid.map(frame => frame.xyz[0] / 100));
  const [ymin, ymax] = bounds(valid.map(frame => frame.xyz[1] / 100));
  const scale = Math.min((width - 24) / (xmax - xmin), (height - 30) / (ymax - ymin));
  const xy = xyz => [width / 2 + (xyz[0] / 100 - (xmin + xmax) / 2) * scale,
                     height / 2 - (xyz[1] / 100 - (ymin + ymax) / 2) * scale];
  for (let x = 0; x < width; x += 26) line(ctx, x, 0, x, height, "#252e39");
  for (let y = 0; y < height; y += 26) line(ctx, 0, y, width, y, "#252e39");
  ctx.fillStyle = muted;
  ctx.fillText("XY · вид сверху", 9, 15);
  const point = i => frames[i].xyz[0] === null ? null : xy(frames[i].xyz);
  trace(ctx, frames.length, point, "#4e6879");
  trace(ctx, index + 1, point, green);
  if (point(index)) dot(ctx, point(index), green, 4);
  if (lag && index >= lag.k) {
    const released = index - lag.k;
    const delayedPoint = i => {
      const xyz = lag.frames[i]?.xyz;
      return xyz?.[0] != null ? xy(xyz) : null;
    };
    delayedTrace(ctx, released + 1, delayedPoint);
    if (delayedPoint(released)) dot(ctx, delayedPoint(released), orange, 3);
  }
}

function drawChart() {
  const {ctx, width, height} = canvasContext(chart);
  const value = frame => axis === "error" ? (frame.error_m ?? null) :
    (frame.xyz[axis] === null ? null : frame.xyz[axis] / 100);
  const values = frames.map(value);
  const delayedValues = lag?.frames.map(value) ?? [];
  const [min, max] = bounds(values.concat(delayedValues).filter(item => item !== null));
  const graphW = width - pad.left - pad.right;
  const graphH = height - pad.top - pad.bottom;
  const x = i => pad.left + i * graphW / Math.max(frames.length - 1, 1);
  const y = value => pad.top + (max - value) * graphH / (max - min);
  ctx.fillStyle = muted;
  for (let i = 0; i <= 4; i++) {
    const v = min + (max - min) * i / 4;
    line(ctx, pad.left, y(v), width - pad.right, y(v), "#303b49");
    ctx.textAlign = "right";
    ctx.fillText(number.format(v), pad.left - 6, y(v) + 3);
  }
  ctx.textAlign = "center";
  for (let i = 0; i <= 5; i++) {
    const frame = Math.round(i * (frames.length - 1) / 5);
    ctx.fillText(number.format(frame / data.metadata.fps) + " с", x(frame), height - 5);
  }
  trace(ctx, values.length, i => values[i] === null ? null : [x(i), y(values[i])], green);
  delayedTrace(ctx, delayedValues.length,
    i => delayedValues[i] === null ? null : [x(i), y(delayedValues[i])]);
  line(ctx, x(index), pad.top, x(index), height - pad.bottom, "#dbe5ef");
}

function paint(frame) {
  index = Math.max(0, Math.min(frame, frames.length - 1));
  const point = frames[index];
  byId("frame").textContent = `Кадр ${point.frame} · ${number.format(point.frame / data.metadata.fps)} с`;
  byId("mode").textContent = names[point.status] || point.status;
  ["x", "y", "z"].forEach((id, i) => {
    byId(id).textContent = point.xyz[i] === null ? "—" : number.format(point.xyz[i] / 100) + " м";
  });
  const cams = point.cams.length ? `Камеры ${point.cams.join(", ")}` : "Без зрительного наблюдения";
  const error = point.error_m == null ? "" : ` · ошибка ${number.format(point.error_m)} м`;
  byId("detail").textContent = cams + error;
  if (lag) {
    const delayed = lag.frames[index - lag.k];
    const position = delayed?.xyz[0] != null ? delayed.xyz.map(x => number.format(x / 100)).join(" / ") : "—";
    byId("lag-current").textContent = delayed
      ? `k = ${lag.k} · кадр ${delayed.frame} · X / Y / Z · ${position}${position === "—" ? "" : " м"}`
      : `Первая сглаженная точка появится на кадре ${lag.k}.`;
  }
  drawMap();
  drawChart();
}

function seek(frame) {
  frame = Math.max(0, Math.min(frame, frames.length - 1));
  if (Number.isFinite(video.duration)) video.currentTime = (frame + 0.1) / data.metadata.fps;
  paint(frame);
}

function sync() {
  if (!frames) return;
  const frame = Math.min(frames.length - 1, Math.floor(video.currentTime * data.metadata.fps + .0001));
  if (frame !== index) paint(frame);
}

document.querySelectorAll("[data-axis]").forEach(button => button.addEventListener("click", () => {
  axis = button.dataset.axis === "error" ? "error" : Number(button.dataset.axis);
  document.querySelectorAll("[data-axis]").forEach(item => item.setAttribute("aria-pressed", item === button));
  if (frames) drawChart();
}));
chart.addEventListener("click", event => {
  if (!frames) return;
  const rect = chart.getBoundingClientRect();
  const fraction = (event.clientX - rect.left - pad.left) / (rect.width - pad.left - pad.right);
  seek(Math.round(fraction * (frames.length - 1)));
});
chart.addEventListener("keydown", event => {
  if (!frames || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  event.preventDefault();
  seek(index + (event.key === "ArrowLeft" ? -1 : 1));
});
video.addEventListener("timeupdate", sync);
if ("requestVideoFrameCallback" in video) {
  const onFrame = () => { sync(); video.requestVideoFrameCallback(onFrame); };
  video.requestVideoFrameCallback(onFrame);
}
window.addEventListener("resize", () => { if (frames) paint(index); });

function showLag() {
  for (const id of ["lag-comparison", "lag-current", "trajectory-legend"]) {
    byId(id).hidden = false;
  }
  byId("lag-label").textContent = `k = ${lag.k} · история ${lag.history} · ${integer.format(Math.round(1000 * lag.k / data.metadata.fps))} мс по видео`;
  byId("lag-legend").textContent = `Сглаженная траектория · k = ${lag.k}`;
  const pending = lag.video_frames - lag.released_frames;
  byId("lag-summary").textContent = `Координаты выданы на ${integer.format(lag.prediction_frames)} из ${integer.format(lag.video_frames)} кадров.` +
    (pending ? ` Кадров в конце записи без выдачи — ${integer.format(pending)}.` : "");
  byId("chart-note").textContent = "График сравнивает координаты по времени оцениваемого кадра. " +
    "На карте сглаженная точка появляется после её выдачи.";
  const table = byId("lag-metrics");
  table.hidden = !lag.metrics?.fixed_lag?.n;
  if (table.hidden) return;
  const {causal, fixed_lag} = lag.metrics;
  byId("lag-metrics-caption").textContent = `Обе траектории оценены на одних и тех же ${integer.format(fixed_lag.n)} кадрах с эталоном.`;
  table.querySelectorAll("tr[data-metric]").forEach(row => {
    const {metric, unit} = row.dataset;
    [causal, fixed_lag].forEach((scores, i) => {
      row.cells[i + 1].textContent = number.format(scores[metric]) + " " + unit;
    });
  });
}

function setLagDownload(id, content, suffix, mime) {
  const link = byId(id);
  if (link.href.startsWith("blob:")) URL.revokeObjectURL(link.href);
  link.href = URL.createObjectURL(new Blob([content], {type: mime}));
  link.download = `trajectory-k${lag.k}-h${lag.history}-${lag.method}.${suffix}`;
  link.hidden = false;
}

async function updateLag(params) {
  const request = new AbortController();
  lagRequest = request;
  try {
    const response = await fetch(`api/smooth?${params}`, {signal: request.signal});
    if (!response.ok) throw new Error(response.statusText);
    const result = await response.json();
    if (request.signal.aborted) return;
    if (result.video_frames !== frames.length) throw new Error("Replay changed");
    lag = result;
    showLag();
    const csv = ["Frame;Available_At_Frame;Loc_X;Loc_Y;Loc_Z",
      ...lag.frames.map(row => [row.frame, row.available_at, ...row.xyz].join(";"))].join("\r\n") + "\r\n";
    setLagDownload("lag-csv", csv, "csv", "text/csv");
    setLagDownload("lag-json", JSON.stringify(lag), "json", "application/json");
    byId("lag-status").textContent = "";
    paint(index);
  } catch (error) {
    if (!request.signal.aborted) byId("lag-status").textContent = "Пересчёт недоступен. Измените параметры или обновите страницу.";
  }
}

function scheduleLag() {
  clearTimeout(lagTimer);
  lagRequest?.abort();
  historyInput.disabled = lockInput.checked;
  if (lockInput.checked) historyInput.value = lagInput.value;
  byId("lag-csv").hidden = true;
  byId("lag-json").hidden = true;
  if (!controls.checkValidity()) {
    byId("lag-status").textContent = "Введите целое число от 0 до 120 для каждого поля.";
    return;
  }
  const params = new URLSearchParams({k: lagInput.valueAsNumber, history: historyInput.valueAsNumber, method: methodInput.value});
  byId("lag-status").textContent = "Пересчитываем траекторию и метрики…";
  lagTimer = setTimeout(() => updateLag(params), 80);
}

controls.addEventListener("input", scheduleLag);
controls.addEventListener("submit", event => event.preventDefault());

const replay = fetch("replay.json").then(response => {
  if (!response.ok) throw new Error(response.statusText);
  return response.json();
});
replay.then(payload => {
  data = payload;
  frames = data.frames;
  controls.hidden = false;
  const {metrics, metadata} = data;
  byId("sequence-name").textContent = metadata.sequence ?? "Way1";
  byId("target-title").textContent = "Траектория в 3D · " + (metadata.object ?? "CargoTruck");
  if (metrics) {
    byId("mse").textContent = metrics.mse_m2 == null ? "—" : number.format(metrics.mse_m2) + " м²";
    byId("mae").textContent = metrics.mae_m == null ? "—" : number.format(metrics.mae_m) + " м";
    byId("rmse").textContent = metrics.rmse_m == null ? "—" : number.format(metrics.rmse_m) + " м";
    byId("max-error").textContent = metrics.max_m == null ? "—" : number.format(metrics.max_m) + " м";
    if (metrics.error_gt_16m_frames != null) {
      byId("rmse-detail").textContent = `95-й перцентиль ${number.format(metrics.p95_m)} м · ошибка > 16 м на ${integer.format(metrics.error_gt_16m_frames)} кадрах`;
    }
    byId("coverage").textContent = number.format(100 * metrics.prediction_frames / frames.length) + "%";
    byId("covered").textContent = `${integer.format(metrics.prediction_frames)} из ${integer.format(frames.length)} кадров`;
    const seen = (metrics.statuses.mono || 0) + (metrics.statuses.multi || 0) + (metrics.statuses.flow || 0) + (metrics.statuses.motion || 0);
    byId("observed").textContent = number.format(100 * seen / frames.length) + "%";
  }
  byId("run-info").textContent = `${metadata.cameras} камер · ${metadata.fps} кадр/с в исходнике · ${number.format(metadata.mean_ms)} мс на шаг трекера`;
  paint(0);
  scheduleLag();
}).catch(() => {
  byId("error").hidden = false;
  byId("error").textContent = "Не удалось загрузить результаты прогона.";
});
