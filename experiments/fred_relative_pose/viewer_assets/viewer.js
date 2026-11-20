"use strict";

const data = window.POSE_DATA;
const $ = (id) => document.getElementById(id);
const cameraCanvas = $("camera");
const orbitCanvas = $("orbit");
const colors = { pose: "#5ee8cf", target: "#ffc881", history: "#b7c8db", faint: "#5c7089" };
const photos = new Map();
const state = { example: 0, frame: 0, yaw: -0.75, pitch: 0.36, zoom: 1, drag: null, play: null };
const add = (a, b) => a.map((x, i) => x + b[i]);
const sub = (a, b) => a.map((x, i) => x - b[i]);
const mul = (a, scalar) => a.map((x) => scalar * x);
const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(...a);
const unit = (a) => mul(a, 1 / norm(a));
const clamp = (x, lo, hi) => Math.max(lo, Math.min(x, hi));
const degrees = (angle) => angle * 180 / Math.PI;
const format = (items, digits) => items.map((value) => value.toFixed(digits)).join(", ");

function sample() { return data.examples[state.example]; }
function frame() { return sample().frames[state.frame]; }

function surface(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  return { ctx, w: rect.width, h: rect.height };
}

function path(ctx, points, color, width = 1, dash = []) {
  ctx.beginPath();
  let pen = false;
  for (const point of points) {
    if (!point || !Number.isFinite(point[0]) || !Number.isFinite(point[1])) { pen = false; continue; }
    if (pen) ctx.lineTo(point[0], point[1]);
    else ctx.moveTo(point[0], point[1]);
    pen = true;
  }
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  ctx.setLineDash(dash);
  ctx.stroke();
  ctx.setLineDash([]);
}

function marker(ctx, point, color, radius = 3) {
  if (!point) return;
  ctx.beginPath();
  ctx.fillStyle = color;
  ctx.arc(point[0], point[1], radius, 0, 2 * Math.PI);
  ctx.fill();
}

function arrow(ctx, start, end, color, dash = []) {
  if (!start || !end) return;
  path(ctx, [start, end], color, 2, dash);
  const direction = sub(end, start);
  if (norm(direction) < 3) return;
  const axis = unit(direction);
  const base = sub(end, mul(axis, Math.min(10, 0.3 * norm(direction))));
  const side = mul([-axis[1], axis[0]], 4);
  path(ctx, [add(base, side), end, sub(base, side)], color, 2);
}

function pixelProject(p, item) {
  if (p[2] <= 0.6) return null;
  return item.focal.map((fx, i) => item.principal_point[i] + fx * p[i] / p[2]);
}

function cameraTransform(w, h, item, index, raw = false, rawBox = null) {
  const [fw, fh] = data.frame_size;
  let [left, top, width, height] = [0, 0, fw, fh];
  if ($("crop").checked && (!raw || rawBox)) {
    const [cx, cy, bw, bh] = raw ? rawBox : item.target_boxes_px[index];
    width = Math.min(fw, Math.max(110, bw * 4.7, bh * 4.7 * fw / fh));
    height = width * fh / fw;
    left = clamp(cx - width / 2, 0, fw - width);
    top = clamp(cy - height / 2, 0, fh - height);
  }
  const scale = Math.min(w / width, h / height);
  const x = (w - width * scale) / 2;
  const y = (h - height * scale) / 2;
  return { left, top, width, height, x, y, scale,
    map: (p) => p && [x + (p[0] - left) * scale, y + (p[1] - top) * scale] };
}

function rotorBasis(up) {
  const axis = unit(cross(up, Math.abs(up[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0]));
  return [axis, cross(up, axis)];
}

function ring(origin, a, b) {
  return Array.from({ length: 41 }, (_, i) => {
    const angle = 2 * Math.PI * i / 40;
    return add(origin, add(mul(a, Math.cos(angle)), mul(b, Math.sin(angle))));
  });
}

function drawBody(ctx, item, index, project, detailed) {
  const center = item.positions[index];
  const up = item.up_over_time[index];
  const [right, crosswise] = rotorBasis(up).map((axis) => mul(axis, 0.5));
  path(ctx, ring(center, right, crosswise).map(project), colors.pose, 1.6);
  if (detailed) {
    const height = mul(up, data.body_half_height);
    path(ctx, ring(center, right, height).map(project), "#5ee8cf70");
    path(ctx, ring(center, crosswise, height).map(project), "#5ee8cf70");
    arrow(ctx, project(center), project(add(center, item.gravity_up)), colors.history, [4, 4]);
  }
  arrow(ctx, project(center), project(add(center, mul(up, 1.15))), colors.pose);
  marker(ctx, project(center), colors.pose, 2.5);
}

function box(ctx, xywh, coords, color, dash) {
  const [cx, cy, w, h] = xywh;
  const corners = [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2],
    [cx - w / 2, cy + h / 2], [cx - w / 2, cy - h / 2]];
  path(ctx, corners.map(coords), color, 1.3, dash);
}

function imageFor(url) {
  if (!photos.has(url)) {
    const img = new Image();
    img.onload = drawCamera;
    img.onerror = drawCamera;
    if (photos.size >= 70) photos.delete(photos.keys().next().value);
    photos.set(url, img);
    img.src = url;
  }
  return photos.get(url);
}

function drawCamera() {
  const { ctx, w, h } = surface(cameraCanvas);
  const item = sample();
  const show = frame();
  const kind = $("rgb-kind").value;
  const raw = kind === "rgb";
  const match = show[kind];
  const image = match ? imageFor(match.url) : null;
  const { map, left, top, width, height, scale, x, y } = cameraTransform(w, h, item, show.index, raw, match?.box_px);
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, width * scale, height * scale);
  ctx.clip();
  let status = "No RGB frame within the matching tolerance.";
  if (image?.naturalWidth) {
    const sx = image.naturalWidth / data.frame_size[0];
    const sy = image.naturalHeight / data.frame_size[1];
    ctx.drawImage(image, left * sx, top * sy, width * sx, height * sy, x, y, width * scale, height * scale);
    const offset = match.offset_ms + data.rgb_offset_ms;
    status = `RGB frame ${offset >= 0 ? "+" : ""}${offset.toFixed(1)} ms from event time.`;
    if (data.rgb_offset_ms) status += ` Lookup shift ${data.rgb_offset_ms} ms.`;
  } else if (match) {
    status = image.complete ? "RGB image could not be loaded." : "Loading RGB image…";
  }
  if (!raw && $("overlay").checked) {
    path(ctx, item.past_boxes_px.map(map), colors.history, 1.3, [4, 5]);
    path(ctx, item.target_boxes_px.map(map), colors.target, 1.5, [4, 4]);
    path(ctx, item.projected_boxes_px.map(map), colors.pose, 1.7);
    box(ctx, item.target_boxes_px[show.index], map, colors.target, [4, 3]);
    box(ctx, item.projected_boxes_px[show.index], map, colors.pose, [2, 4]);
    drawBody(ctx, item, show.index, (p) => map(pixelProject(p, item)), false);
  } else if (raw && match?.box_px && $("overlay").checked) {
    box(ctx, match.box_px, map, colors.target, [4, 4]);
  }
  ctx.restore();
  $("rgb-status").textContent = status;
}

function makeOrbitProject(item, mode, w, h, index) {
  if (mode === "camera") {
    const coords = cameraTransform(w, h, item, index);
    return (point) => coords.map(pixelProject(point, item));
  }
  const pathCenter = mul(add(item.positions[0], item.positions.at(-1)), 0.5);
  const overview = mode === "overview";
  const target = overview ? mul(pathCenter, 0.55) : item.positions[index];
  const extent = Math.max(2.7, ...item.positions.map((p) => norm(sub(p, target))));
  const distance = (overview ? Math.max(5, 2.1 * extent) : Math.max(3.3, 1.3 * extent)) / state.zoom;
  const sy = Math.sin(state.yaw), cy = Math.cos(state.yaw);
  const sp = Math.sin(state.pitch), cp = Math.cos(state.pitch);
  const right = [cy, 0, -sy], down = [sy * sp, cp, cy * sp], forward = cross(right, down);
  const focal = Math.min(w, h) * 0.95;
  return (point) => {
    const delta = sub(point, target);
    const z = distance + dot(delta, forward);
    return z > 0.07 ? [w / 2 + focal * dot(delta, right) / z, h / 2 + focal * dot(delta, down) / z] : null;
  };
}

function drawFrustum(ctx, item, project) {
  const [w, h] = data.frame_size;
  const origin = [0, 0, 0];
  const depth = clamp(item.positions[0][2] * 0.22, 1, 14);
  const corners = [[0, 0], [w, 0], [w, h], [0, h]].map((p) => [
    depth * (p[0] - item.principal_point[0]) / item.focal[0],
    depth * (p[1] - item.principal_point[1]) / item.focal[1], depth
  ]);
  path(ctx, [...corners, corners[0]].map(project), colors.faint);
  for (const corner of corners) path(ctx, [origin, corner].map(project), colors.faint);
  const point = project(origin);
  if (point) {
    marker(ctx, point, colors.history);
    ctx.font = "12px system-ui";
    ctx.fillStyle = colors.history;
    ctx.fillText("Event camera", point[0] + 8, point[1] + 3);
  }
}

function drawGrid(ctx, item, index, project) {
  const [a, b] = rotorBasis(item.gravity_up);
  const origin = sub(item.positions[index], mul(item.gravity_up, 0.85));
  const radius = 3;
  for (let step = -radius; step <= radius; step++) {
    const alongA = add(origin, mul(a, step));
    const alongB = add(origin, mul(b, step));
    path(ctx, [sub(alongA, mul(b, radius)), add(alongA, mul(b, radius))].map(project), "#344253");
    path(ctx, [sub(alongB, mul(a, radius)), add(alongB, mul(a, radius))].map(project), "#344253");
  }
}

function drawOrbit() {
  const { ctx, w, h } = surface(orbitCanvas);
  const item = sample();
  const index = frame().index;
  const mode = $("view-mode").value;
  const project = makeOrbitProject(item, mode, w, h, index);
  if (mode === "overview") drawFrustum(ctx, item, project);
  if (mode === "focus") drawGrid(ctx, item, index, project);
  path(ctx, item.positions.map(project), "#5ee8cf78", 1.3, [5, 3]);
  path(ctx, item.positions.slice(0, index + 1).map(project), colors.pose, 1.9);
  marker(ctx, project(item.positions[0]), colors.history, 2.5);
  drawBody(ctx, item, index, project, true);
}

function stop() {
  if (state.play !== null) clearTimeout(state.play);
  state.play = null;
  $("play").textContent = "Play";
}

function updateFrame() {
  const item = sample();
  const index = frame().index;
  const up = item.up_over_time[index];
  $("time").value = state.frame;
  $("elapsed").textContent = `${Math.round(item.times[index] * 1000)} ms`;
  $("position").textContent = format(item.positions[index], 2);
  $("up").textContent = format(up, 3);
  $("tilt").textContent = `${degrees(Math.acos(clamp(dot(up, item.gravity_up), -1, 1))).toFixed(1)}°`;
  $("velocity").textContent = format(item.velocity_over_time[index], 2);
  $("acceleration").textContent = format(item.acceleration_over_time[index], 2);
  drawCamera();
  drawOrbit();
}

function resetView() {
  Object.assign(state, { yaw: -0.75, pitch: 0.36, zoom: 1 });
  drawOrbit();
}

function updateExample() {
  stop();
  state.frame = 0;
  const item = sample();
  $("time").max = item.frames.length - 1;
  $("scene").textContent = `${item.scene.replaceAll("_", " ")} · ${item.annotation_source ?? "cleaned tracks"}`;
  $("angular-velocity").textContent = format(item.angular_velocity, 2);
  $("thrust").textContent = item.specific_thrust.toFixed(2);
  $("error").textContent = `${item.ade_px.toFixed(1)} / ${item.fde_px.toFixed(1)} px`;
  $("intrinsics").textContent = `${format(item.focal, 0)} / ${format(item.principal_point, 0)}`;
  $("event-figure").hidden = !item.event_preview;
  if (item.event_preview) $("event-image").src = item.event_preview;
  $("event-label").textContent = `${data.event_window_ms} ms event cutout at anchor`;
  resetView();
  updateFrame();
}

function playNext() {
  const frames = sample().frames;
  if (state.frame >= frames.length - 1) { stop(); return; }
  const current = frames[state.frame], next = frames[state.frame + 1];
  const match = next[$("rgb-kind").value];
  if (match) imageFor(match.url);
  state.play = setTimeout(() => { state.frame++; updateFrame(); playNext(); },
    Math.max(20, (next.time_s - current.time_s) * 1000));
}

$("run").textContent = `${data.split === "val" ? "Held-out sequences" : "Training sequences"} · ${data.forecast_ms} ms forecast · epoch ${data.epoch}`;
for (let i = 0; i < data.examples.length; i++) {
  const item = data.examples[i];
  $("example").add(new Option(`Seq ${item.sequence} · Track ${item.track_id} · ${item.anchor_time_s.toFixed(2)} s`, i));
}
$("example").addEventListener("change", (event) => { state.example = Number(event.target.value); updateExample(); });
$("time").addEventListener("input", (event) => { stop(); state.frame = Number(event.target.value); updateFrame(); });
$("play").addEventListener("click", () => {
  if (state.play !== null) { stop(); return; }
  if (state.frame === sample().frames.length - 1) { state.frame = 0; updateFrame(); }
  $("play").textContent = "Pause";
  playNext();
});
$("rgb-kind").addEventListener("change", drawCamera);
$("overlay").addEventListener("change", drawCamera);
$("crop").addEventListener("change", updateFrame);
$("view-mode").addEventListener("change", resetView);
$("reset").addEventListener("click", resetView);
orbitCanvas.addEventListener("pointerdown", (event) => {
  if ($("view-mode").value === "camera") return;
  state.drag = [event.clientX, event.clientY];
  orbitCanvas.setPointerCapture(event.pointerId);
});
orbitCanvas.addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  state.yaw += (event.clientX - state.drag[0]) * 0.009;
  state.pitch = clamp(state.pitch + (event.clientY - state.drag[1]) * 0.009, -1.5, 1.5);
  state.drag = [event.clientX, event.clientY];
  drawOrbit();
});
orbitCanvas.addEventListener("lostpointercapture", () => { state.drag = null; });
orbitCanvas.addEventListener("wheel", (event) => {
  if ($("view-mode").value === "camera") return;
  event.preventDefault();
  state.zoom = clamp(state.zoom * Math.exp(-0.001 * event.deltaY), 0.35, 3.5);
  drawOrbit();
}, { passive: false });
new ResizeObserver(updateFrame).observe(cameraCanvas);
new ResizeObserver(drawOrbit).observe(orbitCanvas);
updateExample();
