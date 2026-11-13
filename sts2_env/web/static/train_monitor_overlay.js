(function () {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const STATUS_ZH = {
    running: "运行",
    waiting: "等待",
    complete: "完成",
    failed: "失败",
    stale: "过期",
    unknown: "未知",
  };
  let lastSnap = null;
  let selectedRunId = null;
  let connected = false;
  let lastEventAt = null;
  let seenDeliveries = new Set();
  let painting = false;
  let framedOnce = false;

  function q(id) {
    return document.querySelector(
      '#cv-world [data-kind="node"][data-id="' +
        CSS.escape(id) +
        '"]:not(.wf-slot), #cv-world [data-kind="step"][data-id="' +
        CSS.escape(id) +
        '"]'
    );
  }

  function hud() {
    let el = document.getElementById("sts2-monitor-hud");
    if (el) return el;
    el = document.createElement("aside");
    el.id = "sts2-monitor-hud";
    el.setAttribute("aria-live", "polite");
    el.innerHTML =
      '<div id="sts2-monitor-test-banner">TEST MODE 夹具目录，不是实机并发训练</div>' +
      '<section id="sts2-monitor-clocks">' +
      '<div id="sts2-monitor-conn"></div>' +
      '<div id="sts2-monitor-collect"></div>' +
      '<div id="sts2-monitor-advance"></div>' +
      '<div id="sts2-monitor-edge"></div>' +
      '<div id="sts2-monitor-empty"></div>' +
      "</section>" +
      '<ol id="sts2-monitor-runs"></ol>' +
      '<div id="sts2-monitor-detail" hidden></div>';
    document.body.appendChild(el);
    return el;
  }

  function installDock() {
    const app = document.getElementById("app");
    const box = hud();
    document.body.classList.add("sts2-docked");
    if (app) {
      app.classList.add("sts2-docked");
      if (box.parentElement !== app) app.appendChild(box);
    } else if (box.parentElement !== document.body) {
      document.body.appendChild(box);
    }
    const layout = document.querySelector(".cv-layout");
    if (layout) layout.classList.add("rail-off", "dt-off");
  }

  function tightenTrain() {
    const root = q("train-root");
    const frame = root && root.closest(".cv-frame");
    if (frame) frame.classList.add("sts2-train-frame");
    const io = q("train-io");
    if (io) io.classList.add("sts2-io-wrap");
    const zips = q("train-zips");
    const cmd = q("train-cmd");
    const inner = io && (io.querySelector(":scope > .wf-children") || io);
    if (inner && zips && zips.parentElement !== inner) {
      inner.insertBefore(zips, inner.firstChild);
    }
    if (io && cmd && cmd.parentElement && io.previousElementSibling !== cmd) {
      cmd.parentElement.insertBefore(io, cmd.nextSibling);
    }
  }

  function hudLeft() {
    const hud = document.getElementById("sts2-monitor-hud");
    const vp = document.getElementById("cv-vp");
    if (hud) return hud.getBoundingClientRect().left;
    return vp ? vp.getBoundingClientRect().right : window.innerWidth;
  }

  function trainInView() {
    const el = q("train-cmd") || q("train-root");
    const io = q("train-io") || q("train-zips");
    const vp = document.getElementById("cv-vp");
    if (!el || !vp) return false;
    const camK = (window.smyw && window.smyw.ui && window.smyw.ui.cam && window.smyw.ui.cam.k) || 1;
    if (Math.abs(camK - 1) > 0.02) return false;
    const a = el.getBoundingClientRect();
    const z = io ? io.getBoundingClientRect() : a;
    const b = vp.getBoundingClientRect();
    const left = Math.min(a.left, z.left);
    const right = Math.max(a.right, z.right);
    const top = Math.min(a.top, z.top);
    const bottom = Math.max(a.bottom, z.bottom);
    return (
      a.width > 80 &&
      left >= b.left - 8 &&
      right <= hudLeft() - 8 &&
      top >= b.top - 20 &&
      bottom <= b.bottom + 12
    );
  }

  function frameTrainCard() {
    if (framedOnce && trainInView()) return;
    const readBtn = document.querySelector('.cv-tools [data-fit="read"]');
    const cardBtn = document.querySelector('.cv-tools [data-fit="card"]');
    if (cardBtn && cardBtn.classList.contains("on") && readBtn) readBtn.click();
    const vp = document.getElementById("cv-vp");
    const world = document.getElementById("cv-world");
    const el = q("train-cmd") || q("train-root");
    const zips = q("train-io") || q("train-zips");
    if (!vp || !world || !el || !window.smyw || !window.smyw.ui) return;
    const k0 = (window.smyw.ui.cam && window.smyw.ui.cam.k) || 1;
    const wr = world.getBoundingClientRect();
    if (wr.width < 2) return;
    const er = el.getBoundingClientRect();
    const zr = zips ? zips.getBoundingClientRect() : er;
    const left = Math.min(er.left, zr.left);
    const top = Math.min(er.top, zr.top);
    const right = Math.max(er.right, zr.right);
    const bottom = Math.max(er.bottom, zr.bottom);
    const box = {
      x: (left - wr.left) / k0,
      y: (top - wr.top) / k0,
      w: (right - left) / k0,
      h: (bottom - top) / k0,
    };
    const k = 1;
    const x = 16 - box.x * k;
    const y = 12 - box.y * k;
    window.smyw.ui.cam = { k: k, x: x, y: y };
    world.style.transform = "translate(" + x + "px," + y + "px) scale(" + k + ")";
    const z = document.getElementById("cv-zoom");
    if (z) z.textContent = "100%";
    if (trainInView()) framedOnce = true;
  }

  function setStatus(id, status) {
    const el = q(id);
    if (el) el.setAttribute("data-status", status);
  }

  function clearCanvasStatus() {
    document.querySelectorAll("#cv-world [data-status]").forEach((el) => {
      if (el.classList.contains("sts2-now")) return;
      el.removeAttribute("data-status");
      el.classList.remove("sts2-pulse");
    });
  }

  function metricText(m) {
    if (m.total != null && m.total > 0) {
      const pct = Math.round((100 * m.value) / m.total);
      return m.label + " " + m.value + " / " + m.total + " " + m.unit + "（" + pct + "%）";
    }
    return m.label + " 累计 " + m.value + " " + m.unit;
  }

  function occupantOf(snapshot, runId) {
    return (snapshot.occupants || []).find((o) => o.run_id === runId) || null;
  }

  function edgeCaption(snapshot) {
    if (!connected) return "过程边 learn→Checkpoints 未流动 · 连接断开";
    if (reduced) return "过程边 learn→Checkpoints 未流动 · prefers-reduced-motion";
    const ae = ((snapshot && snapshot.activity_edges) || []).find(
      (e) => e.edge_from === "train-go" && e.edge_to === "train-zips"
    );
    if (ae && ae.active) return "过程边 learn→Checkpoints 流动中";
    if (ae && ae.missing_signal) return "过程边 learn→Checkpoints 未流动 · 缺 " + ae.missing_signal;
    return "过程边 learn→Checkpoints 无活动证据";
  }

  function paintHud(snapshot) {
    const box = hud();
    box.querySelector("#sts2-monitor-test-banner").style.display = snapshot.test_mode
      ? "block"
      : "none";
    box.querySelector("#sts2-monitor-conn").textContent = connected
      ? "连接 已连接（不等于训练运行）"
      : "连接 断开";
    const gen = snapshot.generated_at || "";
    const age = lastEventAt ? Math.max(0, Date.now() - lastEventAt) : null;
    box.querySelector("#sts2-monitor-collect").textContent =
      age == null ? "采集 尚无采集时钟" : "采集延迟 " + age + " ms";
    const genEl = box.querySelector("#sts2-monitor-advance");
    const highlight = selectedRunId || snapshot.active_run_id;
    const occ = occupantOf(snapshot, highlight);
    const adv = occ && occ.last_advance_at;
    genEl.textContent = adv ? "推进 " + adv : "推进 无推进时钟 · " + gen;
    box.querySelector("#sts2-monitor-edge").textContent = edgeCaption(snapshot);
    const occupants = snapshot.occupants || [];
    const empty = box.querySelector("#sts2-monitor-empty");
    if (!occupants.length) {
      empty.hidden = false;
      empty.textContent = "无近期任务记录（未确认空闲）";
    } else {
      empty.hidden = true;
      empty.textContent = "";
    }
    const list = box.querySelector("#sts2-monitor-runs");
    list.replaceChildren();
    occupants.forEach((row) => {
      const li = document.createElement("li");
      li.setAttribute("data-status", row.status);
      if (row.run_id === highlight) li.classList.add("on");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sts2-task";
      const st = STATUS_ZH[row.status] || row.status;
      const metrics = (row.metrics || []).map(metricText).join("\n");
      btn.innerHTML =
        '<span class="sts2-task-id"></span>' +
        '<span class="sts2-task-lane"></span>' +
        '<span class="sts2-task-step"></span>' +
        '<span class="sts2-task-st"></span>' +
        '<span class="sts2-task-m"></span>';
      btn.querySelector(".sts2-task-id").textContent = row.run_id;
      btn.querySelector(".sts2-task-lane").textContent = row.lane;
      btn.querySelector(".sts2-task-step").textContent = "子步骤 " + row.substep_label;
      btn.querySelector(".sts2-task-st").textContent = "状态 " + st;
      btn.querySelector(".sts2-task-m").textContent = metrics || "指标 未知";
      btn.addEventListener("click", () => {
        selectedRunId = row.run_id;
        fetch("/api/snapshot?run_id=" + encodeURIComponent(row.run_id), {
          cache: "no-store",
        })
          .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
          .then((snap) => applySnapshot(snap, true))
          .catch(() => paintDisconnected());
      });
      li.appendChild(btn);
      list.appendChild(li);
    });
  }

  function paintDisconnected() {
    connected = false;
    paintHud(lastSnap || { test_mode: false, occupants: [], generated_at: "" });
    syncProcessEdge();
  }

  function showDetail(id) {
    const pane = hud().querySelector("#sts2-monitor-detail");
    if (!lastSnap) {
      pane.hidden = true;
      return;
    }
    const occ = (lastSnap.occupants || []).find((o) => o.overlay_id === id);
    const cell = (lastSnap.cells || []).find((c) => c.overlay_id === id);
    if (!occ && !cell) {
      pane.hidden = true;
      return;
    }
    pane.hidden = false;
    const facts = ((cell && cell.facts) || [])
      .map((f) => {
        const bits = [f.kind];
        if (f.tag) bits.push(f.tag);
        if (f.step != null) bits.push("step " + f.step);
        if (f.value != null) bits.push(String(f.value));
        if (f.n_rows != null) bits.push("rows " + f.n_rows);
        if (f.detail) bits.push(f.detail);
        if (f.token) bits.push(f.token);
        return bits.join(" ");
      })
      .join("\n");
    const head = occ
      ? occ.run_id + " · " + occ.substep_label + " · " + occ.status + " · " + occ.reason
      : id + " · " + cell.status + " · " + cell.reason;
    pane.textContent = head + (facts ? "\n" + facts : "");
  }

  function paintNow(occ) {
    document.querySelectorAll(".sts2-now").forEach((el) => el.remove());
    const host = q("train-cmd");
    if (!host || !occ || occ.lane !== "train") return;
    const bar = document.createElement("div");
    bar.className = "sts2-now";
    bar.setAttribute("data-status", occ.status);
    const st = STATUS_ZH[occ.status] || occ.status;
    const metrics = (occ.metrics || []).map(metricText).join(" · ") || "指标 未知";
    bar.innerHTML =
      '<div class="sts2-now-k">当前步骤</div>' +
      '<div class="sts2-now-t"></div>' +
      '<div class="sts2-now-s"></div>' +
      '<div class="sts2-now-m"></div>';
    bar.querySelector(".sts2-now-t").textContent = occ.substep_label;
    bar.querySelector(".sts2-now-s").textContent = "状态 " + st + " · " + occ.reason;
    bar.querySelector(".sts2-now-m").textContent = metrics;
    const edge = document.createElement("div");
    edge.className = "sts2-now-e";
    edge.textContent = edgeCaption(lastSnap);
    bar.appendChild(edge);
    host.insertBefore(bar, host.firstChild);
  }

  function worldPoint(el) {
    const world = document.getElementById("cv-world");
    if (!el || !world) return null;
    const wr = world.getBoundingClientRect();
    const k = wr.width / (world.offsetWidth || 1) || 1;
    const r = el.getBoundingClientRect();
    return {
      x: (r.left + r.width / 2 - wr.left) / k,
      y: (r.top + r.height / 2 - wr.top) / k,
      r: (r.right - wr.left) / k,
      l: (r.left - wr.left) / k,
      t: (r.top - wr.top) / k,
      b: (r.bottom - wr.top) / k,
    };
  }

  function ensureProcessSvg() {
    const world = document.getElementById("cv-world");
    if (!world) return null;
    let svg = document.getElementById("cv-links");
    if (!svg) {
      svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("id", "cv-links");
      svg.setAttribute("class", "cv-links");
      world.appendChild(svg);
    }
    svg.setAttribute("width", String(world.offsetWidth));
    svg.setAttribute("height", String(world.offsetHeight));
    return svg;
  }

  function syncProcessEdge() {
    const svg = ensureProcessSvg();
    const a = q("train-go");
    const b = q("train-zips");
    if (!svg || !a || !b) return;
    const A = worldPoint(a);
    const B = worldPoint(b);
    if (!A || !B) return;
    const x1 = A.r + 2;
    const y1 = (A.t + A.b) / 2;
    const x2 = B.l - 3;
    const y2 = B.t + Math.min(36, (B.b - B.t) / 2);
    const dx = Math.max(80, Math.abs(x2 - x1) * 0.45);
    const d =
      "M" +
      x1 +
      "," +
      y1 +
      " C" +
      (x1 + dx) +
      "," +
      y1 +
      " " +
      (x2 - dx) +
      "," +
      y2 +
      " " +
      x2 +
      "," +
      y2;
    let path = svg.querySelector('path.sts2-process[data-from="train-go"][data-to="train-zips"]');
    if (!path) {
      path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("class", "cv-link sts2-process");
      path.setAttribute("data-from", "train-go");
      path.setAttribute("data-to", "train-zips");
      path.setAttribute("data-kind", "process");
      path.setAttribute("fill", "none");
      svg.appendChild(path);
    }
    path.setAttribute("d", d);
    const edges = (lastSnap && lastSnap.activity_edges) || [];
    const ae = edges.find((e) => e.edge_from === "train-go" && e.edge_to === "train-zips");
    const live = !!(connected && ae && ae.active && !reduced);
    path.classList.toggle("sts2-flow", live);
    path.setAttribute("data-missing", (ae && ae.missing_signal) || "");
    path.setAttribute(
      "aria-label",
      live ? "活动过程边 learn 到 Checkpoints" : edgeCaption(lastSnap)
    );
  }

  function particle(delivery) {
    if (reduced || !delivery.edge_from || !delivery.edge_to) return false;
    if (!connected) return false;
    syncProcessEdge();
    const svg = document.getElementById("cv-links");
    const path =
      svg &&
      svg.querySelector(
        'path.sts2-process[data-from="' +
          CSS.escape(delivery.edge_from) +
          '"][data-to="' +
          CSS.escape(delivery.edge_to) +
          '"]'
      );
    if (!svg || !path) return false;
    const host = document.createElement("div");
    host.className = "sts2-dot-html";
    document.body.appendChild(host);
    const len = path.getTotalLength();
    function put(p) {
      const ctm = path.getScreenCTM();
      if (!ctm) return;
      const loc = new DOMPoint(p.x, p.y).matrixTransform(ctm);
      host.style.left = loc.x + "px";
      host.style.top = loc.y + "px";
    }
    put(path.getPointAtLength(0));
    const t0 = performance.now();
    function frame(t) {
      const u = Math.min(1, (t - t0) / 20000);
      put(path.getPointAtLength(len * u));
      if (u < 1) requestAnimationFrame(frame);
      else host.remove();
    }
    requestAnimationFrame(frame);
    return true;
  }

  function replayDeliveries(snapshot) {
    (snapshot.recent_deliveries || []).forEach((d) => {
      if (!d.delivery_id || seenDeliveries.has(d.delivery_id)) return;
      if (particle(d)) seenDeliveries.add(d.delivery_id);
    });
  }

  function paintCanvas(snapshot) {
    tightenTrain();
    frameTrainCard();
    clearCanvasStatus();
    (snapshot.cells || []).forEach((c) => {
      if (c.overlay_id === "train-go" || c.overlay_id === "train-zips") return;
      setStatus(c.overlay_id, c.status);
    });
    const highlight = selectedRunId || snapshot.active_run_id;
    const occ = occupantOf(snapshot, highlight);
    if (occ && occ.lane === "train") {
      paintNow(occ);
      if (occ.substep_id === "zips") setStatus("train-zips", occ.status);
      else if (occ.substep_id === "launch") setStatus("train-cmd", occ.status);
      else setStatus("train-go", occ.status);
    }
    syncProcessEdge();
  }

  function applySnapshot(snapshot) {
    lastSnap = snapshot;
    lastEventAt = Date.now();
    painting = true;
    try {
      installDock();
      paintCanvas(snapshot);
      paintHud(snapshot);
      replayDeliveries(snapshot);
    } finally {
      painting = false;
    }
  }

  function closeReviewRail() {
    const layout = document.querySelector(".cv-layout");
    if (layout) {
      layout.classList.add("rail-off", "dt-off", "sts2-has-dock");
    }
    const l = document.querySelector(".pane-toggle.l");
    if (l && l.getAttribute("aria-expanded") === "true") l.click();
  }

  function bootShell() {
    const smyw = window.smyw;
    if (!smyw || !smyw.ui) return;
    smyw.ui.tab = "canvas";
    smyw.ui.mode = "view";
    smyw.ui.showLinks = true;
    requestAnimationFrame(() => {
      if (typeof smyw.render === "function") smyw.render();
      closeReviewRail();
      installDock();
      requestAnimationFrame(() => {
        closeReviewRail();
        installDock();
        tightenTrain();
        frameTrainCard();
        syncProcessEdge();
      });
    });
  }

  const overlay = {
    replace(snapshot) {
      connected = true;
      applySnapshot(snapshot);
    },
    apply(event) {
      if (!event || !event.type) return;
      lastEventAt = Date.now();
      if (event.type === "status_changed") setStatus(event.overlay_id, event.status);
      if (event.type === "delivery") {
        if (particle(event) && event.delivery_id) seenDeliveries.add(event.delivery_id);
      }
      if (event.type === "pulse") {
        const el = q(event.overlay_id);
        if (el && connected) {
          el.classList.add("sts2-pulse");
          setTimeout(() => el.classList.remove("sts2-pulse"), 1200);
        }
      }
    },
  };

  window.sts2Monitor = {
    bind() {
      return overlay;
    },
  };

  hud();
  bootShell();
  document.addEventListener("click", (e) => {
    const n = e.target.closest("[data-kind='node'][data-id], [data-kind='step'][data-id]");
    if (n) showDetail(n.getAttribute("data-id"));
  });
  const app = document.getElementById("app");
  if (app && window.MutationObserver) {
    let queued = false;
    new MutationObserver(() => {
      if (painting || queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        installDock();
        closeReviewRail();
        if (lastSnap) {
          painting = true;
          try {
            paintCanvas(lastSnap);
            paintHud(lastSnap);
          } finally {
            painting = false;
          }
        } else {
          tightenTrain();
          frameTrainCard();
          syncProcessEdge();
        }
      });
    }).observe(app, { childList: true, subtree: true });
  }

  const src = new EventSource("/api/events");
  src.addEventListener("snapshot", (e) => {
    connected = true;
    const msg = JSON.parse(e.data);
    if (!msg.snapshot) return;
    if (selectedRunId) pullSnapshot();
    else overlay.replace(msg.snapshot);
  });
  src.addEventListener("event", (e) => {
    connected = true;
    const msg = JSON.parse(e.data);
    if (msg.event) overlay.apply(msg.event);
  });
  src.onerror = () => {
    connected = false;
    paintHud(lastSnap || { test_mode: false, occupants: [], generated_at: "" });
    syncProcessEdge();
  };
  function pullSnapshot() {
    const qs = selectedRunId ? "?run_id=" + encodeURIComponent(selectedRunId) : "";
    fetch("/api/snapshot" + qs, { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((snap) => {
        connected = true;
        overlay.replace(snap);
      })
      .catch(() => paintDisconnected());
  }
  setInterval(pullSnapshot, 2000);
  pullSnapshot();
})();
