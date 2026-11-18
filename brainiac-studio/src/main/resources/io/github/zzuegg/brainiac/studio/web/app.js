// Brainiac Studio front end. Plain JavaScript, no build step.
(() => {
  "use strict";
  const M = window.STUDIO_MOCK;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const icon = (n, cls = "") => `<svg class="${cls}" aria-hidden="true"><use href="#i-${n}"/></svg>`;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const capIcon = { text: "text", image: "image", speech: "speech", vision: "image", train: "train", agent: "agent" };
  const capName = { text: "Text", image: "Image", speech: "Speech", vision: "Vision", train: "Training", agent: "Agents", video: "Video" };

  // What the page remembers. Served by the studio, it lives in page.json in the studio's data
  // folder (the page arrives with it), so every browser on this PC sees the same tabs, layout and
  // choices; conversations are files of their own there. Opened as a plain file, it uses this browser.
  const pageState = (() => {
    const el = document.getElementById("studio-state");
    try { return el ? JSON.parse(el.textContent) : null; } catch { return null; }
  })();
  const local = {
    get(k) { try { const v = localStorage.getItem("studio." + k); return v == null ? undefined : JSON.parse(v); } catch { return undefined; } },
    set(k, v) { try { localStorage.setItem("studio." + k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
    keys() { try { return Object.keys(localStorage).filter(k => k.startsWith("studio.")).map(k => k.slice(7)); } catch { return []; } },
    remove(k) { try { localStorage.removeItem("studio." + k); } catch { /* storage unavailable */ } },
  };
  let stateChanges = {}, stateTimer = 0;
  function sendState() {
    const body = stateChanges;
    stateChanges = {}; stateTimer = 0;
    const token = document.querySelector('meta[name="studio-token"]')?.content;
    return fetch("/api/state", { method: "POST", keepalive: true, headers: { "X-Studio-Token": token, "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then(r => { if (!r.ok) console.warn("cannot save the page state", r.status); return r.ok; })
      .catch(e => { console.warn("cannot save the page state", e); return false; });
  }
  const store = {
    // conversations are not page state: the studio keeps them in its chats folder
    get(k, d) {
      if (pageState && k !== "convos" && k in pageState) return pageState[k];
      if (pageState && k !== "convos") return d;
      const v = local.get(k); return v === undefined ? d : v;
    },
    set(k, v) {
      if (!pageState) { local.set(k, v); return; }
      if (k === "convos") return;
      pageState[k] = v;
      stateChanges[k] = v === undefined ? null : v;
      clearTimeout(stateTimer);
      stateTimer = setTimeout(sendState, 300);
    },
  };
  addEventListener("pagehide", () => { if (stateTimer) { clearTimeout(stateTimer); sendState(); } });
  // What this browser remembered before the studio kept it moves to the studio once, then is cleared here.
  if (pageState) {
    const moved = local.keys().filter(k => k !== "convos" && !(k in pageState));
    if (moved.length) {
      moved.forEach(k => { pageState[k] = local.get(k); stateChanges[k] = pageState[k]; });
      sendState().then(ok => { if (ok) moved.forEach(local.remove); });
    }
    local.keys().filter(k => k !== "convos" && !moved.includes(k)).forEach(local.remove);
  }

  const state = {
    view: "chat",
    layout: store.get("layout", "auto"),      // auto | focus | wide | ultra
    density: store.get("density", "cozy"),    // cozy | compact
    scale: store.get("scale", 1),
    agent: "voice",
    tab: "activity",
    mode: "auto",
    autonomy: "ask",
    playing: false,
    t: 14.2,
  };

  // ---------- layout ----------
  function effectiveLayout() {
    if (state.layout !== "auto") return state.layout;
    const w = window.innerWidth / state.scale;
    if (w >= 2200) return "ultra";
    if (w >= 1400) return "wide";
    return "focus";
  }
  function applyFrame() {
    const app = $("#app");
    app.dataset.layout = effectiveLayout();
    app.dataset.density = state.density;
    app.style.zoom = state.scale === 1 ? "" : state.scale;
  }

  // ---------- procedural pictures (stand-ins for generated media) ----------
  function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }
  function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim() || "#888"; }

  function paintThumb(cv, seed, noise = 0) {
    const w = cv.width = 256, h = cv.height = 256, g = cv.getContext("2d"), r = rng(seed);
    const hues = [[143, 184, 255], [244, 169, 106], [195, 164, 255], [99, 214, 174], [255, 130, 151]];
    const a = hues[seed % 5], b = hues[(seed + 2) % 5];
    const bg = g.createLinearGradient(0, 0, w, h);
    bg.addColorStop(0, `rgb(${10 + a[0] / 14},${12 + a[1] / 14},${20 + a[2] / 12})`);
    bg.addColorStop(1, `rgb(${8 + b[0] / 18},${9 + b[1] / 18},${14 + b[2] / 14})`);
    g.fillStyle = bg; g.fillRect(0, 0, w, h);
    const glow = g.createRadialGradient(w * (.3 + r() * .4), h * (.3 + r() * .4), 4, w / 2, h / 2, w * .7);
    glow.addColorStop(0, `rgba(${a},.55)`); glow.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = glow; g.fillRect(0, 0, w, h);
    // a little node graph, the subject of the thumbnail
    const nodes = [];
    const n = 5 + Math.floor(r() * 3);
    for (let i = 0; i < n; i++) nodes.push([28 + r() * (w - 96), 26 + (i / n) * (h - 70) + r() * 10]);
    g.lineWidth = 2;
    for (let i = 1; i < n; i++) {
      const [x0, y0] = nodes[i - 1], [x1, y1] = nodes[i];
      g.strokeStyle = `rgba(${i % 2 ? a : b},.85)`;
      g.beginPath(); g.moveTo(x0 + 34, y0 + 18); g.bezierCurveTo(x0 + 34, y0 + 50, x1 + 34, y1 - 20, x1 + 34, y1); g.stroke();
    }
    for (let i = 0; i < n; i++) {
      const [x, y] = nodes[i];
      g.fillStyle = "rgba(19,23,33,.92)"; g.strokeStyle = `rgba(${i % 2 ? a : b},.9)`; g.lineWidth = 1.5;
      g.beginPath(); g.roundRect(x, y, 68, 18, 5); g.fill(); g.stroke();
      g.fillStyle = `rgba(${i % 2 ? a : b},1)`; g.fillRect(x + 6, y + 7, 5, 5);
      g.fillStyle = "rgba(231,234,241,.55)"; g.fillRect(x + 16, y + 8, 30 + r() * 16, 3);
    }
    g.fillStyle = "rgba(255,255,255,.92)"; g.font = "700 22px 'Bricolage Grotesque', sans-serif";
    if (seed % 2 === 0) g.fillText("Build it.", 18, h - 20);
    if (noise > 0) {
      const img = g.getImageData(0, 0, w, h), d = img.data, rr = rng(seed * 7 + 1);
      for (let i = 0; i < d.length; i += 4) {
        const v = rr() * 255;
        d[i] = d[i] * (1 - noise) + v * noise; d[i + 1] = d[i + 1] * (1 - noise) + rr() * 255 * noise; d[i + 2] = d[i + 2] * (1 - noise) + rr() * 255 * noise;
      }
      g.putImageData(img, 0, 0);
    }
  }

  function paintFrame(cv, t) {
    const w = cv.width = 1280, h = cv.height = 720, g = cv.getContext("2d");
    const bg = g.createLinearGradient(0, 0, 0, h); bg.addColorStop(0, "#10141d"); bg.addColorStop(1, "#0a0d13");
    g.fillStyle = bg; g.fillRect(0, 0, w, h);
    g.fillStyle = "rgba(143,160,200,.09)";
    for (let x = 20; x < w; x += 28) for (let y = 20; y < h; y += 28) g.fillRect(x, y, 2, 2);
    // the builder being demoed: nodes slide in with time
    const cols = ["#8fb8ff", "#f4a96a", "#63d6ae", "#e6d27e", "#c3a4ff"];
    const k = Math.min(1, t / 20);
    const pts = [[180, 110], [180, 250], [180, 390], [520, 250], [860, 250], [860, 400]];
    g.lineWidth = 3;
    for (let i = 1; i < pts.length; i++) {
      if (i / pts.length > k + .2) break;
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      g.strokeStyle = cols[i % 5]; g.globalAlpha = .8;
      g.beginPath(); g.moveTo(x0 + 110, y0 + 70); g.bezierCurveTo(x0 + 110, y0 + 140, x1 + 110, y1 - 60, x1 + 110, y1); g.stroke();
    }
    g.globalAlpha = 1;
    pts.forEach(([x, y], i) => {
      if (i / pts.length > k + .25) return;
      g.fillStyle = "#151a24"; g.strokeStyle = i === 5 && t > 12 && t < 19 ? "#ff6f7d" : "#2a3242"; g.lineWidth = 2;
      g.beginPath(); g.roundRect(x, y, 220, 70, 14); g.fill(); g.stroke();
      g.fillStyle = cols[i % 5]; g.fillRect(x + 16, y + 18, 10, 10);
      g.fillStyle = "#e7eaf1"; g.font = "600 20px Geist, sans-serif"; g.fillText(["Input", "Conv", "SiLU", "Global pool", "Dense", "Dense"][i], x + 36, y + 30);
      g.fillStyle = "#7f889b"; g.font = "16px 'Geist Mono', monospace"; g.fillText(["[3,32,32]", "[32,32,32]", "[32,32,32]", "[64]", "[10]", "expects [N]"][i], x + 16, y + 56);
    });
    // cursor
    const cx = 300 + Math.sin(t / 2) * 260, cy = 300 + Math.cos(t / 3) * 120;
    g.fillStyle = "#fff"; g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + 18, cy + 18); g.lineTo(cx + 7, cy + 19); g.lineTo(cx, cy + 27); g.closePath(); g.fill();
  }

  function waveBars(n, seed, past = -1) {
    const r = rng(seed); let s = "";
    for (let i = 0; i < n; i++) {
      const env = Math.sin(Math.PI * (i / n)) * .6 + .4;
      const v = Math.max(8, Math.round((.25 + r() * .75) * env * 100));
      s += `<i class="${i < past ? "past" : ""}" style="height:${v}%"></i>`;
    }
    return s;
  }

  // ---------- CHAT ----------
  const toolActions = {
    text: [["Shorten", "edit"], ["Change tone", "wand"], ["Rewrite line", "retry"]],
    speech: [["Retake line", "retry"], ["Change voice", "speech"], ["Slower", "speech"], ["Align to cuts", "video"]],
    image: [["Variations", "image"], ["Upscale ×2", "image"], ["Lock seed", "lock"], ["Inpaint", "edit"]],
  };

  function threadList() {
    return `
    <aside class="threads" aria-label="Conversations">
      <div class="col-head"><span class="h3">Threads</span><span class="grow"></span>
        <button class="iconbtn" aria-label="New thread">${icon("plus")}</button></div>
      <div class="search">${icon("search")}<input class="input" id="thread-search" placeholder="Search threads and agents"></div>
      <div class="scroll">
        ${M.threads.map(g => `
          <div class="tgroup"><span class="eyebrow">${g.group}</span>
            ${g.items.map(t => `
              <div class="titem ${t.active ? "active" : ""} ${t.child ? "child" : ""}" data-thread="${t.id}" tabindex="0">
                <span class="dot ic ${t.attention ? "pulse" : ""}" style="--c:var(--cap-${t.attention ? "agent" : t.cap})"></span>
                <div style="min-width:0"><div class="t">${esc(t.title)}</div><div class="s">${esc(t.sub)}</div></div>
                <span class="when">${t.when}</span>
              </div>`).join("")}
          </div>`).join("")}
      </div>
    </aside>`;
  }

  function agentCard(a) {
    const st = { done: ["done", "c-ok"], running: ["running", "c-text"], waiting: ["needs you", "c-warn"] }[a.status];
    return `
    <div class="agent ${state.agent === a.id ? "selected" : ""}" data-agent="${a.id}" style="--c:var(--cap-${a.cap})" tabindex="0" role="button" aria-label="Open ${esc(a.name)}">
      <div class="glyph">${icon(capIcon[a.cap])}</div>
      <div style="min-width:0">
        <div class="name">${esc(a.name)} <span class="chip ${st[1]}">${st[0]}</span></div>
        <div class="doing">${esc(a.doing)}</div>
      </div>
      <div class="stats"><span>${a.time}</span><span>${a.tokens === "—" ? capName[a.cap].toLowerCase() : a.tokens + " tok"}</span></div>
      <span class="bar" style="width:${a.progress * 100}%"></span>
    </div>`;
  }

  function transcript() {
    const thumbs = [11, 12, 13, 14];
    return `
    <section class="stage" aria-label="Conversation">
      <div class="stage-head">
        <span class="title">Voiceover for the builder demo</span>
        <span class="chip c-agent">orchestrator</span>
        <span class="grow"></span>
        <div class="seg" role="group" aria-label="View">
          <button aria-pressed="true">Thread</button><button aria-pressed="false">Tree</button><button aria-pressed="false">Compare</button>
        </div>
        <button class="iconbtn" id="toggle-insp" aria-label="Show or hide the agent panel">${icon("panel")}</button>
      </div>
      <div class="transcript">
        <div class="lane">
          <div class="msg user">
            <div class="avatar user">MZ</div>
            <div>
              <div class="meta"><b>You</b><span>15:36</span></div>
              <div class="body">Here's the 38 second demo of the model builder. Write a voiceover that follows the cuts, voice it with a calm narrator, and make a thumbnail for the upload.</div>
              <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">
                <span class="chip c-vision">${icon("video")} builder-demo.mp4 · 0:38 · 1080p</span>
              </div>
            </div>
          </div>

          <div class="msg">
            <div class="avatar ai" style="--c:var(--cap-agent)">${icon("agent")}</div>
            <div style="min-width:0">
              <div class="meta"><b>Orchestrator</b><span class="chip c-text">Qwen2.5 7B</span><span class="mono">41 tok/s</span>
                <span class="branch" title="Earlier answers to the same message">${"<button aria-label='Previous answer'>‹</button>"}2 / 3${"<button aria-label='Next answer'>›</button>"}</span></div>
              <div class="body">
                <p>The video has 6 cuts. I'll split this into three parts, each with its own agent, so each one only sees the tools it needs.</p>
                <details class="tool" open>
                  <summary>${icon("video")}<span><b>video.scenes</b> builder-demo.mp4</span><span class="grow"></span><span class="mono muted">6 cuts · 0.4 s</span></summary>
                  <pre>0.0  empty canvas        12.8  conv → dense error     26.4  chat + MCP
6.0  drag first block    19.4  training, loss curve   32.8  logo</pre>
                </details>
                <div class="plan">
                  <div class="plan-head">${icon("check")}<b>Plan</b><span class="grow"></span><span class="mono muted">2 of 4</span></div>
                  <ul class="todo">
                    <li class="done"><span class="box"></span>Write 6 lines timed to the cuts<span class="mono">Script writer</span></li>
                    <li class="run"><span class="box"></span>Voice the lines with a calm narrator<span class="mono">Voiceover</span></li>
                    <li class="run"><span class="box"></span>Render 4 thumbnail candidates<span class="mono">Thumbnail</span></li>
                    <li><span class="box"></span>Mix voice under the video and export</li>
                  </ul>
                </div>
                <div class="agents">${M.agents.map(agentCard).join("")}</div>
              </div>
              <div class="actions">
                <button class="iconbtn" aria-label="Copy">${icon("copy")}</button>
                <button class="iconbtn" aria-label="Try again">${icon("retry")}</button>
                <button class="iconbtn" aria-label="Branch from here">${icon("branch")}</button>
              </div>
            </div>
          </div>

          <div class="checkpoint"><span>${icon("check")}Checkpoint · script saved · 15:36:26</span><button class="btn ghost" title="Go back to this point: conversation, files, or both">Rewind here</button></div>

          <div class="msg">
            <div class="avatar ai" style="--c:var(--cap-speech)">${icon("speech")}</div>
            <div style="min-width:0">
              <div class="meta"><b>Voiceover</b><span class="chip c-speech">OpenAI speech</span><span class="chip plain">remote</span><span class="muted">voice Mara · first two lines</span></div>
              <div class="audio-card">
                <button class="play" aria-label="Play take">${icon("play")}</button>
                <div class="wave">${waveBars(90, 3, 40)}</div>
                <span class="mono muted tnum">0:05 / 0:12</span>
              </div>
            </div>
          </div>

          <div class="msg">
            <div class="avatar ai" style="--c:var(--cap-image)">${icon("image")}</div>
            <div style="min-width:0">
              <div class="meta"><b>Thumbnail</b><span class="chip c-image">thumb-dit-64</span><span class="muted">preview at 12 of 50 steps</span></div>
              <div class="gen-card">
                <div class="imgs">${thumbs.map(s => `<canvas data-thumb="${s}" data-noise=".35" width="256" height="256"></canvas>`).join("")}</div>
                <div class="foot"><span class="chip c-warn">waiting for your OK to finish</span><span class="mono">seed 1207 · DDIM · eta 0</span><span class="grow"></span>
                  <button class="btn" data-agent-open="thumb">Review</button></div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="composer-wrap">
        <div class="composer">
          <div class="route-hint" id="route-hint">${icon("wand")}<span>Auto will send this to</span><span class="chip c-text">Qwen2.5 7B Instruct</span><span class="faint">text · 32k context · tools on</span></div>
          <textarea id="prompt" rows="2" placeholder="Ask anything, attach a file, or @mention an agent">Make the narrator a little slower on the last line, and try a brighter thumbnail</textarea>
          <div class="row">
            <button class="iconbtn" aria-label="Attach a file">${icon("clip")}</button>
            <div class="seg" role="group" aria-label="What to make">
              ${["auto", "text", "image", "speech"].map(m => `<button data-mode="${m}" aria-pressed="${state.mode === m}">${m === "auto" ? icon("wand") : icon(capIcon[m])}${m[0].toUpperCase() + m.slice(1)}</button>`).join("")}
            </div>
            <div class="seg" role="group" aria-label="How much agents may do alone" title="Plan: agents propose, you approve. Ask: they ask before using the GPU or writing files. Auto: they run.">
              ${["plan", "ask", "auto"].map(m => `<button data-autonomy="${m}" aria-pressed="${state.autonomy === m}">${m === "plan" ? "Plan first" : m === "ask" ? "Ask me" : "Run alone"}</button>`).join("")}
            </div>
            <span class="grow"></span>
            <span class="ctx" title="Context used"><span class="meter"><i style="width:7%"></i></span><span class="mono faint">2.1k / 32k</span></span>
            <button class="send" aria-label="Send">${icon("send")}</button>
          </div>
        </div>
      </div>
    </section>`;
  }

  function inspector() {
    const a = M.agents.find(x => x.id === state.agent);
    const st = { done: "Finished", running: "Running", waiting: "Waiting for you" }[a.status];
    const feed = {
      script: [
        ["c-text", "<b>Read</b> 6 cuts from video.scenes", "15:36:12"],
        ["c-text", "<b>Drafted</b> 6 lines, 131 words. Line 3 ran 1.4 s over its cut.", "15:36:19", null],
        ["c-text", "<b>Tightened</b> line 3 to fit 6.0 s", "15:36:24", "“Connect a convolution to a dense layer, and it tells you why that cannot work, and how to fix it.”"],
        ["c-ok", "<b>Done</b> · 112 words · 38.4 s at 175 wpm", "15:36:26"],
      ],
      voice: [
        ["c-speech", "<b>Picked voice</b> Mara · calm narrator", "15:36:27"],
        ["c-speech", "<b>Voiced</b> line 1 · 5.4 s of 5.6 s", "15:36:31"],
        ["c-speech", "<b>Voiced</b> line 2 · 6.1 s of 6.2 s", "15:36:38"],
        ["c-text", "<b>Voicing</b> line 3 …", "15:36:44", "[calm] Connect a convolution to a dense layer, <break 0.3s> and it tells you why that cannot work, and how to fix it."],
      ],
      thumb: [
        ["c-image", "<b>Prompt</b> from the script’s last line", "15:36:28", "dark node graph, glowing edges, big type “Build it.”, 16:9"],
        ["c-image", "<b>Preview</b> 4 candidates at 12 steps", "15:36:52"],
        ["c-warn", "<b>Asks</b> to finish all 4 at 50 steps on the GPU (≈ 9 s, 1.1 GB)", "15:36:53", null, true],
      ],
    }[a.id];
    return `
    <aside class="inspector" aria-label="Agent details" style="--c:var(--cap-${a.cap})">
      <div class="insp-head">
        <div class="top-row">
          <div class="agent" style="padding:0;border:0;background:none;grid-template-columns:auto minmax(0,1fr);--c:var(--cap-${a.cap})">
            <div class="glyph">${icon(capIcon[a.cap])}</div>
            <div><div class="name">${esc(a.name)}</div><div class="doing">${st}</div></div>
          </div>
          <span class="grow"></span>
          <button class="iconbtn" aria-label="Pause agent">${icon("pause")}</button>
          <button class="iconbtn" aria-label="Stop agent">${icon("stop")}</button>
        </div>
        <dl class="kv">
          <dt>Model</dt><dd>${esc(a.model)}</dd>
          <dt>Elapsed</dt><dd>${a.time}</dd>
          <dt>Parent</dt><dd>Orchestrator</dd>
        </dl>
        <div style="display:grid;gap:6px">
          <span class="eyebrow">Tools this agent can use</span>
          <div style="display:flex;gap:6px;flex-wrap:wrap">${a.tools.map(t => `<span class="chip c-${a.cap}">${esc(t)}</span>`).join("")}</div>
        </div>
        ${a.cap === "speech" ? `<div class="flag">${icon("flask")}<div><b>Runs on OpenAI.</b> No local model provides <code>speech.synthesize</code> yet, so this agent uses your OpenAI key and the script leaves this PC. A local voice model takes over once one is installed.</div></div>` : ""}
      </div>
      <div class="tabs" role="tablist">
        ${["activity", "output", "context"].map(t => `<button role="tab" data-tab="${t}" aria-selected="${state.tab === t}">${t[0].toUpperCase() + t.slice(1)}</button>`).join("")}
      </div>
      ${state.tab === "output" && effectiveLayout() !== "ultra" ? `<div class="feed embedded">${mediaPane()}</div>` : ""}
      <div class="feed" ${state.tab === "output" && effectiveLayout() !== "ultra" ? "hidden" : ""}>
        ${feed.map(([c, w, t, pre, ask]) => `
          <div class="ev"><span class="pin ${c}" style="border-color:var(--c)"></span>
            <div><div class="what">${w}</div><div class="t">${t}</div>
              ${pre ? `<pre>${esc(pre)}</pre>` : ""}
              ${ask ? `<div class="approve"><button class="btn primary">Allow once</button><button class="btn">Always for this thread</button><button class="btn ghost">Deny</button></div>` : ""}
            </div></div>`).join("")}
      </div>
      <div class="insp-compose">
        <div style="display:flex;gap:6px;flex-wrap:wrap">${toolActions[a.cap].map(([l, i]) => `<button class="btn" style="height:28px;font-size:12px">${icon(i)}${l}</button>`).join("")}</div>
        <textarea class="input" rows="1" placeholder="Message ${esc(a.name)} directly"></textarea>
      </div>
    </aside>`;
  }

  function mediaPane() {
    const a = M.agents.find(x => x.id === state.agent);
    if (a.cap === "image") {
      return `
      <section class="media" aria-label="Thumbnail candidates">
        <div class="col-head"><span class="h3">Thumbnail candidates</span><span class="grow"></span><span class="chip c-image">image tools</span></div>
        <div class="media-body">
          <div class="batch">${[11, 12, 13, 14].map(s => `<div class="tile"><canvas data-thumb="${s}" data-noise=".35"></canvas><span class="step">12 / 50</span>
            <div class="ov"><button>Vary</button><button>Upscale</button><button>Use</button></div></div>`).join("")}</div>
          <div class="field"><label for="m-prompt">Prompt</label><textarea class="input" id="m-prompt" rows="3">dark node graph, glowing edges, big type “Build it.”, 16:9</textarea></div>
        </div>
      </section>`;
    }
    if (a.cap === "text") {
      return `
      <section class="media" aria-label="Script">
        <div class="col-head"><span class="h3">Script</span><span class="grow"></span><span class="chip c-text">text tools</span></div>
        <div class="media-body">${M.script.map((l, i) => `<div class="para" style="grid-template-columns:64px minmax(0,1fr)"><span class="mono muted">${fmt(l.t)}</span><div class="txt">${esc(l.text)}</div></div>`).join("")}</div>
      </section>`;
    }
    // speech: voiceover over the video, with the player
    const total = 38.4;
    return `
    <section class="media" aria-label="Voiceover preview">
      <div class="col-head"><span class="h3">builder-demo.mp4</span><span class="chip c-speech">voiceover</span><span class="grow"></span>
        <span class="mono muted">1920 × 1080 · 30 fps</span></div>
      <div class="media-body">
        <div class="player">
          <canvas id="frame"></canvas>
          <div class="caption" id="caption">${esc(lineAt(state.t).text)}</div>
        </div>
        <div class="transport">
          <button class="play" id="play" aria-label="Play">${icon(state.playing ? "pause" : "play")}</button>
          <span class="mono tnum" id="clock">${fmt(state.t)} / ${fmt(total)}</span>
          <input type="range" id="scrub" min="0" max="${total}" step="0.1" value="${state.t}" aria-label="Position">
          <div class="seg" role="group" aria-label="Listen to"><button aria-pressed="true">Mix</button><button aria-pressed="false">Voice</button><button aria-pressed="false">Video</button></div>
        </div>
        <div class="tracks">
          <div class="tl-track"><span>${icon("video")} Video</span><div class="tl-lane">${M.script.map(l => `<div class="tl-clip" style="left:${l.t / total * 100}%;width:${l.d / total * 100 - .5}%;background:color-mix(in srgb,var(--cap-vision) 14%,transparent);color:var(--cap-vision)"><span class="mono" style="font-size:10px">cut</span></div>`).join("")}<span class="playhead" style="left:${state.t / total * 100}%"></span></div></div>
          <div class="tl-track"><span>${icon("speech")} Mara</span><div class="tl-lane">${M.script.map((l, i) => `<div class="tl-clip" title="${esc(l.text)}" style="left:${l.t / total * 100}%;width:${l.d / total * 100 - .8}%;background:color-mix(in srgb,var(--cap-speech) ${i < 2 ? 22 : 8}%,transparent);color:var(--cap-speech);${i >= 3 ? "border:1px dashed color-mix(in srgb,var(--cap-speech) 40%,transparent)" : ""}">${i < 3 ? waveBars(36, i + 5) : ""}</div>`).join("")}<span class="playhead" style="left:${state.t / total * 100}%"></span></div></div>
          <div class="tl-track"><span>${icon("text")} Captions</span><div class="tl-lane">${M.script.map(l => `<div class="tl-clip" style="left:${l.t / total * 100}%;width:${l.d / total * 100 - .8}%;background:var(--panel-2);color:var(--fg-2)"><span style="font-size:10.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(l.text)}</span></div>`).join("")}</div></div>
        </div>
        <div class="vo-controls">
          <div class="field"><label for="vo-pace">Pace <output>0.95×</output></label><input type="range" id="vo-pace" min="0.7" max="1.3" step="0.05" value="0.95"></div>
          <div class="field"><label for="vo-stab">Stability <output>0.62</output></label><input type="range" id="vo-stab" min="0" max="1" step="0.01" value="0.62"></div>
          <div class="field"><label for="vo-duck">Duck video audio <output>−12 dB</output></label><input type="range" id="vo-duck" min="-24" max="0" step="1" value="-12"></div>
        </div>
      </div>
    </section>`;
  }

  function lineAt(t) { return M.script.find(l => t >= l.t && t < l.t + l.d + .4) || { text: "" }; }
  function fmt(s) { const m = Math.floor(s / 60), x = (s % 60).toFixed(1).padStart(4, "0"); return `${m}:${x}`; }

  function chatView() {
    if (isLive() && L.sel !== "example") return liveChatView();
    const lay = effectiveLayout();
    const insp = lay !== "focus" || state.inspectorOpen;
    return `<div class="chat lay-${lay} ${insp ? "" : "no-inspector"}">
      ${isLive() ? liveThreadList() : threadList()}${transcript()}${inspector()}${lay === "ultra" ? mediaPane() : ""}
    </div>`;
  }

  // ---------- CREATE (images) ----------
  function createView() {
    if (isLive()) return liveCreateView();
    const batches = [
      { p: "dark node graph, glowing edges, big type “Build it.”, 16:9", seeds: [21, 22, 23, 24, 25, 26, 27, 28], meta: "thumb-dit-64 · 50 steps · DDIM · 9.1 s" },
      { p: "isometric neural network made of glass, soft studio light", seeds: [31, 32, 33, 34], meta: "thumb-dit-64 · 30 steps · flow matching · 5.0 s" },
    ];
    return `
    <div class="create">
      <aside class="side" aria-label="Image settings">
        <div style="display:flex;align-items:center;gap:8px"><span class="h2">Images</span><span class="grow"></span><span class="chip c-image">thumb-dit-64</span></div>
        <div class="flag">${icon("flask")}<div><b>Uses models you train here.</b> <code>nn.diffusion</code> samples any denoiser built in the studio; loading pretrained text-to-image weights is not in the core yet.</div></div>
        <div class="prompt-box"><textarea id="img-prompt" aria-label="Prompt">dark node graph, glowing edges, big type “Build it.”, 16:9</textarea>
          <div style="display:flex;gap:6px;flex-wrap:wrap"><span class="chip plain">− blurry</span><span class="chip plain">− text artifacts</span><button class="btn ghost" style="height:24px;font-size:12px">${icon("plus")}Negative</button></div></div>
        <div class="field"><label>Aspect</label>
          <div class="ratio">${[["1:1", 14, 14], ["4:3", 16, 12], ["16:9", 18, 10], ["3:4", 12, 16], ["9:16", 10, 18]].map(([l, w, h], i) => `<button aria-pressed="${i === 2}"><i style="width:${w}px;height:${h}px"></i>${l}</button>`).join("")}</div></div>
        <div class="field"><label>Sampler</label><div class="seg" style="width:100%"><button aria-pressed="true" style="flex:1">DDIM</button><button aria-pressed="false" style="flex:1">DDPM</button><button aria-pressed="false" style="flex:1">Flow</button></div></div>
        <div class="field"><label for="steps">Steps <output>50</output></label><input type="range" id="steps" min="1" max="200" value="50"></div>
        <div class="field"><label for="eta">Eta (0 = deterministic) <output>0.0</output></label><input type="range" id="eta" min="0" max="1" step="0.05" value="0"></div>
        <div class="field"><label for="guid">Guidance <output>4.5</output></label><input type="range" id="guid" min="1" max="12" step="0.5" value="4.5"></div>
        <div class="field"><label for="seed">Seed</label><div style="display:flex;gap:6px"><input class="input mono" id="seed" value="1207"><button class="iconbtn" aria-label="Lock seed">${icon("lock")}</button></div></div>
        <button class="btn primary" style="height:40px;justify-content:center">${icon("wand")}Generate 4 <span class="kbd" style="color:inherit;border-color:currentColor;background:transparent;opacity:.7">Ctrl ↵</span></button>
      </aside>
      <section class="gallery" aria-label="Generated images">
        <div>
          <div class="batch-head"><span class="eyebrow">Running</span><span class="p">same prompt, seed 1300</span><span class="mono muted">step 18 of 50</span></div>
          <div class="progress-strip">${[0.95, .85, .72, .6, .48, .36, .24, .12].map((n, i) => `<canvas data-thumb="41" data-noise="${n}" title="step ${i * 6 + 1}"></canvas>`).join("")}</div>
        </div>
        ${batches.map(b => `
          <div>
            <div class="batch-head"><span class="p">${esc(b.p)}</span><span class="mono muted">${b.meta}</span></div>
            <div class="batch">${b.seeds.map(s => `<div class="tile"><canvas data-thumb="${s}"></canvas><span class="step">seed ${1000 + s}</span><div class="ov"><button>Vary</button><button>Upscale</button><button>Reuse</button></div></div>`).join("")}</div>
          </div>`).join("")}
      </section>
    </div>`;
  }

  // ---------- VOICE ----------
  function voiceView() {
    if (isLive()) return liveVoiceView();
    const total = 38.4;
    const tags = ["[calm]", "", "[warm]", "", "[smile]", "[slower]"];
    return `
    <div class="voice">
      <aside class="voices" aria-label="Voices">
        <div class="col-head"><span class="h3">Voices</span><span class="grow"></span><button class="btn ghost" style="height:28px">${icon("plus")}Design</button></div>
        ${M.voices.map(v => `
          <div class="vcard ${v.pick ? "active" : ""}" tabindex="0">
            <div class="vface" style="background:${v.hue}">${v.clone ? icon("plus") : v.name[0]}</div>
            <div style="min-width:0"><div class="n">${esc(v.name)}</div><div class="d">${esc(v.desc)}</div></div>
            ${v.clone ? "" : `<button class="iconbtn" aria-label="Preview ${esc(v.name)}">${icon("play")}</button>`}
          </div>`).join("")}
      </aside>
      <section style="display:flex;flex-direction:column;min-height:0;min-width:0">
        <div class="stage-head"><span class="title">Builder demo voiceover</span><span class="chip c-speech">6 lines · 38.4 s</span><span class="grow"></span>
          <button class="btn">${icon("video")}Replace video</button><button class="btn primary">${icon("speech")}Voice all</button></div>
        <div style="padding:14px 24px 0"><div class="flag">${icon("flask")}<div><b>Runs on OpenAI.</b> No local model provides <code>speech.synthesize</code> yet, so voices come from your OpenAI key and the script is sent to OpenAI. When a local voice model is installed, this screen switches to it without changes.</div></div></div>
        <div class="vo-split">
        <div class="vo-video">
          <div class="player"><canvas class="frame" data-t="${state.t}"></canvas><div class="caption">${esc(lineAt(state.t).text)}</div></div>
          <div class="transport"><button class="play" aria-label="Play">${icon("play")}</button><span class="mono tnum">${fmt(state.t)} / ${fmt(total)}</span>
            <input type="range" id="v-scrub" min="0" max="${total}" step="0.1" value="${state.t}" aria-label="Position"></div>
          <div class="mono muted" style="font-size:11.5px">builder-demo.mp4 · 6 cuts found · lines snap to cuts</div>
        </div>
        <div class="script">
          ${M.script.map((l, i) => `
            <div class="para">
              <div class="who"><span class="chip c-speech">Mara</span><span class="mono muted">${fmt(l.t)} · ${l.d.toFixed(1)} s</span></div>
              <div class="txt" contenteditable="true" spellcheck="false">${tags[i] ? `<span class="tag">${tags[i]}</span> ` : ""}${esc(l.text).replace(", and", ", <span class=\"pause\">0.3 s</span> and")}</div>
            </div>`).join("")}
        </div>
        </div>
        <div class="timeline">
          <div class="tl-track"><span>${icon("speech")} Mara</span><div class="tl-lane">${M.script.map((l, i) => `<div class="tl-clip" style="left:${l.t / total * 100}%;width:${l.d / total * 100 - .6}%;background:color-mix(in srgb,var(--cap-speech) 16%,transparent);color:var(--cap-speech)">${waveBars(40, i + 9)}</div>`).join("")}<span class="playhead"></span></div></div>
        </div>
      </section>
      <aside class="voice-settings" aria-label="Voice settings">
        <span class="h3">Mara</span>
        <div class="field"><label for="v-stab">Stability <output>0.62</output></label><input type="range" id="v-stab" min="0" max="1" step="0.01" value="0.62"></div>
        <div class="field" title="This voice model has no similarity control"><label for="v-sim">Similarity <output>not offered</output></label><input type="range" id="v-sim" min="0" max="1" step="0.01" value="0.8" disabled></div>
        <div class="field"><label for="v-style">Style <output>0.15</output></label><input type="range" id="v-style" min="0" max="1" step="0.01" value="0.15"></div>
        <div class="field"><label for="v-pace">Pace <output>0.95×</output></label><input type="range" id="v-pace" min="0.7" max="1.3" step="0.05" value="0.95"></div>
        <div class="field"><label for="v-pitch">Pitch <output>−1 st</output></label><input type="range" id="v-pitch" min="-6" max="6" step="1" value="-1"></div>
        <div class="field"><label>Emotion tags</label><div style="display:flex;gap:6px;flex-wrap:wrap">${["calm", "warm", "smile", "whisper", "excited", "slower", "pause"].map(t => `<span class="tag">[${t}]</span>`).join("")}</div></div>
        <div class="field"><label>Output</label><div class="seg"><button aria-pressed="true">WAV 48k</button><button aria-pressed="false">MP3</button><button aria-pressed="false">Opus</button></div></div>
      </aside>
    </div>`;
  }

  // ---------- BUILDER ----------
  // A left-to-right node editor. Every socket has a tensor kind (socket shape and wire
  // color) and a shape; a wire can only join an output to an input that accepts it.
  const COL = 204, LANE = { a: 64, b: 300 };
  const kindName = { image: "image tensor [C, H, W]", vector: "vector [N]", seq: "sequence [T, D]" };

  function socketRow(dir, s, nodeId, i) {
    const cls = ["sock", s.kind, s.on ? "on" : "", s.state || ""].join(" ");
    return `<div class="row ${dir}"><span class="${cls}" data-sock="${nodeId}.${dir}${i}" title="${kindName[s.kind]}"></span>
      ${dir === "in" ? `<span class="nm">${s.name}</span><span class="shp">${s.shape}</span>` : `<span class="shp">${s.shape}</span><span class="nm">${s.name}</span>`}</div>`;
  }

  function graphNode(n) {
    return `
    <div class="gnode ${n.id === "c2" ? "selected" : ""} ${n.loose ? "loose" : ""}" data-node="${n.id}" style="left:${24 + n.col * COL}px;top:${LANE[n.lane] + (n.dy || 0)}px;--c:var(--cap-${n.cat})">
      <div class="nh"><span class="sw"></span><b>${esc(n.title)}</b><span class="grow"></span><span class="mono faint">${n.tag || ""}</span></div>
      ${n.ins.map((s, i) => socketRow("in", s, n.id, i)).join("")}
      ${n.p.length ? `<div class="params">${n.p.map(([k, v]) => `<div><span>${k}</span><span>${v}</span></div>`).join("")}</div>` : ""}
      ${n.outs.map((s, i) => socketRow("out", s, n.id, i)).join("")}
    </div>`;
  }

  function builderView() {
    const G = M.graph;
    return `
    <div class="builder">
      <aside class="palette" aria-label="Blocks">
        <div class="col-head"><span class="h3">Blocks</span><span class="grow"></span><span class="mono faint">31 found</span></div>
        <div class="search">${icon("search")}<input class="input" placeholder="Search blocks"></div>
        ${M.palette.map(g => `
          <div class="pal-group"><span class="eyebrow"><span>${g.group}</span>${g.experimental ? `<span class="chip c-train" style="font-size:10px;letter-spacing:0;text-transform:none">auto-found</span>` : ""}</span>
            ${g.items.map(i => `<div class="pal-item" draggable="true" style="--c:var(--cap-${i.cap})"><span class="sw">${i.k}</span><span>${esc(i.name)}</span><span class="sig">${esc(i.sig)}</span></div>`).join("")}
          </div>`).join("")}
      </aside>
      <section class="canvas-wrap" id="canvas" aria-label="Model graph">
        <div class="canvas-tools">
          <div class="seg"><button aria-pressed="true">Graph</button><button aria-pressed="false">Code</button><button aria-pressed="false">Summary</button></div>
          <button class="btn">${icon("wand")}Arrange</button>
          <div class="seg"><button aria-label="Zoom out">−</button><button aria-pressed="true" id="zoom-label">Fit</button><button aria-label="Zoom in">+</button></div>
        </div>
        <div class="world" id="world">
          <div class="frame-group" id="group-res"><span>Residual block</span></div>
          <svg class="edges" id="edges"></svg>
          ${G.nodes.map(graphNode).join("")}
          <div class="drop-warn" id="drop-warn">
            <div><b>Dense takes a vector [N].</b> Add gives an image tensor <span class="mono">32 × 32 × 32</span>.</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap"><button class="btn primary" style="height:28px;font-size:12px">Insert Global pool</button><button class="btn" style="height:28px;font-size:12px">Insert Flatten · 32768</button></div>
          </div>
        </div>
        <div class="canvas-status">
          <span class="chip c-ok">6 blocks connected</span><span class="chip c-warn">Dense (128) is not connected</span>
          <span class="legend-k"><i class="sock image on"></i>image [C,H,W]<i class="sock vector on"></i>vector [N]<i class="sock seq on"></i>sequence [T,D]</span>
          <span class="grow"></span>
          <span class="mono muted">drag from a socket · only matching sockets light up</span>
        </div>
      </section>
      <aside class="props" aria-label="Block settings">
        <div><span class="eyebrow">Selected · in Residual block</span><div class="h2" style="margin-top:4px">Conv</div></div>
        <div class="field"><label for="p-f">Filters</label><input class="input mono" id="p-f" value="32"></div>
        <div class="field"><label for="p-k">Kernel</label><div style="display:flex;gap:6px"><input class="input mono" id="p-k" value="3"><input class="input mono" id="p-k2" value="3" aria-label="Kernel width"></div></div>
        <div class="field"><label for="p-s">Stride</label><input class="input mono" id="p-s" value="1"></div>
        <div class="field"><label>Padding</label><div class="seg"><button aria-pressed="false">Valid</button><button aria-pressed="true">Same</button><button aria-pressed="false">Full</button></div></div>
        <div class="field"><label for="p-a">Activation</label><select class="input" id="p-a"><option selected>None</option><option>ReLU</option><option>SiLU</option><option>GELU</option></select></div>
        <div class="field"><label>Shape</label><div class="mono" style="display:flex;gap:8px;align-items:center;color:var(--fg-2)">32 × 32 × 32 ${icon("chev")} <b style="color:var(--fg)">32 × 32 × 32</b></div>
          <span class="muted" style="font-size:12px">Stride 2 would make 32 × 16 × 16, which Add cannot sum with its other input, so the field refuses it and says why.</span></div>
        <div style="display:grid;gap:8px"><span class="eyebrow">Whole model</span>
          <div class="sum">
            <div><b class="tnum">10.5k</b><span>parameters</span></div><div><b class="tnum">41 KB</b><span>weights, float32</span></div>
            <div><b class="tnum">10.3 M</b><span>multiply-adds per image</span></div><div><b class="tnum">[10]</b><span>output</span></div>
          </div>
        </div>
        <button class="btn primary" style="justify-content:center;height:38px">${icon("train")}Train this model</button>
      </aside>
    </div>`;
  }

  function sockPos(id) {
    const s = document.querySelector(`[data-sock="${id}"]`); if (!s) return null;
    const node = s.closest(".gnode"), row = s.parentElement;
    return {
      x: node.offsetLeft + row.offsetLeft + s.offsetLeft + s.offsetWidth / 2,
      y: node.offsetTop + row.offsetTop + s.offsetTop + s.offsetHeight / 2,
      kind: [...s.classList].find(k => kindName[k]),
    };
  }
  function wire(a, b) {
    const k = Math.max(40, Math.abs(b.x - a.x) / 2);
    return `M${a.x},${a.y} C${a.x + k},${a.y} ${b.x - k},${b.y} ${b.x},${b.y}`;
  }

  function drawEdges() {
    const svg = document.getElementById("edges"); if (!svg) return;
    let out = M.graph.edges.map(([from, to]) => {
      const a = sockPos(from), b = sockPos(to); if (!a || !b) return "";
      return `<path d="${wire(a, b)}" class="wire k-${a.kind}"/>`;
    }).join("");
    // the wire being dragged right now, refused by the Dense input
    const a = sockPos("add.out0"), t = sockPos("d128.in0");
    if (a && t) {
      const tip = { x: t.x - 26, y: t.y - 14 };
      out += `<path d="${wire(a, tip)}" class="wire ghost"/><circle cx="${tip.x}" cy="${tip.y}" r="4" class="ghost-end"/>`;
      const warn = document.getElementById("drop-warn");
      const dn = document.querySelector('[data-node="d128"]'); warn.style.left = dn.offsetLeft + "px"; warn.style.top = (dn.offsetTop + dn.offsetHeight + 12) + "px";
    }
    svg.innerHTML = out;
    const g = document.getElementById("group-res"), c2 = document.querySelector('[data-node="c2"]'), add = document.querySelector('[data-node="add"]');
    if (g && c2 && add) {
      const x0 = c2.offsetLeft - 14, y0 = add.offsetTop - 34, x1 = add.offsetLeft + add.offsetWidth + 14, y1 = c2.offsetTop + c2.offsetHeight + 14;
      Object.assign(g.style, { left: x0 + "px", top: y0 + "px", width: (x1 - x0) + "px", height: (y1 - y0) + "px" });
    }
  }

  function fitWorld() {
    const world = document.getElementById("world"), canvas = document.getElementById("canvas"); if (!world) return;
    const nodes = [...world.querySelectorAll(".gnode")];
    const w = Math.max(...nodes.map(n => n.offsetLeft + n.offsetWidth)) + 40;
    const h = Math.max(...nodes.map(n => n.offsetTop + n.offsetHeight)) + 150;
    world.style.width = w + "px"; world.style.height = h + "px";
    const s = Math.min(1, (canvas.clientWidth - 24) / w, (canvas.clientHeight - 70) / h);
    world.style.transform = `scale(${s})`;
    world.dataset.scale = s;
    const z = document.getElementById("zoom-label"); if (z) z.textContent = `Fit · ${Math.round(s * 100)}%`;
  }

  function enableDrag() {
    document.querySelectorAll(".gnode").forEach(el => {
      el.addEventListener("pointerdown", e => {
        if (e.button !== 0 || e.target.classList.contains("sock")) return;
        document.querySelectorAll(".gnode.selected").forEach(n => n.classList.remove("selected"));
        el.classList.add("selected");
        const s = +(document.getElementById("world").dataset.scale || 1);
        const sx = e.clientX, sy = e.clientY, ox = el.offsetLeft, oy = el.offsetTop;
        el.setPointerCapture(e.pointerId);
        const move = ev => { el.style.left = ox + (ev.clientX - sx) / s + "px"; el.style.top = oy + (ev.clientY - sy) / s + "px"; drawEdges(); };
        const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); };
        el.addEventListener("pointermove", move); el.addEventListener("pointerup", up);
      });
    });
  }

  // ---------- live builder ----------
  // Served by the studio, the builder offers every layer the installed modules have, found from
  // their API, and checks each change with the core's own shape rules: a wire or a value that
  // would break a block that worked is refused, and the reason is shown.
  const B = {
    catalog: null,                        // GET /api/builder/layers
    graph: store.get("builderDraft", null) || { name: "my-model", nodes: [{ id: "in", op: "input", shape: [3, 32, 32], x: 40, y: 120 }] },
    check: null,                          // the last check: {ok, nodes: {id: {shape} | {error}}, input, output, outputShape}
    sel: null,                            // the selected block's id
    search: "",
    refusal: null,                        // {id, text}: the last change that was refused, and why
    built: null,                          // the last build: {parameters, file, summary, ...}
    building: false,
    graphs: [],                           // saved graphs
    view: store.get("builderView", { x: 0, y: 0, s: 1 }),
    openGroups: new Set(),
  };
  const opLabel = { input: "Input", add: "Add", multiply: "Multiply", concat: "Concat" };
  // a block still being wired: not an error to refuse, only something to finish
  const incomplete = err => /not connected|needs at least two inputs/.test(err || "");
  const entryOf = n => n.op === "layer" ? B.catalog?.entries.find(e => e.key === n.entry) : null;
  const entryLabel = e => e.group + (["new", "of"].includes(e.name) ? "" : "." + e.name);
  const nodeLabel = n => n.op === "layer" ? (entryOf(n) ? entryLabel(entryOf(n)) : n.entry.split("#").pop()) : opLabel[n.op];
  const shapeOf = id => B.check?.nodes?.[id]?.shape || null;
  const errorOf = id => B.check?.nodes?.[id]?.error || null;
  const kindOf = s => !s ? "vector" : s.length >= 3 ? "image" : s.length === 2 ? "seq" : "vector";
  const fmtShape = s => s ? s.join(" × ") : "?";
  const multiIn = n => n.op === "add" || n.op === "multiply" || n.op === "concat" || (n.op === "layer" && entryOf(n)?.multiInput);
  const fmtCount = n => n >= 1e9 ? (n / 1e9).toFixed(2) + " B" : n >= 1e6 ? (n / 1e6).toFixed(2) + " M" : n >= 1e3 ? (n / 1e3).toFixed(1) + " k" : String(n);
  function saveDraft() { store.set("builderDraft", B.graph); }

  // A first value for each parameter, so a new block works at once; the person changes it after.
  function guess(p) {
    const n = p.name.toLowerCase();
    if (p.kind === "random") return null;
    if (p.kind === "choice") return p.choices.includes("RELU") ? "RELU" : p.choices.includes("HE") ? "HE" : p.choices[0];
    if (p.kind === "boolean") return false;
    if (p.kind === "string") return "";
    let v = 1;
    if (p.kind === "float" || p.kind === "double") v = /rate|drop|prob|eps|alpha|slope|momentum/.test(n) ? 0.1 : 1;
    else if (/kernel|window|size$/.test(n) && p.array) v = 3;
    else if (/kernel|window/.test(n)) v = 3;
    else if (/head/.test(n)) v = 4;
    else if (/units|filters|features|dim|channels|hidden|width|out|size|vocab|capacity|experts|slots/.test(n)) v = 64;
    else if (/group|stride|dilation|multiplier/.test(n)) v = 1;   // the values that change nothing
    else if (/factor|scale/.test(n)) v = 2;
    return p.array ? (/kernel|window/.test(n) ? [v, v] : [v]) : v;
  }

  // The graph as the studio reads it: no positions, options only when switched on.
  function wireGraph(g) {
    return {
      output: g.output || undefined,
      nodes: g.nodes.map(n => ({
        id: n.id, op: n.op, shape: n.shape, entry: n.entry, inputs: n.inputs || [],
        args: n.args, options: Object.entries(n.options || {}).map(([key, args]) => ({ key, args })),
      })),
    };
  }

  let checkSeq = 0;
  async function runCheck(g) {
    const seq = ++checkSeq;
    const r = await postJson("/api/builder/check", { graph: wireGraph(g) });
    return seq === checkSeq ? r : null;
  }

  // Applies a change, checks it, and keeps it only if no block that worked stops working (and
  // the block it is aimed at works). Returns true if it was kept.
  async function tryChange(change, focus) {
    const before = JSON.parse(JSON.stringify(B.graph)), was = B.check;
    change(B.graph);
    let r;
    try { r = await runCheck(B.graph); } catch (e) { B.graph = before; B.refusal = { id: focus, text: e.message }; renderBuilder(); return false; }
    if (!r) return false;
    const broken = Object.entries(r.nodes).filter(([id, v]) => v.error && (!was || was.nodes[id]?.shape || id === focus))
      .filter(([id, v]) => !(id === focus && incomplete(v.error)));
    if (broken.length) {
      B.graph = before;
      const [id, v] = broken.find(([id]) => id === focus) || broken[0];
      B.refusal = { id: focus || id, text: `${nodeLabel(B.graph.nodes.find(n => n.id === id) || { op: "input" })}: ${v.error}` };
      renderBuilder();
      return false;
    }
    B.check = r; B.refusal = null; B.built = null;
    saveDraft();
    renderBuilder();
    return true;
  }

  async function recheck() {
    try { const r = await runCheck(B.graph); if (r) { B.check = r; renderBuilder(); } } catch (e) { console.warn("cannot check the graph", e); }
  }

  // Opens a saved drawing on the builder screen.
  async function openGraph(name) {
    const g = B.graphs.find(x => x.name === name);
    if (!g) return;
    B.graph = { name: g.name, output: g.output, nodes: g.nodes.map((n, i) => ({ x: 40 + i * 250, y: 120, ...n, options: Array.isArray(n.options) ? Object.fromEntries(n.options.map(o => [o.key, o.args])) : n.options || {} })) };
    B.sel = null; B.built = null; B.refusal = null;
    saveDraft();
    if (state.view !== "builder") go("builder");
    await recheck(); renderBuilder();
  }

  async function loadBuilder() {
    try {
      const [cat, gs] = await Promise.all([api("/api/builder/layers"), api("/api/builder/graphs")]);
      B.catalog = cat; B.graphs = gs.graphs || [];
      await recheck();
      if (state.view === "builder") render();
    } catch (e) { console.warn("cannot load the builder", e); }
  }

  function newId(base) {
    const b = base.toLowerCase().replace(/[^a-z0-9]+/g, "").slice(0, 12) || "block";
    for (let i = 1; ; i++) if (!B.graph.nodes.some(n => n.id === b + i)) return b + i;
  }

  // Adds a block; from the selected block when there is one and the wire works.
  async function addBlock(spec, at) {
    const sel = B.graph.nodes.find(n => n.id === B.sel);
    const node = { id: newId(spec.label), inputs: [], x: 0, y: 0, ...spec.node };
    delete node.label;
    if (at) { node.x = at.x; node.y = at.y; }
    else if (sel) { node.x = sel.x + 240; node.y = sel.y; }
    else { node.x = 40 + 240 * Math.min(6, B.graph.nodes.length); node.y = 120; }
    // nudge it off any block already there
    while (B.graph.nodes.some(n => Math.abs(n.x - node.x) < 60 && Math.abs(n.y - node.y) < 60)) node.y += 90;
    B.graph.nodes.push(node);
    B.sel = node.id;
    if (sel && !at && node.op !== "input") {
      const ok = await tryChange(g => { g.nodes.find(n => n.id === node.id).inputs = [sel.id]; }, node.id);
      if (ok) return;
    }
    saveDraft();
    await recheck();
    renderBuilder();
  }

  function paletteSpec(e) {
    return { label: e.group, node: { op: "layer", entry: e.key, args: e.params.map(guess), options: {} } };
  }

  function liveBuilderView() {
    if (!B.catalog) return `<div class="builder live-builder"><div class="empty" style="grid-column:1/-1"><span class="typing"><i></i><i></i><i></i></span><div class="muted">Looking for every layer the installed modules offer…</div></div></div>`;
    return `<div class="builder live-builder">${builderPalette()}${builderCanvas()}${builderProps()}</div>`;
  }

  function builderPalette() {
    const q = B.search.toLowerCase();
    const match = e => !q || (e.group + " " + e.name + " " + e.params.map(p => p.name).join(" ")).toLowerCase().includes(q);
    const groups = new Map();
    B.catalog.entries.filter(match).forEach(e => { const k = (e.experimental ? "x:" : "c:") + e.group; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(e); });
    const item = e => `<button class="pal-item" draggable="true" data-entry="${esc(e.key)}" title="${esc(e.key)}">
        <span class="sw">${esc(e.group.slice(0, 2))}</span><span class="pn">${esc(["new", "of"].includes(e.name) ? e.group : e.name)}</span>
        <span class="sig">${esc(e.params.filter(p => p.kind !== "random").map(p => p.name + (p.array ? "…" : "")).join(", "))}</span></button>`;
    const section = (title, exp) => {
      const gs = [...groups].filter(([k]) => k.startsWith(exp ? "x:" : "c:"));
      if (!gs.length) return "";
      return `<div class="pal-sec"><span class="eyebrow">${title} <span class="faint">${gs.reduce((a, [, l]) => a + l.length, 0)}</span></span>
        ${gs.map(([k, list]) => {
          const name = k.slice(2), open = q || B.openGroups.has(k) || list.length === 1;
          return list.length === 1 ? item(list[0]) : `<div class="pal-fold ${open ? "open" : ""}">
            <button class="pal-head" data-pal-group="${esc(k)}" aria-expanded="${!!open}">${icon("chev")}<span>${esc(name)}</span><span class="faint">${list.length}</span></button>
            ${open ? list.map(item).join("") : ""}</div>`;
        }).join("")}</div>`;
    };
    return `<aside class="palette" aria-label="Blocks">
      <div class="col-head"><span class="h3">Blocks</span><span class="grow"></span><span class="mono faint" title="Found in the installed modules' API${B.catalog.skipped ? `; ${B.catalog.skipped} left out because they need something the builder cannot fill in, such as a tensor of weights` : ""}">${B.catalog.entries.length} found</span></div>
      <div class="search">${icon("search")}<input class="input" id="block-search" placeholder="Search blocks and options" value="${esc(B.search)}"></div>
      <div class="pal-sec"><span class="eyebrow">Wiring</span>
        ${[["input", "Input", "shape"], ["add", "Add", "same shapes"], ["multiply", "Multiply", "same shapes"], ["concat", "Concat", "along the first dim"]].map(([op, name, sig]) =>
          `<button class="pal-item" draggable="true" data-op="${op}"><span class="sw">${name.slice(0, 2)}</span><span class="pn">${name}</span><span class="sig">${sig}</span></button>`).join("")}
      </div>
      ${section("Core", false)}${section("Experimental", true)}
      ${groups.size ? "" : `<div class="muted" style="padding:8px 16px;font-size:12.5px">Nothing matches.</div>`}
    </aside>`;
  }

  function builderCanvas() {
    const v = B.view, nodes = B.graph.nodes;
    const errs = nodes.filter(n => errorOf(n.id)), loose = errs.filter(n => incomplete(errorOf(n.id)));
    const outId = B.check?.output;
    return `<section class="canvas-wrap" id="canvas" aria-label="Model graph">
      <div class="canvas-tools">
        <input class="input mono" id="graph-name" aria-label="Model name" value="${esc(B.graph.name || "")}" style="width:180px;height:32px">
        <select class="input" id="graph-open" aria-label="Open a saved model" style="width:auto;height:32px">
          <option value="">Open…</option>${B.graphs.map(g => `<option value="${esc(g.name)}">${esc(g.name)}${g.built ? " · built" : ""}</option>`).join("")}
          <option value="__new">New, empty</option></select>
        <button class="btn" id="arrange">${icon("wand")}Arrange</button>
        <div class="seg"><button id="zoom-out" aria-label="Zoom out">−</button><button id="zoom-fit">${Math.round(v.s * 100)}%</button><button id="zoom-in" aria-label="Zoom in">+</button></div>
      </div>
      <div class="world" id="world" style="transform:translate(${v.x}px,${v.y}px) scale(${v.s})">
        <svg class="edges" id="edges"></svg>
        ${nodes.map(n => liveNode(n, n.id === outId)).join("")}
      </div>
      <div class="canvas-status">
        ${!B.check ? `<span class="chip plain">checking…</span>` : B.check.ok ? `<span class="chip c-ok">${icon("check")}${nodes.length} blocks · checks out · output ${fmtShape(B.check.outputShape)}</span>`
          : `<span class="chip c-warn">${errs.length ? `${errs.length} block${errs.length === 1 ? "" : "s"} ${loose.length === errs.length ? "not connected" : "to fix"}` : esc(B.check.problem || "not finished")}</span>`}
        ${B.refusal ? `<span class="chip c-bad" title="${esc(B.refusal.text)}">Refused: ${esc(B.refusal.text.length > 90 ? B.refusal.text.slice(0, 90) + "…" : B.refusal.text)}</span>` : ""}
        <span class="legend-k"><i class="sock image on"></i>image [C,H,W]<i class="sock vector on"></i>vector [N]<i class="sock seq on"></i>sequence [T,D]</span>
        <span class="grow"></span>
        <span class="mono muted">drag a block in · drag from an output to an input · Delete removes</span>
      </div>
    </section>`;
  }

  function liveNode(n, isOut) {
    const e = entryOf(n), err = errorOf(n.id), shp = shapeOf(n.id);
    const ins = n.op === "input" ? [] : multiIn(n) ? [...(n.inputs || []), null] : [(n.inputs || [])[0] || null];
    const vals = [];
    if (n.op === "input") vals.push(["shape", fmtShape(n.shape)]);
    if (e) e.params.forEach((p, i) => { if (p.kind !== "random") vals.push([p.name, fmtVal(n.args?.[i])]); });
    Object.entries(n.options || {}).forEach(([k, a]) => vals.push([k.replace(/\(.*$/, ""), a.map(fmtVal).join(", ") || "on"]));
    return `<div class="gnode live ${n.id === B.sel ? "selected" : ""} ${err ? (incomplete(err) ? "loose" : "bad") : ""} ${e?.experimental ? "exp" : ""}"
        data-node="${esc(n.id)}" style="left:${n.x}px;top:${n.y}px;--c:var(--cap-${n.op === "input" ? "vision" : n.op === "layer" ? (e?.experimental ? "train" : "text") : "agent"})">
      <div class="nh"><span class="sw"></span><b>${esc(nodeLabel(n))}</b><span class="grow"></span>${isOut ? `<span class="chip c-ok" style="font-size:10px">output</span>` : `<span class="mono faint">${esc(n.id)}</span>`}</div>
      ${ins.map((src, i) => `<div class="row in"><span class="sock ${kindOf(shapeOf(src))} ${src ? "on" : ""}" data-in="${esc(n.id)}" data-slot="${i}" title="${src ? "from " + esc(src) : "drop a wire here"}"></span>
        <span class="nm">${src ? esc(src) : multiIn(n) && i ? "+" : "in"}</span><span class="shp">${src ? fmtShape(shapeOf(src)) : ""}</span></div>`).join("")}
      ${vals.length ? `<div class="params">${vals.slice(0, 6).map(([k, v]) => `<div><span>${esc(k)}</span><span>${esc(v)}</span></div>`).join("")}${vals.length > 6 ? `<div><span>+${vals.length - 6} more</span><span></span></div>` : ""}</div>` : ""}
      ${err ? `<div class="nerr">${esc(err)}</div>` : ""}
      <div class="row out"><span class="shp">${shp ? fmtShape(shp) : ""}</span><span class="nm">out</span><span class="sock ${kindOf(shp)} ${shp ? "on" : ""}" data-out="${esc(n.id)}" title="drag to an input"></span></div>
    </div>`;
  }

  function fmtVal(v) {
    if (v == null || v === "") return "–";
    if (Array.isArray(v)) return v.join(" × ");
    if (typeof v === "boolean") return v ? "yes" : "no";
    return String(v);
  }

  function fieldFor(p, v, path) {
    const id = "f-" + path.replace(/[^a-z0-9]/gi, "_");
    const label = `<label for="${id}">${esc(p.name)} <span class="faint mono">${esc(p.type)}</span></label>`;
    if (p.kind === "random") return "";
    if (p.kind === "boolean" && !p.array) return `<label class="check"><input type="checkbox" id="${id}" data-field="${esc(path)}" ${v ? "checked" : ""}> ${esc(p.name)}</label>`;
    if (p.kind === "choice" && !p.array) return `<div class="field">${label}<select class="input" id="${id}" data-field="${esc(path)}">${p.choices.map(c => `<option ${c === v ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></div>`;
    const shown = Array.isArray(v) ? v.join(", ") : v == null ? "" : String(v);
    return `<div class="field">${label}<input class="input mono" id="${id}" data-field="${esc(path)}" value="${esc(shown)}" ${p.array ? `placeholder="${p.kind === "choice" ? "names, comma separated" : "numbers, comma separated"}"` : ""} spellcheck="false"></div>`;
  }

  function parseVal(p, raw) {
    if (p.kind === "boolean") return !!raw;
    if (p.kind === "choice" && !p.array) return raw;
    const one = s => p.kind === "string" || p.kind === "choice" ? s : s === "" ? null : Number(s);
    if (p.array) return String(raw).split(/[\s,×x]+/).filter(Boolean).map(one);
    return one(String(raw).trim());
  }

  function builderProps() {
    const n = B.graph.nodes.find(x => x.id === B.sel);
    if (!n) {
      const c = B.check, b = B.built;
      return `<aside class="props" aria-label="Model">
        <div><span class="eyebrow">Whole model</span><div class="h2" style="margin-top:4px">${esc(B.graph.name || "Untitled")}</div></div>
        <div class="sum">
          <div><b class="tnum">${B.graph.nodes.length}</b><span>blocks</span></div>
          <div><b class="tnum">${c?.input ? fmtShape(shapeOf(c.input)) : "–"}</b><span>input</span></div>
          <div><b class="tnum">${c?.outputShape ? fmtShape(c.outputShape) : "–"}</b><span>output</span></div>
          <div><b class="tnum">${b ? fmtCount(b.parameters) : "–"}</b><span>${b ? `parameters · ${fmtCount(b.parameters * 4)}B float32` : "parameters, once built"}</span></div>
        </div>
        ${c && !c.ok ? `<div class="flag">${icon("flask")}<div>${esc(c.problem || "Some blocks do not check out yet; they are marked on the canvas.")}</div></div>` : ""}
        <button class="btn primary" id="build" style="justify-content:center;height:38px" ${c?.ok && !B.building ? "" : "disabled"}>${icon("build")}${B.building ? "Building…" : "Build and save"}</button>
        ${b ? `<div class="flag ok">${icon("check")}<div><b>Built.</b> ${b.file ? `Saved to <code>${esc(b.file)}</code>.` : esc(b.note || "")}</div></div>
          <details class="tool"><summary>${icon("layout")}<span><b>Summary</b></span></summary><pre>${esc(b.summary)}</pre></details>` : ""}
        <p class="muted" style="font-size:12px;margin:0">Click a block to change it. Every layer and option here comes from the installed modules; a change that would break a block that works is refused.</p>
      </aside>`;
    }
    const e = entryOf(n), err = errorOf(n.id), refused = B.refusal && B.refusal.id === n.id ? B.refusal.text : null;
    const inShape = (n.inputs || []).map(shapeOf).filter(Boolean).map(fmtShape).join(" + ");
    const opts = e ? e.options : [];
    const oq = (B.optSearch || "").toLowerCase();
    return `<aside class="props" aria-label="Block settings">
      <div><span class="eyebrow">${n.op === "layer" ? esc(e?.module?.replace("io.github.zzuegg.", "") || "") : "wiring"}${e?.experimental ? ` · <span class="chip c-train" style="font-size:10px">experimental</span>` : ""}</span>
        <div class="h2" style="margin-top:4px">${esc(nodeLabel(n))}</div>${e ? `<div class="mono faint" style="font-size:11px;overflow-wrap:anywhere">${esc(e.key.replace(/^io\.github\.zzuegg\.brainiac\./, ""))}</div>` : ""}</div>
      <div class="field"><label>Shape</label><div class="mono" style="display:flex;gap:8px;align-items:center;color:var(--fg-2)">${esc(inShape || (n.op === "input" ? "" : "not connected"))} ${n.op === "input" ? "" : icon("chev")} <b style="color:var(--fg)">${esc(fmtShape(shapeOf(n.id)))}</b></div></div>
      ${refused ? `<div class="flag bad">${icon("flask")}<div><b>Refused.</b> ${esc(refused)}</div></div>` : err ? `<div class="flag">${icon("flask")}<div>${esc(err)}</div></div>` : ""}
      ${n.op === "input" ? `<div class="field"><label for="f-shape">Shape of one sample <span class="faint">channels first, e.g. 3, 32, 32</span></label><input class="input mono" id="f-shape" data-field="shape" value="${esc((n.shape || []).join(", "))}"></div>` : ""}
      ${e ? e.params.map((p, i) => fieldFor(p, n.args?.[i], "arg." + i)).join("") : ""}
      ${opts.length ? `<div class="optbox"><div class="optbox-head"><span class="eyebrow">Options <span class="faint">${Object.keys(n.options || {}).length} of ${opts.length} on</span></span></div>
        ${opts.length > 6 ? `<input class="input" id="opt-search" placeholder="Find an option" value="${esc(B.optSearch || "")}" style="height:30px">` : ""}
        ${opts.filter(o => !oq || o.key.toLowerCase().includes(oq)).map(o => {
          const on = n.options && o.key in n.options;
          return `<div class="opt ${on ? "on" : ""}"><label class="check"><input type="checkbox" data-opt="${esc(o.key)}" ${on ? "checked" : ""}> <span class="mono">${esc(o.name)}</span><span class="faint mono" style="font-size:10.5px">(${esc(o.params.map(p => p.type).join(", "))})</span></label>
            ${on ? o.params.map((p, i) => fieldFor(p, n.options[o.key][i], "opt." + o.key + "." + i)).join("") : ""}</div>`;
        }).join("")}</div>` : ""}
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        ${n.op !== "input" ? `<button class="btn ghost sm" id="mark-output">${B.graph.output === n.id ? "Output (chosen)" : "Make this the output"}</button>` : ""}
        <button class="btn ghost sm" id="del-block">Remove block</button>
      </div>
    </aside>`;
  }

  function renderBuilder() {
    if (state.view !== "builder" || !isLive() || !B.catalog) return;
    const keep = document.activeElement?.id, pos = document.activeElement?.selectionStart;
    const pal = $(".palette")?.scrollTop, props = $(".props")?.scrollTop;
    $("#main .view").innerHTML = liveBuilderView();
    if ($(".palette")) $(".palette").scrollTop = pal || 0;
    if ($(".props")) $(".props").scrollTop = props || 0;
    bindLiveBuilder();
    if (keep) { const el = document.getElementById(keep); if (el) { el.focus({ preventScroll: true }); try { if (pos != null) el.setSelectionRange(pos, pos); } catch { /* not a text field */ } } }
  }

  function sockXY(el) {
    const world = $("#world"), wr = world.getBoundingClientRect(), r = el.getBoundingClientRect(), s = B.view.s;
    return { x: (r.left + r.width / 2 - wr.left) / s, y: (r.top + r.height / 2 - wr.top) / s };
  }

  function drawLiveEdges(extra) {
    const svg = $("#edges"); if (!svg) return;
    let out = "";
    B.graph.nodes.forEach(n => (n.inputs || []).forEach((src, i) => {
      const a = $(`[data-out="${CSS.escape(src)}"]`), b = $(`[data-in="${CSS.escape(n.id)}"][data-slot="${i}"]`);
      if (!a || !b) return;
      out += `<path d="${wire(sockXY(a), sockXY(b))}" class="wire k-${kindOf(shapeOf(src))}"/>`;
    }));
    if (extra) out += `<path d="${wire(extra.a, extra.b)}" class="wire drafting k-${extra.kind}"/>`;
    svg.innerHTML = out;
  }

  function setView(v) {
    B.view = v; store.set("builderView", v);
    const w = $("#world"); if (w) w.style.transform = `translate(${v.x}px,${v.y}px) scale(${v.s})`;
    const z = $("#zoom-fit"); if (z) z.textContent = Math.round(v.s * 100) + "%";
  }

  function fitLive() {
    const c = $("#canvas"), nodes = $$(".gnode.live"); if (!c || !nodes.length) return;
    const xs = nodes.map(n => n.offsetLeft), ys = nodes.map(n => n.offsetTop);
    const x2 = Math.max(...nodes.map(n => n.offsetLeft + n.offsetWidth)), y2 = Math.max(...nodes.map(n => n.offsetTop + n.offsetHeight));
    const x1 = Math.min(...xs), y1 = Math.min(...ys);
    const s = Math.max(0.3, Math.min(1.25, (c.clientWidth - 48) / (x2 - x1 + 1), (c.clientHeight - 130) / (y2 - y1 + 1)));
    setView({ s, x: 24 - x1 * s, y: 60 - y1 * s });
    drawLiveEdges();
  }

  // Lays the blocks out left to right by depth.
  function arrange() {
    const depth = {}, byId = Object.fromEntries(B.graph.nodes.map(n => [n.id, n]));
    const d = (id, seen = new Set()) => {
      if (depth[id] != null) return depth[id];
      if (seen.has(id)) return 0;
      seen.add(id);
      const n = byId[id];
      return depth[id] = n && n.inputs?.length ? 1 + Math.max(...n.inputs.map(s => d(s, seen))) : 0;
    };
    const cols = {};
    B.graph.nodes.forEach(n => { const k = d(n.id); (cols[k] ||= []).push(n); });
    Object.entries(cols).forEach(([k, list]) => list.forEach((n, i) => { n.x = 40 + k * 250; n.y = 80 + i * 210; }));
    saveDraft();
    renderBuilder();
    requestAnimationFrame(fitLive);
  }

  async function removeBlock(id) {
    B.graph.nodes = B.graph.nodes.filter(n => n.id !== id);
    B.graph.nodes.forEach(n => { n.inputs = (n.inputs || []).filter(s => s !== id); });
    if (B.graph.output === id) delete B.graph.output;
    B.sel = null; B.refusal = null; B.built = null;
    saveDraft();
    await recheck();
    renderBuilder();
  }

  async function connect(src, dst, slot) {
    if (src === dst) return;
    const target = B.graph.nodes.find(n => n.id === dst);
    if (!target || target.op === "input") return;
    await tryChange(g => {
      const t = g.nodes.find(n => n.id === dst);
      const ins = [...(t.inputs || [])];
      if (multiIn(t)) { if (slot < ins.length) ins[slot] = src; else ins.push(src); } else ins[0] = src;
      t.inputs = [...new Set(ins)];
    }, dst);
  }

  function bindLiveBuilder() {
    $("#block-search")?.addEventListener("input", e => { B.search = e.target.value; renderBuilder(); });
    $$("[data-pal-group]").forEach(b => b.addEventListener("click", () => { const k = b.dataset.palGroup; B.openGroups.has(k) ? B.openGroups.delete(k) : B.openGroups.add(k); renderBuilder(); }));
    const specOf = el => el.dataset.op ? { label: el.dataset.op, node: el.dataset.op === "input" ? { op: "input", shape: [16] } : { op: el.dataset.op } }
      : paletteSpec(B.catalog.entries.find(e => e.key === el.dataset.entry));
    $$(".pal-item").forEach(el => {
      el.addEventListener("click", () => addBlock(specOf(el)));
      el.addEventListener("dragstart", e => { e.dataTransfer.setData("text/x-block", el.dataset.entry || "op:" + el.dataset.op); e.dataTransfer.effectAllowed = "copy"; });
    });
    const canvas = $("#canvas");
    if (canvas) {
      canvas.addEventListener("dragover", e => { if (e.dataTransfer.types.includes("text/x-block")) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } });
      canvas.addEventListener("drop", e => {
        const k = e.dataTransfer.getData("text/x-block"); if (!k) return;
        e.preventDefault();
        const wr = $("#world").getBoundingClientRect();
        const at = { x: (e.clientX - wr.left) / B.view.s - 90, y: (e.clientY - wr.top) / B.view.s - 20 };
        const el = k.startsWith("op:") ? { dataset: { op: k.slice(3) } } : { dataset: { entry: k } };
        addBlock(specOf(el), at);
      });
      // pan on the background, zoom with the wheel
      canvas.addEventListener("pointerdown", e => {
        if (e.button !== 0 || e.target.closest(".gnode, .canvas-tools, .canvas-status")) return;
        B.sel = null; B.refusal = null;
        const sx = e.clientX, sy = e.clientY, v0 = { ...B.view };
        let moved = false;
        canvas.setPointerCapture(e.pointerId);
        const move = ev => { moved = true; setView({ ...v0, x: v0.x + ev.clientX - sx, y: v0.y + ev.clientY - sy }); };
        const up = () => { canvas.removeEventListener("pointermove", move); canvas.removeEventListener("pointerup", up); if (!moved) renderBuilder(); };
        canvas.addEventListener("pointermove", move); canvas.addEventListener("pointerup", up);
      });
      canvas.addEventListener("wheel", e => {
        e.preventDefault();
        const r = canvas.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top, v = B.view;
        const s = Math.max(0.25, Math.min(2, v.s * (e.deltaY < 0 ? 1.1 : 1 / 1.1)));
        setView({ s, x: px - (px - v.x) * s / v.s, y: py - (py - v.y) * s / v.s });
        drawLiveEdges();
      }, { passive: false });
    }
    $("#zoom-in")?.addEventListener("click", () => { setView({ ...B.view, s: Math.min(2, B.view.s * 1.2) }); drawLiveEdges(); });
    $("#zoom-out")?.addEventListener("click", () => { setView({ ...B.view, s: Math.max(0.25, B.view.s / 1.2) }); drawLiveEdges(); });
    $("#zoom-fit")?.addEventListener("click", fitLive);
    $("#arrange")?.addEventListener("click", arrange);
    $("#graph-name")?.addEventListener("change", e => { B.graph.name = e.target.value.trim() || "my-model"; saveDraft(); });
    $("#graph-open")?.addEventListener("change", e => {
      const v = e.target.value; if (!v) return;
      if (v === "__new") newModel(); else openGraph(v);
    });
    // blocks: select and move; wires from an output socket to an input socket
    $$(".gnode.live").forEach(el => {
      el.addEventListener("pointerdown", e => {
        if (e.button !== 0) return;
        const out = e.target.closest("[data-out]");
        const n = B.graph.nodes.find(x => x.id === el.dataset.node);
        e.stopPropagation();
        if (out) {
          const a = sockXY(out), kind = kindOf(shapeOf(n.id));
          const move = ev => { const wr = $("#world").getBoundingClientRect(); drawLiveEdges({ a, b: { x: (ev.clientX - wr.left) / B.view.s, y: (ev.clientY - wr.top) / B.view.s }, kind }); };
          const up = ev => {
            document.removeEventListener("pointermove", move); document.removeEventListener("pointerup", up);
            const hit = document.elementFromPoint(ev.clientX, ev.clientY);
            const sock = hit?.closest("[data-in]"), node = hit?.closest(".gnode.live");
            drawLiveEdges();
            if (sock) connect(n.id, sock.dataset.in, +sock.dataset.slot);
            else if (node && node.dataset.node !== n.id) connect(n.id, node.dataset.node, 99);
          };
          document.addEventListener("pointermove", move); document.addEventListener("pointerup", up);
          return;
        }
        const inSock = e.target.closest("[data-in]");
        if (inSock && inSock.classList.contains("on")) {
          // pulling a wire off an input disconnects it
          const slot = +inSock.dataset.slot;
          n.inputs = n.inputs.filter((_, i) => i !== slot);
          B.built = null; saveDraft(); recheck().then(renderBuilder);
          return;
        }
        if (B.sel !== n.id) { B.sel = n.id; B.optSearch = ""; B.refusal = B.refusal && B.refusal.id === n.id ? B.refusal : null; }
        const sx = e.clientX, sy = e.clientY, ox = n.x, oy = n.y;
        let moved = false;
        el.setPointerCapture(e.pointerId);
        const move = ev => {
          moved = true;
          n.x = Math.round(ox + (ev.clientX - sx) / B.view.s); n.y = Math.round(oy + (ev.clientY - sy) / B.view.s);
          el.style.left = n.x + "px"; el.style.top = n.y + "px"; drawLiveEdges();
        };
        const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); saveDraft(); renderBuilder(); };
        el.addEventListener("pointermove", move); el.addEventListener("pointerup", up);
      });
    });
    // the settings of the selected block
    const n = B.graph.nodes.find(x => x.id === B.sel), e = n && entryOf(n);
    $$("[data-field]").forEach(f => f.addEventListener("change", () => {
      const path = f.dataset.field, raw = f.type === "checkbox" ? f.checked : f.value;
      if (path === "shape") {
        const dims = String(raw).split(/[\s,×x]+/).filter(Boolean).map(Number);
        tryChange(g => { g.nodes.find(x => x.id === n.id).shape = dims; }, n.id);
      } else if (path.startsWith("arg.")) {
        const i = +path.slice(4), p = e.params[i];
        tryChange(g => { const t = g.nodes.find(x => x.id === n.id); t.args = [...(t.args || [])]; t.args[i] = parseVal(p, raw); }, n.id);
      } else if (path.startsWith("opt.")) {
        const rest = path.slice(4), at = rest.lastIndexOf("."), key = rest.slice(0, at), i = +rest.slice(at + 1);
        const p = e.options.find(o => o.key === key).params[i];
        tryChange(g => { const t = g.nodes.find(x => x.id === n.id); t.options[key] = [...t.options[key]]; t.options[key][i] = parseVal(p, raw); }, n.id);
      }
    }));
    $$("[data-opt]").forEach(c => c.addEventListener("change", () => {
      const o = e.options.find(x => x.key === c.dataset.opt);
      tryChange(g => {
        const t = g.nodes.find(x => x.id === n.id); t.options = { ...(t.options || {}) };
        if (c.checked) t.options[o.key] = o.params.map(guess); else delete t.options[o.key];
      }, n.id);
    }));
    $("#opt-search")?.addEventListener("input", ev => { B.optSearch = ev.target.value; renderBuilder(); });
    $("#mark-output")?.addEventListener("click", () => tryChange(g => { g.output = g.output === n.id ? undefined : n.id; }, n.id));
    $("#del-block")?.addEventListener("click", () => removeBlock(n.id));
    $("#build")?.addEventListener("click", async () => {
      B.building = true; renderBuilder();
      try {
        B.built = await postJson("/api/builder/build", { name: B.graph.name || "my-model", graph: wireGraph(B.graph) });
        await postJson("/api/builder/graphs", { name: B.graph.name || "my-model", graph: { output: B.graph.output, nodes: B.graph.nodes } });
        B.graphs = (await api("/api/builder/graphs")).graphs || [];
      } catch (err) { B.built = null; B.refusal = { id: null, text: err.message }; }
      B.building = false; renderBuilder();
    });
    requestAnimationFrame(() => drawLiveEdges());
  }
  document.addEventListener("keydown", e => {
    if (state.view !== "builder" || !isLive() || !B.sel || !["Delete", "Backspace", "Escape"].includes(e.key)) return;
    if (e.target.closest("input, textarea, select") || document.querySelector(".cmdk-back, #pop")) return;
    if (e.key === "Escape") { B.sel = null; B.refusal = null; renderBuilder(); return; }
    e.preventDefault();
    removeBlock(B.sel);
  });

  // ---------- TRAIN (hosts, launch, live runs) ----------
  function spark(pts, c, w = 120, h = 28) {
    if (!pts.length) return `<svg width="${w}" height="${h}" aria-hidden="true"></svg>`;
    const max = 2.4, xs = i => (i / 21) * (w - 4) + 2, ys = v => h - 3 - (v / max) * (h - 6);
    const d = pts.map((v, i) => `${i ? "L" : "M"}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(" ");
    const last = pts.length - 1;
    return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true"><path d="${d} L${xs(last)},${h} L2,${h} Z" fill="${c}" opacity=".12"/><path d="${d}" fill="none" stroke="${c}" stroke-width="1.6"/><circle cx="${xs(last)}" cy="${ys(pts[last])}" r="2.6" fill="${c}"/></svg>`;
  }

  function hostCard(h) {
    const st = { busy: ["c-text", "training"], idle: ["c-ok", "idle"], offline: ["c-bad", "offline"] }[h.state];
    return `
    <div class="host ${h.state}">
      <div class="host-head"><span class="dot ${h.state === "busy" ? "pulse" : ""}" style="--c:var(--${h.state === "offline" ? "bad" : h.state === "busy" ? "accent" : "ok"})"></span>
        <b>${esc(h.name)}</b><span class="chip ${st[0]}">${st[1]}</span><span class="grow"></span><span class="mono faint">${esc(h.addr)}</span></div>
      ${h.devices.map(d => `<div class="dev-row"><span>${esc(d.name)}</span><span class="meter wide"><i style="width:${d.util}%"></i></span><span class="mono tnum">${d.mem}</span></div>`).join("")}
      <div class="host-foot"><span class="muted">${esc(h.foot)}</span></div>
    </div>`;
  }

  function trainView() {
    const runs = M.runs;
    const W = 1000, H = 220, pad = 30, maxY = 2.4;
    const xs = n => pad + (n / 21) * (W - pad * 2), ys = v => H - pad - (v / maxY) * (H - pad * 2);
    const lines = runs.filter(r => r.pts.length).map(r => {
      const d = r.pts.map((v, i) => `${i ? "L" : "M"}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(" ");
      const last = r.pts.length - 1;
      return `<path d="${d}" fill="none" stroke="${r.c}" stroke-width="2"/><circle cx="${xs(last)}" cy="${ys(r.pts[last])}" r="3.5" fill="${r.c}"/>`;
    }).join("");
    const grid = [0, .5, 1, 1.5, 2].map(v => `<line x1="${pad}" x2="${W - pad}" y1="${ys(v)}" y2="${ys(v)}" stroke="var(--line-soft)"/><text x="${pad - 8}" y="${ys(v) + 4}" text-anchor="end" fill="var(--muted)" font-size="11" font-family="var(--font-mono)">${v.toFixed(1)}</text>`).join("");
    const xt = [0, 7, 14, 21].map(i => `<text x="${xs(i)}" y="${H - 8}" text-anchor="middle" fill="var(--muted)" font-size="11" font-family="var(--font-mono)">${Math.round(i / 21 * 30)}k</text>`).join("");
    return `
    <div class="train">
      <section class="hosts" aria-label="Hosts">
        ${M.hosts.map(hostCard).join("")}
        <button class="host add" aria-label="Add a host">${icon("plus")}<b>Add a host</b><span class="muted">Run <code>brainiac-studio --worker</code> on another machine and paste its address and token.</span></button>
      </section>
      <div class="train-body">
        <aside class="launch panel" aria-label="Start a training run">
          <div class="col-head" style="padding:0 0 4px"><span class="h3">New run</span><span class="grow"></span><span class="chip c-train">resnet-s</span></div>
          <div class="field"><label for="t-data">Data</label><select class="input" id="t-data"><option>CIFAR-10 · 50,000 train / 10,000 test</option><option>data/my-images (folder)</option></select></div>
          <div class="two">
            <div class="field"><label for="t-opt">Optimizer</label><select class="input" id="t-opt"><option>AdamW</option><option selected>Muon + AdamW</option><option>Lion</option><option>Adam 8-bit</option></select></div>
            <div class="field"><label for="t-lr">Learning rate</label><input class="input mono" id="t-lr" value="3e-4"></div>
          </div>
          <div class="two">
            <div class="field"><label for="t-bs">Batch</label><input class="input mono" id="t-bs" value="256"></div>
            <div class="field"><label for="t-steps">Steps</label><input class="input mono" id="t-steps" value="30,000"></div>
          </div>
          <div class="field"><label>Schedule</label><div class="seg"><button aria-pressed="false">Constant</button><button aria-pressed="true">Warmup + cosine</button><button aria-pressed="false">Step</button></div></div>
          <div class="field"><label>Precision</label><div class="seg"><button aria-pressed="false">float32</button><button aria-pressed="true">bfloat16</button><button aria-pressed="false">float16</button></div></div>
          <div class="field"><label for="t-host">Host</label><select class="input" id="t-host"><option>Auto: least busy that fits (lab workstation)</option><option>This PC · RX 9070 XT</option><option>Lab workstation · 2 × RTX 4090</option></select></div>
          <label class="sweep"><input type="checkbox" checked> Sweep <span class="mono muted">lr ∈ {1e-4, 3e-4, 1e-3} × optimizer ∈ {AdamW, Muon}</span></label>
          <div class="muted" style="font-size:12px">6 runs · about 14 min each · spread over 2 hosts · checkpoint every 1,000 steps</div>
          <button class="btn primary" style="height:38px;justify-content:center">${icon("train")}Start 6 runs</button>
        </aside>
        <section class="runs-col">
          <div class="panel">
            <div class="col-head"><span class="h3">Runs</span><span class="chip c-text">3 running</span><span class="chip plain">1 queued</span><span class="grow"></span><div class="seg"><button aria-pressed="true">All</button><button aria-pressed="false">Running</button><button aria-pressed="false">Finished</button></div></div>
            <div class="tablebox" style="border:0;border-radius:0">
            <table class="models runs-table">
              <thead><tr><th>Run</th><th>Host</th><th>Progress</th><th>Loss</th><th>Accuracy</th><th>Speed</th><th></th></tr></thead>
              <tbody>${runs.map(r => `<tr>
                <td><div style="display:flex;gap:8px;align-items:center"><span class="dot" style="--c:${r.c}"></span><span class="nm">${esc(r.name)}</span></div><div class="src">${r.state}</div></td>
                <td>${esc(r.host)}</td>
                <td><div class="prog"><span class="bar"><i style="width:${r.pct}%;background:${r.c}"></i></span><span class="mono tnum">${r.step}</span></div><div class="src">${r.eta === "—" ? "" : "done in " + r.eta}</div></td>
                <td><div style="display:flex;gap:8px;align-items:center">${spark(r.pts, r.c)}<span class="mono tnum">${r.loss ?? "—"}</span></div></td>
                <td class="mono tnum">${r.acc}</td>
                <td class="mono tnum">${r.speed}</td>
                <td><div class="row-actions">${r.state === "running" ? `<button class="iconbtn" aria-label="Pause">${icon("pause")}</button><button class="iconbtn" aria-label="Stop">${icon("stop")}</button>` : r.state === "queued" ? `<button class="iconbtn" aria-label="Start now">${icon("play")}</button>` : `<button class="iconbtn" aria-label="Run again">${icon("retry")}</button>`}</div></td>
              </tr>`).join("")}</tbody>
            </table></div>
          </div>
          <div class="panel chartbox">
            <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px"><span class="h3">Training loss</span><span class="grow"></span>
              <div class="legend">${runs.filter(r => r.pts.length).map(r => `<span style="--c:${r.c}"><i></i>${esc(r.name)}</span>`).join("")}</div></div>
            <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Loss against step for each run">${grid}${xt}${lines}</svg>
          </div>
        </section>
        <aside class="run-detail panel" aria-label="Selected run">
          <div class="col-head" style="padding:0"><span class="dot" style="--c:var(--cap-speech)"></span><span class="h3">resnet-s · muon</span><span class="grow"></span><span class="chip c-text">running</span></div>
          <dl class="kv"><dt>Host</dt><dd>Lab workstation · GPU 0</dd><dt>Step</dt><dd>18,400 / 30,000</dd><dt>Learning rate</dt><dd>1.9e-4 (cosine)</dd><dt>Gradient norm</dt><dd>0.84</dd><dt>Best checkpoint</dt><dd>step 18,000 · 92.6%</dd></dl>
          <div class="eyebrow">Events</div>
          <div class="feed" style="padding:0">
            ${[["c-ok", "Checkpoint saved · step 18,000", "16:02"], ["c-text", "Validation 92.6% (+0.4)", "16:02"], ["c-warn", "resnet-s · lion stopped early: no gain in 5 evaluations", "15:51"], ["c-text", "Started on Lab workstation · GPU 0", "15:31"]].map(([c, w, t]) => `<div class="ev"><span class="pin ${c}" style="border-color:var(--c)"></span><div><div class="what">${w}</div><div class="t">${t}</div></div></div>`).join("")}
          </div>
          <div style="display:flex;gap:6px;flex-wrap:wrap"><button class="btn">Open in builder</button><button class="btn">Chat with it</button><button class="btn ghost">Clone settings</button></div>
        </aside>
      </div>
    </div>`;
  }

  // ---------- TRAIN (live) ----------
  // Runs train a model from the builder on built-in data or files in the data folder. The
  // optimizers, losses, schedules and metrics are found in the installed modules, like layers.
  const T = {
    options: null, data: null, models: null, runs: [], active: 0, loaded: false, loading: false,
    sel: store.get("trainSel", null), form: store.get("trainForm", null), check: null, checkSeq: 0,
    starting: false, error: null, timer: 0, scale: store.get("trainScale", "linear"), show: "all", hover: null,
    hosts: [], hostsAt: 0, addingHost: false, hostDraft: { name: "", url: "", token: "" }, hostError: null, hostBusy: false, removing: null,
  };
  const SERIES = 8;
  const runColor = r => `var(--series-${((parseInt(r.id, 10) || 1) - 1) % SERIES + 1})`;
  const trainEntries = kind => T.options ? T.options[{ optimizer: "optimizers", loss: "losses", schedule: "schedules", metric: "metrics" }[kind]] : [];
  const trainEntry = (kind, key) => trainEntries(kind).find(e => e.key === key);
  const optLabel = e => !e ? "?" : `${e.group === "Optimizer" || e.group === "Loss" || e.group === "Metric" || e.group === "LearningRateSchedule" ? "" : e.group + "."}${e.name === "new" ? e.group : e.name}`;
  const optSig = e => e && e.params.length ? ` (${e.params.map(p => p.name).join(", ")})` : "";
  const fmtNum = v => v == null ? "–" : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : Math.abs(v) >= 1 ? (+v).toFixed(3) : (+v).toPrecision(3);
  const stateChip = s => ({ queued: "plain", loading: "c-text", running: "c-text", finished: "c-ok", stopped: "c-warn", failed: "c-bad", interrupted: "c-warn" }[s] || "plain");
  const stateName = s => ({ queued: "queued", loading: "loading", running: "training", finished: "finished", stopped: "stopped", failed: "failed", interrupted: "interrupted" }[s] || s);
  const isActive = r => ["queued", "loading", "running"].includes(r.state);

  function trainGuess(p) {
    const n = p.name.toLowerCase();
    if (/learningrate|^lr$|rate$/.test(n)) return 0.001;
    if (/weightdecay|decay/.test(n)) return 0.01;
    if (/momentum|beta1/.test(n)) return 0.9;
    if (/beta2/.test(n)) return 0.999;
    if (/epsilon|eps/.test(n)) return 1e-8;
    if (/totalepochs|epochs/.test(n)) return T.form?.epochs || 20;
    if (/minrate/.test(n)) return 0;
    if (/factor|gamma/.test(n)) return 0.5;
    if (/every/.test(n)) return 10;
    if (/delta|smooth/.test(n)) return 1;
    return guess(p);
  }
  const choiceFor = (kind, key) => { const e = trainEntry(kind, key); return e ? { entry: e.key, args: e.params.map(trainGuess) } : null; };
  function pickKey(kind, ...wanted) {
    const keys = trainEntries(kind).map(e => e.key);
    for (const w of wanted) { const k = keys.find(k => k.endsWith(w)); if (k) return k; }
    return keys[0] || null;
  }
  const dataInfo = () => {
    const d = T.form?.data || {};
    if (d.kind === "builtin") return (T.data?.builtins || []).find(b => b.name === d.name);
    return (T.data?.files || []).find(f => f.name === (d.file || d.images || d.folder));
  };
  const classesData = () => { const i = dataInfo(); return i ? i.classes === true || Array.isArray(i.classes) || i.kind === "idx" || (i.kind === "csv" && T.form.data.classes !== false) : true; };
  function defaultsFor(classes) {
    return {
      loss: choiceFor("loss", pickKey("loss", classes ? "#CROSS_ENTROPY" : "#MSE")),
      metrics: [pickKey("metric", classes ? "#ACCURACY" : "#MEAN_ABSOLUTE_ERROR")].filter(Boolean),
    };
  }
  function ensureForm() {
    if (!T.options || !T.data || !T.models) return;
    const f = T.form || {};
    if (!f.model || !T.models.some(m => m.name === f.model)) f.model = T.models.find(m => !m.problem)?.name || null;
    if (!f.data) f.data = { kind: "builtin", name: "spirals" };
    if (!f.optimizer || !trainEntry("optimizer", f.optimizer.entry)) f.optimizer = choiceFor("optimizer", pickKey("optimizer", "Optimizer#adamW(float,float)", "Optimizer#adam(float)"));
    if (f.schedule && !trainEntry("schedule", f.schedule.entry)) f.schedule = null;
    if (f.host && f.host !== "local" && !T.hosts.some(h => h.id === f.host)) f.host = "local";
    f.host ??= "local";
    f.epochs ??= 20; f.batchSize ??= 32; f.validation ??= 0.1; f.patience ??= 0; f.seed ??= 42; f.device ??= "best"; f.clipNorm ??= 0; f.sweep ??= "";
    T.form = f;
    if (!f.loss || !trainEntry("loss", f.loss.entry)) Object.assign(f, { loss: defaultsFor(classesData()).loss });
    if (!f.metrics) f.metrics = defaultsFor(classesData()).metrics;
  }
  const saveForm = () => store.set("trainForm", T.form);

  function specOf(f) {
    const s = { model: f.model, data: f.data, optimizer: f.optimizer, loss: f.loss, metrics: f.metrics, epochs: +f.epochs, batchSize: +f.batchSize,
      validation: +f.validation, patience: +f.patience, seed: +f.seed, device: f.device };
    if (f.schedule) s.schedule = f.schedule;
    if (+f.clipNorm > 0) s.clipNorm = +f.clipNorm;
    if (f.name) s.name = f.name;
    if (f.host && f.host !== "local") s.host = f.host;
    return s;
  }

  async function loadTrain() {
    if (T.loading) return;
    T.loading = true;
    try {
      const [o, d, m, h] = await Promise.all([api("/api/train/options"), api("/api/train/data"), api("/api/train/models"), api("/api/hosts")]);
      T.options = o; T.data = d; T.models = m.models || []; T.hosts = h.hosts || []; T.hostsAt = Date.now();
      ensureForm();
      T.loaded = true;
      loadVast();
      await pollRuns(true);
      checkTrain();
      if (state.view === "train") render();
    } catch (e) { T.error = String(e.message || e); console.warn("cannot load training", e); if (state.view === "train") render(); }
    finally { T.loading = false; }
  }

  async function pollRuns(once) {
    clearTimeout(T.timer);
    try {
      const [j, h] = await Promise.all([api("/api/train/runs"), Date.now() - T.hostsAt > 4000 && T.hosts.length ? api("/api/hosts") : null]);
      if (h) { T.hosts = h.hosts || []; T.hostsAt = Date.now(); }
      const wasActive = T.active;
      T.runs = j.runs || []; T.active = j.active || 0;
      if (wasActive && !T.active) api("/api/train/models").then(m => { T.models = m.models || []; if (state.view === "train") renderTrainForm(); });
      if (!once && state.view === "train") renderTrainLive();
    } catch (e) { console.warn("cannot read runs", e); }
    if (state.view === "train" && isLive()) T.timer = setTimeout(() => pollRuns(false), T.active ? 1000 : 5000);
  }

  let checkTimer = 0;
  function checkTrain() {
    clearTimeout(checkTimer);
    checkTimer = setTimeout(async () => {
      if (!T.form?.model) { T.check = null; renderTrainCheck(); return; }
      const seq = ++T.checkSeq;
      try {
        const r = await postJson("/api/train/check", { spec: specOf(T.form) });
        if (seq === T.checkSeq) { T.check = r; renderTrainCheck(); }
      } catch (e) { if (seq === T.checkSeq) { T.check = { ok: false, problem: String(e.message || e) }; renderTrainCheck(); } }
    }, 250);
  }

  // ----- the view
  function liveTrainView() {
    if (!T.loaded) return `<div class="train live-train"><div class="empty" style="padding:60px"><span class="typing"><i></i><i></i><i></i></span>
      <div class="muted">${T.error ? esc(T.error) : "Looking for models, data and every optimizer the installed modules offer…"}</div></div></div>`;
    return `
    <div class="train live-train">
      <section class="hosts" id="train-hosts" aria-label="Hosts">${trainHosts()}</section>
      <section id="train-vast" aria-label="GPUs to rent">${vastPanel()}</section>
      <div class="train-body">
        <aside class="launch panel" id="train-form" aria-label="Start a training run">${trainForm()}</aside>
        <section class="runs-col">
          <div class="panel" id="train-runs">${trainRunsTable()}</div>
          <div class="panel chartbox" id="train-chart">${trainChartFrame()}</div>
        </section>
        <aside class="run-detail panel" id="train-detail" aria-label="Selected run">${trainDetail()}</aside>
      </div>
    </div>`;
  }

  function trainHosts() {
    const devs = M.system?.devices || [];
    const run = T.runs.find(r => r.host === "This PC" && (r.state === "running" || r.state === "loading"));
    const queued = T.runs.filter(r => r.host === "This PC" && r.state === "queued").length;
    const here = `<div class="host ${run ? "busy" : "idle"}">
      <div class="host-head"><span class="dot ${run ? "pulse" : ""}" style="--c:var(--${run ? "accent" : "ok"})"></span><b>This PC</b>
        <span class="chip ${run ? "c-text" : "c-ok"}">${run ? "training" : "idle"}</span><span class="grow"></span><span class="mono faint">${queued ? queued + " queued" : "one run at a time"}</span></div>
      ${devs.map(d => `<div class="dev-row"><span>${esc(d.name)}</span><span class="mono faint">${d.kind === "gpu" ? "GPU" : d.name.toLowerCase() === "cpu" ? "" : "CPU"}</span><span class="mono tnum">${run && run.device === d.name && run.samplesPerSecond ? fmtCount(run.samplesPerSecond) + "/s" : ""}</span></div>`).join("")}
      <div class="host-foot"><span class="muted">${run ? `${esc(run.name)} · epoch ${run.epoch + (run.state === "running" ? 1 : 0)} of ${run.epochs}` : "Ready for a run."}</span></div>
    </div>`;
    const remote = T.hosts.map(h => {
      const st = !h.online ? ["c-bad", "offline", "bad"] : h.active ? ["c-text", "training", "accent"] : ["c-ok", "idle", "ok"];
      const tr = h.training;
      return `<div class="host ${h.online ? h.active ? "busy" : "idle" : "offline"}">
        <div class="host-head"><span class="dot ${h.active ? "pulse" : ""}" style="--c:var(--${st[2]})"></span><b>${esc(h.name)}</b><span class="chip ${st[0]}">${st[1]}</span><span class="grow"></span>
          <span class="mono faint" title="Token ${esc(h.token)}">${esc(h.url.replace(/^https?:\/\//, "").replace(/\/$/, ""))}</span>
          <button class="iconbtn sm" data-host-remove="${esc(h.id)}" aria-label="Forget ${esc(h.name)}" title="${T.removing === h.id ? "Click again to forget it" : "Forget this host"}">${T.removing === h.id ? icon("check") : icon("trash")}</button></div>
        ${(h.devices || []).map(d => `<div class="dev-row"><span>${esc(d.name)}</span><span class="mono faint">${d.kind === "gpu" ? "GPU" : d.name.toLowerCase() === "cpu" ? "" : "CPU"}</span><span class="mono tnum">${tr && tr.samplesPerSecond && d.kind === "gpu" ? fmtCount(tr.samplesPerSecond) + "/s" : ""}</span></div>`).join("")}
        <div class="host-foot"><span class="muted">${!h.online ? esc(h.error || "does not answer") : tr ? `${esc(tr.name)} · epoch ${tr.epoch + 1} of ${tr.epochs}` : h.queued ? `${h.queued} queued` : "Ready for a run."}</span></div>
      </div>`;
    }).join("");
    const add = T.addingHost ? `<form class="host add-form" id="host-form" autocomplete="off">
        <div class="host-head"><b>Add a host</b><span class="grow"></span><button type="button" class="iconbtn sm" id="host-cancel" aria-label="Cancel">${icon("x")}</button></div>
        <span class="muted" style="font-size:12px">On the other machine run <code>brainiac-studio --worker</code>. It prints its addresses and a token.</span>
        <input class="input" id="host-url" placeholder="Address, such as 192.168.1.20:7870" value="${esc(T.hostDraft.url)}" spellcheck="false">
        <input class="input mono" id="host-token" placeholder="Token" value="${esc(T.hostDraft.token)}" spellcheck="false" type="password">
        <input class="input" id="host-name" placeholder="Name (optional)" value="${esc(T.hostDraft.name)}">
        ${T.hostError ? `<div class="flag bad">${icon("alert")}<span>${esc(T.hostError)}</span></div>` : ""}
        <button class="btn primary" type="submit" ${T.hostBusy ? "disabled" : ""}>${T.hostBusy ? "Connecting…" : "Connect"}</button>
      </form>`
      : `<button class="host add" id="host-add" aria-label="Add a host">${icon("plus")}<b>Add a host</b><span class="muted">Run <code>brainiac-studio --worker</code> on another machine and paste its address and token.</span></button>`;
    return here + remote + vastCard() + add;
  }

  function entrySelect(kind, cur, label, allowNone) {
    const es = trainEntries(kind);
    const core = es.filter(e => !e.experimental), exp = es.filter(e => e.experimental);
    const opt = e => `<option value="${esc(e.key)}" ${cur && cur.entry === e.key ? "selected" : ""}>${esc(optLabel(e) + optSig(e))}</option>`;
    return `<div class="field"><label for="t-${kind}">${label}</label><select class="input" id="t-${kind}" data-tentry="${kind}">
      ${allowNone ? `<option value="" ${!cur ? "selected" : ""}>${esc(allowNone)}</option>` : ""}
      <optgroup label="Core">${core.map(opt).join("")}</optgroup>${exp.length ? `<optgroup label="Experimental">${exp.map(opt).join("")}</optgroup>` : ""}</select></div>`;
  }
  function entryParams(kind, cur) {
    const e = cur && trainEntry(kind, cur.entry);
    if (!e || !e.params.length) return "";
    return `<div class="two">${e.params.map((p, i) => fieldFor(p, cur.args?.[i], `${kind}.args.${i}`)).join("")}</div>`;
  }

  function dataSelect() {
    const d = T.form.data, files = T.data.files || [];
    const val = d.kind === "builtin" ? "builtin:" + d.name : "file:" + (d.file || d.images || d.folder);
    return `<div class="field"><label for="t-data">Data</label><select class="input" id="t-data">
      <optgroup label="Built in">${T.data.builtins.map(b => `<option value="builtin:${esc(b.name)}" ${val === "builtin:" + b.name ? "selected" : ""}>${esc(b.name)} · ${fmtShape(b.input)} → ${b.targets}</option>`).join("")}</optgroup>
      ${files.length ? `<optgroup label="In the data folder">${files.map(f => `<option value="file:${esc(f.name)}" ${val === "file:" + f.name ? "selected" : ""}>${esc(f.name)} · ${f.kind === "images" ? "image folder" : f.kind.toUpperCase()}</option>`).join("")}</optgroup>` : ""}
    </select><span class="faint" style="font-size:12px">${esc(dataInfo()?.what || "")}${files.length ? "" : ` Put CSV files, MNIST IDX files or image folders (one subfolder per class) in <code>${esc(T.data.folder)}</code> to train on them.`}</span></div>
    ${dataExtras()}`;
  }
  function dataExtras() {
    const d = T.form.data, info = dataInfo();
    if (d.kind === "csv" && info) {
      const cols = info.columns || [];
      return `<div class="two"><div class="field"><label for="t-label">Answer column</label><select class="input" id="t-label" data-tf="data.label">
        ${cols.map(c => `<option ${c === (d.label || cols[cols.length - 1]) ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></div>
        <div class="field"><label>&nbsp;</label><label class="check"><input type="checkbox" data-tf="data.classes" ${d.classes !== false ? "checked" : ""}> classes</label>
        <label class="check"><input type="checkbox" data-tf="data.normalize" ${d.normalize ? "checked" : ""}> standardize</label></div></div>`;
    }
    if (d.kind === "images") {
      return `<div class="two">${[["channels", 3], ["height", 32], ["width", 32]].map(([k, def]) => `<div class="field"><label for="t-img-${k}">${k}</label><input class="input mono" id="t-img-${k}" data-tf="data.${k}" value="${esc(d[k] ?? def)}"></div>`).join("")}</div>`;
    }
    if (d.kind === "idx") {
      return `<div class="field"><label for="t-idx-l">Labels file</label><input class="input mono" id="t-idx-l" data-tf="data.labels" value="${esc(d.labels || "")}"></div>`;
    }
    return "";
  }

  function trainForm() {
    const f = T.form;
    if (!T.models.length) return `<div class="col-head" style="padding:0 0 4px"><span class="h3">New run</span></div>
      <p class="muted" style="margin:0">There is no model to train yet. Draw one in the builder and press <b>Build and save</b>.</p>
      <button class="btn primary" data-go="builder">${icon("build")}Open the builder</button>`;
    const m = T.models.find(x => x.name === f.model);
    const metrics = trainEntries("metric");
    return `
      <div class="col-head" style="padding:0 0 4px"><span class="h3">New run</span><span class="grow"></span>${m ? `<span class="chip plain mono">${fmtShape(m.input)} → ${fmtShape(m.output)}</span>` : ""}</div>
      <div class="field"><label for="t-model">Model</label><select class="input" id="t-model" data-tf="model">
        ${T.models.map(x => `<option value="${esc(x.name)}" ${x.name === f.model ? "selected" : ""} ${x.problem ? "disabled" : ""}>${esc(x.name)}${x.parameters != null ? " · " + fmtCount(x.parameters) + " params" : ""}</option>`).join("")}</select></div>
      ${dataSelect()}
      ${entrySelect("optimizer", f.optimizer, "Optimizer")}${entryParams("optimizer", f.optimizer)}
      ${entrySelect("loss", f.loss, "Loss")}${entryParams("loss", f.loss)}
      ${entrySelect("schedule", f.schedule, "Learning rate schedule", "Constant")}${entryParams("schedule", f.schedule)}
      <div class="two">
        <div class="field"><label for="t-epochs">Epochs</label><input class="input mono" id="t-epochs" data-tf="epochs" value="${esc(f.epochs)}"></div>
        <div class="field"><label for="t-batch">Batch</label><input class="input mono" id="t-batch" data-tf="batchSize" value="${esc(f.batchSize)}"></div>
        <div class="field"><label for="t-val" title="The share of the data held back to check the model on, from 0 to 0.5">Validation share</label><input class="input mono" id="t-val" data-tf="validation" value="${esc(f.validation)}"></div>
        <div class="field"><label for="t-pat" title="Stop when the validation loss has not improved for this many epochs, and keep the best weights. 0 trains every epoch.">Patience <span class="faint">0 = off</span></label><input class="input mono" id="t-pat" data-tf="patience" value="${esc(f.patience)}"></div>
        <div class="field"><label for="t-seed">Seed</label><input class="input mono" id="t-seed" data-tf="seed" value="${esc(f.seed)}"></div>
        <div class="field"><label for="t-clip">Clip gradients <span class="faint">0 = off</span></label><input class="input mono" id="t-clip" data-tf="clipNorm" value="${esc(f.clipNorm)}"></div>
      </div>
      ${metrics.length ? `<div class="field"><label>Metrics</label><div class="metric-picks">${metrics.map(e => `<label class="check"><input type="checkbox" data-metric="${esc(e.key)}" ${f.metrics.includes(e.key) ? "checked" : ""}> ${esc(optLabel(e).toLowerCase().replace(/_/g, " "))}</label>`).join("")}</div></div>` : ""}
      <div class="two">
        <div class="field"><label for="t-host">Train on</label><select class="input" id="t-host" data-tf="host"><option value="local" ${f.host === "local" ? "selected" : ""}>This PC</option>
          ${T.hosts.map(h => `<option value="${esc(h.id)}" ${f.host === h.id ? "selected" : ""} ${h.online ? "" : "disabled"}>${esc(h.name)}${h.online ? "" : " (offline)"}</option>`).join("")}</select></div>
        <div class="field"><label for="t-device">Device</label><select class="input" id="t-device" data-tf="device">${deviceOptions(f).map(d => `<option value="${esc(d.key)}" ${d.key === f.device ? "selected" : ""}>${esc(d.name)}</option>`).join("")}</select></div>
      </div>
      <div class="field"><label for="t-sweep">Also try learning rates <span class="faint">one run each</span></label><input class="input mono" id="t-sweep" data-tf="sweep" value="${esc(f.sweep)}" placeholder="for example 3e-4, 3e-3"></div>
      <div id="train-check">${trainCheckLine()}</div>
      <button class="btn primary" id="train-start" style="height:38px;justify-content:center" ${T.starting || !T.check?.ok ? "disabled" : ""}>${icon("train")}${startLabel()}</button>`;
  }
  function deviceOptions(f) {
    if (!f.host || f.host === "local") return T.options.devices;
    const h = T.hosts.find(x => x.id === f.host), devs = h?.devices || [];
    const gpu = devs.find(d => d.kind === "gpu");
    return [{ key: "best", name: `Best available (${gpu ? gpu.name : "cpu"})` }, ...devs.map(d => ({ key: d.kind, name: d.name }))]
      .filter((d, i, all) => all.findIndex(x => x.key === d.key) === i);
  }
  const sweepRates = () => String(T.form.sweep || "").split(/[\s,;]+/).filter(Boolean).map(Number).filter(v => isFinite(v) && v > 0);
  const startLabel = () => { const n = 1 + sweepRates().length; return T.starting ? "Starting…" : n > 1 ? `Start ${n} runs` : "Start training"; };
  function trainCheckLine() {
    const c = T.check;
    if (!c) return `<div class="faint" style="font-size:12px">Checking the model against the data…</div>`;
    if (!c.ok) return `<div class="flag bad" role="alert">${icon("alert")}<span>${esc(c.problem || "The model and the data do not fit together.")}</span></div>`;
    const where = !T.form.host || T.form.host === "local" ? "This PC" : T.hosts.find(h => h.id === T.form.host)?.name;
    const busy = where === "This PC" ? T.runs.some(r => r.host === "This PC" && (r.state === "running" || r.state === "loading")) : (T.hosts.find(h => h.id === T.form.host)?.active > 0);
    return `<div class="muted" style="font-size:12px">${(c.trainSamples || 0).toLocaleString()} samples to learn from${+T.form.validation > 0 ? `, ${((c.samples || 0) - (c.trainSamples || 0)).toLocaleString()} to check` : ""} · ${(c.stepsPerEpoch || 0).toLocaleString()} steps per epoch${c.classes?.length ? ` · ${c.classes.length} classes` : ""}${busy ? " · queued behind the run training now" : ""}</div>`;
  }
  function renderTrainCheck() {
    const box = $("#train-check"); if (!box) return;
    box.innerHTML = trainCheckLine();
    const b = $("#train-start"); if (b) { b.disabled = T.starting || !T.check?.ok; b.innerHTML = icon("train") + startLabel(); }
  }

  function trainRunsTable() {
    const runs = T.runs.filter(r => T.show === "all" || (T.show === "active" ? isActive(r) : !isActive(r)));
    const counts = { running: T.runs.filter(r => r.state === "running" || r.state === "loading").length, queued: T.runs.filter(r => r.state === "queued").length };
    const seg = (k, l) => `<button data-tshow="${k}" aria-pressed="${T.show === k}">${l}</button>`;
    const metricOf = r => { const h = r.history?.[r.history.length - 1]; if (!h) return null; const vm = h.validationMetrics || {}, tm = h.metrics || {}; const k = Object.keys(vm)[0] || Object.keys(tm)[0]; return k ? { k, v: vm[k] ?? tm[k], val: k in vm } : null; };
    return `<div class="col-head"><span class="h3">Runs</span>${counts.running ? `<span class="chip c-text">${counts.running} training</span>` : ""}${counts.queued ? `<span class="chip plain">${counts.queued} queued</span>` : ""}<span class="grow"></span>
      <div class="seg" role="group" aria-label="Show runs">${seg("all", "All")}${seg("active", "Running")}${seg("ended", "Ended")}</div></div>
      ${runs.length ? `<div class="tablebox" style="border:0;border-radius:0"><table class="models runs-table">
        <thead><tr><th>Run</th><th>Progress</th><th>Loss</th><th>Metric</th><th>Speed</th><th></th></tr></thead>
        <tbody>${runs.map(r => {
          const pct = r.steps ? Math.min(100, r.step / r.steps * 100) : r.state === "finished" ? 100 : 0;
          const last = r.history?.[r.history.length - 1], mt = metricOf(r);
          return `<tr class="${r.id === selectedRun()?.id ? "sel" : ""}" data-run="${esc(r.id)}">
            <td><div style="display:flex;gap:8px;align-items:center"><span class="dot" style="--c:${runColor(r)}"></span><span class="nm">${esc(r.name)}</span></div>
              <div class="src"><span class="chip ${stateChip(r.state)}" style="height:18px;font-size:10.5px">${stateName(r.state)}</span> ${esc(r.data || "")}</div></td>
            <td><div class="prog"><span class="bar"><i style="width:${pct.toFixed(1)}%;background:${runColor(r)}"></i></span><span class="mono tnum">${r.epoch}/${r.epochs}</span></div>
              <div class="src">${r.state === "running" && r.samplesPerSecond && r.steps ? "done in " + etaOf(r) : r.step ? r.step.toLocaleString() + " steps" : ""}</div></td>
            <td class="mono tnum">${fmtNum(last ? (last.validationLoss ?? last.loss) : r.loss)}${last?.validationLoss != null ? ` <span class="faint">val</span>` : ""}</td>
            <td class="mono tnum">${mt ? `${fmtNum(mt.v)} <span class="faint">${esc((mt.val ? "val " : "") + mt.k)}</span>` : "–"}</td>
            <td class="mono tnum">${r.samplesPerSecond ? fmtCount(r.samplesPerSecond) + "/s" : "–"}</td>
            <td><div class="row-actions">${isActive(r) ? `<button class="iconbtn" data-tstop="${esc(r.id)}" aria-label="Stop ${esc(r.name)}" title="Stop and keep what it learned">${icon("stop")}</button>`
              : `<button class="iconbtn" data-tagain="${esc(r.id)}" aria-label="Use the settings of ${esc(r.name)}" title="Use these settings">${icon("retry")}</button><button class="iconbtn" data-tdel="${esc(r.id)}" aria-label="Remove ${esc(r.name)}" title="Remove from the list (the saved model stays)">${icon("trash")}</button>`}</div></td>
          </tr>`; }).join("")}</tbody></table></div>`
        : `<div class="muted" style="padding:14px 16px">${T.runs.length ? "No runs here." : "No runs yet. Pick a model and data on the left and start one."}</div>`}`;
  }
  function etaOf(r) {
    const left = (r.steps - r.step) * r.batchSize / r.samplesPerSecond;
    return left < 60 ? Math.ceil(left) + " s" : left < 3600 ? Math.ceil(left / 60) + " min" : (left / 3600).toFixed(1) + " h";
  }
  // ----- the loss chart: one line per run, hover for values
  const chartRuns = () => T.runs.filter(r => (r.curve || []).length > 1 && (T.show === "all" || (T.show === "active" ? isActive(r) : !isActive(r)))).slice(0, SERIES);
  function trainChartFrame() {
    const runs = chartRuns();
    return `<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px"><span class="h3">Training loss</span><span class="faint" style="font-size:12px">per step</span><span class="grow"></span>
      <div class="legend">${runs.map(r => `<span style="--c:${runColor(r)}"><i></i>${esc(r.name)}</span>`).join("")}</div>
      <div class="seg" role="group" aria-label="Scale"><button data-tscale="linear" aria-pressed="${T.scale === "linear"}">Linear</button><button data-tscale="log" aria-pressed="${T.scale === "log"}">Log</button></div></div>
      <div class="loss-chart" id="loss-chart">${runs.length ? "" : `<div class="muted" style="padding:40px 0;text-align:center">The loss of every step shows here once a run starts.</div>`}</div>`;
  }
  function drawLossChart() {
    const box = $("#loss-chart"); if (!box) return;
    const runs = chartRuns();
    if (!runs.length) return;
    const W = Math.max(320, box.clientWidth), H = 240, L = 46, R = 12, Tp = 10, Bt = 26;
    const log = T.scale === "log";
    const f = v => log ? Math.log10(Math.max(v, 1e-8)) : v;
    let maxX = 1, lo = Infinity, hi = -Infinity;
    for (const r of runs) {
      const c = r.curve; maxX = Math.max(maxX, c[c.length - 1][0]);
      const skip = Math.floor(c.length * .02);
      for (let i = 0; i < c.length; i++) { const v = f(c[i][1]); if (!isFinite(v)) continue; lo = Math.min(lo, v); if (i >= skip) hi = Math.max(hi, v); }
    }
    if (!log) lo = Math.min(0, lo);
    if (!(hi > lo)) hi = lo + 1;
    const X = s => L + s / maxX * (W - L - R), Y = v => Tp + (1 - (Math.min(f(v), hi) - lo) / (hi - lo)) * (H - Tp - Bt);
    const ticks = 4, grid = [];
    for (let i = 0; i <= ticks; i++) {
      const v = lo + (hi - lo) * i / ticks, y = Tp + (1 - i / ticks) * (H - Tp - Bt), lab = log ? fmtNum(Math.pow(10, v)) : fmtNum(v);
      grid.push(`<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="var(--line-soft)"/><text x="${L - 8}" y="${y + 4}" text-anchor="end" fill="var(--muted)" font-size="11" font-family="var(--font-mono)">${lab}</text>`);
    }
    for (let i = 0; i <= 4; i++) { const s = maxX * i / 4; grid.push(`<text x="${X(s)}" y="${H - 6}" text-anchor="${i === 0 ? "start" : i === 4 ? "end" : "middle"}" fill="var(--muted)" font-size="11" font-family="var(--font-mono)">${fmtCount(Math.round(s))}</text>`); }
    const lines = runs.map(r => {
      const c = r.curve, step = Math.max(1, Math.floor(c.length / (W / 2)));
      const d = c.filter((_, i) => i % step === 0 || i === c.length - 1).map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ");
      const lp = c[c.length - 1];
      return `<path d="${d}" fill="none" stroke="${runColor(r)}" stroke-width="2" stroke-linejoin="round"/><circle cx="${X(lp[0]).toFixed(1)}" cy="${Y(lp[1]).toFixed(1)}" r="4" fill="${runColor(r)}" stroke="var(--panel)" stroke-width="2"/>`;
    }).join("");
    box.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Loss against step for ${runs.length} run${runs.length === 1 ? "" : "s"}">${grid.join("")}${lines}<line id="lc-x" y1="${Tp}" y2="${H - Bt}" stroke="var(--muted)" stroke-dasharray="3 3" visibility="hidden"/></svg><div class="chart-tip" id="lc-tip" hidden></div>`;
    const svg = box.querySelector("svg"), tip = $("#lc-tip"), cross = $("#lc-x");
    svg.addEventListener("mousemove", e => {
      const rect = svg.getBoundingClientRect(), px = e.clientX - rect.left;
      if (px < L || px > W - R) { tip.hidden = true; cross.setAttribute("visibility", "hidden"); return; }
      const s = (px - L) / (W - L - R) * maxX;
      const rows = runs.map(r => { let best = null; for (const p of r.curve) { if (!best || Math.abs(p[0] - s) < Math.abs(best[0] - s)) best = p; } return { r, p: best }; })
        .filter(x => x.p && Math.abs(x.p[0] - s) <= maxX * .03);
      if (!rows.length) { tip.hidden = true; cross.setAttribute("visibility", "hidden"); return; }
      cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
      tip.innerHTML = `<div class="faint mono">step ${Math.round(s).toLocaleString()}</div>` + rows.map(x => `<div><span class="dot" style="--c:${runColor(x.r)}"></span>${esc(x.r.name)} <b class="mono">${fmtNum(x.p[1])}</b></div>`).join("");
      tip.hidden = false;
      tip.style.left = Math.min(px + 14, W - tip.offsetWidth - 4) + "px"; tip.style.top = "8px";
    });
    svg.addEventListener("mouseleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
  }

  const selectedRun = () => T.runs.find(r => r.id === T.sel) || T.runs[0] || null;

  // ----- trying a trained model: one sample of its data, or values typed in
  T.tries = {};   // run id -> {busy, result, error, values}
  function tryPanel(r) {
    const t = T.tries[r.id] || {}, res = t.result, id = esc(r.id);
    const m = (T.models || []).find(x => x.name === r.output);
    const size = m?.input ? m.input.reduce((a, b) => a * b, 1) : null;
    let shown = "";
    if (res) {
      const name = i => res.classes ? res.classes[i] : "output " + i;
      const one = res.outputSize === 1;
      const right = res.expectedBest != null ? res.expectedBest === res.best : null;
      const top = res.output.map((v, i) => ({ v, i })).filter(x => x.v != null).sort((a, b) => b.v - a.v).slice(0, 5);
      const span = Math.max(1e-9, ...top.map(x => Math.abs(x.v)));
      shown = `<div class="try-result">
        ${res.index != null ? `<div class="faint mono" style="font-size:11.5px">sample ${(res.index + 1).toLocaleString()} of ${res.samples.toLocaleString()} · ${res.input.slice(0, 6).map(fmtNum).join(", ")}${res.inputSize > 6 ? ", … " + res.inputSize + " values" : ""}</div>` : ""}
        <div class="try-answer">${one ? `<b class="mono">${fmtNum(res.output[0])}</b>` : `<b>${esc(res.label ?? name(res.best))}</b>`}
          ${right === true ? `<span class="chip c-ok">${icon("check")}right</span>` : right === false ? `<span class="chip c-bad">should be ${esc(res.expectedLabel ?? name(res.expectedBest))}</span>`
            : res.expected ? `<span class="chip plain mono">should be ${res.expected.slice(0, 4).map(fmtNum).join(", ")}${res.expected.length > 4 ? ", …" : ""}</span>` : ""}</div>
        ${one ? "" : `<div class="try-bars">${top.map(x => `<div><span class="nm">${esc(name(x.i))}</span><span class="bar"><i style="width:${(Math.abs(x.v) / span * 100).toFixed(1)}%"></i></span><span class="mono tnum">${fmtNum(x.v)}</span></div>`).join("")}</div>`}
      </div>`;
    }
    return `<div class="trybox"><div class="eyebrow">Try it</div>
      <div class="try-row"><button class="btn sm" data-try-sample="${id}" ${t.busy ? "disabled" : ""}>${icon("play")}${res && res.index != null ? "Another sample" : "A sample of its data"}</button></div>
      <form class="try-row" data-try-own="${id}" autocomplete="off"><input class="input mono" data-try-values="${id}" value="${esc(t.values || "")}" placeholder="or your own ${size ? size.toLocaleString() + " " : ""}values, comma separated" spellcheck="false" aria-label="Your own input values">
        <button class="btn sm" type="submit" ${t.busy ? "disabled" : ""}>Run</button></form>
      ${t.error ? `<div class="flag bad" role="alert">${icon("alert")}<span>${esc(t.error)}</span></div>` : ""}${shown}</div>`;
  }
  async function tryModel(r, own) {
    const t = T.tries[r.id] ||= {};
    if (t.busy) return;
    const body = { model: r.output };
    if (own) {
      const vals = String(t.values || "").split(/[\s,;]+/).filter(Boolean).map(Number);
      if (!vals.length || vals.some(v => !isFinite(v))) { t.error = "Type numbers, separated by commas."; renderTrainLive(true); return; }
      body.values = vals;
    } else body.data = r.spec.data;
    t.busy = true; t.error = null; renderTrainLive(true);
    try { t.result = await postJson("/api/train/predict", body); }
    catch (e) { t.error = String(e.message || e); t.result = null; }
    t.busy = false; renderTrainLive(true);
  }
  function trainDetail() {
    const r = selectedRun();
    if (!r) return `<div class="col-head" style="padding:0"><span class="h3">Run</span></div><p class="muted" style="margin:0">Details of the selected run show here: its settings, every epoch and where the trained model was saved.</p>`;
    const s = r.spec || {}, o = trainEntry("optimizer", s.optimizer?.entry), lo = trainEntry("loss", s.loss?.entry), sc = s.schedule && trainEntry("schedule", s.schedule.entry);
    const last = r.history?.[r.history.length - 1];
    const args = (e, c) => e && e.params.length ? ` (${e.params.map((p, i) => `${p.name} ${fmtVal(c.args?.[i])}`).join(", ")})` : "";
    const hist = (r.history || []).slice(-8).reverse();
    const mkeys = [...new Set(hist.flatMap(h => [...Object.keys(h.validationMetrics || {}).map(k => "val " + k), ...Object.keys(h.metrics || {})]))].slice(0, 2);
    const mval = (h, k) => k.startsWith("val ") ? h.validationMetrics?.[k.slice(4)] : h.metrics?.[k];
    return `
      <div class="col-head" style="padding:0"><span class="dot" style="--c:${runColor(r)}"></span><span class="h3">${esc(r.name)}</span><span class="grow"></span><span class="chip ${stateChip(r.state)}">${stateName(r.state)}</span></div>
      ${r.error ? `<div class="flag bad" role="alert">${icon("alert")}<span>${esc(r.error)}</span></div>` : ""}
      ${r.output ? `<div class="flag ok">${icon("check")}<span>Saved as <b>${esc(r.output)}</b> in the models folder${r.state === "stopped" ? ", with what it learned before it was stopped" : ""}${r.stoppedEarly ? `; stopped early, best at epoch ${r.bestEpoch}` : ""}.</span></div>` : ""}
      ${r.output && !isActive(r) ? tryPanel(r) : ""}
      <dl class="kv"><dt>Host</dt><dd>${esc(r.host || "This PC")}${r.device ? " · " + esc(r.device) : ""}</dd><dt>Model</dt><dd>${esc(s.model || "?")}</dd><dt>Data</dt><dd>${esc(r.data || "?")}${r.samples ? ` · ${r.samples.toLocaleString()} samples` : ""}</dd>
        <dt>Optimizer</dt><dd>${esc(optLabel(o) + args(o, s.optimizer || {}))}</dd><dt>Loss</dt><dd>${esc(optLabel(lo) + args(lo, s.loss || {}))}</dd>
        ${sc ? `<dt>Schedule</dt><dd>${esc(optLabel(sc) + args(sc, s.schedule))}</dd>` : ""}
        <dt>Epoch</dt><dd>${r.epoch} of ${r.epochs}${r.steps ? ` · step ${r.step.toLocaleString()} of ${r.steps.toLocaleString()}` : ""}</dd>
        <dt>Batch</dt><dd>${s.batchSize}${r.samplesPerSecond ? ` · ${fmtCount(r.samplesPerSecond)} samples/s` : ""}</dd>
        ${last?.learningRate != null ? `<dt>Learning rate</dt><dd class="mono">${fmtNum(last.learningRate)}</dd>` : ""}</dl>
      ${hist.length ? `<div class="eyebrow">Epochs</div><div class="tablebox"><table class="models epochs-table"><thead><tr><th>#</th><th>Loss</th>${hist.some(h => h.validationLoss != null) ? "<th>Val loss</th>" : ""}${mkeys.map(k => `<th>${esc(k)}</th>`).join("")}</tr></thead>
        <tbody>${hist.map(h => `<tr><td class="mono">${h.epoch}</td><td class="mono tnum">${fmtNum(h.loss)}</td>${hist.some(x => x.validationLoss != null) ? `<td class="mono tnum">${fmtNum(h.validationLoss)}</td>` : ""}${mkeys.map(k => `<td class="mono tnum">${fmtNum(mval(h, k))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>` : ""}
      <div style="display:flex;gap:6px;flex-wrap:wrap">${isActive(r) ? `<button class="btn" data-tstop="${esc(r.id)}">${icon("stop")}Stop and keep</button>` : `<button class="btn" data-tagain="${esc(r.id)}">${icon("retry")}Use these settings</button><button class="btn ghost" data-tdel="${esc(r.id)}">Remove</button>`}</div>`;
  }

  // ----- partial updates while runs train, so typing in the form is never interrupted
  function renderTrainLive(force) {
    if (state.view !== "train" || !T.loaded) return;
    // values being typed for Try it are not redrawn away by the poll
    const trying = !force && !!document.activeElement?.matches?.("#train-detail input");
    const set = (id, html) => { const el = $("#" + id); if (el) el.innerHTML = html; };
    const typing = !!document.activeElement?.matches?.("#host-form input, #vast-key, #vast-hours");
    if (!typing) set("train-hosts", trainHosts());
    set("train-runs", trainRunsTable()); if (!trying) set("train-detail", trainDetail());
    const crumb = $("#crumb .muted"); if (crumb) crumb.textContent = `· ${T.runs.length} run${T.runs.length === 1 ? "" : "s"}${T.active ? `, ${T.active} active` : ""}`;
    const tip = $("#lc-tip");
    if (!tip || tip.hidden) { set("train-chart", trainChartFrame()); drawLossChart(); }
    renderTrainCheck();
    bindTrainLists();
    if (!typing) { bindHosts(); bindVastCard(); }
  }
  function renderTrainForm() {
    const box = $("#train-form"); if (!box) return;
    const focus = document.activeElement?.id, pos = document.activeElement?.selectionStart;
    ensureForm();
    box.innerHTML = trainForm();
    bindTrainForm();
    if (focus) { const el = document.getElementById(focus); if (el) { el.focus(); try { if (pos != null) el.setSelectionRange(pos, pos); } catch (_) { /* not a text field */ } } }
  }

  function setPath(obj, path, v) {
    const ks = path.split("."); let o = obj;
    for (let i = 0; i < ks.length - 1; i++) { const k = ks[i]; o[k] ??= /^\d+$/.test(ks[i + 1]) ? [] : {}; o = o[k]; }
    o[ks[ks.length - 1]] = v;
  }
  function bindTrainForm() {
    const box = $("#train-form"); if (!box) return;
    box.querySelector("[data-go]")?.addEventListener("click", e => go(e.currentTarget.dataset.go));
    $("#t-data")?.addEventListener("change", e => {
      const [kind, name] = [e.target.value.slice(0, e.target.value.indexOf(":")), e.target.value.slice(e.target.value.indexOf(":") + 1)];
      if (kind === "builtin") T.form.data = { kind: "builtin", name };
      else {
        const file = T.data.files.find(x => x.name === name);
        const m = T.models.find(x => x.name === T.form.model), shape = m?.input?.length === 3 ? m.input : [3, 32, 32];
        T.form.data = file.kind === "csv" ? { kind: "csv", file: name } : file.kind === "idx" ? { kind: "idx", images: name, labels: file.labels || "" }
          : { kind: "images", folder: name, channels: shape[0], height: shape[1], width: shape[2] };
      }
      if (!T.form.lossTouched) Object.assign(T.form, defaultsFor(classesData()));
      saveForm(); renderTrainForm(); checkTrain();
    });
    $$("[data-tentry]", box).forEach(sel => sel.addEventListener("change", () => {
      const kind = sel.dataset.tentry;
      T.form[kind] = sel.value ? choiceFor(kind, sel.value) : null;
      if (kind === "loss") T.form.lossTouched = true;
      saveForm(); renderTrainForm();
    }));
    $$("[data-metric]", box).forEach(c => c.addEventListener("change", () => {
      const k = c.dataset.metric;
      T.form.metrics = c.checked ? [...new Set([...T.form.metrics, k])] : T.form.metrics.filter(x => x !== k);
      saveForm();
    }));
    const numeric = new Set(["epochs", "batchSize", "validation", "patience", "seed", "clipNorm", "data.channels", "data.height", "data.width"]);
    $$("[data-tf]", box).forEach(el => {
      const ev = el.tagName === "SELECT" || el.type === "checkbox" ? "change" : "input";
      el.addEventListener(ev, () => {
        const path = el.dataset.tf;
        let v = el.type === "checkbox" ? el.checked : el.value;
        if (numeric.has(path)) v = v === "" ? "" : Number(v);
        setPath(T.form, path, v);
        if (path === "data.classes" && !T.form.lossTouched) Object.assign(T.form, defaultsFor(classesData()));
        if (path === "host") T.form.device = "best";
        saveForm();
        if (path === "sweep") { renderTrainCheck(); return; }
        if (el.tagName === "SELECT" || el.type === "checkbox") renderTrainForm();
        checkTrain();
      });
    });
    $$("[data-field]", box).forEach(el => {
      const ev = el.tagName === "SELECT" || el.type === "checkbox" ? "change" : "input";
      el.addEventListener(ev, () => {
        const [kind, , i] = el.dataset.field.split(".");
        const e = trainEntry(kind, T.form[kind]?.entry); if (!e) return;
        T.form[kind].args[+i] = parseVal(e.params[+i], el.type === "checkbox" ? el.checked : el.value);
        saveForm();
      });
    });
    $("#train-start")?.addEventListener("click", startTraining);
  }

  // ----- Vast.ai: GPU machines to rent. Renting shows the price and asks each time; a rented
  // machine is set up as a worker host and destroyed after the hours chosen.
  const V = { status: null, loading: false, error: null, busy: false, keyError: null, removing: false, open: false,
    filters: { gpu: "", minGpuRam: 0, gpus: 1, maxPrice: 0 }, offers: null, searching: false, offerError: null,
    rentals: [], rentUnavailable: null, maxHours: 48, confirm: null, renting: false, rentError: null, destroying: null, poll: 0, cardError: null };
  const RENT_STAGE = { renting: "renting", starting: "starting up", copying: "copying the studio", building: "building the worker",
    ready: "ready as a host", destroying: "destroying", destroyed: "destroyed", failed: "failed" };
  const rentalActive = r => !r.ended;
  const usd = n => n == null ? "—" : "$" + (n < 1 ? n.toFixed(3) : n.toFixed(2));

  async function loadVast() {
    if (V.loading) return;
    V.loading = true;
    try { V.status = await api("/api/vast"); V.error = null; } catch (e) { V.error = String(e.message || e); }
    finally { V.loading = false; }
    await loadRentals(false);
    if (state.view === "train" && T.loaded) renderHosts();
    else if (state.view === "library") render();
  }

  // the machines rented from the studio; watched while any is being set up or runs
  async function loadRentals(draw = true) {
    try { takeRentals(await api("/api/vast/rentals")); } catch (e) { console.warn(e); }
    if (draw && state.view === "train" && T.loaded && !document.activeElement?.matches?.("#host-form input, #vast-key, #vast-hours")) renderHosts();
    watchRentals();
  }
  function takeRentals(r) {
    const was = V.rentals.filter(x => x.stage === "ready").length;
    V.rentals = r.rentals || []; V.rentUnavailable = r.unavailable || null; V.maxHours = r.maxHours || 48;
    // a machine that just became ready is a new host
    if (V.rentals.filter(x => x.stage === "ready").length !== was) api("/api/hosts").then(h => { T.hosts = h.hosts || []; T.hostsAt = Date.now(); if (state.view === "train" && T.loaded) renderHosts(); }).catch(() => {});
  }
  function watchRentals() {
    clearTimeout(V.poll);
    if (!V.rentals.some(rentalActive)) return;
    const settingUp = V.rentals.some(r => rentalActive(r) && r.stage !== "ready");
    V.poll = setTimeout(() => { if (state.view === "train" || state.view === "library") loadRentals(); else watchRentals(); }, settingUp ? 4000 : 20000);
  }
  const hoursLeft = r => Math.max(0, (r.until - Date.now()) / 3600000);
  const fmtHours = h => h >= 1 ? `${h.toFixed(h >= 10 ? 0 : 1)} h` : `${Math.max(1, Math.round(h * 60))} min`;

  function rentalRow(r) {
    const active = rentalActive(r);
    const tone = r.stage === "ready" ? "ok" : r.stage === "failed" ? "bad" : active ? "warn" : "muted";
    const btn = !active || r.stage === "destroying" ? ""
      : r.instance ? `<button class="btn sm ${V.destroying === r.instance ? "danger" : "ghost"}" data-vast-destroy="${r.instance}">${V.destroying === r.instance ? "Click to destroy" : "Destroy"}</button>`
      : `<button class="btn sm ghost" data-vast-cancel="${esc(r.key)}">Cancel</button>`;
    return `<div class="rental ${active ? "" : "ended"}">
        <div class="rental-head"><span class="dot ${active && r.stage !== "ready" ? "pulse" : ""}" style="--c:var(--${tone})"></span><b title="${esc(r.gpu)}">${esc(r.gpu)}</b><span class="grow"></span>
          <span class="mono tnum" title="Cost so far at ${usd(r.pricePerHour)} an hour">${usd(r.cost)}</span>${btn}</div>
        <div class="faint rental-sub"><span class="chip plain">${esc(RENT_STAGE[r.stage] || r.stage)}</span>${active && r.stage !== "destroying" ? `<span>${usd(r.pricePerHour)}/h · destroyed in ${fmtHours(hoursLeft(r))}${r.instance ? ` · #${r.instance}` : ""}</span>` : ""}</div>
        ${r.leftover ? `<div class="faint" style="font-size:12px">Left running when the studio last stopped without closing${r.instance ? ` (#${r.instance})` : ""}, so it was destroyed when the studio started.</div>` : ""}
        ${r.last ? `<div class="mono faint rental-log" title="${esc(r.last)}">${esc(r.last)}</div>` : ""}
        ${r.error ? `<div class="flag bad">${icon("alert")}<span>${esc(r.error)}</span></div>` : ""}
      </div>`;
  }

  function vastCard() {
    const s = V.status;
    if (!s) return `<div class="host vast"><div class="host-head"><span class="dot" style="--c:var(--muted)"></span><b>Vast.ai</b><span class="grow"></span><span class="typing"><i></i><i></i><i></i></span></div></div>`;
    if (!s.key) return `<form class="host vast" id="vast-key-form" autocomplete="off">
        <div class="host-head"><span class="dot" style="--c:var(--muted)"></span><b>Vast.ai</b><span class="chip plain">rent GPUs</span></div>
        <span class="muted" style="font-size:12px">Rent GPU machines by the hour. Paste an API key from <a href="https://cloud.vast.ai/manage-keys/" target="_blank" rel="noopener">cloud.vast.ai</a>. It is kept in <code>keys.properties</code> and sent only to Vast.ai.</span>
        <div class="vast-row"><input class="input mono" id="vast-key" type="password" placeholder="API key" spellcheck="false" aria-label="Vast.ai API key">
          <button class="btn" type="submit" ${V.busy ? "disabled" : ""}>${V.busy ? "Checking…" : "Save"}</button></div>
        ${V.keyError ? `<div class="flag bad">${icon("alert")}<span>${esc(V.keyError)}</span></div>` : ""}
      </form>`;
    const ins = s.instances || [];
    const mine = new Set(V.rentals.filter(r => r.instance && rentalActive(r)).map(r => r.instance));
    const others = ins.filter(i => !mine.has(i.id));
    return `<div class="host vast">
        <div class="host-head"><span class="dot" style="--c:var(--${s.error ? "bad" : "ok"})"></span><b>Vast.ai</b><span class="chip plain" title="Key ${esc(s.key.source === "environment" ? "from " + s.key.label : "saved in keys.properties")}">${esc(s.key.hint)}</span>
          <span class="grow"></span>${s.balance != null ? `<span class="mono tnum" title="Balance on Vast.ai">${usd(s.balance)}</span>` : ""}
          ${s.key.source === "saved" ? `<button class="iconbtn sm" id="vast-remove" aria-label="Remove the Vast.ai key" title="${V.removing ? "Click again to remove the key" : "Remove the key"}">${V.removing ? icon("check") : icon("trash")}</button>` : ""}</div>
        ${s.error ? `<div class="flag bad">${icon("alert")}<span>${esc(s.error)}</span></div>` : ""}
        ${V.cardError ? `<div class="flag bad">${icon("alert")}<span>${esc(V.cardError)}</span></div>` : ""}
        ${V.rentals.map(rentalRow).join("")}
        ${others.map(i => `<div class="dev-row vast-other"><span>${esc(i.gpu)}${i.gpus > 1 ? " × " + i.gpus : ""}</span><span class="mono faint">${esc(i.status || "")}</span><span class="mono tnum">${usd(i.pricePerHour)}/h</span>
          <button class="iconbtn sm ${V.destroying === i.id ? "armed" : ""}" data-vast-destroy="${i.id}" aria-label="Destroy instance ${i.id}" title="${V.destroying === i.id ? "Click again to destroy it; billing ends" : `Destroy instance ${i.id}`}">${V.destroying === i.id ? icon("check") : icon("trash")}</button></div>`).join("")}
        <div class="host-foot"><span class="muted">${ins.length ? `${ins.length} rented machine${ins.length === 1 ? "" : "s"}` : "No machines rented."}</span><span class="grow"></span>
          <button class="btn sm" id="vast-open" aria-expanded="${V.open}">${icon("search")}${V.open ? "Hide offers" : "Find GPUs"}</button></div>
      </div>`;
  }

  // Vast.ai on the Models screen, under Compute: the account at a glance; renting and offers are on Training.
  function vastCompute() {
    const s = V.status;
    const open = `<button class="btn sm" data-vast-train>${icon("search")}${s?.key ? "Find GPUs on Training" : "Add the key on Training"}</button>`;
    if (!s) return `<div class="prov off"><div style="display:flex;gap:10px;align-items:center"><span class="plogo">V</span><b>Vast.ai</b><span class="grow"></span>${V.error ? `<span class="chip plain">unreachable</span>` : `<span class="typing"><i></i><i></i><i></i></span>`}</div></div>`;
    const ins = s.instances || [];
    return `<div class="prov ${s.key && !s.error ? "on" : "off"}">
        <div style="display:flex;gap:10px;align-items:center"><span class="plogo">V</span><b>Vast.ai</b><span class="grow"></span>
          ${s.key ? `<span class="chip plain">${esc(s.key.hint)}</span>${s.balance != null ? `<span class="mono tnum" title="Balance on Vast.ai">${usd(s.balance)}</span>` : ""}` : `<span class="chip plain">no key</span>`}</div>
        <div class="muted" style="font-size:12.5px">GPU machines rented by the hour. A rented machine runs the studio as a worker and shows up here as a host.</div>
        ${s.error ? `<div class="keyerr">${esc(s.error)}</div>` : ""}
        ${ins.map(i => `<div class="dev-row"><span>${esc(i.gpu)}${i.gpus > 1 ? " × " + i.gpus : ""}</span><span class="mono faint">${esc(i.status || "")}</span><span class="mono tnum">${usd(i.pricePerHour)}/h</span></div>`).join("")}
        <div class="provset"><span class="mono faint" style="font-size:11px">${s.key ? (ins.length ? `${ins.length} rented` : "nothing rented") : "rent GPUs by the hour"}</span><span class="grow"></span>${open}</div>
      </div>`;
  }

  function vastPanel() {
    if (!V.open || !V.status?.key) return "";
    const f = V.filters, o = V.offers;
    const rows = !o ? "" : !o.length ? `<div class="faint pad">No offers match. Loosen the filters.</div>`
      : `<div class="tablebox vast-table"><table class="models"><thead><tr><th>GPU</th><th>Memory</th><th>$/hour</th><th>Reliability</th><th>Where</th><th>CUDA</th><th>CPU · RAM</th><th>Download</th><th aria-label="Rent"></th></tr></thead><tbody>
        ${o.map(x => `<tr class="${V.confirm?.id === x.id ? "sel" : ""}"><td><b>${esc(x.gpu)}</b>${x.gpus > 1 ? ` <span class="faint">× ${x.gpus}</span>` : ""}</td><td class="mono tnum">${x.gpuRamGb ?? "?"} GB</td><td class="mono tnum">${usd(x.pricePerHour)}</td>
          <td class="mono tnum">${x.reliability != null ? (x.reliability * 100).toFixed(1) + "%" : "—"}</td><td>${esc(x.location || "—")}</td><td class="mono">${x.cuda ?? "—"}</td>
          <td class="mono tnum">${x.cpuCores != null ? Math.round(x.cpuCores) : "?"} · ${x.ramGb ?? "?"} GB</td><td class="mono tnum">${x.downMbps != null ? Math.round(x.downMbps) + " Mb/s" : "—"}</td>
          <td><button class="btn sm" data-vast-rent="${x.id}" ${V.rentUnavailable ? `disabled title="${esc(V.rentUnavailable)}"` : ""}>Rent…</button></td></tr>
          ${V.confirm?.id === x.id ? `<tr class="rent-confirm"><td colspan="9">${rentConfirm(x)}</td></tr>` : ""}`).join("")}</tbody></table></div>`;
    return `<div class="panel vast-panel">
      <div class="col-head"><span class="h3">GPUs on Vast.ai</span><span class="muted" style="font-size:12px">verified machines you can rent now, cheapest first</span><span class="grow"></span>
        <button class="iconbtn sm" id="vast-close" aria-label="Close">${icon("x")}</button></div>
      <form class="vast-filters" id="vast-filters" autocomplete="off">
        <label>GPU<input class="input" id="vf-gpu" placeholder="any, or 4090, A100…" value="${esc(f.gpu)}"></label>
        <label>Memory per GPU<input class="input mono" id="vf-mem" type="number" min="0" step="1" value="${f.minGpuRam || ""}" placeholder="any GB"></label>
        <label>GPUs<input class="input mono" id="vf-gpus" type="number" min="1" max="16" value="${f.gpus || 1}"></label>
        <label>Most per hour<input class="input mono" id="vf-price" type="number" min="0" step="0.05" value="${f.maxPrice || ""}" placeholder="any $"></label>
        <button class="btn primary" type="submit" ${V.searching ? "disabled" : ""}>${icon("search")}${V.searching ? "Looking…" : "Search"}</button>
      </form>
      ${V.offerError ? `<div class="flag bad" style="margin:0 16px 12px">${icon("alert")}<span>${esc(V.offerError)}</span></div>` : ""}
      ${rows}
      ${V.rentUnavailable ? `<div class="flag warn" style="margin:12px 16px 0">${icon("alert")}<span>${esc(V.rentUnavailable)}</span></div>` : ""}
      <p class="muted vast-note">Renting asks you each time, with the price. The machine gets this checkout of the studio, builds it, and joins as a host through an SSH tunnel, which takes 10 to 20 minutes and is billed too. It logs in with your SSH key from <code>~/.ssh</code>, which must be added at <a href="https://cloud.vast.ai/manage-keys/" target="_blank" rel="noopener">cloud.vast.ai</a>. It is destroyed after the hours you choose, when you destroy it, or when the studio closes.</p>
    </div>`;
  }

  function rentConfirm(x) {
    const h = V.confirm.hours, total = x.pricePerHour * h;
    return `<form class="rent-box" id="vast-rent-form" autocomplete="off">
        <div><b>Rent ${x.gpus > 1 ? x.gpus + " × " : ""}${esc(x.gpu)} for ${usd(x.pricePerHour)} an hour?</b></div>
        <div class="muted">Billing starts as soon as Vast.ai rents it to you and runs until it is destroyed. Storage is billed on top, usually a few cents an hour.</div>
        <div class="rent-row"><label>Destroy it after<input class="input mono" id="vast-hours" type="number" min="0.25" max="${V.maxHours}" step="0.25" value="${h}"> hours</label>
          <span class="muted">at most <b class="mono tnum">${usd(total)}</b> for the machine</span><span class="grow"></span>
          <button class="btn ghost" type="button" id="vast-rent-cancel">Cancel</button>
          <button class="btn primary" type="submit" ${V.renting ? "disabled" : ""}>${V.renting ? "Renting…" : `Rent for ${usd(x.pricePerHour)}/h`}</button></div>
        ${V.rentError ? `<div class="flag bad">${icon("alert")}<span>${esc(V.rentError)}</span></div>` : ""}
      </form>`;
  }

  async function rentOffer() {
    const x = (V.offers || []).find(o => o.id === V.confirm?.id); if (!x) return;
    const hours = +($("#vast-hours")?.value || V.confirm.hours);
    if (!(hours > 0) || hours > V.maxHours) { V.rentError = `Choose up to ${V.maxHours} hours.`; renderVast(); return; }
    V.confirm.hours = hours; V.renting = true; V.rentError = null; renderVast();
    try {
      takeRentals(await postJson("/api/vast/rent", { offer: x.id, pricePerHour: x.pricePerHour, hours }));
      V.confirm = null; V.offers = V.offers.filter(o => o.id !== x.id);
      loadVast();
    } catch (e) { V.rentError = String(e.message || e); }
    V.renting = false; renderVast(); renderHosts(); watchRentals();
  }

  function renderVast() {
    const box = $("#train-vast"); if (!box) return;
    box.innerHTML = vastPanel(); bindVastPanel();
  }

  async function searchVast() {
    V.searching = true; V.offerError = null; renderVast();
    try { V.offers = (await postJson("/api/vast/offers", V.filters)).offers; }
    catch (e) { V.offerError = String(e.message || e); }
    V.searching = false; renderVast();
  }

  function bindVastCard() {
    $("#vast-key-form")?.addEventListener("submit", async e => {
      e.preventDefault();
      const key = $("#vast-key").value.trim();
      if (!key) { V.keyError = "Paste the key first."; renderHosts(); return; }
      V.busy = true; V.keyError = null; renderHosts();
      try { V.status = await postJson("/api/vast/key", { key }); if (V.status.error) V.keyError = V.status.error; }
      catch (err) { V.keyError = String(err.message || err); }
      V.busy = false; renderHosts();
    });
    $("#vast-remove")?.addEventListener("click", async () => {
      if (!V.removing) { V.removing = true; renderHosts(); setTimeout(() => { if (V.removing) { V.removing = false; renderHosts(); } }, 4000); return; }
      V.removing = false; V.open = false; V.offers = null;
      try { V.status = await postJson("/api/vast/key", { remove: true }); } catch (e) { console.warn(e); }
      renderHosts(); renderVast();
    });
    $("#vast-open")?.addEventListener("click", () => { V.open = !V.open; renderHosts(); renderVast(); if (V.open && !V.offers) searchVast(); });
    $$("[data-vast-destroy]").forEach(b => b.addEventListener("click", async () => {
      const id = +b.dataset.vastDestroy;
      // two clicks: the first arms it
      if (V.destroying !== id) { V.destroying = id; renderHosts(); setTimeout(() => { if (V.destroying === id) { V.destroying = null; renderHosts(); } }, 4000); return; }
      V.destroying = null; b.disabled = true;
      try { takeRentals(await postJson("/api/vast/destroy", { id })); V.cardError = null; }
      catch (e) { V.cardError = String(e.message || e); }
      loadVast();
    }));
    $$("[data-vast-cancel]").forEach(b => b.addEventListener("click", async () => {
      b.disabled = true;
      try { takeRentals(await postJson("/api/vast/cancel", { key: b.dataset.vastCancel })); V.cardError = null; } catch (e) { V.cardError = String(e.message || e); }
      renderHosts(); watchRentals();
    }));
  }

  function bindVastPanel() {
    $("#vast-close")?.addEventListener("click", () => { V.open = false; V.confirm = null; renderHosts(); renderVast(); });
    $$("[data-vast-rent]").forEach(b => b.addEventListener("click", () => {
      const id = +b.dataset.vastRent;
      V.confirm = V.confirm?.id === id ? null : { id, hours: V.confirm?.hours || 2 }; V.rentError = null; renderVast();
      $("#vast-hours")?.focus();
    }));
    $("#vast-rent-cancel")?.addEventListener("click", () => { V.confirm = null; V.rentError = null; renderVast(); });
    $("#vast-rent-form")?.addEventListener("submit", e => { e.preventDefault(); rentOffer(); });
    $("#vast-hours")?.addEventListener("input", e => {
      const x = (V.offers || []).find(o => o.id === V.confirm?.id), h = +e.target.value;
      const out = $("#vast-rent-form .rent-row .muted b");
      if (x && out && h > 0) { V.confirm.hours = h; out.textContent = usd(x.pricePerHour * h); }
    });
    $("#vast-filters")?.addEventListener("submit", e => {
      e.preventDefault();
      V.filters = { gpu: $("#vf-gpu").value.trim(), minGpuRam: +$("#vf-mem").value || 0, gpus: +$("#vf-gpus").value || 1, maxPrice: +$("#vf-price").value || 0 };
      searchVast();
    });
  }

  function renderHosts(focus) {
    const box = $("#train-hosts"); if (!box) return;
    box.innerHTML = trainHosts(); bindHosts(); bindVastCard();
    if (focus) $(focus)?.focus();
  }
  function bindHosts() {
    $("#host-add")?.addEventListener("click", () => { T.addingHost = true; T.hostError = null; renderHosts("#host-url"); });
    $("#host-cancel")?.addEventListener("click", () => { T.addingHost = false; T.hostError = null; renderHosts(); });
    ["url", "token", "name"].forEach(k => $("#host-" + k)?.addEventListener("input", e => { T.hostDraft[k] = e.target.value; }));
    $("#host-form")?.addEventListener("submit", async e => {
      e.preventDefault();
      T.hostBusy = true; T.hostError = null; renderHosts();
      try {
        await postJson("/api/hosts", T.hostDraft);
        T.addingHost = false; T.hostDraft = { name: "", url: "", token: "" };
        T.hostsAt = 0;
        const h = await api("/api/hosts"); T.hosts = h.hosts || []; T.hostsAt = Date.now();
        renderTrainForm();
      } catch (err) { T.hostError = String(err.message || err).replace(/^\/api\/hosts: /, ""); }
      finally { T.hostBusy = false; renderHosts(T.addingHost ? "#host-token" : null); }
    });
    $$("[data-host-remove]").forEach(b => b.addEventListener("click", async () => {
      const id = b.dataset.hostRemove;
      if (T.removing !== id) { T.removing = id; renderHosts(); setTimeout(() => { if (T.removing === id) { T.removing = null; renderHosts(); } }, 4000); return; }
      T.removing = null;
      await postJson("/api/hosts", { id, remove: true }).catch(() => {});
      T.hosts = T.hosts.filter(h => h.id !== id);
      if (T.form.host === id) { T.form.host = "local"; saveForm(); }
      renderHosts(); renderTrainForm();
    }));
  }
  function bindTrainLists() {
    $$("[data-tshow]").forEach(b => b.addEventListener("click", () => { T.show = b.dataset.tshow; renderTrainLive(); }));
    $$("[data-tscale]").forEach(b => b.addEventListener("click", () => { T.scale = b.dataset.tscale; store.set("trainScale", T.scale); renderTrainLive(); }));
    $$("#train-runs tr[data-run]").forEach(tr => tr.addEventListener("click", e => { if (e.target.closest("button")) return; T.sel = tr.dataset.run; store.set("trainSel", T.sel); renderTrainLive(); }));
    $$("[data-tstop]").forEach(b => b.addEventListener("click", async () => { b.disabled = true; await postJson("/api/train/stop", { id: b.dataset.tstop }).catch(() => {}); pollRuns(false); }));
    $$("[data-tdel]").forEach(b => b.addEventListener("click", async () => { b.disabled = true; await postJson("/api/train/delete", { id: b.dataset.tdel }).catch(() => {}); if (T.sel === b.dataset.tdel) T.sel = null; pollRuns(false); }));
    $$("[data-try-values]").forEach(el => el.addEventListener("input", () => { (T.tries[el.dataset.tryValues] ||= {}).values = el.value; }));
    $$("[data-try-sample]").forEach(b => b.addEventListener("click", () => { const r = T.runs.find(x => x.id === b.dataset.trySample); if (r) tryModel(r, false); }));
    $$("[data-try-own]").forEach(f => f.addEventListener("submit", e => { e.preventDefault(); const r = T.runs.find(x => x.id === f.dataset.tryOwn); if (r) tryModel(r, true); }));
    $$("[data-tagain]").forEach(b => b.addEventListener("click", () => {
      const r = T.runs.find(x => x.id === b.dataset.tagain); if (!r) return;
      const s = JSON.parse(JSON.stringify(r.spec));
      T.form = { ...T.form, ...s, name: "", sweep: "", schedule: s.schedule || null, lossTouched: true };
      saveForm(); renderTrainForm(); checkTrain();
      $("#train-form")?.scrollIntoView({ block: "start", behavior: "smooth" });
    }));
  }
  async function startTraining() {
    if (T.starting || !T.check?.ok) return;
    T.starting = true; renderTrainCheck();
    const base = specOf(T.form), e = trainEntry("optimizer", base.optimizer.entry);
    const lrAt = e ? e.params.findIndex(p => /learningrate|^lr$/i.test(p.name)) : -1;
    const specs = [base, ...(lrAt >= 0 ? sweepRates() : []).map(lr => {
      const s = JSON.parse(JSON.stringify(base)); s.optimizer.args[lrAt] = lr;
      s.name = `${base.model} · ${optLabel(e)} · lr ${lr}`; return s;
    })];
    if (specs.length > 1 && !base.name) base.name = `${base.model} · ${optLabel(e)} · lr ${base.optimizer.args[lrAt]}`;
    try {
      let first = null;
      for (const s of specs) { const r = await postJson("/api/train/start", { spec: s }); first ??= r.id; }
      T.sel = first; store.set("trainSel", T.sel);
      T.check = { ...T.check };   // keep the line, now with the queue in mind
    } catch (err) {
      T.check = { ok: false, problem: String(err.message || err) };
    } finally {
      T.starting = false; renderTrainCheck(); pollRuns(false);
    }
  }
  function bindLiveTrain() {
    bindTrainForm(); bindTrainLists(); bindHosts(); bindVastCard(); bindVastPanel(); drawLossChart();
    if (!T.timer) pollRuns(false);
  }

  // ---------- LIBRARY ----------
  // Before loading, say where a model fits: 16 GB of VRAM with 6.1 GB free in this example.
  function fitCell(m) {
    const gb = parseFloat(m.size) * (m.size.includes("MB") ? .001 : 1);
    if (!gb) return m.state === "remote" ? `<span class="chip plain">provider’s servers</span>` : `<span class="faint">—</span>`;
    if (m.state === "loaded" || m.state === "draft") return `<span class="chip c-ok">${m.device}</span>`;
    const need = gb * 1.12;
    return need <= 6.1 ? `<span class="chip c-ok">GPU · ${need.toFixed(1)} GB</span>`
      : `<span class="chip c-warn" title="Unload Qwen2.5 7B to fit on the GPU">GPU after unload · ${need.toFixed(1)} GB</span>`;
  }
  function libraryView() {
    return `
    <div class="library"><div class="lib-inner">
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><span class="h2">Models</span>${isLive() && P.note ? `<span class="chip c-ok" role="status">${esc(P.note)}</span>` : ""}<span class="grow"></span>
        <button class="btn" id="hub-open" aria-expanded="${!!(isLive() && H.open)}">${icon("download")}Hugging Face</button><button class="btn" id="add-place" aria-expanded="${!!(isLive() && P.open)}">${icon("plus")}Add folder or GGUF</button><button class="btn primary" id="new-model">${icon("build")}New model</button></div>
      ${isLive() && H.open ? hubPanel() : isLive() && allDownloads().length ? `<div id="hub-box">${hubDownloads()}</div>` : ""}
      ${isLive() && P.open ? placePanel() : ""}
      ${isLive() ? liveModelsTable() : `<div class="tablebox"><table class="models">
        <thead><tr><th>Model</th><th>Can</th><th>Format</th><th>Size</th><th>Context</th><th>Fits on</th><th>State</th></tr></thead>
        <tbody>${M.models.map(m => `<tr>
          <td><div class="nm">${esc(m.name)}</div><div class="src">${esc(m.src)}</div></td>
          <td>${m.cap.map(c => `<span class="chip c-${c}">${capName[c]}</span>`).join(" ")}</td>
          <td class="mono">${m.fmt}</td><td class="mono tnum">${m.size}</td><td class="mono">${m.ctx}</td><td>${fitCell(m)}</td>
          <td>${m.state === "loaded" ? `<span class="chip c-ok">loaded · ${m.tps}</span>` : m.state === "remote" ? `<span class="chip c-agent">remote · your key</span>` : m.state === "planned" ? `<span class="chip c-warn">planned</span>` : m.state === "draft" ? `<span class="chip c-text">draft model</span>` : `<button class="btn" style="height:26px;font-size:12px">Load</button>`}</td>
        </tr>`).join("")}</tbody></table></div>`}
      ${isLive() ? "" : `<div style="display:flex;gap:10px;align-items:baseline;flex-wrap:wrap"><span class="h2">Providers</span><span class="muted">Remote models sit next to local ones in routing; their model lists come from each provider.</span></div>`}
      ${isLive() ? liveProviders() : `<div class="providers">${M.providers.map(p => `
        <div class="prov ${p.state}">
          <div style="display:flex;gap:10px;align-items:center"><span class="plogo">${esc(p.name[0])}</span><b>${esc(p.name)}</b><span class="grow"></span>
            ${p.key === "local" ? `<span class="chip c-ok">always on</span>` : `<label class="switch"><input type="checkbox" ${p.state === "on" ? "checked" : ""} aria-label="Use ${esc(p.name)}"><i></i></label>`}</div>
          <div style="display:flex;gap:5px;flex-wrap:wrap">${p.caps.map(c => `<span class="chip c-${c}">${capName[c] || c}</span>`).join("")}</div>
          <div class="muted" style="font-size:12.5px">${esc(p.models)}</div>
          <div style="display:flex;gap:8px;align-items:center"><span class="mono faint" style="font-size:11px">${esc(p.note)}</span><span class="grow"></span>
            ${p.key === "local" ? "" : `<button class="btn" style="height:28px;font-size:12px">${p.state === "on" ? "Key and models" : "Add API key"}</button>`}</div>
        </div>`).join("")}
        <div class="prov routing">
          <b>Routing</b>
          <div class="seg" role="group" aria-label="Routing preference"><button aria-pressed="true">This PC first</button><button aria-pressed="false">Best quality</button><button aria-pressed="false">Cheapest</button></div>
          <label style="display:flex;gap:8px;align-items:center;font-size:13px"><input type="checkbox" checked> Ask before sending anything to a remote provider</label>
          <span class="muted" style="font-size:12px">Keys stay on this PC, readable only by your user account.</span>
        </div>
      </div>`}
      <div style="display:flex;gap:10px;align-items:baseline;flex-wrap:wrap"><span class="h2">Capabilities</span>${M.system ? `<span class="chip c-ok">live</span>` : ""}<span class="muted">Found at startup from the core and experimental modules. New ones appear here, in routing and over MCP without UI changes.</span></div>
      <div class="caps">${M.capabilities.map(c => `
        <div class="cap ${c.missing ? "missing" : ""}" style="--c:var(--cap-${c.cap})">
          <div style="display:flex;gap:8px;align-items:center"><span class="k">${c.key}</span><span class="grow"></span><span class="mono faint">${c.missing ? "planned" : ""}</span></div>
          <div class="v">${esc(c.what)}</div>
          <div class="mono faint" style="font-size:11px">${c.missing ? "no provider installed" : c.note ? esc(c.note) : esc(c.source)}</div>
        </div>`).join("")}</div>
    </div></div>`;
  }

  // ---------- MCP ----------
  function mcpView() {
    return `
    <div class="mcp"><div class="mcp-grid">
      <div style="display:grid;gap:14px;align-content:start;min-width:0">
        <div style="display:flex;gap:10px;align-items:center"><span class="h2">MCP server</span><span class="chip c-ok">on</span></div>
        <div class="endpoint"><span class="dot pulse" style="--c:var(--ok)"></span><code>http://127.0.0.1:7860/mcp</code><span class="chip plain">${icon("lock")}this machine only · token per launch</span><span class="grow"></span><button class="btn">${icon("copy")}Copy config</button></div>
        <div class="seg" role="tablist" aria-label="MCP view"><button aria-pressed="true">Tools</button><button aria-pressed="false">Try a tool</button><button aria-pressed="false">Long calls</button><button aria-pressed="false">Raw protocol</button></div>
        <div class="panel"><div class="col-head"><span class="h3">Tools agents can call</span><span class="grow"></span><span class="mono faint">generated from capabilities</span></div>
          <div class="toollist">${M.mcpTools.map(t => `<div class="tl"><code>${t.name}</code><span class="mono faint">${esc(t.args)}</span><span class="ds">${esc(t.desc)}</span></div>`).join("")}</div></div>
      </div>
      <div style="display:grid;gap:14px;align-content:start;min-width:0">
        <div class="panel"><div class="col-head"><span class="h3">Connected</span></div>
          <div style="display:grid;gap:8px;padding:0 14px 14px">
            ${[["Claude Code", "stdio bridge · 2 calls/min", "var(--cap-agent)"], ["Cursor", "HTTP · idle 3 min", "var(--cap-text)"]].map(([n, d, c]) => `<div style="display:flex;gap:10px;align-items:center"><span class="dot" style="--c:${c}"></span><b style="font-size:13px">${n}</b><span class="muted" style="font-size:12px">${d}</span></div>`).join("")}
          </div></div>
        <div class="panel"><div class="col-head"><span class="h3">Recent calls</span></div>
          <div class="log">${M.mcpLog.map(([t, c, m]) => `<div class="ln"><span class="faint">${t}</span><span style="color:var(--cap-agent)">${c}</span><span>${esc(m)}</span></div>`).join("")}</div></div>
        <div class="panel" style="padding:14px;display:grid;gap:10px"><span class="h3">Permissions</span>
          ${[["Read models and capabilities", true], ["Chat and generate", true], ["Build and train models", true], ["Delete models or runs", false]].map(([l, on]) => `<label style="display:flex;gap:10px;align-items:center;font-size:13px"><input type="checkbox" ${on ? "checked" : ""}>${l}</label>`).join("")}
        </div>
      </div>
    </div></div>`;
  }

  // ---------- AGENTS (every agent in every thread, and what waits on you) ----------
  function agentsView() {
    const all = [
      ...M.agents.map(a => ({ ...a, thread: "Voiceover for the builder demo" })),
      { id: "ppo", name: "PPO tuner", cap: "train", model: "Qwen2.5 7B + brainiac-rl", status: "waiting", doing: "Wants to run 8 racing environments on the GPU for about 20 min", tokens: "3.4k", time: "4 min", progress: .5, thread: "Racing cars PPO tuning" },
      { id: "eval", name: "Evaluator", cap: "train", model: "brainiac-rl", status: "running", doing: "Scoring checkpoint 140 over 10 seeds · mean return 412", tokens: "—", time: "2 min", progress: .64, thread: "Racing cars PPO tuning" },
      { id: "sweep", name: "Sweep runner", cap: "train", model: "brainiac-train", status: "running", doing: "resnet-s · muon at step 18.4k of 30k", tokens: "—", time: "41 min", progress: .61, thread: "CIFAR-10 ResNet sweep" },
      { id: "lora", name: "LoRA explainer", cap: "text", model: "Qwen2.5 7B Instruct", status: "done", doing: "Found the cause: learning rate 2e-3 with alpha 32 overshoots", tokens: "5.8k", time: "38 s", progress: 1, thread: "Why does LoRA rank 8 diverge?" },
    ];
    const lane = (st, title) => {
      const xs = all.filter(a => a.status === st);
      return `<div class="lane-col"><div class="col-head" style="padding:0 2px 8px"><span class="h3">${title}</span><span class="mono faint">${xs.length}</span></div>
        <div style="display:grid;gap:8px">${xs.map(a => `<div style="display:grid;gap:4px"><span class="mono faint" style="font-size:11px;padding-left:2px">${esc(a.thread)}</span>${agentCard(a)}</div>`).join("")}</div></div>`;
    };
    return `
    <div class="mission">
      <aside class="inbox" aria-label="Waiting for you">
        <div class="col-head"><span class="h3">Waiting for you</span><span class="chip c-warn">2</span></div>
        ${[["Thumbnail", "image", "Render 4 candidates at 50 steps on RX 9070 XT", "≈ 9 s · 1.1 GB VRAM", "Voiceover for the builder demo"],
           ["PPO tuner", "train", "Run 8 racing environments for 20 min", "GPU busy until 16:05 · checkpoints every 5 min", "Racing cars PPO tuning"]].map(([n, c, w, d, t]) => `
          <div class="ask" style="--c:var(--cap-${c})">
            <div style="display:flex;gap:8px;align-items:center"><span class="dot" style="--c:var(--cap-${c})"></span><b>${n}</b><span class="grow"></span><span class="mono faint" style="font-size:11px">${esc(t)}</span></div>
            <div>${esc(w)}</div><div class="mono muted" style="font-size:11.5px">${esc(d)}</div>
            <div class="approve"><button class="btn primary">Allow</button><button class="btn">Edit plan</button><button class="btn ghost">Deny</button></div>
          </div>`).join("")}
        <div class="col-head" style="padding-top:18px"><span class="h3">Limits</span></div>
        <dl class="kv" style="padding:0 16px"><dt>Agents at once</dt><dd>4 of 6</dd><dt>Nesting depth</dt><dd>2</dd><dt>GPU jobs at once</dt><dd>1</dd><dt>Default mode</dt><dd>Ask me</dd></dl>
      </aside>
      <section class="lanes">${lane("running", "Running")}${lane("waiting", "Needs you")}${lane("done", "Finished today")}</section>
    </div>`;
  }

  const dataFolder = document.querySelector('meta[name="studio-folder"]')?.content || "";
  // ---------- layout popover ----------
  function layoutPopover() {
    const opt = (k, v, l) => `<button data-${k}="${v}" aria-pressed="${state[k] === v}">${l}</button>`;
    return `
    <div class="pop" id="pop" role="dialog" aria-label="Layout">
      <div class="field"><label>Layout <output>${effectiveLayout()}</output></label>
        <div class="seg">${opt("layout", "auto", "Auto")}${opt("layout", "focus", "Focus")}${opt("layout", "wide", "Wide")}${opt("layout", "ultra", "Ultrawide")}</div></div>
      <div class="lay-pics">
        <i data-l="focus"><b style="flex:1"></b></i>
        <i data-l="wide"><b style="flex:.5"></b><b style="flex:1.4"></b><b style="flex:.8"></b></i>
        <i data-l="ultra"><b style="flex:.5"></b><b style="flex:1.3"></b><b style="flex:.8"></b><b style="flex:1.3"></b></i>
      </div>
      <div class="field"><label>Density</label><div class="seg">${opt("density", "cozy", "Comfortable")}${opt("density", "compact", "Compact")}</div></div>
      <div class="field"><label for="ui-scale">Interface size <output>${Math.round(state.scale * 100)}%</output></label>
        <input type="range" id="ui-scale" min="0.8" max="1.6" step="0.05" value="${state.scale}"></div>
      <p class="muted" style="margin:0;font-size:12px">Auto picks Ultrawide from 2200 px: threads, conversation, agent and media side by side.</p>
      ${dataFolder ? `<div class="field"><label>Kept in</label><code class="datafolder" title="Keys, conversations, these settings and every picture and take. Start the studio with --data to use another folder.">${esc(dataFolder)}</code>
        <span class="faint" style="font-size:12px">Start with <code>--data &lt;folder&gt;</code> to use another one.</span></div>` : ""}
    </div>`;
  }

  // ---------- render ----------
  const views = { chat: chatView, agents: () => isLive() ? liveAgentsView() : agentsView(), create: createView, voice: voiceView, builder: () => isLive() ? liveBuilderView() : builderView(), train: () => isLive() ? liveTrainView() : trainView(), library: libraryView, mcp: mcpView };
  const crumbs = {
    chat: `Chat <span class="muted">· Voiceover for the builder demo</span>`, agents: `Agents <span class="muted">· 4 running · 2 waiting for you</span>`, create: `Images <span class="muted">· thumb-dit-64</span>`,
    voice: `Voice <span class="muted">· Builder demo voiceover</span>`, builder: `Model builder <span class="muted">· resnet-s</span>`,
    train: `Training <span class="muted">· 3 hosts · 2 runs live</span>`, library: `Models <span class="muted">· 6 on this machine</span>`, mcp: `MCP <span class="muted">· 2 agents connected</span>`,
  };
  const routes = {
    chat: ["c-text", "Qwen2.5 7B Instruct"], agents: ["c-agent", "per agent"], create: ["c-image", "thumb-dit-64"], voice: ["c-speech", "OpenAI speech · remote"],
    builder: ["c-train", "resnet-s (unsaved)"], train: ["c-train", "2 hosts busy"], library: ["c-text", "Qwen2.5 7B Instruct"], mcp: ["c-agent", "per call"],
  };

  let builderLoading = false;
  function render() {
    applyFrame();
    $("#main").innerHTML = `<div class="view">${views[state.view]()}</div>`;
    const liveChat = isLive() && state.view === "chat" && L.sel !== "example";
    const liveLib = isLive() && state.view === "library";
    const liveMedia = isLive() && (state.view === "create" || state.view === "voice");
    const liveBuilder = isLive() && state.view === "builder";
    const liveTrain = isLive() && state.view === "train";
    const liveAgents = isLive() && state.view === "agents";
    const ready = (L.info?.providers || []).filter(p => p.ready).length;
    $("#crumb").innerHTML = liveChat ? `Chat <span class="muted">· ${esc(current()?.title || "New conversation")}</span>`
      : liveMedia ? (state.view === "create" ? `Images <span class="muted">· ${G.batches.length} batch${G.batches.length === 1 ? "" : "es"} this session</span>` : `Voice <span class="muted">· ${G.takes.length} take${G.takes.length === 1 ? "" : "s"} this session</span>`)
      : liveLib ? `Models <span class="muted">· ${L.info ? `${L.info.models.length} from ${ready} provider${ready === 1 ? "" : "s"}` : "looking…"}</span>`
      : liveBuilder ? `Model builder <span class="muted">· ${esc(B.graph.name || "Untitled")}</span>`
      : liveTrain ? `Training <span class="muted">· ${T.loaded ? `${T.runs.length} run${T.runs.length === 1 ? "" : "s"}${T.active ? `, ${T.active} active` : ""}` : "looking…"}</span>`
      : liveAgents ? agentCrumb() : crumbs[state.view];
    const picked = liveChat ? paneModel(L.active) : "auto";
    const autoModel = (liveChat || liveLib) && L.info ? modelByKey(picked === "auto" ? L.info.auto.text : picked) : null;
    const mediaModel = liveMedia ? (state.view === "create" ? pickedModel(G.imgModel, "image") : pickedModel(G.voModel, "speech")) : null;
    const [c, n] = liveMedia ? [state.view === "create" ? "c-image" : "c-speech", mediaModel ? mediaModel.name : L.info ? "no model yet" : "…"]
      : liveChat || liveLib ? ["c-text", autoModel ? autoModel.name : L.info ? "no chat model yet" : "…"]
      : liveBuilder ? ["c-train", B.catalog ? `${B.catalog.entries.length} layers found` : "…"]
      : liveTrain ? ["c-train", T.options ? `${T.options.optimizers.length} optimizers found` : "…"]
      : liveAgents ? ["c-agent", A.model === "auto" ? (L.info?.auto?.text ? modelByKey(L.info.auto.text)?.name || "auto" : "no chat model yet") : modelByKey(A.model)?.name || A.model] : routes[state.view];
    $("#router .chip").className = "chip " + c; $("#router .chip").textContent = n;
    $$(".rail a").forEach(a => { if (a.dataset.view === state.view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
    paintAll();
    renderDataNote();
    if (state.view === "builder") {
      if (!isLive()) { drawEdges(); fitWorld(); enableDrag(); }
      else if (B.catalog) bindLiveBuilder();
      else if (!builderLoading) { builderLoading = true; loadBuilder().finally(() => { builderLoading = false; }); }
    }
    if (state.view === "train" && isLive()) { if (T.loaded) bindLiveTrain(); else loadTrain(); }
    if (state.view === "chat") bindChat();
    if (state.view === "library" && isLive()) bindLiveLibrary();
    if (state.view === "create" && isLive()) bindLiveCreate();
    if (state.view === "voice" && isLive()) bindLiveVoice();
    if (liveAgents) { if (A.loaded) bindLiveAgents(); if (!A.loading && !A.timer) loadAgents(); }
    if (state.view === "agents" && !isLive()) document.querySelectorAll(".agent[data-agent]").forEach(el => el.addEventListener("click", () => { if (["script", "voice", "thumb"].includes(el.dataset.agent)) { state.agent = el.dataset.agent; state.inspectorOpen = true; go("chat"); } }));
  }

  function paintAll() {
    $$("canvas[data-thumb]").forEach(cv => paintThumb(cv, +cv.dataset.thumb, +(cv.dataset.noise || 0)));
    const f = $("#frame"); if (f) paintFrame(f, state.t);
    $$("canvas.frame").forEach(c => paintFrame(c, +c.dataset.t));
  }

  function bindChat() {
    if (isLive()) {
      bindLiveChat();
      if (L.sel !== "example") return;
    }
    $$("[data-agent], [data-agent-open]").forEach(el => el.addEventListener("click", () => {
      state.agent = el.dataset.agent || el.dataset.agentOpen; state.inspectorOpen = true; render();
    }));
    $$(".agent[data-agent]").forEach(el => el.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.click(); } }));
    const ti = $("#toggle-insp"); if (ti) ti.addEventListener("click", () => { state.inspectorOpen = !state.inspectorOpen; render(); });
    $$("[data-tab]").forEach(b => b.addEventListener("click", () => { state.tab = b.dataset.tab; render(); }));
    $$("[data-mode]").forEach(b => b.addEventListener("click", () => { state.mode = b.dataset.mode; $$("[data-mode]").forEach(x => x.setAttribute("aria-pressed", x === b)); routeHint(); }));
    $$("[data-autonomy]").forEach(b => b.addEventListener("click", () => { state.autonomy = b.dataset.autonomy; $$("[data-autonomy]").forEach(x => x.setAttribute("aria-pressed", x === b)); }));
    const p = $("#prompt"); if (p) p.addEventListener("input", routeHint);
    routeHint();
    const play = $("#play"), scrub = $("#scrub");
    if (play) play.addEventListener("click", () => { state.playing = !state.playing; play.innerHTML = icon(state.playing ? "pause" : "play"); if (state.playing) tick(); });
    if (scrub) scrub.addEventListener("input", () => { state.t = +scrub.value; updatePlayer(); });
  }

  // The composer names the model before you send: the route follows what you ask for.
  // A message that asks for several media goes to the orchestrator, which hands each part
  // to an agent that only has that medium's tools.
  function routeHint() {
    const el = $("#route-hint"); if (!el) return;
    const text = ($("#prompt")?.value || "").toLowerCase();
    const target = { text: ["Qwen2.5 7B Instruct", "text · 32k context · tools on"], image: ["thumb-dit-64", "image · 64 × 64 · 50 steps"], speech: ["OpenAI speech", "speech · remote, uses your key"] };
    let caps = [state.mode];
    if (state.mode === "auto") {
      caps = [];
      if (/(draw|image|picture|photo|thumbnail|render)/.test(text)) caps.push("image");
      if (/(say|speak|voice|narrat|read aloud|voiceover)/.test(text)) caps.push("speech");
      if (!caps.length) caps.push("text");
    }
    if (caps.length > 1) {
      el.innerHTML = `${icon("wand")}<span>Auto splits this:</span>${caps.map(c => `<span class="chip c-${c}">${c} → ${target[c][0]}</span>`).join("")}<span class="faint">via the orchestrator</span>`;
    } else {
      const c = caps[0];
      el.innerHTML = `${icon("wand")}<span>${state.mode === "auto" ? "Auto will send this to" : "Sending to"}</span><span class="chip c-${c}">${target[c][0]}</span><span class="faint">${target[c][1]}</span>`;
    }
  }

  function updatePlayer() {
    const f = $("#frame"); if (!f) return;
    paintFrame(f, state.t);
    $("#caption").textContent = lineAt(state.t).text;
    $("#clock").textContent = `${fmt(state.t)} / ${fmt(38.4)}`;
    $("#scrub").value = state.t;
    $$(".media .playhead").forEach(p => p.style.left = (state.t / 38.4 * 100) + "%");
  }
  let last = 0;
  function tick(ts) {
    if (!state.playing || state.view !== "chat") return;
    if (ts && last) { state.t = (state.t + (ts - last) / 1000) % 38.4; updatePlayer(); }
    last = ts || 0; requestAnimationFrame(tick);
  }

  // ---------- command palette ----------
  // Every screen, the conversations, the chat models, the saved models in the builder and the
  // runs, so anything can be reached from the keyboard.
  function commandList() {
    const layout = (v, l) => ({ label: `Layout: ${l}`, icon: "layout", run: () => { state.layout = v; store.set("layout", v); render(); } });
    const out = [
      { label: "Go to Chat", icon: "chat", run: () => go("chat") }, { label: "Go to Images", icon: "image", run: () => go("create") },
      { label: "Go to Voice", icon: "speech", run: () => go("voice") }, { label: "Go to Model builder", icon: "build", run: () => go("builder") },
      { label: "Go to Training", icon: "train", run: () => go("train") }, { label: "Go to Models", icon: "library", run: () => go("library") },
      { label: "Go to Agents", icon: "agent", run: () => go("agents") },
    ];
    if (!isLive()) return [...out, { label: "Go to MCP server", icon: "plug", run: () => go("mcp") },
      layout("auto", "Auto"), layout("focus", "Focus"), layout("wide", "Wide"), layout("ultra", "Ultrawide")];
    out.push(
      { label: "New conversation", icon: "plus", run: () => { if (state.view !== "chat") go("chat"); newChat(false); } },
      { label: "New model in the builder", icon: "build", run: newModel },
      { label: "Start an agent", icon: "agent", run: () => { go("agents"); setTimeout(() => $("#ag-goal")?.focus(), 300); } },
      { label: "Add models from a folder or GGUF file…", icon: "folder", run: () => { P.open = true; P.error = null; go("library"); $("#place-path")?.focus(); if (!P.list) browsePlace(""); } },
      { label: "Download a model from Hugging Face…", icon: "download", run: () => { H.open = true; go("library"); $("#hub-q")?.focus(); pollHub(); } },
      { label: "Add an API key…", icon: "lock", run: () => { go("library"); $("[data-add-key]")?.scrollIntoView({ block: "center" }); } },
      { label: "Start a training run…", icon: "train", run: () => go("train") },
      layout("auto", "Auto"), layout("focus", "Focus"), layout("wide", "Wide"), layout("ultra", "Ultrawide"),
      { label: "Switch light or dark", icon: "sun", run: () => $("#theme-btn").click() },
    );
    [...L.convos].sort((x, y) => (y.updated || 0) - (x.updated || 0)).slice(0, 60).forEach(c =>
      out.push({ label: c.title || "New conversation", hint: c.status === "finished" ? "finished conversation" : "conversation", icon: "chat",
        run: () => { if (state.view !== "chat") go("chat"); openConvo(c.id); } }));
    chatModels().forEach(m => out.push({ label: `Chat with ${m.name}`, hint: m.provider === "local" ? "This PC" : providerName(m.provider), icon: "text",
      run: () => { if (state.view !== "chat") go("chat"); newChat(false); L.panes[L.active].model = m.key; L.model = m.key; store.set("chatModel", m.key); savePanes(); render(); } }));
    B.graphs.forEach(g => out.push({ label: `Open ${g.name} in the builder`, hint: g.built ? "built" : "drawing", icon: "build", run: () => openGraph(g.name) }));
    T.runs.slice(0, 30).forEach(r => out.push({ label: `Run ${r.id}: ${r.name}`, hint: r.state, icon: "train", run: () => { T.sel = r.id; store.set("trainSel", r.id); go("train"); } }));
    return out;
  }
  function openCmdk() {
    if (document.querySelector(".cmdk-back")) { $("#cmdk-in")?.focus(); return; }
    const back = document.createElement("div");
    back.className = "cmdk-back";
    back.innerHTML = `<div class="cmdk" role="dialog" aria-label="Commands"><input id="cmdk-in" placeholder="Type a command, a model, a conversation or a run" aria-controls="cmdk-list" aria-autocomplete="list"><ul id="cmdk-list" role="listbox"></ul></div>`;
    document.body.appendChild(back);
    const all = commandList();
    let sel = 0, items = all;
    const list = () => {
      const words = $("#cmdk-in").value.toLowerCase().split(/\s+/).filter(Boolean);
      items = all.filter(c => words.every(w => (c.label + " " + (c.hint || "")).toLowerCase().includes(w))).slice(0, 80);
      sel = Math.min(sel, Math.max(0, items.length - 1));
      $("#cmdk-list").innerHTML = items.length ? items.map((c, i) => `<li role="option" data-i="${i}" class="${i === sel ? "on" : ""}" aria-selected="${i === sel}">${icon(c.icon)}<span class="grow">${esc(c.label)}</span>${c.hint ? `<span class="faint">${esc(c.hint)}</span>` : ""}</li>`).join("")
        : `<li class="none faint">Nothing matches.</li>`;
      $("#cmdk-list li.on")?.scrollIntoView({ block: "nearest" });
    };
    const run = c => { back.remove(); if (c) c.run(); };
    list();
    $("#cmdk-in").focus();
    $("#cmdk-in").addEventListener("input", () => { sel = 0; list(); });
    back.addEventListener("keydown", e => {
      if (e.key === "Escape") back.remove();
      if (e.key === "ArrowDown") { sel = Math.min(items.length - 1, sel + 1); list(); e.preventDefault(); }
      if (e.key === "ArrowUp") { sel = Math.max(0, sel - 1); list(); e.preventDefault(); }
      if (e.key === "Enter") { e.preventDefault(); run(items[sel]); }
    });
    $("#cmdk-list").addEventListener("mousemove", e => { const li = e.target.closest("li[data-i]"); if (li && +li.dataset.i !== sel) { sel = +li.dataset.i; $$("#cmdk-list li").forEach(x => { x.classList.toggle("on", x === li); x.setAttribute("aria-selected", String(x === li)); }); } });
    $("#cmdk-list").addEventListener("click", e => { const li = e.target.closest("li[data-i]"); if (li) run(items[+li.dataset.i]); });
    back.addEventListener("click", e => { if (e.target === back) back.remove(); });
  }

  function go(v) { if (v === "library" && state.view !== "library") L.usage = null; state.view = v; if (location.hash !== "#" + v) history.replaceState(null, "", "#" + v); render(); }

  // ---------- wiring ----------
  window.addEventListener("hashchange", () => { const v = location.hash.slice(1); if (views[v]) { if (v === "library" && state.view !== "library") L.usage = null; state.view = v; render(); } });
  window.addEventListener("resize", () => { const before = $("#app").dataset.layout; applyFrame(); if ($("#app").dataset.layout !== before) render(); else if (state.view === "builder") { if (isLive()) drawLiveEdges(); else fitWorld(); } else if (state.view === "train" && isLive()) drawLossChart(); else if (state.view === "chat" && isLive() && L.sel !== "example" && L.panes.length > maxPanes()) render(); });
  document.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openCmdk(); } });
  document.addEventListener("keydown", e => { if (e.key === "Escape" && L.editing && !document.querySelector(".cmdk-back")) { L.editing = null; L.draft = null; render(); } });
  $("#cmdk-open").addEventListener("click", openCmdk);
  $("#theme-btn").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : !matchMedia("(prefers-color-scheme: light)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    paintAll();
  });
  $("#layout-btn").addEventListener("click", e => {
    e.stopPropagation();
    const old = $("#pop"); if (old) { old.remove(); return; }
    document.body.insertAdjacentHTML("beforeend", layoutPopover());
    const pop = $("#pop");
    pop.addEventListener("click", ev => {
      const b = ev.target.closest("button"); if (!b) return;
      for (const k of ["layout", "density"]) if (b.dataset[k]) { state[k] = b.dataset[k]; store.set(k, state[k]); }
      pop.remove(); render(); $("#layout-btn").click();
    });
    $("#ui-scale").addEventListener("change", ev => { state.scale = +ev.target.value; store.set("scale", state.scale); pop.remove(); render(); });
  });
  document.addEventListener("click", e => { const p = $("#pop"); if (p && !p.contains(e.target)) p.remove(); });

  // ---------- live chat ----------
  // Served by the studio, Chat holds real conversations, kept in the studio's data folder. The
  // invented example conversation and the MCP mock-up show only when the page is opened as a file.
  const L = {
    info: null,                           // GET /api/models: providers, models, what auto picks
    convos: store.get("convos", []),
    sel: null,                            // the design preview ("example") shows only when the page is opened as a plain file
    model: store.get("chatModel", "auto"),  // what a new conversation starts with
    tabs: store.get("chatTabs", []),      // open conversations, in tab order
    panes: store.get("chatPanes", [{ id: null }]),
    active: store.get("chatActive", 0),   // the pane that has the focus
    running: {},                          // conversation id: the AbortController of the answer being written
    drafts: {},                           // conversation id ("#" and the pane for a new one): what is typed but not sent
    scroll: {},                           // conversation id: where its transcript was scrolled to
    showFinished: false,
    convosLoaded: false,                  // true once the studio's conversations are in
    usage: null,                          // GET /api/usage: calls and tokens by provider, key, model, day
    balances: null,                       // GET /api/balances: what is left per key, where providers say
    editing: null,                        // the provider open in the editor, and its unsaved draft
    draft: null,
    adding: null,                         // the provider whose key form is open
  };
  const isLive = () => !!document.querySelector('meta[name="studio-token"]');
  const current = () => paneConvo(L.active);
  const chatModels = sel => (L.info?.models || []).filter(m => m.media.includes("text") && (m.shown !== false || m.key === sel));
  const modelByKey = k => (L.info?.models || []).find(m => m.key === k);
  const providerName = id => (L.info?.providers || []).find(p => p.id === id)?.name || id;
  const providerKind = id => id === "local" ? "local" : (L.info?.providers || []).find(p => p.id === id)?.kind || "api";
  const mediaChip = { text: "c-text", vision: "c-vision", image: "c-image", speech: "c-speech", transcribe: "c-speech", video: "c-video" };
  const mediaName = { text: "Text", vision: "Vision", image: "Image", speech: "Speech", transcribe: "Transcribe", video: "Video" };

  // A little Markdown: code fences, inline code, bold, paragraphs. Everything is escaped first.
  function md(text) {
    // code spans are set aside first, so nothing inside them is read as bold, a link or an address
    const link = (href, label) => `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
    const plain = s => s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, href) => link(href, label))
      .replace(/(^|[\s(])(https?:\/\/[^\s<]*[^\s<.,;:!?)\]])/g, (_, pre, href) => pre + link(href, href))
      .replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>")
      .replace(/(^|[\s(])\*([^*\s][^*\n]*?)\*(?=[\s.,;:!?)]|$)/g, "$1<i>$2</i>").replace(/\n/g, "<br>");
    const inline = s => esc(s).split(/(`[^`\n]+`)/).map((x, i) => i % 2 ? `<code>${x.slice(1, -1)}</code>` : plain(x)).join("");
    const cells = line => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(x => x.trim());
    const isRow = line => /^\s*\|.*\|\s*$/.test(line);
    const isRule = line => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line);
    return text.split("```").map((part, i) => {
      if (i % 2) {
        const nl = part.indexOf("\n");
        return `<pre class="code">${esc((nl >= 0 ? part.slice(nl + 1) : part).replace(/\n$/, ""))}</pre>`;
      }
      // blocks of lines: headings, rules, bullet and numbered lists (which a blank line does not end), paragraphs
      const out = [];
      let para = [], list = null;
      const flushPara = () => { if (para.length) out.push(`<p>${inline(para.join("\n"))}</p>`); para = []; };
      const flushList = () => {
        if (list) out.push(`<${list.tag}${list.start > 1 ? ` start="${list.start}"` : ""}>${list.items.map(x => `<li>${inline(x)}</li>`).join("")}</${list.tag}>`);
        list = null;
      };
      const lines = part.split("\n");
      for (let n = 0; n < lines.length; n++) {
        const raw = lines[n], line = raw.trimEnd();
        if (isRow(line) && n + 1 < lines.length && isRule(lines[n + 1]) && lines[n + 1].includes("-")) {
          flushPara(); flushList();
          const head = cells(line), rows = [];
          for (n += 2; n < lines.length && isRow(lines[n]); n++) rows.push(cells(lines[n]));
          n--;
          out.push(`<div class="mdtable"><table><thead><tr>${head.map(c => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${rows.map(r =>
            `<tr>${head.map((_, k) => `<td>${inline(r[k] || "")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
          continue;
        }
        const h = /^(#{1,4})\s+(.*)$/.exec(line), ul = /^\s*[-*•]\s+(.*)$/.exec(line), ol = /^\s*(\d+)[.)]\s+(.*)$/.exec(line);
        if (!line.trim()) { flushPara(); continue; }
        if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flushPara(); flushList(); out.push("<hr>"); }
        else if (h) { flushPara(); flushList(); out.push(`<p class="mdh mdh${h[1].length}">${inline(h[2])}</p>`); }
        else if (ul || ol) {
          flushPara();
          const tag = ul ? "ul" : "ol";
          if (!list || list.tag !== tag) { flushList(); list = { tag, start: ol ? +ol[1] : 1, items: [] }; }
          list.items.push(ul ? ul[1] : ol[2]);
        } else if (list && /^\s{2,}/.test(raw)) list.items[list.items.length - 1] += "\n" + line.trim();
        else { flushList(); para.push(line); }
      }
      flushPara(); flushList();
      return out.join("");
    }).join("");
  }

  function when(t) {
    const d = new Date(t), now = new Date();
    return d.toDateString() === now.toDateString() ? d.toTimeString().slice(0, 5) : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }

  // ---------- live chat: tabs, side-by-side panes, finished conversations ----------
  // Open conversations sit in tabs above one or more panes; each pane shows one conversation
  // (or a new one) and answers on its own, so several can stream at once. A finished
  // conversation leaves the tabs and waits under Finished until it is reopened.
  const convoById = id => L.convos.find(c => c.id === id) || null;
  const paneConvo = i => convoById((L.panes[i] || {}).id);
  const running = id => !!L.running[id];
  const draftKey = i => (paneConvo(i) || {}).id || "#" + i;   // a new conversation's draft belongs to its pane
  const anyRunning = () => Object.keys(L.running).length > 0;
  // a model that went away (key removed, provider edited) falls back to Auto
  const usable = k => !k || (L.info && k !== "auto" && !modelByKey(k)) ? "auto" : k;
  const paneModel = i => { const c = paneConvo(i); return usable((c ? c.model : (L.panes[i] || {}).model) || L.model); };
  const maxPanes = () => {
    if (effectiveLayout() === "focus") return 1;
    const w = window.innerWidth / state.scale - 64 - 268;
    return Math.max(1, Math.min(4, Math.floor(w / 480)));
  };
  // What a conversation is told before it starts and how long an answer may get. A new
  // conversation keeps them on its pane until the first message makes it a conversation.
  const paneSettings = i => paneConvo(i) || L.panes[i] || {};
  const hasSettings = o => !!((o.system || "").trim() || +o.maxTokens > 0 || o.temperature != null || o.topP != null || o.topK != null || o.repeatPenalty != null);
  function savePanes() {
    store.set("chatPanes", L.panes.map(p => ({ id: p.id || null, model: p.model || null, system: p.id ? undefined : p.system || undefined, maxTokens: p.id ? undefined : p.maxTokens || undefined, temperature: p.id ? undefined : p.temperature ?? undefined, topP: p.id ? undefined : p.topP ?? undefined, topK: p.id ? undefined : p.topK ?? undefined, repeatPenalty: p.id ? undefined : p.repeatPenalty ?? undefined })));
    store.set("chatTabs", L.tabs);
    store.set("chatActive", L.active);
  }
  function fitPanes() {
    // until the studio's conversations arrive, tabs and panes naming them are kept as they are
    if (L.convosLoaded) {
      L.tabs = L.tabs.filter(id => { const c = convoById(id); return c && c.status !== "finished"; });
      L.panes = L.panes.filter(p => !p.id || convoById(p.id));
    }
    L.panes = L.panes.slice(0, maxPanes());
    if (!L.panes.length) L.panes = [{ id: L.tabs[0] || null }];
    L.active = Math.min(Math.max(0, L.active), L.panes.length - 1);
  }
  // Shows a conversation: focuses the pane that has it, or puts it in the active pane.
  function openConvo(id, beside) {
    const c = convoById(id);
    if (!c) return;
    L.sel = null;
    if (c.status !== "finished" && !L.tabs.includes(id)) L.tabs.push(id);
    const at = L.panes.findIndex(p => p.id === id);
    if (at >= 0) L.active = at;
    else if (beside && L.panes.length < maxPanes()) { L.panes.push({ id }); L.active = L.panes.length - 1; }
    else L.panes[L.active] = { id };
    savePanes();
    render();
  }
  function newChat(beside) {
    L.sel = null; store.set("convo", null);
    const empty = L.panes.findIndex(p => !p.id);
    if (beside && L.panes.length < maxPanes()) { L.panes.push({ id: null }); L.active = L.panes.length - 1; }
    else if (empty >= 0) L.active = empty;
    else L.panes[L.active] = { id: null };
    savePanes();
    render();
    $(`.pane[data-pane="${L.active}"] textarea`)?.focus();
  }
  function closeTab(id) {
    L.tabs = L.tabs.filter(t => t !== id);
    const at = L.panes.findIndex(p => p.id === id);
    if (at >= 0) {
      if (L.panes.length > 1) { L.panes.splice(at, 1); if (L.active >= at) L.active = Math.max(0, L.active - 1); }
      else L.panes[0] = { id: L.tabs.find(t => !L.panes.some(p => p.id === t)) || null };
    }
    savePanes();
    render();
  }
  function closePane(i) {
    if (L.panes.length < 2) return;
    L.panes.splice(i, 1);
    if (L.active >= i) L.active = Math.max(0, L.active - 1);
    savePanes();
    render();
  }
  function setFinished(id, done) {
    const c = convoById(id);
    if (!c) return;
    c.status = done ? "finished" : "open";
    c.finishedAt = done ? Date.now() : undefined;
    if (done) L.tabs = L.tabs.filter(t => t !== id);
    else if (!L.tabs.includes(id)) L.tabs.push(id);
    persist(c);
    savePanes();
    render();
  }

  // Conversations live in the studio (one file each, next to the keys); this browser keeps a copy.
  // (without the pictures, which are too large for a browser's storage)
  const saveConvos = () => store.set("convos", L.convos.slice(0, 200).map(c => c.messages.some(m => m.images) ? { ...c, messages: c.messages.map(m => { const { images, ...rest } = m; return rest; }) } : c));

  // ----- pictures attached to a message, for models that can look at them
  L.attach = {};   // draft key -> [{type, data}]
  const MAX_PICTURES = 8;
  const picUrl = p => `data:${p.type};base64,${p.data}`;
  // a picture is made small enough to send: 1568 px on its long side, as JPEG (PNG if it has transparency to keep)
  function shrinkPicture(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file), img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        const scale = Math.min(1, 1568 / Math.max(img.naturalWidth, img.naturalHeight));
        const cv = document.createElement("canvas");
        cv.width = Math.max(1, Math.round(img.naturalWidth * scale)); cv.height = Math.max(1, Math.round(img.naturalHeight * scale));
        const g = cv.getContext("2d");
        const png = file.type === "image/png" && cv.width * cv.height <= 512 * 512;
        if (!png) { g.fillStyle = "#fff"; g.fillRect(0, 0, cv.width, cv.height); }
        g.drawImage(img, 0, 0, cv.width, cv.height);
        const type = png ? "image/png" : "image/jpeg";
        resolve({ type, data: cv.toDataURL(type, 0.88).split(",")[1] });
      };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("not a picture the browser can read")); };
      img.src = url;
    });
  }
  async function attachPictures(i, files) {
    const key = draftKey(i), list = L.attach[key] ||= [];
    for (const f of files) {
      if (!f.type.startsWith("image/") || list.length >= MAX_PICTURES) continue;
      try { list.push(await shrinkPicture(f)); } catch (e) { console.warn("cannot attach " + f.name, e); }
    }
    const p = $(`[data-prompt="${i}"]`); if (p) L.drafts[key] = p.value;
    render();
    $(`[data-prompt="${i}"]`)?.focus();
  }
  const pending = new Map();
  function persist(c) {
    saveConvos();
    if (!isLive()) return;
    clearTimeout(pending.get(c.id));
    pending.set(c.id, setTimeout(() => { pending.delete(c.id); upload(c); }, 400));
  }
  function upload(c) {
    const body = { ...c, messages: c.messages.map(m => { const { streaming, ...rest } = m; return rest; }) };
    return fetch("/api/chats", { method: "POST", headers: { "X-Studio-Token": tokenMeta.content, "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then(r => { if (!r.ok) console.warn("cannot save the conversation", r.status); return r.ok; })
      .catch(e => { console.warn("cannot save the conversation", e); return false; });
  }
  async function deleteConvo(id) {
    L.convos = L.convos.filter(c => c.id !== id);
    L.tabs = L.tabs.filter(t => t !== id);
    L.panes = L.panes.map(p => p.id === id ? { id: null } : p);
    saveConvos(); savePanes();
    render();
    try {
      await fetch("/api/chats", { method: "POST", headers: { "X-Studio-Token": tokenMeta.content, "Content-Type": "application/json" }, body: JSON.stringify({ id, delete: true }) });
    } catch (e) { console.warn("cannot delete the conversation", e); }
  }
  // The studio's copy wins; conversations only this browser has (from before) are sent up once.
  async function loadConvos() {
    try {
      const saved = (await api("/api/chats")).chats || [];
      const known = new Set(saved.map(c => c.id));
      const onlyHere = L.convos.filter(c => !known.has(c.id) && /^[a-z0-9]{4,40}$/.test(c.id));
      L.convos = [...saved, ...onlyHere].sort((a, b) => b.updated - a.updated);
      L.convosLoaded = true;
      saveConvos();
      // once the studio has every one, this browser lets go of its copy
      Promise.all(onlyHere.map(upload)).then(oks => { if (pageState && oks.every(Boolean)) local.remove("convos"); });
      // a browser that has not opened any tabs yet starts with the open conversations as tabs
      if (store.get("chatTabs", null) === null) L.tabs = L.convos.filter(c => c.status !== "finished").slice(0, 6).map(c => c.id);
      fitPanes();
      if (state.view === "chat" && !anyRunning()) render();
    } catch (e) { console.warn("cannot read conversations", e); }
  }

  function liveThreadList() {
    const q = (L.search || "").toLowerCase();
    const match = c => !q || c.title.toLowerCase().includes(q);
    const open = L.convos.filter(c => c.status !== "finished" && match(c)).sort((a, b) => b.updated - a.updated);
    const done = L.convos.filter(c => c.status === "finished" && match(c)).sort((a, b) => (b.finishedAt || b.updated) - (a.finishedAt || a.updated));
    const shownIn = id => L.panes.findIndex(p => p.id === id);
    const item = c => {
      const last = c.messages.findLast(m => m.role === "assistant" && m.model);
      const at = shownIn(c.id), fin = c.status === "finished";
      return `<div class="titem ${at === L.active && L.sel !== "example" ? "active" : at >= 0 ? "shown" : ""} ${fin ? "done" : ""}" data-convo="${c.id}" tabindex="0">
        ${running(c.id) ? `<span class="live-dot" title="Answering"></span>` : fin ? `<span class="ic donemark">${icon("check")}</span>` : `<span class="dot ic" style="--c:var(--cap-text)"></span>`}
        <div style="min-width:0"><div class="t">${esc(c.title)}</div><div class="s">${esc(last ? last.model.name : "no answer yet")}</div></div>
        <span class="when">${when(fin ? c.finishedAt || c.updated : c.updated)}</span>
        <span class="tacts">
          ${!fin && maxPanes() > 1 && at < 0 ? `<button class="iconbtn sm" data-beside="${c.id}" aria-label="Open beside" title="Open beside">${icon("panel")}</button>` : ""}
          ${fin ? `<button class="iconbtn sm" data-reopen="${c.id}" aria-label="Reopen" title="Reopen">${icon("retry")}</button>
            <button class="iconbtn sm" data-delete="${c.id}" aria-label="Delete" title="Delete">${icon("trash")}</button>`
            : `<button class="iconbtn sm" data-finish="${c.id}" aria-label="Mark finished" title="Mark finished">${icon("check")}</button>`}
        </span></div>`;
    };
    return `
    <aside class="threads" aria-label="Conversations">
      <div class="col-head"><span class="h3">Threads</span><span class="grow"></span>
        <button class="iconbtn" id="new-convo" aria-label="New conversation" title="New conversation">${icon("plus")}</button></div>
      <div class="search">${icon("search")}<input class="input" id="thread-search" placeholder="Search conversations" value="${esc(L.search || "")}"></div>
      <div class="scroll">
        <div class="tgroup"><span class="eyebrow">Open <span class="faint">${open.length || ""}</span></span>
          ${open.length ? open.map(item).join("") : `<div class="muted" style="padding:2px 8px 8px;font-size:12.5px">${q ? "Nothing matches." : "Your conversations appear here."}</div>`}
        </div>
        ${done.length ? `<div class="tgroup"><button class="eyebrow fold" id="fold-finished" aria-expanded="${!!(L.showFinished || q)}">${icon("chev")}Finished <span class="faint">${done.length}</span></button>
          ${L.showFinished || q ? done.map(item).join("") : ""}</div>` : ""}
      </div>
    </aside>`;
  }

  // answers saved before the studio split plain-text <think> blocks off: show them as reasoning too
  function splitThink(m) {
    if (m.role === "user" || m.thinking || !m.text) return m;
    const r = /^\s*<think>([\s\S]*?)(?:<\/think>|$)\s*/.exec(m.text);
    return r ? { ...m, thinking: r[1].trim(), text: m.text.slice(r[0].length) } : m;
  }

  // The Reasoning bubble: the one place a model's thinking is drawn, in Chat and in Agents.
  const thinkHtml = (text, { open = false, live = false, attrs = "" } = {}) => `<details class="tool think" ${attrs} ${open ? "open" : ""}>
      <summary>${icon("wand")}<span><b>Reasoning</b></span><span class="grow"></span><span class="mono muted">${live ? "thinking…" : "summary"}</span></summary><pre>${esc(text)}</pre></details>`;

  function msgHtml(m, i) {
    m = splitThink(m);
    if (m.role === "user") {
      return `<div class="msg user" data-i="${i}"><div class="avatar user">${icon("chat")}</div><div style="min-width:0">
        <div class="meta"><b>You</b><span>${m.at ? when(m.at) : ""}</span></div>${m.images?.length ? `<div class="msg-pics">${m.images.map(p => `<a href="${picUrl(p)}" target="_blank" rel="noopener"><img src="${picUrl(p)}" alt="Attached picture"></a>`).join("")}</div>` : ""}<div class="body">${md(m.text)}</div>
        <div class="actions">
          <button class="iconbtn" data-copy="${i}" aria-label="Copy" title="Copy">${icon("copy")}</button>
          <button class="iconbtn" data-edit-msg="${i}" aria-label="Edit and send again" title="Edit and send again: the answers after it are replaced">${icon("edit")}</button></div></div></div>`;
    }
    const s = m.stats, model = m.model;
    const rate = s && s.outputTokens > 0 && s.seconds > 0 ? `${(s.outputTokens / s.seconds).toFixed(0)} tok/s` : "";
    const notes = [];
    if (s && s.stop === "length") notes.push(`<span class="chip c-warn">cut at the length limit</span>`);
    if (s && s.stop === "refusal") notes.push(`<span class="chip c-warn">the model declined</span>`);
    if (m.stopped) notes.push(`<span class="chip plain">stopped</span>`);
    return `<div class="msg" data-i="${i}">
      <div class="avatar ai" style="--c:var(--cap-text)">${icon("text")}</div>
      <div style="min-width:0">
        <div class="meta"><b>${esc(model ? model.name : "Choosing a model…")}</b>
          ${model ? (model.local ? `<span class="chip c-ok">this PC</span>` : `<span class="chip plain">${esc(providerName(model.provider))}</span>`) : ""}
          ${s ? `<span class="mono">${s.outputTokens >= 0 ? s.outputTokens + " tok" : ""}${rate ? " · " + rate : ""}${s.seconds ? " · " + s.seconds.toFixed(1) + " s" : ""}</span>` : ""}
          ${notes.join("")}</div>
        <div class="body">
          ${m.thinking ? thinkHtml(m.thinking, { open: m.streaming && !m.text, live: m.streaming && !m.text }) : ""}
          ${md(m.text)}
          ${m.streaming && !m.text && !m.thinking ? `<span class="typing" aria-label="Writing"><i></i><i></i><i></i></span>` : ""}
          ${m.error ? `<div class="flag bad">${icon("flask")}<div><b>No answer.</b> ${esc(m.error)}</div></div>` : ""}
        </div>
        ${m.streaming ? "" : `<div class="actions">
          <button class="iconbtn" data-copy="${i}" aria-label="Copy" title="Copy">${icon("copy")}</button>
          <button class="iconbtn" data-retry="${i}" aria-label="Answer again" title="Answer again">${icon("retry")}</button></div>`}
      </div></div>`;
  }

  function routeLine(model) {
    if (!L.info) return `${icon("wand")}<span>Looking for models on this PC and at your providers…</span>`;
    const key = model === "auto" ? L.info.auto.text : model;
    const m = modelByKey(key);
    if (!key) return `${icon("wand")}<span>No chat model yet.</span>`;
    const pid = m ? m.provider : key.split("/")[0], kind = providerKind(pid);
    const where = kind === "local" ? "this PC · nothing leaves it" : kind === "host" ? `${providerName(pid)} · your machine` : `${providerName(pid)} · uses your key`;
    return `${icon("wand")}<span>${model === "auto" ? "Auto will send this to" : "Sending to"}</span><span class="chip c-text">${esc(m ? m.name : key)}</span><span class="faint">${esc(where)}${m && m.detail ? " · " + esc(m.detail) : ""}</span>`;
  }

  function modelPicker(i) {
    const sel = paneModel(i);
    const groups = {};
    chatModels(sel).forEach(m => (groups[m.provider] ||= []).push(m));
    const auto = L.info && L.info.auto.text ? modelByKey(L.info.auto.text) : null;
    return `<select class="input model-pick" data-pick-model="${i}" aria-label="Model">
      <option value="auto" ${sel === "auto" ? "selected" : ""}>Auto${auto ? " · " + esc(auto.name) : ""}</option>
      ${Object.entries(groups).map(([p, ms]) => `<optgroup label="${esc(providerName(p))}">${ms.map(m => `<option value="${esc(m.key)}" ${m.key === sel ? "selected" : ""}>${esc(m.name)}</option>`).join("")}</optgroup>`).join("")}
    </select>`;
  }

  function emptyState() {
    if (!L.info) return `<div class="empty"><span class="typing"><i></i><i></i><i></i></span></div>`;
    if (chatModels().length) {
      return `<div class="empty"><div class="h2">What should we work on?</div>
        <div class="muted">${L.info.auto.text ? `Auto picks ${esc(modelByKey(L.info.auto.text)?.name || L.info.auto.text)}, this PC first. Choose any model above.` : ""}</div></div>`;
    }
    const local = L.info.providers.find(p => p.local);
    return `<div class="empty"><div class="flag">${icon("flask")}<div><b>No chat model yet.</b> Put a Hugging Face model folder or a <code>.gguf</code> file in
      ${(local?.folders || []).map(f => `<code>${esc(f)}</code>`).join(" or ")}, or add an API key under Models.
      <div style="margin-top:8px"><button class="btn" data-go="library">${icon("library")}Open Models and providers</button></div></div></div></div>`;
  }

  function paneHtml(i) {
    const c = paneConvo(i), fin = c && c.status === "finished", busy = c && running(c.id);
    const key = c ? c.id : "";
    return `
    <section class="stage pane ${i === L.active ? "active" : ""}" data-pane="${i}" data-convo="${key}" aria-label="${esc(c ? c.title : "New conversation")}">
      <div class="stage-head">
        <span class="title">${c ? esc(c.title) : "New conversation"}</span>
        ${fin ? `<span class="chip c-ok">finished</span>` : ""}
        <span class="grow"></span>
        ${fin ? "" : modelPicker(i)}
        ${fin ? "" : `<button class="iconbtn ${hasSettings(paneSettings(i)) ? "has-set" : ""}" data-chat-settings="${i}" aria-label="Instructions, answer length and sampling" title="Instructions, answer length and sampling${hasSettings(paneSettings(i)) ? " (set)" : ""}" aria-expanded="${L.settingsOpen === (key || "#" + i)}">${icon("gear")}</button>`}
        ${c ? (fin ? `<button class="btn sm" data-reopen="${c.id}">${icon("retry")}Reopen</button>`
          : `<button class="btn ghost sm" data-finish="${c.id}" title="Mark finished: it leaves the tabs and waits under Finished">${icon("check")}Finish</button>`) : ""}
        ${L.panes.length > 1 ? `<button class="iconbtn" data-close-pane="${i}" aria-label="Close this pane" title="Close this pane"><span aria-hidden="true" style="font-size:17px;line-height:1">×</span></button>` : ""}
      </div>
      ${!fin && (L.settingsOpen === (key || "#" + i)) ? `<div class="chat-settings">
        <div class="field"><label for="cs-system-${i}">Instructions <span class="faint">a system prompt, sent before every message of this conversation</span></label>
          <textarea class="input" id="cs-system-${i}" data-cs-system="${i}" rows="3" placeholder="For example: You are a terse code reviewer. Answer in German.">${esc(paneSettings(i).system || "")}</textarea></div>
        <div class="field cs-len"><label for="cs-max-${i}">Longest answer <span class="faint">tokens</span></label>
          <input class="input mono" id="cs-max-${i}" data-cs-max="${i}" type="number" min="1" max="65536" step="1" placeholder="8192" value="${esc(paneSettings(i).maxTokens || "")}">
          <label for="cs-temp-${i}" title="0 always picks the likeliest word; higher is more varied. Empty leaves it to the model.">Temperature <span class="faint">0 to 2</span></label>
          <input class="input mono" id="cs-temp-${i}" data-cs-temp="${i}" type="number" min="0" max="2" step="0.1" placeholder="model's own" value="${esc(paneSettings(i).temperature ?? "")}"></div>
        <div class="field cs-len cs-sampling">
          <label for="cs-topp-${i}" title="The model picks among the likeliest words that together reach this probability. Lower is more focused. Empty leaves it to the model.">Top-p <span class="faint">0 to 1</span></label>
          <input class="input mono" id="cs-topp-${i}" data-cs-num="${i}" data-k="topP" data-min="0.01" data-max="1" type="number" min="0.01" max="1" step="0.05" placeholder="model's own" value="${esc(paneSettings(i).topP ?? "")}">
          <label for="cs-topk-${i}" title="The model picks among this many likeliest words. Empty leaves it to the model. Claude and GPT models have no such setting.">Top-k <span class="faint">words</span></label>
          <input class="input mono" id="cs-topk-${i}" data-cs-num="${i}" data-k="topK" data-min="1" data-max="1000" data-int="1" type="number" min="1" max="1000" step="1" placeholder="model's own" value="${esc(paneSettings(i).topK ?? "")}">
          <label for="cs-rep-${i}" title="How strongly words already written are held back: 1 is not at all. For models on this PC and on worker hosts.">Repeat penalty <span class="faint">1 to 2</span></label>
          <input class="input mono" id="cs-rep-${i}" data-cs-num="${i}" data-k="repeatPenalty" data-min="1" data-max="2" type="number" min="1" max="2" step="0.05" placeholder="model's own" value="${esc(paneSettings(i).repeatPenalty ?? "")}"></div>
      </div>` : ""}
      <div class="transcript" data-transcript="${i}"><div class="lane">${c && c.messages.length ? c.messages.map(msgHtml).join("") : emptyState()}</div></div>
      <div class="composer-wrap">
        ${fin ? `<div class="finished-bar">${icon("check")}<span>Finished ${when(c.finishedAt || c.updated)}. Reopen it to keep going.</span><span class="grow"></span>
            <button class="btn sm" data-reopen="${c.id}">Reopen</button></div>`
        : `<div class="composer">
          ${c && L.editingMsg && L.editingMsg.convo === c.id ? `<div class="route-hint editing">${icon("edit")}<span>Editing an earlier message. Sending replaces it and the answers after it.</span><span class="grow"></span><button class="btn ghost sm" data-edit-cancel>Cancel</button></div>` : ""}
          <div class="route-hint" data-route="${i}">${routeLine(paneModel(i))}</div>
          ${(L.attach[draftKey(i)] || []).length ? `<div class="attached">${L.attach[draftKey(i)].map((p, n) => `<span class="pic"><img src="${picUrl(p)}" alt="Picture ${n + 1}"><button type="button" data-unattach="${i}:${n}" aria-label="Remove picture ${n + 1}" title="Remove">×</button></span>`).join("")}</div>` : ""}
          <textarea data-prompt="${i}" rows="2" placeholder="Ask anything. Enter sends, Shift+Enter adds a line">${esc(L.drafts[draftKey(i)] || "")}</textarea>
          <div class="row">
            <button type="button" class="iconbtn" data-attach="${i}" aria-label="Attach pictures" title="Attach pictures for a model that can look at them (Claude, GPT, Gemini). You can also paste or drop them.">${icon("image")}</button>
            <input type="file" accept="image/*" multiple hidden data-attach-file="${i}">
            <span class="faint" style="font-size:12px;padding-left:2px">Saved in the studio</span>
            <span class="grow"></span>
            ${busy ? `<button class="send stop" data-stop="${key}" aria-label="Stop">${icon("stop")}</button>` : `<button class="send" data-send="${i}" aria-label="Send">${icon("send")}</button>`}
          </div>
        </div>`}
      </div>
    </section>`;
  }

  function tabsBar() {
    const tabs = L.tabs.map(convoById).filter(Boolean);
    const room = L.panes.length < maxPanes();
    return `<div class="chattabs" role="tablist" aria-label="Open conversations">
      <button class="iconbtn sm threads-toggle" id="threads-toggle" aria-label="All conversations" title="All conversations, open and finished" aria-expanded="${!!L.showThreads}">${icon("chat")}</button>
      ${tabs.map(c => {
        const at = L.panes.findIndex(p => p.id === c.id);
        return `<div class="ctab ${at === L.active ? "active" : at >= 0 ? "shown" : ""}" role="tab" aria-selected="${at === L.active}">
          <button class="ctab-open" data-tab="${c.id}" title="${esc(c.title)}">${running(c.id) ? `<span class="live-dot"></span>` : ""}<span>${esc(c.title)}</span></button>
          <button class="ctab-x" data-close-tab="${c.id}" aria-label="Close tab ${esc(c.title)}" title="Close tab (the conversation stays under Open)">×</button></div>`;
      }).join("")}
      <button class="ctab-new" id="tab-new" aria-label="New conversation" title="New conversation">${icon("plus")}</button>
      <span class="grow"></span>
      ${maxPanes() > 1 ? `<button class="btn ghost sm" id="split" ${room ? "" : "disabled"} title="${room ? "Open a new conversation beside this one" : "No room for another pane at this width"}">${icon("panel")}Split</button>` : ""}
    </div>`;
  }

  function liveChatView() {
    fitPanes();
    return `<div class="chat livechat lay-${effectiveLayout()} no-inspector ${L.showThreads ? "show-threads" : ""}">${liveThreadList()}
      <div class="chatmain">${tabsBar()}<div class="panes" style="--n:${L.panes.length}">${L.panes.map((_, i) => paneHtml(i)).join("")}</div></div></div>`;
  }

  // Repaints one message while it streams, at most once a frame, in every pane that shows it.
  const paintQueued = new Map();
  function paintMsg(c, i) {
    if (paintQueued.has(c.id)) return;
    paintQueued.set(c.id, requestAnimationFrame(() => {
      paintQueued.delete(c.id);
      if (state.view !== "chat") return;
      $$(`.pane[data-convo="${c.id}"]`).forEach(pane => {
        const el = pane.querySelector(`.msg[data-i="${i}"]`), box = pane.querySelector(".transcript");
        if (!el || !box) return;
        const stick = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
        el.outerHTML = msgHtml(c.messages[i], i);
        if (stick) box.scrollTop = box.scrollHeight;
      });
    }));
  }

  // The provider wants alternating turns: a failed answer is left out and the user's turns around it are joined.
  function turns(messages) {
    const out = [];
    for (const m of messages) {
      if (m.role === "assistant" && (m.error || !m.text)) continue;
      const last = out[out.length - 1];
      if (last && last.role === m.role) { last.text += "\n\n" + m.text; if (m.images?.length) last.images = [...(last.images || []), ...m.images].slice(0, MAX_PICTURES); }
      else out.push(m.images?.length ? { role: m.role, text: m.text, images: m.images } : { role: m.role, text: m.text });
    }
    return out;
  }

  async function answer(c) {
    const a = { role: "assistant", text: "", thinking: "", model: null, streaming: true, at: Date.now() };
    const history = turns(c.messages);
    c.messages.push(a);
    const i = c.messages.length - 1;
    const ctrl = new AbortController();
    L.running[c.id] = ctrl;
    render();
    const onEvent = (event, d) => {
      if (event === "route") a.model = d;
      else if (event === "text") a.text += d.text;
      else if (event === "thinking") a.thinking += d.text;
      else if (event === "done") a.stats = d;
      else if (event === "error") a.error = d.message;
      paintMsg(c, i);
    };
    try {
      const r = await fetch("/api/chat", {
        method: "POST", signal: ctrl.signal,
        headers: { "X-Studio-Token": tokenMeta.content, "Content-Type": "application/json" },
        body: JSON.stringify({ model: usable(c.model || L.model), messages: history, system: (c.system || "").trim() || undefined, maxTokens: +c.maxTokens > 0 ? Math.floor(+c.maxTokens) : undefined, temperature: c.temperature ?? undefined, topP: c.topP ?? undefined, topK: c.topK ?? undefined, repeatPenalty: c.repeatPenalty ?? undefined }),
      });
      await readEvents(r, onEvent);
    } catch (e) {
      if (e.name === "AbortError") a.stopped = true; else a.error = e.message;
    }
    a.streaming = false;
    delete L.running[c.id];
    c.updated = Date.now();
    persist(c);
    if (state.view === "chat") render();
  }

  // reads a server-sent event stream from the studio, after checking the answer is one
  async function readEvents(r, onEvent) {
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `the studio answered ${r.status}`);
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, cut); buf = buf.slice(cut + 2);
        let event = "message", data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7);
          else if (line.startsWith("data: ")) data += line.slice(6);
        }
        if (data) onEvent(event, JSON.parse(data));
      }
    }
  }

  function send(i) {
    const p = $(`[data-prompt="${i}"]`), text = (p?.value || "").trim();
    let c = paneConvo(i);
    const pics = L.attach[draftKey(i)] || [];
    if ((!text && !pics.length) || !L.info || (c && running(c.id))) return;
    delete L.attach[draftKey(i)];
    if (!c) {
      c = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), title: (text || "Picture").replace(/\s+/g, " ").slice(0, 70),
        messages: [], updated: Date.now(), status: "open", model: L.panes[i].model || L.model };
      if ((L.panes[i].system || "").trim()) c.system = L.panes[i].system;
      if (+L.panes[i].maxTokens > 0) c.maxTokens = +L.panes[i].maxTokens;
      if (L.panes[i].temperature != null) c.temperature = L.panes[i].temperature;
      for (const k of ["topP", "topK", "repeatPenalty"]) if (L.panes[i][k] != null) c[k] = L.panes[i][k];
      if (L.settingsOpen === "#" + i) L.settingsOpen = c.id;
      L.convos.unshift(c);
      L.panes[i] = { id: c.id };
      L.tabs.push(c.id);
      L.drafts["#" + i] = "";
      savePanes();
    }
    L.active = i;
    L.drafts[c.id] = "";
    // an edited message takes the place of the old one; what was answered after it goes
    if (L.editingMsg && L.editingMsg.convo === c.id) { c.messages.splice(L.editingMsg.at); L.editingMsg = null; }
    c.messages.push(pics.length ? { role: "user", text, images: pics, at: Date.now() } : { role: "user", text, at: Date.now() });
    persist(c);
    answer(c);
  }

  function bindLiveChat() {
    $$(".titem[data-convo]").forEach(el => {
      const open = e => {
        if (e.target.closest(".tacts")) return;
        if (el.dataset.convo === "example") { L.sel = "example"; store.set("convo", "example"); render(); return; }
        store.set("convo", null);
        L.showThreads = false;
        openConvo(el.dataset.convo, e.ctrlKey || e.metaKey);
      };
      el.addEventListener("click", open);
      el.addEventListener("keydown", e => { if (e.key === "Enter") open(e); });
    });
    $$("[data-beside]").forEach(b => b.addEventListener("click", () => openConvo(b.dataset.beside, true)));
    $$("[data-finish]").forEach(b => b.addEventListener("click", () => setFinished(b.dataset.finish, true)));
    $$("[data-reopen]").forEach(b => b.addEventListener("click", () => { setFinished(b.dataset.reopen, false); openConvo(b.dataset.reopen); }));
    $$("[data-delete]").forEach(b => b.addEventListener("click", () => {
      const c = convoById(b.dataset.delete);
      if (c && confirmDelete(b)) deleteConvo(c.id);
    }));
    $("#threads-toggle")?.addEventListener("click", () => { L.showThreads = !L.showThreads; render(); });
    $("#fold-finished")?.addEventListener("click", () => { L.showFinished = !L.showFinished; render(); });
    $("#new-convo")?.addEventListener("click", () => newChat(false));
    $("#tab-new")?.addEventListener("click", () => newChat(false));
    $("#split")?.addEventListener("click", () => newChat(true));
    $$("[data-tab]").forEach(b => b.addEventListener("click", () => openConvo(b.dataset.tab)));
    $$("[data-close-tab]").forEach(b => b.addEventListener("click", () => closeTab(b.dataset.closeTab)));
    $$("[data-close-pane]").forEach(b => b.addEventListener("click", () => closePane(+b.dataset.closePane)));
    $("#thread-search")?.addEventListener("input", e => {
      L.search = e.target.value; const at = e.target.selectionStart; render();
      const s = $("#thread-search"); s.focus(); s.setSelectionRange(at, at);
    });
    $$(".pane").forEach(pane => {
      const i = +pane.dataset.pane;
      pane.addEventListener("focusin", () => { if (L.active !== i) { L.active = i; savePanes(); $$(".pane").forEach(x => x.classList.toggle("active", +x.dataset.pane === i)); } });
      const box = pane.querySelector(".transcript"), key = pane.dataset.convo;
      if (box) {
        box.scrollTop = key && L.scroll[key] != null ? L.scroll[key] : box.scrollHeight;
        box.addEventListener("scroll", () => { if (key) L.scroll[key] = box.scrollHeight - box.scrollTop - box.clientHeight < 80 ? null : box.scrollTop; }, { passive: true });
      }
    });
    $$("[data-pick-model]").forEach(s => s.addEventListener("change", () => {
      const i = +s.dataset.pickModel, c = paneConvo(i);
      if (c) { c.model = s.value; persist(c); } else { L.panes[i].model = s.value; L.model = s.value; store.set("chatModel", s.value); savePanes(); }
      $(`[data-route="${i}"]`).innerHTML = routeLine(s.value);
    }));
    $$("[data-chat-settings]").forEach(b => b.addEventListener("click", () => {
      const k = draftKey(+b.dataset.chatSettings);
      L.settingsOpen = L.settingsOpen === k ? null : k;
      render();
      if (L.settingsOpen) $(`[data-cs-system="${b.dataset.chatSettings}"]`)?.focus();
    }));
    const setSetting = (i, k, v) => {
      const c = paneConvo(i);
      const unset = v == null || v === "" || (v === 0 && (k === "maxTokens" || k === "system"));
      if (c) { if (unset) delete c[k]; else c[k] = v; persist(c); } else { L.panes[i][k] = unset ? undefined : v; savePanes(); }
      $(`[data-chat-settings="${i}"]`)?.classList.toggle("has-set", hasSettings(paneSettings(i)));
    };
    $$("[data-cs-system]").forEach(t => t.addEventListener("input", () => setSetting(+t.dataset.csSystem, "system", t.value)));
    $$("[data-cs-max]").forEach(t => t.addEventListener("input", () => setSetting(+t.dataset.csMax, "maxTokens", Math.max(0, Math.min(65536, Math.floor(+t.value || 0))))));
    $$("[data-cs-temp]").forEach(t => t.addEventListener("input", () => setSetting(+t.dataset.csTemp, "temperature", t.value.trim() === "" || !isFinite(+t.value) ? null : Math.max(0, Math.min(2, +t.value)))));
    $$("[data-cs-num]").forEach(t => t.addEventListener("input", () => {
      const raw = t.value.trim(), n = t.dataset.int ? Math.round(+raw) : +raw;
      setSetting(+t.dataset.csNum, t.dataset.k, raw === "" || !isFinite(n) ? null : Math.max(+t.dataset.min, Math.min(+t.dataset.max, n)));
    }));
    $$("[data-edit-msg]").forEach(b => b.addEventListener("click", () => {
      const pane = b.closest(".pane"), c = convoById(pane?.dataset.convo);
      if (!c || running(c.id) || c.status === "finished") return;
      const at = +b.dataset.editMsg, i = +pane.dataset.pane;
      L.editingMsg = { convo: c.id, at };
      L.drafts[c.id] = c.messages[at].text;
      if (c.messages[at].images?.length) L.attach[c.id] = c.messages[at].images.slice(); else delete L.attach[c.id];
      render();
      const p = $(`[data-prompt="${i}"]`); p?.focus(); p?.setSelectionRange(p.value.length, p.value.length);
    }));
    $$("[data-edit-cancel]").forEach(b => b.addEventListener("click", () => { const e = L.editingMsg; L.editingMsg = null; if (e) { L.drafts[e.convo] = ""; delete L.attach[e.convo]; } render(); }));
    $$("[data-prompt]").forEach(p => {
      const i = +p.dataset.prompt;
      p.addEventListener("input", () => { L.drafts[draftKey(i)] = p.value; });
      p.addEventListener("paste", e => {
        const files = [...(e.clipboardData?.files || [])].filter(f => f.type.startsWith("image/"));
        if (files.length) { e.preventDefault(); attachPictures(i, files); }
      });
      const box = p.closest(".composer");
      box?.addEventListener("dragover", e => { if ([...(e.dataTransfer?.types || [])].includes("Files")) e.preventDefault(); });
      box?.addEventListener("drop", e => {
        const files = [...(e.dataTransfer?.files || [])].filter(f => f.type.startsWith("image/"));
        if (files.length) { e.preventDefault(); attachPictures(i, files); }
      });
      p.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(i); } });
    });
    $$("[data-send]").forEach(b => b.addEventListener("click", () => send(+b.dataset.send)));
    $$("[data-attach]").forEach(b => b.addEventListener("click", () => $(`[data-attach-file="${b.dataset.attach}"]`)?.click()));
    $$("[data-attach-file]").forEach(f => f.addEventListener("change", () => attachPictures(+f.dataset.attachFile, [...f.files])));
    $$("[data-unattach]").forEach(b => b.addEventListener("click", () => {
      const [i, n] = b.dataset.unattach.split(":").map(Number), key = draftKey(i);
      const p = $(`[data-prompt="${i}"]`); if (p) L.drafts[key] = p.value;
      (L.attach[key] || []).splice(n, 1);
      render();
    }));
    $$("[data-stop]").forEach(b => b.addEventListener("click", () => L.running[b.dataset.stop]?.abort()));
    $$("[data-go]").forEach(b => b.addEventListener("click", () => go(b.dataset.go)));
    $$("[data-copy]").forEach(b => b.addEventListener("click", () => {
      const m = convoById(b.closest(".pane")?.dataset.convo)?.messages[+b.dataset.copy];
      if (m) navigator.clipboard?.writeText(m.text).then(() => { b.innerHTML = icon("check"); setTimeout(() => { b.innerHTML = icon("copy"); }, 1200); });
    }));
    $$("[data-retry]").forEach(b => b.addEventListener("click", () => {
      const c = convoById(b.closest(".pane")?.dataset.convo); if (!c || running(c.id)) return;
      c.messages.splice(+b.dataset.retry);
      answer(c);
    }));
    const a = document.activeElement;
    if (!a || a === document.body) $(`[data-prompt="${L.active}"]`)?.focus({ preventScroll: true });
  }
  // Deleting asks twice on the same button: the first press arms it for three seconds.
  function confirmDelete(b) {
    if (b.dataset.armed) return true;
    b.dataset.armed = "1"; b.classList.add("armed"); b.title = "Press again to delete";
    setTimeout(() => { delete b.dataset.armed; b.classList.remove("armed"); b.title = "Delete"; }, 3000);
    return false;
  }

  // ---------- live images and voice ----------
  // Pictures and takes stay in this tab (as blob URLs); the studio also saves each file to its
  // outputs folder, and the tile or take shows where.
  const G = {
    imgModel: store.get("imgModel", "auto"), aspect: store.get("aspect", "1:1"), count: store.get("imgCount", 2),
    prompt: "", batches: [], drawing: null,
    // drawing on this PC: what is not sent anywhere else, and kept per model where it differs
    neg: "", steps: {}, guidance: {}, seed: "", sampler: store.get("sampler", "DPM_PP_2M_KARRAS"), opts: {}, stop: null,
    voModel: store.get("voModel", "auto"), voice: store.get("voice", null), speed: 1, instructions: "", format: "mp3",
    // a second voice mixed in, for models whose voices blend, and how much of the first to keep
    blend: "", blendW: 0.7,
    text: "", takes: [], speaking: false, video: null,
    // a voice made from a recording, for models that clone (F5-TTS)
    clone: { name: "", text: "", sample: null, rec: null, busy: false, error: "" },
    // writing down what is said, on this PC
    mode: store.get("voMode", "speak"), trModel: store.get("trModel", "auto"), trLang: store.get("trLang", "auto"),
    trTranslate: false, trTimes: true, trOpts: {}, transcripts: [], listening: null, recorder: null,
  };
  const mediaModels = medium => (L.info?.models || []).filter(m => m.media.includes(medium) && m.shown !== false);
  const pickedModel = (sel, medium) => modelByKey(sel === "auto" ? L.info?.auto?.[medium] : sel);

  function mediaPicker(id, medium, sel) {
    const groups = {};
    mediaModels(medium).forEach(m => (groups[m.provider] ||= []).push(m));
    const auto = L.info?.auto?.[medium] ? modelByKey(L.info.auto[medium]) : null;
    return `<select class="input" id="${id}" aria-label="Model">
      <option value="auto" ${sel === "auto" ? "selected" : ""}>Auto${auto ? " · " + esc(auto.name) : ""}</option>
      ${Object.entries(groups).map(([p, ms]) => `<optgroup label="${esc(providerName(p))}">${ms.map(m => `<option value="${esc(m.key)}" ${m.key === sel ? "selected" : ""}>${esc(m.name)}</option>`).join("")}</optgroup>`).join("")}
    </select>`;
  }

  function remoteNote(m, what) {
    if (!m) return "";
    return `<div class="flag">${icon("flask")}<div><b>Runs at ${esc(providerName(m.provider))}.</b> ${what} is sent to ${esc(providerName(m.provider))} and billed to your key. No local model offers this yet; one that does will show up here.</div></div>`;
  }

  function noModel(medium) {
    return `<div class="empty"><div class="flag">${icon("flask")}<div><b>No ${medium === "image" ? "image" : "speech"} model yet.</b>
      ${L.info ? (medium === "image" ? "Put a Stable Diffusion model (1.x, 2.x or XL, as a diffusers folder or a single .safetensors file) in a model folder to draw on this PC, or add an OpenAI or Google API key and their image models appear here."
        : "Nothing on this PC can speak yet. Add an OpenAI or Google API key and their voice models appear here.") : "Looking for models…"}
      <div style="margin-top:8px"><button class="btn" data-go="library">${icon("library")}Open Models and providers</button></div></div></div></div>`;
  }

  async function postJson(path, body) {
    const r = await fetch(path, { method: "POST", headers: { "X-Studio-Token": tokenMeta.content, "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || `the studio answered ${r.status}`);
    return j;
  }

  function blobUrl(item) {
    const bin = atob(item.data), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return URL.createObjectURL(new Blob([bytes], { type: item.mime }));
  }
  const fileName = p => p.split(/[\\/]/).pop();
  const ratioCss = a => a.replace(":", " / ");

  // the usual steps and guidance of a model on this PC, asked for once
  function drawOpts(m) {
    if (!m || m.provider !== "local") return null;
    const o = G.opts[m.key];
    if (!o) {
      G.opts[m.key] = { loading: true };
      postJson("/api/images/options", { model: m.key }).then(j => { G.opts[m.key] = j; if (state.view === "create") renderCreateSide(); })
        .catch(() => { G.opts[m.key] = { samplers: ["DPM_PP_2M_KARRAS"] }; });
      return null;
    }
    return o.loading ? null : o;
  }
  const samplerName = n => ({ EULER: "Euler", EULER_ANCESTRAL: "Euler ancestral", DPM_PP_2M: "DPM++ 2M", DPM_PP_2M_KARRAS: "DPM++ 2M Karras", EULER_KARRAS: "Euler Karras" })[n] || n;

  function localSettings(m) {
    const o = drawOpts(m) || {};
    const steps = G.steps[m.key] ?? o.steps ?? 25, guidance = G.guidance[m.key] ?? o.guidance ?? 7;
    // settings a model ignores are left out (FLUX.1 has no negative prompt or sampler choice, schnell no guidance)
    const neg = o.usesNegative !== false, guid = o.usesGuidance !== false, samp = o.usesSampler !== false;
    return `<div class="flag ok-note">${icon("lock")}<div><b>Draws on this PC.</b> Nothing is sent anywhere.${o.label ? ` <span class="muted">${esc(o.label)}</span>` : ""}</div></div>
      ${neg ? `<div class="field"><label for="img-neg">Leave out</label><input class="input" id="img-neg" placeholder="blurry, text, watermark" value="${esc(G.neg)}"></div>` : ""}
      <div class="img-grid">
        <div class="field"><label for="img-steps">Steps</label><input class="input mono" id="img-steps" type="number" min="1" max="150" value="${steps}"></div>
        ${guid ? `<div class="field"><label for="img-guidance" title="How closely to follow the prompt">Guidance</label><input class="input mono" id="img-guidance" type="number" min="0" max="30" step="0.5" value="${guidance}"></div>` : ""}
        <div class="field"><label for="img-seed">Seed</label><div class="seed-row"><input class="input mono" id="img-seed" inputmode="numeric" placeholder="random" value="${esc(G.seed)}">
          <button class="iconbtn sm" id="img-seed-dice" type="button" aria-label="Random seed" title="Random seed each time">${icon("retry")}</button></div></div>
        ${samp ? `<div class="field"><label for="img-sampler">Sampler</label><select class="input" id="img-sampler">${(o.samplers || [G.sampler]).map(n => `<option value="${n}" ${n === G.sampler ? "selected" : ""}>${esc(samplerName(n))}</option>`).join("")}</select></div>` : ""}
      </div>`;
  }

  function renderCreateSide() {
    const side = $(".create .side"); if (!side) return;
    const focus = document.activeElement?.id;
    keepCreateFields();
    render();
    if (focus) document.getElementById(focus)?.focus();
  }
  // what is typed in the side panel survives a redraw
  function keepCreateFields() {
    const m = pickedModel(G.imgModel, "image");
    if ($("#img-prompt")) G.prompt = $("#img-prompt").value;
    if ($("#img-neg")) G.neg = $("#img-neg").value;
    if ($("#img-seed")) G.seed = $("#img-seed").value.trim();
    // only what was changed by hand is kept: a shown default may be a stand-in until the model's own arrive
    const edited = el => el && el.value !== el.defaultValue;
    if (m && edited($("#img-steps"))) G.steps[m.key] = +$("#img-steps").value || undefined;
    if (m && edited($("#img-guidance"))) { const g = parseFloat($("#img-guidance").value); G.guidance[m.key] = isNaN(g) ? undefined : g; }
  }

  function liveCreateView() {
    const m = pickedModel(G.imgModel, "image");
    const local = m?.provider === "local";
    const ratios = [["1:1", 14, 14], ["4:3", 16, 12], ["16:9", 18, 10], ["3:4", 12, 16], ["9:16", 10, 18]];
    return `
    <div class="create">
      <aside class="side" aria-label="Image settings">
        <div style="display:flex;align-items:center;gap:8px"><span class="h2">Images</span><span class="grow"></span>${m ? `<span class="chip c-image">${esc(m.name)}</span>` : ""}</div>
        <div class="field"><label for="img-model">Model</label>${mediaPicker("img-model", "image", G.imgModel)}</div>
        ${local ? "" : remoteNote(m, "The prompt")}
        <div class="prompt-box"><textarea id="img-prompt" aria-label="Prompt" placeholder="What should the picture show?">${esc(G.prompt)}</textarea></div>
        <div class="field"><label>Aspect</label>
          <div class="ratio">${ratios.map(([l, w, h]) => `<button data-aspect="${l}" aria-pressed="${G.aspect === l}"><i style="width:${w}px;height:${h}px"></i>${l}</button>`).join("")}</div></div>
        <div class="field"><label>Pictures</label><div class="seg" style="width:100%">${[1, 2, 3, 4].map(n => `<button data-count="${n}" aria-pressed="${G.count === n}" style="flex:1">${n}</button>`).join("")}</div></div>
        ${local ? localSettings(m) : ""}
        ${G.drawing && G.stop ? `<button class="btn" id="img-stop" style="height:40px;justify-content:center">${icon("stop")}Stop drawing</button>` : `<button class="btn primary" id="img-go" style="height:40px;justify-content:center" ${!m || G.drawing ? "disabled" : ""}>`}${G.drawing && G.stop ? "" : `${icon("wand")}${G.drawing ? "Drawing…" : `Draw ${G.count}`} <span class="kbd" style="color:inherit;border-color:currentColor;background:transparent;opacity:.7">Ctrl ↵</span></button>`}
      </aside>
      <section class="gallery" aria-label="Pictures">
        ${!mediaModels("image").length ? noModel("image") : ""}
        ${G.drawing ? `<div id="drawing">${drawingHtml()}</div>` : ""}
        ${G.batches.map((b, bi) => `
          <div>
            <div class="batch-head"><span class="p">${esc(b.prompt)}</span><span class="mono muted">${esc(b.model)} · ${b.aspect}${b.local ? ` · ${b.local.steps} steps · ${samplerName(b.local.sampler)}` : ""}${b.seconds ? ` · ${b.seconds.toFixed(1)} s` : ""}</span></div>
            ${b.error ? `<div class="flag bad">${icon("flask")}<div><b>${b.stopped ? "Stopped." : "No picture."}</b> ${b.stopped ? "" : esc(b.error)}</div></div>` : ""}
            ${b.items.length ? `<div class="batch">${b.items.map(it => `<div class="tile" title="Saved as ${esc(it.file)}${it.seed != null ? ` · seed ${it.seed}` : ""}"><img src="${it.url}" alt="${esc(b.prompt)}" style="aspect-ratio:${ratioCss(b.aspect)}">
              ${it.seed != null ? `<span class="step">seed ${it.seed}</span>` : ""}
              <div class="ov"><a href="${it.url}" download="${esc(fileName(it.file))}">Save</a><button data-reuse="${bi}">Reuse prompt</button>${it.seed != null ? `<button data-reseed="${bi}:${it.seed}">Reuse seed</button>` : ""}</div></div>`).join("")}</div>` : ""}
          </div>`).join("")}
        ${mediaModels("image").length && !G.batches.length && !G.drawing ? `<div class="empty"><div class="h2">Describe a picture</div><div class="muted">Pictures are also saved in the studio's outputs folder.</div></div>` : ""}
      </section>
    </div>`;
  }

  // the pictures being drawn: shimmering tiles, or on this PC the latest rough preview and the step
  function drawingHtml() {
    const d = G.drawing;
    const pct = d.total ? Math.round(100 * d.step / d.total) : 0;
    const tiles = Array.from({ length: d.count }, (_, i) => {
      const done = d.items?.[i];
      if (done) return `<div class="tile"><img src="${done.url}" alt="" style="aspect-ratio:${ratioCss(d.aspect)}"></div>`;
      const now = d.total && Math.floor((d.step - 1) / (d.total / d.count)) === i;
      return `<div class="tile ${now && d.preview ? "previewing" : "wait"}" style="aspect-ratio:${ratioCss(d.aspect)}">${now && d.preview ? `<img src="${d.preview}" alt="" style="aspect-ratio:${ratioCss(d.aspect)}">` : ""}</div>`;
    }).join("");
    return `<div class="batch-head"><span class="eyebrow">Drawing</span><span class="p">${esc(d.prompt)}</span><span class="mono muted">${esc(d.model)}${d.total ? ` · step ${d.step} of ${d.total}` : d.local ? " · loading the model…" : ""}</span></div>
      ${d.total ? `<div class="draw-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"><i style="width:${pct}%"></i></div>` : ""}
      <div class="batch">${tiles}</div>`;
  }
  function paintDrawing() {
    const box = $("#drawing");
    if (box && G.drawing) box.innerHTML = drawingHtml(); else if (state.view === "create") render();
  }

  async function drawHere(m, prompt) {
    keepCreateFields();
    const o = G.opts[m.key] || {};
    const seed = G.seed === "" ? null : Math.max(0, Math.floor(+G.seed) || 0);
    const body = { model: m.key, prompt, negativePrompt: G.neg, aspect: G.aspect, count: G.count,
      steps: G.steps[m.key] ?? o.steps ?? 25, guidance: G.guidance[m.key] ?? o.guidance ?? 7, sampler: G.sampler };
    if (seed != null) body.seed = seed;
    const ctrl = new AbortController();
    G.stop = ctrl;
    G.drawing = { prompt, model: m.name, aspect: G.aspect, count: G.count, local: true, step: 0, total: 0, items: [] };
    const batch = { prompt, model: m.name, aspect: G.aspect, items: [], local: { steps: body.steps, sampler: body.sampler } };
    render();
    let last = 0;
    try {
      const r = await fetch("/api/images/draw", { method: "POST", signal: ctrl.signal,
        headers: { "X-Studio-Token": tokenMeta.content, "Content-Type": "application/json" }, body: JSON.stringify(body) });
      await readEvents(r, (event, d) => {
        if (event === "route") { batch.model = G.drawing.model = d.name; }
        else if (event === "step") { G.drawing.step = d.step; G.drawing.total = d.total; }
        else if (event === "preview") G.drawing.preview = "data:image/png;base64," + d.data;
        else if (event === "image") { const it = { url: blobUrl(d), file: d.file, seed: d.seed }; batch.items.push(it); G.drawing.items.push(it); }
        else if (event === "done") batch.seconds = d.seconds;
        else if (event === "error") { batch.error = d.message; batch.stopped = d.message === "stopped"; }
        // a preview a few times a second is plenty
        const now = performance.now();
        if (event !== "step" || now - last > 120) { last = now; paintDrawing(); }
      });
    } catch (e) {
      if (e.name === "AbortError") { batch.error = "stopped"; batch.stopped = true; } else batch.error = e.message;
    }
    G.batches.unshift(batch);
    G.drawing = null; G.stop = null;
    if (state.view === "create") render();
  }

  async function draw() {
    const m = pickedModel(G.imgModel, "image");
    const prompt = ($("#img-prompt")?.value || "").trim();
    if (!m || !prompt || G.drawing) return;
    if (m.provider === "local") return drawHere(m, prompt);
    G.prompt = prompt;
    G.drawing = { prompt, model: m.name, aspect: G.aspect, count: G.count };
    render();
    const batch = { prompt, model: m.name, aspect: G.aspect, items: [] };
    try {
      const j = await postJson("/api/images", { model: G.imgModel, prompt, aspect: G.aspect, count: G.count });
      batch.model = j.model.name; batch.seconds = j.seconds;
      batch.items = j.items.map(it => ({ url: blobUrl(it), file: it.file }));
    } catch (e) { batch.error = e.message; }
    G.batches.unshift(batch);
    G.drawing = null;
    if (state.view === "create") render();
  }

  function bindLiveCreate() {
    $("#img-model")?.addEventListener("change", e => { keepCreateFields(); G.imgModel = e.target.value; store.set("imgModel", G.imgModel); render(); });
    $$("[data-aspect]").forEach(b => b.addEventListener("click", () => { keepCreateFields(); G.aspect = b.dataset.aspect; store.set("aspect", G.aspect); render(); }));
    $$("[data-count]").forEach(b => b.addEventListener("click", () => { keepCreateFields(); G.count = +b.dataset.count; store.set("imgCount", G.count); render(); }));
    $("#img-prompt")?.addEventListener("input", e => { G.prompt = e.target.value; });
    $("#img-prompt")?.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") draw(); });
    $("#img-go")?.addEventListener("click", draw);
    $("#img-stop")?.addEventListener("click", () => G.stop?.abort());
    $("#img-sampler")?.addEventListener("change", e => { G.sampler = e.target.value; store.set("sampler", G.sampler); });
    $("#img-seed-dice")?.addEventListener("click", () => { G.seed = ""; const f = $("#img-seed"); if (f) { f.value = ""; f.focus(); } });
    ["#img-neg", "#img-seed", "#img-steps", "#img-guidance"].forEach(id => $(id)?.addEventListener("change", keepCreateFields));
    $$("[data-reseed]").forEach(b => b.addEventListener("click", () => {
      const [bi, seed] = b.dataset.reseed.split(":");
      keepCreateFields(); G.prompt = G.batches[+bi].prompt; G.seed = seed; render(); $("#img-prompt").focus();
    }));
    $$("[data-reuse]").forEach(b => b.addEventListener("click", () => { G.prompt = G.batches[+b.dataset.reuse].prompt; render(); $("#img-prompt").focus(); }));
    $$("[data-go]").forEach(b => b.addEventListener("click", () => go(b.dataset.go)));
  }

  function hue(name) { let h = 0; for (const c of name) h = (h * 31 + c.charCodeAt(0)) % 360; return `oklch(0.78 0.11 ${h})`; }

  const trModels = () => mediaModels("transcribe").filter(m => m.provider === "local");
  const trPicked = () => G.trModel === "auto" ? trModels()[0] : trModels().find(m => m.key === G.trModel);
  const clock = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  const modeSwitch = () => `<div class="seg vo-mode" role="group" aria-label="What to do"><button data-vomode="speak" aria-pressed="${G.mode === "speak"}">Speak</button><button data-vomode="listen" aria-pressed="${G.mode === "listen"}">Transcribe</button></div>`;

  // what a transcribing model knows, asked for once (this loads it)
  function trOptions(m) {
    if (!m) return null;
    const o = G.trOpts[m.key];
    if (!o) {
      G.trOpts[m.key] = { loading: true };
      postJson("/api/transcribe/options", { model: m.key }).then(j => { G.trOpts[m.key] = j; if (state.view === "voice" && G.mode === "listen") render(); })
        .catch(() => { G.trOpts[m.key] = { languages: [], translate: false }; });
      return null;
    }
    return o.loading ? null : o;
  }

  function transcriptHtml(t, i) {
    const segs = t.segments || [];
    return `<div class="take transcript">
      <div class="who"><span class="chip c-speech">${esc(t.name)}</span><span class="mono muted">${esc(t.model || "")}${t.audioSeconds ? ` · ${t.audioSeconds < 60 ? t.audioSeconds.toFixed(1) + " s" : clock(t.audioSeconds)} of audio` : ""}${t.language ? ` · ${esc(t.language)}` : ""}${t.seconds ? ` · ${t.seconds.toFixed(1)} s` : ""}</span><span class="grow"></span>
        ${t.running ? `<button class="btn ghost sm" id="tr-stop">${icon("stop")}Stop</button>` : t.text ? `<button class="btn ghost sm" data-tr-copy="${i}">${icon("copy")}Copy</button><button class="btn ghost sm" data-tr-chat="${i}">${icon("chat")}To chat</button>` : ""}</div>
      ${t.error ? `<div class="flag bad">${icon("flask")}<div><b>${t.error === "stopped" ? "Stopped." : "Nothing written down."}</b> ${t.error === "stopped" ? "" : esc(t.error)}</div></div>` : ""}
      ${segs.length ? `<div class="segs">${segs.map(g => `<div class="seg-line">${G.trTimes || t.times ? `<span class="mono faint">${clock(g.start)}</span>` : ""}<span>${esc(g.text.trim())}</span></div>`).join("")}</div>`
        : t.running ? `<div class="muted" style="font-size:13px"><span class="typing"><i></i><i></i><i></i></span> ${t.stage || "listening…"}</div>` : ""}
    </div>`;
  }

  function listenView() {
    const m = trPicked(), o = trOptions(m) || {};
    const rec = G.recorder;
    return `
    <div class="voice listen">
      <section style="display:flex;flex-direction:column;min-height:0;min-width:0">
        <div class="stage-head"><span class="title">Voice</span>${modeSwitch()}${m ? `<span class="chip c-speech">${esc(m.name)}</span>` : ""}<span class="grow"></span>
          ${rec ? `<button class="btn primary" id="tr-rec-stop">${icon("stop")}Stop · <span class="mono" id="tr-rec-time">${clock((Date.now() - rec.at) / 1000)}</span></button>`
            : `<button class="btn" id="tr-rec" ${!m || G.listening ? "disabled" : ""}>${icon("speech")}Record</button>`}
          <label class="btn primary ${!m || G.listening ? "disabled" : ""}" for="tr-file">${icon("plus")}Choose a recording</label><input type="file" id="tr-file" accept="audio/*,video/*" hidden ${!m || G.listening ? "disabled" : ""}></div>
        ${!m ? `<div class="empty"><div class="flag">${icon("flask")}<div><b>No transcribing model yet.</b> Put a Whisper model (such as openai/whisper-base, a Hugging Face folder) in a model folder to write down what is said, on this PC.
          <div style="margin-top:8px"><button class="btn" data-go="library">${icon("library")}Open Models and providers</button></div></div></div></div>` : `
        <div class="script tr-list" id="tr-drop">
          ${!G.transcripts.length ? `<label class="video-drop" for="tr-file">${icon("speech")}<b>Drop a recording here, or record one</b><span class="muted">Any audio or video the browser plays. It is written down on this PC and never uploaded anywhere.</span></label>` : ""}
          ${G.transcripts.map(transcriptHtml).join("")}
        </div>`}
      </section>
      <aside class="voice-settings" aria-label="Transcription settings">
        <div class="field"><label for="tr-model">Model</label><select class="input" id="tr-model" aria-label="Model">
          <option value="auto" ${G.trModel === "auto" ? "selected" : ""}>Auto${trModels()[0] ? " · " + esc(trModels()[0].name) : ""}</option>
          ${trModels().map(x => `<option value="${esc(x.key)}" ${x.key === G.trModel ? "selected" : ""}>${esc(x.name)}</option>`).join("")}</select></div>
        <div class="field"><label for="tr-lang">Spoken language</label><select class="input" id="tr-lang">
          <option value="auto">Detect it</option>${(o.languages || []).map(l => `<option value="${esc(l)}" ${l === G.trLang ? "selected" : ""}>${esc(langName(l))}</option>`).join("")}</select></div>
        <label class="check"><input type="checkbox" id="tr-translate" ${G.trTranslate ? "checked" : ""} ${o.translate ? "" : "disabled"}> Write it in English${o.translate ? "" : ` <output>not offered</output>`}</label>
        <label class="check"><input type="checkbox" id="tr-times" ${G.trTimes ? "checked" : ""}> Show times</label>
        <p class="muted" style="font-size:12px;margin:0">The recording is turned into 16 kHz mono in this browser, then written down by ${m ? esc(m.name) : "the model"} on this PC.</p>
      </aside>
    </div>`;
  }
  const langName = code => { try { return new Intl.DisplayNames([navigator.language || "en"], { type: "language" }).of(code) || code; } catch (_) { return code; } };

  // any audio the browser can play, as 16 kHz mono 16-bit WAV, base64
  const toWav16k = blob => toWav(blob, 16000);

  async function toWav(blob, rate) {
    const buf = await blob.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    let decoded;
    try { decoded = await ctx.decodeAudioData(buf); } finally { ctx.close?.(); }
    const n = Math.max(1, Math.ceil(decoded.duration * rate));
    const off = new OfflineAudioContext(1, n, rate);
    const src = off.createBufferSource(); src.buffer = decoded; src.connect(off.destination); src.start();
    const pcm = (await off.startRendering()).getChannelData(0);
    const out = new DataView(new ArrayBuffer(44 + pcm.length * 2));
    const str = (o, t) => { for (let i = 0; i < t.length; i++) out.setUint8(o + i, t.charCodeAt(i)); };
    str(0, "RIFF"); out.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVE"); str(12, "fmt ");
    out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true); out.setUint32(24, rate, true);
    out.setUint32(28, rate * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true); str(36, "data"); out.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i++) out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 0x7fff, true);
    const bytes = new Uint8Array(out.buffer);
    let bin = "";
    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return { data: btoa(bin), seconds: decoded.duration };
  }

  async function transcribe(blob, name) {
    const m = trPicked();
    if (!m || G.listening) return;
    const t = { name, model: m.name, segments: [], running: true, stage: "reading the recording…", times: G.trTimes };
    G.transcripts.unshift(t);
    const ctrl = new AbortController();
    G.listening = ctrl;
    render();
    const paint = () => { const box = $(".tr-list"); if (box && state.view === "voice" && G.mode === "listen") { box.innerHTML = G.transcripts.map(transcriptHtml).join(""); bindTranscripts(); } };
    try {
      const wav = await toWav16k(blob);
      t.audioSeconds = wav.seconds; t.stage = "writing it down…"; paint();
      const body = { model: m.key, audio: wav.data, language: G.trLang, translate: G.trTranslate, timestamps: true };
      const r = await fetch("/api/transcribe", { method: "POST", signal: ctrl.signal,
        headers: { "X-Studio-Token": tokenMeta.content, "Content-Type": "application/json" }, body: JSON.stringify(body) });
      await readEvents(r, (event, d) => {
        if (event === "route") t.model = d.name;
        else if (event === "segment") t.segments.push(d);
        else if (event === "done") Object.assign(t, { text: d.text, language: d.language, seconds: d.seconds });
        else if (event === "error") t.error = d.message;
        paint();
      });
      if (!t.text && t.segments.length) t.text = t.segments.map(g => g.text.trim()).join(" ");
    } catch (e) {
      t.error = e.name === "AbortError" ? "stopped" : e.name === "EncodingError" ? "the browser cannot read that recording" : e.message;
    }
    t.running = false; G.listening = null;
    if (state.view === "voice") render();
  }

  async function record() {
    if (G.recorder || G.listening) return;
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (e) { G.transcripts.unshift({ name: "Recording", error: "the microphone is not available: " + e.message }); render(); return; }
    const chunks = [], rec = new MediaRecorder(stream);
    rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = () => {
      stream.getTracks().forEach(tr => tr.stop());
      clearInterval(G.recorder?.tick); G.recorder = null;
      transcribe(new Blob(chunks, { type: rec.mimeType || "audio/webm" }), "Recording " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    };
    G.recorder = { rec, at: Date.now(), tick: setInterval(() => { const el = $("#tr-rec-time"); if (el && G.recorder) el.textContent = clock((Date.now() - G.recorder.at) / 1000); }, 500) };
    rec.start(1000);
    render();
  }

  function bindTranscripts() {
    $("#tr-stop")?.addEventListener("click", () => G.listening?.abort());
    $$("[data-tr-copy]").forEach(b => b.addEventListener("click", () => navigator.clipboard?.writeText(G.transcripts[+b.dataset.trCopy].text || "")));
    $$("[data-tr-chat]").forEach(b => b.addEventListener("click", () => {
      const text = G.transcripts[+b.dataset.trChat].text || "";
      go("chat");
      setTimeout(() => { const p = $('[data-prompt="0"]'); if (p) { p.value = (p.value ? p.value + "\n" : "") + text; p.focus(); p.dispatchEvent(new Event("input")); } }, 50);
    }));
  }

  function bindListen() {
    $$("[data-vomode]").forEach(b => b.addEventListener("click", () => { G.mode = b.dataset.vomode; store.set("voMode", G.mode); render(); }));
    $("#tr-model")?.addEventListener("change", e => { G.trModel = e.target.value; store.set("trModel", G.trModel); render(); });
    $("#tr-lang")?.addEventListener("change", e => { G.trLang = e.target.value; store.set("trLang", G.trLang); });
    $("#tr-translate")?.addEventListener("change", e => { G.trTranslate = e.target.checked; });
    $("#tr-times")?.addEventListener("change", e => { G.trTimes = e.target.checked; G.transcripts.forEach(t => { t.times = G.trTimes; }); render(); });
    $("#tr-file")?.addEventListener("change", e => { const f = e.target.files[0]; if (f) transcribe(f, f.name); e.target.value = ""; });
    $("#tr-rec")?.addEventListener("click", record);
    $("#tr-rec-stop")?.addEventListener("click", () => G.recorder?.rec.stop());
    const drop = $("#tr-drop");
    drop?.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
    drop?.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop?.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) transcribe(f, f.name); });
    bindTranscripts();
    $$("[data-go]").forEach(b => b.addEventListener("click", () => go(b.dataset.go)));
  }

  function liveVoiceView() {
    if (G.mode === "listen") return listenView();
    const m = pickedModel(G.voModel, "speech");
    const local = m?.provider === "local";
    const opt = m?.speech || { voices: [], speed: false, instructions: false, formats: [] };
    if (!opt.voices.includes(G.voice)) G.voice = opt.voices[0] || null;
    if (!opt.mix || !opt.voices.includes(G.blend) || G.blend === G.voice) G.blend = "";
    if (opt.formats.length && !opt.formats.includes(G.format)) G.format = opt.formats[0];
    const names = opt.names || {};
    return `
    <div class="voice">
      <aside class="voices" aria-label="Voices">
        <div class="col-head"><span class="h3">Voices</span><span class="grow"></span><span class="mono faint">${opt.voices.length || ""}</span></div>
        ${!opt.voices.length && opt.clone ? `<div class="faint pad" style="font-size:13px">No voices yet. This model speaks in a voice made from a recording: add one under <b>New voice</b>.</div>` : ""}
        ${opt.voices.map(v => { const [n, d] = voiceName(v, names); return `
          <div class="vcard ${v === G.voice ? "active" : ""}" data-voice="${esc(v)}" tabindex="0" role="button" aria-pressed="${v === G.voice}">
            <div class="vface" style="background:${hue(v)}">${esc(n[0].toUpperCase())}</div>
            <div style="min-width:0"><div class="n">${esc(n)}${v === G.blend ? ` <span class="mono faint">blended in</span>` : ""}</div><div class="d">${esc(d ? d + " · " + providerName(m.provider) : providerName(m.provider))}</div></div>
            <button class="iconbtn" data-preview="${esc(v)}" aria-label="Hear ${esc(n)}" title="${local ? "Speaks one short sentence" : "Speaks one short sentence, billed to your key"}">${icon("play")}</button>
          </div>`; }).join("")}
      </aside>
      <section style="display:flex;flex-direction:column;min-height:0;min-width:0">
        <div class="stage-head"><span class="title">Voice</span>${modeSwitch()}${m ? `<span class="chip c-speech">${esc(m.name)}</span>` : ""}<span class="grow"></span>
          <label class="btn" for="vo-video-file">${icon("video")}${G.video ? "Change video" : "Add a video"}</label><input type="file" id="vo-video-file" accept="video/*" hidden>
          <button class="btn primary" id="vo-go" ${!m || G.speaking ? "disabled" : ""}>${icon("speech")}${G.speaking ? "Speaking…" : "Speak"}</button></div>
        ${m ? `<div style="padding:14px 24px 0">${local ? `<div class="flag ok-note">${icon("lock")}<div><b>Speaks on this PC.</b> The text is not sent anywhere.</div></div>` : remoteNote(m, "The text")}</div>` : ""}
        ${!mediaModels("speech").length ? noModel("speech") : `
        <div class="vo-split">
          <div class="script">
            <textarea class="input vo-text" id="vo-text" placeholder="Write what the voice should say. Ctrl+Enter speaks it.">${esc(G.text)}</textarea>
            ${G.takes.map((t, i) => `<div class="take">
              <div class="who"><span class="chip c-speech">${esc(t.voice)}</span><span class="mono muted">${esc(t.model)}${t.seconds ? ` · ${t.seconds.toFixed(1)} s` : ""}</span><span class="grow"></span>
                ${t.url ? `${G.video ? `<button class="btn ghost sm" data-with-video="${i}">${icon("play")}With video</button>` : ""}<a class="btn ghost sm" href="${t.url}" download="${esc(fileName(t.file))}" title="Saved as ${esc(t.file)}">Save</a>` : ""}</div>
              <div class="muted" style="font-size:13px">${esc(t.text.length > 160 ? t.text.slice(0, 160) + "…" : t.text)}</div>
              ${t.error ? `<div class="flag bad">${icon("flask")}<div><b>No audio.</b> ${esc(t.error)}</div></div>` : `<audio controls src="${t.url}" data-take="${i}"></audio>`}
            </div>`).join("")}
          </div>
          <div class="vo-video">
            ${G.video ? `<video id="vo-video" src="${G.video.url}" controls playsinline></video><div class="mono muted" style="font-size:11.5px">${esc(G.video.name)} · plays here only, never uploaded</div>`
              : `<label class="video-drop" for="vo-video-file">${icon("video")}<b>Voiceover for a video?</b><span class="muted">Add it here to hear takes over the picture. It stays in this browser.</span></label>`}
          </div>
        </div>`}
      </section>
      <aside class="voice-settings" aria-label="Voice settings">
        <div class="field"><label for="vo-model">Model</label>${mediaPicker("vo-model", "speech", G.voModel)}</div>
        <div class="field" ${opt.speed ? "" : `title="${m ? esc(m.name) + " has no pace setting" : ""}"`}><label for="vo-speed">Pace <output>${opt.speed ? G.speed.toFixed(2) + "×" : "not offered"}</output></label>
          <input type="range" id="vo-speed" min="0.5" max="2" step="0.05" value="${G.speed}" ${opt.speed ? "" : "disabled"}></div>
        ${opt.mix && opt.voices.length > 1 ? `<div class="field"><label for="vo-blend">Blend with</label>
          <select class="input" id="vo-blend"><option value="">Nothing, just ${esc(voiceName(G.voice, names)[0])}</option>${opt.voices.filter(v => v !== G.voice).map(v => `<option value="${esc(v)}" ${v === G.blend ? "selected" : ""}>${esc(voiceName(v, names)[0])}${voiceName(v, names)[1] ? " · " + esc(voiceName(v, names)[1]) : ""}</option>`).join("")}</select></div>
        ${G.blend ? `<div class="field"><label for="vo-mix">Mix <output>${mixLabel(names)}</output></label>
          <input type="range" id="vo-mix" min="0.1" max="0.9" step="0.05" value="${G.blendW}" aria-label="How much of ${esc(voiceName(G.voice, names)[0])}"></div>` : ""}` : ""}
        ${opt.clone ? cloneBox() : ""}
        ${opt.tags?.length ? `<div class="field"><label>Expressions <output>put in the text</output></label><div class="tag-row">${opt.tags.map(t => `<button class="btn ghost sm" data-vo-tag="${esc(t)}">&lt;${esc(t)}&gt;</button>`).join("")}</div></div>` : ""}
        <div class="field"><label for="vo-instr">How to say it ${opt.instructions ? "" : "<output>not offered</output>"}</label>
          <textarea class="input" id="vo-instr" rows="3" placeholder="Calm narrator, a little slower, warm" ${opt.instructions ? "" : "disabled"}>${esc(G.instructions)}</textarea></div>
        ${opt.formats.length > 1 ? `<div class="field"><label>File</label><div class="seg">${opt.formats.map(f => `<button data-format="${f}" aria-pressed="${G.format === f}">${f.toUpperCase()}</button>`).join("")}</div></div>`
          : opt.formats.length ? `<div class="field"><label>File <output>${opt.formats[0].toUpperCase()}</output></label></div>` : ""}
      </aside>
    </div>`;
  }

  // making a voice from a recording: a name, 3 to 12 seconds of speech, and what is said in it
  function cloneBox() {
    const c = G.clone;
    const ready = c.sample && c.name.trim() && !c.busy && !c.rec;
    return `<div class="field clone-box"><label for="vo-clone-name">New voice <output>from a recording</output></label>
      <input class="input" id="vo-clone-name" placeholder="Its name, such as Me" maxlength="40" value="${esc(c.name)}">
      <div class="clone-row">
        <label class="btn sm ${c.rec || c.busy ? "disabled" : ""}" for="vo-clone-file">${icon("clip")}Choose a recording</label><input type="file" id="vo-clone-file" accept="audio/*,video/*" hidden>
        <button class="btn sm" id="vo-clone-rec" type="button" ${c.busy ? "disabled" : ""}>${c.rec ? `${icon("stop")}Stop <span class="mono" id="vo-clone-time">0:00</span>` : `${icon("speech")}Record`}</button>
      </div>
      <div class="mono faint" style="font-size:12px">${c.sample ? `${esc(c.sample.name)} · ${c.sample.seconds.toFixed(1)} s${c.sample.seconds > 12 ? " (only the first 12 s are used)" : ""}` : "3 to 12 seconds of clear speech works best."}</div>
      <textarea class="input" id="vo-clone-text" rows="2" placeholder="What is said in it. Left empty, a Whisper model writes it down.">${esc(c.text)}</textarea>
      <button class="btn primary sm" id="vo-clone-save" type="button" ${ready ? "" : "disabled"}>${c.busy ? "Adding…" : "Add the voice"}</button>
      ${c.error ? `<div class="flag bad">${icon("flask")}<div>${esc(c.error)}</div></div>` : ""}
    </div>`;
  }

  async function cloneSample(blob, name) {
    try {
      const w = await toWav(blob, 24000);
      G.clone.sample = { data: w.data, seconds: w.seconds, name };
      G.clone.error = w.seconds < 1 ? "The recording is too short: use 3 to 12 seconds of speech." : "";
    } catch (e) { G.clone.error = "This file could not be read as audio: " + e.message; }
    if (state.view === "voice") render();
  }

  async function cloneRecord() {
    const c = G.clone;
    if (c.rec) { c.rec.rec.stop(); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (e) { c.error = "The microphone is not available: " + e.message; render(); return; }
    const chunks = [], rec = new MediaRecorder(stream);
    rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      clearInterval(c.rec?.tick); c.rec = null;
      cloneSample(new Blob(chunks, { type: rec.mimeType || "audio/webm" }), "Recording");
    };
    c.rec = { rec, at: Date.now(), tick: setInterval(() => { const el = $("#vo-clone-time"); if (el && c.rec) el.textContent = clock((Date.now() - c.rec.at) / 1000); }, 500) };
    rec.start(1000);
    render();
  }

  async function cloneSave() {
    const c = G.clone, m = pickedModel(G.voModel, "speech");
    if (!m || !c.sample) return;
    c.busy = true; c.error = ""; render();
    try {
      const j = await postJson("/api/speech/voice", { model: m.key, name: c.name.trim(), audio: c.sample.data, transcript: c.text.trim() });
      G.voice = j.voice; store.set("voice", G.voice);
      Object.assign(c, { name: "", text: "", sample: null });
      await loadModels();
    } catch (e) { c.error = e.message; }
    c.busy = false;
    if (state.view === "voice") render();
  }

  // a voice's readable name and description, from "Heart (American female)" or the bare id
  function voiceName(v, names) {
    if (!v) return [""];
    const full = names?.[v];
    const hit = full && /^(.*?)\s*\((.*)\)$/.exec(full);
    if (hit) return [hit[1], hit[2]];
    if (full) return [full, ""];
    return [v[0].toUpperCase() + v.slice(1), ""];
  }

  function mixLabel(names) {
    const a = Math.round(G.blendW * 100);
    return `${a}% ${voiceName(G.voice, names)[0]} · ${100 - a}% ${voiceName(G.blend, names)[0]}`;
  }

  // what is sent: the voice, or a blend such as af_heart*0.70+am_adam*0.30
  function spokenVoice() {
    return G.blend ? `${G.voice}*${G.blendW.toFixed(2)}+${G.blend}*${(1 - G.blendW).toFixed(2)}` : G.voice;
  }

  function spokenLabel() {
    const names = pickedModel(G.voModel, "speech")?.speech?.names || {};
    return G.blend ? mixLabel(names) : voiceName(G.voice, names)[0];
  }

  async function speak(text, voice, preview, label) {
    const m = pickedModel(G.voModel, "speech");
    if (!m || !text || !voice) return;
    const take = { text, voice: label || voice, model: m.name };
    if (!preview) { G.speaking = true; render(); }
    try {
      const j = await postJson("/api/speech", { model: G.voModel, text, voice, speed: G.speed, instructions: G.instructions, format: G.format });
      Object.assign(take, { url: blobUrl(j.items[0]), file: j.items[0].file, seconds: j.seconds, model: j.model.name });
    } catch (e) { take.error = e.message; }
    if (preview) {
      if (take.url) new Audio(take.url).play().catch(() => {});
      else { G.takes.unshift(take); render(); }
      return;
    }
    G.takes.unshift(take);
    G.speaking = false;
    if (state.view === "voice") render();
  }

  function bindLiveVoice() {
    if (G.mode === "listen") return bindListen();
    $$("[data-vomode]").forEach(b => b.addEventListener("click", () => { G.text = $("#vo-text")?.value ?? G.text; G.mode = b.dataset.vomode; store.set("voMode", G.mode); render(); }));
    const keep = () => { G.text = $("#vo-text")?.value ?? G.text; G.instructions = $("#vo-instr")?.value ?? G.instructions; };
    $("#vo-model")?.addEventListener("change", e => { keep(); G.voModel = e.target.value; store.set("voModel", G.voModel); render(); });
    $$("[data-voice]").forEach(el => el.addEventListener("click", e => {
      if (e.target.closest("[data-preview]")) return;
      keep(); G.voice = el.dataset.voice; store.set("voice", G.voice); render();
    }));
    $$("[data-preview]").forEach(b => b.addEventListener("click", () => {
      const n = voiceName(b.dataset.preview, pickedModel(G.voModel, "speech")?.speech?.names)[0];
      speak(`Hello, I'm ${n}. This is how I sound.`, b.dataset.preview, true, n);
    }));
    $("#vo-blend")?.addEventListener("change", e => { keep(); G.blend = e.target.value; render(); });
    $("#vo-clone-name")?.addEventListener("input", e => {
      G.clone.name = e.target.value;
      const b = $("#vo-clone-save"); if (b) b.disabled = !(G.clone.sample && G.clone.name.trim() && !G.clone.busy && !G.clone.rec);
    });
    $("#vo-clone-text")?.addEventListener("input", e => { G.clone.text = e.target.value; });
    $("#vo-clone-file")?.addEventListener("change", e => { const f = e.target.files[0]; if (f) { keep(); cloneSample(f, f.name); } });
    $("#vo-clone-rec")?.addEventListener("click", () => { keep(); cloneRecord(); });
    $("#vo-clone-save")?.addEventListener("click", () => { keep(); cloneSave(); });
    $("#vo-mix")?.addEventListener("input", e => {
      G.blendW = +e.target.value;
      e.target.previousElementSibling.querySelector("output").textContent = mixLabel(pickedModel(G.voModel, "speech")?.speech?.names);
    });
    $$("[data-vo-tag]").forEach(b => b.addEventListener("click", () => {
      const t = $("#vo-text"); if (!t) return;
      const at = t.selectionStart ?? t.value.length, tag = `<${b.dataset.voTag}>`;
      t.value = t.value.slice(0, at) + tag + t.value.slice(t.selectionEnd ?? at);
      G.text = t.value; t.focus(); t.selectionStart = t.selectionEnd = at + tag.length;
    }));
    $("#vo-speed")?.addEventListener("input", e => { G.speed = +e.target.value; e.target.previousElementSibling.querySelector("output").textContent = G.speed.toFixed(2) + "×"; });
    $("#vo-instr")?.addEventListener("input", e => { G.instructions = e.target.value; });
    $("#vo-text")?.addEventListener("input", e => { G.text = e.target.value; });
    $("#vo-text")?.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { keep(); speak(G.text.trim(), spokenVoice(), false, spokenLabel()); } });
    $$("[data-format]").forEach(b => b.addEventListener("click", () => { keep(); G.format = b.dataset.format; render(); }));
    $("#vo-go")?.addEventListener("click", () => { keep(); speak(G.text.trim(), spokenVoice(), false, spokenLabel()); });
    $("#vo-video-file")?.addEventListener("change", e => {
      const f = e.target.files[0]; if (!f) return;
      keep();
      if (G.video) URL.revokeObjectURL(G.video.url);
      G.video = { url: URL.createObjectURL(f), name: f.name };
      render();
    });
    $$("[data-with-video]").forEach(b => b.addEventListener("click", () => {
      const v = $("#vo-video"), a = $(`audio[data-take="${b.dataset.withVideo}"]`);
      if (!v || !a) return;
      v.currentTime = 0; a.currentTime = 0; v.muted = true;
      v.play(); a.play();
    }));
    $$("[data-go]").forEach(b => b.addEventListener("click", () => go(b.dataset.go)));
  }

  // ---------- live models and providers ----------
  // Models grouped by provider; each group folds, and remembers whether it was folded.
  const isFolded = (id, n) => { const f = store.get("libFolded", {}); return id in f ? f[id] : n > 25; };
  function setFolded(id, v) { const f = store.get("libFolded", {}); f[id] = v; store.set("libFolded", f); }
  function liveModelsTable() {
    const ms = L.info ? L.info.models : [];
    const groups = new Map();
    ms.forEach(m => { if (!groups.has(m.provider)) groups.set(m.provider, []); groups.get(m.provider).push(m); });
    const row = m => `<tr data-find="${esc((m.name + " " + m.key).toLowerCase())}" class="mrow ${m.shown === false ? "hidden-model" : ""}">
        <td><div class="nm">${esc(m.name)}</div><div class="src">${esc(m.key)}</div></td>
        <td>${m.media.map(x => `<span class="chip ${mediaChip[x] || "plain"}">${mediaName[x] || esc(x)}</span>`).join(" ")}</td>
        <td>${m.shown === false ? `<span class="chip plain" title="Not offered in the model pickers or by Auto. Change it with Edit on the provider.">hidden</span>` : Object.values(L.info.auto || {}).includes(m.key) ? `<span class="chip c-ok">Auto picks it</span>` : ""}</td>
        <td class="mono">${esc(m.detail)}</td></tr>`;
    const body = [...groups].map(([id, list]) => {
      const folded = isFolded(id, list.length), name = id === "local" ? "This PC" : providerName(id);
      const hidden = list.filter(m => m.shown === false).length;
      const p = (L.info.providers || []).find(x => x.id === id);
      return `<tbody class="mgroup ${folded ? "folded" : ""}" data-group="${esc(id)}">
        <tr class="ghead"><td colspan="4"><div class="ghead-in">
          <button class="gtoggle" data-fold="${esc(id)}" aria-expanded="${!folded}">${icon("chev")}<b>${esc(name)}</b>
            <span class="faint tnum">${list.length} model${list.length === 1 ? "" : "s"}${hidden ? ` · ${hidden} hidden` : ""}</span></button>
          <span class="grow"></span>
          ${id === "local" ? `<span class="chip c-ok">free · private</span>` : providerKind(id) === "host" ? `<span class="chip c-ok">your machine</span>` : `<span class="chip c-agent">your key</span>`}
          ${p && p.settings ? `<button class="btn ghost sm" data-edit-provider="${esc(id)}">${icon("edit")}Edit</button>` : ""}
        </div></td></tr>
        ${list.map(row).join("")}</tbody>`;
    }).join("");
    return `${ms.length ? `<div class="modelbar">
        <input class="input" id="model-find" type="search" placeholder="Find a model in every provider" aria-label="Find a model">
        <span class="grow"></span>
        ${groups.size > 1 ? `<button class="btn ghost sm" data-fold-all="false">Expand all</button><button class="btn ghost sm" data-fold-all="true">Collapse all</button>` : ""}
      </div>` : ""}
      <div class="tablebox models-box"><table class="models">
      <thead><tr><th>Model</th><th>Can</th><th>Use</th><th>Detail</th></tr></thead>
      ${body || `<tbody><tr><td colspan="4" class="muted">${L.info ? "No models yet: use Add folder or GGUF for models on this PC, or add an API key below." : "Looking for models…"}</td></tr></tbody>`}</table></div>
      <div class="muted nofind" hidden style="font-size:13px">No model matches.</div>`;
  }

  // ---------- usage and balances ----------
  const compact = n => n >= 1e9 ? (n / 1e9).toFixed(1) + "B" : n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M"
    : n >= 1e3 ? (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k" : String(Math.round(n));
  const plural = (n, one, many) => `${compact(n)} ${n === 1 ? one : many || one + "s"}`;
  const money = b => b.amount === undefined ? "no limit" : `${b.amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${b.currency}`;
  function usageLine(t) {
    if (!t || !t.calls) return "";
    const bits = [plural(t.calls, "call")];
    if (t.in + t.out) bits.push(`${compact(t.in)} in · ${compact(t.out)} out`);
    if (t.failed) bits.push(`${t.failed} failed`);
    return bits.join(" · ");
  }
  function keyBalance(p, label) {
    const list = L.balances && L.balances[p.id];
    const b = list && list.find(x => x.label === label);
    if (!p.balance) return "";
    if (!b) return L.balances ? "" : `<span class="chip plain faint">balance…</span>`;
    if (b.error) return `<span class="chip plain" title="${esc(b.error)}">balance unknown</span>`;
    const low = b.balance.amount !== undefined && b.balance.amount < 1;
    return `<span class="chip ${low ? "c-warn" : "c-ok"}" title="${esc(b.balance.detail)}">${esc(money(b.balance))} left</span>`;
  }

  function liveUsage() {
    const u = L.usage;
    if (!u) return `<div class="usage"><span class="muted">Reading the usage log…</span></div>`;
    const days = u.daily;
    const sum = (row, f) => Object.values(row.providers).reduce((a, t) => a + f(t), 0);
    const tot = { calls: 0, in: 0, out: 0, failed: 0, images: 0, chars: 0 };
    days.forEach(d => Object.values(d.providers).forEach(t => { tot.calls += t.calls; tot.in += t.in; tot.out += t.out; tot.failed += t.failed; }));
    const kinds = { image: 0, speech: 0 };
    // pictures and characters come from the per-model totals, split by the model's medium
    Object.entries(u.models).forEach(([k, t]) => {
      const m = modelByKey(k);
      if (m && m.media.includes("image")) kinds.image += t.units;
      else if (m && m.media.includes("speech")) kinds.speech += t.units;
    });
    const tokens = days.map(d => sum(d, t => t.in + t.out));
    const max = Math.max(1, ...tokens);
    const bars = days.map((d, i) => {
      const v = tokens[i];
      const label = new Date(d.day + "T12:00").toLocaleDateString(undefined, { month: "short", day: "numeric" });
      const per = Object.entries(d.providers).map(([p, t]) => `${providerName(p)}: ${t.in + t.out ? compact(t.in + t.out) + " tokens, " : ""}${plural(t.calls, "call")}${t.failed ? `, ${t.failed} failed` : ""}`).join("; ");
      return `<button class="bar" data-tip="${esc(label)}|${esc(per || "no calls")}" aria-label="${esc(label)}: ${esc(per || "no calls")}">${v
        ? `<i style="height:${Math.max(2, v / max * 100).toFixed(1)}%"></i>` : `<i class="zero"></i>`}</button>`;
    }).join("");
    const first = new Date(days[0].day + "T12:00"), last = new Date(days[days.length - 1].day + "T12:00");
    const fmtDay = d => d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
    const provRows = Object.entries(u.providers).sort((a, b) => (b[1].in + b[1].out) - (a[1].in + a[1].out) || b[1].calls - a[1].calls);
    const tile = (v, l) => `<div class="tile"><b class="tnum">${v}</b><span>${l}</span></div>`;
    return `<div class="usage">
      <div class="usage-head"><span class="h3">Usage</span><span class="muted">every call this studio made, kept on this PC; tokens as the providers count them</span></div>
      ${provRows.length ? `<div class="tiles">
        ${tile(compact(tot.calls), "calls, last 30 days")}${tile(compact(tot.in), "tokens in")}${tile(compact(tot.out), "tokens out")}
        ${kinds.image ? tile(compact(kinds.image), "pictures, all time") : ""}${kinds.speech ? tile(compact(kinds.speech), "characters spoken, all time") : ""}
        ${tot.failed ? tile(compact(tot.failed), "failed calls") : ""}</div>
      <figure class="daychart">
        <figcaption>Tokens per day, last 30 days <span class="faint">· most ${compact(max)}</span></figcaption>
        <div class="chartwrap"><div class="bars" role="group" aria-label="Tokens per day for the last 30 days">${bars}</div><div class="charttip" hidden></div></div>
        <div class="axis"><span>${esc(fmtDay(first))}</span><span>today</span></div>
      </figure>
      <div class="tablebox"><table class="usage-table">
        <thead><tr><th>Provider and model</th><th class="r">Calls</th><th class="r">Failed</th><th class="r">Tokens in</th><th class="r">Tokens out</th><th class="r">Pictures or characters</th><th class="r">Avg time</th></tr></thead>
        <tbody>${provRows.map(([p, t]) => {
          const ms = Object.entries(u.models).filter(([k]) => k.startsWith(p + "/")).sort((a, b) => b[1].calls - a[1].calls);
          const row = (name, t, cls) => `<tr class="${cls}"><td>${name}</td><td class="r tnum">${compact(t.calls)}</td><td class="r tnum ${t.failed ? "bad" : "faint"}">${t.failed}</td>
            <td class="r tnum">${compact(t.in)}</td><td class="r tnum">${compact(t.out)}</td><td class="r tnum">${t.units ? compact(t.units) : "—"}</td><td class="r tnum">${t.calls ? (t.seconds / t.calls).toFixed(1) + " s" : "—"}</td></tr>`;
          return row(`<b>${esc(p === "local" ? "This PC" : providerName(p))}</b>`, t, "p-total") + ms.map(([k, mt]) => row(`<span class="mono">${esc(k.slice(p.length + 1))}</span>`, mt, "p-model")).join("");
        }).join("")}</tbody></table></div>`
      : `<div class="muted" style="font-size:13px">No calls yet. Chats, pictures and speech show up here with their tokens, per provider, key and model.</div>`}
    </div>`;
  }

  function liveProviders() {
    if (!L.info) return `<div class="providers"><div class="prov"><span class="muted">Asking the providers…</span></div></div>`;
    const apis = L.info.providers.filter(p => p.kind === "api");
    const ready = apis.filter(p => p.keys.length);
    const more = apis.filter(p => !p.keys.length);
    const local = L.info.providers.filter(p => p.local);
    const hosts = L.info.providers.filter(p => p.kind === "host");
    const keyUse = (p, label) => usageLine(L.usage && L.usage.keys[p.id + "/" + label]);
    const form = (p, open) => `<form class="keyform" data-provider="${p.id}" ${open ? "" : "hidden"}>
          <input class="input" name="label" placeholder="Name" value="${p.keys.some(k => k.source === "saved") ? "" : "default"}" aria-label="Key name, such as work" required maxlength="40">
          <input class="input mono" name="key" type="password" placeholder="API key" aria-label="${esc(p.name)} API key" required autocomplete="off">
          <button class="btn" type="submit">${icon("plus")}Add</button>
        </form>`;
    const card = p => {
      const head = `<div style="display:flex;gap:10px;align-items:center"><span class="plogo">${esc(p.name[0])}</span><b>${esc(p.name)}</b><span class="grow"></span>
        ${p.ready && !(p.kind === "host" && p.error) ? `<button class="chip c-ok linkchip" data-show-models="${p.id}" title="Show ${esc(p.name)} models in the table">${p.models} model${p.models === 1 ? "" : "s"}</button>` : `<span class="chip plain">${p.local ? "no models" : p.kind === "host" ? "offline" : "no key"}</span>`}</div>`;
      if (p.kind === "host") {
        return `<div class="prov ${p.error ? "off" : "on"}">${head}
          <div class="muted" style="font-size:12.5px">A machine of yours running <code>brainiac-studio --worker</code>. Its models answer with its own engine and hardware.</div>
          ${p.error ? `<div class="keyerr">${esc(p.error)}</div>` : ""}
          <div class="provset"><span class="mono faint" style="font-size:11px">free · your machine${L.usage && L.usage.providers[p.id] ? " · " + esc(usageLine(L.usage.providers[p.id])) : ""}</span><span class="grow"></span>
            <button class="btn sm" data-go="train">${icon("train")}Hosts on Training</button></div></div>`;
      }
      if (p.local) {
        return `<div class="prov ${p.ready ? "on" : "off"}">${head}
          <div class="muted" style="font-size:12.5px">Hugging Face model folders (${esc((p.families || ["llama", "mistral", "qwen2"]).join(", "))}) and <code>.gguf</code> files in:</div>
          ${p.folders.map(f => `<div class="mono faint" style="font-size:11.5px;word-break:break-all">${esc(f)}</div>`).join("")}
          ${(p.added || []).length ? `<div class="muted" style="font-size:12.5px">and the ones you added:</div>
            ${p.added.map(f => `<div class="placerow"><span class="mono faint">${esc(f)}</span>
              <button class="iconbtn sm" data-place-remove="${esc(f)}" aria-label="Forget ${esc(f)}" title="${P.removing === f ? "Click again to forget it (the files stay)" : "Forget this place (the files stay)"}">${icon(P.removing === f ? "check" : "trash")}</button></div>`).join("")}` : ""}
          <span class="mono faint" style="font-size:11px">free · private · works offline${L.usage && L.usage.providers.local ? " · " + esc(usageLine(L.usage.providers.local)) : ""}</span></div>`;
      }
      const total = L.usage && L.usage.providers[p.id];
      const st = p.settings || {};
      const flags = [st.auto === false ? "not used by Auto" : "", st.models ? `${st.models.length} of ${p.models} models shown` : "",
        st.url ? "own address" : "", st.options && st.options.billTo ? `bills ${st.options.billTo}` : ""].filter(Boolean);
      return `<div class="prov ${p.ready ? "on" : "off"}">${head}
        ${p.note ? `<div class="muted" style="font-size:12.5px">${esc(p.note)}</div>` : ""}
        <div class="provset">${flags.map(f => `<span class="chip plain">${esc(f)}</span>`).join("")}<span class="grow"></span>
          <button class="btn sm" data-edit-provider="${p.id}">${icon("edit")}Edit</button></div>
        ${p.error ? `<div class="keyerr">${esc(p.error)}</div>` : ""}
        <div class="keys">${p.keys.map((k, n) => `<div class="keyrow">
            <span class="dot" style="--c:var(--${k.error ? "bad" : "ok"})"></span>
            <span class="lbl">${esc(k.label)}</span><span class="mono faint">${esc(k.hint)}</span>
            ${n === 0 && p.keys.length > 1 ? `<span class="chip plain">tried first</span>` : ""}
            ${keyBalance(p, k.label)}
            <span class="grow"></span>
            ${k.source === "saved" ? `${n > 0 ? `<button class="btn ghost sm" data-key-first="${p.id}" data-label="${esc(k.label)}">Try first</button>` : ""}<button class="btn ghost sm" data-key-remove="${p.id}" data-label="${esc(k.label)}" aria-label="Remove key ${esc(k.label)}">Remove</button>` : `<span class="chip plain" title="Set in the environment the studio started from">environment</span>`}
            ${keyUse(p, k.label) ? `<div class="keyuse mono">${esc(keyUse(p, k.label))}</div>` : ""}
            ${k.error ? `<div class="keyerr">${esc(k.error)}</div>` : ""}</div>`).join("")}</div>
        ${form(p, true)}
        <span class="mono faint" style="font-size:11px">${total && p.keys.length > 1 ? esc(usageLine(total)) + " in all · " : ""}several keys: tried in order, the next one takes over when a key is refused or out of credit</span></div>`;
    };
    return `<div class="sect-head"><span class="h2">Compute</span><span class="muted">Machines that run models: this PC, your other machines running as workers, and GPUs you rent.</span></div>
      <div class="providers">${local.map(card).join("")}${hosts.map(card).join("")}${hosts.length ? "" : `<div class="prov off hint-card">
          <b>Another machine of yours</b>
          <div class="muted" style="font-size:12.5px">Run <code>brainiac-studio --worker</code> on it, then add it on Training. Its models show up here and answer in Chat; it can also train.</div>
          <div class="provset"><span class="grow"></span><button class="btn sm" data-go="train">${icon("plus")}Add a host</button></div></div>`}${vastCompute()}</div>
      <div class="sect-head"><span class="h2">Model APIs</span><span class="muted">Hosted models you reach with an API key. Auto tries them after your own machines.</span></div>
      ${ready.length ? `<div class="providers">${ready.map(card).join("")}</div>` : ""}
      ${more.length ? `<div class="more-head"><span class="h3">Add a provider</span><span class="muted">${more.length} more take an API key. Each also reads its key from the environment variable shown.</span></div>
      <div class="more">${more.map(p => `<div class="prov-row ${L.adding === p.id ? "open" : ""}">
          <span class="plogo">${esc(p.name[0])}</span>
          <div class="who"><b>${esc(p.name)}</b><span class="muted">${esc(p.note || "")}</span></div>
          <span class="mono faint env">${esc((p.env || []).join(" or "))}</span>
          <button class="btn sm" data-add-key="${p.id}" aria-expanded="${L.adding === p.id}">${icon("plus")}Add key</button>
          ${form(p, L.adding === p.id)}
        </div>`).join("")}</div>` : ""}
      <div class="prov routing">
        <b>Routing</b>
        <span class="muted" style="font-size:12.5px">Auto uses a model on this PC when there is one, then your worker hosts, then the providers with keys, in the order shown here.</span>
        <button class="btn" id="refresh-models">${icon("retry")}List models again</button>
        ${L.info.providers.some(p => p.balance && p.keys.length) ? `<button class="btn" id="refresh-balances">${icon("retry")}Check balances</button>` : ""}
        <span class="muted" style="font-size:12px">Keys stay on this PC, readable only by your user account. The page only ever sees their last four characters. Anthropic, OpenAI and Google report no balance to an API key; DeepSeek, Kimi and OpenRouter do.</span>
      </div>
      ${liveUsage()}
      ${L.editing ? providerEditor() : ""}`;
  }

  // ---------- provider editor ----------
  const optionInfo = {
    billTo: ["Organization to bill", "my-org", "Hugging Face charges this organization's credits instead of yours (sent as X-HF-Bill-To). Leave empty to pay yourself."],
  };
  function startEdit(id) {
    const p = L.info.providers.find(x => x.id === id);
    const st = p.settings || {};
    L.editing = id;
    L.draft = { auto: st.auto !== false, pick: !!st.models, picked: new Set(st.models || []), url: st.url || "", options: { ...(st.options || {}) }, error: "" };
    render();
    $("#ped")?.focus();
  }
  function providerEditor() {
    const p = L.info.providers.find(x => x.id === L.editing);
    if (!p) return "";
    const d = L.draft, st = p.settings || {};
    const ms = L.info.models.filter(m => m.provider === p.id);
    // ids saved earlier that the provider no longer lists stay visible, so they can be removed
    const gone = [...d.picked].filter(id => !ms.some(m => m.id === id));
    const row = (id, name, detail, media) => `<label class="pick" data-find="${esc((name + " " + id).toLowerCase())}">
        <input type="checkbox" data-pick="${esc(id)}" ${d.picked.has(id) ? "checked" : ""} ${d.pick ? "" : "disabled"}>
        <span class="who"><b>${esc(name)}</b>${name === id ? "" : `<span class="mono faint">${esc(id)}</span>`}</span>
        <span class="caps">${(media || []).map(x => `<span class="chip ${mediaChip[x] || "plain"}">${mediaName[x] || esc(x)}</span>`).join("")}</span>
        <span class="mono faint det">${esc(detail || "")}</span></label>`;
    return `<div class="ped-back" id="ped-back"><aside class="ped" id="ped" role="dialog" aria-modal="true" aria-label="Edit ${esc(p.name)}" tabindex="-1">
      <header><span class="plogo">${esc(p.name[0])}</span><div><b>${esc(p.name)}</b><div class="muted">${esc(p.note || "")}</div></div>
        <span class="grow"></span><button class="iconbtn" id="ped-close" aria-label="Close"><span aria-hidden="true" style="font-size:18px;line-height:1">×</span></button></header>
      <div class="ped-body">
        <label class="ped-row"><span><b>Use in Auto</b><span class="muted">Auto may pick this provider's models. Off, they can still be chosen by name.</span></span>
          <span class="switch"><input type="checkbox" id="ped-auto" ${d.auto ? "checked" : ""} aria-label="Use ${esc(p.name)} in Auto"><i></i></span></label>
        <div class="ped-field"><label for="ped-url"><b>Address</b></label>
          <input class="input mono" id="ped-url" value="${esc(d.url)}" placeholder="${esc(st.home || "")}" spellcheck="false" autocomplete="off">
          <span class="muted">Empty uses ${esc(st.home || "the provider's own")}. Set a gateway, a proxy or another region${p.id === "qwen" ? ", such as https://dashscope.aliyuncs.com/compatible-mode/v1/ for mainland China" : ""}.</span></div>
        ${Object.keys(st.options || {}).map(o => `<div class="ped-field"><label for="ped-opt-${o}"><b>${esc(optionInfo[o] ? optionInfo[o][0] : o)}</b></label>
          <input class="input" id="ped-opt-${o}" data-option="${o}" value="${esc(d.options[o] || "")}" placeholder="${esc(optionInfo[o] ? optionInfo[o][1] : "")}" spellcheck="false" autocomplete="off">
          <span class="muted">${esc(optionInfo[o] ? optionInfo[o][2] : "")}</span></div>`).join("")}
        ${p.id === "huggingface" ? `<div class="flag">${icon("flask")}<div>Hugging Face's plan decides how calls are paid, not which models you may call: Free has no included credits (buy some first), PRO includes $2 a month. Gated models also need their license accepted on the model page. Prices below are each model's cheapest live host.</div></div>` : ""}
        <div class="ped-field grow-field"><b>Models</b>
          <div class="seg" role="group" aria-label="Which models to offer">
            <button data-pick-mode="all" aria-pressed="${!d.pick}">All ${ms.length}</button>
            <button data-pick-mode="pick" aria-pressed="${d.pick}">Only the ones I pick</button></div>
          <span class="muted" id="ped-count">${d.pick ? `${d.picked.size} picked. Only these appear in the model pickers and in Auto.` : "Every model the provider lists appears in the pickers, including new ones."}</span>
          ${ms.length > 8 ? `<div class="ped-tools"><input class="input" id="ped-find" type="search" placeholder="Find a model" aria-label="Find a model">
            <button class="btn sm" id="ped-all" ${d.pick ? "" : "disabled"}>Pick shown</button><button class="btn sm" id="ped-none" ${d.pick ? "" : "disabled"}>Clear</button></div>` : ""}
          <div class="picks ${d.pick ? "" : "off"}">${ms.map(m => row(m.id, m.name, m.detail, m.media)).join("")}
            ${gone.map(id => row(id, id, "no longer listed", [])).join("")}
            ${ms.length || gone.length ? "" : `<div class="muted" style="padding:10px">${p.ready ? "The provider listed no models. Check its key or address." : "Add a key to list this provider's models."}</div>`}</div>
        </div>
      </div>
      <footer>${d.error ? `<div class="keyerr">${esc(d.error)}</div>` : ""}<span class="grow"></span>
        <button class="btn ghost" id="ped-cancel">Cancel</button><button class="btn primary" id="ped-save">Save</button></footer>
    </aside></div>`;
  }
  function bindProviderEditor() {
    if (!L.editing) return;
    const d = L.draft;
    const close = () => { L.editing = null; L.draft = null; render(); };
    $("#ped-close").addEventListener("click", close);
    $("#ped-cancel").addEventListener("click", close);
    $("#ped-back").addEventListener("click", e => { if (e.target.id === "ped-back") close(); });
    if (!document.activeElement || document.activeElement === document.body) $("#ped").focus();
    $("#ped-auto").addEventListener("change", e => d.auto = e.target.checked);
    $("#ped-url").addEventListener("input", e => d.url = e.target.value);
    $$("[data-option]").forEach(i => i.addEventListener("input", () => d.options[i.dataset.option] = i.value));
    const count = () => { $("#ped-count").textContent = `${d.picked.size} picked. Only these appear in the model pickers and in Auto.`; };
    $$("[data-pick]").forEach(c => c.addEventListener("change", () => { if (c.checked) d.picked.add(c.dataset.pick); else d.picked.delete(c.dataset.pick); count(); }));
    $$("[data-pick-mode]").forEach(b => b.addEventListener("click", () => {
      d.pick = b.dataset.pickMode === "pick";
      if (d.pick && !d.picked.size) {   // start from what Auto would use, so the list is not empty
        const auto = Object.values(L.info.auto).find(k => k.startsWith(L.editing + "/"));
        if (auto) d.picked.add(auto.slice(L.editing.length + 1));
      }
      render();
    }));
    const find = $("#ped-find");
    find?.addEventListener("input", () => {
      const q = find.value.trim().toLowerCase();
      $$(".picks .pick").forEach(l => l.hidden = !!q && !l.dataset.find.includes(q));
    });
    $("#ped-all")?.addEventListener("click", () => { $$(".picks .pick:not([hidden]) input").forEach(c => { c.checked = true; d.picked.add(c.dataset.pick); }); count(); });
    $("#ped-none")?.addEventListener("click", () => { $$(".picks .pick:not([hidden]) input").forEach(c => { c.checked = false; d.picked.delete(c.dataset.pick); }); count(); });
    $("#ped-save").addEventListener("click", async () => {
      if (d.pick && !d.picked.size) { d.error = "Pick at least one model, or choose All."; render(); return; }
      const btn = $("#ped-save"); btn.disabled = true; btn.textContent = "Saving…";
      try {
        const r = await fetch("/api/providers", { method: "POST", headers: { "X-Studio-Token": tokenMeta.content, "Content-Type": "application/json" },
          body: JSON.stringify({ provider: L.editing, auto: d.auto, models: d.pick ? [...d.picked] : null, url: d.url.trim(), options: d.options }) });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(j.error || r.status);
        L.info = j; L.editing = null; L.draft = null;
        if (L.model !== "auto" && !modelByKey(L.model)) L.model = "auto";
        render();
      } catch (e) { d.error = e.message; render(); }
    });
  }

  async function postKeys(body) {
    const r = await fetch("/api/keys", { method: "POST", headers: { "X-Studio-Token": tokenMeta.content, "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || r.status);
    L.info = j;
    render();
  }

  // ---------- adding model files and folders from this PC ----------
  const P = { open: false, path: "", list: null, error: null, busy: false, removing: null, note: null };
  function placePanel() {
    const l = P.list;
    const kindChip = k => k === "gguf" ? `<span class="chip c-text">GGUF</span>` : k === "model" ? `<span class="chip c-text">model folder</span>` : "";
    const inside = l ? l.entries.filter(e => e.kind !== "folder").length : 0;
    const here = l && l.path ? (l.kind === "model" ? "This folder is a model." : `${inside || "No"} model${inside === 1 ? "" : "s"} directly in this folder.`) : "";
    return `<form class="placebox" id="place-form" autocomplete="off">
      <div class="place-head"><div><b>Add models from this PC</b> <span class="muted">A <code>.gguf</code> file, a Hugging Face model folder (Llama, Qwen2, Mistral), or a folder holding them. The files stay where they are.</span></div>
        <button type="button" class="iconbtn sm" id="place-close" aria-label="Close">${icon("x")}</button></div>
      <div class="place-path"><input class="input mono" id="place-path" placeholder="A file or folder, such as D:/Models" value="${esc(P.path)}" spellcheck="false" aria-label="File or folder">
        <button class="btn primary" type="submit" ${P.busy || !P.path.trim() ? "disabled" : ""}>${P.busy ? "Adding…" : "Add"}</button></div>
      ${P.error ? `<div class="flag bad">${icon("alert")}<span>${esc(P.error)}</span></div>` : ""}
      <div class="place-browse" aria-label="Folders on this PC">
        <div class="place-crumb">${l && l.path ? `<button type="button" class="btn ghost sm up" data-browse="${esc(l.parent ?? "")}">${icon("chev")}Up</button><code>${esc(l.path)}</code><span class="grow"></span><span class="faint here">${esc(here)}</span>` : `<span class="faint">Start from</span>`}</div>
        <div class="place-list">${!l ? `<div class="faint">Looking…</div>` : l.entries.length ? l.entries.map(e => `<button type="button" class="place-item ${e.path === P.path ? "on" : ""}" data-${e.kind === "folder" ? "browse" : "pick"}="${esc(e.path)}" title="${esc(e.path)}">
            ${icon(e.kind === "folder" ? "folder" : "library")}<span class="nm">${esc(e.name)}</span>${kindChip(e.kind)}</button>`).join("") : `<div class="faint">No folders or models here.</div>`}</div>
      </div>
    </form>`;
  }
  async function browsePlace(path) {
    try { P.list = await postJson("/api/local/browse", { path: path || "" }); P.error = null; if (path) P.path = P.list.path; }
    catch (e) { P.error = e.message; }
    if (state.view === "library") renderPlace();
  }
  function renderPlace() {
    const box = $("#place-form");
    if (!box) { render(); return; }
    const focus = document.activeElement?.id === "place-path", caret = focus ? $("#place-path").selectionStart : 0;
    box.outerHTML = placePanel();
    bindPlace();
    if (focus) { const i = $("#place-path"); i.focus(); i.setSelectionRange(caret, caret); }
  }
  function bindPlace() {
    $("#place-close")?.addEventListener("click", () => { P.open = false; P.error = null; render(); });
    $("#place-path")?.addEventListener("input", e => { P.path = e.target.value; const b = $("#place-form button[type=submit]"); if (b) b.disabled = P.busy || !P.path.trim(); });
    $$("#place-form [data-browse]").forEach(b => b.addEventListener("click", () => browsePlace(b.dataset.browse)));
    $$("#place-form [data-pick]").forEach(b => b.addEventListener("click", () => { P.path = b.dataset.pick; renderPlace(); }));
    $("#place-form")?.addEventListener("submit", async e => {
      e.preventDefault();
      P.busy = true; P.error = null; renderPlace();
      try {
        const r = await postJson("/api/local", { path: P.path.trim() });
        P.open = false; P.busy = false; P.path = ""; P.list = null;
        P.note = r.models ? `Added ${r.models} model${r.models === 1 ? "" : "s"}; they are under This PC below.` : "Added. Models put there later show up too.";
        setTimeout(() => { P.note = null; if (state.view === "library") render(); }, 8000);
        await loadModels(false);
      } catch (err) { P.busy = false; P.error = err.message; renderPlace(); }
    });
  }
  function newModel() {
    const taken = new Set(B.graphs.map(g => g.name));
    let name = "my-model";
    for (let i = 2; taken.has(name); i++) name = "my-model-" + i;
    B.graph = { name, nodes: [{ id: "in", op: "input", shape: [3, 32, 32], x: 40, y: 120 }] };
    B.sel = null; B.built = null; B.refusal = null; B.check = null;
    saveDraft();
    if (state.view !== "builder") go("builder"); else renderBuilder();
    recheck();
  }

  // ---------- downloading models from Hugging Face ----------
  const H = { open: false, query: "", kind: "text", results: null, hidden: 0, runs: null, searching: false, error: null,
    repo: null, listing: null, loadingRepo: false, repoError: null, downloads: [], folder: "", timer: 0, deleting: null };
  const fmtBytes = n => n >= 1e9 ? (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + " GB" : n >= 1e6 ? Math.round(n / 1e6) + " MB" : n >= 1e3 ? Math.round(n / 1e3) + " kB" : n + " B";
  const fmtWait = s => !isFinite(s) || s <= 0 ? "" : s < 90 ? Math.round(s) + " s left" : s < 5400 ? Math.round(s / 60) + " min left" : (s / 3600).toFixed(1) + " h left";
  const hubActive = d => d.state === "queued" || d.state === "downloading" || d.state === "checking";
  // Where a download goes: this PC, or a worker host, which fetches the model itself so it can run it.
  H.target = "local"; H.remote = {};   // remote: host id -> that host's downloads
  const hubHosts = () => (L.info?.providers || []).filter(p => p.kind === "host" && !p.error).map(p => ({ id: p.id.replace(/^host-/, ""), name: p.name }));
  const hubTarget = () => hubHosts().find(h => h.id === H.target) || null;
  const allDownloads = () => [...H.downloads, ...Object.entries(H.remote).flatMap(([id, ds]) => ds.map(d => ({ ...d, host: id, model: `host-${id}/${d.name}` })))];
  // The token the downloader sends to huggingface.co, shown only by its last four characters.
  function hubToken() {
    const t = H.token;
    const where = !t ? "" : t.source === "saved" ? `saved as key “${esc(t.label)}” under Hugging Face`
      : t.managed ? "in <code>~/.brainiac/hf_token</code>, which the core's downloader reads too"
      : t.source === "environment" ? `from <code>${esc(t.label)}</code> in the environment` : "from <code>huggingface-cli login</code>";
    if (t && !H.tokenEdit) return `<div class="hub-token">${icon("lock")}<span>Token <code>${esc(t.hint)}</code> ${where}.</span><span class="grow"></span>
      <button type="button" class="btn ghost sm" id="hub-token-change">Change</button>${t.managed ? `<button type="button" class="btn ghost sm" id="hub-token-forget">${H.tokenForget ? "Click again to forget" : "Forget"}</button>` : ""}</div>`;
    return `<form class="hub-token edit" id="hub-token-form" autocomplete="off">${icon("lock")}
      <span>${t ? "A new token replaces the one " + where + "." : `Gated models such as Llama or Gemma need a token. <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">Make a read token</a>, accept the model's license on its page, and paste the token here.`}</span>
      <input class="input mono" id="hub-token" type="password" placeholder="hf_…" spellcheck="false" aria-label="Hugging Face token">
      <button class="btn" type="submit">Save</button>${t ? `<button type="button" class="btn ghost" id="hub-token-cancel">Cancel</button>` : ""}
      ${H.tokenError ? `<div class="flag bad">${icon("alert")}<span>${esc(H.tokenError)}</span></div>` : ""}</form>`;
  }

  function hubPanel() {
    const res = H.results;
    const list = H.searching ? `<div class="faint pad">Searching…</div>`
      : !res ? (H.kind === "speech" ? `<div class="faint pad">Whisper models write down what is said (<code>openai/whisper-base</code>), VITS models speak (<code>facebook/mms-tts-eng</code>, one per language). Both work on the Voice screen.</div>`
        : `<div class="faint pad">Search by name, such as <code>qwen2.5 instruct</code>, <code>llama 3.2 1b</code> or <code>tinyllama</code>.</div>`)
      : !res.length ? `<div class="faint pad">Nothing found that the studio runs${H.hidden ? ` (${H.hidden} other model${H.hidden === 1 ? "" : "s"} found)` : ""}.</div>`
      : res.map(r => `<button type="button" class="hub-item ${H.repo === r.id ? "on" : ""}" data-repo="${esc(r.id)}">
          <span class="nm">${esc(r.id)}</span>
          <span class="meta">${r.family ? `<span class="chip c-ok">${esc(r.family)}</span>` : ""}${r.gguf ? `<span class="chip c-text">GGUF</span>` : ""}${r.safetensors ? `<span class="chip c-text">safetensors</span>` : ""}
            ${r.parameters ? `<span class="chip plain">${r.parameters >= 1e9 ? (r.parameters / 1e9).toFixed(r.parameters < 1e10 ? 1 : 0) + "B" : Math.round(r.parameters / 1e6) + "M"} params</span>` : ""}${r.sees ? `<span class="chip plain">looks at pictures</span>` : ""}${r.gated ? `<span class="chip c-warn" title="Accept its license on huggingface.co and add your token above">gated</span>` : ""}
            <span class="faint tnum" title="downloads">${icon("download")}${fmtCount(r.downloads)}</span></span></button>`).join("")
        + (H.hidden ? `<div class="faint pad">${H.hidden} more found that the studio cannot run yet.</div>` : "");
    return `<section class="placebox hubbox" id="hub-box" aria-label="Download from Hugging Face">
      <div class="place-head"><div><b>Download from Hugging Face</b> <span class="muted">Shows only models the studio runs${H.runs ? `: ${esc(H.runs.families.join(", "))} model folders, and ${esc(H.runs.architectures.join(" and "))} GGUF files` : ""}. When the core learns more, more show up. Downloads go to <code>${esc(H.folder || "downloads")}</code>.</span></div>
        <button type="button" class="iconbtn sm" id="hub-close" aria-label="Close">${icon("x")}</button></div>
      <div id="hub-token-box">${hubToken()}</div>
      <form class="place-path" id="hub-search" autocomplete="off">
        <div class="seg" role="group" aria-label="What kind of model"><button type="button" data-hub-kind="text" aria-pressed="${H.kind === "text"}">Chat</button><button type="button" data-hub-kind="speech" aria-pressed="${H.kind === "speech"}">Speech</button></div>
        <input class="input" id="hub-q" type="search" placeholder="${H.kind === "speech" ? "Such as whisper base or mms-tts-deu" : "Find a model on Hugging Face"}" value="${esc(H.query)}" aria-label="Model name">
        <button class="btn primary" type="submit" ${H.searching ? "disabled" : ""}>${icon("search")}Search</button></form>
      ${H.error ? `<div class="flag bad">${icon("alert")}<span>${esc(H.error)}</span></div>` : ""}
      <div class="hub-cols">
        <div class="place-browse"><div class="hub-list">${list}</div></div>
        <div class="place-browse hub-files">${hubFiles()}</div>
      </div>
      ${hubDownloads()}
    </section>`;
  }
  function hubFiles() {
    if (!H.repo) return `<div class="faint pad">Pick a model to see its files and which ones the studio runs.</div>`;
    const head = `<div class="place-crumb"><code>${esc(H.repo)}</code><span class="grow"></span><a class="faint" href="https://huggingface.co/${esc(H.repo)}" target="_blank" rel="noopener">Model page</a></div>`;
    if (H.loadingRepo) return head + `<div class="faint pad">Reading its files…</div>`;
    if (H.repoError) return head + `<div class="flag bad">${icon("alert")}<span>${esc(H.repoError)}</span></div>`;
    const opts = (H.listing?.options || []).slice().sort((a, b) => (b.supported - a.supported) || (a.size - b.size));
    if (!opts.length) return head + `<div class="faint pad">No GGUF file or model folder here.</div>`;
    const hosts = hubHosts();
    const target = hosts.length ? `<label class="hub-target">Download to <select class="input" id="hub-target">
        <option value="local" ${hubTarget() ? "" : "selected"}>This PC</option>
        ${hosts.map(h => `<option value="${esc(h.id)}" ${h.id === H.target ? "selected" : ""}>${esc(h.name)}</option>`).join("")}</select>
        ${hubTarget() ? `<span class="faint">The host fetches it with its own connection and then offers it in Chat. Gated models need <code>HF_TOKEN</code> set on the host.</span>` : ""}</label>` : "";
    const to = hubTarget();
    const list = to ? (H.remote[to.id] || []) : H.downloads;
    const same = (d, o) => d.repo === H.repo && (o.kind === "folder" ? d.kind === "folder" : d.name === o.name);
    const busy = o => list.find(d => same(d, o) && hubActive(d));
    const have = o => to ? list.some(d => same(d, o) && d.state === "done") : o.have;
    return head + target + `<div class="hub-opts">${opts.map(o => `<div class="hub-opt ${o.supported ? "" : "off"}">
        <div class="nm">${esc(o.kind === "folder" ? "Whole model folder" : o.name)}${o.quant ? ` <span class="chip plain">${esc(o.quant)}</span>` : ""}${o.kind === "folder" ? ` <span class="chip plain">safetensors</span>` : ""}${o.sees ? ` <span class="chip plain" title="${o.projector ? "Its image projector (" + esc(o.projector.split("/").pop()) + ", " + fmtBytes(o.projectorSize) + ") is downloaded with it" : "Attach pictures to a message in chat"}">looks at pictures</span>` : ""}</div>
        <span class="mono faint tnum">${o.size ? fmtBytes(o.size + (o.projectorSize || 0)) : ""}</span>
        ${have(o) ? `<span class="chip c-ok">on ${to ? esc(to.name) : "this PC"}</span>` : busy(o) ? `<span class="chip c-text">downloading</span>`
          : o.supported ? `<button class="btn sm" data-download="${esc(o.id)}">${icon("download")}${to ? "Download to " + esc(to.name) : "Download"}</button>` : `<span class="chip c-warn">won’t run</span>`}
        ${o.reason ? `<div class="why">${esc(o.reason)}</div>` : ""}</div>`).join("")}</div>`;
  }
  function hubDownloads() {
    const all = allDownloads();
    if (!all.length) return "";
    const hostName = id => hubHosts().find(h => h.id === id)?.name || "the host";
    return `<div class="hub-downs" aria-label="Downloads">${all.map(d => {
      const where = d.host ? hostName(d.host) : "this PC", hostAttr = d.host ? ` data-host="${esc(d.host)}"` : "";
      const pct = d.total ? Math.min(100, d.done / d.total * 100) : 0;
      const status = d.state === "done" ? "on " + where : d.state === "checking" ? "checking…" : d.state === "queued" ? "starting…"
        : d.state === "downloading" ? `${fmtBytes(d.done)} of ${fmtBytes(d.total)}${d.speed ? ` · ${fmtBytes(d.speed)}/s · ${fmtWait((d.total - d.done) / d.speed)}` : ""}`
        : d.state === "stopped" ? `stopped at ${fmtBytes(d.done)} of ${fmtBytes(d.total)}` : "failed";
      return `<div class="hub-down ${d.state}">
        <div class="nm"><b>${esc(d.name)}</b><span class="faint">${esc(d.repo)}</span>${d.host ? `<span class="chip plain">to ${esc(where)}</span>` : ""}</div>
        <div class="meter wide"><i style="width:${pct.toFixed(1)}%;${d.state === "failed" ? "background:var(--bad)" : d.state === "done" ? "background:var(--ok)" : ""}"></i></div>
        <span class="mono faint tnum st">${esc(status)}</span>
        <span class="acts">${hubActive(d) ? `<button class="btn sm ghost" data-hub-cancel="${esc(d.id)}"${hostAttr}>${icon("stop")}Stop</button>`
          : d.state === "done" ? `${(d.family === "whisper" || d.family === "vits") ? (d.host ? "" : `<button class="btn sm" data-voice-with="${esc(d.model)}" data-voice-mode="${d.family === "whisper" ? "listen" : "speak"}">${icon("speech")}${d.family === "whisper" ? "Transcribe with it" : "Speak with it"}</button>`) : `<button class="btn sm" data-chat-with="${esc(d.model)}">${icon("chat")}Chat with it</button>`}${d.host ? "" : `<button class="iconbtn sm" data-hub-delete="${esc(d.name)}" aria-label="Delete ${esc(d.name)}" title="${H.deleting === d.name ? "Click again to delete the files" : "Delete it from this PC"}">${icon(H.deleting === d.name ? "check" : "trash")}</button>`}`
          : `<button class="btn sm" data-hub-retry="${esc(d.id)}"${hostAttr}>${icon("retry")}${d.state === "stopped" ? "Resume" : "Try again"}</button>`}</span>
        ${d.error ? `<div class="why bad">${esc(d.error)}</div>` : ""}
      </div>`;
    }).join("")}</div>`;
  }
  function renderHub() {
    const box = $("#hub-box");
    if (!box || state.view !== "library") return;
    const focus = document.activeElement?.id === "hub-q", caret = focus ? $("#hub-q").selectionStart : 0;
    const scroll = $(".hub-list")?.scrollTop || 0, fscroll = $(".hub-opts")?.scrollTop || 0;
    box.outerHTML = H.open ? hubPanel() : `<div id="hub-box">${hubDownloads()}</div>`;
    bindHub();
    if ($(".hub-list")) $(".hub-list").scrollTop = scroll;
    if ($(".hub-opts")) $(".hub-opts").scrollTop = fscroll;
    if (focus) { const i = $("#hub-q"); i.focus(); i.setSelectionRange(caret, caret); }
  }
  async function hubSearch() {
    H.searching = true; H.error = null; renderHub();
    try { const r = await postJson("/api/hub/search", { query: H.query, kind: H.kind }); H.results = r.models || []; H.hidden = r.hidden || 0; }
    catch (e) { H.error = e.message; H.results = null; }
    H.searching = false; renderHub();
  }
  async function hubRepo(repo) {
    H.repo = repo; H.listing = null; H.repoError = null; H.loadingRepo = true; renderHub();
    try { const l = await postJson("/api/hub/files", { repo }); if (H.repo === repo) H.listing = l; }
    catch (e) { if (H.repo === repo) H.repoError = e.message; }
    if (H.repo === repo) { H.loadingRepo = false; renderHub(); }
  }
  async function pollHub() {
    clearTimeout(H.timer);
    try {
      const j = await api("/api/hub/downloads");
      const before = new Map(H.downloads.map(d => [d.id, d.state]));
      H.downloads = j.downloads || []; H.folder = j.folder || H.folder; H.runs = j.runs || H.runs;
      if ("token" in j && JSON.stringify(j.token) !== JSON.stringify(H.token)) { H.token = j.token; if (!H.tokenEdit && $("#hub-token-box")) { $("#hub-token-box").innerHTML = hubToken(); bindHubToken(); } }
      // the host that downloads go to, and every host still fetching something
      let hostFinished = false;
      const watch = new Set([...Object.keys(H.remote).filter(id => H.remote[id].some(hubActive)), ...(hubTarget() ? [H.target] : [])]);
      await Promise.all([...watch].map(async id => {
        try {
          const was = new Map((H.remote[id] || []).map(d => [d.id, d.state]));
          H.remote[id] = (await postJson("/api/hosts/downloads", { host: id })).downloads || [];
          if (H.remote[id].some(d => d.state === "done" && was.has(d.id) && was.get(d.id) !== "done")) hostFinished = true;
        } catch (e) { console.warn("cannot read the downloads of host " + id, e); }
      }));
      const finished = H.downloads.some(d => d.state === "done" && before.has(d.id) && before.get(d.id) !== "done");
      if (hostFinished) await loadModels(true);   // the host now serves the model: list it
      else if (finished) {
        if (H.listing) H.listing.options.forEach(o => { if (H.downloads.some(d => d.state === "done" && d.repo === H.repo && (o.kind === "folder" ? d.kind === "folder" : d.name === o.name))) o.have = true; });
        await loadModels(false);
      } else renderHub();
    } catch (e) { console.warn("cannot read the downloads", e); }
    if (allDownloads().some(hubActive)) H.timer = setTimeout(pollHub, 1000);
  }
  function chatWith(key) {
    go("chat"); newChat(false);
    L.panes[L.active].model = key; L.model = key; store.set("chatModel", key); savePanes(); render();
    $(`.pane[data-pane="${L.active}"] textarea`)?.focus();
  }
  function bindHubToken() {
    const redraw = () => { $("#hub-token-box").innerHTML = hubToken(); bindHubToken(); };
    $("#hub-token-change")?.addEventListener("click", () => { H.tokenEdit = true; H.tokenError = null; redraw(); $("#hub-token")?.focus(); });
    $("#hub-token-cancel")?.addEventListener("click", () => { H.tokenEdit = false; H.tokenError = null; redraw(); });
    $("#hub-token-forget")?.addEventListener("click", async () => {
      if (!H.tokenForget) { H.tokenForget = true; redraw(); setTimeout(() => { if (H.tokenForget) { H.tokenForget = false; redraw(); } }, 4000); return; }
      H.tokenForget = false;
      try { H.token = (await postJson("/api/hub/token", { remove: true })).token; } catch (e) { H.tokenError = e.message; }
      redraw(); loadModels(false);
    });
    $("#hub-token-form")?.addEventListener("submit", async e => {
      e.preventDefault();
      const input = $("#hub-token"), token = input.value.trim();
      if (!token) { H.tokenError = "Paste the token first."; redraw(); return; }
      try {
        // a token saved as a key comes first, so a new one replaces that key
        if (H.token?.source === "saved") { await postJson("/api/keys", { provider: "huggingface", label: H.token.label, key: token }); H.token = (await api("/api/hub/downloads")).token; }
        else H.token = (await postJson("/api/hub/token", { token })).token;
        H.tokenEdit = false; H.tokenError = null;
        if (H.repo) hubRepo(H.repo);
      } catch (err) { H.tokenError = err.message; }
      input.value = "";
      redraw(); loadModels(false);
    });
  }

  function bindHub() {
    bindHubToken();
    $("#hub-close")?.addEventListener("click", () => { H.open = false; render(); });
    $("#hub-q")?.addEventListener("input", e => { H.query = e.target.value; });
    $$("[data-hub-kind]").forEach(b => b.addEventListener("click", () => {
      if (H.kind === b.dataset.hubKind) return;
      H.kind = b.dataset.hubKind; H.results = null; H.repo = null; H.listing = null; renderHub(); $("#hub-q")?.focus();
    }));
    $$("[data-voice-with]").forEach(b => b.addEventListener("click", () => {
      G.mode = b.dataset.voiceMode; store.set("voMode", G.mode);
      if (G.mode === "listen") { G.trModel = b.dataset.voiceWith; store.set("trModel", G.trModel); }
      else { G.voModel = b.dataset.voiceWith; store.set("voModel", G.voModel); }
      go("voice");
    }));
    $("#hub-search")?.addEventListener("submit", e => { e.preventDefault(); hubSearch(); });
    $$("[data-repo]").forEach(b => b.addEventListener("click", () => hubRepo(b.dataset.repo)));
    $$("[data-download]").forEach(b => b.addEventListener("click", async () => {
      b.disabled = true;
      try {
        if (hubTarget()) await postJson("/api/hosts/download", { host: H.target, repo: H.repo, option: b.dataset.download });
        else await postJson("/api/hub/download", { repo: H.repo, option: b.dataset.download });
        H.repoError = null;
      }
      catch (e) { H.repoError = e.message; }
      await pollHub(); renderHub();
    }));
    $$("[data-hub-cancel]").forEach(b => b.addEventListener("click", async () => {
      b.disabled = true;
      await (b.dataset.host ? postJson("/api/hosts/cancel", { host: b.dataset.host, id: b.dataset.hubCancel }) : postJson("/api/hub/cancel", { id: b.dataset.hubCancel })).catch(() => {});
      setTimeout(pollHub, 300);
    }));
    $("#hub-target")?.addEventListener("change", e => { H.target = e.target.value; renderHub(); pollHub(); });
    $$("[data-hub-retry]").forEach(b => b.addEventListener("click", async () => {
      const host = b.dataset.host;
      const d = (host ? H.remote[host] || [] : H.downloads).find(x => x.id === b.dataset.hubRetry); if (!d) return;
      b.disabled = true;
      try { await postJson(host ? "/api/hosts/download" : "/api/hub/download", host ? { host, repo: d.repo, option: d.option } : { repo: d.repo, option: d.option }); }
      catch (e) { d.error = e.message; }
      await pollHub(); renderHub();
    }));
    $$("[data-chat-with]").forEach(b => b.addEventListener("click", () => chatWith(b.dataset.chatWith)));
    $$("[data-hub-delete]").forEach(b => b.addEventListener("click", async () => {
      const n = b.dataset.hubDelete;
      if (H.deleting !== n) { H.deleting = n; renderHub(); setTimeout(() => { if (H.deleting === n) { H.deleting = null; renderHub(); } }, 4000); return; }
      H.deleting = null;
      try { await postJson("/api/hub/delete", { name: n }); } catch (e) { console.warn(e); }
      H.downloads = H.downloads.filter(d => d.name !== n || d.state !== "done");
      if (H.listing) H.listing.options.forEach(o => { if (o.name === n || (o.kind === "folder" && o.name === n)) o.have = false; });
      await loadModels(false);
    }));
  }

  function bindLiveLibrary() {
    $("#hub-open")?.addEventListener("click", () => {
      H.open = !H.open; render();
      if (H.open) { $("#hub-q")?.focus(); pollHub(); if (!H.results) hubSearch(); }
    });
    bindHub();
    if (!H.loaded) { H.loaded = true; pollHub(); }
    $("#add-place")?.addEventListener("click", () => {
      P.open = !P.open; P.error = null; render();
      if (P.open) { $("#place-path")?.focus(); if (!P.list) browsePlace(""); }
    });
    $("#new-model")?.addEventListener("click", newModel);
    bindPlace();
    if (!V.status && !V.loading && !V.error) loadVast();
    $$(".providers [data-go]").forEach(b => b.addEventListener("click", () => go(b.dataset.go)));
    $("[data-vast-train]")?.addEventListener("click", () => {
      if (V.status?.key) { V.open = true; if (!V.offers) searchVast(); }
      go("train");
    });
    $$("[data-place-remove]").forEach(b => b.addEventListener("click", async () => {
      const f = b.dataset.placeRemove;
      if (P.removing !== f) { P.removing = f; render(); setTimeout(() => { if (P.removing === f) { P.removing = null; if (state.view === "library") render(); } }, 4000); return; }
      P.removing = null;
      try { await postJson("/api/local", { path: f, remove: true }); } catch (e) { console.warn(e); }
      await loadModels(false);
    }));
    $$(".keyform").forEach(f => f.addEventListener("submit", async e => {
      e.preventDefault();
      const btn = f.querySelector("button"); btn.disabled = true;
      try { await postKeys({ provider: f.dataset.provider, label: f.label.value.trim(), key: f.key.value.trim() }); }
      catch (err) { btn.disabled = false; f.insertAdjacentHTML("afterend", `<div class="keyerr">${esc(err.message)}</div>`); }
    }));
    $$("[data-key-remove]").forEach(b => b.addEventListener("click", () => postKeys({ provider: b.dataset.keyRemove, label: b.dataset.label, remove: true }).catch(e => console.warn(e))));
    $$("[data-key-first]").forEach(b => b.addEventListener("click", () => postKeys({ provider: b.dataset.keyFirst, label: b.dataset.label, first: true }).catch(e => console.warn(e))));
    $("#refresh-models")?.addEventListener("click", () => loadModels(true));
    $("#refresh-balances")?.addEventListener("click", () => loadBalances());
    $$("[data-edit-provider]").forEach(b => b.addEventListener("click", () => startEdit(b.dataset.editProvider)));
    bindProviderEditor();
    $$("[data-add-key]").forEach(b => b.addEventListener("click", () => {
      L.adding = L.adding === b.dataset.addKey ? null : b.dataset.addKey;
      render();
      $(`.keyform[data-provider="${L.adding}"] input[name=key]`)?.focus();
    }));
    const fold = (g, v) => {
      g.classList.toggle("folded", v);
      g.querySelector("[data-fold]").setAttribute("aria-expanded", String(!v));
      setFolded(g.dataset.group, v);
    };
    $$("[data-fold]").forEach(b => b.addEventListener("click", () => { const g = b.closest(".mgroup"); fold(g, !g.classList.contains("folded")); }));
    $$("[data-fold-all]").forEach(b => b.addEventListener("click", () => $$(".mgroup").forEach(g => fold(g, b.dataset.foldAll === "true"))));
    $$("[data-show-models]").forEach(b => b.addEventListener("click", () => {
      const g = $(`.mgroup[data-group="${b.dataset.showModels}"]`);
      if (!g) return;
      fold(g, false);
      g.scrollIntoView({ behavior: "smooth", block: "start" });
    }));
    // while searching, every group with a match opens and the others step aside; folding comes back after
    const find = $("#model-find");
    find?.addEventListener("input", () => {
      const q = find.value.trim().toLowerCase();
      let any = false;
      $$(".mgroup").forEach(g => {
        let hits = 0;
        g.querySelectorAll("tr.mrow").forEach(tr => { const hit = !q || tr.dataset.find.includes(q); tr.hidden = !hit; hits += hit ? 1 : 0; });
        g.classList.toggle("searching", !!q);
        g.hidden = !!q && !hits;
        any = any || hits > 0;
      });
      const none = $(".nofind"); if (none) none.hidden = !q || any;
    });
    const tip = $(".charttip");
    $$(".daychart .bar").forEach(g => {
      const show = () => {
        const [day, per] = g.dataset.tip.split("|");
        tip.innerHTML = `<b>${esc(day)}</b>${per.split("; ").map(x => `<div>${esc(x)}</div>`).join("")}`;
        const box = g.closest(".chartwrap").getBoundingClientRect(), r = g.getBoundingClientRect();
        tip.hidden = false;
        tip.style.left = Math.min(Math.max(0, r.left - box.left + r.width / 2 - tip.offsetWidth / 2), box.width - tip.offsetWidth) + "px";
        g.classList.add("on");
      };
      const hide = () => { tip.hidden = true; g.classList.remove("on"); };
      g.addEventListener("mouseenter", show); g.addEventListener("focus", show);
      g.addEventListener("mouseleave", hide); g.addEventListener("blur", hide);
    });
    if (!L.usage) loadUsage();
    if (!L.balances && L.info && L.info.providers.some(p => p.balance && p.keys.length)) loadBalances();
  }

  async function loadUsage() {
    try { L.usage = await api("/api/usage"); } catch (e) { console.warn("cannot read usage", e); L.usage = { providers: {}, keys: {}, models: {}, daily: [], recent: [] }; }
    if (state.view === "library") render();
  }

  let balancesBusy = false;
  async function loadBalances() {
    if (balancesBusy) return;
    balancesBusy = true;
    try { L.balances = await api("/api/balances"); } catch (e) { console.warn("cannot read balances", e); L.balances = {}; }
    balancesBusy = false;
    if (state.view === "library") render();
  }

  async function loadModels(refresh) {
    try {
      L.info = await api("/api/models" + (refresh ? "?refresh=1" : ""));
      if (L.model !== "auto" && !modelByKey(L.model)) L.model = "auto";
      if (["chat", "library", "create", "voice"].includes(state.view) && !(state.view === "chat" && anyRunning())) render();
    } catch (e) {
      console.warn("cannot list models", e);
    }
  }

  // ---------- AGENTS, live ----------
  // Agents the user starts: a goal, a chat model and the studio's tools. In Ask mode a tool
  // that acts (downloads, training, asking a paid model) waits here for Allow or Deny.
  const A = { loaded: false, loading: false, error: null, runs: [], tools: [], maxSteps: 30, sel: null, detail: null,
    model: "auto", mode: "ask", steps: 12, goal: "", busy: false, formError: null, notes: {}, reply: "", removing: null,
    open: new Set(), toolsOpen: false, timer: 0, asksKey: "", detailKey: "" };
  const agentStates = { running: ["working", "c-text"], waiting: ["needs you", "c-warn"], done: ["done", "c-ok"], failed: ["failed", "c-bad"],
    stopped: ["stopped", "c-warn"], "out of steps": ["out of steps", "c-warn"] };
  const agentActive = r => r.state === "running" || r.state === "waiting";
  const shortDur = ms => { const s = Math.max(0, Math.round(ms / 1000)); return s < 60 ? s + " s" : s < 3600 ? Math.round(s / 60) + " min" : (s / 3600).toFixed(1) + " h"; };
  // the goal's first line as plain text: Markdown marks belong in the rendered goal, not in a title
  const plainMd = t => t.replace(/^#+\s+/, "").replace(/\*\*([^*]+)\*\*/g, "$1").replace(/(^|\W)\*([^*\s][^*]*)\*/g, "$1$2").replace(/`([^`]+)`/g, "$1");
  const agentTitle = g => { const t = plainMd(g.split("\n")[0]); return t.length > 90 ? t.slice(0, 87) + "…" : t; };
  const argText = a => { const s = JSON.stringify(a || {}); return s === "{}" ? "" : s; };

  function liveAgentCard(r) {
    const st = agentStates[r.state] || [r.state, "c-text"];
    const time = shortDur((r.ended || Date.now()) - r.created);
    const tok = (r.inputTokens || 0) + (r.outputTokens || 0);
    return `<div class="agent ${A.sel === r.id ? "selected" : ""}" data-run="${esc(r.id)}" style="--c:var(--cap-agent)" tabindex="0" role="button" aria-label="Open agent: ${esc(agentTitle(r.goal))}">
      <div class="glyph">${icon("agent")}</div>
      <div style="min-width:0">
        <div class="name"><span class="agent-goal" title="${esc(r.goal)}">${esc(agentTitle(r.goal))}</span></div>
        <div class="doing"><span class="chip ${st[1]}">${st[0]}</span> ${esc((r.state === "waiting" && r.pending ? "wants to call " + r.pending.tool : r.error || r.doing || "").replace(/[*`]/g, ""))}</div>
      </div>
      <div class="stats"><span>${time}</span><span>${tok ? fmtCount(tok) + " tok" : r.calls + " call" + (r.calls === 1 ? "" : "s")}</span></div>
      ${agentActive(r) ? `<span class="bar" style="width:${Math.min(100, Math.max(4, r.turns / r.maxSteps * 100))}%"></span>` : ""}
    </div>`;
  }

  function agentAsk(r, inline) {
    const p = r.pending;
    return `<div class="ask" style="--c:var(--cap-agent)">
      ${inline ? "" : `<div style="display:flex;gap:8px;align-items:center;min-width:0"><span class="dot pulse" style="--c:var(--warn)"></span><b class="agent-goal">${esc(agentTitle(r.goal))}</b></div>`}
      <div>Wants to call <code>${esc(p.tool)}</code>: ${esc(p.acts)}.</div>
      ${argText(p.arguments) ? `<div class="mono muted ask-args">${esc(argText(p.arguments))}</div>` : ""}
      <input class="input" data-ag-note="${esc(r.id)}" placeholder="Note for the agent (optional)" value="${esc(A.notes[r.id] || "")}">
      <div class="approve"><button class="btn primary" data-ag-allow="${esc(r.id)}">Allow</button><button class="btn ghost" data-ag-deny="${esc(r.id)}">Deny</button>
        ${inline ? "" : `<span class="grow"></span><button class="btn ghost" data-run="${esc(r.id)}">Open</button>`}</div>
    </div>`;
  }

  function agentAsks() {
    const xs = A.runs.filter(r => r.state === "waiting" && r.pending);
    return `<div class="col-head"><span class="h3">Waiting for you</span>${xs.length ? `<span class="chip c-warn">${xs.length}</span>` : ""}</div>
      ${xs.length ? xs.map(r => agentAsk(r, false)).join("") : `<p class="muted ag-none">Nothing waits for you. In Ask mode an agent stops here before it downloads, trains or asks a model that may cost money.</p>`}`;
  }

  function agentForm() {
    return `<form class="ag-form" id="ag-form" autocomplete="off">
      <div class="col-head" style="padding:14px 0 8px"><span class="h3">Start an agent</span></div>
      <textarea class="input" id="ag-goal" rows="3" placeholder="What should it do? For example: find a small Qwen model on Hugging Face that runs here, and download it.">${esc(A.goal)}</textarea>
      <div class="field"><label for="ag-model">Thinks with</label>${mediaPicker("ag-model", "text", A.model)}</div>
      <div class="ag-row">
        <div class="seg" role="group" aria-label="How much it may do alone">
          <button type="button" data-ag-mode="ask" aria-pressed="${A.mode === "ask"}" title="It waits for you before a tool that acts">Ask first</button>
          <button type="button" data-ag-mode="auto" aria-pressed="${A.mode === "auto"}" title="It calls every tool itself">Auto</button></div>
        <label class="muted ag-steps">Steps <input class="input mono" id="ag-steps" type="number" min="1" max="${A.maxSteps}" value="${A.steps}"></label>
      </div>
      ${A.formError ? `<div class="flag bad">${icon("alert")}<span>${esc(A.formError)}</span></div>` : ""}
      <button class="btn primary" type="submit" ${A.busy ? "disabled" : ""}>${icon("play")}${A.busy ? "Starting…" : "Start"}</button>
    </form>`;
  }

  function agentTools() {
    return `<div class="col-head" style="padding-top:18px"><span class="h3">Tools agents can call</span><span class="grow"></span>
        <button class="btn ghost sm" id="ag-tools-toggle" aria-expanded="${A.toolsOpen}">${A.toolsOpen ? "Hide" : A.tools.length + " tools"}</button></div>
      ${A.toolsOpen ? `<ul class="ag-tools">${A.tools.map(t => `<li><code>${esc(t.name)}</code>${t.acts ? `<span class="chip c-warn" title="${esc(t.acts)}">asks first</span>` : ""}<div class="muted">${esc(t.description)}</div></li>`).join("")}</ul>
        <p class="muted ag-none">Generated from the studio's own methods. Programs holding the studio's token can call them too, at <code>/api/tools</code>.</p>` : ""}`;
  }

  function agentLanes() {
    const lane = (title, xs) => `<div class="lane-col"><div class="col-head" style="padding:0 2px 8px"><span class="h3">${title}</span><span class="mono faint">${xs.length}</span></div>
      <div style="display:grid;gap:8px">${xs.map(liveAgentCard).join("") || `<p class="muted ag-none" style="padding:0 2px">None.</p>`}</div></div>`;
    if (!A.runs.length) return `<div class="empty ag-empty">${icon("agent")}<div><b>No agents yet.</b></div>
      <div class="muted">Say what you want done on the left. The agent picks from the studio's tools, such as searching Hugging Face, downloading a model, or starting a training run, and tells you what it did.</div></div>`;
    return lane("Working", A.runs.filter(r => r.state === "running"))
      + lane("Needs you", A.runs.filter(r => r.state === "waiting"))
      + lane("Finished", A.runs.filter(r => !agentActive(r)));
  }

  // the chat's Reasoning bubble, for an agent's thinking
  const agentThink = (text, open, i, live) => `<div class="st md-body">${thinkHtml(text, { open, live, attrs: i != null ? `data-ag-open="${i}"` : "" })}</div>`;

  function agentStep(s, i) {
    switch (s.kind) {
      case "goal": return `<div class="st st-goal"><span class="eyebrow">You</span><div class="md-body">${md(s.text || "")}</div></div>`;
      case "think": return agentThink(s.text || "", A.open.has(i), i);
      case "say": return `<div class="st st-say"><div class="md-body">${md(s.text || "")}</div></div>`;
      case "call": return `<div class="st st-call">${icon("plug")}<span>Calls <code>${esc(s.tool)}</code>${argText(s.arguments) ? ` <span class="mono muted">${esc(argText(s.arguments))}</span>` : ""}</span></div>`;
      case "result": return `<details class="st st-result ${s.ok ? "" : "bad"}" data-ag-open="${i}" ${A.open.has(i) ? "open" : ""}><summary>${s.ok ? icon("check") : icon("alert")}<span><code>${esc(s.tool)}</code> ${s.ok ? "answered" : "failed"}</span><span class="mono faint">${fmtCount((s.text || "").length)} chars</span></summary><pre>${esc(s.text || "")}</pre></details>`;
      case "decision": return `<div class="st st-decision">${icon(s.allow ? "check" : "x")}<span>You ${s.allow ? "allowed" : "did not allow"} <code>${esc(s.tool)}</code>${s.note ? `: “${esc(s.note)}”` : ""}</span></div>`;
      case "answer": return `<div class="st st-answer"><span class="eyebrow">Answer</span><div class="md-body">${md(s.text || "")}</div></div>`;
      case "error": return `<div class="flag bad">${icon("alert")}<span>${esc(s.text || "")}</span></div>`;
      default: return "";
    }
  }

  function agentDetailBody() {
    const r = A.detail;
    if (!r) return `<div class="empty" style="padding:40px"><span class="typing"><i></i><i></i><i></i></span></div>`;
    const st = agentStates[r.state] || [r.state, "c-text"];
    const by = r.by ? modelByKey(r.by)?.name || r.by : r.model === "auto" ? "Auto" : modelByKey(r.model)?.name || r.model;
    const tok = (r.inputTokens || 0) + (r.outputTokens || 0);
    return `<div class="ag-meta"><span class="chip ${st[1]}">${st[0]}</span><span class="mono faint">${esc(by)} · ${r.mode === "ask" ? "asks first" : "auto"} · step ${r.turns} of ${r.maxSteps}${tok ? " · " + fmtCount(tok) + " tok" : ""}</span>
        <span class="grow"></span>${agentActive(r) ? `<button class="btn" data-ag-stop="${esc(r.id)}">${icon("stop")}Stop</button>`
          : `<button class="btn ghost" data-ag-remove="${esc(r.id)}">${icon("trash")}${A.removing === r.id ? "Click again to remove" : "Remove"}</button>`}</div>
      <div class="ag-steps-list" id="ag-steps-list">${(r.steps || []).map(agentStep).join("")}
        ${r.state === "waiting" && r.pending ? agentAsk(r, true) : ""}
        ${r.state === "running" ? (r.thinking ? agentThink(r.thinking, true, null, true) : `<div class="st st-say"><span class="typing"><i></i><i></i><i></i></span></div>`) : ""}</div>
      ${agentActive(r) ? "" : `<form class="ag-reply" id="ag-reply"><textarea class="input" id="ag-reply-text" rows="2" placeholder="Tell it more and let it go on. Enter sends, Shift+Enter adds a line">${esc(A.reply)}</textarea><button class="btn primary" type="submit">${icon("send")}Send</button></form>`}`;
  }

  function liveAgentsView() {
    if (!A.loaded) return `<div class="empty" style="padding:60px"><span class="typing"><i></i><i></i><i></i></span><div class="muted">${A.error ? esc(A.error) : "Looking for agents and tools…"}</div></div>`;
    const sel = A.sel && A.runs.find(r => r.id === A.sel);
    return `<div class="mission live-agents ${sel ? "with-detail" : ""}">
      <aside class="inbox" aria-label="Start an agent and what waits for you">
        <div style="padding:0 16px">${agentForm()}</div>
        <div id="ag-asks">${agentAsks()}</div>
        <div id="ag-tools">${agentTools()}</div>
      </aside>
      <section class="lanes" id="ag-lanes">${agentLanes()}</section>
      ${sel ? `<aside class="agent-detail" id="ag-detail" aria-label="Agent">
        <div class="col-head"><span class="h3 agent-goal" title="${esc(sel.goal)}">${esc(agentTitle(sel.goal))}</span><span class="grow"></span><button class="iconbtn sm" id="ag-close" aria-label="Close">${icon("x")}</button></div>
        <div id="ag-detail-body">${agentDetailBody()}</div></aside>` : ""}
    </div>`;
  }

  const agentCrumb = () => `Agents <span class="muted">· ${A.loaded ? `${A.runs.filter(r => r.state === "running").length} working · ${A.runs.filter(r => r.state === "waiting").length} waiting for you` : "looking…"}</span>`;

  function agentBadge() {
    const n = A.runs.filter(r => r.state === "waiting").length;
    const a = $('.rail a[data-view="agents"]');
    if (!a) return;
    let b = a.querySelector(".badge");
    if (!n) { b?.remove(); return; }
    if (!b) { b = document.createElement("span"); b.className = "badge"; a.appendChild(b); }
    b.textContent = n;
  }

  // Redraw what changed without touching what the user is typing in.
  function refreshAgents() {
    agentBadge();
    if (state.view !== "agents" || !isLive()) return;
    if (!$(".live-agents") || !!A.sel !== !!$("#ag-detail")) { render(); return; }
    $("#ag-lanes").innerHTML = agentLanes();
    $("#crumb").innerHTML = agentCrumb();
    const asksKey = A.runs.filter(r => r.state === "waiting").map(r => r.id + ":" + r.pending?.since).join(",");
    if (asksKey !== A.asksKey) { A.asksKey = asksKey; $("#ag-asks").innerHTML = agentAsks(); }
    const d = A.detail;
    const detailKey = d ? [d.id, d.state, (d.steps || []).length, d.turns, A.removing, (d.thinking || "").length].join(":") : "";
    if (detailKey !== A.detailKey && $("#ag-detail-body")) {
      A.detailKey = detailKey;
      const list = $("#ag-steps-list"), atEnd = !list || list.scrollHeight - list.scrollTop - list.clientHeight < 40, top = list?.scrollTop;
      const focused = document.activeElement?.id === "ag-reply-text";
      $("#ag-detail-body").innerHTML = agentDetailBody();
      const nl = $("#ag-steps-list");
      if (nl) nl.scrollTop = atEnd ? nl.scrollHeight : top;
      if (focused) $("#ag-reply-text")?.focus();
    }
    bindAgentParts();
  }

  async function loadAgents() {
    if (A.loading) return;
    A.loading = true;
    try {
      if (!L.info) await loadModels(false);
      const j = await api("/api/agents");
      A.runs = j.runs; A.tools = j.tools; A.maxSteps = j.maxSteps; A.loaded = true; A.error = null;
      if (A.sel && !A.runs.some(r => r.id === A.sel)) { A.sel = null; A.detail = null; }
      if (A.sel) A.detail = await postJson("/api/agents/get", { id: A.sel });
    } catch (e) {
      A.error = String(e.message || e);
    } finally {
      A.loading = false;
    }
    clearTimeout(A.timer);
    A.timer = 0;
    const active = A.runs.some(agentActive);
    if (active || state.view === "agents") A.timer = setTimeout(loadAgents, active ? (state.view === "agents" ? 1000 : 3000) : 5000);
    if (state.view === "agents" && !$(".live-agents")) render(); else refreshAgents();
  }

  async function agentDo(path, body) {
    try { await postJson(path, body); } catch (e) { A.formError = String(e.message || e); render(); }
    loadAgents();
  }

  function openAgent(id) {
    A.sel = id; A.detail = A.runs.find(r => r.id === id) ? null : A.detail; A.open = new Set(); A.reply = ""; A.removing = null; A.detailKey = "";
    render();
    loadAgents();
    if (innerWidth < 1080) setTimeout(() => $("#ag-detail")?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  }

  function bindAgentParts() {
    $$("[data-run]").forEach(el => {
      if (el.dataset.bound) return; el.dataset.bound = "1";
      el.addEventListener("click", () => openAgent(el.dataset.run));
      el.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openAgent(el.dataset.run); } });
    });
    $$("[data-ag-note]").forEach(el => el.oninput = () => { A.notes[el.dataset.agNote] = el.value; });
    const decide = (id, allow) => { const note = A.notes[id] || ""; delete A.notes[id]; A.asksKey = null; A.detailKey = null; agentDo("/api/agents/answer", { id, allow, note }); };
    $$("[data-ag-allow]").forEach(el => el.onclick = () => decide(el.dataset.agAllow, true));
    $$("[data-ag-deny]").forEach(el => el.onclick = () => decide(el.dataset.agDeny, false));
    $$("[data-ag-stop]").forEach(el => el.onclick = () => agentDo("/api/agents/stop", { id: el.dataset.agStop }));
    $$("[data-ag-remove]").forEach(el => el.onclick = () => {
      const id = el.dataset.agRemove;
      if (A.removing !== id) { A.removing = id; A.detailKey = null; refreshAgents(); return; }
      A.removing = null; A.sel = null; A.detail = null; agentDo("/api/agents/delete", { id });
    });
    $$("details[data-ag-open]").forEach(el => el.ontoggle = () => { const i = +el.dataset.agOpen; if (el.open) A.open.add(i); else A.open.delete(i); });
    const reply = $("#ag-reply");
    if (reply) {
      $("#ag-reply-text").oninput = e => { A.reply = e.target.value; };
      reply.onsubmit = e => {
        e.preventDefault();
        const text = A.reply.trim();
        if (!text) return;
        A.reply = ""; A.detailKey = null;
        agentDo("/api/agents/reply", { id: A.sel, text });
      };
      // like the chat composer: Enter sends, Shift+Enter adds a line
      $("#ag-reply-text").onkeydown = e => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); reply.requestSubmit(); } };
    }
  }

  function bindLiveAgents() {
    A.asksKey = A.runs.filter(r => r.state === "waiting").map(r => r.id + ":" + r.pending?.since).join(",");
    A.detailKey = "";
    const goal = $("#ag-goal");
    if (!goal) return;
    goal.oninput = () => { A.goal = goal.value; };
    goal.onkeydown = e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("#ag-form").requestSubmit(); };
    $("#ag-model").onchange = e => { A.model = e.target.value; };
    $("#ag-steps").oninput = e => { A.steps = Math.max(1, Math.min(A.maxSteps, +e.target.value || 12)); };
    $$("[data-ag-mode]").forEach(b => b.onclick = () => { A.mode = b.dataset.agMode; $$("[data-ag-mode]").forEach(x => x.setAttribute("aria-pressed", x === b)); });
    $("#ag-form").onsubmit = async e => {
      e.preventDefault();
      if (!A.goal.trim()) { A.formError = "Say what the agent should do."; render(); return; }
      A.busy = true; A.formError = null; render();
      try {
        const r = await postJson("/api/agents/start", { goal: A.goal, model: A.model, mode: A.mode, maxSteps: A.steps });
        A.goal = ""; A.sel = r.id; A.detail = null; A.open = new Set();
      } catch (err) {
        A.formError = String(err.message || err);
      }
      A.busy = false;
      render();
      loadAgents();
    };
    $("#ag-tools-toggle").onclick = () => { A.toolsOpen = !A.toolsOpen; $("#ag-tools").innerHTML = agentTools(); bindLiveAgents(); };
    $("#ag-close")?.addEventListener("click", () => { A.sel = null; A.detail = null; render(); });
    bindAgentParts();
    if (A.detail) { const nl = $("#ag-steps-list"); if (nl) nl.scrollTop = nl.scrollHeight; }
  }

  // ---------- live data ----------
  // Served by the studio, the page carries a token and reads real data from the API. Opened
  // as a plain file or preview, it shows the example data in mock.js and says so.
  const tokenMeta = document.querySelector('meta[name="studio-token"]');
  const mediumToCap = { text: "text", image: "image", speech: "speech", vision: "vision", video: "vision", training: "train", agent: "agent" };
  // screens that show only real data
  const showsLiveData = () => ["library", "create", "voice", "builder", "train", "agents"].includes(state.view) || (state.view === "chat" && L.sel !== "example");

  async function api(path) {
    const r = await fetch(path, { headers: { "X-Studio-Token": tokenMeta.content } });
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.json();
  }

  function renderDevices() {
    const box = $("#devices");
    if (!M.system) return;   // keep the example meters
    // the device this PC's models run on is marked; a click chooses another, and a click on a chosen one goes back to the best
    const ld = M.localDevice, pinned = ld && ld.choice !== "best";
    box.innerHTML = M.system.devices.map(d => {
      const on = ld && ld.device === d.name;
      const what = d.kind === "gpu" ? "GPU" : "Processor";
      const tip = !ld ? `${what} found by a Brainiac backend`
        : on ? (pinned ? `Models on this PC run here, as you chose. Click to go back to the best device.` : `Models on this PC run here, the best device found. Click to keep them here.`)
        : `${what}. Click to run this PC's chat, image and voice models here instead.`;
      return `<button type="button" class="dev ${on ? "on" : ""} ${on && pinned ? "pinned" : ""}" data-local-device="${esc(d.name)}" aria-pressed="${on}" title="${esc(tip)}">
      <span class="dot" style="--c:${d.kind === "gpu" ? "var(--ok)" : "var(--muted)"}"></span>${esc(d.name)}${on && pinned ? `<span class="faint">chosen</span>` : ""}</button>`;
    }).join("") + gpuLimitHtml();
    wireGpuLimit(box);
    $$("[data-local-device]", box).forEach(b => b.addEventListener("click", async () => {
      if (!M.localDevice) return;
      const name = b.dataset.localDevice;
      const next = M.localDevice.choice === name ? "best" : name;
      try { M.localDevice = await postJson("/api/local/device", { device: next }); }
      catch (e) { console.warn("cannot choose the device", e); }
      renderDevices();
    }));
  }

  // The GPU memory limit: the studio takes no more than this from the card and puts the rest in system memory.
  // The core reads it once, so a change applies when the studio next starts.
  let gpuLimitEditing = false;
  const fmtGb = mb => (mb / 1024).toFixed(mb % 1024 ? 1 : 0) + " GB";
  function gpuLimitHtml() {
    const ld = M.localDevice;
    if (!ld || ld.memory === undefined || !M.system.devices.some(d => d.kind === "gpu")) return "";
    if (gpuLimitEditing) {
      return `<span class="dev gpu-limit"><label for="gpu-limit">GPU limit</label><input id="gpu-limit" data-gpu-limit-input type="number" min="0" step="0.5" inputmode="decimal" placeholder="none" value="${ld.memory ? ld.memory / 1024 : ""}" aria-label="GPU memory limit in GB, empty for none"><span class="faint">GB</span></span>`;
    }
    const waits = ld.memory !== ld.memoryInEffect;
    const tip = "The most GPU memory the studio takes; what does not fit goes to system memory, which is slower. "
      + "Set it below what other programs leave free if models fail to load."
      + (waits ? ` Now in effect: ${ld.memoryInEffect ? fmtGb(ld.memoryInEffect) : "no limit"}. The change applies when the studio is started again.` : "");
    return `<button type="button" class="dev ${waits ? "pinned" : ""}" data-gpu-limit title="${esc(tip)}">GPU limit <span class="mono tnum">${ld.memory ? fmtGb(ld.memory) : "none"}</span>${waits ? `<span class="faint">after restart</span>` : ""}</button>`;
  }
  function wireGpuLimit(box) {
    const open = $("[data-gpu-limit]", box);
    if (open) open.addEventListener("click", () => { gpuLimitEditing = true; renderDevices(); const i = $("[data-gpu-limit-input]"); if (i) { i.focus(); i.select(); } });
    const input = $("[data-gpu-limit-input]", box);
    if (!input) return;
    let done = false;
    const close = async save => {
      if (done) return;
      done = true;
      gpuLimitEditing = false;
      const gb = parseFloat(input.value);
      const mb = isFinite(gb) && gb > 0 ? Math.round(gb * 1024) : 0;
      if (save && mb !== M.localDevice.memory) {
        try { M.localDevice = await postJson("/api/local/device", { memory: mb }); }
        catch (e) { console.warn("cannot set the GPU memory limit", e); }
      }
      renderDevices();
    };
    input.addEventListener("keydown", e => { if (e.key === "Enter") close(true); else if (e.key === "Escape") close(false); });
    input.addEventListener("blur", () => close(true));
  }

  function renderDataNote() {
    const note = $("#data-note");
    note.hidden = !!tokenMeta && showsLiveData();
    note.textContent = tokenMeta ? "Example data on this screen" : "Example data";
  }

  async function loadLive() {
    if (!tokenMeta) return;
    try {
      const [caps, sys] = await Promise.all([api("/api/capabilities"), api("/api/system")]);
      M.capabilities = caps.capabilities.map(c => ({
        key: c.key, cap: mediumToCap[c.medium] || "agent", what: c.description, source: c.source, note: c.note, missing: c.status === "planned",
      }));
      M.system = sys;
      api("/api/local/device").then(d => { M.localDevice = d; renderDevices(); }).catch(() => {});
      renderDevices();
      render();
      loadModels(false);
      loadConvos();
      loadAgents();
    } catch (e) {
      console.warn("studio API not reachable, showing example data", e);
    }
  }

  const start = location.hash.slice(1);
  state.view = views[start] ? start : "chat";
  // the badge counts real agents waiting for the user
  if (isLive()) {
    $('.rail a[data-view="agents"] .badge')?.remove();
    // nothing serves MCP yet: the screen is a mock-up, so the studio does not offer it as if it were real
    $('.rail a[data-view="mcp"]')?.remove();
    if (state.view === "mcp") state.view = "chat";
  }
  render();
  loadLive();
  if (document.fonts) document.fonts.ready.then(paintAll);
})();
