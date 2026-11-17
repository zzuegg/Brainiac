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

  const store = {
    get(k, d) { try { const v = localStorage.getItem("studio." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("studio." + k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  };

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
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><span class="h2">Models</span><span class="grow"></span>
        <button class="btn">${icon("plus")}Add folder or GGUF</button><button class="btn primary">${icon("build")}New model</button></div>
      ${isLive() ? liveModelsTable() : `<div class="tablebox"><table class="models">
        <thead><tr><th>Model</th><th>Can</th><th>Format</th><th>Size</th><th>Context</th><th>Fits on</th><th>State</th></tr></thead>
        <tbody>${M.models.map(m => `<tr>
          <td><div class="nm">${esc(m.name)}</div><div class="src">${esc(m.src)}</div></td>
          <td>${m.cap.map(c => `<span class="chip c-${c}">${capName[c]}</span>`).join(" ")}</td>
          <td class="mono">${m.fmt}</td><td class="mono tnum">${m.size}</td><td class="mono">${m.ctx}</td><td>${fitCell(m)}</td>
          <td>${m.state === "loaded" ? `<span class="chip c-ok">loaded · ${m.tps}</span>` : m.state === "remote" ? `<span class="chip c-agent">remote · your key</span>` : m.state === "planned" ? `<span class="chip c-warn">planned</span>` : m.state === "draft" ? `<span class="chip c-text">draft model</span>` : `<button class="btn" style="height:26px;font-size:12px">Load</button>`}</td>
        </tr>`).join("")}</tbody></table></div>`}
      <div style="display:flex;gap:10px;align-items:baseline;flex-wrap:wrap"><span class="h2">Providers</span><span class="muted">Remote models sit next to local ones in routing; their model lists come from each provider.</span></div>
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
    </div>`;
  }

  // ---------- render ----------
  const views = { chat: chatView, agents: agentsView, create: createView, voice: voiceView, builder: builderView, train: trainView, library: libraryView, mcp: mcpView };
  const crumbs = {
    chat: `Chat <span class="muted">· Voiceover for the builder demo</span>`, agents: `Agents <span class="muted">· 4 running · 2 waiting for you</span>`, create: `Images <span class="muted">· thumb-dit-64</span>`,
    voice: `Voice <span class="muted">· Builder demo voiceover</span>`, builder: `Model builder <span class="muted">· resnet-s</span>`,
    train: `Training <span class="muted">· 3 hosts · 2 runs live</span>`, library: `Models <span class="muted">· 6 on this machine</span>`, mcp: `MCP <span class="muted">· 2 agents connected</span>`,
  };
  const routes = {
    chat: ["c-text", "Qwen2.5 7B Instruct"], agents: ["c-agent", "per agent"], create: ["c-image", "thumb-dit-64"], voice: ["c-speech", "OpenAI speech · remote"],
    builder: ["c-train", "resnet-s (unsaved)"], train: ["c-train", "2 hosts busy"], library: ["c-text", "Qwen2.5 7B Instruct"], mcp: ["c-agent", "per call"],
  };

  function render() {
    applyFrame();
    $("#main").innerHTML = `<div class="view">${views[state.view]()}</div>`;
    const liveChat = isLive() && state.view === "chat" && L.sel !== "example";
    const liveLib = isLive() && state.view === "library";
    const liveMedia = isLive() && (state.view === "create" || state.view === "voice");
    const ready = (L.info?.providers || []).filter(p => p.ready).length;
    $("#crumb").innerHTML = liveChat ? `Chat <span class="muted">· ${esc(current()?.title || "New conversation")}</span>`
      : liveMedia ? (state.view === "create" ? `Images <span class="muted">· ${G.batches.length} batch${G.batches.length === 1 ? "" : "es"} this session</span>` : `Voice <span class="muted">· ${G.takes.length} take${G.takes.length === 1 ? "" : "s"} this session</span>`)
      : liveLib ? `Models <span class="muted">· ${L.info ? `${L.info.models.length} from ${ready} provider${ready === 1 ? "" : "s"}` : "looking…"}</span>` : crumbs[state.view];
    const picked = liveChat ? paneModel(L.active) : "auto";
    const autoModel = (liveChat || liveLib) && L.info ? modelByKey(picked === "auto" ? L.info.auto.text : picked) : null;
    const mediaModel = liveMedia ? (state.view === "create" ? pickedModel(G.imgModel, "image") : pickedModel(G.voModel, "speech")) : null;
    const [c, n] = liveMedia ? [state.view === "create" ? "c-image" : "c-speech", mediaModel ? mediaModel.name : L.info ? "no model yet" : "…"]
      : liveChat || liveLib ? ["c-text", autoModel ? autoModel.name : L.info ? "no chat model yet" : "…"] : routes[state.view];
    $("#router .chip").className = "chip " + c; $("#router .chip").textContent = n;
    $$(".rail a").forEach(a => { if (a.dataset.view === state.view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
    paintAll();
    renderDataNote();
    if (state.view === "builder") { drawEdges(); fitWorld(); enableDrag(); }
    if (state.view === "chat") bindChat();
    if (state.view === "library" && isLive()) bindLiveLibrary();
    if (state.view === "create" && isLive()) bindLiveCreate();
    if (state.view === "voice" && isLive()) bindLiveVoice();
    if (state.view === "agents") document.querySelectorAll(".agent[data-agent]").forEach(el => el.addEventListener("click", () => { if (["script", "voice", "thumb"].includes(el.dataset.agent)) { state.agent = el.dataset.agent; state.inspectorOpen = true; go("chat"); } }));
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
  const commands = [
    ["Go to Chat", "chat", "chat"], ["Go to Images", "image", "create"], ["Go to Voice", "speech", "voice"], ["Go to Model builder", "build", "builder"],
    ["Go to Training", "train", "train"], ["Go to Models", "library", "library"], ["Go to MCP server", "plug", "mcp"],
    ["Load a model from a folder or GGUF…", "plus", null], ["New thread", "chat", "chat"], ["Switch layout: Ultrawide", "layout", "ultra"], ["Switch layout: Focus", "layout", "focus"],
  ];
  function openCmdk() {
    const back = document.createElement("div");
    back.className = "cmdk-back";
    back.innerHTML = `<div class="cmdk" role="dialog" aria-label="Commands"><input id="cmdk-in" placeholder="Type a command, a model or a thread"><ul id="cmdk-list"></ul></div>`;
    document.body.appendChild(back);
    let sel = 0;
    const list = () => {
      const q = $("#cmdk-in").value.toLowerCase();
      const items = commands.filter(c => c[0].toLowerCase().includes(q));
      sel = Math.min(sel, Math.max(0, items.length - 1));
      $("#cmdk-list").innerHTML = items.map((c, i) => `<li class="${i === sel ? "on" : ""}">${icon(c[1])}<span class="grow">${c[0]}</span></li>`).join("");
      return items;
    };
    const run = c => { back.remove(); if (!c || !c[2]) return; if (views[c[2]]) go(c[2]); else { state.layout = c[2]; store.set("layout", c[2]); render(); } };
    list();
    $("#cmdk-in").focus();
    $("#cmdk-in").addEventListener("input", list);
    back.addEventListener("keydown", e => {
      const items = list();
      if (e.key === "Escape") back.remove();
      if (e.key === "ArrowDown") { sel = Math.min(items.length - 1, sel + 1); list(); e.preventDefault(); }
      if (e.key === "ArrowUp") { sel = Math.max(0, sel - 1); list(); e.preventDefault(); }
      if (e.key === "Enter") run(items[sel]);
    });
    back.addEventListener("click", e => { if (e.target === back) back.remove(); });
  }

  function go(v) { if (v === "library" && state.view !== "library") L.usage = null; state.view = v; if (location.hash !== "#" + v) history.replaceState(null, "", "#" + v); render(); }

  // ---------- wiring ----------
  window.addEventListener("hashchange", () => { const v = location.hash.slice(1); if (views[v]) { if (v === "library" && state.view !== "library") L.usage = null; state.view = v; render(); } });
  window.addEventListener("resize", () => { const before = $("#app").dataset.layout; applyFrame(); if ($("#app").dataset.layout !== before) render(); else if (state.view === "builder") fitWorld(); else if (state.view === "chat" && isLive() && L.sel !== "example" && L.panes.length > maxPanes()) render(); });
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
  // Served by the studio, Chat holds real conversations. They are kept in this browser; the
  // example conversation stays in the list so the agent screens can still be seen.
  const L = {
    info: null,                           // GET /api/models: providers, models, what auto picks
    convos: store.get("convos", []),
    sel: store.get("convo", null) === "example" ? "example" : null,   // "example" while the design preview shows
    model: store.get("chatModel", "auto"),  // what a new conversation starts with
    tabs: store.get("chatTabs", []),      // open conversations, in tab order
    panes: store.get("chatPanes", [{ id: store.get("convo", null) === "example" ? null : store.get("convo", null) }]),
    active: store.get("chatActive", 0),   // the pane that has the focus
    running: {},                          // conversation id: the AbortController of the answer being written
    drafts: {},                           // conversation id ("#" and the pane for a new one): what is typed but not sent
    scroll: {},                           // conversation id: where its transcript was scrolled to
    showFinished: false,
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
  const mediaChip = { text: "c-text", vision: "c-vision", image: "c-image", speech: "c-speech", transcribe: "c-speech", video: "c-video" };
  const mediaName = { text: "Text", vision: "Vision", image: "Image", speech: "Speech", transcribe: "Transcribe", video: "Video" };

  // A little Markdown: code fences, inline code, bold, paragraphs. Everything is escaped first.
  function md(text) {
    const inline = s => esc(s).replace(/`([^`\n]+)`/g, "<code>$1</code>").replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>")
      .replace(/(^|[\s(])\*([^*\s][^*\n]*?)\*(?=[\s.,;:!?)]|$)/g, "$1<i>$2</i>").replace(/\n/g, "<br>");
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
      for (const raw of part.split("\n")) {
        const line = raw.trimEnd();
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
  function savePanes() {
    store.set("chatPanes", L.panes.map(p => ({ id: p.id || null, model: p.model || null })));
    store.set("chatTabs", L.tabs);
    store.set("chatActive", L.active);
  }
  function fitPanes() {
    L.tabs = L.tabs.filter(id => { const c = convoById(id); return c && c.status !== "finished"; });
    L.panes = L.panes.filter(p => !p.id || convoById(p.id)).slice(0, maxPanes());
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
  const saveConvos = () => store.set("convos", L.convos.slice(0, 200));
  const pending = new Map();
  function persist(c) {
    saveConvos();
    if (!isLive()) return;
    clearTimeout(pending.get(c.id));
    pending.set(c.id, setTimeout(() => {
      pending.delete(c.id);
      const body = { ...c, messages: c.messages.map(m => { const { streaming, ...rest } = m; return rest; }) };
      fetch("/api/chats", { method: "POST", headers: { "X-Studio-Token": tokenMeta.content, "Content-Type": "application/json" }, body: JSON.stringify(body) })
        .then(r => { if (!r.ok) console.warn("cannot save the conversation", r.status); })
        .catch(e => console.warn("cannot save the conversation", e));
    }, 400));
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
      onlyHere.forEach(persist);
      saveConvos();
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
            <button class="iconbtn sm" data-delete="${c.id}" aria-label="Delete" title="Delete">${icon("stop")}</button>`
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
        <div class="tgroup"><span class="eyebrow">Example</span>
          <div class="titem ${L.sel === "example" ? "active" : ""}" data-convo="example" tabindex="0">
            <span class="dot ic" style="--c:var(--cap-agent)"></span>
            <div style="min-width:0"><div class="t">Voiceover for the builder demo</div><div class="s">design preview · agents</div></div>
            <span class="when"></span></div>
        </div>
      </div>
    </aside>`;
  }

  function msgHtml(m, i) {
    if (m.role === "user") {
      return `<div class="msg user" data-i="${i}"><div class="avatar user">${icon("chat")}</div><div style="min-width:0">
        <div class="meta"><b>You</b><span>${m.at ? when(m.at) : ""}</span></div><div class="body">${md(m.text)}</div></div></div>`;
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
          ${m.thinking ? `<details class="tool think" ${m.streaming && !m.text ? "open" : ""}><summary>${icon("wand")}<span><b>Reasoning</b></span><span class="grow"></span><span class="mono muted">${m.streaming && !m.text ? "thinking…" : "summary"}</span></summary><pre>${esc(m.thinking)}</pre></details>` : ""}
          ${md(m.text)}
          ${m.streaming && !m.text && !m.thinking ? `<span class="typing" aria-label="Writing"><i></i><i></i><i></i></span>` : ""}
          ${m.error ? `<div class="flag bad">${icon("flask")}<div><b>No answer.</b> ${esc(m.error)}</div></div>` : ""}
        </div>
        ${m.streaming ? "" : `<div class="actions">
          <button class="iconbtn" data-copy="${i}" aria-label="Copy">${icon("copy")}</button>
          <button class="iconbtn" data-retry="${i}" aria-label="Answer again">${icon("retry")}</button></div>`}
      </div></div>`;
  }

  function routeLine(model) {
    if (!L.info) return `${icon("wand")}<span>Looking for models on this PC and at your providers…</span>`;
    const key = model === "auto" ? L.info.auto.text : model;
    const m = modelByKey(key);
    if (!key) return `${icon("wand")}<span>No chat model yet.</span>`;
    const where = m && m.provider === "local" ? "this PC · nothing leaves it" : `${providerName(m ? m.provider : key.split("/")[0])} · uses your key`;
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
        ${c ? (fin ? `<button class="btn sm" data-reopen="${c.id}">${icon("retry")}Reopen</button>`
          : `<button class="btn ghost sm" data-finish="${c.id}" title="Mark finished: it leaves the tabs and waits under Finished">${icon("check")}Finish</button>`) : ""}
        ${L.panes.length > 1 ? `<button class="iconbtn" data-close-pane="${i}" aria-label="Close this pane" title="Close this pane"><span aria-hidden="true" style="font-size:17px;line-height:1">×</span></button>` : ""}
      </div>
      <div class="transcript" data-transcript="${i}"><div class="lane">${c && c.messages.length ? c.messages.map(msgHtml).join("") : emptyState()}</div></div>
      <div class="composer-wrap">
        ${fin ? `<div class="finished-bar">${icon("check")}<span>Finished ${when(c.finishedAt || c.updated)}. Reopen it to keep going.</span><span class="grow"></span>
            <button class="btn sm" data-reopen="${c.id}">Reopen</button></div>`
        : `<div class="composer">
          <div class="route-hint" data-route="${i}">${routeLine(paneModel(i))}</div>
          <textarea data-prompt="${i}" rows="2" placeholder="Ask anything. Enter sends, Shift+Enter adds a line">${esc(L.drafts[draftKey(i)] || "")}</textarea>
          <div class="row">
            <span class="faint" style="font-size:12px;padding-left:6px">Saved in the studio</span>
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
      if (last && last.role === m.role) last.text += "\n\n" + m.text; else out.push({ role: m.role, text: m.text });
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
        body: JSON.stringify({ model: usable(c.model || L.model), messages: history }),
      });
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
    } catch (e) {
      if (e.name === "AbortError") a.stopped = true; else a.error = e.message;
    }
    a.streaming = false;
    delete L.running[c.id];
    c.updated = Date.now();
    persist(c);
    if (state.view === "chat") render();
  }

  function send(i) {
    const p = $(`[data-prompt="${i}"]`), text = (p?.value || "").trim();
    let c = paneConvo(i);
    if (!text || !L.info || (c && running(c.id))) return;
    if (!c) {
      c = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), title: text.replace(/\s+/g, " ").slice(0, 70),
        messages: [], updated: Date.now(), status: "open", model: L.panes[i].model || L.model };
      L.convos.unshift(c);
      L.panes[i] = { id: c.id };
      L.tabs.push(c.id);
      L.drafts["#" + i] = "";
      savePanes();
    }
    L.active = i;
    L.drafts[c.id] = "";
    c.messages.push({ role: "user", text, at: Date.now() });
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
    $$("[data-prompt]").forEach(p => {
      const i = +p.dataset.prompt;
      p.addEventListener("input", () => { L.drafts[draftKey(i)] = p.value; });
      p.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(i); } });
    });
    $$("[data-send]").forEach(b => b.addEventListener("click", () => send(+b.dataset.send)));
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
    voModel: store.get("voModel", "auto"), voice: store.get("voice", null), speed: 1, instructions: "", format: "mp3",
    text: "", takes: [], speaking: false, video: null,
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
      ${L.info ? `Nothing on this PC can ${medium === "image" ? "draw" : "speak"} yet. Add an OpenAI or Google API key and their ${medium === "image" ? "image" : "voice"} models appear here.` : "Looking for models…"}
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

  function liveCreateView() {
    const m = pickedModel(G.imgModel, "image");
    const ratios = [["1:1", 14, 14], ["4:3", 16, 12], ["16:9", 18, 10], ["3:4", 12, 16], ["9:16", 10, 18]];
    return `
    <div class="create">
      <aside class="side" aria-label="Image settings">
        <div style="display:flex;align-items:center;gap:8px"><span class="h2">Images</span><span class="grow"></span>${m ? `<span class="chip c-image">${esc(m.name)}</span>` : ""}</div>
        <div class="field"><label for="img-model">Model</label>${mediaPicker("img-model", "image", G.imgModel)}</div>
        ${remoteNote(m, "The prompt")}
        <div class="prompt-box"><textarea id="img-prompt" aria-label="Prompt" placeholder="What should the picture show?">${esc(G.prompt)}</textarea></div>
        <div class="field"><label>Aspect</label>
          <div class="ratio">${ratios.map(([l, w, h]) => `<button data-aspect="${l}" aria-pressed="${G.aspect === l}"><i style="width:${w}px;height:${h}px"></i>${l}</button>`).join("")}</div></div>
        <div class="field"><label>Pictures</label><div class="seg" style="width:100%">${[1, 2, 3, 4].map(n => `<button data-count="${n}" aria-pressed="${G.count === n}" style="flex:1">${n}</button>`).join("")}</div></div>
        <button class="btn primary" id="img-go" style="height:40px;justify-content:center" ${!m || G.drawing ? "disabled" : ""}>${icon("wand")}${G.drawing ? "Drawing…" : `Draw ${G.count}`} <span class="kbd" style="color:inherit;border-color:currentColor;background:transparent;opacity:.7">Ctrl ↵</span></button>
      </aside>
      <section class="gallery" aria-label="Pictures">
        ${!mediaModels("image").length ? noModel("image") : ""}
        ${G.drawing ? `<div><div class="batch-head"><span class="eyebrow">Drawing</span><span class="p">${esc(G.drawing.prompt)}</span><span class="mono muted">${esc(G.drawing.model)}</span></div>
          <div class="batch">${Array.from({ length: G.drawing.count }, () => `<div class="tile wait" style="aspect-ratio:${ratioCss(G.drawing.aspect)}"></div>`).join("")}</div></div>` : ""}
        ${G.batches.map((b, bi) => `
          <div>
            <div class="batch-head"><span class="p">${esc(b.prompt)}</span><span class="mono muted">${esc(b.model)} · ${b.aspect}${b.seconds ? ` · ${b.seconds.toFixed(1)} s` : ""}</span></div>
            ${b.error ? `<div class="flag bad">${icon("flask")}<div><b>No picture.</b> ${esc(b.error)}</div></div>` : `
            <div class="batch">${b.items.map(it => `<div class="tile" title="Saved as ${esc(it.file)}"><img src="${it.url}" alt="${esc(b.prompt)}" style="aspect-ratio:${ratioCss(b.aspect)}">
              <div class="ov"><a href="${it.url}" download="${esc(fileName(it.file))}">Save</a><button data-reuse="${bi}">Reuse prompt</button></div></div>`).join("")}</div>`}
          </div>`).join("")}
        ${mediaModels("image").length && !G.batches.length && !G.drawing ? `<div class="empty"><div class="h2">Describe a picture</div><div class="muted">Pictures are also saved in the studio's outputs folder.</div></div>` : ""}
      </section>
    </div>`;
  }

  async function draw() {
    const m = pickedModel(G.imgModel, "image");
    const prompt = ($("#img-prompt")?.value || "").trim();
    if (!m || !prompt || G.drawing) return;
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
    $("#img-model")?.addEventListener("change", e => { G.imgModel = e.target.value; store.set("imgModel", G.imgModel); G.prompt = $("#img-prompt").value; render(); });
    $$("[data-aspect]").forEach(b => b.addEventListener("click", () => { G.aspect = b.dataset.aspect; store.set("aspect", G.aspect); G.prompt = $("#img-prompt").value; render(); }));
    $$("[data-count]").forEach(b => b.addEventListener("click", () => { G.count = +b.dataset.count; store.set("imgCount", G.count); G.prompt = $("#img-prompt").value; render(); }));
    $("#img-prompt")?.addEventListener("input", e => { G.prompt = e.target.value; });
    $("#img-prompt")?.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") draw(); });
    $("#img-go")?.addEventListener("click", draw);
    $$("[data-reuse]").forEach(b => b.addEventListener("click", () => { G.prompt = G.batches[+b.dataset.reuse].prompt; render(); $("#img-prompt").focus(); }));
    $$("[data-go]").forEach(b => b.addEventListener("click", () => go(b.dataset.go)));
  }

  function hue(name) { let h = 0; for (const c of name) h = (h * 31 + c.charCodeAt(0)) % 360; return `oklch(0.78 0.11 ${h})`; }

  function liveVoiceView() {
    const m = pickedModel(G.voModel, "speech");
    const opt = m?.speech || { voices: [], speed: false, instructions: false, formats: [] };
    if (!opt.voices.includes(G.voice)) G.voice = opt.voices[0] || null;
    if (opt.formats.length && !opt.formats.includes(G.format)) G.format = opt.formats[0];
    return `
    <div class="voice">
      <aside class="voices" aria-label="Voices">
        <div class="col-head"><span class="h3">Voices</span><span class="grow"></span><span class="mono faint">${opt.voices.length || ""}</span></div>
        ${opt.voices.map(v => `
          <div class="vcard ${v === G.voice ? "active" : ""}" data-voice="${esc(v)}" tabindex="0" role="button" aria-pressed="${v === G.voice}">
            <div class="vface" style="background:${hue(v)}">${esc(v[0].toUpperCase())}</div>
            <div style="min-width:0"><div class="n">${esc(v[0].toUpperCase() + v.slice(1))}</div><div class="d">${esc(providerName(m.provider))}</div></div>
            <button class="iconbtn" data-preview="${esc(v)}" aria-label="Hear ${esc(v)}" title="Speaks one short sentence, billed to your key">${icon("play")}</button>
          </div>`).join("")}
      </aside>
      <section style="display:flex;flex-direction:column;min-height:0;min-width:0">
        <div class="stage-head"><span class="title">Voice</span>${m ? `<span class="chip c-speech">${esc(m.name)}</span>` : ""}<span class="grow"></span>
          <label class="btn" for="vo-video-file">${icon("video")}${G.video ? "Change video" : "Add a video"}</label><input type="file" id="vo-video-file" accept="video/*" hidden>
          <button class="btn primary" id="vo-go" ${!m || G.speaking ? "disabled" : ""}>${icon("speech")}${G.speaking ? "Speaking…" : "Speak"}</button></div>
        ${m ? `<div style="padding:14px 24px 0">${remoteNote(m, "The text")}</div>` : ""}
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
        <div class="field"><label for="vo-instr">How to say it ${opt.instructions ? "" : "<output>not offered</output>"}</label>
          <textarea class="input" id="vo-instr" rows="3" placeholder="Calm narrator, a little slower, warm" ${opt.instructions ? "" : "disabled"}>${esc(G.instructions)}</textarea></div>
        ${opt.formats.length > 1 ? `<div class="field"><label>File</label><div class="seg">${opt.formats.map(f => `<button data-format="${f}" aria-pressed="${G.format === f}">${f.toUpperCase()}</button>`).join("")}</div></div>`
          : opt.formats.length ? `<div class="field"><label>File <output>${opt.formats[0].toUpperCase()}</output></label></div>` : ""}
      </aside>
    </div>`;
  }

  async function speak(text, voice, preview) {
    const m = pickedModel(G.voModel, "speech");
    if (!m || !text || !voice) return;
    const take = { text, voice, model: m.name };
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
    const keep = () => { G.text = $("#vo-text")?.value ?? G.text; G.instructions = $("#vo-instr")?.value ?? G.instructions; };
    $("#vo-model")?.addEventListener("change", e => { keep(); G.voModel = e.target.value; store.set("voModel", G.voModel); render(); });
    $$("[data-voice]").forEach(el => el.addEventListener("click", e => {
      if (e.target.closest("[data-preview]")) return;
      keep(); G.voice = el.dataset.voice; store.set("voice", G.voice); render();
    }));
    $$("[data-preview]").forEach(b => b.addEventListener("click", () => speak(`Hello, I'm ${b.dataset.preview}. This is how I sound.`, b.dataset.preview, true)));
    $("#vo-speed")?.addEventListener("input", e => { G.speed = +e.target.value; e.target.previousElementSibling.querySelector("output").textContent = G.speed.toFixed(2) + "×"; });
    $("#vo-instr")?.addEventListener("input", e => { G.instructions = e.target.value; });
    $("#vo-text")?.addEventListener("input", e => { G.text = e.target.value; });
    $("#vo-text")?.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { keep(); speak(G.text.trim(), G.voice); } });
    $$("[data-format]").forEach(b => b.addEventListener("click", () => { keep(); G.format = b.dataset.format; render(); }));
    $("#vo-go")?.addEventListener("click", () => { keep(); speak(G.text.trim(), G.voice); });
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
          ${id === "local" ? `<span class="chip c-ok">free · private</span>` : `<span class="chip c-agent">your key</span>`}
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
      ${body || `<tbody><tr><td colspan="4" class="muted">${L.info ? "No models yet: add a model to a model folder, or an API key below." : "Looking for models…"}</td></tr></tbody>`}</table></div>
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
    const ready = L.info.providers.filter(p => p.local || p.keys.length);
    const more = L.info.providers.filter(p => !p.local && !p.keys.length);
    const keyUse = (p, label) => usageLine(L.usage && L.usage.keys[p.id + "/" + label]);
    const form = (p, open) => `<form class="keyform" data-provider="${p.id}" ${open ? "" : "hidden"}>
          <input class="input" name="label" placeholder="Name" value="${p.keys.some(k => k.source === "saved") ? "" : "default"}" aria-label="Key name, such as work" required maxlength="40">
          <input class="input mono" name="key" type="password" placeholder="API key" aria-label="${esc(p.name)} API key" required autocomplete="off">
          <button class="btn" type="submit">${icon("plus")}Add</button>
        </form>`;
    const card = p => {
      const head = `<div style="display:flex;gap:10px;align-items:center"><span class="plogo">${esc(p.name[0])}</span><b>${esc(p.name)}</b><span class="grow"></span>
        ${p.ready ? `<button class="chip c-ok linkchip" data-show-models="${p.id}" title="Show ${esc(p.name)} models in the table">${p.models} model${p.models === 1 ? "" : "s"}</button>` : `<span class="chip plain">${p.local ? "no models" : "no key"}</span>`}</div>`;
      if (p.local) {
        return `<div class="prov ${p.ready ? "on" : "off"}">${head}
          <div class="muted" style="font-size:12.5px">Hugging Face model folders (Llama, Qwen2, Mistral) and <code>.gguf</code> files in:</div>
          ${p.folders.map(f => `<div class="mono faint" style="font-size:11.5px;word-break:break-all">${esc(f)}</div>`).join("")}
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
    return `<div class="providers">${ready.map(card).join("")}</div>
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
        <span class="muted" style="font-size:12.5px">Auto uses a model on this PC when there is one, then the providers with keys, in the order shown here.</span>
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

  function bindLiveLibrary() {
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

  // ---------- live data ----------
  // Served by the studio, the page carries a token and reads real data from the API. Opened
  // as a plain file or preview, it shows the example data in mock.js and says so.
  const tokenMeta = document.querySelector('meta[name="studio-token"]');
  const mediumToCap = { text: "text", image: "image", speech: "speech", vision: "vision", video: "vision", training: "train", agent: "agent" };
  // screens that show only real data
  const showsLiveData = () => ["library", "create", "voice"].includes(state.view) || (state.view === "chat" && L.sel !== "example");

  async function api(path) {
    const r = await fetch(path, { headers: { "X-Studio-Token": tokenMeta.content } });
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.json();
  }

  function renderDevices() {
    const box = $("#devices");
    if (!M.system) return;   // keep the example meters
    box.innerHTML = M.system.devices.map(d => `<span class="dev" title="${d.kind === "gpu" ? "GPU" : "Processor"} found by a Brainiac backend">
      <span class="dot" style="--c:${d.kind === "gpu" ? "var(--ok)" : "var(--muted)"}"></span>${esc(d.name)}</span>`).join("");
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
      renderDevices();
      render();
      loadModels(false);
      loadConvos();
    } catch (e) {
      console.warn("studio API not reachable, showing example data", e);
    }
  }

  const start = location.hash.slice(1);
  state.view = views[start] ? start : "chat";
  render();
  loadLive();
  if (document.fonts) document.fonts.ready.then(paintAll);
})();
