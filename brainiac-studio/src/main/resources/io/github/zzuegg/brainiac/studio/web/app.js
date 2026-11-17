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
    const lay = effectiveLayout();
    const insp = lay !== "focus" || state.inspectorOpen;
    return `<div class="chat lay-${lay} ${insp ? "" : "no-inspector"}">
      ${threadList()}${transcript()}${inspector()}${lay === "ultra" ? mediaPane() : ""}
    </div>`;
  }

  // ---------- CREATE (images) ----------
  function createView() {
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
      <div class="tablebox"><table class="models">
        <thead><tr><th>Model</th><th>Can</th><th>Format</th><th>Size</th><th>Context</th><th>Fits on</th><th>State</th></tr></thead>
        <tbody>${M.models.map(m => `<tr>
          <td><div class="nm">${esc(m.name)}</div><div class="src">${esc(m.src)}</div></td>
          <td>${m.cap.map(c => `<span class="chip c-${c}">${capName[c]}</span>`).join(" ")}</td>
          <td class="mono">${m.fmt}</td><td class="mono tnum">${m.size}</td><td class="mono">${m.ctx}</td><td>${fitCell(m)}</td>
          <td>${m.state === "loaded" ? `<span class="chip c-ok">loaded · ${m.tps}</span>` : m.state === "remote" ? `<span class="chip c-agent">remote · your key</span>` : m.state === "planned" ? `<span class="chip c-warn">planned</span>` : m.state === "draft" ? `<span class="chip c-text">draft model</span>` : `<button class="btn" style="height:26px;font-size:12px">Load</button>`}</td>
        </tr>`).join("")}</tbody></table></div>
      <div style="display:flex;gap:10px;align-items:baseline;flex-wrap:wrap"><span class="h2">Providers</span><span class="muted">Remote models sit next to local ones in routing; their model lists come from each provider.</span></div>
      <div class="providers">${M.providers.map(p => `
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
      </div>
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
    $("#crumb").innerHTML = crumbs[state.view];
    const [c, n] = routes[state.view];
    $("#router .chip").className = "chip " + c; $("#router .chip").textContent = n;
    $$(".rail a").forEach(a => { if (a.dataset.view === state.view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
    paintAll();
    renderDataNote();
    if (state.view === "builder") { drawEdges(); fitWorld(); enableDrag(); }
    if (state.view === "chat") bindChat();
    if (state.view === "agents") document.querySelectorAll(".agent[data-agent]").forEach(el => el.addEventListener("click", () => { if (["script", "voice", "thumb"].includes(el.dataset.agent)) { state.agent = el.dataset.agent; state.inspectorOpen = true; go("chat"); } }));
  }

  function paintAll() {
    $$("canvas[data-thumb]").forEach(cv => paintThumb(cv, +cv.dataset.thumb, +(cv.dataset.noise || 0)));
    const f = $("#frame"); if (f) paintFrame(f, state.t);
    $$("canvas.frame").forEach(c => paintFrame(c, +c.dataset.t));
  }

  function bindChat() {
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

  function go(v) { state.view = v; if (location.hash !== "#" + v) history.replaceState(null, "", "#" + v); render(); }

  // ---------- wiring ----------
  window.addEventListener("hashchange", () => { const v = location.hash.slice(1); if (views[v]) { state.view = v; render(); } });
  window.addEventListener("resize", () => { const before = $("#app").dataset.layout; applyFrame(); if ($("#app").dataset.layout !== before) render(); else if (state.view === "builder") fitWorld(); });
  document.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openCmdk(); } });
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

  // ---------- live data ----------
  // Served by the studio, the page carries a token and reads real data from the API. Opened
  // as a plain file or preview, it shows the example data in mock.js and says so.
  const tokenMeta = document.querySelector('meta[name="studio-token"]');
  const mediumToCap = { text: "text", image: "image", speech: "speech", vision: "vision", video: "vision", training: "train", agent: "agent" };
  const liveViews = new Set();   // screens that show only real data; none yet

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
    note.hidden = !!tokenMeta && liveViews.has(state.view);
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
