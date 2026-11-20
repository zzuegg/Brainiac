import { html } from "/v2/lib/html.js";

// Model text as paragraphs and code blocks. Everything is drawn as text, never as markup.

export function Text({ text = "" }) {
  const parts = [];
  const re = /```([^\n`]*)\n([\s\S]*?)(?:```|$)/g;
  let at = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > at) parts.push({ prose: text.slice(at, m.index) });
    // a block still streaming in may end in the first backticks of its closing fence
    parts.push({ code: m[2].replace(/\n?`{1,2}$/, "").replace(/\n$/, ""), lang: m[1].trim() });
    at = re.lastIndex;
  }
  if (at < text.length) parts.push({ prose: text.slice(at) });
  return html`${parts.map(p => p.code != null
    ? html`<pre class="code" data-lang=${p.lang || undefined}><code>${p.code}</code></pre>`
    : p.prose.split(/\n{2,}/).filter(s => s.trim()).map(s => html`<p>${s.trim()}</p>`))}`;
}
