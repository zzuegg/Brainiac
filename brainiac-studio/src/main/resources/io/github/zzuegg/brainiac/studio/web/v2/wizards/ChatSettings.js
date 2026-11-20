import { html, useState } from "/v2/lib/html.js";
import { Layer } from "/v2/ui/Layer.js";
import { Field } from "/v2/ui/Wizard.js";
import { Button } from "/v2/ui/Button.js";
import { configure } from "/v2/state/chats.js";

// A conversation's own settings, in a drawer so the conversation stays in view: instructions
// the model gets first, and how freely it answers. Each change is kept at once.

export function ChatSettings({ s, close, top }) {
  const c = s.chat.value;
  const [system, setSystem] = useState(c.system || "");
  const num = (k, v, min, max, int) => {
    const n = int ? Math.round(+v) : +v;
    configure(s, { [k]: v === "" || !isFinite(n) ? null : Math.max(min, Math.min(max, n)) });
  };
  const done = () => {
    if (system.trim() !== (c.system || "")) configure(s, { system: system.trim() || null });
    close();
  };
  return html`<${Layer} kind="drawer" title="Chat settings" subtitle="For this conversation only." close=${done} top=${top}
      footer=${html`<${Button} variant="primary" onClick=${done}>Done<//>`}>
    <${Field} label="Instructions" hint="What the model reads before every message: a role, a style, facts to keep in mind.">
      <textarea class="input" rows="6" value=${system} placeholder="You are a concise assistant. Answer in German."
        onInput=${e => setSystem(e.target.value)} onBlur=${() => configure(s, { system: system.trim() || null })} /><//>
    <${Field} label="Temperature" hint="Lower answers the same way each time; higher varies more. Empty uses the model's own.">
      <input class="input narrow" type="number" min="0" max="2" step="0.1" value=${c.temperature ?? ""} placeholder="model's"
        onChange=${e => num("temperature", e.target.value, 0, 2)} /><//>
    <${Field} label="Longest answer" hint="In tokens, about ¾ of a word each. Empty lets the model decide.">
      <input class="input narrow" type="number" min="1" max="65536" step="256" value=${c.maxTokens ?? ""} placeholder="model's"
        onChange=${e => num("maxTokens", e.target.value, 1, 65536, true)} /><//>
    <${Field} label="Top-p" hint="Only the likeliest words that add up to this share are picked from. Empty uses the model's own.">
      <input class="input narrow" type="number" min="0" max="1" step="0.05" value=${c.topP ?? ""} placeholder="model's"
        onChange=${e => num("topP", e.target.value, 0, 1)} /><//>
  <//>`;
}
