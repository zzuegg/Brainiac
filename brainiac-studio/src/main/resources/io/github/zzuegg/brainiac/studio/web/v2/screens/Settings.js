import { html } from "/v2/lib/html.js";
import { Choices, Field } from "/v2/ui/Wizard.js";
import { theme, readingWidth } from "/v2/state/prefs.js";
import { Button } from "/v2/ui/Button.js";
import { navigate } from "/v2/state/workspace.js";

// Settings for every project: how the page looks. Models, keys and machines have pages of their own.

export function Settings({ tab }) {
  return html`<div class="page page-narrow">
    <header class="page-head"><div><p class="eyebrow">Studio</p><h1 class="page-title">Studio settings</h1><p class="page-lead">For every project. A project’s own folder, models and tools are in its Project settings.</p></div></header>
    <section class="card">
      <h2 class="card-title">Appearance</h2>
      <${Field} label="Theme"><${Choices} name="Theme" value=${theme.value} onChange=${v => (theme.value = v)} options=${[
        { value: "system", label: "Like the system" }, { value: "light", label: "Light" }, { value: "dark", label: "Dark" },
      ]} /><//>
      <${Field} label="Reading width" hint="How wide chats and agent transcripts are; on a wide screen the rest stays free for side panels.">
        <${Choices} name="Reading width" value=${readingWidth.value} onChange=${v => (readingWidth.value = v)} options=${[
          { value: "narrow", label: "Narrow", text: "640 px" }, { value: "normal", label: "Normal", text: "760 px" },
          { value: "wide", label: "Wide", text: "960 px" }, { value: "full", label: "Full", text: "the whole width" },
        ]} /><//>
    </section>
    <section class="card">
      <h2 class="card-title">Models, machines and agents</h2>
      <p class="muted">Each has a page of its own, also in the sidebar.</p>
      <div class="row-btns">
        <${Button} icon="layers" onClick=${() => navigate(tab, { screen: "models" })}>Models and API keys<//>
        <${Button} icon="server" onClick=${() => navigate(tab, { screen: "machines" })}>Machines and GPU memory<//>
        <${Button} icon="agent" onClick=${() => navigate(tab, { screen: "agents" })}>Agent roles, rules and services<//>
      </div>
    </section>
  </div>`;
}
