import { html } from "/v2/lib/html.js";
import { Choices, Field } from "/v2/ui/Wizard.js";
import { theme, readingWidth } from "/v2/state/prefs.js";
import { Icon } from "/v2/ui/Icon.js";

// Settings for this page. Studio-wide setup (providers, keys, machines) moves here with the
// Placement step; until then it lives in the classic page.

export function Settings() {
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
      <h2 class="card-title">Models, keys and machines</h2>
      <p class="muted">These still live in the classic Studio while this one grows.</p>
      <a class="btn" href="/" target="_blank" rel="noopener"><${Icon} name="classic" /><span>Open classic Studio</span></a>
    </section>
  </div>`;
}
