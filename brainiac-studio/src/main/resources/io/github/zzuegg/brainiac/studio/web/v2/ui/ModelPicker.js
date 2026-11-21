import { html } from "/v2/lib/html.js";
import { chatModels, autoModel, modelByKey, placeOf, modelInfo } from "/v2/state/models.js";

// Picks a chat model: Auto first (saying what it picks now), then the models grouped by where
// they run. A model chosen before that is gone stays listed, so the choice is not lost.

export function ModelPicker({ value = "auto", onChange, disabled, label = "Model" }) {
  const groups = new Map();
  for (const m of chatModels.value) {
    const g = placeOf(m);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(m);
  }
  const auto = autoModel.value;
  const known = value === "auto" || modelByKey(value);
  return html`<select class="input select-sm model-pick" value=${value} disabled=${disabled} aria-label=${label}
      title=${value === "auto" ? (auto ? "Auto picks " + auto.name : "Auto picks a model") : value}
      onChange=${e => onChange(e.target.value)}>
    <option value="auto">${auto ? "Auto · " + auto.name : modelInfo.value ? "Auto" : "Auto (looking for models…)"}</option>
    ${!known && html`<option value=${value}>${value} (not found)</option>`}
    ${[...groups].map(([g, ms]) => html`<optgroup label=${g}>${ms.map(m => html`<option value=${m.key}>${m.name}</option>`)}</optgroup>`)}
  </select>`;
}
