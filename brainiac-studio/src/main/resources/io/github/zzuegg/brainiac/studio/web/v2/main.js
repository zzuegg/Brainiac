// Brainiac Studio v2: the page's entry point. It loads the projects, reopens the tabs this
// browser had (or the one the address names) and draws the shell.
import { html, render } from "/v2/lib/html.js";
import { App } from "/v2/shell/App.js";
import { loadProjects } from "/v2/state/projects.js";
import { restore } from "/v2/state/workspace.js";
import { fail } from "/v2/state/toasts.js";
import "/v2/state/prefs.js";

async function start() {
  let fallback = "default";
  try {
    const list = await loadProjects();
    fallback = list[0]?.id || "default";
  } catch (e) {
    fail(e);
  }
  restore(fallback);
  render(html`<${App} />`, document.getElementById("app"));
}

start();
