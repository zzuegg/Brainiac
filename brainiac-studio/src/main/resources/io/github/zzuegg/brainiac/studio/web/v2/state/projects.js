import { signal, computed } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";

// The projects, shared by every tab. A tab names its project; this says what it is called.

export const projects = signal([]);
export const loaded = signal(false);
export const byId = computed(() => new Map(projects.value.map(p => [p.id, p])));

export async function loadProjects() {
  const r = await api("/api/projects");
  projects.value = r.projects || [];
  loaded.value = true;
  return projects.value;
}

export const projectName = id => byId.value.get(id)?.name ?? id;
