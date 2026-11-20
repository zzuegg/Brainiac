// Small text helpers shared by every screen.

const minute = 60_000, hour = 60 * minute, day = 24 * hour;

/** "just now", "5 min", "3 h", "yesterday", "12 Mar". */
export function when(ms) {
  if (!ms) return "";
  const d = Date.now() - ms;
  if (d < minute) return "just now";
  if (d < hour) return `${Math.floor(d / minute)} min`;
  if (d < day) return `${Math.floor(d / hour)} h`;
  if (d < 2 * day) return "yesterday";
  const date = new Date(ms);
  const sameYear = date.getFullYear() === new Date().getFullYear();
  return date.toLocaleDateString(undefined, sameYear ? { day: "numeric", month: "short" } : { day: "numeric", month: "short", year: "numeric" });
}

/** "in 29 days", for when something in the trash goes for good. */
export function until(ms) {
  const d = ms - Date.now();
  if (d <= 0) return "soon";
  if (d < day) return `in ${Math.max(1, Math.round(d / hour))} h`;
  return `in ${Math.round(d / day)} days`;
}

/** A project id from a name: lowercase letters, digits and dashes, at most 40. */
export function slug(name) {
  const s = name.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40).replace(/-+$/, "");
  return s || "project";
}

export const plural = (n, one, many = one + "s") => `${n} ${n === 1 ? one : many}`;
