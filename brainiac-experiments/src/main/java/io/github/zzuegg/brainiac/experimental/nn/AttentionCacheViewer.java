package io.github.zzuegg.brainiac.experimental.nn;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Writes a self-contained HTML page that shows, for every token fed to an
 * {@link EvictingKvCache} with recording on, which earlier tokens it read (a heat map row of
 * attention weights over all positions) and what the cache had kept or dropped at that
 * moment: cells of evicted tokens are hatched, the token dropped at a step is outlined.
 * Hovering a cell shows the exact weight. No scripts or external files; light and dark mode.
 */
public final class AttentionCacheViewer {

    private AttentionCacheViewer() {
    }

    /**
     * Renders recorded steps.
     *
     * @param title the page title
     * @param steps the steps, from {@link EvictingKvCache#steps()}
     * @return the HTML page
     */
    public static String html(String title, List<EvictingKvCache.Step> steps) {
        Map<Integer, String> labels = new HashMap<>();
        int maxPos = 0;
        for (EvictingKvCache.Step s : steps) {
            labels.put(s.position(), s.label());
            maxPos = Math.max(maxPos, s.position());
        }
        int columns = maxPos + 1;
        StringBuilder b = new StringBuilder(4096);
        b.append("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">")
                .append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
                .append("<title>").append(escape(title)).append("</title>\n<style>\n")
                .append(":root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--grid:#e4e4df;--hot:31,119,180;")
                .append("--gone:#d0d0ca;--drop:#c2410c}\n")
                .append("@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecece8;--muted:#9a9a93;")
                .append("--grid:#2a2a28;--hot:94,166,224;--gone:#3a3a37;--drop:#fb923c}}\n")
                .append("body{margin:0;padding:16px;background:var(--bg);color:var(--fg);")
                .append("font:14px/1.4 system-ui,sans-serif}\n")
                .append("h1{font-size:18px;margin:0 0 4px}p{color:var(--muted);margin:0 0 12px}\n")
                .append(".wrap{overflow-x:auto;max-width:100%}\n")
                .append("table{border-collapse:collapse;font-size:11px}\n")
                .append("th{font-weight:500;color:var(--muted);padding:2px 4px;white-space:nowrap}\n")
                .append("thead th{writing-mode:vertical-rl;transform:rotate(180deg);height:60px}\n")
                .append("td{width:14px;height:14px;padding:0;border:1px solid var(--grid)}\n")
                .append("td.gone{background:repeating-linear-gradient(45deg,var(--gone) 0 2px,transparent 2px 5px)}\n")
                .append("td.drop{outline:2px solid var(--drop);outline-offset:-2px}\n")
                .append("td.self{border-color:var(--fg)}\n")
                .append("tr:hover th{color:var(--fg)}\n")
                .append(".legend{display:flex;gap:16px;margin:0 0 12px;color:var(--muted);flex-wrap:wrap}\n")
                .append(".legend span{display:inline-flex;align-items:center;gap:6px}\n")
                .append(".legend i{display:inline-block;width:14px;height:14px;border:1px solid var(--grid)}\n")
                .append("</style></head><body>\n<h1>").append(escape(title)).append("</h1>\n")
                .append("<p>Each row is one new token; each column an earlier token. Colour is the attention ")
                .append("weight (mean over heads).</p>\n<div class=\"legend\">")
                .append("<span><i style=\"background:rgba(var(--hot),0.9)\"></i>attended</span>")
                .append("<span><i class=\"gone\" style=\"background:repeating-linear-gradient(45deg,var(--gone) 0 2px,")
                .append("transparent 2px 5px)\"></i>evicted earlier</span>")
                .append("<span><i style=\"outline:2px solid var(--drop);outline-offset:-2px\"></i>dropped now</span>")
                .append("</div>\n<div class=\"wrap\"><table>\n<thead><tr><th></th>");
        for (int p = 0; p < columns; p++) {
            b.append("<th>").append(escape(labels.getOrDefault(p, Integer.toString(p)))).append("</th>");
        }
        b.append("</tr></thead>\n<tbody>\n");
        Set<Integer> gone = new HashSet<>();
        for (EvictingKvCache.Step s : steps) {
            Map<Integer, Float> w = new HashMap<>();
            float max = 1e-9f;
            for (int i = 0; i < s.keys().length; i++) {
                w.put(s.keys()[i], s.weights()[i]);
                max = Math.max(max, s.weights()[i]);
            }
            Set<Integer> now = new HashSet<>();
            for (int e : s.evicted()) {
                now.add(e);
            }
            b.append("<tr><th>").append(escape(s.label())).append("</th>");
            for (int p = 0; p < columns; p++) {
                if (p > s.position()) {
                    b.append("<td></td>");
                    continue;
                }
                Float weight = w.get(p);
                StringBuilder cls = new StringBuilder();
                if (weight == null && (gone.contains(p) || now.contains(p))) {
                    cls.append("gone");
                }
                if (now.contains(p)) {
                    cls.append(cls.isEmpty() ? "" : " ").append("drop");
                }
                if (p == s.position()) {
                    cls.append(cls.isEmpty() ? "" : " ").append("self");
                }
                b.append("<td");
                if (!cls.isEmpty()) {
                    b.append(" class=\"").append(cls).append('"');
                }
                if (weight != null) {
                    b.append(String.format(Locale.ROOT, " style=\"background:rgba(var(--hot),%.3f)\"",
                            Math.sqrt(weight / max)));
                    b.append(String.format(Locale.ROOT, " title=\"%s → %s: %.4f\"", escape(s.label()),
                            escape(labels.getOrDefault(p, Integer.toString(p))), weight));
                } else {
                    b.append(" title=\"evicted\"");
                }
                b.append("></td>");
            }
            b.append("</tr>\n");
            gone.addAll(now);
        }
        b.append("</tbody></table></div>\n</body></html>\n");
        return b.toString();
    }

    /**
     * Writes the page to a file.
     *
     * @param path the file
     * @param title the page title
     * @param steps the recorded steps
     * @throws IOException if writing fails
     */
    public static void write(Path path, String title, List<EvictingKvCache.Step> steps) throws IOException {
        Files.writeString(path, html(title, steps), StandardCharsets.UTF_8);
    }

    private static String escape(String s) {
        StringBuilder b = new StringBuilder(s.length());
        for (char c : s.toCharArray()) {
            switch (c) {
                case '<' -> b.append("&lt;");
                case '>' -> b.append("&gt;");
                case '&' -> b.append("&amp;");
                case '"' -> b.append("&quot;");
                default -> b.append(c);
            }
        }
        return b.toString();
    }
}
