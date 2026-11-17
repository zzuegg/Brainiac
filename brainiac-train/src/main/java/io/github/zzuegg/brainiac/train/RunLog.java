package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.io.Json;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Records a training run in a folder, for comparing runs later: {@code config.json} with the
 * settings it was started with, {@code metrics.jsonl} with one line per logged point,
 * {@code metrics.csv} with the same points as a table, and {@code dashboard.html}, a page with
 * one chart per metric that opens in any browser without a server.
 *
 * <p>Add it to a {@link Trainer} as a {@link TrainingListener} to log every epoch (and, with
 * {@link #everyBatches}, batch losses), or call {@link #log} from your own loop.
 *
 * <pre>{@code
 * RunLog run = RunLog.create(Path.of("runs/mlp-adam"), Map.of("lr", 0.001, "layers", 3));
 * Trainer.of(model).listener(run).fit(data);
 * run.close();   // writes the dashboard
 * }</pre>
 */
public final class RunLog implements TrainingListener, AutoCloseable {

    private final Path dir;
    private final BufferedWriter jsonl;
    private final List<Map<String, Object>> points = new ArrayList<>();
    private final long started = System.nanoTime();
    private int batchEvery;

    private RunLog(Path dir, Map<String, ?> config) throws IOException {
        this.dir = dir;
        Files.createDirectories(dir);
        Map<String, Object> c = new LinkedHashMap<>();
        c.put("started", Instant.now().toString());
        c.put("config", new TreeMap<>(config));
        Files.writeString(dir.resolve("config.json"), Json.write(c), StandardCharsets.UTF_8);
        jsonl = Files.newBufferedWriter(dir.resolve("metrics.jsonl"), StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    /**
     * Starts a run log; existing files in the folder are replaced.
     *
     * @param dir the folder, created if needed
     * @param config the run's settings: numbers, strings, booleans, lists or maps
     * @return the log
     */
    public static RunLog create(Path dir, Map<String, ?> config) {
        try {
            return new RunLog(dir, config);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Also logs the loss of every {@code n}-th batch, as metric {@code batch_loss}.
     *
     * @param n the interval in steps; 0 turns it off
     * @return this log
     */
    public RunLog everyBatches(int n) {
        this.batchEvery = n;
        return this;
    }

    /**
     * Logs values at a step.
     *
     * @param step the step: an epoch, an optimizer step or anything increasing
     * @param values the metrics by name
     */
    public synchronized void log(long step, Map<String, ? extends Number> values) {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("step", step);
        p.put("seconds", (System.nanoTime() - started) / 1e9);
        for (Map.Entry<String, ? extends Number> e : values.entrySet()) {
            double v = e.getValue().doubleValue();
            p.put(e.getKey(), Double.isFinite(v) ? v : null);
        }
        points.add(p);
        try {
            jsonl.write(Json.write(p));
            jsonl.write('\n');
            jsonl.flush();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void onEpoch(EpochResult r) {
        Map<String, Number> v = new LinkedHashMap<>();
        v.put("loss", r.loss());
        r.metrics().forEach(v::put);
        if (r.hasValidation()) {
            v.put("val_loss", r.validationLoss());
            r.validationMetrics().forEach((k, x) -> v.put("val_" + k, x));
        }
        v.put("lr", r.learningRate());
        log(r.epoch(), v);
    }

    @Override
    public void onBatch(int epoch, long step, float loss) {
        if (batchEvery > 0 && step % batchEvery == 0) {
            log(step, Map.of("batch_loss", loss));
        }
    }

    /** @return the logged points, oldest first */
    public synchronized List<Map<String, Object>> points() {
        return List.copyOf(points);
    }

    /**
     * Writes {@code metrics.csv} and {@code dashboard.html} and closes the JSONL file.
     */
    @Override
    public synchronized void close() {
        try {
            jsonl.close();
            List<String> names = names();
            StringBuilder csv = new StringBuilder("step,seconds");
            for (String n : names) {
                csv.append(',').append(n);
            }
            csv.append('\n');
            for (Map<String, Object> p : points) {
                csv.append(p.get("step")).append(',').append(p.get("seconds"));
                for (String n : names) {
                    Object v = p.get(n);
                    csv.append(',').append(v == null ? "" : v);
                }
                csv.append('\n');
            }
            Files.writeString(dir.resolve("metrics.csv"), csv, StandardCharsets.UTF_8);
            Files.writeString(dir.resolve("dashboard.html"), dashboard(names), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private List<String> names() {
        List<String> names = new ArrayList<>();
        for (Map<String, Object> p : points) {
            for (String k : p.keySet()) {
                if (!k.equals("step") && !k.equals("seconds") && !names.contains(k)) {
                    names.add(k);
                }
            }
        }
        return names;
    }

    /** One SVG line chart per metric, data inlined. */
    private String dashboard(List<String> names) throws IOException {
        StringBuilder h = new StringBuilder();
        h.append("<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
                + "content=\"width=device-width, initial-scale=1\"><title>")
                .append(escape(dir.getFileName().toString())).append("</title><style>")
                .append(":root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#2563eb;--grid:#e5e5ea}")
                .append("@media (prefers-color-scheme: dark){:root{--bg:#1c1c1e;--fg:#f2f2f7;--muted:#98989d;"
                        + "--line:#60a5fa;--grid:#3a3a3c}}")
                .append("body{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:16px}")
                .append(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}")
                .append("figure{margin:0}figcaption{font-weight:600;margin-bottom:4px}svg{width:100%;height:auto}")
                .append("pre{color:var(--muted);white-space:pre-wrap}</style></head><body><h1>")
                .append(escape(dir.getFileName().toString())).append("</h1><pre>")
                .append(escape(Files.readString(dir.resolve("config.json")))).append("</pre><div class=\"grid\">");
        for (String n : names) {
            List<double[]> xy = new ArrayList<>();
            for (Map<String, Object> p : points) {
                if (p.get(n) instanceof Number v) {
                    xy.add(new double[] {((Number) p.get("step")).doubleValue(), v.doubleValue()});
                }
            }
            h.append("<figure><figcaption>").append(escape(n)).append("</figcaption>").append(chart(xy))
                    .append("</figure>");
        }
        return h.append("</div></body></html>\n").toString();
    }

    private static String chart(List<double[]> xy) {
        int w = 320;
        int ht = 180;
        int pad = 36;
        if (xy.isEmpty()) {
            return "<p>no values</p>";
        }
        double x0 = xy.getFirst()[0];
        double x1 = xy.getLast()[0];
        double y0 = Double.POSITIVE_INFINITY;
        double y1 = Double.NEGATIVE_INFINITY;
        for (double[] p : xy) {
            y0 = Math.min(y0, p[1]);
            y1 = Math.max(y1, p[1]);
        }
        if (x1 == x0) {
            x1 = x0 + 1;
        }
        if (y1 == y0) {
            y1 = y0 + 1;
        }
        StringBuilder s = new StringBuilder("<svg viewBox=\"0 0 " + w + " " + ht + "\" role=\"img\">");
        s.append("<line x1=\"").append(pad).append("\" y1=\"").append(ht - pad).append("\" x2=\"").append(w - 8)
                .append("\" y2=\"").append(ht - pad).append("\" stroke=\"var(--grid)\"/>");
        s.append(label(4, 14, y1)).append(label(4, ht - pad, y0)).append(label(pad, ht - 8, x0))
                .append(label(w - 60, ht - 8, x1));
        s.append("<polyline fill=\"none\" stroke=\"var(--line)\" stroke-width=\"2\" points=\"");
        for (double[] p : xy) {
            double x = pad + (p[0] - x0) / (x1 - x0) * (w - pad - 8);
            double y = 8 + (1 - (p[1] - y0) / (y1 - y0)) * (ht - pad - 8);
            s.append(String.format(java.util.Locale.ROOT, "%.1f,%.1f ", x, y));
        }
        return s.append("\"/></svg>").toString();
    }

    private static String label(int x, int y, double v) {
        return "<text x=\"" + x + "\" y=\"" + y + "\" font-size=\"10\" fill=\"var(--muted)\">"
                + String.format(java.util.Locale.ROOT, "%.4g", v) + "</text>";
    }

    private static String escape(String s) {
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
    }
}
