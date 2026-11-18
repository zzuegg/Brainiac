package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.llm.Tools;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

/**
 * Tools for tables and charts in the project's folder, and the project's content calendar.
 */
public final class DataTools implements RiskRefiner, Checkpointed {

    /** The most bytes of a table that are read. */
    static final long MAX_TABLE = 64L << 20;

    /** The calendar's file in the project's folder. */
    static final String CALENDAR = "content-calendar.json";

    private volatile Checkpoints checkpoints;

    /** Creates the tools. */
    public DataTools() {
    }

    @Override
    public void checkpoints(Checkpoints c) {
        this.checkpoints = c;
    }

    @Override
    public Risk refine(String tool, Map<String, Object> arguments, Risk declared) {
        try {
            if (tool.equals("content_calendar")) {
                String action = String.valueOf(arguments.getOrDefault("action", "list")).toLowerCase(Locale.ROOT);
                return action.equals("list") ? Risk.READ : Risk.CREATE;
            }
            if (tool.equals("make_chart")) {
                Object path = arguments.get("path");
                Path f = path == null ? null : Reach.locate(svg(String.valueOf(path)));
                return f != null && Files.exists(f) ? Risk.WRITE : Risk.CREATE;
            }
        } catch (RuntimeException e) {
            return declared;
        }
        return declared;
    }

    @Override
    public List<Reach.Access> reach(String tool, Map<String, Object> arguments) {
        return switch (tool) {
            case "query_table" -> Reach.of(arguments, false, "path");
            case "make_chart" -> {
                Map<String, Object> a = new LinkedHashMap<>(arguments);
                if (a.get("path") instanceof String p) {
                    a.put("path", svg(p));
                }
                yield Reach.of(a, true, "path");
            }
            default -> List.of();
        };
    }

    // ---- tables

    /** A condition on a column. */
    public record Filter(@Tools.Param("the column's name") String column,
            @Tools.Param("one of = != < <= > >= contains startswith empty notempty") String op,
            @Tools.Param(value = "the value to compare with", optional = true) String value) {
    }

    /** A sum or the like over a column. */
    public record Aggregate(@Tools.Param("count, sum, avg, min, max or distinct") String op,
            @Tools.Param(value = "the column; not needed for count", optional = true) String column) {
    }

    /**
     * Queries a table.
     *
     * @param path the table
     * @param where conditions
     * @param columns which columns
     * @param groupBy grouping columns
     * @param aggregates per group
     * @param sort the order
     * @param limit how many rows
     * @param describe whether to describe the columns
     * @return the rows
     */
    @Tools.Tool(name = "query_table", value = "Reads a CSV or TSV table in the project's folder: filters rows, picks "
            + "columns, groups and sums, sorts. With describe=true it says each column's type, range and blanks "
            + "instead. Use it before guessing at numbers.")
    @Tier(value = Risk.READ, resultChars = 14_000)
    public Map<String, Object> queryTable(@Tools.Param("the table, inside the project's folder") String path,
            @Tools.Param("rows must meet all of these") Optional<List<Filter>> where,
            @Tools.Param("the columns to give; all if left out") Optional<List<String>> columns,
            @Tools.Param("columns to group by") Optional<List<String>> groupBy,
            @Tools.Param("per group, such as {\"op\": \"sum\", \"column\": \"sales\"}; a count if grouped and left out")
            Optional<List<Aggregate>> aggregates,
            @Tools.Param("the column to sort by, with '-' before it for largest first") Optional<String> sort,
            @Tools.Param("how many rows to give, 50 if left out, at most 500") Optional<Integer> limit,
            @Tools.Param("true to describe the columns instead") Optional<Boolean> describe) {
        Table t = read(Reach.file(path));
        if (describe.orElse(false)) {
            return describe(t);
        }
        List<List<String>> rows = new ArrayList<>();
        for (List<String> r : t.rows()) {
            boolean ok = true;
            for (Filter f : where.orElse(List.of())) {
                ok &= test(r.get(t.index(f.column())), f);
            }
            if (ok) {
                rows.add(r);
            }
        }
        List<String> header;
        List<List<String>> out;
        if (groupBy.isPresent() && !groupBy.get().isEmpty() || aggregates.isPresent() && !aggregates.get().isEmpty()) {
            List<String> keys = groupBy.orElse(List.of());
            List<Aggregate> aggs = aggregates.filter(a -> !a.isEmpty()).orElse(List.of(new Aggregate("count", null)));
            Map<List<String>, List<List<String>>> groups = new LinkedHashMap<>();
            for (List<String> r : rows) {
                List<String> k = new ArrayList<>();
                for (String c : keys) {
                    k.add(r.get(t.index(c)));
                }
                groups.computeIfAbsent(k, x -> new ArrayList<>()).add(r);
            }
            header = new ArrayList<>(keys);
            for (Aggregate a : aggs) {
                header.add(a.op().toLowerCase(Locale.ROOT) + (a.column() == null || a.column().isBlank() ? "" : "(" + a.column() + ")"));
            }
            out = new ArrayList<>();
            for (Map.Entry<List<String>, List<List<String>>> g : groups.entrySet()) {
                List<String> line = new ArrayList<>(g.getKey());
                for (Aggregate a : aggs) {
                    line.add(aggregate(t, g.getValue(), a));
                }
                out.add(line);
            }
        } else {
            List<String> cols = columns.filter(c -> !c.isEmpty()).orElse(t.header());
            header = cols;
            out = new ArrayList<>();
            for (List<String> r : rows) {
                List<String> line = new ArrayList<>();
                for (String c : cols) {
                    line.add(r.get(t.index(c)));
                }
                out.add(line);
            }
        }
        if (sort.isPresent() && !sort.get().isBlank()) {
            String s = sort.get().strip();
            boolean desc = s.startsWith("-");
            String col = desc ? s.substring(1) : s;
            int i = header.indexOf(col);
            if (i < 0) {
                throw new IllegalArgumentException("cannot sort by " + col + ": the result's columns are " + header);
            }
            Comparator<List<String>> cmp = (a, b) -> compare(a.get(i), b.get(i));
            out.sort(desc ? cmp.reversed() : cmp);
        }
        int n = Math.clamp(limit.orElse(50), 1, 500);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("columns", header);
        result.put("rows", out.subList(0, Math.min(n, out.size())));
        result.put("matched", out.size());
        if (out.size() > n) {
            result.put("note", "the first " + n + " of " + out.size() + " rows");
        }
        return result;
    }

    /** A table read. */
    record Table(List<String> header, List<List<String>> rows) {
        int index(String column) {
            int i = header.indexOf(column);
            if (i < 0) {
                for (int k = 0; k < header.size(); k++) {
                    if (header.get(k).equalsIgnoreCase(column)) {
                        return k;
                    }
                }
                throw new IllegalArgumentException("there is no column " + column + "; the columns are " + header);
            }
            return i;
        }
    }

    static Table read(Path f) {
        if (!Files.isRegularFile(f)) {
            throw new IllegalArgumentException("there is no table " + f.getFileName() + " in the project's folder");
        }
        try {
            if (Files.size(f) > MAX_TABLE) {
                throw new IllegalArgumentException("the table is over " + (MAX_TABLE >> 20) + " MB");
            }
            String text = Files.readString(f, StandardCharsets.UTF_8);
            if (text.startsWith("﻿")) {
                text = text.substring(1);
            }
            String first = text.lines().findFirst().orElse("");
            char sep = f.toString().toLowerCase(Locale.ROOT).endsWith(".tsv") || count(first, '\t') > count(first, ',')
                    ? '\t' : count(first, ';') > count(first, ',') ? ';' : ',';
            List<List<String>> all = csv(text, sep);
            if (all.isEmpty()) {
                throw new IllegalArgumentException("the table is empty");
            }
            List<String> header = new ArrayList<>();
            for (String h : all.getFirst()) {
                header.add(h.strip());
            }
            List<List<String>> rows = new ArrayList<>();
            for (List<String> r : all.subList(1, all.size())) {
                if (r.size() == 1 && r.getFirst().isBlank()) {
                    continue;
                }
                List<String> line = new ArrayList<>(r);
                while (line.size() < header.size()) {
                    line.add("");
                }
                rows.add(line.subList(0, header.size()));
            }
            return new Table(header, rows);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    private static int count(String s, char c) {
        return (int) s.chars().filter(x -> x == c).count();
    }

    /**
     * Parses CSV text with quoted fields.
     *
     * @param text the text
     * @param sep the separator
     * @return the rows
     */
    static List<List<String>> csv(String text, char sep) {
        List<List<String>> rows = new ArrayList<>();
        List<String> row = new ArrayList<>();
        StringBuilder field = new StringBuilder();
        boolean quoted = false;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (quoted) {
                if (c == '"') {
                    if (i + 1 < text.length() && text.charAt(i + 1) == '"') {
                        field.append('"');
                        i++;
                    } else {
                        quoted = false;
                    }
                } else {
                    field.append(c);
                }
            } else if (c == '"' && field.isEmpty()) {
                quoted = true;
            } else if (c == sep) {
                row.add(field.toString());
                field.setLength(0);
            } else if (c == '\n' || c == '\r') {
                if (c == '\r' && i + 1 < text.length() && text.charAt(i + 1) == '\n') {
                    i++;
                }
                row.add(field.toString());
                field.setLength(0);
                rows.add(row);
                row = new ArrayList<>();
            } else {
                field.append(c);
            }
        }
        if (!field.isEmpty() || !row.isEmpty()) {
            row.add(field.toString());
            rows.add(row);
        }
        return rows;
    }

    private static Double number(String s) {
        if (s == null) {
            return null;
        }
        String t = s.strip().replace("_", "");
        if (t.isEmpty()) {
            return null;
        }
        try {
            return Double.parseDouble(t);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static int compare(String a, String b) {
        Double x = number(a);
        Double y = number(b);
        if (x != null && y != null) {
            return Double.compare(x, y);
        }
        return a.compareToIgnoreCase(b);
    }

    private static boolean test(String v, Filter f) {
        String op = f.op() == null ? "=" : f.op().strip().toLowerCase(Locale.ROOT);
        String w = f.value() == null ? "" : f.value();
        return switch (op) {
            case "=", "==" -> compare(v, w) == 0;
            case "!=", "<>" -> compare(v, w) != 0;
            case "<" -> compare(v, w) < 0;
            case "<=" -> compare(v, w) <= 0;
            case ">" -> compare(v, w) > 0;
            case ">=" -> compare(v, w) >= 0;
            case "contains" -> v.toLowerCase(Locale.ROOT).contains(w.toLowerCase(Locale.ROOT));
            case "startswith" -> v.toLowerCase(Locale.ROOT).startsWith(w.toLowerCase(Locale.ROOT));
            case "empty" -> v.isBlank();
            case "notempty" -> !v.isBlank();
            default -> throw new IllegalArgumentException("unknown condition " + f.op());
        };
    }

    private static String aggregate(Table t, List<List<String>> rows, Aggregate a) {
        String op = a.op().strip().toLowerCase(Locale.ROOT);
        if (op.equals("count")) {
            return String.valueOf(rows.size());
        }
        if (a.column() == null || a.column().isBlank()) {
            throw new IllegalArgumentException(op + " needs a column");
        }
        int i = t.index(a.column());
        if (op.equals("distinct")) {
            return String.valueOf(rows.stream().map(r -> r.get(i)).distinct().count());
        }
        double sum = 0;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        int n = 0;
        for (List<String> r : rows) {
            Double d = number(r.get(i));
            if (d != null) {
                sum += d;
                min = Math.min(min, d);
                max = Math.max(max, d);
                n++;
            }
        }
        double v = switch (op) {
            case "sum" -> sum;
            case "avg", "mean" -> n == 0 ? Double.NaN : sum / n;
            case "min" -> n == 0 ? Double.NaN : min;
            case "max" -> n == 0 ? Double.NaN : max;
            default -> throw new IllegalArgumentException("unknown aggregate " + a.op() + "; use count, sum, avg, min, max or distinct");
        };
        return fmt(v);
    }

    private static String fmt(double v) {
        if (Double.isNaN(v)) {
            return "";
        }
        if (v == Math.rint(v) && Math.abs(v) < 1e15) {
            return String.valueOf((long) v);
        }
        return String.format(Locale.ROOT, "%.6g", v).replaceAll("(\\.\\d*?)0+(e|$)", "$1$2").replaceAll("\\.(e|$)", "$1");
    }

    private static Map<String, Object> describe(Table t) {
        List<Map<String, Object>> cols = new ArrayList<>();
        for (int i = 0; i < t.header().size(); i++) {
            int blanks = 0;
            int numbers = 0;
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            double sum = 0;
            Map<String, Integer> distinct = new LinkedHashMap<>();
            for (List<String> r : t.rows()) {
                String v = r.get(i);
                if (v.isBlank()) {
                    blanks++;
                    continue;
                }
                Double d = number(v);
                if (d != null) {
                    numbers++;
                    min = Math.min(min, d);
                    max = Math.max(max, d);
                    sum += d;
                }
                if (distinct.size() < 1000) {
                    distinct.merge(v, 1, Integer::sum);
                }
            }
            Map<String, Object> c = new LinkedHashMap<>();
            c.put("name", t.header().get(i));
            int filled = t.rows().size() - blanks;
            boolean numeric = filled > 0 && numbers == filled;
            c.put("type", numeric ? "number" : "text");
            c.put("blanks", blanks);
            if (numeric) {
                c.put("min", fmt(min));
                c.put("max", fmt(max));
                c.put("mean", fmt(sum / numbers));
            } else {
                c.put("distinct", distinct.size() >= 1000 ? "1000+" : distinct.size());
                List<String> top = distinct.entrySet().stream().sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                        .limit(5).map(e -> e.getKey() + " (" + e.getValue() + ")").toList();
                c.put("common", top);
            }
            cols.add(c);
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("rows", t.rows().size());
        out.put("columns", cols);
        return out;
    }

    // ---- charts

    /** One line or set of bars. */
    public record Series(@Tools.Param("the series' name, shown in the legend") String name,
            @Tools.Param("one value per label") List<Double> values) {
    }

    /**
     * Draws a chart.
     *
     * @param path where to save it
     * @param kind bar, line or pie
     * @param title the title
     * @param labels the x labels
     * @param series the values
     * @param yLabel the y axis' name
     * @return the file
     */
    @Tools.Tool(name = "make_chart", value = "Draws a bar, line or pie chart as an SVG file in the project's folder. "
            + "Give the labels along the bottom and one or more series of values, one value per label; a pie takes "
            + "one series.")
    @Tier(value = Risk.CREATE, what = "draws a chart into the project's folder")
    public Map<String, Object> makeChart(@Tools.Param("the file to save, such as 'charts/sales.svg'") String path,
            @Tools.Param("bar, line or pie") String kind,
            @Tools.Param("the chart's title") Optional<String> title,
            @Tools.Param("the labels along the bottom, or the slices of a pie") List<String> labels,
            @Tools.Param("the values") List<Series> series,
            @Tools.Param("what the values are, such as 'sales in EUR'") Optional<String> yLabel) {
        String k = kind == null ? "bar" : kind.strip().toLowerCase(Locale.ROOT);
        if (!List.of("bar", "line", "pie").contains(k)) {
            throw new IllegalArgumentException("a chart is bar, line or pie, not " + kind);
        }
        if (labels == null || labels.isEmpty() || series == null || series.isEmpty()) {
            throw new IllegalArgumentException("give labels and at least one series");
        }
        if (labels.size() > 200 || series.size() > 12) {
            throw new IllegalArgumentException("a chart takes at most 200 labels and 12 series");
        }
        for (Series s : series) {
            if (s.values() == null || s.values().size() != labels.size()) {
                throw new IllegalArgumentException("the series " + s.name() + " has " + (s.values() == null ? 0 : s.values().size())
                        + " values for " + labels.size() + " labels");
            }
        }
        String svg = k.equals("pie") ? pie(title.orElse(""), labels, series.getFirst())
                : xy(k, title.orElse(""), labels, series, yLabel.orElse(""));
        Path f = Reach.file(svg(path));
        try {
            Checkpointed.before(checkpoints, f);
            Files.createDirectories(f.getParent());
            Files.writeString(f, svg, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("file", svg(path));
        out.put("kind", k);
        out.put("bytes", svg.length());
        return out;
    }

    private static String svg(String path) {
        return path.toLowerCase(Locale.ROOT).endsWith(".svg") ? path : path + ".svg";
    }

    private static final String[] COLORS = {"#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948",
            "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#1f77b4", "#2ca02c"};

    static String xy(String kind, String title, List<String> labels, List<Series> series, String yLabel) {
        int w = 800;
        int h = 480;
        int left = 70;
        int right = 20;
        int top = title.isBlank() ? 20 : 50;
        int bottom = 90;
        double lo = 0;
        double hi = 0;
        for (Series s : series) {
            for (Double v : s.values()) {
                if (v != null && Double.isFinite(v)) {
                    lo = Math.min(lo, v);
                    hi = Math.max(hi, v);
                }
            }
        }
        if (hi == lo) {
            hi = lo + 1;
        }
        double step = nice((hi - lo) / 5);
        lo = Math.floor(lo / step) * step;
        hi = Math.ceil(hi / step) * step;
        double plotW = w - left - right;
        double plotH = h - top - bottom;
        double finalLo = lo;
        double finalHi = hi;
        java.util.function.DoubleUnaryOperator y = v -> top + plotH * (1 - (v - finalLo) / (finalHi - finalLo));
        StringBuilder b = head(w, h, title);
        for (double t = lo; t <= hi + step / 2; t += step) {
            double yy = y.applyAsDouble(t);
            b.append(String.format(Locale.ROOT, "<line x1='%d' x2='%d' y1='%.1f' y2='%.1f' stroke='#ddd'/>", left, w - right, yy, yy));
            b.append(String.format(Locale.ROOT, "<text x='%d' y='%.1f' text-anchor='end' font-size='12' fill='#555'>%s</text>",
                    left - 6, yy + 4, esc(fmt(t))));
        }
        if (!yLabel.isBlank()) {
            b.append(String.format(Locale.ROOT, "<text transform='translate(16,%.1f) rotate(-90)' text-anchor='middle' "
                    + "font-size='13' fill='#333'>%s</text>", top + plotH / 2, esc(yLabel)));
        }
        int n = labels.size();
        double slot = plotW / n;
        int every = Math.max(1, (int) Math.ceil(n / 30.0));
        for (int i = 0; i < n; i += every) {
            double x = left + slot * (i + 0.5);
            b.append(String.format(Locale.ROOT, "<text transform='translate(%.1f,%d) rotate(-35)' text-anchor='end' "
                    + "font-size='12' fill='#555'>%s</text>", x, h - bottom + 16, esc(cut(labels.get(i), 18))));
        }
        double zero = y.applyAsDouble(Math.max(lo, Math.min(hi, 0)));
        if (kind.equals("bar")) {
            double bw = slot * 0.8 / series.size();
            for (int s = 0; s < series.size(); s++) {
                for (int i = 0; i < n; i++) {
                    Double v = series.get(s).values().get(i);
                    if (v == null || !Double.isFinite(v)) {
                        continue;
                    }
                    double x = left + slot * i + slot * 0.1 + bw * s;
                    double yy = y.applyAsDouble(v);
                    b.append(String.format(Locale.ROOT, "<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' fill='%s'>"
                            + "<title>%s: %s</title></rect>", x, Math.min(yy, zero), Math.max(1, bw - 1), Math.abs(zero - yy),
                            COLORS[s % COLORS.length], esc(labels.get(i)), esc(fmt(v))));
                }
            }
        } else {
            for (int s = 0; s < series.size(); s++) {
                StringBuilder pts = new StringBuilder();
                for (int i = 0; i < n; i++) {
                    Double v = series.get(s).values().get(i);
                    if (v == null || !Double.isFinite(v)) {
                        continue;
                    }
                    pts.append(String.format(Locale.ROOT, "%.1f,%.1f ", left + slot * (i + 0.5), y.applyAsDouble(v)));
                }
                b.append(String.format(Locale.ROOT, "<polyline points='%s' fill='none' stroke='%s' stroke-width='2.5'/>",
                        pts.toString().strip(), COLORS[s % COLORS.length]));
            }
        }
        b.append(String.format(Locale.ROOT, "<line x1='%d' x2='%d' y1='%.1f' y2='%.1f' stroke='#888'/>", left, w - right, zero, zero));
        if (series.size() > 1) {
            legend(b, series.stream().map(Series::name).toList(), left, h - 18);
        }
        return b.append("</svg>\n").toString();
    }

    static String pie(String title, List<String> labels, Series s) {
        int w = 640;
        int h = 480;
        double total = 0;
        for (Double v : s.values()) {
            total += v == null || v < 0 ? 0 : v;
        }
        if (total <= 0) {
            throw new IllegalArgumentException("a pie needs positive values");
        }
        StringBuilder b = head(w, h, title);
        double cx = 220;
        double cy = h / 2.0 + 10;
        double r = 170;
        double a = -Math.PI / 2;
        for (int i = 0; i < labels.size(); i++) {
            Double v = s.values().get(i);
            if (v == null || v <= 0) {
                continue;
            }
            double sweep = 2 * Math.PI * v / total;
            String color = COLORS[i % COLORS.length];
            if (sweep >= 2 * Math.PI - 1e-9) {
                b.append(String.format(Locale.ROOT, "<circle cx='%.1f' cy='%.1f' r='%.1f' fill='%s'/>", cx, cy, r, color));
            } else {
                double x1 = cx + r * Math.cos(a);
                double y1 = cy + r * Math.sin(a);
                double x2 = cx + r * Math.cos(a + sweep);
                double y2 = cy + r * Math.sin(a + sweep);
                b.append(String.format(Locale.ROOT, "<path d='M%.1f,%.1f L%.1f,%.1f A%.1f,%.1f 0 %d 1 %.1f,%.1f Z' "
                        + "fill='%s' stroke='#fff'><title>%s: %s</title></path>", cx, cy, x1, y1, r, r,
                        sweep > Math.PI ? 1 : 0, x2, y2, color, esc(labels.get(i)), esc(fmt(v))));
            }
            a += sweep;
        }
        double ly = 70;
        for (int i = 0; i < labels.size() && i < 20; i++) {
            Double v = s.values().get(i);
            b.append(String.format(Locale.ROOT, "<rect x='420' y='%.1f' width='12' height='12' fill='%s'/>"
                    + "<text x='438' y='%.1f' font-size='13' fill='#333'>%s (%.0f%%)</text>", ly, COLORS[i % COLORS.length],
                    ly + 11, esc(cut(labels.get(i), 22)), v == null || v < 0 ? 0 : 100 * v / total));
            ly += 20;
        }
        return b.append("</svg>\n").toString();
    }

    private static StringBuilder head(int w, int h, String title) {
        StringBuilder b = new StringBuilder();
        b.append(String.format(Locale.ROOT, "<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d' viewBox='0 0 %d %d' "
                + "font-family='system-ui, sans-serif'><rect width='100%%' height='100%%' fill='#fff'/>", w, h, w, h));
        if (!title.isBlank()) {
            b.append(String.format(Locale.ROOT, "<text x='%d' y='30' text-anchor='middle' font-size='18' font-weight='600' "
                    + "fill='#222'>%s</text>", w / 2, esc(title)));
        }
        return b;
    }

    private static void legend(StringBuilder b, List<String> names, int x, int y) {
        double at = x;
        for (int i = 0; i < names.size(); i++) {
            b.append(String.format(Locale.ROOT, "<rect x='%.1f' y='%d' width='12' height='12' fill='%s'/>"
                    + "<text x='%.1f' y='%d' font-size='12' fill='#333'>%s</text>", at, y - 10, COLORS[i % COLORS.length],
                    at + 16, y, esc(cut(names.get(i), 20))));
            at += 30 + 7 * Math.min(20, names.get(i).length());
        }
    }

    private static double nice(double raw) {
        double mag = Math.pow(10, Math.floor(Math.log10(raw)));
        double f = raw / mag;
        return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * mag;
    }

    private static String cut(String s, int n) {
        return s.length() > n ? s.substring(0, n - 1) + "…" : s;
    }

    static String esc(String s) {
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("'", "&#39;").replace("\"", "&quot;");
    }

    // ---- content calendar

    /** An entry of the content calendar. */
    public record Entry(@Tools.Param(value = "the entry's id; needed to update or remove", optional = true) String id,
            @Tools.Param(value = "the day, as 2026-12-01, or a time, as 2026-12-01T09:00", optional = true) String date,
            @Tools.Param(value = "where it goes, such as mastodon, bluesky, blog or newsletter", optional = true) String channel,
            @Tools.Param(value = "a short title", optional = true) String title,
            @Tools.Param(value = "the post's text or an outline", optional = true) String text,
            @Tools.Param(value = "idea, draft, ready, scheduled or posted", optional = true) String state,
            @Tools.Param(value = "files that go with it, in the project's folder", optional = true) List<String> media,
            @Tools.Param(value = "notes", optional = true) String notes) {
    }

    /**
     * Keeps the content calendar.
     *
     * @param action what to do
     * @param entry the entry
     * @param from the first day listed
     * @param to the last day listed
     * @return the calendar or the entry
     */
    @Tools.Tool(name = "content_calendar", value = "Keeps the project's content calendar: what goes out, where and when. "
            + "'list' shows entries, optionally between two days; 'add' adds one; 'update' changes the fields given of "
            + "the entry with that id; 'remove' drops it. Nothing is posted from here.")
    @Tier(value = Risk.CREATE, what = "changes the project's content calendar")
    public Map<String, Object> contentCalendar(@Tools.Param("list, add, update or remove") String action,
            @Tools.Param("the entry, for add, update or remove") Optional<Entry> entry,
            @Tools.Param("for list: the first day, as 2026-12-01") Optional<String> from,
            @Tools.Param("for list: the last day") Optional<String> to) {
        String a = action == null ? "list" : action.strip().toLowerCase(Locale.ROOT);
        Path f = FileTools.project().inFolder(CALENDAR);
        synchronized (DataTools.class) {
            List<Map<String, Object>> items = calendar(f);
            Map<String, Object> out = new LinkedHashMap<>();
            switch (a) {
                case "list" -> {
                    String lo = from.orElse("");
                    String hi = to.orElse("");
                    List<Map<String, Object>> shown = new ArrayList<>();
                    for (Map<String, Object> m : items) {
                        String d = Json.string(m, "date", "");
                        if ((lo.isBlank() || d.compareTo(lo) >= 0) && (hi.isBlank() || d.isBlank()
                                || d.substring(0, Math.min(d.length(), 10)).compareTo(hi) <= 0)) {
                            shown.add(m);
                        }
                    }
                    out.put("entries", shown);
                    out.put("total", items.size());
                    return out;
                }
                case "add" -> {
                    Entry e = entry.orElseThrow(() -> new IllegalArgumentException("give the entry to add"));
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("id", UUID.randomUUID().toString().substring(0, 8));
                    merge(m, e);
                    m.putIfAbsent("state", "idea");
                    m.put("created", System.currentTimeMillis());
                    items.add(m);
                    out.put("added", m);
                }
                case "update", "remove" -> {
                    Entry e = entry.orElseThrow(() -> new IllegalArgumentException("give the entry's id"));
                    Map<String, Object> m = items.stream().filter(x -> Json.string(x, "id", "").equals(e.id())).findFirst()
                            .orElseThrow(() -> new IllegalArgumentException("there is no entry " + e.id()
                                    + "; list the calendar to see the ids"));
                    if (a.equals("remove")) {
                        items.remove(m);
                        out.put("removed", e.id());
                    } else {
                        merge(m, e);
                        m.put("updated", System.currentTimeMillis());
                        out.put("updated", m);
                    }
                }
                default -> throw new IllegalArgumentException("the action is list, add, update or remove, not " + action);
            }
            items.sort(Comparator.comparing(m -> Json.string(m, "date", "~")));
            try {
                Checkpointed.before(checkpoints, f);
                Files.createDirectories(f.getParent());
                Files.writeString(f, Json.write(Map.of("entries", items)), StandardCharsets.UTF_8);
            } catch (IOException ex) {
                throw new UncheckedIOException(ex.getMessage(), ex);
            }
            return out;
        }
    }

    private static void merge(Map<String, Object> m, Entry e) {
        put(m, "date", e.date());
        put(m, "channel", e.channel());
        put(m, "title", e.title());
        put(m, "text", e.text());
        if (e.state() != null && !e.state().isBlank()) {
            String s = e.state().strip().toLowerCase(Locale.ROOT);
            if (!List.of("idea", "draft", "ready", "scheduled", "posted").contains(s)) {
                throw new IllegalArgumentException("the state is idea, draft, ready, scheduled or posted");
            }
            m.put("state", s);
        }
        if (e.media() != null) {
            m.put("media", e.media());
        }
        put(m, "notes", e.notes());
    }

    private static void put(Map<String, Object> m, String k, String v) {
        if (v != null) {
            if (v.isBlank()) {
                m.remove(k);
            } else {
                m.put(k, v);
            }
        }
    }

    /**
     * Reads the calendar.
     *
     * @param f its file
     * @return its entries
     */
    static List<Map<String, Object>> calendar(Path f) {
        List<Map<String, Object>> items = new ArrayList<>();
        if (!Files.isRegularFile(f)) {
            return items;
        }
        try {
            Object entries = Json.object(Json.parse(Files.readString(f, StandardCharsets.UTF_8))).get("entries");
            if (entries instanceof List<?> list) {
                for (Object o : list) {
                    items.add(new LinkedHashMap<>(Json.object(o)));
                }
            }
            return items;
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }
}
