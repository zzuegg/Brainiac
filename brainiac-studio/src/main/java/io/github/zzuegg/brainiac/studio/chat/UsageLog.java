package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Every call the studio makes to a model, one JSON line each, so the page can show tokens,
 * calls and failures per provider, key, model and day. Nothing in it is secret: keys appear
 * by name only, and prompts are not kept.
 */
public final class UsageLog {

    private static final System.Logger LOG = System.getLogger(UsageLog.class.getName());
    private static final int KEEP = 200_000;

    /**
     * One call.
     *
     * @param at when it ended, in milliseconds since 1970
     * @param provider the provider id
     * @param key the name of the key used, or an empty string
     * @param model the provider's model id
     * @param kind {@code chat}, {@code image}, {@code speech} or {@code video}
     * @param in tokens read, or -1 if unknown
     * @param out tokens written, or -1 if unknown
     * @param units pictures made, or characters spoken; 0 for chat
     * @param seconds how long it took
     * @param ok false if it failed
     */
    public record Entry(long at, String provider, String key, String model, String kind, long in, long out, long units,
            double seconds, boolean ok) {

        Map<String, Object> toJson() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("at", at);
            m.put("provider", provider);
            m.put("key", key);
            m.put("model", model);
            m.put("kind", kind);
            m.put("in", in);
            m.put("out", out);
            m.put("units", units);
            m.put("seconds", Math.round(seconds * 100) / 100.0);
            m.put("ok", ok);
            return m;
        }

        static Entry fromJson(Map<String, Object> m) {
            return new Entry(Json.integer(m, "at", 0), Json.string(m, "provider", ""), Json.string(m, "key", ""),
                    Json.string(m, "model", ""), Json.string(m, "kind", "chat"), Json.integer(m, "in", -1),
                    Json.integer(m, "out", -1), Json.integer(m, "units", 0), Json.number(m, "seconds", 0),
                    !Boolean.FALSE.equals(m.get("ok")));
        }
    }

    private final Path file;
    private final List<Entry> entries = new ArrayList<>();

    /**
     * Opens the log, reading what is already in the file.
     *
     * @param file the file; it need not exist yet
     */
    public UsageLog(Path file) {
        this.file = file;
        if (Files.isRegularFile(file)) {
            try {
                for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
                    if (!line.isBlank()) {
                        try {
                            entries.add(Entry.fromJson(Json.object(Json.parse(line))));
                        } catch (IllegalArgumentException bad) {
                            // a line cut short by a crash; skip it
                        }
                    }
                }
            } catch (IOException e) {
                throw new UncheckedIOException("cannot read " + file, e);
            }
            trim();
        }
    }

    /**
     * Adds a call. A log that cannot be written is reported once per call and otherwise ignored:
     * usage numbers are never a reason to fail a call.
     *
     * @param e the call
     */
    public synchronized void add(Entry e) {
        entries.add(e);
        trim();
        try {
            Files.createDirectories(file.getParent());
            Files.writeString(file, Json.write(e.toJson()) + "\n", StandardCharsets.UTF_8, StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } catch (IOException ex) {
            LOG.log(System.Logger.Level.WARNING, "cannot write the usage log " + file + ": " + ex.getMessage());
        }
    }

    private void trim() {
        if (entries.size() > KEEP) {
            entries.subList(0, entries.size() - KEEP).clear();
        }
    }

    /**
     * Sums the log: per provider, per key, per model, per day (the last 30), and the latest calls.
     *
     * @param zone the zone days are counted in
     * @return the summary, ready to write as JSON
     */
    public synchronized Map<String, Object> summary(ZoneId zone) {
        Map<String, Totals> providers = new LinkedHashMap<>();
        Map<String, Totals> keys = new LinkedHashMap<>();
        Map<String, Totals> models = new LinkedHashMap<>();
        TreeMap<String, Map<String, Totals>> days = new TreeMap<>();
        LocalDate from = LocalDate.now(zone).minusDays(29);
        for (Entry e : entries) {
            providers.computeIfAbsent(e.provider(), k -> new Totals()).add(e);
            if (!e.key().isEmpty()) {
                keys.computeIfAbsent(e.provider() + "/" + e.key(), k -> new Totals()).add(e);
            }
            models.computeIfAbsent(e.provider() + "/" + e.model(), k -> new Totals()).add(e);
            LocalDate day = Instant.ofEpochMilli(e.at()).atZone(zone).toLocalDate();
            if (!day.isBefore(from)) {
                days.computeIfAbsent(day.toString(), d -> new LinkedHashMap<>())
                        .computeIfAbsent(e.provider(), p -> new Totals()).add(e);
            }
        }
        List<Object> daily = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            String d = from.plusDays(i).toString();
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("day", d);
            Map<String, Object> per = new LinkedHashMap<>();
            days.getOrDefault(d, Map.of()).forEach((p, t) -> per.put(p, t.toJson()));
            row.put("providers", per);
            daily.add(row);
        }
        List<Object> recent = new ArrayList<>();
        for (int i = entries.size() - 1; i >= 0 && recent.size() < 40; i--) {
            recent.add(entries.get(i).toJson());
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("providers", json(providers));
        out.put("keys", json(keys));
        out.put("models", json(models));
        out.put("daily", daily);
        out.put("recent", recent);
        return out;
    }

    private static Map<String, Object> json(Map<String, Totals> m) {
        Map<String, Object> out = new LinkedHashMap<>();
        m.forEach((k, t) -> out.put(k, t.toJson()));
        return out;
    }

    private static final class Totals {
        long calls;
        long failed;
        long in;
        long out;
        long units;
        double seconds;

        void add(Entry e) {
            calls++;
            failed += e.ok() ? 0 : 1;
            in += Math.max(0, e.in());
            out += Math.max(0, e.out());
            units += e.units();
            seconds += e.seconds();
        }

        Map<String, Object> toJson() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("calls", calls);
            m.put("failed", failed);
            m.put("in", in);
            m.put("out", out);
            m.put("units", units);
            m.put("seconds", Math.round(seconds * 10) / 10.0);
            return m;
        }
    }
}
