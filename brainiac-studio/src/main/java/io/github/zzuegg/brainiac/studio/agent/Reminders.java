package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Reminders the user or an agent set: at their time each shows on the Agents page until it is
 * dismissed, and one with a goal also starts an agent on it, which asks before it acts. A
 * reminder can repeat daily or weekly.
 */
public final class Reminders implements AutoCloseable {

    private static final Pattern RELATIVE = Pattern.compile("\\+?\\s*(?:in\\s+)?(\\d+)\\s*(m|min|minutes?|h|hours?|d|days?)",
            Pattern.CASE_INSENSITIVE);

    private final Path file;
    private final Consumer<Map<String, Object>> fire;
    private final Thread clock;
    private volatile boolean closed;

    /**
     * Keeps reminders and starts watching their times.
     *
     * @param file where they are kept
     * @param fire what to do when one is due, after it is marked due
     */
    public Reminders(Path file, Consumer<Map<String, Object>> fire) {
        this.file = file;
        this.fire = fire;
        this.clock = Thread.ofVirtual().name("studio-reminders").start(this::loop);
    }

    @Override
    public void close() {
        closed = true;
        clock.interrupt();
    }

    /**
     * Reads a time: {@code 2026-12-01T09:00}, or {@code in 30 minutes}, {@code +2h}, {@code 1d}.
     *
     * @param at the time
     * @return milliseconds since the epoch
     */
    public static long when(String at) {
        String s = at == null ? "" : at.strip();
        Matcher m = RELATIVE.matcher(s);
        if (m.matches()) {
            long n = Long.parseLong(m.group(1));
            char unit = m.group(2).toLowerCase(Locale.ROOT).charAt(0);
            Duration d = unit == 'm' ? Duration.ofMinutes(n) : unit == 'h' ? Duration.ofHours(n) : Duration.ofDays(n);
            return System.currentTimeMillis() + d.toMillis();
        }
        try {
            return LocalDateTime.parse(s.length() == 10 ? s + "T09:00" : s).atZone(ZoneId.systemDefault()).toInstant()
                    .toEpochMilli();
        } catch (DateTimeParseException e) {
            throw new IllegalArgumentException("write the time as 2026-12-01T09:00, or as 'in 30 minutes'");
        }
    }

    /**
     * Adds a reminder.
     *
     * @param at when
     * @param text what to remind of
     * @param goal what an agent should then do, or blank for none
     * @param role the agent's role
     * @param project the project
     * @param repeat {@code daily}, {@code weekly} or blank
     * @param by who set it: {@code you} or an agent's id
     * @return the reminder
     */
    public synchronized Map<String, Object> add(String at, String text, String goal, String role, String project,
            String repeat, String by) {
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("say what to remind of");
        }
        String rep = repeat == null ? "" : repeat.strip().toLowerCase(Locale.ROOT);
        if (!List.of("", "daily", "weekly").contains(rep)) {
            throw new IllegalArgumentException("a reminder repeats daily or weekly, or not at all");
        }
        long t = when(at);
        if (t < System.currentTimeMillis() - 60_000) {
            throw new IllegalArgumentException("that time is past");
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", UUID.randomUUID().toString().substring(0, 8));
        m.put("at", t);
        m.put("text", text.strip());
        if (goal != null && !goal.isBlank()) {
            m.put("goal", goal.strip());
            m.put("role", role == null || role.isBlank() ? Roles.DEFAULT : role);
        }
        m.put("project", project == null ? "" : project);
        if (!rep.isEmpty()) {
            m.put("repeat", rep);
        }
        m.put("by", by);
        m.put("state", "waiting");
        List<Map<String, Object>> all = load();
        if (all.stream().filter(x -> "waiting".equals(x.get("state"))).count() >= 500) {
            throw new IllegalArgumentException("there are 500 reminders waiting already");
        }
        all.add(m);
        store(all);
        return m;
    }

    /**
     * Lists reminders, soonest first.
     *
     * @return every reminder not dismissed
     */
    public synchronized List<Map<String, Object>> list() {
        List<Map<String, Object>> all = load();
        all.removeIf(m -> "dismissed".equals(m.get("state")));
        all.sort((a, b) -> Long.compare(Json.integer(a, "at", 0), Json.integer(b, "at", 0)));
        return all;
    }

    /**
     * Dismisses a reminder that is due, or cancels one that waits.
     *
     * @param id the reminder
     * @return false if there is none
     */
    public synchronized boolean dismiss(String id) {
        List<Map<String, Object>> all = load();
        boolean hit = all.removeIf(m -> id.equals(m.get("id")));
        if (hit) {
            store(all);
        }
        return hit;
    }

    /** Fires the reminders whose time has come. */
    void tick() {
        List<Map<String, Object>> due = new ArrayList<>();
        synchronized (this) {
            List<Map<String, Object>> all = load();
            long now = System.currentTimeMillis();
            boolean changed = false;
            for (Map<String, Object> m : new ArrayList<>(all)) {
                if (!"waiting".equals(m.get("state")) || Json.integer(m, "at", Long.MAX_VALUE) > now) {
                    continue;
                }
                Map<String, Object> fired = new LinkedHashMap<>(m);
                fired.put("state", "due");
                fired.put("fired", now);
                String rep = Json.string(m, "repeat", "");
                if (rep.isEmpty()) {
                    all.set(all.indexOf(m), fired);
                } else {
                    // the repeating one moves on; a copy shows as due
                    long step = rep.equals("daily") ? 86_400_000L : 7 * 86_400_000L;
                    long next = Json.integer(m, "at", now);
                    while (next <= now) {
                        next += step;
                    }
                    m.put("at", next);
                    fired.put("id", UUID.randomUUID().toString().substring(0, 8));
                    fired.remove("repeat");
                    all.add(fired);
                }
                due.add(fired);
                changed = true;
            }
            if (changed) {
                store(all);
            }
        }
        for (Map<String, Object> m : due) {
            try {
                fire.accept(m);
            } catch (RuntimeException e) {
                synchronized (this) {
                    List<Map<String, Object>> all = load();
                    for (Map<String, Object> x : all) {
                        if (m.get("id").equals(x.get("id"))) {
                            x.put("error", e.getMessage());
                        }
                    }
                    store(all);
                }
            }
        }
    }

    /**
     * Notes the agent a reminder started.
     *
     * @param id the reminder
     * @param run the agent's id
     */
    synchronized void started(String id, String run) {
        List<Map<String, Object>> all = load();
        for (Map<String, Object> x : all) {
            if (id.equals(x.get("id"))) {
                x.put("run", run);
            }
        }
        store(all);
    }

    private void loop() {
        while (!closed) {
            try {
                tick();
                Thread.sleep(20_000);
            } catch (InterruptedException e) {
                return;
            } catch (RuntimeException e) {
                System.getLogger(Reminders.class.getName()).log(System.Logger.Level.WARNING, "reminders", e);
            }
        }
    }

    private List<Map<String, Object>> load() {
        List<Map<String, Object>> out = new ArrayList<>();
        if (Files.isRegularFile(file)) {
            try {
                for (Object o : Json.array(Json.object(Json.parse(Files.readString(file, StandardCharsets.UTF_8)))
                        .getOrDefault("reminders", List.of()))) {
                    out.add(new LinkedHashMap<>(Json.object(o)));
                }
            } catch (IOException | RuntimeException e) {
                System.getLogger(Reminders.class.getName()).log(System.Logger.Level.WARNING, "cannot read " + file, e);
            }
        }
        return out;
    }

    private void store(List<Map<String, Object>> all) {
        try {
            Files.createDirectories(file.toAbsolutePath().getParent());
            Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
            Files.writeString(tmp, Json.write(Map.of("reminders", all)), StandardCharsets.UTF_8);
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new java.io.UncheckedIOException(e.getMessage(), e);
        }
    }
}
