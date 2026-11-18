package io.github.zzuegg.brainiac.studio.train;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.Keys;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * The other machines this studio can train on: studios started with {@code --worker}, each with
 * a name, an address and the token it printed. They are kept in {@code hosts.properties} in the
 * data folder, readable only by this account; a token never goes to the page.
 */
public final class Hosts {

    /**
     * A host.
     *
     * @param id its id here
     * @param name what it is called
     * @param url its address
     * @param token the token it accepts
     */
    public record Host(String id, String name, String url, String token) {

        /** Hides the token. */
        @Override
        public String toString() {
            return "Host[" + id + ", " + name + ", " + url + "]";
        }
    }

    private static final long FRESH_MILLIS = 3000;

    private final Path file;
    private final Map<String, Host> hosts = new LinkedHashMap<>();
    private final Map<String, Status> status = new ConcurrentHashMap<>();
    private final Map<String, java.util.Set<String>> lends = new java.util.HashMap<>();   // host id -> secrets it is lent
    private volatile java.util.function.Function<String, Optional<String>> secrets = name -> Optional.empty();

    /**
     * Says where this studio's secrets come from when a host is lent one.
     *
     * @param source gives a secret by name, such as {@code huggingface}
     */
    public void secrets(java.util.function.Function<String, Optional<String>> source) {
        this.secrets = source;
    }

    /**
     * Lends a host one of this studio's secrets, or takes it back, and remembers the choice: a
     * host that lost the secret, because it restarted, is lent it again when it is next needed.
     *
     * @param id the host
     * @param name the secret, such as {@code huggingface}
     * @param lend whether to lend it or take it back
     * @throws IOException if there is no such host or secret, or the host cannot be reached
     */
    public void lend(String id, String name, boolean lend) throws IOException {
        Host h = find(id).orElseThrow(() -> new IOException("no such host"));
        if (lend) {
            String value = secrets.apply(name).orElseThrow(() -> new IOException(
                    "this studio has no " + name + " key to lend: add one under Providers first"));
            client(h).lend(name, value);
        } else {
            try {
                client(h).lend(name, "");
            } catch (IOException e) {
                // a host that is off holds nothing any more: the choice is still remembered
                if (!String.valueOf(e.getMessage()).startsWith("nothing answers")) {
                    throw e;
                }
            }
        }
        synchronized (this) {
            java.util.Set<String> set = lends.computeIfAbsent(id, k -> new java.util.TreeSet<>());
            if (lend ? set.add(name) : set.remove(name)) {
                store();
            }
        }
        status.remove(id);
    }

    /**
     * Takes back everything a host was lent, as far as it still answers; used before it is forgotten.
     *
     * @param id the host
     */
    public void takeBack(String id) {
        Host h = find(id).orElse(null);
        java.util.Set<String> names;
        synchronized (this) {
            names = java.util.Set.copyOf(lends.getOrDefault(id, java.util.Set.of()));
        }
        for (String name : h == null ? java.util.Set.<String>of() : names) {
            try {
                client(h).lend(name, "");
            } catch (IOException | RuntimeException e) {
                // a host that is off or gone holds nothing any more
                System.getLogger(Hosts.class.getName()).log(System.Logger.Level.DEBUG, h.name() + " did not answer");
            }
        }
    }

    /**
     * Lends a host again whatever it is meant to hold and lost, before work that may need it.
     * A secret that cannot be sent is left out: the work then fails with its own reason.
     *
     * @param h the host
     */
    public void relend(Host h) {
        java.util.Set<String> want;
        synchronized (this) {
            want = java.util.Set.copyOf(lends.getOrDefault(h.id(), java.util.Set.of()));
        }
        if (want.isEmpty()) {
            return;
        }
        try {
            WorkerClient c = client(h);
            List<String> held = Json.array(c.info().getOrDefault("secrets", List.of())).stream().map(String::valueOf).toList();
            for (String name : want) {
                Optional<String> value = held.contains(name) ? Optional.empty() : secrets.apply(name);
                if (value.isPresent()) {
                    c.lend(name, value.get());
                }
            }
        } catch (IOException | RuntimeException e) {
            System.getLogger(Hosts.class.getName()).log(System.Logger.Level.WARNING,
                    "cannot lend " + h.name() + " its secrets again: " + e.getMessage());
        }
    }

    private record Status(long at, Map<String, Object> info, String error) {
    }

    /**
     * Reads the hosts kept in a file.
     *
     * @param file {@code hosts.properties}
     */
    public Hosts(Path file) {
        this.file = file;
        load();
    }

    private synchronized void load() {
        if (!Files.isRegularFile(file)) {
            return;
        }
        Properties p = new Properties();
        try (InputStream in = Files.newInputStream(file)) {
            p.load(in);
        } catch (IOException e) {
            return;
        }
        TreeSet<String> ids = new TreeSet<>();
        for (String k : p.stringPropertyNames()) {
            if (k.startsWith("host.") && k.endsWith(".url")) {
                ids.add(k.substring(5, k.length() - 4));
            }
        }
        for (String id : ids) {
            hosts.put(id, new Host(id, p.getProperty("host." + id + ".name", id), p.getProperty("host." + id + ".url"),
                    p.getProperty("host." + id + ".token", "")));
            java.util.Set<String> set = new java.util.TreeSet<>(List.of(p.getProperty("host." + id + ".lend", "").split(",")));
            set.remove("");
            if (!set.isEmpty()) {
                lends.put(id, set);
            }
        }
    }

    private void store() throws IOException {
        Properties p = new Properties();
        for (Host h : hosts.values()) {
            p.setProperty("host." + h.id() + ".name", h.name());
            p.setProperty("host." + h.id() + ".url", h.url());
            p.setProperty("host." + h.id() + ".token", h.token());
            if (!lends.getOrDefault(h.id(), java.util.Set.of()).isEmpty()) {
                p.setProperty("host." + h.id() + ".lend", String.join(",", lends.get(h.id())));   // names, never values
            }
        }
        Files.createDirectories(file.getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        Files.deleteIfExists(tmp);
        Keys.ownerOnly(Files.createFile(tmp));
        try (OutputStream out = Files.newOutputStream(tmp)) {
            p.store(out, "Brainiac Studio training hosts. Readable only by this account.");
        }
        try {
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /**
     * Returns every host.
     *
     * @return the hosts, in the order they were added
     */
    public synchronized List<Host> list() {
        return List.copyOf(hosts.values());
    }

    /**
     * Finds a host.
     *
     * @param id its id
     * @return the host
     */
    public synchronized Optional<Host> find(String id) {
        return Optional.ofNullable(hosts.get(id));
    }

    /**
     * Adds a host after checking that it answers and accepts the token.
     *
     * @param name what to call it, or blank to use the name it reports
     * @param url its address
     * @param token the token it printed
     * @return the host
     * @throws IOException if it cannot be reached, refuses the token, or cannot be saved
     */
    public Host add(String name, String url, String token) throws IOException {
        if (token == null || token.isBlank()) {
            throw new IllegalArgumentException("paste the token the worker printed when it started");
        }
        WorkerClient c = new WorkerClient(url, token.strip());
        Map<String, Object> info = c.info();
        String shown = name == null || name.isBlank() ? Json.string(info, "name", c.base().getHost()) : name.strip();
        if (shown.length() > 60) {
            shown = shown.substring(0, 60);
        }
        synchronized (this) {
            int n = 1;
            while (hosts.containsKey("h" + n)) {
                n++;
            }
            Host h = new Host("h" + n, shown, c.base().toString(), token.strip());
            hosts.put(h.id(), h);
            store();
            status.put(h.id(), new Status(System.currentTimeMillis(), info, null));
            return h;
        }
    }

    /**
     * Forgets a host.
     *
     * @param id its id
     * @return false if there was none
     * @throws IOException if the file cannot be written
     */
    public synchronized boolean remove(String id) throws IOException {
        if (hosts.remove(id) == null) {
            return false;
        }
        lends.remove(id);
        status.remove(id);
        store();
        return true;
    }

    /**
     * Returns a client for a host.
     *
     * @param h the host
     * @return the client
     */
    public WorkerClient client(Host h) {
        return new WorkerClient(h.url(), h.token());
    }

    /**
     * Describes every host for the page, asking each one how it is (at most every few seconds).
     * Tokens are shown only as their last four characters.
     *
     * @return the hosts
     */
    public List<Map<String, Object>> toJson() {
        List<Host> list = list();
        Map<String, Future<Status>> asks = new LinkedHashMap<>();
        try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
            for (Host h : list) {
                Status s = status.get(h.id());
                if (s == null || System.currentTimeMillis() - s.at() > FRESH_MILLIS) {
                    asks.put(h.id(), pool.submit(() -> ask(h)));
                }
            }
            for (var e : asks.entrySet()) {
                try {
                    status.put(e.getKey(), e.getValue().get(6, TimeUnit.SECONDS));
                } catch (Exception ex) {
                    status.put(e.getKey(), new Status(System.currentTimeMillis(), null, "does not answer"));
                }
            }
        }
        List<Map<String, Object>> out = new ArrayList<>();
        for (Host h : list) {
            Status s = status.get(h.id());
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", h.id());
            m.put("name", h.name());
            m.put("url", h.url());
            m.put("token", h.token().length() > 4 ? "…" + h.token().substring(h.token().length() - 4) : "…");
            m.put("online", s != null && s.info() != null);
            synchronized (this) {
                m.put("lend", List.copyOf(lends.getOrDefault(h.id(), java.util.Set.of())));
            }
            if (s != null && s.info() != null) {
                m.put("holds", s.info().getOrDefault("secrets", List.of()));
                m.put("lendable", s.info().containsKey("secrets"));
            }
            if (s != null && s.info() != null) {
                m.putAll(Map.of("devices", s.info().getOrDefault("devices", List.of()),
                        "active", s.info().getOrDefault("active", 0), "queued", s.info().getOrDefault("queued", 0)));
                if (s.info().get("training") != null) {
                    m.put("training", s.info().get("training"));
                }
            }
            if (s != null && s.error() != null) {
                m.put("error", s.error());
            }
            out.add(m);
        }
        return out;
    }

    private Status ask(Host h) {
        try {
            return new Status(System.currentTimeMillis(), client(h).info(), null);
        } catch (IOException | RuntimeException e) {
            String m = e.getMessage();
            return new Status(System.currentTimeMillis(), null, m == null || m.isBlank() ? "does not answer" : m);
        }
    }
}
