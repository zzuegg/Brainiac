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
        }
    }

    private void store() throws IOException {
        Properties p = new Properties();
        for (Host h : hosts.values()) {
            p.setProperty("host." + h.id() + ".name", h.name());
            p.setProperty("host." + h.id() + ".url", h.url());
            p.setProperty("host." + h.id() + ".token", h.token());
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
