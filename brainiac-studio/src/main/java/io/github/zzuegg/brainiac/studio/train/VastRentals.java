package io.github.zzuegg.brainiac.studio.train;

import io.github.zzuegg.brainiac.cloud.VastBadHosts;
import io.github.zzuegg.brainiac.cloud.VastClient;
import io.github.zzuegg.brainiac.cloud.VastInstance;
import io.github.zzuegg.brainiac.cloud.VastLaunch;
import io.github.zzuegg.brainiac.cloud.VastOffer;
import io.github.zzuegg.brainiac.cloud.VastRunner;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Writer;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.SecureRandom;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Machines rented on Vast.ai from the studio, each set up as a worker host: rented, started, sent
 * this checkout, built, and its worker reached through an SSH tunnel and added on the Train
 * screen, where it trains runs and serves its models.
 *
 * <p>Nothing is rented without the price the user saw and agreed to: {@link #rent} takes that
 * price and refuses when the offer now costs more. A rented machine is destroyed when the user
 * says so, when the hours they chose are up, when setting it up fails, and when the studio
 * closes, so it never runs on unseen. Each rented instance is also written down, so a studio that
 * was killed before it could destroy its machines destroys them the next time it starts
 * ({@link #recover}).
 */
public final class VastRentals implements AutoCloseable {

    private static final System.Logger LOG = System.getLogger(VastRentals.class.getName());

    /** The most machines the studio keeps rented at once. */
    public static final int MAX_ACTIVE = 3;

    /** The longest a rental may run before it is destroyed. */
    public static final int MAX_HOURS = 48;

    /** What renting and setting up a machine takes. */
    public interface Machines {
        /**
         * Rents an offer. Billing starts here.
         *
         * @param offerId the offer
         * @return the instance id
         * @throws IOException if Vast.ai refuses
         */
        long rent(long offerId) throws IOException;

        /**
         * Waits for the instance to run.
         *
         * @param id the instance
         * @param timeout how long to wait
         * @return the running instance
         * @throws IOException if it does not start
         * @throws InterruptedException if interrupted
         */
        VastInstance awaitReady(long id, Duration timeout) throws IOException, InterruptedException;

        /**
         * Logs in and checks, before the checkout is sent, that the machine can run Brainiac
         * (a usable NVIDIA Vulkan driver).
         *
         * @param i the instance
         * @param log a file that gets the check's output
         * @return false if this machine cannot run Brainiac
         * @throws IOException if the check itself fails
         * @throws InterruptedException if interrupted
         */
        boolean usable(VastInstance i, Path log) throws IOException, InterruptedException;

        /**
         * Sends the checkout.
         *
         * @param i the instance
         * @throws IOException if that fails
         * @throws InterruptedException if interrupted
         */
        void prepare(VastInstance i) throws IOException, InterruptedException;

        /**
         * Builds and starts the worker there, reachable on a local port.
         *
         * @param i the instance
         * @param token the worker's token
         * @param localPort the port on this machine
         * @return the connection, whose output is the build's
         * @throws IOException if it cannot be started
         */
        Tunnel startWorker(VastInstance i, String token, int localPort) throws IOException;

        /**
         * Stops the worker there.
         *
         * @param i the instance
         * @throws IOException if that fails
         * @throws InterruptedException if interrupted
         */
        void stopWorker(VastInstance i) throws IOException, InterruptedException;

        /**
         * Destroys an instance; billing ends.
         *
         * @param id the instance
         * @throws IOException if Vast.ai refuses
         */
        void destroy(long id) throws IOException;

        /**
         * Says whether the account still has an instance.
         *
         * @param id the instance
         * @return false if it is gone
         * @throws IOException if Vast.ai cannot be asked
         */
        default boolean exists(long id) throws IOException {
            return true;
        }
    }

    /** The connection to a worker on a rented machine. */
    public interface Tunnel extends AutoCloseable {
        /**
         * Says whether it is still up.
         *
         * @return true while it runs
         */
        boolean alive();

        /**
         * Returns what the machine prints while it builds and runs the worker.
         *
         * @return the output
         */
        InputStream output();

        @Override
        void close();
    }

    /** A machine rented from the studio. */
    private static final class Rental {
        final String key;
        final VastOffer offer;
        final String gpu;
        final double pricePerHour;
        final long started;
        final long until;
        final boolean leftover;
        final Deque<String> log = new ArrayDeque<>();
        volatile long instance;
        volatile VastInstance running;
        volatile String stage = "renting";
        volatile String error;
        volatile String hostId;
        volatile long ended;
        volatile boolean prepared;
        volatile boolean asking;
        volatile Tunnel tunnel;
        volatile Thread thread;

        Rental(String key, VastOffer offer, double pricePerHour, long until) {
            this.key = key;
            this.offer = offer;
            this.gpu = offer.gpus() > 1 ? offer.gpus() + "× " + offer.gpuName() : offer.gpuName();
            this.pricePerHour = pricePerHour;
            this.until = until;
            this.started = System.currentTimeMillis();
            this.leftover = false;
        }

        // a machine an earlier studio rented and did not get to destroy
        Rental(String key, long instance, String gpu, double pricePerHour, long started) {
            this.key = key;
            this.offer = null;
            this.gpu = gpu;
            this.pricePerHour = pricePerHour;
            this.started = started;
            this.until = started;
            this.instance = instance;
            this.leftover = true;
        }

        boolean active() {
            return ended == 0;
        }

        // moves the setup on, unless the rental is being ended
        synchronized void advance(String next) throws InterruptedException {
            if ("destroying".equals(stage) || ended != 0) {
                throw new InterruptedException("ended");
            }
            stage = next;
        }

        synchronized void line(String s) {
            log.addLast(s);
            while (log.size() > 12) {
                log.removeFirst();
            }
        }

        synchronized Map<String, Object> toJson() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("key", key);
            m.put("instance", instance == 0 ? null : instance);
            m.put("offer", offer == null ? null : offer.id());
            if (leftover) {
                m.put("leftover", true);
            }
            m.put("gpu", gpu);
            m.put("pricePerHour", pricePerHour);
            m.put("stage", stage);
            m.put("started", started);
            m.put("until", until);
            m.put("ended", ended);
            long end = ended == 0 ? System.currentTimeMillis() : ended;
            m.put("cost", Math.round(pricePerHour * (end - started) / 3_600_000.0 * 1000) / 1000.0);
            if (error != null) {
                m.put("error", error);
            }
            if (hostId != null) {
                m.put("host", hostId);
            }
            if (ended == 0 && !"ready".equals(stage) && !log.isEmpty()) {
                m.put("last", log.getLast());
            }
            return m;
        }
    }

    private final Hosts hosts;
    private final Machines machines;
    private final Path logs;
    private final Path ledger;
    private final String unavailable;
    private final VastBadHosts bad;
    private final Map<String, Rental> rentals = new ConcurrentHashMap<>();
    private final AtomicLong counter = new AtomicLong();
    private final Thread watchdog;
    private final Duration tick;
    private volatile boolean closed;

    /**
     * Creates the service.
     *
     * @param hosts where a ready machine is added as a host
     * @param machines how machines are rented and set up
     * @param logs where each machine's setup output is written
     * @param unavailable why renting cannot work here (no checkout to send, say), or null
     * @param bad the machines that cannot run Brainiac; one found is added
     */
    public VastRentals(Hosts hosts, Machines machines, Path logs, String unavailable, VastBadHosts bad) {
        this(hosts, machines, logs, unavailable, bad, Duration.ofSeconds(15));
    }

    // the watchdog looks every tick
    VastRentals(Hosts hosts, Machines machines, Path logs, String unavailable, VastBadHosts bad, Duration tick) {
        this.tick = tick;
        this.bad = bad;
        this.hosts = hosts;
        this.machines = machines;
        this.logs = logs;
        this.ledger = logs.resolve("rented.txt");
        this.unavailable = unavailable;
        this.watchdog = Thread.ofVirtual().name("studio-vast-watchdog").start(this::watch);
    }

    /**
     * Uses Vast.ai through brainiac-cloud: the instance runs the {@code cloud/vast/setup.sh} worker
     * of a checkout sent from here, behind an SSH tunnel.
     *
     * @param client gives a client with the user's key, or throws when there is no key
     * @param repo the checkout to send, or null if none was found
     * @return the machines
     */
    public static Machines vast(ClientSource client, Path repo) {
        return new Machines() {
            private VastRunner runner;

            private synchronized VastRunner runner() throws IOException {
                if (repo == null) {
                    throw new IOException("the studio cannot find the Brainiac checkout to send");
                }
                if (runner == null) {
                    runner = new VastRunner(repo, null);
                }
                return runner;
            }

            @Override
            public long rent(long offerId) throws IOException {
                return client.get().rent(offerId, VastLaunch.defaults().withDisk(60));
            }

            @Override
            public VastInstance awaitReady(long id, Duration timeout) throws IOException, InterruptedException {
                return client.get().awaitReady(id, timeout);
            }

            @Override
            public boolean usable(VastInstance i, Path log) throws IOException, InterruptedException {
                runner().awaitSsh(i, Duration.ofMinutes(5));
                int code = runner().check(i, log);
                if (code != 0 && code != 2) {
                    throw new IOException("checking the machine failed (setup.sh check ended with " + code + ")");
                }
                return code == 0;
            }

            @Override
            public void prepare(VastInstance i) throws IOException, InterruptedException {
                runner().ship(i);
            }

            @Override
            public Tunnel startWorker(VastInstance i, String token, int localPort) throws IOException {
                Process p = runner().startWorker(i, token, localPort, List.of());
                return new Tunnel() {
                    @Override
                    public boolean alive() {
                        return p.isAlive();
                    }

                    @Override
                    public InputStream output() {
                        return p.getInputStream();
                    }

                    @Override
                    public void close() {
                        p.destroy();
                    }
                };
            }

            @Override
            public void stopWorker(VastInstance i) throws IOException, InterruptedException {
                runner().stopWorker(i);
            }

            @Override
            public void destroy(long id) throws IOException {
                client.get().destroy(id);
            }

            @Override
            public boolean exists(long id) throws IOException {
                return client.get().instances().stream().anyMatch(i -> i.id() == id);
            }
        };
    }

    /** Gives a Vast.ai client with the user's key. */
    @FunctionalInterface
    public interface ClientSource {
        /**
         * Returns the client.
         *
         * @return the client
         * @throws IOException if there is no key
         */
        VastClient get() throws IOException;
    }

    /**
     * Says why renting cannot work here.
     *
     * @return the reason, or empty when it can
     */
    public Optional<String> unavailable() {
        return Optional.ofNullable(unavailable);
    }

    /**
     * Rents a machine and sets it up as a worker host, in the background.
     *
     * @param offer the offer the user chose, as it is now (its price looked up again)
     * @param agreedPrice the price per hour the user agreed to
     * @param hours how long it may run before it is destroyed
     * @return the rental
     * @throws IllegalArgumentException if the price went up, the hours are out of range or too
     *     many machines are rented already
     * @throws IllegalStateException if renting cannot work here
     */
    public Map<String, Object> rent(VastOffer offer, double agreedPrice, double hours) {
        double seenPrice = offer.dollarsPerHour();
        if (unavailable != null) {
            throw new IllegalStateException(unavailable);
        }
        if (!(agreedPrice > 0) || seenPrice > agreedPrice + 1e-6) {
            throw new IllegalArgumentException(String.format(java.util.Locale.ROOT,
                    "the offer now costs $%.3f an hour, not $%.3f; look at it again before renting", seenPrice, agreedPrice));
        }
        if (!(hours > 0) || hours > MAX_HOURS) {
            throw new IllegalArgumentException("choose between a few minutes and " + MAX_HOURS + " hours");
        }
        synchronized (this) {
            if (rentals.values().stream().filter(Rental::active).count() >= MAX_ACTIVE) {
                throw new IllegalArgumentException("the studio keeps at most " + MAX_ACTIVE
                        + " machines rented at once; destroy one first");
            }
            Rental r = new Rental("r" + counter.incrementAndGet(), offer, seenPrice,
                    System.currentTimeMillis() + (long) (hours * 3_600_000));
            rentals.put(r.key, r);
            // what was asked for, before the setup moves it on
            Map<String, Object> first = r.toJson();
            r.thread = Thread.ofVirtual().name("studio-vast-" + r.key).start(() -> setUp(r));
            return first;
        }
    }

    /**
     * Lists the machines rented from the studio, newest first, with what each has cost so far.
     *
     * @return the rentals
     */
    public List<Map<String, Object>> list() {
        List<Rental> all = new ArrayList<>(rentals.values());
        all.sort((a, b) -> Long.compare(b.started, a.started));
        return all.stream().map(Rental::toJson).toList();
    }

    /**
     * Destroys a machine: one the studio rented (in the background, its worker stopped and its
     * host removed first; {@link #list} shows when it is gone) or any other instance of the account.
     *
     * @param instance the instance id
     * @throws IOException if Vast.ai refuses
     */
    public void destroy(long instance) throws IOException {
        Rental r = rentals.values().stream().filter(x -> x.instance == instance && x.active()).findFirst().orElse(null);
        if (r == null) {
            machines.destroy(instance);
            return;
        }
        endLater(r);
    }

    /**
     * Stops setting up a rental and destroys its machine, in the background.
     *
     * @param key the rental's key
     * @return false if there is no such rental still running
     */
    public boolean cancel(String key) {
        Rental r = rentals.get(key);
        if (r == null || !r.active()) {
            return false;
        }
        endLater(r);
        return true;
    }

    private void endLater(Rental r) {
        if (claim(r)) {
            Thread.ofVirtual().name("studio-vast-end-" + r.key).start(() -> finish(r, null));
        }
    }

    private void setUp(Rental r) {
        try {
            // not interrupted while Vast.ai rents: an instance it gives must be known, to be destroyed
            synchronized (r) {
                if ("destroying".equals(r.stage) || r.ended != 0) {
                    return;   // cancelled before anything was rented
                }
                r.asking = true;
            }
            try {
                r.instance = machines.rent(r.offer.id());
                remember(r);
            } finally {
                r.asking = false;
            }
            r.advance("starting");
            r.running = machines.awaitReady(r.instance, Duration.ofMinutes(15));
            r.advance("checking");
            Files.createDirectories(logs);
            if (!machines.usable(r.running, logs.resolve("vast-" + r.instance + "-check.log"))) {
                bad.add(r.offer, "no usable NVIDIA Vulkan driver");
                throw new IOException("this machine has no usable NVIDIA Vulkan driver, so it is skipped in searches from now on");
            }
            r.advance("copying");
            machines.prepare(r.running);
            r.prepared = true;
            r.advance("building");
            String token = token();
            int port = freePort();
            r.tunnel = machines.startWorker(r.running, token, port);
            Thread.ofVirtual().name("studio-vast-log-" + r.key).start(() -> drain(r));
            long deadline = System.nanoTime() + Duration.ofMinutes(30).toNanos();
            while (true) {
                if (!r.tunnel.alive()) {
                    throw new IOException("the worker stopped while it was set up" + (r.log.isEmpty() ? "" : ": " + r.log.getLast()));
                }
                try {
                    Hosts.Host h = hosts.add("Vast " + r.gpu, "127.0.0.1:" + port, token);
                    r.hostId = h.id();
                    r.advance("ready");
                    LOG.log(System.Logger.Level.INFO, "Vast.ai instance " + r.instance + " is ready as host " + h.id());
                    return;
                } catch (IOException notYet) {
                    if (System.nanoTime() > deadline) {
                        throw new IOException("the worker did not start within 30 minutes");
                    }
                }
                Thread.sleep(5000);
            }
        } catch (InterruptedException e) {
            // ended by the user, the clock or the studio closing; end() cleans up
        } catch (IOException | RuntimeException e) {
            LOG.log(System.Logger.Level.WARNING, "renting " + r.gpu + " failed", e);
            end(r, String.valueOf(e.getMessage()));
        }
    }

    private void drain(Rental r) {
        Path file = logs.resolve("vast-" + (r.instance == 0 ? r.key : String.valueOf(r.instance)) + ".log");
        try {
            Files.createDirectories(logs);
            try (BufferedReader in = new BufferedReader(new InputStreamReader(r.tunnel.output(), StandardCharsets.UTF_8));
                 Writer out = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
                String line;
                while ((line = in.readLine()) != null) {
                    out.write(line);
                    out.write('\n');
                    out.flush();
                    if (!line.isBlank()) {
                        r.line(line.strip());
                    }
                }
            }
        } catch (IOException e) {
            // the connection ended
        }
    }

    // stops the setup, the worker and the tunnel, removes the host and destroys the instance
    private void end(Rental r, String error) {
        if (claim(r)) {
            finish(r, error);
        }
    }

    // marks a rental as ending; false if it already is
    private static boolean claim(Rental r) {
        synchronized (r) {
            if (!r.active() || "destroying".equals(r.stage)) {
                return false;
            }
            r.stage = "destroying";
            return true;
        }
    }

    private void finish(Rental r, String error) {
        Thread t = r.thread;
        if (t != null && t != Thread.currentThread()) {
            synchronized (r) {
                if (!r.asking) {
                    t.interrupt();
                }
            }
            // the setup may be waiting on Vast.ai, which an interrupt does not stop: wait for it,
            // so an instance it is about to be given is destroyed too
            try {
                if (!t.join(Duration.ofSeconds(90))) {
                    LOG.log(System.Logger.Level.WARNING, "setting up " + r.gpu + " did not stop in time");
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        List<String> problems = new ArrayList<>();
        if (r.hostId != null) {
            try {
                hosts.remove(r.hostId);
            } catch (IOException e) {
                problems.add("the host stayed listed: " + e.getMessage());
            }
        }
        if (r.prepared && r.running != null) {
            try {
                machines.stopWorker(r.running);
            } catch (IOException | InterruptedException e) {
                // the instance goes anyway
            }
        }
        if (r.tunnel != null) {
            r.tunnel.close();
        }
        // the instance may be rented although the setup was stopped before it said so
        boolean destroyed = r.instance == 0;
        for (int attempt = 0; attempt < 3 && !destroyed; attempt++) {
            try {
                machines.destroy(r.instance);
                destroyed = true;
            } catch (IOException e) {
                problems.add(e.getMessage());
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException ie) {
                    break;
                }
            }
        }
        if (destroyed && r.instance != 0) {
            forget(r.instance);
        }
        synchronized (r) {
            r.ended = System.currentTimeMillis();
            if (!destroyed) {
                r.stage = "failed";
                r.error = "could not destroy instance " + r.instance + ", so it may still bill: destroy it at "
                        + "cloud.vast.ai/instances (" + String.join("; ", problems) + ")";
            } else if (error != null) {
                r.stage = "failed";
                r.error = error + "; the machine was destroyed";
            } else {
                r.stage = "destroyed";
            }
        }
    }

    /**
     * Destroys the machines an earlier studio rented and did not get to destroy, because it was
     * killed or the PC went down. They cannot be used any more (their worker's token went with
     * that studio) and would bill until someone noticed. Called once the studio is up; each shows
     * in {@link #list} as it goes.
     */
    public void recover() {
        for (String line : ledgerLines()) {
            String[] f = line.split("\\|", 4);
            try {
                Rental r = new Rental("r" + counter.incrementAndGet(), Long.parseLong(f[0]), f[3],
                        Double.parseDouble(f[1]), Long.parseLong(f[2]));
                if (rentals.values().stream().anyMatch(x -> x.instance == r.instance)) {
                    continue;
                }
                Thread.ofVirtual().name("studio-vast-leftover-" + r.instance).start(() -> {
                    try {
                        if (!machines.exists(r.instance)) {
                            forget(r.instance);   // it went some other way
                            return;
                        }
                    } catch (IOException | RuntimeException e) {
                        // cannot tell, so it is destroyed
                    }
                    LOG.log(System.Logger.Level.WARNING, "instance " + r.instance
                            + " was left running by an earlier studio; destroying it");
                    rentals.put(r.key, r);
                    if (claim(r)) {
                        finish(r, null);
                    }
                });
            } catch (RuntimeException e) {
                LOG.log(System.Logger.Level.WARNING, "cannot read a line of " + ledger + ": " + line);
            }
        }
    }

    // the instances rented and not yet destroyed, one a line: id|price|started|gpu
    private List<String> ledgerLines() {
        synchronized (ledger) {
            try {
                return Files.isRegularFile(ledger) ? Files.readAllLines(ledger).stream().filter(l -> !l.isBlank()).toList()
                        : List.of();
            } catch (IOException e) {
                LOG.log(System.Logger.Level.WARNING, "cannot read " + ledger, e);
                return List.of();
            }
        }
    }

    private void remember(Rental r) {
        synchronized (ledger) {
            try {
                Files.createDirectories(logs);
                List<String> lines = new ArrayList<>(ledgerLines());
                lines.add(r.instance + "|" + r.pricePerHour + "|" + r.started + "|" + r.gpu.replace('\n', ' '));
                Files.write(ledger, lines);
            } catch (IOException e) {
                LOG.log(System.Logger.Level.WARNING, "cannot write " + ledger
                        + "; if the studio is killed, instance " + r.instance + " is not destroyed when it starts again", e);
            }
        }
    }

    private void forget(long instance) {
        synchronized (ledger) {
            try {
                List<String> lines = ledgerLines();
                List<String> kept = lines.stream().filter(l -> !l.startsWith(instance + "|")).toList();
                if (kept.size() != lines.size()) {
                    Files.write(ledger, kept);
                }
            } catch (IOException e) {
                LOG.log(System.Logger.Level.WARNING, "cannot write " + ledger, e);
            }
        }
    }

    private void watch() {
        while (!closed) {
            try {
                Thread.sleep(tick);
            } catch (InterruptedException e) {
                return;
            }
            long now = System.currentTimeMillis();
            for (Rental r : rentals.values()) {
                if (r.active() && now >= r.until) {
                    LOG.log(System.Logger.Level.INFO, "the hours chosen for " + r.gpu + " are up; destroying it");
                    endLater(r);
                }
            }
        }
    }

    private static String token() {
        byte[] b = new byte[24];
        new SecureRandom().nextBytes(b);
        return HexFormat.of().formatHex(b);
    }

    private static int freePort() throws IOException {
        try (ServerSocket s = new ServerSocket(0, 1, InetAddress.getLoopbackAddress())) {
            return s.getLocalPort();
        }
    }

    /**
     * Finds the Brainiac checkout the studio runs from, which is what a rented machine builds.
     *
     * @param env the environment ({@code BRAINIAC_REPO} names it)
     * @return the checkout, or empty
     */
    public static Optional<Path> findRepo(Map<String, String> env) {
        String given = System.getProperty("studio.repo", env.get("BRAINIAC_REPO"));
        List<Path> starts = new ArrayList<>();
        if (given != null && !given.isBlank()) {
            starts.add(Path.of(given));
        }
        starts.add(Path.of("").toAbsolutePath());
        try {
            starts.add(Path.of(VastRentals.class.getProtectionDomain().getCodeSource().getLocation().toURI()));
        } catch (Exception e) {
            // run from somewhere without a location
        }
        for (Path start : starts) {
            for (Path p = start.toAbsolutePath(); p != null; p = p.getParent()) {
                if (Files.isRegularFile(p.resolve("settings.gradle.kts")) && Files.isRegularFile(p.resolve("cloud/vast/setup.sh"))
                        && Files.exists(p.resolve(".git"))) {
                    return Optional.of(p);
                }
            }
        }
        return Optional.empty();
    }

    /**
     * Says whether the {@code ssh} command is on the PATH, which reaching a rented machine needs.
     *
     * @param env the environment
     * @return true if it is
     */
    public static boolean hasSsh(Map<String, String> env) {
        String path = env.getOrDefault("PATH", env.getOrDefault("Path", ""));
        for (String dir : path.split(java.io.File.pathSeparator)) {
            if (!dir.isBlank() && (Files.isExecutable(Path.of(dir, "ssh")) || Files.isExecutable(Path.of(dir, "ssh.exe")))) {
                return true;
            }
        }
        return false;
    }

    /** Destroys every machine still rented from the studio, so none bills on unseen. */
    @Override
    public void close() {
        closed = true;
        watchdog.interrupt();
        List<Thread> ending = new ArrayList<>();
        for (Rental r : rentals.values()) {
            if (claim(r)) {
                ending.add(Thread.ofVirtual().name("studio-vast-end-" + r.key).start(() -> finish(r, null)));
            }
        }
        for (Thread t : ending) {
            try {
                t.join(Duration.ofMinutes(3));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }
}
