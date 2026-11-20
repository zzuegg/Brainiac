package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * Decides which of this PC's models stay in GPU memory. The chat, image, speaking and
 * transcribing models each keep one model loaded (a {@link GpuSlot}); before one of them loads a
 * model, it asks here for room, and what happens when the model does not fit in what is free is
 * the {@linkplain Policy policy} for that kind of task:
 *
 * <ul>
 *   <li>{@link Policy#SWAP}: the models of the other kinds that are not working and not pinned are
 *       unloaded, the one used longest ago first, until it fits; a model that is working is waited for.
 *   <li>{@link Policy#QUEUE}: nothing is unloaded for it; it waits until there is room, which comes
 *       when a model is unloaded by hand or after being idle for {@link #idleMinutes()}.
 *   <li>{@link Policy#REMOTE}: it is not run here; a {@link NoRoom} says so, and the caller sends the
 *       task to the runner its route names.
 *   <li>{@link Policy#ASK}: a {@link NoRoom} carries the choices back to whoever asked.
 *   <li>{@link Policy#SPILL}: it loads anyway and what does not fit spills to system memory, which
 *       is slow; this is how the studio worked before.
 * </ul>
 *
 * <p>A project can choose another policy for a task with {@link #POLICY}. Room is counted on
 * the device the model loads on; a device that does not count its memory (the processor) always
 * has room. A model's size is first guessed from its files and then measured when it loads.
 */
public final class GpuResidency {

    private static final System.Logger LOG = System.getLogger(GpuResidency.class.getName());

    /** The kinds of task that keep a model loaded on this PC. */
    public static final List<String> KINDS = List.of("chat", "image", "speech", "transcribe", "video");

    /** What to do with a task whose model does not fit in the GPU memory that is free. */
    public enum Policy {
        /** Unload idle models of other kinds to make room. */
        SWAP,
        /** Wait until there is room. */
        QUEUE,
        /** Run it on the route's runner instead. */
        REMOTE,
        /** Ask which of the others to do. */
        ASK,
        /** Load anyway and let what does not fit spill to system memory. */
        SPILL;

        /**
         * Reads a policy by name, ignoring case.
         *
         * @param name the name, such as {@code swap}
         * @return the policy
         * @throws IllegalArgumentException if there is no such policy
         */
        public static Policy of(String name) {
            try {
                return valueOf(name.strip().toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("no scheduling policy " + name
                        + ": use swap, queue, remote, ask or spill");
            }
        }

        /** @return the name the page and API use */
        public String key() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    /**
     * A policy that holds for the tasks run while it is bound, such as a project's choice for its
     * tasks. Unbound, the studio's own choice for the kind of task holds.
     */
    public static final ScopedValue<Policy> POLICY = ScopedValue.newInstance();

    /**
     * Says a model was not loaded because it does not fit, and why.
     *
     * <p>Its message is meant for the user; {@link #policy()} says whether the caller should send
     * the task elsewhere ({@link Policy#REMOTE}) or ask ({@link Policy#ASK}).
     */
    public static final class NoRoom extends IOException {

        private static final long serialVersionUID = 1L;

        private final transient Policy policy;
        private final String kind;
        private final long need;
        private final long free;
        private final transient List<String> holding;

        NoRoom(Policy policy, String kind, long need, long free, List<String> holding, String why) {
            super(why);
            this.policy = policy;
            this.kind = kind;
            this.need = need;
            this.free = free;
            this.holding = List.copyOf(holding);
        }

        /** @return the policy that gave up: what the caller is asked to do next */
        public Policy policy() {
            return policy;
        }

        /** @return the kind of task */
        public String kind() {
            return kind;
        }

        /** @return the bytes the model is thought to need */
        public long need() {
            return need;
        }

        /** @return the bytes that were free */
        public long free() {
            return free;
        }

        /** @return the models holding the memory, as {@code kind: model} */
        public List<String> holding() {
            return holding;
        }

        /**
         * Describes it for the page: what is needed and what could be done.
         *
         * @return {@code {kind, need, free, holding, choices}}
         */
        public Map<String, Object> toJson() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("kind", kind);
            m.put("need", need);
            m.put("free", free);
            m.put("holding", holding);
            m.put("choices", List.of("swap", "queue", "remote", "spill"));
            m.put("message", getMessage());
            return m;
        }
    }

    /** One task waiting for room, for the page. */
    record Waiting(long id, String kind, String model, long need, long since) {
    }

    private final Function<Device, Optional<Device.Memory>> memory;
    private final Path file;
    private final List<GpuSlot> slots = new CopyOnWriteArrayList<>();
    private final Map<String, Long> measured = new ConcurrentHashMap<>();
    private final Map<Long, Waiting> waiting = new ConcurrentHashMap<>();
    private final AtomicLong ids = new AtomicLong();
    private final Map<String, Policy> policies = new EnumMapByKind();
    private final Set<String> pinned = ConcurrentHashMap.newKeySet();
    /** Minutes an unused model stays loaded until the user chooses otherwise. */
    static final int DEFAULT_IDLE_MINUTES = 5;

    private volatile int idleMinutes = DEFAULT_IDLE_MINUTES;
    // whether the user chose the idle minutes; until then a newer default applies
    private volatile boolean idleChosen;
    private volatile long waitMillis = 30 * 60_000L;
    private volatile long pollMillis = 500;

    /** Kinds to policies, with every kind present. */
    private static final class EnumMapByKind extends ConcurrentHashMap<String, Policy> {
        private static final long serialVersionUID = 1L;

        EnumMapByKind() {
            for (String k : KINDS) {
                put(k, Policy.SWAP);
            }
        }
    }

    /**
     * Schedules with the devices' own memory counts, keeping its choices in a file.
     *
     * @param file where the choices are kept, or null to keep them only while running
     */
    public GpuResidency(Path file) {
        this(file, d -> {
            try {
                return d.memory();
            } catch (RuntimeException | LinkageError e) {
                return Optional.empty();
            }
        });
    }

    /**
     * Schedules with other memory counts, for tests.
     *
     * @param file where the choices are kept, or null
     * @param memory what a device has in use and may use
     */
    GpuResidency(Path file, Function<Device, Optional<Device.Memory>> memory) {
        this.file = file;
        this.memory = memory;
        read();
    }

    void register(GpuSlot slot) {
        slots.add(slot);
    }

    List<GpuSlot> slots() {
        return List.copyOf(slots);
    }

    /** For tests: how long a queued or swapping task waits, and how often it looks. */
    void timing(long waitMillis, long pollMillis) {
        this.waitMillis = waitMillis;
        this.pollMillis = pollMillis;
    }

    /**
     * Returns the studio's own policy for a kind of task.
     *
     * @param kind one of {@link #KINDS}
     * @return the policy
     */
    public Policy policy(String kind) {
        return policies.getOrDefault(kind, Policy.SWAP);
    }

    /**
     * Returns the policy that holds now for a kind of task: a bound {@link #POLICY}, or else the
     * studio's own.
     *
     * @param kind the kind
     * @return the policy
     */
    public Policy effective(String kind) {
        return POLICY.isBound() ? POLICY.get() : policy(kind);
    }

    /**
     * Returns the minutes an unused model stays loaded before it is unloaded.
     *
     * @return the minutes, 0 for as long as nothing else needs the room
     */
    public int idleMinutes() {
        return idleMinutes;
    }

    /**
     * Changes the choices. Fields left out stay as they are.
     *
     * @param change {@code {policies?: {kind: policy}, pinned?: [kind], idleMinutes?}}
     * @throws IllegalArgumentException if a kind or policy is unknown
     * @throws IOException if the choices cannot be kept
     */
    public void change(Map<String, Object> change) throws IOException {
        Map<String, Policy> ps = new LinkedHashMap<>();
        if (change.get("policies") != null) {
            for (Map.Entry<String, Object> e : Json.object(change.get("policies")).entrySet()) {
                ps.put(kind(e.getKey()), Policy.of(String.valueOf(e.getValue())));
            }
        }
        Set<String> pins = null;
        if (change.get("pinned") != null) {
            pins = new LinkedHashSet<>();
            for (Object o : Json.array(change.get("pinned"))) {
                pins.add(kind(String.valueOf(o)));
            }
        }
        int idle = (int) Json.integer(change, "idleMinutes", idleMinutes);
        if (idle < 0 || idle > 24 * 60) {
            throw new IllegalArgumentException("idle minutes go from 0 to 1440");
        }
        policies.putAll(ps);
        if (pins != null) {
            pinned.clear();
            pinned.addAll(pins);
        }
        if (change.get("idleMinutes") != null) {
            idleMinutes = idle;
            idleChosen = true;
        }
        write();
    }

    private static String kind(String k) {
        if (!KINDS.contains(k)) {
            throw new IllegalArgumentException("no kind of task " + k + ": use " + String.join(", ", KINDS));
        }
        return k;
    }

    /**
     * Pins or unpins a kind: a pinned kind's model is never unloaded to make room.
     *
     * @param kind the kind
     * @param pin whether to pin it
     * @throws IOException if the choice cannot be kept
     */
    public void pin(String kind, boolean pin) throws IOException {
        if (pin) {
            pinned.add(kind(kind));
        } else {
            pinned.remove(kind);
        }
        write();
    }

    /**
     * Unloads the models of some kinds that are not working now.
     *
     * @param kind a kind, or {@code all}
     * @return the kinds unloaded
     */
    public List<String> unload(String kind) {
        List<String> out = new ArrayList<>();
        for (GpuSlot s : slots) {
            if (("all".equals(kind) || s.kind().equals(kind)) && s.loadedModel() != null && s.tryUnload()) {
                out.add(s.kind());
            }
        }
        if (!out.isEmpty()) {
            settle();
        }
        return out;
    }

    /** Unloads the models idle for longer than {@link #idleMinutes()}; the studio calls it now and then. */
    public void reap() {
        int minutes = idleMinutes;
        if (minutes <= 0) {
            return;
        }
        long before = System.currentTimeMillis() - minutes * 60_000L;
        boolean any = false;
        for (GpuSlot s : slots) {
            if (s.loadedModel() != null && !s.busy() && s.lastUsed() < before && s.tryUnload()) {
                LOG.log(System.Logger.Level.INFO, "unloaded the " + s.kind() + " model after " + minutes + " idle minutes");
                any = true;
            }
        }
        if (any) {
            settle();
        }
    }

    /**
     * Guesses the GPU memory a model takes: what was measured when it last loaded, or else the
     * size of its files.
     *
     * @param key the model and device, as {@link GpuSlot} names them
     * @param path the model's file or folder
     * @return the bytes
     */
    long estimate(String key, Path path) {
        Long m = measured.get(key);
        return m != null ? m : sizeOnDisk(path);
    }

    boolean wasMeasured(String key) {
        return measured.containsKey(key);
    }

    void measuredAs(String key, long bytes) {
        if (bytes > 0) {
            measured.put(key, bytes);
        }
    }

    /**
     * Returns the size of a model's weights on disk.
     *
     * @param path a file or a folder
     * @return the bytes of its weight files, or of everything when none are recognised
     */
    static long sizeOnDisk(Path path) {
        if (path == null) {
            return 0;
        }
        try {
            if (Files.isRegularFile(path)) {
                return Files.size(path);
            }
            if (!Files.isDirectory(path)) {
                return 0;
            }
            try (Stream<Path> s = Files.walk(path, 4)) {
                long[] all = {0, 0};
                s.filter(Files::isRegularFile).forEach(p -> {
                    try {
                        long n = Files.size(p);
                        String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
                        all[0] += n;
                        if (name.endsWith(".safetensors") || name.endsWith(".gguf") || name.endsWith(".bin")
                                || name.endsWith(".pt") || name.endsWith(".pth") || name.endsWith(".ckpt")
                                || name.endsWith(".onnx")) {
                            all[1] += n;
                        }
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                });
                return all[1] > 0 ? all[1] : all[0];
            }
        } catch (IOException | UncheckedIOException e) {
            return 0;
        }
    }

    /**
     * Makes room for a model before a slot loads it, as the policy that holds says. The slot holds
     * its own lock and has already let go of its old model.
     *
     * @param asking the slot that loads
     * @param model the model's name, for messages
     * @param device where it loads
     * @param need the bytes it is thought to need
     * @param cancelled says when to give up waiting
     * @throws NoRoom if it does not fit and the policy says to run it elsewhere or to ask, or
     *     waiting took too long
     * @throws IOException if waiting was interrupted
     */
    void admit(GpuSlot asking, String model, Device device, long need, BooleanSupplier cancelled) throws IOException {
        Policy policy = effective(asking.kind());
        if (policy == Policy.SPILL || need <= 0 || memory.apply(device).isEmpty()) {
            return;
        }
        long free = free(device);
        if (need <= free) {
            return;
        }
        if (policy == Policy.REMOTE || policy == Policy.ASK) {
            throw noRoom(policy, asking, model, need, free);
        }
        long id = ids.incrementAndGet();
        waiting.put(id, new Waiting(id, asking.kind(), model, need, System.currentTimeMillis()));
        try {
            long deadline = System.currentTimeMillis() + waitMillis;
            while (true) {
                if (policy == Policy.SWAP) {
                    for (GpuSlot s : candidates(asking)) {
                        if (s.tryUnload()) {
                            LOG.log(System.Logger.Level.INFO, "unloaded the " + s.kind() + " model to make room for "
                                    + model);
                            settle();
                            free = free(device);
                            if (need <= free) {
                                return;
                            }
                        }
                    }
                }
                free = free(device);
                if (need <= free) {
                    return;
                }
                if (policy == Policy.SWAP && slots.stream().noneMatch(s -> s != asking && s.loadedModel() != null
                        && !pinned.contains(s.kind()))) {
                    // nothing left to unload: only pinned models or memory the studio does not hold
                    throw noRoom(Policy.ASK, asking, model, need, free);
                }
                if (cancelled.getAsBoolean() || System.currentTimeMillis() > deadline) {
                    throw noRoom(Policy.ASK, asking, model, need, free);
                }
                try {
                    Thread.sleep(pollMillis);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("stopped while waiting for GPU memory", e);
                }
            }
        } finally {
            waiting.remove(id);
        }
    }

    /**
     * Unloads every other model that may go, for a model whose own check found too little room
     * after {@link #admit} let it load, as the swap policy allows.
     *
     * @param asking the slot that loads
     * @param model the model's name, for the log
     * @return whether any was unloaded
     */
    boolean makeRoom(GpuSlot asking, String model) {
        if (effective(asking.kind()) != Policy.SWAP) {
            return false;
        }
        boolean any = false;
        for (GpuSlot s : candidates(asking)) {
            if (s.tryUnload()) {
                LOG.log(System.Logger.Level.INFO, "unloaded the " + s.kind() + " model to make room for " + model);
                any = true;
            }
        }
        if (any) {
            settle();
        }
        return any;
    }

    // the other loaded models that may go, the one used longest ago first
    private List<GpuSlot> candidates(GpuSlot asking) {
        return slots.stream()
                .filter(s -> s != asking && s.loadedModel() != null && !pinned.contains(s.kind()) && !s.busy())
                .sorted(Comparator.comparingLong(GpuSlot::lastUsed))
                .toList();
    }

    private NoRoom noRoom(Policy policy, GpuSlot asking, String model, long need, long free) {
        List<String> holding = new ArrayList<>();
        for (GpuSlot s : slots) {
            if (s != asking && s.loadedModel() != null) {
                holding.add(s.kind() + ": " + s.loadedModel() + (pinned.contains(s.kind()) ? " (pinned)" : "")
                        + (s.busy() ? " (working)" : ""));
            }
        }
        String why = model + " needs about " + gb(need) + " of GPU memory and " + gb(Math.max(0, free)) + " is free"
                + (holding.isEmpty() ? "" : " (" + String.join(", ", holding) + ")")
                + switch (policy) {
                    case REMOTE -> ": it goes to a remote runner";
                    case ASK -> ": unload a model, wait, run it on a remote runner or let it spill to system memory";
                    default -> "";
                };
        return new NoRoom(policy, asking.kind(), need, free, holding, why);
    }

    long inUse(Device device) {
        return memory.apply(device).map(Device.Memory::inUse).orElse(0L);
    }

    private long free(Device device) {
        Optional<Device.Memory> m = memory.apply(device);
        if (m.isEmpty()) {
            return Long.MAX_VALUE;
        }
        Device.Memory mem = m.get();
        return mem.budget() - mem.inUse();
    }

    // freed tensors count until collected
    private static void settle() {
        System.gc();
    }

    static String gb(long bytes) {
        return String.format(Locale.ROOT, "%.1f GB", bytes / (double) (1L << 30));
    }

    /**
     * Describes what is loaded and waiting, and the choices, for the page.
     *
     * @param device the device models load on now
     * @return {@code {device, inUse, budget, slots, waiting, policies, pinned, idleMinutes}}
     */
    public Map<String, Object> toJson(Device device) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("device", device == null ? null : device.name());
        Optional<Device.Memory> m = device == null ? Optional.empty() : memory.apply(device);
        out.put("inUse", m.map(Device.Memory::inUse).orElse(null));
        out.put("budget", m.map(Device.Memory::budget).orElse(null));
        List<Map<String, Object>> ss = new ArrayList<>();
        for (GpuSlot s : slots) {
            Map<String, Object> e = new LinkedHashMap<>();
            e.put("kind", s.kind());
            e.put("model", s.loadedModel());
            e.put("bytes", s.loadedModel() == null ? 0 : s.bytes());
            e.put("busy", s.busy());
            e.put("lastUsed", s.lastUsed());
            e.put("pinned", pinned.contains(s.kind()));
            ss.add(e);
        }
        out.put("slots", ss);
        List<Map<String, Object>> ws = new ArrayList<>();
        for (Waiting w : waiting.values().stream().sorted(Comparator.comparingLong(Waiting::id)).toList()) {
            ws.add(Map.of("id", w.id(), "kind", w.kind(), "model", w.model(), "need", w.need(), "since", w.since()));
        }
        out.put("waiting", ws);
        Map<String, Object> ps = new LinkedHashMap<>();
        for (String k : KINDS) {
            ps.put(k, policy(k).key());
        }
        out.put("policies", ps);
        out.put("pinned", KINDS.stream().filter(pinned::contains).toList());
        out.put("idleMinutes", idleMinutes);
        return out;
    }

    private void read() {
        if (file == null || !Files.isRegularFile(file)) {
            return;
        }
        try {
            Map<String, Object> m = Json.object(Json.read(file));
            if (m.get("policies") != null) {
                for (Map.Entry<String, Object> e : Json.object(m.get("policies")).entrySet()) {
                    if (KINDS.contains(e.getKey())) {
                        policies.put(e.getKey(), Policy.of(String.valueOf(e.getValue())));
                    }
                }
            }
            if (m.get("pinned") != null) {
                for (Object o : Json.array(m.get("pinned"))) {
                    if (KINDS.contains(String.valueOf(o))) {
                        pinned.add(String.valueOf(o));
                    }
                }
            }
            // files from before the choice was kept hold the old default of 0, which no one chose
            idleChosen = Boolean.TRUE.equals(m.get("idleChosen"));
            if (idleChosen) {
                idleMinutes = (int) Math.max(0, Json.integer(m, "idleMinutes", DEFAULT_IDLE_MINUTES));
            }
            if (m.get("measured") != null) {
                for (Map.Entry<String, Object> e : Json.object(m.get("measured")).entrySet()) {
                    if (e.getValue() instanceof Number n && n.longValue() > 0) {
                        measured.put(e.getKey(), n.longValue());
                    }
                }
            }
        } catch (IOException | RuntimeException e) {
            LOG.log(System.Logger.Level.WARNING, "cannot read " + file, e);
        }
    }

    private synchronized void write() throws IOException {
        if (file == null) {
            return;
        }
        Map<String, Object> m = new LinkedHashMap<>();
        Map<String, Object> ps = new LinkedHashMap<>();
        for (String k : KINDS) {
            ps.put(k, policy(k).key());
        }
        m.put("policies", ps);
        m.put("pinned", KINDS.stream().filter(pinned::contains).toList());
        m.put("idleMinutes", idleMinutes);
        m.put("idleChosen", idleChosen);
        m.put("measured", new LinkedHashMap<>(measured));
        Files.createDirectories(file.toAbsolutePath().getParent());
        Files.writeString(file, Json.write(m));
    }

    /** Keeps what was measured, so the next start guesses better; failures are only logged. */
    void keepMeasured() {
        try {
            write();
        } catch (IOException e) {
            LOG.log(System.Logger.Level.WARNING, "cannot write " + file, e);
        }
    }
}
