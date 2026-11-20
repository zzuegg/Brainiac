package io.github.zzuegg.brainiac.studio.train;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.train.EpochResult;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One training run and how far it is: its settings, its state, the loss of every step (thinned
 * out as the run gets long), and each epoch's results. It is safe to read from one thread while
 * the training thread updates it.
 */
public final class TrainingRun {

    /** Where a run is. */
    public enum State {
        /** Waiting for the run before it to end. */
        QUEUED,
        /** Reading the model and the data. */
        LOADING,
        /** Training. */
        RUNNING,
        /** Trained for every epoch, or stopped early because validation stopped improving. */
        FINISHED,
        /** Stopped on request; what it learned so far was saved. */
        STOPPED,
        /** Could not start or broke off with an error. */
        FAILED,
        /** Was running when the studio closed. */
        INTERRUPTED;

        boolean active() {
            return this == QUEUED || this == LOADING || this == RUNNING;
        }

        String key() {
            return name().toLowerCase(java.util.Locale.ROOT);
        }
    }

    /** The most loss points kept; older points are averaged in pairs beyond that. */
    static final int MAX_POINTS = 1200;

    private final String id;
    private volatile String name;
    private final Map<String, Object> spec;
    private final long created;
    private final String host;
    private State state = State.QUEUED;
    private long started;
    private long ended;
    private String error;
    private String output;
    private String device;
    private int epochs;
    private int epoch;
    private long step;
    private long stepsPerEpoch;
    private int samples;
    private int batchSize;
    private int bestEpoch;
    private boolean stoppedEarly;
    private final List<Map<String, Object>> history = new ArrayList<>();
    private final List<float[]> curve = new ArrayList<>();   // {step, loss}
    private int stride = 1;
    private double pendingLoss;
    private int pendingCount;
    private volatile boolean stopAsked;
    private String remoteId;
    private long remoteSpeed = -1;

    TrainingRun(String id, String name, Map<String, Object> spec, String host) {
        this(id, name, spec, host, System.currentTimeMillis());
    }

    private TrainingRun(String id, String name, Map<String, Object> spec, String host, long created) {
        this.id = id;
        this.name = name;
        this.spec = Map.copyOf(spec);
        this.host = host;
        this.created = created;
        this.epochs = (int) Json.integer(spec, "epochs", 10);
        this.batchSize = (int) Json.integer(spec, "batchSize", 32);
    }

    /**
     * Returns the run's id.
     *
     * @return the id
     */
    public String id() {
        return id;
    }

    /**
     * Returns the run's name.
     *
     * @return the name
     */
    public String name() {
        return name;
    }

    /**
     * Gives the run another name.
     *
     * @param name the new name
     */
    void rename(String name) {
        this.name = name;
    }

    /**
     * Returns the settings it was started with.
     *
     * @return the settings
     */
    public Map<String, Object> spec() {
        return spec;
    }

    /**
     * Returns where it is.
     *
     * @return the state
     */
    public synchronized State state() {
        return state;
    }

    /**
     * Returns where the run is on its host: the run's id on that host, or null for this machine.
     *
     * @return the id there
     */
    synchronized String remoteId() {
        return remoteId;
    }

    synchronized void remoteId(String id) {
        this.remoteId = id;
    }

    /**
     * Returns the host the run trains on.
     *
     * @return {@link TrainingService#THIS_PC} or a host's name
     */
    public String host() {
        return host;
    }

    /**
     * Takes over what a host reports about the run it trains for this one.
     *
     * @param m the host's description of its run
     */
    synchronized void mirror(Map<String, Object> m) {
        String s = Json.string(m, "state", "running").toUpperCase(java.util.Locale.ROOT);
        try {
            State there = State.valueOf(s);
            state = there.active() ? there : State.RUNNING;   // the end is decided after the model is fetched
        } catch (IllegalArgumentException e) {
            state = State.RUNNING;
        }
        device = Json.string(m, "device", device);
        epoch = (int) Json.integer(m, "epoch", epoch);
        step = Json.integer(m, "step", step);
        samples = (int) Json.integer(m, "samples", samples);
        long steps = Json.integer(m, "steps", 0);
        stepsPerEpoch = epochs > 0 ? steps / epochs : 0;
        remoteSpeed = Json.integer(m, "samplesPerSecond", -1);
        if (m.get("history") != null) {
            history.clear();
            for (Object h : Json.array(m.get("history"))) {
                history.add(Json.object(h));
            }
        }
        if (m.get("curve") != null) {
            curve.clear();
            for (Object p : Json.array(m.get("curve"))) {
                List<Object> pt = Json.array(p);
                if (pt.size() == 2 && pt.get(0) instanceof Number a && pt.get(1) instanceof Number b) {
                    curve.add(new float[] {a.floatValue(), b.floatValue()});
                }
            }
            stride = (int) Math.max(1, Json.integer(m, "stride", 1));
        }
    }

    boolean stopAsked() {
        return stopAsked;
    }

    void askToStop() {
        stopAsked = true;
    }

    synchronized void loading() {
        state = State.LOADING;
        started = System.currentTimeMillis();
    }

    synchronized void running(String device, int samples, long stepsPerEpoch) {
        this.state = State.RUNNING;
        this.device = device;
        this.samples = samples;
        this.stepsPerEpoch = stepsPerEpoch;
    }

    synchronized void batch(int epoch, long step, float loss) {
        this.epoch = Math.max(this.epoch, epoch - 1);
        this.step = step;
        if (!Float.isFinite(loss)) {
            return;
        }
        pendingLoss += loss;
        pendingCount++;
        if (pendingCount < stride) {
            return;
        }
        curve.add(new float[] {step, (float) (pendingLoss / pendingCount)});
        pendingLoss = 0;
        pendingCount = 0;
        if (curve.size() >= MAX_POINTS) {
            // keep the whole run in view: average neighbours and take twice as many steps per point
            List<float[]> half = new ArrayList<>(curve.size() / 2 + 1);
            for (int i = 0; i + 1 < curve.size(); i += 2) {
                half.add(new float[] {curve.get(i + 1)[0], (curve.get(i)[1] + curve.get(i + 1)[1]) / 2});
            }
            curve.clear();
            curve.addAll(half);
            stride *= 2;
        }
    }

    synchronized void epoch(EpochResult r, long millis) {
        epoch = r.epoch();
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("epoch", r.epoch());
        m.put("loss", finite(r.loss()));
        m.put("metrics", finite(r.metrics()));
        if (r.hasValidation()) {
            m.put("validationLoss", finite(r.validationLoss()));
            m.put("validationMetrics", finite(r.validationMetrics()));
        }
        m.put("learningRate", finite(r.learningRate()));
        m.put("millis", millis);
        history.add(m);
    }

    synchronized void finish(State end, String output, String error, int bestEpoch, boolean stoppedEarly) {
        this.state = end;
        this.output = output;
        this.error = error;
        this.bestEpoch = bestEpoch;
        this.stoppedEarly = stoppedEarly;
        this.ended = System.currentTimeMillis();
    }

    synchronized void interrupted() {
        if (state.active()) {
            state = state == State.QUEUED ? State.STOPPED : State.INTERRUPTED;
            error = state == State.INTERRUPTED ? "the studio closed while this run was training" : null;
            ended = ended == 0 ? System.currentTimeMillis() : ended;
        }
    }

    private static Object finite(float v) {
        return Float.isFinite(v) ? (Object) v : null;
    }

    private static Map<String, Object> finite(Map<String, Float> m) {
        Map<String, Object> out = new LinkedHashMap<>();
        m.forEach((k, v) -> out.put(k, finite(v)));
        return out;
    }

    /**
     * Describes the run for the page and for saving.
     *
     * @param withCurve true to include every loss point
     * @return the run as JSON-ready values
     */
    public synchronized Map<String, Object> toJson(boolean withCurve) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", id);
        m.put("name", name);
        m.put("host", host);
        m.put("state", state.key());
        m.put("created", created);
        m.put("started", started);
        m.put("ended", ended);
        m.put("spec", spec);
        m.put("data", DataSources.label(Json.object(spec.getOrDefault("data", Map.of()))));
        m.put("epochs", epochs);
        m.put("epoch", epoch);
        m.put("step", step);
        m.put("steps", stepsPerEpoch * epochs);
        m.put("samples", samples);
        m.put("batchSize", batchSize);
        if (device != null) {
            m.put("device", device);
        }
        if (output != null) {
            m.put("output", output);
        }
        if (error != null) {
            m.put("error", error);
        }
        if (bestEpoch > 0) {
            m.put("bestEpoch", bestEpoch);
        }
        if (remoteId != null) {
            m.put("remoteId", remoteId);
        }
        m.put("stoppedEarly", stoppedEarly);
        long now = ended > 0 ? ended : System.currentTimeMillis();
        if (remoteSpeed >= 0) {
            m.put("samplesPerSecond", remoteSpeed);
        } else if (started > 0 && step > 0) {
            double seconds = Math.max(1e-3, (now - started) / 1000.0);
            m.put("samplesPerSecond", Math.round(step * batchSize / seconds));
        }
        m.put("history", List.copyOf(history));
        if (!curve.isEmpty()) {
            float[] last = curve.getLast();
            m.put("loss", last[1]);
        }
        if (withCurve) {
            List<Object> pts = new ArrayList<>(curve.size());
            for (float[] p : curve) {
                pts.add(List.of((long) p[0], p[1]));
            }
            m.put("curve", pts);
            m.put("stride", stride);
        }
        return m;
    }

    /**
     * Reads a saved run back. A run that was active is still marked so; the caller decides
     * whether it can carry on.
     *
     * @param m what {@link #toJson} wrote
     * @return the run
     */
    static TrainingRun fromJson(Map<String, Object> m) {
        TrainingRun r = new TrainingRun(Json.string(m, "id", "?"), Json.string(m, "name", "?"),
                Json.object(m.getOrDefault("spec", Map.of())), Json.string(m, "host", "This PC"),
                Json.integer(m, "created", 0));
        String s = Json.string(m, "state", "failed").toUpperCase(java.util.Locale.ROOT);
        try {
            r.state = State.valueOf(s);
        } catch (IllegalArgumentException e) {
            r.state = State.FAILED;
        }
        r.started = Json.integer(m, "started", 0);
        r.ended = Json.integer(m, "ended", 0);
        r.error = Json.string(m, "error", null);
        r.output = Json.string(m, "output", null);
        r.device = Json.string(m, "device", null);
        r.epoch = (int) Json.integer(m, "epoch", 0);
        r.step = Json.integer(m, "step", 0);
        r.samples = (int) Json.integer(m, "samples", 0);
        long steps = Json.integer(m, "steps", 0);
        r.stepsPerEpoch = r.epochs > 0 ? steps / r.epochs : 0;
        r.bestEpoch = (int) Json.integer(m, "bestEpoch", 0);
        r.stoppedEarly = Boolean.TRUE.equals(m.get("stoppedEarly"));
        r.stride = (int) Math.max(1, Json.integer(m, "stride", 1));
        r.remoteId = Json.string(m, "remoteId", null);
        long speed = Json.integer(m, "samplesPerSecond", -1);
        r.remoteSpeed = r.remoteId != null ? speed : -1;
        for (Object h : m.get("history") == null ? List.of() : Json.array(m.get("history"))) {
            r.history.add(Json.object(h));
        }
        for (Object p : m.get("curve") == null ? List.of() : Json.array(m.get("curve"))) {
            List<Object> pt = Json.array(p);
            if (pt.size() == 2 && pt.get(0) instanceof Number a && pt.get(1) instanceof Number b) {
                r.curve.add(new float[] {a.floatValue(), b.floatValue()});
            }
        }
        return r;
    }
}
