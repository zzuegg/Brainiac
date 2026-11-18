package io.github.zzuegg.brainiac.experimental.parallel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A pipeline-parallel training schedule: which forward and backward operations each device runs,
 * in which order, for one step of {@code microBatches} micro-batches through a model split into
 * {@code stages} consecutive stages.
 *
 * <p>Generators for the published schedules:
 * <ul>
 *   <li>{@link #gpipe GPipe}: every forward, then every backward; bubble {@code (p-1)/(m+p-1)} and
 *       every stage holds all {@code m} micro-batches' activations (Huang et al., 2019);</li>
 *   <li>{@link #oneFOneB 1F1B} (PipeDream-Flush): after {@code p-d-1} warm-up forwards, device
 *       {@code d} alternates one forward and one backward; the same bubble, at most {@code p}
 *       micro-batches in flight (Narayanan et al., 2019);</li>
 *   <li>{@link #interleaved interleaved 1F1B}: each device holds {@code v} non-adjacent chunks,
 *       which divides the bubble by {@code v} at {@code v} times the messages (Narayanan et al., 2021);</li>
 *   <li>{@link #zeroBubble zero-bubble} ZB-H1 and ZB-H2: the backward split into B (input gradient,
 *       on the critical path) and W (weight gradient, which nobody waits for); deferred W fill
 *       what would be bubbles (Qi et al., 2024);</li>
 *   <li>{@link #dualPipe DualPipe}: micro-batches enter from both ends at once, each device holding
 *       stage {@code d} for one direction and stage {@code p-1-d} for the other (DeepSeek-V3).</li>
 * </ul>
 * The zero-bubble and DualPipe orders come from a greedy list scheduler driven by a
 * {@link ParallelCostModel}: a free device runs a ready B first, then a forward if its in-flight
 * limit allows, and otherwise a pending W, which is how the zero-bubble paper's heuristic fills
 * bubbles. {@link #validate()} checks that every operation appears exactly once on the right
 * device and that the order cannot deadlock.
 *
 * <p>Papers: <a href="https://arxiv.org/abs/1811.06965">arXiv:1811.06965</a>,
 * <a href="https://arxiv.org/abs/1806.03377">arXiv:1806.03377</a>,
 * <a href="https://arxiv.org/abs/2104.04473">arXiv:2104.04473</a>,
 * <a href="https://arxiv.org/abs/2401.10241">arXiv:2401.10241</a>,
 * <a href="https://arxiv.org/abs/2412.19437">arXiv:2412.19437</a>.
 */
public final class PipelineSchedule {

    /** The kind of an operation. */
    public enum Kind {
        /** The forward pass of a stage. */
        F,
        /** The input-gradient half of a split backward pass. */
        B,
        /** The weight-gradient half of a split backward pass. */
        W,
        /** A whole backward pass, input and weight gradients together. */
        BW
    }

    /**
     * One operation.
     *
     * @param kind what runs
     * @param microBatch the micro-batch
     * @param stage the model stage, from 0 (the input end)
     * @param direction 0 for micro-batches that enter at device 0; 1 for DualPipe's reverse direction
     */
    public record Op(Kind kind, int microBatch, int stage, int direction) {

        @Override
        public String toString() {
            return kind + "" + microBatch + "s" + stage + (direction == 1 ? "r" : "");
        }
    }

    private final String name;
    private final int devices;
    private final int stages;
    private final int microBatches;
    private final boolean bidirectional;
    private final boolean split;
    private final boolean vShape;
    private final List<List<Op>> order;

    /**
     * Creates a schedule from explicit per-device orders.
     *
     * @param name a label
     * @param devices the number of devices
     * @param stages the number of model stages; a multiple of {@code devices}
     * @param microBatches the number of micro-batches
     * @param bidirectional whether micro-batches of the second half run in DualPipe's reverse
     *     direction
     * @param order the operations of each device, in execution order
     */
    public PipelineSchedule(String name, int devices, int stages, int microBatches, boolean bidirectional,
            List<List<Op>> order) {
        this(name, devices, stages, microBatches, bidirectional,
                order.stream().flatMap(List::stream).anyMatch(o -> o.kind() == Kind.W), order);
    }

    /**
     * Creates a schedule whose chunks are placed in a V shape (Qi et al., NeurIPS 2024): chunk
     * {@code c} of device {@code d} holds stage {@code c p + d} for even {@code c} and
     * {@code c p + p - 1 - d} for odd {@code c}, so with two chunks every device holds one early
     * and one late stage and the first and last stages share device 0.
     *
     * @param name a label
     * @param devices the number of devices
     * @param stages the number of model stages; a multiple of {@code devices}
     * @param microBatches the number of micro-batches
     * @param vShape whether to use the V-shape placement (false: the loop placement of
     *     {@link #interleaved})
     * @param order the operations of each device, in execution order
     * @return the schedule
     */
    public static PipelineSchedule placed(String name, int devices, int stages, int microBatches, boolean vShape,
            List<List<Op>> order) {
        return new PipelineSchedule(name, devices, stages, microBatches, false,
                order.stream().flatMap(List::stream).anyMatch(o -> o.kind() == Kind.W), vShape, order);
    }

    private PipelineSchedule(String name, int devices, int stages, int microBatches, boolean bidirectional,
            boolean split, List<List<Op>> order) {
        this(name, devices, stages, microBatches, bidirectional, split, false, order);
    }

    private PipelineSchedule(String name, int devices, int stages, int microBatches, boolean bidirectional,
            boolean split, boolean vShape, List<List<Op>> order) {
        if (stages % devices != 0) {
            throw new IllegalArgumentException("stages " + stages + " must be a multiple of devices " + devices);
        }
        if (bidirectional && stages != devices) {
            throw new IllegalArgumentException("a bidirectional schedule has one stage per device");
        }
        this.name = name;
        this.devices = devices;
        this.stages = stages;
        this.microBatches = microBatches;
        this.bidirectional = bidirectional;
        this.split = split;
        this.vShape = vShape;
        List<List<Op>> copy = new ArrayList<>();
        for (List<Op> l : order) {
            copy.add(List.copyOf(l));
        }
        this.order = List.copyOf(copy);
    }

    /** @return the label */
    public String name() {
        return name;
    }

    /** @return the number of devices */
    public int devices() {
        return devices;
    }

    /** @return the number of model stages */
    public int stages() {
        return stages;
    }

    /** @return the number of chunks each device holds per direction */
    public int chunks() {
        return stages / devices;
    }

    /** @return the number of micro-batches */
    public int microBatches() {
        return microBatches;
    }

    /** @return whether micro-batches flow in both directions (DualPipe) */
    public boolean bidirectional() {
        return bidirectional;
    }

    /**
     * @param device a device
     * @return its operations in execution order
     */
    public List<Op> ops(int device) {
        return order.get(device);
    }

    /**
     * @param microBatch a micro-batch
     * @return the direction it flows in
     */
    public int direction(int microBatch) {
        return bidirectional && microBatch >= (microBatches + 1) / 2 ? 1 : 0;
    }

    /**
     * Where a stage runs.
     *
     * @param stage the stage
     * @param direction the direction
     * @return the device
     */
    public int device(int stage, int direction) {
        int d = stage % devices;
        if (vShape && (stage / devices) % 2 == 1) {
            d = devices - 1 - d;
        }
        return direction == 1 ? devices - 1 - d : d;
    }

    /** @return whether chunks are placed in a V shape */
    public boolean vShape() {
        return vShape;
    }

    /** @return whether the backward pass is split into B and W */
    public boolean split() {
        return split;
    }

    /**
     * The operations an operation waits for.
     *
     * @param op the operation
     * @return its dependencies
     */
    public List<Op> dependencies(Op op) {
        List<Op> deps = new ArrayList<>();
        int mb = op.microBatch();
        int s = op.stage();
        int dir = op.direction();
        Kind back = split() ? Kind.B : Kind.BW;
        switch (op.kind()) {
            case F -> {
                if (s > 0) {
                    deps.add(new Op(Kind.F, mb, s - 1, dir));
                }
            }
            case B, BW -> {
                deps.add(new Op(Kind.F, mb, s, dir));
                if (s < stages - 1) {
                    deps.add(new Op(back, mb, s + 1, dir));
                }
            }
            case W -> deps.add(new Op(Kind.B, mb, s, dir));
            default -> throw new IllegalStateException();
        }
        return deps;
    }

    /**
     * Checks that every operation appears exactly once, on the device holding its stage, and
     * that executing the per-device orders cannot deadlock.
     *
     * @throws IllegalStateException if not
     */
    public void validate() {
        boolean split = split();
        Map<Op, Integer> seen = new HashMap<>();
        for (int d = 0; d < devices; d++) {
            for (Op op : order.get(d)) {
                if (seen.put(op, d) != null) {
                    throw new IllegalStateException(name + ": " + op + " appears twice");
                }
                if (device(op.stage(), op.direction()) != d) {
                    throw new IllegalStateException(name + ": " + op + " on device " + d);
                }
                if (op.direction() != direction(op.microBatch())) {
                    throw new IllegalStateException(name + ": " + op + " in the wrong direction");
                }
            }
        }
        int perMicroBatch = stages * (split ? 3 : 2);
        if (seen.size() != perMicroBatch * microBatches) {
            throw new IllegalStateException(name + ": " + seen.size() + " operations, expected "
                    + perMicroBatch * microBatches);
        }
        // replay: repeatedly run any device whose next op has all dependencies done
        int[] next = new int[devices];
        java.util.Set<Op> done = new java.util.HashSet<>();
        int remaining = seen.size();
        boolean progress = true;
        while (remaining > 0 && progress) {
            progress = false;
            for (int d = 0; d < devices; d++) {
                while (next[d] < order.get(d).size() && done.containsAll(dependencies(order.get(d).get(next[d])))) {
                    done.add(order.get(d).get(next[d]++));
                    remaining--;
                    progress = true;
                }
            }
        }
        if (remaining > 0) {
            throw new IllegalStateException(name + ": deadlock with " + remaining + " operations left, next "
                    + Arrays.toString(next));
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(name).append('\n');
        for (int d = 0; d < devices; d++) {
            sb.append("  device ").append(d).append(": ").append(order.get(d)).append('\n');
        }
        return sb.toString();
    }

    // ---------------------------------------------------------------- generators

    /**
     * GPipe: all forwards in micro-batch order, then all backwards.
     *
     * @param devices pipeline depth {@code p}, one stage per device
     * @param microBatches {@code m}
     * @return the schedule
     */
    public static PipelineSchedule gpipe(int devices, int microBatches) {
        List<List<Op>> order = new ArrayList<>();
        for (int d = 0; d < devices; d++) {
            List<Op> ops = new ArrayList<>();
            for (int mb = 0; mb < microBatches; mb++) {
                ops.add(new Op(Kind.F, mb, d, 0));
            }
            for (int mb = 0; mb < microBatches; mb++) {
                ops.add(new Op(Kind.BW, mb, d, 0));
            }
            order.add(ops);
        }
        return new PipelineSchedule("GPipe", devices, devices, microBatches, false, order);
    }

    /**
     * 1F1B (PipeDream-Flush): {@code min(p-d-1, m)} warm-up forwards on device {@code d}, then
     * alternating forward and backward, then the remaining backwards.
     *
     * @param devices pipeline depth {@code p}
     * @param microBatches {@code m}
     * @return the schedule
     */
    public static PipelineSchedule oneFOneB(int devices, int microBatches) {
        List<List<Op>> order = new ArrayList<>();
        for (int d = 0; d < devices; d++) {
            List<Op> ops = new ArrayList<>();
            int warm = Math.min(devices - d - 1, microBatches);
            for (int mb = 0; mb < warm; mb++) {
                ops.add(new Op(Kind.F, mb, d, 0));
            }
            for (int i = 0; i < microBatches - warm; i++) {
                ops.add(new Op(Kind.F, warm + i, d, 0));
                ops.add(new Op(Kind.BW, i, d, 0));
            }
            for (int mb = microBatches - warm; mb < microBatches; mb++) {
                ops.add(new Op(Kind.BW, mb, d, 0));
            }
            order.add(ops);
        }
        return new PipelineSchedule("1F1B", devices, devices, microBatches, false, order);
    }

    /**
     * Megatron-LM's interleaved 1F1B: device {@code d} holds chunks {@code c} as stages
     * {@code c p + d}; micro-batches go through the chunks in groups of {@code p}.
     *
     * @param devices pipeline depth {@code p}
     * @param microBatches {@code m}, a multiple of {@code p}
     * @param chunks {@code v}, chunks per device
     * @return the schedule
     */
    public static PipelineSchedule interleaved(int devices, int microBatches, int chunks) {
        if (microBatches % devices != 0) {
            throw new IllegalArgumentException("interleaving needs micro-batches divisible by devices");
        }
        int p = devices;
        int total = microBatches * chunks;
        List<List<Op>> order = new ArrayList<>();
        for (int d = 0; d < p; d++) {
            List<Op> ops = new ArrayList<>();
            int warm = Math.min((p - d - 1) * 2 + (chunks - 1) * p, total);
            if (microBatches == p) {
                warm = total;   // Megatron: no steady state when m == p
            }
            for (int k = 0; k < warm; k++) {
                ops.add(interleavedOp(Kind.F, k, d, p, chunks));
            }
            for (int k = 0; k < total - warm; k++) {
                ops.add(interleavedOp(Kind.F, k + warm, d, p, chunks));
                ops.add(interleavedOp(Kind.BW, k, d, p, chunks));
            }
            for (int k = total - warm; k < total; k++) {
                ops.add(interleavedOp(Kind.BW, k, d, p, chunks));
            }
            order.add(ops);
        }
        return new PipelineSchedule("interleaved 1F1B v=" + chunks, p, p * chunks, microBatches, false, order);
    }

    private static Op interleavedOp(Kind kind, int k, int device, int p, int chunks) {
        int inGroup = k % (p * chunks);
        int chunk = inGroup / p;
        if (kind != Kind.F) {
            chunk = chunks - 1 - chunk;
        }
        int mb = k / (p * chunks) * p + k % p;
        return new Op(kind, mb, chunk * p + device, 0);
    }

    /**
     * A zero-bubble schedule from the greedy list scheduler with split backward passes.
     *
     * @param devices pipeline depth {@code p}
     * @param microBatches {@code m}
     * @param h2 false for ZB-H1 (at most {@code p - d} micro-batches in flight on device
     *     {@code d}, the memory of 1F1B), true for ZB-H2 (about twice that, for no bubble)
     * @param cost the per-operation times the scheduler plans with
     * @return the schedule
     */
    public static PipelineSchedule zeroBubble(int devices, int microBatches, boolean h2, ParallelCostModel cost) {
        int[] limit = new int[devices];
        for (int d = 0; d < devices; d++) {
            limit[d] = h2 ? 2 * (devices - d) - 1 : devices - d;
        }
        return greedy(h2 ? "ZB-H2" : "ZB-H1", devices, microBatches, false, true, limit, cost);
    }

    /**
     * DualPipe: half the micro-batches enter at device 0 and half at device {@code p-1}; each
     * device holds stage {@code d} of the first direction and stage {@code p-1-d} of the second
     * (twice the parameters), with split backward passes.
     *
     * @param devices pipeline depth {@code p}
     * @param microBatches {@code m}, even
     * @param cost the per-operation times the scheduler plans with
     * @return the schedule
     */
    public static PipelineSchedule dualPipe(int devices, int microBatches, ParallelCostModel cost) {
        int[] limit = new int[devices];
        Arrays.fill(limit, devices + 1);
        return greedy("DualPipe", devices, microBatches, true, true, limit, cost);
    }

    /**
     * The greedy list scheduler behind the zero-bubble and DualPipe schedules: devices are
     * simulated in time; a free device runs a ready backward (B or BW) first, else a ready
     * forward while fewer than {@code limit[d]} micro-batches hold activations on it, else a
     * pending W.
     *
     * @param name the label
     * @param devices {@code p}
     * @param microBatches {@code m}
     * @param bidirectional DualPipe's two directions
     * @param split whether backward passes split into B and W
     * @param limit the in-flight limit per device
     * @param cost operation and link times
     * @return the schedule
     */
    public static PipelineSchedule greedy(String name, int devices, int microBatches, boolean bidirectional,
            boolean split, int[] limit, ParallelCostModel cost) {
        int p = devices;
        PipelineSchedule shape = new PipelineSchedule(name, p, p, microBatches, bidirectional, split,
                emptyOrders(p));
        Map<Op, Double> end = new HashMap<>();
        List<List<Op>> order = emptyOrders(p);
        double[] free = new double[p];
        int[] inFlight = new int[p];
        List<List<Op>> pendingW = emptyOrders(p);
        int total = microBatches * p * (split ? 3 : 2);
        int scheduled = 0;
        Kind back = split ? Kind.B : Kind.BW;
        int idle = 0;
        while (scheduled < total) {
            if (idle > 100 * p) {
                throw new IllegalStateException(name + ": no device can make progress");
            }
            // every device has waited in turn without progress: the in-flight limits would deadlock
            // (all devices full of forwards whose backwards wait on each other), so let one exceed its limit
            boolean relax = idle > 4 * p;
            int d = 0;
            for (int i = 1; i < p; i++) {
                if (free[i] < free[d]) {
                    d = i;
                }
            }
            double t = free[d];
            // candidate forward and backward ops on device d, with their ready times
            Op bestB = null;
            double bestBReady = Double.POSITIVE_INFINITY;
            Op bestF = null;
            double bestFReady = Double.POSITIVE_INFINITY;
            boolean backwardOutstanding = false;
            for (int mb = 0; mb < microBatches; mb++) {
                int dir = shape.direction(mb);
                int stage = dir == 0 ? d : p - 1 - d;
                Op f = new Op(Kind.F, mb, stage, dir);
                Op b = new Op(back, mb, stage, dir);
                if (!end.containsKey(f)) {
                    double r = ready(shape, f, end, cost);
                    if (r < bestFReady) {
                        bestFReady = r;
                        bestF = f;
                    }
                } else if (!end.containsKey(b)) {
                    // a backward waiting on a later stage (ready time still unknown) frees memory later
                    backwardOutstanding = true;
                    double r = ready(shape, b, end, cost);
                    if (r < bestBReady) {
                        bestBReady = r;
                        bestB = b;
                    }
                }
            }
            Op run = null;
            if (bestB != null && bestBReady <= t) {
                run = bestB;
            } else if (bestF != null && bestFReady <= t && (relax || inFlight[d] < limit[d])) {
                run = bestF;
            } else if (!pendingW.get(d).isEmpty()) {
                run = pendingW.get(d).removeFirst();
            }
            if (run == null) {
                double wake = Double.POSITIVE_INFINITY;
                if (bestB != null && bestBReady > t) {
                    wake = Math.min(wake, bestBReady);
                }
                if (bestF != null && bestFReady > t && (relax || inFlight[d] < limit[d])) {
                    wake = Math.min(wake, bestFReady);
                }
                for (int i = 0; i < p; i++) {
                    if (free[i] > t) {
                        wake = Math.min(wake, free[i]);
                    }
                }
                if (wake == Double.POSITIVE_INFINITY) {
                    if (inFlight[d] >= limit[d] && bestF != null && !backwardOutstanding
                            && pendingW.get(d).isEmpty()) {
                        throw new IllegalStateException(name + ": deadlock, in-flight limit too small");
                    }
                    // waits on something not yet scheduled anywhere: let the others move first
                    wake = Math.nextUp(t);
                }
                free[d] = wake;
                idle++;
                continue;
            }
            idle = 0;
            double start = Math.max(t, run.kind() == Kind.W ? t : ready(shape, run, end, cost));
            double finish = start + cost.time(run.kind(), shape.chunks());
            if (!cost.overlapComm() && run.kind() != Kind.W && crossesDevice(shape, run)) {
                finish += cost.transferTime();
            }
            end.put(run, finish);
            free[d] = finish;
            order.get(d).add(run);
            scheduled++;
            switch (run.kind()) {
                case F -> inFlight[d]++;
                case B -> pendingW.get(d).add(new Op(Kind.W, run.microBatch(), run.stage(), run.direction()));
                case W, BW -> inFlight[d]--;
                default -> throw new IllegalStateException();
            }
        }
        PipelineSchedule s = new PipelineSchedule(name, p, p, microBatches, bidirectional, order);
        s.validate();
        return s;
    }

    private static boolean crossesDevice(PipelineSchedule s, Op op) {
        int d = s.device(op.stage(), op.direction());
        if (op.kind() == Kind.F) {
            return op.stage() < s.stages() - 1 && s.device(op.stage() + 1, op.direction()) != d;
        }
        return op.stage() > 0 && s.device(op.stage() - 1, op.direction()) != d;
    }

    private static double ready(PipelineSchedule s, Op op, Map<Op, Double> end, ParallelCostModel cost) {
        double r = 0;
        int d = s.device(op.stage(), op.direction());
        for (Op dep : s.dependencies(op)) {
            Double e = end.get(dep);
            if (e == null) {
                return Double.POSITIVE_INFINITY;
            }
            double arrive = e + (s.device(dep.stage(), dep.direction()) != d ? cost.transferTime() : 0);
            r = Math.max(r, arrive);
        }
        return r;
    }

    private static List<List<Op>> emptyOrders(int p) {
        List<List<Op>> o = new ArrayList<>();
        for (int d = 0; d < p; d++) {
            o.add(new ArrayList<>());
        }
        return o;
    }
}
