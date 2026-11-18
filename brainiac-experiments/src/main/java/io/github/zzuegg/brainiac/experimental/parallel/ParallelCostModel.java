package io.github.zzuegg.brainiac.experimental.parallel;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A cost model for parallel training, calibrated with measured CPU times: how long a stage's
 * forward pass, input-gradient pass (B) and weight-gradient pass (W) take, and how long a
 * point-to-point message of one micro-batch's activations takes (latency plus bytes over
 * bandwidth). {@link #simulate} replays a {@link PipelineSchedule} with these times and reports
 * the step time, the idle ("bubble") fraction, the peak number of micro-batches whose
 * activations each device holds, and the messages and bytes sent.
 *
 * <p>With {@link #overlapComm()} a device computes while its messages travel (as with
 * asynchronous sends on separate streams); without it, a device is busy for the transfer after
 * each operation whose output goes to another device. {@link #collectiveTime} prices ring
 * collectives the same way, for data, tensor and expert parallelism.
 */
public final class ParallelCostModel {

    /**
     * The outcome of replaying a schedule.
     *
     * @param stepTime the time until the last device finishes
     * @param bubble the idle fraction {@code 1 - busy / (devices * stepTime)}
     * @param peakActivations per device, the most micro-batch activations held at once, in
     *     units of one stage's activations for one micro-batch
     * @param messages point-to-point messages between different devices
     * @param bytes bytes those messages carry
     */
    public record Result(double stepTime, double bubble, double[] peakActivations, long messages, long bytes) {

        /** @return the largest per-device peak */
        public double maxPeakActivations() {
            return Arrays.stream(peakActivations).max().orElse(0);
        }
    }

    private final double forward;
    private final double backwardInput;
    private final double backwardWeight;
    private final double latency;
    private final double bytesPerSecond;
    private final long messageBytes;
    private final boolean overlapComm;

    /**
     * Creates a model from known times.
     *
     * @param forward one stage's forward pass for one micro-batch, seconds
     * @param backwardInput its input-gradient pass, seconds
     * @param backwardWeight its weight-gradient pass, seconds
     * @param latency per-message latency, seconds
     * @param bytesPerSecond link bandwidth
     * @param messageBytes bytes of one micro-batch's activations at a stage boundary
     * @param overlapComm whether devices compute while messages travel
     */
    public ParallelCostModel(double forward, double backwardInput, double backwardWeight, double latency,
            double bytesPerSecond, long messageBytes, boolean overlapComm) {
        this.forward = forward;
        this.backwardInput = backwardInput;
        this.backwardWeight = backwardWeight;
        this.latency = latency;
        this.bytesPerSecond = bytesPerSecond;
        this.messageBytes = messageBytes;
        this.overlapComm = overlapComm;
    }

    /**
     * Calibrates the compute times by running the three passes (median of {@code repeats}
     * timed runs after one warm-up).
     *
     * @param forward runs one stage's forward pass on one micro-batch
     * @param backwardInput runs its input-gradient pass
     * @param backwardWeight runs its weight-gradient pass
     * @param repeats timed runs, at least 1
     * @param latency per-message latency, seconds
     * @param bytesPerSecond link bandwidth
     * @param messageBytes bytes per boundary message
     * @param overlapComm whether devices compute while messages travel
     * @return the model
     */
    public static ParallelCostModel calibrate(Runnable forward, Runnable backwardInput, Runnable backwardWeight,
            int repeats, double latency, double bytesPerSecond, long messageBytes, boolean overlapComm) {
        return new ParallelCostModel(median(forward, repeats), median(backwardInput, repeats),
                median(backwardWeight, repeats), latency, bytesPerSecond, messageBytes, overlapComm);
    }

    /**
     * The median wall time of a task.
     *
     * @param task the task
     * @param repeats timed runs after one warm-up
     * @return seconds
     */
    public static double median(Runnable task, int repeats) {
        task.run();
        double[] t = new double[Math.max(1, repeats)];
        for (int i = 0; i < t.length; i++) {
            long s = System.nanoTime();
            task.run();
            t[i] = (System.nanoTime() - s) / 1e9;
        }
        Arrays.sort(t);
        return t[t.length / 2];
    }

    /**
     * @param overlap whether devices compute while messages travel
     * @return a copy with that setting
     */
    public ParallelCostModel withOverlap(boolean overlap) {
        return new ParallelCostModel(forward, backwardInput, backwardWeight, latency, bytesPerSecond, messageBytes,
                overlap);
    }

    /** @return whether devices compute while messages travel */
    public boolean overlapComm() {
        return overlapComm;
    }

    /** @return the forward time of one stage */
    public double forward() {
        return forward;
    }

    /** @return the input-gradient time of one stage */
    public double backwardInput() {
        return backwardInput;
    }

    /** @return the weight-gradient time of one stage */
    public double backwardWeight() {
        return backwardWeight;
    }

    /** @return the time of one boundary message */
    public double transferTime() {
        return latency + messageBytes / bytesPerSecond;
    }

    /**
     * The time of a ring collective.
     *
     * @param bytesPerRank what one rank sends (from {@link Collectives})
     * @param steps sequential ring steps, each paying the latency
     * @return seconds
     */
    public double collectiveTime(long bytesPerRank, int steps) {
        return steps * latency + bytesPerRank / bytesPerSecond;
    }

    /**
     * The compute time of an operation.
     *
     * @param kind the operation
     * @param chunks chunks per device; a chunk is {@code 1 / chunks} of a stage
     * @return seconds
     */
    public double time(PipelineSchedule.Kind kind, int chunks) {
        double t = switch (kind) {
            case F -> forward;
            case B -> backwardInput;
            case W -> backwardWeight;
            case BW -> backwardInput + backwardWeight;
        };
        return t / chunks;
    }

    /**
     * Replays a schedule: every device runs its operations in order, each starting once the
     * device is free and its inputs have arrived.
     *
     * @param s the schedule
     * @return the timing, memory and traffic
     */
    public Result simulate(PipelineSchedule s) {
        int p = s.devices();
        int chunks = s.chunks();
        Map<PipelineSchedule.Op, Double> end = new HashMap<>();
        int[] next = new int[p];
        double[] free = new double[p];
        double[] busy = new double[p];
        double[] held = new double[p];
        double[] peak = new double[p];
        long messages = 0;
        int remaining = 0;
        for (int d = 0; d < p; d++) {
            remaining += s.ops(d).size();
        }
        while (remaining > 0) {
            boolean progress = false;
            for (int d = 0; d < p; d++) {
                List<PipelineSchedule.Op> ops = s.ops(d);
                while (next[d] < ops.size()) {
                    PipelineSchedule.Op op = ops.get(next[d]);
                    double ready = 0;
                    boolean ok = true;
                    for (PipelineSchedule.Op dep : s.dependencies(op)) {
                        Double e = end.get(dep);
                        if (e == null) {
                            ok = false;
                            break;
                        }
                        boolean remote = s.device(dep.stage(), dep.direction()) != d;
                        ready = Math.max(ready, e + (remote ? transferTime() : 0));
                    }
                    if (!ok) {
                        break;
                    }
                    double start = Math.max(free[d], ready);
                    double compute = time(op.kind(), chunks);
                    double finish = start + compute;
                    boolean sends = sendsRemote(s, op, d);
                    if (sends) {
                        messages++;
                        if (!overlapComm) {
                            finish += transferTime();
                        }
                    }
                    busy[d] += compute;
                    free[d] = finish;
                    end.put(op, finish);
                    switch (op.kind()) {
                        case F -> {
                            held[d] += 1.0 / chunks;
                            peak[d] = Math.max(peak[d], held[d]);
                        }
                        case W, BW -> held[d] -= 1.0 / chunks;
                        default -> {
                        }
                    }
                    next[d]++;
                    remaining--;
                    progress = true;
                }
            }
            if (!progress) {
                throw new IllegalStateException(s.name() + " deadlocks");
            }
        }
        double step = 0;
        double work = 0;
        for (int d = 0; d < p; d++) {
            step = Math.max(step, free[d]);
            work += busy[d];
        }
        return new Result(step, 1 - work / (p * step), peak, messages, messages * messageBytes);
    }

    private static boolean sendsRemote(PipelineSchedule s, PipelineSchedule.Op op, int d) {
        return switch (op.kind()) {
            case F -> op.stage() < s.stages() - 1 && s.device(op.stage() + 1, op.direction()) != d;
            case B, BW -> op.stage() > 0 && s.device(op.stage() - 1, op.direction()) != d;
            case W -> false;
        };
    }

    /**
     * The textbook bubble fraction of GPipe and 1F1B.
     *
     * @param devices {@code p}
     * @param microBatches {@code m}
     * @param chunks {@code v} for interleaving, else 1
     * @return {@code (p - 1) / (v m + p - 1)}
     */
    public static double theoreticalBubble(int devices, int microBatches, int chunks) {
        return (devices - 1.0) / ((double) chunks * microBatches + devices - 1);
    }
}
