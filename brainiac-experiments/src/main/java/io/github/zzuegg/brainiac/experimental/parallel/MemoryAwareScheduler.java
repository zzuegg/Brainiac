package io.github.zzuegg.brainiac.experimental.parallel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Pipeline schedules with split backward passes (F, B, W) under a per-device activation
 * memory limit, as a {@link PipelineSchedule} that its validator accepts and
 * {@link PipelineRunner} can run.
 *
 * <p>A schedule is built by event-driven list scheduling: the device that can start an
 * operation earliest picks one of its candidates, the lowest unscheduled micro-batch of each
 * (kind, stage) whose dependencies are scheduled; a micro-batch's first forward on a device
 * reserves the activations of all its chunks there (one unit per micro-batch, {@code 1 / chunks}
 * per chunk), allowed only while the device's reservations stay within the limit, and each W
 * releases its chunk's share. Reserving whole micro-batches keeps the V-shape placement free of
 * deadlocks (a device full of first-chunk activations could otherwise never run a second-chunk
 * forward), and the actual peak never exceeds the reservation. {@link #greedy} takes the first candidate in the
 * order B, F, W (the zero-bubble heuristic, now honouring an explicit memory limit, so it can
 * use ZB-H2's extra memory, which wave 9's in-flight count did not). {@link #search} is a
 * depth-first branch and bound over all candidate choices with the greedy schedule as the
 * first incumbent and the bound {@code max_d (free_d + remaining work_d)}; when it finishes
 * within its node budget the schedule is optimal among these non-delay list schedules.
 *
 * <p>Paper: Qi et al., "Pipeline Parallelism with Controllable Memory", NeurIPS 2024,
 * <a href="https://arxiv.org/abs/2405.15362">arXiv:2405.15362</a>.
 */
public final class MemoryAwareScheduler {

    /**
     * A search result.
     *
     * @param schedule the best schedule found
     * @param makespan its planned step time
     * @param optimal whether the search finished (so no better list schedule exists)
     * @param nodes search nodes visited
     */
    public record Result(PipelineSchedule schedule, double makespan, boolean optimal, long nodes) {
    }

    private final int p;
    private final int stages;
    private final int m;
    private final int chunks;
    private final boolean vShape;
    private final double limit;
    private final ParallelCostModel cost;
    private final PipelineSchedule shape;
    private final double[] end;
    private final double[] free;
    private final double[] held;
    private final double[] remaining;
    private final int[][] nextMb;
    private final List<List<PipelineSchedule.Op>> order = new ArrayList<>();
    private final int[] stageDevice;
    private final boolean[] firstChunk;
    private double best = Double.MAX_VALUE;
    private List<List<PipelineSchedule.Op>> bestOrder;
    private long nodes;
    private long budget;
    private boolean exhausted;
    private boolean greedyOnly;

    private MemoryAwareScheduler(int devices, int stages, int microBatches, boolean vShape, double limit,
            ParallelCostModel cost) {
        this.p = devices;
        this.stages = stages;
        this.m = microBatches;
        this.chunks = stages / devices;
        this.vShape = vShape;
        this.limit = limit;
        this.cost = cost;
        List<List<PipelineSchedule.Op>> empty = new ArrayList<>();
        for (int d = 0; d < devices; d++) {
            empty.add(List.of());
            order.add(new ArrayList<>());
        }
        this.shape = PipelineSchedule.placed("shape", devices, stages, microBatches, vShape, empty);
        this.end = new double[3 * stages * microBatches];
        Arrays.fill(end, Double.NaN);
        this.free = new double[devices];
        this.held = new double[devices];
        this.remaining = new double[devices];
        this.nextMb = new int[3][stages];
        this.stageDevice = new int[stages];
        this.firstChunk = new boolean[stages];
        boolean[] seen = new boolean[devices];
        for (int s = 0; s < stages; s++) {
            stageDevice[s] = shape.device(s, 0);
            firstChunk[s] = !seen[stageDevice[s]];
            seen[stageDevice[s]] = true;
            remaining[stageDevice[s]] += m * (time(0) + time(1) + time(2));
        }
    }

    private double time(int kind) {
        return cost.time(kind == 0 ? PipelineSchedule.Kind.F : kind == 1 ? PipelineSchedule.Kind.B
                : PipelineSchedule.Kind.W, chunks);
    }

    private int id(int kind, int stage, int mb) {
        return (kind * stages + stage) * m + mb;
    }

    /** The earliest start of an op whose dependencies are scheduled, or NaN. */
    private double ready(int kind, int stage, int mb) {
        int d = stageDevice[stage];
        double r = free[d];
        double t = cost.transferTime();
        switch (kind) {
            case 0 -> {
                if (stage > 0) {
                    double e = end[id(0, stage - 1, mb)];
                    if (Double.isNaN(e)) {
                        return Double.NaN;
                    }
                    r = Math.max(r, e + (stageDevice[stage - 1] != d ? t : 0));
                }
            }
            case 1 -> {
                double f = end[id(0, stage, mb)];
                if (Double.isNaN(f)) {
                    return Double.NaN;
                }
                r = Math.max(r, f);
                if (stage < stages - 1) {
                    double e = end[id(1, stage + 1, mb)];
                    if (Double.isNaN(e)) {
                        return Double.NaN;
                    }
                    r = Math.max(r, e + (stageDevice[stage + 1] != d ? t : 0));
                }
            }
            default -> {
                double b = end[id(1, stage, mb)];
                if (Double.isNaN(b)) {
                    return Double.NaN;
                }
                r = Math.max(r, b);
            }
        }
        return r;
    }

    private record Candidate(int kind, int stage, int mb, double start) {
    }

    private List<Candidate> candidates(int d) {
        List<Candidate> c = new ArrayList<>();
        int[] priority = {1, 0, 2};
        for (int kind : priority) {
            for (int s = 0; s < stages; s++) {
                if (stageDevice[s] != d || nextMb[kind][s] >= m) {
                    continue;
                }
                int mb = nextMb[kind][s];
                if (kind == 0 && firstChunk[s] && held[d] + 1.0 > limit + 1e-9) {
                    continue;
                }
                double r = ready(kind, s, mb);
                if (!Double.isNaN(r)) {
                    c.add(new Candidate(kind, s, mb, r));
                }
            }
        }
        return c;
    }

    private void dfs(int left) {
        if (nodes++ > budget) {
            exhausted = true;
            return;
        }
        if (left == 0) {
            double makespan = Arrays.stream(free).max().orElse(0);
            if (makespan < best - 1e-12) {
                best = makespan;
                bestOrder = new ArrayList<>();
                for (List<PipelineSchedule.Op> l : order) {
                    bestOrder.add(new ArrayList<>(l));
                }
            }
            if (greedyOnly) {
                exhausted = true;
            }
            return;
        }
        double bound = 0;
        for (int d = 0; d < p; d++) {
            bound = Math.max(bound, free[d] + remaining[d]);
        }
        if (bound >= best - 1e-12) {
            return;
        }
        int device = -1;
        double earliest = Double.MAX_VALUE;
        List<Candidate> chosen = null;
        for (int d = 0; d < p; d++) {
            List<Candidate> c = candidates(d);
            for (Candidate x : c) {
                if (x.start() < earliest - 1e-12) {
                    earliest = x.start();
                    device = d;
                    chosen = c;
                }
            }
        }
        if (device < 0) {
            return; // blocked by the memory limit
        }
        int d = device;
        for (Candidate x : chosen) {
            double t = time(x.kind());
            double oldFree = free[d];
            double oldHeld = held[d];
            int idx = id(x.kind(), x.stage(), x.mb());
            end[idx] = x.start() + t;
            free[d] = x.start() + t;
            remaining[d] -= t;
            held[d] += x.kind() == 0 ? (firstChunk[x.stage()] ? 1.0 : 0) : x.kind() == 2 ? -1.0 / chunks : 0;
            nextMb[x.kind()][x.stage()]++;
            PipelineSchedule.Kind k = x.kind() == 0 ? PipelineSchedule.Kind.F : x.kind() == 1 ? PipelineSchedule.Kind.B
                    : PipelineSchedule.Kind.W;
            order.get(d).add(new PipelineSchedule.Op(k, x.mb(), x.stage(), 0));
            dfs(left - 1);
            order.get(d).remove(order.get(d).size() - 1);
            nextMb[x.kind()][x.stage()]--;
            held[d] = oldHeld;
            remaining[d] += t;
            free[d] = oldFree;
            end[idx] = Double.NaN;
            if (exhausted) {
                return;
            }
        }
    }

    private Result run(String name, long nodeBudget) {
        budget = nodeBudget;
        dfs(3 * stages * m);
        if (bestOrder == null) {
            throw new IllegalStateException("no schedule fits an activation limit of " + limit);
        }
        PipelineSchedule s = PipelineSchedule.placed(name, p, stages, m, vShape, bestOrder);
        s.validate();
        return new Result(s, best, !exhausted, nodes);
    }

    /**
     * The greedy list schedule (first leaf of the search).
     *
     * @param name the label
     * @param devices {@code p}
     * @param stages the model stages (a multiple of {@code p})
     * @param microBatches {@code m}
     * @param vShape the V-shape placement of chunks
     * @param memoryLimit per-device activations, in stage-micro-batch units (1F1B's device 0
     *     holds {@code p})
     * @param cost the operation and link times
     * @return the schedule
     */
    public static PipelineSchedule greedy(String name, int devices, int stages, int microBatches, boolean vShape,
            double memoryLimit, ParallelCostModel cost) {
        MemoryAwareScheduler s = new MemoryAwareScheduler(devices, stages, microBatches, vShape, memoryLimit, cost);
        s.budget = Long.MAX_VALUE;
        s.greedyOnly = true;
        s.dfs(3 * stages * microBatches);
        if (s.bestOrder == null) {
            throw new IllegalStateException("no greedy schedule fits an activation limit of " + memoryLimit);
        }
        PipelineSchedule out = PipelineSchedule.placed(name, devices, stages, microBatches, vShape, s.bestOrder);
        out.validate();
        return out;
    }

    /**
     * Branch and bound for the shortest step under the memory limit.
     *
     * @param name the label
     * @param devices {@code p}
     * @param stages the model stages
     * @param microBatches {@code m}
     * @param vShape the V-shape placement
     * @param memoryLimit per-device activations, in stage-micro-batch units
     * @param cost the operation and link times
     * @param nodeBudget the most search nodes
     * @return the best schedule found
     */
    public static Result search(String name, int devices, int stages, int microBatches, boolean vShape,
            double memoryLimit, ParallelCostModel cost, long nodeBudget) {
        return new MemoryAwareScheduler(devices, stages, microBatches, vShape, memoryLimit, cost).run(name, nodeBudget);
    }
}
