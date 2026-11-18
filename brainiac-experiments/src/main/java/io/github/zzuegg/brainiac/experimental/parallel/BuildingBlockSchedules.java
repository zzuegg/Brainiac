package io.github.zzuegg.brainiac.experimental.parallel;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Pipeline schedules built by repeating one building block (Qi et al., "Pipeline Parallelism
 * with Controllable Memory", NeurIPS 2024), on the V-shape placement where each device holds
 * two chunks, stages {@code d} and {@code 2p - 1 - d} ({@link PipelineSchedule#placed}).
 *
 * <p>The block is the planned timeline of one micro-batch's {@code 6p} operations (F, B, W per
 * chunk): forwards run down the stages, the second chunk's forwards delayed by {@code delta0};
 * backward input passes run back up; every W starts {@code delta1} after its B. Micro-batch
 * {@code i} repeats the block shifted by {@code i T}. Each device runs its operations in
 * planned order, so the activation memory of a chunk is set by its lifespan (from its F to the
 * end of its W) divided by the interval {@code T}: a short lifespan means little memory, and the
 * offsets trade memory against bubbles. {@link #build} constructs one schedule from the offsets;
 * {@link #vMin}, {@link #vHalf} and {@link #vZb} search a small grid of offsets and intervals
 * for the lowest simulated bubble whose peak activation memory stays within about 1/3, 1/2 and
 * all of 1F1B's (the paper's V-Min, V-Half and V-ZB targets; the paper derives its offsets in
 * closed form, here they are searched).
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2405.15362">arXiv:2405.15362</a>.
 */
public final class BuildingBlockSchedules {

    private BuildingBlockSchedules() {
    }

    private record Planned(PipelineSchedule.Op op, double time, int kind) {
    }

    /**
     * Builds a V-shape schedule from a building block.
     *
     * @param name the label
     * @param devices {@code p}
     * @param microBatches {@code m}
     * @param delta0 the delay of the second chunk's forward passes
     * @param delta1 the delay of each W after its B
     * @param interval the repetition interval {@code T}
     * @param cost the operation times (a chunk is half a stage)
     * @return the schedule
     */
    public static PipelineSchedule build(String name, int devices, int microBatches, double delta0, double delta1,
            double interval, ParallelCostModel cost) {
        int p = devices;
        int stages = 2 * p;
        double f = cost.time(PipelineSchedule.Kind.F, 2);
        double b = cost.time(PipelineSchedule.Kind.B, 2);
        double[] tf = new double[stages];
        double[] tb = new double[stages];
        for (int s = 0; s < stages; s++) {
            tf[s] = s * f + (s >= p ? delta0 : 0);
        }
        tb[stages - 1] = tf[stages - 1] + f;
        for (int s = stages - 2; s >= 0; s--) {
            tb[s] = tb[s + 1] + b;
        }
        List<List<Planned>> per = new ArrayList<>();
        for (int d = 0; d < p; d++) {
            per.add(new ArrayList<>());
        }
        PipelineSchedule shape = PipelineSchedule.placed("shape", p, stages, microBatches, true, emptyOrders(p));
        for (int i = 0; i < microBatches; i++) {
            double shift = i * interval;
            for (int s = 0; s < stages; s++) {
                int d = shape.device(s, 0);
                per.get(d).add(new Planned(new PipelineSchedule.Op(PipelineSchedule.Kind.F, i, s, 0), tf[s] + shift, 1));
                per.get(d).add(new Planned(new PipelineSchedule.Op(PipelineSchedule.Kind.B, i, s, 0), tb[s] + shift, 0));
                per.get(d).add(new Planned(new PipelineSchedule.Op(PipelineSchedule.Kind.W, i, s, 0),
                        tb[s] + b + delta1 + shift, 2));
            }
        }
        List<List<PipelineSchedule.Op>> order = new ArrayList<>();
        for (List<Planned> l : per) {
            l.sort(Comparator.comparingDouble(Planned::time).thenComparingInt(Planned::kind));
            List<PipelineSchedule.Op> ops = new ArrayList<>();
            for (Planned x : l) {
                ops.add(x.op());
            }
            order.add(ops);
        }
        PipelineSchedule s = PipelineSchedule.placed(name, p, stages, microBatches, true, order);
        s.validate();
        return s;
    }

    private static List<List<PipelineSchedule.Op>> emptyOrders(int p) {
        List<List<PipelineSchedule.Op>> o = new ArrayList<>();
        for (int d = 0; d < p; d++) {
            o.add(List.of());
        }
        return o;
    }

    /**
     * Searches offsets and intervals for the lowest simulated bubble under a memory target.
     *
     * @param name the label
     * @param devices {@code p}
     * @param microBatches {@code m}
     * @param memoryFraction the largest peak activation memory as a fraction of 1F1B's ({@code p}
     *     stage-micro-batch units)
     * @param cost the operation and link times
     * @return the best schedule found, or the one with the least memory if none meets the target
     */
    public static PipelineSchedule search(String name, int devices, int microBatches, double memoryFraction,
            ParallelCostModel cost) {
        double f = cost.time(PipelineSchedule.Kind.F, 2);
        double b = cost.time(PipelineSchedule.Kind.B, 2);
        double w = cost.time(PipelineSchedule.Kind.W, 2);
        double work = 2 * (f + b + w);
        double target = memoryFraction * devices + 1e-9;
        PipelineSchedule best = null;
        double bestBubble = Double.MAX_VALUE;
        PipelineSchedule leanest = null;
        double leanestMemory = Double.MAX_VALUE;
        for (double tScale : new double[] {1.0, 1.15, 1.3, 1.5, 2.0, 3.0}) {
            for (int k0 = 0; k0 <= 2 * devices; k0++) {
                for (int k1 = 0; k1 <= 2 * devices; k1++) {
                    PipelineSchedule s = build(name, devices, microBatches, k0 * f, k1 * w, tScale * work, cost);
                    ParallelCostModel.Result r = cost.simulate(s);
                    double mem = r.maxPeakActivations();
                    if (mem < leanestMemory - 1e-9 || (Math.abs(mem - leanestMemory) < 1e-9 && leanest != null
                            && r.bubble() < cost.simulate(leanest).bubble())) {
                        leanestMemory = mem;
                        leanest = s;
                    }
                    if (mem <= target && r.bubble() < bestBubble - 1e-12) {
                        bestBubble = r.bubble();
                        best = s;
                    }
                }
            }
        }
        return best != null ? best : leanest;
    }

    /**
     * V-Min: about a third of 1F1B's activation memory.
     *
     * @param devices {@code p}
     * @param microBatches {@code m}
     * @param cost the costs
     * @return the schedule
     */
    public static PipelineSchedule vMin(int devices, int microBatches, ParallelCostModel cost) {
        return search("V-Min", devices, microBatches, 1.0 / 3 + 1.0 / (2 * devices), cost);
    }

    /**
     * V-Half: about half of 1F1B's activation memory.
     *
     * @param devices {@code p}
     * @param microBatches {@code m}
     * @param cost the costs
     * @return the schedule
     */
    public static PipelineSchedule vHalf(int devices, int microBatches, ParallelCostModel cost) {
        return search("V-Half", devices, microBatches, 0.5 + 1.0 / (2 * devices), cost);
    }

    /**
     * V-ZB: 1F1B's activation memory, nearly no bubble.
     *
     * @param devices {@code p}
     * @param microBatches {@code m}
     * @param cost the costs
     * @return the schedule
     */
    public static PipelineSchedule vZb(int devices, int microBatches, ParallelCostModel cost) {
        return search("V-ZB", devices, microBatches, 1.0, cost);
    }
}
