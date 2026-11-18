package io.github.zzuegg.brainiac.experimental.parallel;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.function.BiFunction;

/**
 * Executes a {@link PipelineSchedule} for real: one Java thread per device, a blocking queue per
 * directed device link carrying activations forward and gradients backward, and the stages'
 * ordinary {@link Layer}s doing the work. Gradients accumulate over the micro-batches, so after
 * {@link #run} each stage's parameters hold the gradient of the mean loss over the whole batch,
 * which equals single-device training to float rounding; {@link #referenceGradients} computes
 * that single-device gradient for the check.
 *
 * <p>Split schedules (B and W) need stages built from {@link SplitBackward} layers, with one tape
 * per stage. DualPipe needs a second copy of every stage for the reverse direction (the same
 * initial weights); the gradients of the two copies are summed. The runner also measures, per
 * device, the time spent computing, so the measured bubble {@code 1 - busy / (p * wall)} can be
 * compared with the cost model's.
 */
public final class PipelineRunner {

    /**
     * The outcome of a step.
     *
     * @param loss the mean loss over the batch
     * @param gradients per stage, the gradients of its parameters (copies summed for DualPipe)
     * @param wallSeconds the step's wall time
     * @param busySeconds per device, the time spent computing
     * @param messages messages sent between devices
     * @param bytes bytes those messages carried
     */
    public record Result(float loss, List<List<Tensor>> gradients, double wallSeconds, double[] busySeconds,
            long messages, long bytes) {

        /** @return the measured idle fraction {@code 1 - sum(busy) / (p * wall)} */
        public double measuredBubble() {
            double s = 0;
            for (double b : busySeconds) {
                s += b;
            }
            return 1 - s / (busySeconds.length * wallSeconds);
        }
    }

    private record Message(PipelineSchedule.Op from, Tensor value) {
    }

    private final List<Layer> stages;
    private final List<Layer> reverse;
    private final List<SplitBackward> tapes;
    private final List<SplitBackward> reverseTapes;
    private final BiFunction<Var, Tensor, Var> loss;

    /**
     * Creates a runner.
     *
     * @param stages the model's stages in order
     * @param tapes one tape per stage for split schedules, or {@code null}
     * @param reverse DualPipe's second copy of the stages (same weights), or {@code null}
     * @param reverseTapes tapes of the second copy, or {@code null}
     * @param loss the mean loss of a micro-batch from the last stage's output and its targets
     */
    public PipelineRunner(List<Layer> stages, List<SplitBackward> tapes, List<Layer> reverse,
            List<SplitBackward> reverseTapes, BiFunction<Var, Tensor, Var> loss) {
        this.stages = List.copyOf(stages);
        this.tapes = tapes == null ? null : List.copyOf(tapes);
        this.reverse = reverse == null ? null : List.copyOf(reverse);
        this.reverseTapes = reverseTapes == null ? null : List.copyOf(reverseTapes);
        this.loss = loss;
    }

    private Layer stage(int s, int dir) {
        return dir == 1 ? reverse.get(s) : stages.get(s);
    }

    private SplitBackward tape(int s, int dir) {
        List<SplitBackward> t = dir == 1 ? reverseTapes : tapes;
        return t == null ? null : t.get(s);
    }

    /**
     * Runs one step's forward and backward passes (no optimizer step); the stages' gradients
     * are cleared first.
     *
     * @param schedule the schedule, whose stage count matches the stages
     * @param inputs the micro-batches' inputs
     * @param targets the micro-batches' targets
     * @return the loss, gradients and timing
     */
    public Result run(PipelineSchedule schedule, List<Tensor> inputs, List<Tensor> targets) {
        int p = schedule.devices();
        int m = schedule.microBatches();
        if (schedule.stages() != stages.size() || inputs.size() != m) {
            throw new IllegalArgumentException("schedule does not match the stages or micro-batches");
        }
        if (schedule.bidirectional() && reverse == null) {
            throw new IllegalArgumentException("DualPipe needs the reverse copy of the stages");
        }
        if (schedule.split() && tapes == null) {
            throw new IllegalArgumentException("a split schedule needs SplitBackward stages");
        }
        clear(stages);
        if (reverse != null) {
            clear(reverse);
        }
        long rows = 0;
        for (Tensor x : inputs) {
            rows += x.shape().dim(0);
        }
        long totalRows = rows;
        List<List<LinkedBlockingQueue<Message>>> links = new ArrayList<>();
        for (int i = 0; i < p; i++) {
            List<LinkedBlockingQueue<Message>> row = new ArrayList<>();
            for (int j = 0; j < p; j++) {
                row.add(new LinkedBlockingQueue<>());
            }
            links.add(row);
        }
        long[] traffic = new long[2];
        long start = System.nanoTime();
        List<double[]> perDevice = Collectives.run(p, d -> {
            Map<PipelineSchedule.Op, Message> stash = new HashMap<>();
            Map<PipelineSchedule.Op, Var[]> saved = new HashMap<>();
            double busy = 0;
            double lossSum = 0;
            long msgs = 0;
            long bytes = 0;
            for (PipelineSchedule.Op op : schedule.ops(d)) {
                int mb = op.microBatch();
                int s = op.stage();
                int dir = op.direction();
                Tensor received = null;
                boolean last = s == schedule.stages() - 1;
                if (op.kind() == PipelineSchedule.Kind.F && s > 0) {
                    received = await(links, stash, schedule, d, new PipelineSchedule.Op(PipelineSchedule.Kind.F, mb,
                            s - 1, dir));
                } else if (op.kind() != PipelineSchedule.Kind.F && op.kind() != PipelineSchedule.Kind.W && !last) {
                    PipelineSchedule.Kind back = schedule.split() ? PipelineSchedule.Kind.B : PipelineSchedule.Kind.BW;
                    received = await(links, stash, schedule, d, new PipelineSchedule.Op(back, mb, s + 1, dir));
                }
                long t0 = System.nanoTime();
                Tensor send = null;
                PipelineSchedule.Op key = new PipelineSchedule.Op(PipelineSchedule.Kind.F, mb, s, dir);
                switch (op.kind()) {
                    case F -> {
                        Var in = Var.param(s == 0 ? inputs.get(mb) : received);
                        Var out = stage(s, dir).forward(in, true);
                        if (last) {
                            float w = (float) inputs.get(mb).shape().dim(0) / totalRows;
                            out = loss.apply(out, targets.get(mb)).mul(w);
                            lossSum += out.value().item();
                        } else {
                            send = out.value();
                        }
                        saved.put(key, new Var[] {in, out});
                    }
                    case B, BW -> {
                        Var[] ctx = saved.get(key);
                        Tensor seed = last ? Tensor.scalar(1f) : received;
                        SplitBackward tape = tape(s, dir);
                        if (tape != null) {
                            tape.backwardInput(mb, ctx[1], seed, ctx[0]);
                            if (op.kind() == PipelineSchedule.Kind.BW) {
                                tape.backwardWeight(mb);
                            }
                        } else {
                            ctx[1].backward(seed);
                        }
                        if (s > 0) {
                            send = ctx[0].grad();
                        }
                        if (op.kind() == PipelineSchedule.Kind.BW) {
                            saved.remove(key);
                        }
                    }
                    case W -> {
                        tape(s, dir).backwardWeight(mb);
                        saved.remove(key);
                    }
                    default -> throw new IllegalStateException();
                }
                busy += (System.nanoTime() - t0) / 1e9;
                if (send != null) {
                    int to = op.kind() == PipelineSchedule.Kind.F ? schedule.device(s + 1, dir)
                            : schedule.device(s - 1, dir);
                    links.get(d).get(to).add(new Message(op, send));
                    if (to != d) {
                        msgs++;
                        bytes += send.size() * Float.BYTES;
                    }
                }
            }
            return new double[] {busy, lossSum, msgs, bytes};
        });
        double wall = (System.nanoTime() - start) / 1e9;
        double[] busy = new double[p];
        double total = 0;
        for (int d = 0; d < p; d++) {
            busy[d] = perDevice.get(d)[0];
            total += perDevice.get(d)[1];
            traffic[0] += (long) perDevice.get(d)[2];
            traffic[1] += (long) perDevice.get(d)[3];
        }
        List<List<Tensor>> grads = new ArrayList<>();
        for (int s = 0; s < stages.size(); s++) {
            List<Tensor> g = new ArrayList<>();
            List<Var> ps = stages.get(s).parameters();
            for (int i = 0; i < ps.size(); i++) {
                Tensor a = ps.get(i).grad();
                if (reverse != null) {
                    Tensor b = reverse.get(s).parameters().get(i).grad();
                    a = a == null ? b : b == null ? a : a.add(b);
                }
                g.add(a);
            }
            grads.add(g);
        }
        return new Result((float) total, grads, wall, busy, traffic[0], traffic[1]);
    }

    private static Tensor await(List<List<LinkedBlockingQueue<Message>>> links, Map<PipelineSchedule.Op, Message> stash,
            PipelineSchedule schedule, int d, PipelineSchedule.Op from) {
        int src = schedule.device(from.stage(), from.direction());
        while (true) {
            Message m = stash.remove(from);
            if (m != null) {
                return m.value();
            }
            try {
                Message got = links.get(src).get(d).poll(120, java.util.concurrent.TimeUnit.SECONDS);
                if (got == null) {
                    throw new IllegalStateException("device " + d + " timed out waiting for " + from);
                }
                stash.put(got.from(), got);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException(e);
            }
        }
    }

    private static void clear(List<Layer> layers) {
        for (Layer l : layers) {
            for (Var v : l.parameters()) {
                v.zeroGrad();
            }
        }
    }

    /**
     * The single-device gradient of the mean loss over the whole batch, for the exactness
     * check: all stages in sequence on the concatenated micro-batches.
     *
     * @param stages the stages
     * @param inputs the micro-batches' inputs
     * @param targets their targets
     * @param loss the mean loss
     * @return the loss and, per stage, its parameters' gradients
     */
    public static Result referenceGradients(List<Layer> stages, List<Tensor> inputs, List<Tensor> targets,
            BiFunction<Var, Tensor, Var> loss) {
        clear(stages);
        long t0 = System.nanoTime();
        Var x = Var.param(Tensor.concat(0, inputs.toArray(Tensor[]::new)));
        Var h = x;
        for (Layer l : stages) {
            h = l.forward(h, true);
        }
        Var l = loss.apply(h, Tensor.concat(0, targets.toArray(Tensor[]::new)));
        l.backward();
        double wall = (System.nanoTime() - t0) / 1e9;
        List<List<Tensor>> grads = new ArrayList<>();
        for (Layer s : stages) {
            List<Tensor> g = new ArrayList<>();
            for (Var v : s.parameters()) {
                g.add(v.grad());
            }
            grads.add(g);
        }
        return new Result(l.value().item(), grads, wall, new double[] {wall}, 0, 0);
    }

    /**
     * The largest absolute difference between two sets of gradients.
     *
     * @param a gradients per stage
     * @param b gradients per stage
     * @return the largest elementwise difference
     */
    public static double maxDifference(List<List<Tensor>> a, List<List<Tensor>> b) {
        double worst = 0;
        for (int s = 0; s < a.size(); s++) {
            for (int i = 0; i < a.get(s).size(); i++) {
                float[] x = a.get(s).get(i).toArray();
                float[] y = b.get(s).get(i).toArray();
                for (int j = 0; j < x.length; j++) {
                    worst = Math.max(worst, Math.abs(x[j] - y[j]));
                }
            }
        }
        return worst;
    }
}
