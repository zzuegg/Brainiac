package io.github.zzuegg.brainiac.experiments.parallel;

import io.github.zzuegg.brainiac.autograd.Recompute;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.parallel.CheckpointPlanner;
import io.github.zzuegg.brainiac.experimental.parallel.ParallelCostModel;
import io.github.zzuegg.brainiac.experimental.train.CompressedActivations;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * 9.6.5: activation checkpointing schedules for toy transformers on the memory/compute plane.
 *
 * <p>Each layer is a pre-norm block (RMS norm, fused QKV projection, causal attention, output
 * projection, RMS norm, ReLU MLP of width {@code 4d}). Its activations are listed as nine chain
 * nodes with their sizes from the real shapes and their times measured on the CPU backend: norm,
 * QKV, attention scores and probabilities ({@code 2 b h s^2} floats, as an unfused kernel keeps
 * them), attention context, output projection, norm, MLP up, ReLU, MLP down. A
 * {@link CheckpointPlanner} then prices no checkpointing, whole-block {@link Recompute}, Chen's
 * {@code sqrt(n)}, selective recomputation of the attention region and the exact optimum under
 * a budget; ActNN-style 2-bit compressed activations and reversible blocks are priced by their
 * usual accounting. {@link #execute} runs the executable plans for real and checks that their
 * gradients match.
 */
public final class MemorySchedules {

    private MemorySchedules() {
    }

    /**
     * One point on the memory/compute plane.
     *
     * @param method the method
     * @param peakBytes peak activation bytes
     * @param memorySaving share of no-checkpointing's peak saved
     * @param extraCompute extra time over a forward plus backward pass (taken as 3 forwards)
     */
    public record Point(String method, long peakBytes, double memorySaving, double extraCompute) {
    }

    /** The toy transformer's weights and its layer function. */
    private static final class Toy {
        final int width;
        final int heads;
        final List<Var[]> layers = new ArrayList<>();

        Toy(int depth, int width, int heads, long seed) {
            this.width = width;
            this.heads = heads;
            SplittableRandom r = new SplittableRandom(seed);
            float a = (float) (1 / Math.sqrt(width));
            for (int l = 0; l < depth; l++) {
                layers.add(new Var[] {Var.param(Tensor.normal(Shape.of(width, 3 * width), 0, a, r)),
                        Var.param(Tensor.normal(Shape.of(width, width), 0, a, r)),
                        Var.param(Tensor.normal(Shape.of(width, 4 * width), 0, a, r)),
                        Var.param(Tensor.normal(Shape.of(4 * width, width), 0, a / 2, r))});
            }
        }

        static Var rms(Var x) {
            return x.div(x.square().mean(2, true).add(1e-5f).sqrt());
        }

        /** The attention core: QKV {@code [b, s, 3d]} to context {@code [b, s, d]}. */
        Var core(Var qkv) {
            int b = qkv.shape().dim(0);
            int s = qkv.shape().dim(1);
            int dh = width / heads;
            Var q = qkv.slice(2, 0, width).reshape(b, s, heads, dh).permute(0, 2, 1, 3);
            Var k = qkv.slice(2, width, 2 * width).reshape(b, s, heads, dh).permute(0, 2, 1, 3);
            Var v = qkv.slice(2, 2 * width, 3 * width).reshape(b, s, heads, dh).permute(0, 2, 1, 3);
            Var a = Functional.attention(q, k, v, true, (float) (1 / Math.sqrt(dh)));
            return a.permute(0, 2, 1, 3).reshape(b, s, width);
        }

        Var layer(int l, Var x, boolean selective) {
            Var[] w = layers.get(l);
            Var qkv = rms(x).matmul(w[0]);
            Var ctx = selective ? Recompute.apply(this::core, qkv, List.of()) : core(qkv);
            Var h = x.add(ctx.matmul(w[1]));
            return h.add(rms(h).matmul(w[2]).relu().matmul(w[3]));
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            layers.forEach(w -> p.addAll(List.of(w)));
            return p;
        }
    }

    /**
     * Builds the planner chain of a toy transformer, timing each operation on the CPU.
     *
     * @param depth layers
     * @param sequence the sequence length
     * @param width the model width
     * @param seed the seed
     * @return the planner
     */
    public static CheckpointPlanner chain(int depth, int sequence, int width, long seed) {
        int b = 2;
        int heads = 4;
        long rows = (long) b * sequence;
        SplittableRandom r = new SplittableRandom(seed);
        Tensor x = Tensor.normal(Shape.of(b, sequence, width), 0, 1, r);
        Var wqkv = Var.of(Tensor.normal(Shape.of(width, 3 * width), 0, 0.1f, r));
        Var wo = Var.of(Tensor.normal(Shape.of(width, width), 0, 0.1f, r));
        Var w1 = Var.of(Tensor.normal(Shape.of(width, 4 * width), 0, 0.1f, r));
        Var w2 = Var.of(Tensor.normal(Shape.of(4 * width, width), 0, 0.1f, r));
        Var xv = Var.of(x);
        Var qkv = xv.matmul(wqkv);
        int dh = width / heads;
        Var q = qkv.slice(2, 0, width).reshape(b, sequence, heads, dh).permute(0, 2, 1, 3);
        Var k = qkv.slice(2, width, 2 * width).reshape(b, sequence, heads, dh).permute(0, 2, 1, 3);
        Var v = qkv.slice(2, 2 * width, 3 * width).reshape(b, sequence, heads, dh).permute(0, 2, 1, 3);
        Var up = xv.matmul(w1);
        int rep = 5;
        double tNorm = ParallelCostModel.median(() -> Var.noGrad(() -> Toy.rms(xv)), rep);
        double tQkv = ParallelCostModel.median(() -> Var.noGrad(() -> xv.matmul(wqkv)), rep);
        // unfused attention: scores and softmax, then the weighted sum of values
        double tScores = ParallelCostModel.median(() -> Var.noGrad(() -> Activation.softmax(3)
                .forward(q.matmul(k.permute(0, 1, 3, 2)).mul((float) (1 / Math.sqrt(dh))))), rep);
        Var probs = Var.noGrad(() -> Activation.softmax(3).forward(q.matmul(k.permute(0, 1, 3, 2))));
        double tContext = ParallelCostModel.median(() -> Var.noGrad(() -> probs.matmul(v)), rep);
        double tProj = ParallelCostModel.median(() -> Var.noGrad(() -> xv.matmul(wo).add(xv)), rep);
        double tUp = ParallelCostModel.median(() -> Var.noGrad(() -> xv.matmul(w1)), rep);
        double tRelu = ParallelCostModel.median(() -> Var.noGrad(up::relu), rep);
        double tDown = ParallelCostModel.median(() -> Var.noGrad(() -> up.matmul(w2).add(xv)), rep);
        long d4 = 4L * width;
        List<CheckpointPlanner.Node> nodes = new ArrayList<>();
        for (int l = 0; l < depth; l++) {
            String p = "L" + l + ".";
            nodes.add(new CheckpointPlanner.Node(p + "norm1", "norm", rows * width * 4, tNorm));
            nodes.add(new CheckpointPlanner.Node(p + "qkv", "matmul", rows * 3 * width * 4, tQkv));
            nodes.add(new CheckpointPlanner.Node(p + "scores", "attention",
                    2L * b * heads * sequence * sequence * 4, tScores));
            nodes.add(new CheckpointPlanner.Node(p + "context", "attention", rows * width * 4, tContext));
            nodes.add(new CheckpointPlanner.Node(p + "proj", "matmul", rows * width * 4, tProj));
            nodes.add(new CheckpointPlanner.Node(p + "norm2", "norm", rows * width * 4, tNorm));
            nodes.add(new CheckpointPlanner.Node(p + "up", "matmul", rows * d4 * 4, tUp));
            nodes.add(new CheckpointPlanner.Node(p + "relu", "elementwise", rows * d4 * 4, tRelu));
            nodes.add(new CheckpointPlanner.Node(p + "down", "matmul", rows * width * 4, tDown));
        }
        return new CheckpointPlanner(nodes);
    }

    /**
     * The memory/compute points of every method on one toy transformer.
     *
     * @param depth layers
     * @param sequence the sequence length
     * @param seed the seed
     * @return points by method
     */
    public static Map<String, Point> pareto(int depth, int sequence, long seed) {
        int width = 32;
        CheckpointPlanner planner = chain(depth, sequence, width, seed);
        double fb = 3 * planner.forwardCost();
        long none = planner.none().peakBytes();
        Map<String, Point> out = new LinkedHashMap<>();
        List<CheckpointPlanner.Plan> plans = new ArrayList<>();
        plans.add(planner.none());
        CheckpointPlanner.Plan block = planner.everyKth(9);
        plans.add(new CheckpointPlanner.Plan("Recompute every block", block.keep(), block.peakBytes(),
                block.extraCost()));
        CheckpointPlanner.Plan seg = planner.everyKth(9 * (int) Math.ceil(Math.sqrt(depth)));
        CheckpointPlanner.Plan sqrt = new CheckpointPlanner.Plan("sqrt(n) layer segments", seg.keep(),
                seg.peakBytes(), seg.extraCost());
        plans.add(sqrt);
        plans.add(planner.selective(Set.of("attention")));
        plans.add(planner.selective(Set.of("attention", "elementwise", "norm")));
        for (CheckpointPlanner.Plan p : plans) {
            out.put(p.name(), point(p.name(), p.peakBytes(), p.extraCost(), none, fb));
        }
        for (CheckpointPlanner.Plan ref : List.of(sqrt, block, planner.selective(Set.of("attention")))) {
            CheckpointPlanner.Plan opt = planner.optimal(ref.peakBytes());
            String name = "optimal at the peak of " + (ref == block ? "Recompute every block" : ref.name());
            out.put(name, point(name, opt.peakBytes(), opt.extraCost(), none, fb));
        }
        // ActNN-style 2-bit activations: every activation stored at 2 bits plus a scale and
        // offset per group of 256; the cost is quantising and dequantising, timed on the CPU
        long total = 0;
        for (CheckpointPlanner.Node n : planner.chain()) {
            total += n.bytes();
        }
        Tensor probe = Tensor.normal(Shape.of(1 << 15), 0, 1, new SplittableRandom(seed));
        CompressedActivations act = new CompressedActivations(2, 256, seed);
        double perByte = ParallelCostModel.median(() -> act.store(probe), 3) / probe.size() / 4;
        long actnn = (long) (total * (2 / 32.0 + 8 / (4.0 * 256)));
        out.put("ActNN 2-bit (CompressedActivations)", point("ActNN", actnn, 2 * perByte * total, none, fb));
        // reversible blocks: inputs rebuilt from outputs, one layer's activations at a time,
        // at the cost of one more forward pass
        long layerBytes = total / depth;
        out.put("ReversibleBlock", point("reversible", 2 * layerBytes, planner.forwardCost(), none, fb));
        return out;
    }

    private static Point point(String name, long peak, double extra, long none, double fb) {
        return new Point(name, peak, 1 - (double) peak / none, extra / fb);
    }

    /**
     * Whether the optimal plan is at least as cheap as every-k-th checkpointing at each k's
     * peak (the optimum dominates the {@code sqrt(n)} family).
     *
     * @param depth layers
     * @param sequence the sequence length
     * @param seed the seed
     * @return {@code [k, peak, every-k extra, optimal extra]} rows
     */
    public static List<double[]> dominance(int depth, int sequence, long seed) {
        CheckpointPlanner planner = chain(depth, sequence, 32, seed);
        List<double[]> rows = new ArrayList<>();
        for (int k = 1; k <= 3 * 9; k += 2) {
            CheckpointPlanner.Plan p = planner.everyKth(k);
            CheckpointPlanner.Plan o = planner.optimal(p.peakBytes());
            rows.add(new double[] {k, p.peakBytes(), p.extraCost(), o.extraCost()});
        }
        return rows;
    }

    /**
     * A plan run for real on the CPU backend.
     *
     * @param method the plan
     * @param millis milliseconds per forward plus backward pass (median)
     * @param overhead the time over no checkpointing
     * @param gradientError the largest gradient difference from no checkpointing
     */
    public record Executed(String method, double millis, double overhead, double gradientError) {
    }

    /**
     * Runs no checkpointing, whole-block recomputation, {@code sqrt(n)} segments and selective
     * attention recomputation through core {@link Recompute}.
     *
     * @param depth layers
     * @param sequence the sequence length
     * @param seed the seed
     * @return one row per plan
     */
    public static List<Executed> execute(int depth, int sequence, long seed) {
        int width = 32;
        Toy toy = new Toy(depth, width, 4, seed);
        Tensor x = Tensor.normal(Shape.of(2, sequence, width), 0, 1, new SplittableRandom(seed + 1));
        int k = (int) Math.ceil(Math.sqrt(depth));
        Map<String, Function<Var, Var>> modes = new LinkedHashMap<>();
        modes.put("no checkpointing", in -> {
            Var h = in;
            for (int l = 0; l < depth; l++) {
                h = toy.layer(l, h, false);
            }
            return h;
        });
        modes.put("Recompute every block", in -> {
            Var h = in;
            for (int l = 0; l < depth; l++) {
                int layer = l;
                h = Recompute.apply(v -> toy.layer(layer, v, false), h, List.of(toy.layers.get(l)));
            }
            return h;
        });
        modes.put("sqrt(n) segments", in -> {
            Var h = in;
            for (int s = 0; s < depth; s += k) {
                int from = s;
                int to = Math.min(depth, s + k);
                List<Var> ps = new ArrayList<>();
                for (int l = from; l < to; l++) {
                    ps.addAll(List.of(toy.layers.get(l)));
                }
                h = Recompute.apply(v -> {
                    Var y = v;
                    for (int l = from; l < to; l++) {
                        y = toy.layer(l, y, false);
                    }
                    return y;
                }, h, ps);
            }
            return h;
        });
        modes.put("selective (attention)", in -> {
            Var h = in;
            for (int l = 0; l < depth; l++) {
                h = toy.layer(l, h, true);
            }
            return h;
        });
        List<Executed> out = new ArrayList<>();
        List<Tensor> reference = null;
        double base = 0;
        for (Map.Entry<String, Function<Var, Var>> e : modes.entrySet()) {
            List<Tensor> grads = new ArrayList<>();
            double ms = 1000 * ParallelCostModel.median(() -> {
                for (Var p : toy.parameters()) {
                    p.zeroGrad();
                }
                e.getValue().apply(Var.of(x)).square().mean().backward();
            }, 3);
            for (Var p : toy.parameters()) {
                grads.add(p.grad());
            }
            double err = 0;
            if (reference == null) {
                reference = grads;
                base = ms;
            } else {
                for (int i = 0; i < grads.size(); i++) {
                    float[] a = grads.get(i).toArray();
                    float[] b = reference.get(i).toArray();
                    for (int j = 0; j < a.length; j++) {
                        err = Math.max(err, Math.abs(a[j] - b[j]));
                    }
                }
            }
            out.add(new Executed(e.getKey(), ms, ms / base - 1, err));
        }
        return out;
    }
}
