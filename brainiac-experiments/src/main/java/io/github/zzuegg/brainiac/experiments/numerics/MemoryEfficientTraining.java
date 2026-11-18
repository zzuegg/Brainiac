package io.github.zzuegg.brainiac.experiments.numerics;

import io.github.zzuegg.brainiac.autograd.Recompute;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ReversibleBlock;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.AdamMini;
import io.github.zzuegg.brainiac.experimental.train.CompressedActivations;
import io.github.zzuegg.brainiac.experimental.train.GaLore;
import io.github.zzuegg.brainiac.experimental.train.LowBitOptimizerStates;
import io.github.zzuegg.brainiac.experimental.train.ReLora;
import io.github.zzuegg.brainiac.experimental.train.StochasticRounding;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Memory-efficient training (wave 7, items 7.3.12 to 7.3.15) on {@link ToyTasks#digits}, each
 * split measuring accuracy or loss against the memory it saves:
 *
 * <ul>
 *   <li>{@link #activations}: {@link CompressedActivations} (ActNN) at 8, 4 and 2 bits against
 *       float32, bf16 storage and core {@link Recompute} checkpointing, metered in bytes kept for
 *       the backward pass;</li>
 *   <li>{@link #reversible}: a 12-block {@link ReversibleBlock} stack against the same blocks
 *       with an ordinary graph and with {@link Recompute}, plus the reconstruction drift in fp32
 *       and bf16;</li>
 *   <li>{@link #optimizerStates}: {@link LowBitOptimizerStates} (4-bit, its zero-point ablation,
 *       COAT FP8) against fp32 Adam, core 8-bit Adam and Adam-mini, in state bytes per
 *       parameter, over seeds;</li>
 *   <li>{@link #lowRank}: {@link ReLora} against full-rank AdamW, plain LoRA and GaLore, with the
 *       effective rank of the total update.</li>
 * </ul>
 */
public final class MemoryEfficientTraining {

    private static final float LR = 0.01f;

    private MemoryEfficientTraining() {
    }

    /**
     * One configuration's outcome.
     *
     * @param accuracy test accuracy (mean over seeds where several run)
     * @param loss final test cross-entropy
     * @param memory the memory measured by the split (bytes kept for backward per step, or
     *     optimizer-state bytes per parameter)
     * @param millisPerStep wall time per training step
     * @param extra a split-specific number (drift, divergences, effective rank or trainable values)
     */
    public record Row(double accuracy, double loss, double memory, double millisPerStep, double extra) {
    }

    private static ToyTasks.Data train(long seed) {
        return ToyTasks.digits(10, 30, 8, 0.1, seed);
    }

    private static ToyTasks.Data test(long seed) {
        return ToyTasks.digits(10, 20, 8, 0.1, seed, 99);
    }

    private interface Model {
        Var logits(Tensor x);
    }

    private static Row fit(Model model, List<Var> params, Optimizer opt, ToyTasks.Data train, ToyTasks.Data test,
            int steps, long seed, java.util.function.LongSupplier meter, Runnable resetMeter) {
        SplittableRandom r = new SplittableRandom(seed + 3);
        long maxSaved = 0;
        long t0 = System.nanoTime();
        for (int t = 0; t < steps; t++) {
            Tensor[] b = NumericsTasks.batch(train, 64, r);
            resetMeter.run();
            Var loss = Functional.crossEntropy(model.logits(b[0]), b[1]);
            maxSaved = Math.max(maxSaved, meter.getAsLong());
            loss.backward();
            opt.step(params);
        }
        double ms = (System.nanoTime() - t0) / 1e6 / steps;
        Tensor logits = Var.noGrad(() -> model.logits(test.inputs())).value();
        double loss = Functional.crossEntropy(Var.of(logits), test.ids()).value().item();
        return new Row(ToyTasks.accuracy(logits, test.y()), loss, maxSaved, ms, 0);
    }

    /**
     * The activation-compression split: a 64-128-128-10 ReLU MLP.
     *
     * @param steps training steps (batch 64)
     * @param seed the seed
     * @return one row per storage scheme; {@code memory} is the bytes kept for backward per step
     */
    public static Map<String, Row> activations(int steps, long seed) {
        ToyTasks.Data train = train(seed);
        ToyTasks.Data test = test(seed);
        Map<String, Row> out = new LinkedHashMap<>();
        for (int bits : new int[] {32, 16, 8, 4, 2}) {
            SimpleMlp net = SimpleMlp.relu(seed, 64, 128, 128, 10);
            CompressedActivations act = new CompressedActivations(bits, 64, seed);
            Model m = x -> {
                Var h = Var.of(x);
                for (int l = 0; l < net.depth(); l++) {
                    h = act.dense(h, net.weight(l), net.bias(l));
                    if (l < net.depth() - 1) {
                        h = bits == 32 ? reluFloat(h, act) : act.relu(h);
                    }
                }
                return h;
            };
            String name = bits == 32 ? "fp32" : bits == 16 ? "bf16 storage" : "ActNN " + bits + "-bit";
            out.put(name, fit(m, net.parameters(), Optimizer.adam(LR), train, test, steps, seed,
                    act::savedBytes, act::resetMeter));
        }
        SimpleMlp net = SimpleMlp.relu(seed, 64, 128, 128, 10);
        List<Var> hidden = List.of(net.weight(0), net.bias(0), net.weight(1), net.bias(1));
        long[] saved = new long[1];
        Model rec = x -> {
            Var h = Recompute.apply(in -> net.layer(1, net.layer(0, in)), Var.of(x), hidden);
            // kept: the block input and output, and the head's input
            saved[0] = 4L * (x.size() + 2 * h.value().size());
            return net.linear(2, h);
        };
        out.put("Recompute", fit(rec, net.parameters(), Optimizer.adam(LR), train, test, steps, seed,
                () -> saved[0], () -> { }));
        return out;
    }

    /** A ReLU kept as autograd keeps it, its float output, metered. */
    private static Var reluFloat(Var h, CompressedActivations meter) {
        meter.store(h.value());
        return h.relu();
    }

    /**
     * The reversible split: a 64-to-32 lift, a stack of reversible blocks and a linear head.
     *
     * @param blocks the depth
     * @param steps training steps
     * @param seed the seed
     * @return one row per variant; {@code memory} is activation floats kept per example (from the
     *     architecture), {@code extra} the trained stack's reconstruction drift (fp32 for
     *     "reversible", bf16 for "reversible bf16 drift")
     */
    public static Map<String, Row> reversible(int blocks, int steps, long seed) {
        ToyTasks.Data train = train(seed);
        ToyTasks.Data test = test(seed);
        Map<String, Row> out = new LinkedHashMap<>();
        int width = 32;
        int half = width / 2;
        for (String variant : List.of("reversible", "ordinary graph", "Recompute")) {
            SplittableRandom r = new SplittableRandom(seed);
            Var lift = Var.param(Tensor.normal(Shape.of(64, width), 0, (float) Math.sqrt(1.0 / 64), r));
            Var head = Var.param(Tensor.normal(Shape.of(width, 10), 0, (float) Math.sqrt(1.0 / width), r));
            Var headBias = Var.param(Tensor.zeros(Shape.of(1, 10)));
            List<ReversibleBlock> list = new ArrayList<>();
            for (int k = 0; k < blocks; k++) {
                list.add(ReversibleBlock.dense(half, r));
            }
            io.github.zzuegg.brainiac.nn.Layer stack = ReversibleBlock.stack(list, UnaryOperator.identity());
            List<Var> blockParams = stack.parameters();
            Function<Var, Var> plain = x -> {
                Var h = x;
                for (ReversibleBlock b : list) {
                    h = b.forward(h);
                }
                return h;
            };
            Model m = switch (variant) {
                case "reversible" -> x -> stack.forward(Var.of(x).matmul(lift)).matmul(head).add(headBias);
                case "ordinary graph" -> x -> plain.apply(Var.of(x).matmul(lift)).matmul(head).add(headBias);
                default -> x -> Recompute.apply(plain, Var.of(x).matmul(lift), blockParams).matmul(head).add(headBias);
            };
            List<Var> params = new ArrayList<>(blockParams);
            params.addAll(List.of(lift, head, headBias));
            int stored = switch (variant) {
                case "reversible" -> width;
                case "ordinary graph" -> blocks * 4 * half + width;
                default -> 2 * width;
            };
            Row row = fit(m, params, Optimizer.adam(LR), train, test, steps, seed, () -> 0, () -> { });
            double drift = 0;
            if (variant.equals("reversible")) {
                Tensor h = Var.noGrad(() -> Var.of(test.inputs()).matmul(lift)).value();
                drift = ReversibleBlock.reconstructionError(list, h, UnaryOperator.identity());
                double bf16 = ReversibleBlock.reconstructionError(list, h, StochasticRounding::nearestBf16);
                out.put(variant, new Row(row.accuracy(), row.loss(), stored, row.millisPerStep(), drift));
                out.put("reversible bf16 drift", new Row(Double.NaN, Double.NaN, stored, 0, bf16));
                continue;
            }
            out.put(variant, new Row(row.accuracy(), row.loss(), stored, row.millisPerStep(), drift));
        }
        return out;
    }

    /**
     * The optimizer-state split: a 64-64-64-10 ReLU MLP trained by each state format.
     *
     * @param steps training steps
     * @param seeds the seeds
     * @return one row per optimizer; {@code memory} is state bytes per parameter, {@code extra}
     *     the number of diverged seeds (test loss non-finite or above 5)
     */
    public static Map<String, Row> optimizerStates(int steps, long... seeds) {
        Map<String, Function<Float, Optimizer>> opts = new LinkedHashMap<>();
        opts.put("fp32 Adam", lr -> Optimizer.adamW(lr, 0f));
        opts.put("core 8-bit Adam", lr -> Optimizer.adam8bit(lr));
        opts.put("Adam-mini", lr -> AdamMini.of(lr, 0f));
        opts.put("4-bit (rank-1, no zero)", lr -> LowBitOptimizerStates.fourBit(lr, 0f));
        opts.put("4-bit naive (zero point)", lr -> LowBitOptimizerStates.naiveFourBit(lr, 0f));
        opts.put("COAT FP8", lr -> LowBitOptimizerStates.coatFp8(lr, 0f));
        Map<String, Row> out = new LinkedHashMap<>();
        for (Map.Entry<String, Function<Float, Optimizer>> e : opts.entrySet()) {
            double acc = 0;
            double loss = 0;
            double ms = 0;
            double bytes = 0;
            int diverged = 0;
            int ok = 0;
            for (long seed : seeds) {
                SimpleMlp net = SimpleMlp.relu(seed, 64, 64, 64, 10);
                Optimizer opt = e.getValue().apply(LR);
                Row row = fit(x -> net.forward(Var.of(x)), net.parameters(), opt, train(seed), test(seed), steps, seed,
                        () -> 0, () -> { });
                long params = 0;
                for (Var p : net.parameters()) {
                    params += p.shape().size();
                }
                long state = opt instanceof LowBitOptimizerStates low ? low.stateBytes()
                        : AdamMini.stateBytes(opt, net.parameters());
                bytes += (double) state / params;
                ms += row.millisPerStep();
                if (!Double.isFinite(row.loss()) || row.loss() > 5) {
                    diverged++;
                } else {
                    acc += row.accuracy();
                    loss += row.loss();
                    ok++;
                }
            }
            int n = seeds.length;
            out.put(e.getKey(), new Row(ok == 0 ? 0 : acc / ok, ok == 0 ? Double.NaN : loss / ok, bytes / n, ms / n,
                    diverged));
        }
        return out;
    }

    /**
     * The low-rank pretraining split: a 64-64-64-10 tanh MLP trained from scratch, with the two
     * hidden 64x64 weights trained through adapters where the method uses them.
     *
     * @param rank the adapter / projection rank
     * @param restartEvery ReLoRA's merge period
     * @param steps training steps
     * @param seed the seed
     * @return one row per method; {@code memory} is optimizer-state bytes (Adam: 8 per trainable
     *     value), {@code extra} the effective rank of the second hidden weight's total update
     */
    public static Map<String, Row> lowRank(int rank, int restartEvery, int steps, long seed) {
        ToyTasks.Data train = train(seed);
        ToyTasks.Data test = test(seed);
        Map<String, Row> out = new LinkedHashMap<>();
        for (String method : List.of("full-rank AdamW", "GaLore", "LoRA", "ReLoRA")) {
            SimpleMlp net = SimpleMlp.tanh(seed, 64, 64, 64, 10);
            Tensor before = net.weight(1).value();
            ReLora lora = method.endsWith("LoRA") ? new ReLora(List.of(net.weight(0), net.weight(1)), rank, 2f * rank,
                    method.equals("ReLoRA") ? restartEvery : 0, 10, seed) : null;
            List<Var> params = new ArrayList<>();
            if (lora == null) {
                params.addAll(net.parameters());
            } else {
                params.addAll(lora.adapters());
                params.addAll(List.of(net.bias(0), net.bias(1), net.weight(2), net.bias(2)));
            }
            float base = 0.01f;
            Optimizer opt = method.equals("GaLore") ? new GaLore(base, rank, 50, 1f, 0f) : Optimizer.adamW(base, 0f);
            Model m = x -> {
                Var h = Var.of(x);
                for (int l = 0; l < net.depth(); l++) {
                    Var w = lora != null && l < 2 ? lora.weight(l) : net.weight(l);
                    h = h.matmul(w).add(net.bias(l));
                    h = l < net.depth() - 1 ? h.tanh() : h;
                }
                return h;
            };
            SplittableRandom r = new SplittableRandom(seed + 3);
            long t0 = System.nanoTime();
            for (int t = 0; t < steps; t++) {
                if (lora != null) {
                    opt.learningRate(base * lora.learningRateFactor(t));
                }
                Tensor[] b = NumericsTasks.batch(train, 64, r);
                Functional.crossEntropy(m.logits(b[0]), b[1]).backward();
                opt.step(params);
                if (lora != null) {
                    lora.afterStep(t, opt);
                }
            }
            double ms = (System.nanoTime() - t0) / 1e6 / steps;
            Tensor logits = Var.noGrad(() -> m.logits(test.inputs())).value();
            double loss = Functional.crossEntropy(Var.of(logits), test.ids()).value().item();
            Tensor update = lora != null ? lora.totalUpdate(1) : net.weight(1).value().sub(before);
            long trainable = 0;
            for (Var p : params) {
                trainable += p.shape().size();
            }
            long state = method.equals("GaLore") ? AdamMini.stateBytes(opt, params) : 8 * trainable;
            out.put(method, new Row(ToyTasks.accuracy(logits, test.y()), loss, state, ms,
                    ReLora.effectiveRank(update)));
        }
        return out;
    }
}
