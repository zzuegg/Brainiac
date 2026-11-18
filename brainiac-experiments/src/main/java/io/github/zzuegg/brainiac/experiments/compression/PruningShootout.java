package io.github.zzuegg.brainiac.experiments.compression;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.compress.FixedSparsityOptimizer;
import io.github.zzuegg.brainiac.experimental.compress.GradualPruningSchedule;
import io.github.zzuegg.brainiac.experimental.compress.MovementPruning;
import io.github.zzuegg.brainiac.experimental.compress.PruningAtInit;
import io.github.zzuegg.brainiac.experimental.compress.WoodFisherPruning;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.nn.WeightPruning;
import io.github.zzuegg.brainiac.experimental.train.LotteryTicket;
import io.github.zzuegg.brainiac.experimental.train.RigL;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;

/**
 * Unstructured pruning criteria and schedules on the {@link CompressionBench} digits MLP, every row
 * at the same sparsity per weight matrix and the same number of training steps:
 * <ul>
 * <li>{@link #schedules} (wave 8 item 8.5.1): gradual magnitude pruning against one-shot
 *     magnitude pruning with fine-tuning, lottery tickets with rewinding and RigL;</li>
 * <li>{@link #transfer} (8.5.2): movement pruning (hard and soft) against magnitude pruning
 *     during fine-tuning and one-shot Wanda, after moving a pretrained MLP to a new task;</li>
 * <li>{@link #oneShot} (8.5.3): WoodFisher against magnitude, Wanda, SparseGPT and optimal brain
 *     damage, straight after pruning and after a short recovery;</li>
 * <li>{@link #atInit} (8.5.4): SNIP, GraSP and SynFlow against random pruning (uniform and with
 *     matched layer-wise ratios) and a lottery ticket;</li>
 * <li>{@link #languageModel} (8.5.5): structured pruning of a toy transformer language model
 *     (LLM-Pruner, FASP, 2SSP) against SliceGPT, Wanda and dropping blocks;</li>
 * <li>{@link #cnn} (8.5.6): network slimming of the shapes CNN against filter-magnitude pruning
 *     and thinner networks trained from scratch;</li>
 * <li>{@link #moe} (8.5.7): expert pruning and MC-SMoE expert merging of a mixture of experts.</li>
 * </ul>
 */
public final class PruningShootout {

    /**
     * One pruned network.
     *
     * @param method the method
     * @param sparsity the measured share of zero (or masked) weights
     * @param accuracy test accuracy (after any recovery); validation loss for language-model rows
     * @param extra a second number the method reports (accuracy or loss before recovery, or -1)
     */
    public record Row(String method, double sparsity, double accuracy, double extra) {
    }

    private PruningShootout() {
    }

    // ------------------------------------------------------------------ 8.5.1

    /**
     * Pruning schedules trained from scratch for {@code steps} in total.
     *
     * @param sparsities the target sparsities
     * @param steps total training steps of every row
     * @param seed the seed
     * @return dense first, then per sparsity: one-shot, GMP, lottery ticket, RigL
     */
    public static List<Row> schedules(double[] sparsities, int steps, long seed) {
        CompressionBench.Split data = CompressionBench.digits(seed);
        List<Row> rows = new ArrayList<>();
        SimpleMlp dense = CompressionBench.untrained(seed);
        CompressionBench.train(dense::forward, dense.parameters(), Optimizer.adam(0.01f), data.train(), steps);
        rows.add(new Row("dense", 0, CompressionBench.accuracy(dense::forward, data.test()), -1));
        for (double s : sparsities) {
            // one-shot: half dense, prune, half fine-tune
            SimpleMlp one = CompressionBench.untrained(seed);
            List<Var> w1 = CompressionBench.weights(one);
            CompressionBench.train(one::forward, one.parameters(), Optimizer.adam(0.01f), data.train(), steps / 2);
            for (Var w : w1) {
                w.assign(WeightPruning.magnitude(w.value(), s));
            }
            double before = CompressionBench.accuracy(one::forward, data.test());
            CompressionBench.train(one::forward, one.parameters(),
                    FixedSparsityOptimizer.ofZeros(Optimizer.adam(0.01f), w1), data.train(), steps - steps / 2);
            rows.add(row("one-shot+ft", w1, one, data.test(), before));
            // gradual magnitude pruning
            SimpleMlp gmp = CompressionBench.untrained(seed);
            List<Var> w2 = CompressionBench.weights(gmp);
            GradualPruningSchedule sched = GradualPruningSchedule.magnitude(Optimizer.adam(0.01f), w2, s, steps / 10,
                    steps * 6 / 10, Math.max(1, steps / 30));
            CompressionBench.train(gmp::forward, gmp.parameters(), sched, data.train(), steps);
            rows.add(row("gmp", w2, gmp, data.test(), -1));
            // lottery ticket: three prune-rewind rounds and a final run, equal total steps
            SimpleMlp lt = CompressionBench.untrained(seed);
            List<Var> w3 = CompressionBench.weights(lt);
            int rounds = 3;
            int each = steps / (rounds + 1);
            CompressionBench.train(lt::forward, lt.parameters(), Optimizer.adam(0.01f), data.train(), 5);
            LotteryTicket ticket = new LotteryTicket(w3);   // rewind point: step 5
            double per = 1 - Math.pow(1 - s, 1.0 / rounds);
            for (int r = 0; r < rounds; r++) {
                CompressionBench.train(lt::forward, lt.parameters(), ticket.masked(Optimizer.adam(0.01f)),
                        data.train(), r == 0 ? each - 5 : each);
                ticket.prune(per);
                ticket.rewind();
            }
            CompressionBench.train(lt::forward, lt.parameters(), ticket.masked(Optimizer.adam(0.01f)), data.train(),
                    each);
            rows.add(row("lottery", w3, lt, data.test(), -1));
            // RigL
            SimpleMlp rig = CompressionBench.untrained(seed);
            List<Var> w4 = CompressionBench.weights(rig);
            RigL rigl = new RigL(Optimizer.adam(0.01f), w4, 1 - s, 0.3, Math.max(1, steps / 15), steps * 3 / 4, seed);
            CompressionBench.train(rig::forward, rig.parameters(), rigl, data.train(), steps);
            rows.add(row("rigl", w4, rig, data.test(), -1));
        }
        return rows;
    }

    // ------------------------------------------------------------------ 8.5.2

    /**
     * Pruning while fine-tuning the reference MLP (pretrained on prototypes {@code seed}) on a
     * new task (prototypes {@code seed + 100}).
     *
     * @param sparsities the target sparsities
     * @param steps fine-tuning steps
     * @param softLambda the L1 penalty of soft movement pruning
     * @param seed the seed
     * @return per sparsity: magnitude (GMP during fine-tuning), Wanda (one-shot after dense
     *     fine-tuning), hard movement; then soft movement and magnitude at soft's sparsity
     */
    public static List<Row> transfer(double[] sparsities, int steps, double softLambda, long seed) {
        CompressionBench.Split task = CompressionBench.digits(seed + 100);
        List<Row> rows = new ArrayList<>();
        SimpleMlp tuned = CompressionBench.trained(seed);
        CompressionBench.train(tuned::forward, tuned.parameters(), Optimizer.adam(0.005f), task.train(), steps);
        for (double s : sparsities) {
            int start = steps / 10;
            int end = steps * 7 / 10;
            int freq = Math.max(1, steps / 40);
            SimpleMlp mag = CompressionBench.trained(seed);
            List<Var> wm = CompressionBench.weights(mag);
            CompressionBench.train(mag::forward, mag.parameters(),
                    GradualPruningSchedule.magnitude(Optimizer.adam(0.005f), wm, s, start, end, freq), task.train(),
                    steps);
            rows.add(row("magnitude", wm, mag, task.test(), -1));
            rows.add(wanda(tuned, task, s));
            SimpleMlp mov = CompressionBench.trained(seed);
            List<Var> wv = CompressionBench.weights(mov);
            MovementPruning mp = MovementPruning.hard(Optimizer.adam(0.005f), wv, s, start, end, freq);
            UnaryOperator<Var> fwd = CompressionBench.forward(mov, (l, w) -> mp.masked(w), null);
            CompressionBench.train(fwd, mov.parameters(), mp, task.train(), steps);
            mp.finish();
            rows.add(row("movement", wv, mov, task.test(), -1));
        }
        SimpleMlp soft = CompressionBench.trained(seed);
        List<Var> ws = CompressionBench.weights(soft);
        MovementPruning sp = MovementPruning.soft(Optimizer.adam(0.005f), ws, 0, softLambda, 0.05);
        CompressionBench.train(CompressionBench.forward(soft, (l, w) -> sp.masked(w), null), soft.parameters(), sp,
                task.train(), steps);
        sp.finish();
        Row softRow = row("soft movement", ws, soft, task.test(), -1);
        rows.add(softRow);
        SimpleMlp mag = CompressionBench.trained(seed);
        List<Var> wm = CompressionBench.weights(mag);
        CompressionBench.train(mag::forward, mag.parameters(), GradualPruningSchedule.magnitude(Optimizer.adam(0.005f),
                wm, softRow.sparsity(), steps / 10, steps * 7 / 10, Math.max(1, steps / 40)), task.train(), steps);
        rows.add(row("magnitude@soft", wm, mag, task.test(), -1));
        return rows;
    }

    private static Row wanda(SimpleMlp tuned, CompressionBench.Split task, double s) {
        SimpleMlp copy = CompressionBench.untrained(0);
        for (int i = 0; i < copy.parameters().size(); i++) {
            copy.parameters().get(i).assign(tuned.parameters().get(i).value());
        }
        Tensor[] acts = copy.activations(task.train().inputs());
        List<Var> w = CompressionBench.weights(copy);
        for (int l = 0; l < w.size(); l++) {
            w.get(l).assign(WeightPruning.wanda(w.get(l).value(), acts[l], s));
        }
        return row("wanda", w, copy, task.test(), -1);
    }

    // ------------------------------------------------------------------ 8.5.3

    /**
     * One-shot pruning of the trained reference MLP, then a short masked recovery.
     *
     * @param sparsities the sparsities
     * @param samples per-sample gradients for the Fisher
     * @param recovery recovery steps
     * @param seed the seed
     * @return per sparsity: magnitude, Wanda, SparseGPT, OBD, WoodFisher (one step), WoodFisher
     *     (four gradual steps); {@code extra} is the accuracy straight after pruning
     */
    public static List<Row> oneShot(double[] sparsities, int samples, int recovery, long seed) {
        CompressionBench.Split data = CompressionBench.digits(seed);
        List<Row> rows = new ArrayList<>();
        ToyTasks.Data calib = data.train().slice(0, samples);
        WoodFisherPruning wf = new WoodFisherPruning(64, 0.1);
        for (double s : sparsities) {
            for (String method : List.of("magnitude", "wanda", "sparsegpt", "obd", "woodfisher", "woodfisher x4")) {
                SimpleMlp net = CompressionBench.trained(seed);
                List<Var> w = CompressionBench.weights(net);
                Tensor[] acts = net.activations(calib.inputs());
                switch (method) {
                    case "magnitude" -> w.forEach(v -> v.assign(WeightPruning.magnitude(v.value(), s)));
                    case "wanda" -> {
                        for (int l = 0; l < w.size(); l++) {
                            w.get(l).assign(WeightPruning.wanda(w.get(l).value(), acts[l], s));
                        }
                    }
                    case "sparsegpt" -> {
                        for (int l = 0; l < w.size(); l++) {
                            w.get(l).assign(WeightPruning.sparseGpt(w.get(l).value(), acts[l], s, 0, 0, 32, 0.01));
                        }
                    }
                    case "obd" -> WoodFisherPruning.obd(w, grads(net, calib, w), s);
                    case "woodfisher" -> wf.prune(w, grads(net, calib, w), s, true);
                    default -> {
                        for (int k = 1; k <= 4; k++) {
                            double sk = 1 - Math.pow(1 - s, k / 4.0);
                            wf.prune(w, grads(net, calib, w), sk, true);
                        }
                    }
                }
                double before = CompressionBench.accuracy(net::forward, data.test());
                CompressionBench.train(net::forward, net.parameters(),
                        FixedSparsityOptimizer.ofZeros(Optimizer.adam(0.003f), w), data.train(), recovery);
                rows.add(row(method, w, net, data.test(), before));
            }
        }
        return rows;
    }

    private static double[][] grads(SimpleMlp net, ToyTasks.Data calib, List<Var> w) {
        Tensor x = calib.inputs();
        Tensor ids = calib.ids();
        return WoodFisherPruning.perSampleGradients(
                n -> Functional.crossEntropy(net.forward(Var.of(x.slice(0, n, n + 1))), ids.slice(0, n, n + 1)),
                calib.size(), w);
    }

    // ------------------------------------------------------------------ 8.5.4

    /**
     * Pruning at initialisation, then training the sparse network.
     *
     * @param sparsities the sparsities
     * @param steps training steps after pruning
     * @param seed the seed
     * @return per sparsity: SNIP, GraSP, SynFlow, random, random with SNIP's and SynFlow's
     *     layer ratios, and a lottery ticket (pruned after training, rewound to the
     *     initialisation, an upper bound); {@code extra} encodes the last layer's survival
     */
    public static List<Row> atInit(double[] sparsities, int steps, long seed) {
        CompressionBench.Split data = CompressionBench.digits(seed);
        ToyTasks.Data batch = data.train();
        List<Row> rows = new ArrayList<>();
        for (double s : sparsities) {
            List<Tensor> snipRef = null;
            List<Tensor> synRef = null;
            for (String method : List.of("snip", "grasp", "synflow", "random", "random@snip", "random@synflow",
                    "lottery")) {
                SimpleMlp net = CompressionBench.untrained(seed);
                List<Var> w = CompressionBench.weights(net);
                java.util.function.Supplier<Var> loss = () -> Functional.crossEntropy(
                        net.forward(Var.of(batch.inputs())), batch.ids());
                Map<Var, Tensor> masks = switch (method) {
                    case "snip" -> PruningAtInit.snip(w, loss, s);
                    case "grasp" -> PruningAtInit.grasp(w, loss, s);
                    case "synflow" -> PruningAtInit.synflow(w, s, 50);
                    case "random" -> PruningAtInit.random(w, s, seed);
                    case "random@snip" -> PruningAtInit.randomLayerwise(w, remap(snipRef, w), seed);
                    case "random@synflow" -> PruningAtInit.randomLayerwise(w, remap(synRef, w), seed);
                    default -> lotteryMask(net, w, data, s, steps);
                };
                if (method.equals("snip")) {
                    snipRef = w.stream().map(masks::get).toList();
                } else if (method.equals("synflow")) {
                    synRef = w.stream().map(masks::get).toList();
                }
                FixedSparsityOptimizer.apply(masks);
                CompressionBench.train(net::forward, net.parameters(),
                        new FixedSparsityOptimizer(Optimizer.adam(0.01f), masks), data.train(), steps);
                double[] survival = PruningAtInit.survival(w, masks);
                rows.add(new Row(method, 1 - kept(w, masks), CompressionBench.accuracy(net::forward, data.test()),
                        survival[survival.length - 1]));
            }
        }
        return rows;
    }

    private static Map<Var, Tensor> lotteryMask(SimpleMlp net, List<Var> w, CompressionBench.Split data, double s,
            int steps) {
        LotteryTicket ticket = new LotteryTicket(w);
        CompressionBench.train(net::forward, net.parameters(), Optimizer.adam(0.01f), data.train(), steps);
        ticket.prune(s);
        ticket.rewind();
        Map<Var, Tensor> m = new java.util.IdentityHashMap<>();
        for (Var v : w) {
            m.put(v, ticket.mask(v));
        }
        return m;
    }

    /** Masks of another network's weights, matched by position. */
    private static Map<Var, Tensor> remap(List<Tensor> ref, List<Var> w) {
        Map<Var, Tensor> m = new java.util.IdentityHashMap<>();
        for (int i = 0; i < w.size(); i++) {
            m.put(w.get(i), ref.get(i));
        }
        return m;
    }

    private static double kept(List<Var> w, Map<Var, Tensor> masks) {
        double k = 0;
        double n = 0;
        for (Var v : w) {
            k += masks.get(v).sum().item();
            n += v.shape().size();
        }
        return k / n;
    }

    // ------------------------------------------------------------------ 8.5.5

    /**
     * Structured pruning of a toy language model; see {@link LmStructuredPruning}.
     *
     * @param shares shares of block-matrix weights to remove
     * @param steps training steps of the language model
     * @param recovery recovery fine-tuning steps of every row
     * @param seed the seed
     * @return dense first, then per share: LLM-Pruner, FASP, 2SSP, Wanda, SliceGPT and (where it
     *     fits the share) dropping blocks; {@code accuracy} is the validation loss after recovery,
     *     {@code extra} the loss right after pruning, {@code sparsity} the share removed
     */
    public static List<Row> languageModel(double[] shares, int steps, int recovery, long seed) {
        return LmStructuredPruning.run(shares, steps, recovery, seed);
    }

    // ------------------------------------------------------------------ 8.5.18 (end to end)

    /**
     * The toy language model of {@link #languageModel} pruned to 50% unstructured, 2:4, and in
     * 4x4 and 8x8 blocks (the formats of {@code experiments.kernels.KernelTuning#sparse}).
     *
     * @param steps training steps of the language model
     * @param recovery masked recovery steps
     * @param seed the seed
     * @return dense first; {@code accuracy} is the validation loss after recovery, {@code extra}
     *     straight after pruning
     */
    public static List<Row> sparseFormats(int steps, int recovery, long seed) {
        return LmStructuredPruning.sparseFormats(steps, recovery, seed);
    }

    // ------------------------------------------------------------------ 8.5.6

    /**
     * Channel pruning of the shapes CNN; see {@link CnnSlimming}.
     *
     * @param shares shares of all channels to remove
     * @param steps training steps of the reference networks
     * @param fineTune fine-tuning steps after pruning
     * @param lambda the L1 weight on the batch-norm scales
     * @param seed the seed
     * @return dense and dense-with-L1 first, then per share: slimming, filter-L1, the slimmed
     *     shape from scratch and a uniform width of equal multiply-adds from scratch;
     *     {@code metric} is test accuracy, {@code cost} multiply-adds per image, {@code size}
     *     parameters, {@code millis} the forward latency of the test set
     */
    public static List<CompactRow> cnn(double[] shares, int steps, int fineTune, float lambda, long seed) {
        return CnnSlimming.run(shares, steps, fineTune, lambda, seed);
    }

    // ------------------------------------------------------------------ 8.5.7

    /**
     * Halving a mixture of experts; see {@link MoeCompression}.
     *
     * @param regimeCounts the regime counts of the tasks (4: redundant experts, 8: specialised)
     * @param steps training steps
     * @param seed the seed
     * @return per task: full, search-prune, drop least-used, merge, average all, dense of the
     *     active width; {@code metric} is validation MSE, {@code cost} the number of experts,
     *     {@code size} parameters
     */
    public static List<CompactRow> moe(int[] regimeCounts, int steps, long seed) {
        return MoeCompression.run(regimeCounts, steps, seed);
    }

    private static Row row(String method, List<Var> w, SimpleMlp net, ToyTasks.Data test, double extra) {
        return new Row(method, CompressionBench.sparsity(w), CompressionBench.accuracy(net::forward, test), extra);
    }
}
