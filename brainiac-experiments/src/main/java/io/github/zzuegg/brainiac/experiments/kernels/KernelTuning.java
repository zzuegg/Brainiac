package io.github.zzuegg.brainiac.experiments.kernels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.kernels.AutoTuner;
import io.github.zzuegg.brainiac.experimental.kernels.BlockSparseMatmul;
import io.github.zzuegg.brainiac.experimental.kernels.CsrMatmul;
import io.github.zzuegg.brainiac.experimental.kernels.EGraph;
import io.github.zzuegg.brainiac.experimental.kernels.GreedyRewriter;
import io.github.zzuegg.brainiac.experimental.kernels.KernelTimer;
import io.github.zzuegg.brainiac.experimental.kernels.LearnedCostModel;
import io.github.zzuegg.brainiac.experimental.kernels.NmSparseMatmul;
import io.github.zzuegg.brainiac.experimental.kernels.ScheduleSpace;
import io.github.zzuegg.brainiac.experimental.kernels.StrassenMatmul;
import io.github.zzuegg.brainiac.experimental.kernels.TensorDecompositionGame;
import io.github.zzuegg.brainiac.experimental.kernels.TensorExpressionGraph;
import io.github.zzuegg.brainiac.experimental.kernels.TensorExpressionGraph.Op;
import io.github.zzuegg.brainiac.experimental.kernels.WinogradConv;
import io.github.zzuegg.brainiac.experiments.compression.ShapesCnn;
import io.github.zzuegg.brainiac.experiments.evolution.EvolvedKernels;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.SparseMatrix;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 8 kernel experiments (items 8.5.14 to 8.5.18), all timed for real on the CPU with
 * warm-up and checked against the core kernels:
 * <ul>
 * <li>{@link #autotune} (8.5.14): {@link AutoTuner} strategies over a {@link ScheduleSpace}
 *     against the core matmul and the simulated-cache optimum of {@link EvolvedKernels};</li>
 * <li>{@link #fastAlgorithms} (8.5.15): {@link WinogradConv} and {@link StrassenMatmul}
 *     against the core convolution and matmul, and the shapes CNN in fp32 and int8;</li>
 * <li>{@link #discovery} (8.5.16): MCTS on the {@link TensorDecompositionGame} against
 *     ALS with rounding, CMA-ES and random search;</li>
 * <li>{@link #graphs} (8.5.17): {@link EGraph} equality saturation against
 *     {@link GreedyRewriter} on an attention-projection graph and a conv + batch-norm graph;</li>
 * <li>{@link #sparse} (8.5.18): {@link CsrMatmul}, {@link BlockSparseMatmul} and
 *     {@link NmSparseMatmul} against the core dense matmul and core {@link SparseMatrix}.</li>
 * </ul>
 */
public final class KernelTuning {

    private KernelTuning() {
    }

    /**
     * One result line.
     *
     * @param method the method
     * @param metric the measured value (GFLOP/s, speed-up, accuracy, rank, cost...)
     * @param baseline the matching baseline value, or NaN
     * @param detail a short note
     */
    public record Row(String method, double metric, double baseline, String detail) {
    }

    private static float[] random(int n, SplittableRandom r) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) r.nextGaussian();
        }
        return v;
    }

    // ------------------------------------------------------------------ 8.5.14

    /**
     * Autotunes a square matmul.
     *
     * @param size the side
     * @param budget real measurements per strategy
     * @param heldOut extra random schedules to score the cost model on
     * @param seed the seed
     * @return the core matmul ({@code metric} GFLOP/s); per strategy the best GFLOP/s
     *     ({@code baseline} the core's) and measurements to 95% of the best found overall; the
     *     simulated-cache optimum measured for real; the cost model's Spearman correlation on
     *     held-out schedules; the best schedule's relative error against the core
     */
    public static List<Row> autotune(int size, int budget, int heldOut, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] a = random(size * size, r);
        float[] b = random(size * size, r);
        ScheduleSpace space = new ScheduleSpace(size, size, size);
        Tensor ta = Tensor.of(Shape.of(size, size), a);
        Tensor tb = Tensor.of(Shape.of(size, size), b);
        SplittableRandom warm = new SplittableRandom(seed + 99);
        for (int i = 0; i < 48; i++) {   // JIT warm-up of every loop order, so the first strategy is not penalised
            space.multiply(a, b, space.random(warm));
        }
        double core = KernelTimer.time(() -> ta.matmul(tb).toArray(), 20, 15).peakGflops(space.flops());
        Map<ScheduleSpace.Schedule, Double> memo = new HashMap<>();
        AutoTuner tuner = new AutoTuner(space, s -> memo.computeIfAbsent(s, k -> space.gflops(a, b, k, 5)));
        List<AutoTuner.Result> results = List.of(tuner.random(budget, seed + 1), tuner.grid(budget),
                tuner.evolution(budget, 8, seed + 2), tuner.costModelGuided(budget, 8, 4, seed + 3));
        List<String> names = List.of("random", "grid", "evolution", "cost-model");
        double best = 0;
        for (AutoTuner.Result res : results) {
            best = Math.max(best, res.bestGflops());
        }
        List<Row> rows = new ArrayList<>();
        rows.add(new Row("core matmul", core, Double.NaN, ""));
        for (int i = 0; i < results.size(); i++) {
            AutoTuner.Result res = results.get(i);
            rows.add(new Row(names.get(i), res.bestGflops(), core,
                    res.best() + ", to95%=" + res.measurementsTo(0.95 * best)));
        }
        // simulated-cache optimum (EvolvedKernels' 1 KiB LRU on a 32x32 product), measured for real
        SplittableRandom sr = new SplittableRandom(seed + 4);
        String[] orders = {"ijk", "ikj", "jik", "jki", "kij", "kji"};
        int[] tiles = {4, 8, 16, 32};
        EvolvedKernels.Matmul simBest = null;
        long fewest = Long.MAX_VALUE;
        for (int e = 0; e < 60; e++) {
            EvolvedKernels.Matmul cand = new EvolvedKernels.Matmul(orders[sr.nextInt(6)], tiles[sr.nextInt(4)],
                    tiles[sr.nextInt(4)], tiles[sr.nextInt(4)]);
            long misses = EvolvedKernels.matmulMisses(cand);
            if (misses >= 0 && misses < fewest) {
                fewest = misses;
                simBest = cand;
            }
        }
        ScheduleSpace.Schedule simSchedule = new ScheduleSpace.Schedule(simBest.order(), simBest.ti(), simBest.tj(),
                simBest.tk(), 1, 1);
        rows.add(new Row("sim-cache optimum", memo.computeIfAbsent(simSchedule, k -> space.gflops(a, b, k, 5)), best,
                simSchedule + ", misses=" + fewest));
        LearnedCostModel model = results.get(3).model();
        double[] predicted = new double[heldOut];
        double[] measured = new double[heldOut];
        for (int i = 0; i < heldOut; i++) {
            ScheduleSpace.Schedule s = space.random(sr);
            predicted[i] = model.predict(s);
            measured[i] = memo.computeIfAbsent(s, k -> space.gflops(a, b, k, 5));
        }
        rows.add(new Row("cost-model spearman", LearnedCostModel.spearman(predicted, measured), Double.NaN,
                heldOut + " held-out schedules"));
        float[] got = space.multiply(a, b, results.get(3).best());
        rows.add(new Row("relative error", KernelTimer.relativeError(got, ta.matmul(tb).toArray()), Double.NaN,
                "best schedule vs core"));
        return rows;
    }

    // ------------------------------------------------------------------ 8.5.15

    /**
     * Winograd convolution and Strassen multiplication.
     *
     * @param matmulSides square sides for Strassen (cut-over 128)
     * @param cnnSteps training steps of the shapes CNN
     * @param seed the seed
     * @return per Winograd variant the speed-up over the core convolution ({@code baseline}
     *     relative error); per side Strassen's speed-up ({@code baseline} relative error); then
     *     the shapes CNN's accuracy with its first convolution in fp32, F(2,3), F(4,3), direct
     *     int8 and Winograd int8
     */
    public static List<Row> fastAlgorithms(int[] matmulSides, int cnnSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Row> rows = new ArrayList<>();
        Tensor x = Tensor.of(Shape.of(4, 16, 32, 32), random(4 * 16 * 32 * 32, r));
        Tensor w = Tensor.of(Shape.of(16, 16, 3, 3), random(16 * 16 * 9, r));
        Function<Tensor, Tensor> direct = in -> Var.noGrad(() -> Functional.conv(Var.of(in), Var.of(w), null,
                new int[] {1, 1}, new int[] {1, 1}, new int[] {1, 1}, new int[] {1, 1}, 1).value());
        float[] ref = direct.apply(x).toArray();
        double coreMs = KernelTimer.millis(() -> direct.apply(x).toArray(), 2, 5);
        for (int tile : new int[] {2, 4}) {
            WinogradConv wc = new WinogradConv(tile, 0);
            float[][] u = wc.transformFilters(w);
            double ms = KernelTimer.millis(() -> wc.forward(x, 16, u).toArray(), 2, 5);
            double err = KernelTimer.relativeError(wc.forward(x, 16, u).toArray(), ref);
            rows.add(new Row("winograd F(" + tile + ",3) speed-up", coreMs / ms, err,
                    String.format("%.1fms vs core %.1fms", ms, coreMs)));
        }
        StrassenMatmul strassen = new StrassenMatmul(128);
        for (int n : matmulSides) {
            Tensor a = Tensor.of(Shape.of(n, n), random(n * n, r));
            Tensor b = Tensor.of(Shape.of(n, n), random(n * n, r));
            double coreMatmul = KernelTimer.millis(() -> a.matmul(b).toArray(), 1, 3);
            double ms = KernelTimer.millis(() -> strassen.multiply(a, b).toArray(), 1, 3);
            double err = KernelTimer.relativeError(strassen.multiply(a, b).toArray(), a.matmul(b).toArray());
            rows.add(new Row("strassen n=" + n + " speed-up", coreMatmul / ms, err,
                    String.format("%.1fms vs core %.1fms", ms, coreMatmul)));
        }
        Tensor[] train = ShapesCnn.data(192, seed);
        Tensor[] test = ShapesCnn.data(192, seed + 1);
        ShapesCnn cnn = ShapesCnn.create(16, 32, seed);
        cnn.train(train, Optimizer.adam(0.01f), cnnSteps, 0);
        rows.add(new Row("cnn fp32 direct", cnn.accuracy(test), Double.NaN, ""));
        for (int tile : new int[] {2, 4}) {
            WinogradConv wc = new WinogradConv(tile, 0);
            rows.add(new Row("cnn fp32 F(" + tile + ",3)", cnn.accuracy(test, (in, filters, stride) ->
                    Var.of(wc.forward(in.value(), filters.value())), ShapesCnn.CORE), Double.NaN, ""));
        }
        ShapesCnn.Convolution directInt8 = (in, filters, stride) -> {
            Tensor qx = Tensor.of(in.shape(), WinogradConv.fakeQuant(in.value().toArray(), 8));
            Tensor qw = Tensor.of(filters.shape(), WinogradConv.fakeQuant(filters.value().toArray(), 8));
            return ShapesCnn.CORE.apply(Var.of(qx), Var.of(qw), stride);
        };
        rows.add(new Row("cnn int8 direct", cnn.accuracy(test, directInt8, ShapesCnn.CORE), Double.NaN, ""));
        for (int tile : new int[] {2, 4}) {
            WinogradConv wc = new WinogradConv(tile, 8);
            rows.add(new Row("cnn int8 F(" + tile + ",3)", cnn.accuracy(test, (in, filters, stride) ->
                    Var.of(wc.forward(in.value(), filters.value())), ShapesCnn.CORE), Double.NaN, ""));
        }
        for (int bits : new int[] {6}) {
            WinogradConv w2 = new WinogradConv(2, bits);
            WinogradConv w4 = new WinogradConv(4, bits);
            rows.add(new Row("cnn int" + bits + " F(2,3)", cnn.accuracy(test, (in, filters, stride) ->
                    Var.of(w2.forward(in.value(), filters.value())), ShapesCnn.CORE), Double.NaN, ""));
            rows.add(new Row("cnn int" + bits + " F(4,3)", cnn.accuracy(test, (in, filters, stride) ->
                    Var.of(w4.forward(in.value(), filters.value())), ShapesCnn.CORE), Double.NaN, ""));
        }
        return rows;
    }

    // ------------------------------------------------------------------ 8.5.16

    /**
     * Searches a low-rank decomposition of one matmul tensor.
     *
     * @param n rows of A
     * @param m columns of A
     * @param p columns of B
     * @param rank the target rank
     * @param mctsEpisodes MCTS episodes (simulations per move fixed at 32)
     * @param alsRestarts ALS restarts (300 sweeps each)
     * @param cmaRestarts CMA-ES restarts
     * @param cmaEvaluations CMA-ES evaluations per restart
     * @param randomEvaluations random-search samples
     * @param seed the seed
     * @return per method: {@code metric} 1 if the target rank was found exactly,
     *     {@code baseline} the naive rank, {@code detail} the episodes, restarts or evaluations
     *     used, the wall time and the random-matrix error of the algorithm found
     */
    public static List<Row> discovery(int n, int m, int p, int rank, int mctsEpisodes, int alsRestarts,
            int cmaRestarts, int cmaEvaluations, int randomEvaluations, long seed) {
        List<Row> rows = new ArrayList<>();
        TensorDecompositionGame game = new TensorDecompositionGame(n, m, p, rank);
        String name = n + "x" + m + "x" + p + " r" + rank + " ";
        for (String method : List.of("mcts", "als+round", "cma-es", "random")) {
            long t0 = System.nanoTime();
            TensorDecompositionGame.Search s = switch (method) {
                case "mcts" -> game.mcts(32, mctsEpisodes, seed);
                case "als+round" -> game.alsRounding(rank, alsRestarts, 300, seed);
                case "cma-es" -> game.cmaEs(rank, cmaRestarts, cmaEvaluations, seed);
                default -> game.randomSearch(rank, randomEvaluations, seed);
            };
            double ms = (System.nanoTime() - t0) / 1e6;
            double err = s.found() == null ? Double.NaN : game.randomCheck(s.found(), 20, seed);
            rows.add(new Row(name + method, s.found() == null ? 0 : 1, game.naiveRank(), String.format(
                    "evals=%d, %.0fms%s", s.evaluations(), ms, s.found() == null ? "" : String.format(", err=%.1e", err))));
        }
        return rows;
    }

    // ------------------------------------------------------------------ 8.5.17

    /**
     * The attention projections {@code q = (x / sqrt(d)) Wq + bq}, {@code k = x Wk + bk},
     * {@code v = (x Wv + b1) + b2}.
     *
     * @param tokens rows of {@code x}
     * @param dim the width
     * @param seed the seed
     * @return the graph (input {@code x})
     */
    public static TensorExpressionGraph attentionGraph(int tokens, int dim, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        TensorExpressionGraph g = new TensorExpressionGraph();
        int x = g.input("x", Shape.of(tokens, dim));
        int wq = g.constant(Tensor.of(Shape.of(dim, dim), random(dim * dim, r)));
        int wk = g.constant(Tensor.of(Shape.of(dim, dim), random(dim * dim, r)));
        int wv = g.constant(Tensor.of(Shape.of(dim, dim), random(dim * dim, r)));
        int bq = g.constant(Tensor.of(Shape.of(dim), random(dim, r)));
        int bk = g.constant(Tensor.of(Shape.of(dim), random(dim, r)));
        int b1 = g.constant(Tensor.of(Shape.of(dim), random(dim, r)));
        int b2 = g.constant(Tensor.of(Shape.of(dim), random(dim, r)));
        int xs = g.op(Op.SCALE, (float) (1 / Math.sqrt(dim)), x);
        g.output(g.op(Op.ADD, null, g.op(Op.MATMUL, null, xs, wq), bq));
        g.output(g.op(Op.ADD, null, g.op(Op.MATMUL, null, x, wk), bk));
        g.output(g.op(Op.ADD, null, g.op(Op.ADD, null, g.op(Op.MATMUL, null, x, wv), b1), b2));
        return g;
    }

    /**
     * {@code relu(batchnorm(conv3x3(x)))} with inference statistics folded into a per-channel
     * affine.
     *
     * @param batch images
     * @param channels channels in and out
     * @param side image side
     * @param seed the seed
     * @return the graph (input {@code x})
     */
    public static TensorExpressionGraph convBnGraph(int batch, int channels, int side, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        TensorExpressionGraph g = new TensorExpressionGraph();
        int x = g.input("x", Shape.of(batch, channels, side, side));
        int w = g.constant(Tensor.of(Shape.of(channels, channels, 3, 3), random(channels * channels * 9, r)));
        float[] scale = random(channels, r);
        int s = g.constant(Tensor.of(Shape.of(channels), scale));
        int t = g.constant(Tensor.of(Shape.of(channels), random(channels, r)));
        int conv = g.op(Op.CONV, new int[] {1, 1}, x, w);
        g.output(g.op(Op.RELU, null, g.op(Op.AFFINE, null, conv, s, t)));
        return g;
    }

    /**
     * Rewrites both graphs with a single greedy pass, greedy to a fixpoint, and equality
     * saturation.
     *
     * @param seed the seed
     * @return per graph and method: {@code metric} the cost (FLOPs plus kernel overhead),
     *     {@code baseline} the original's, {@code detail} kernels, FLOPs, latency and the
     *     relative error against the original
     */
    public static List<Row> graphs(long seed) {
        List<Row> rows = new ArrayList<>();
        SplittableRandom r = new SplittableRandom(seed + 9);
        Map<String, TensorExpressionGraph> cases = new java.util.LinkedHashMap<>();
        cases.put("qkv", attentionGraph(64, 64, seed));
        cases.put("conv-bn", convBnGraph(4, 16, 16, seed));
        // the hand-written fusion order: fuse first, then fold scales, biases and norms
        List<EGraph.Rule> phases = List.of(EGraph.fuseSharedMatmuls(), EGraph.scaleIntoWeight(),
                EGraph.reassociateAdd(), EGraph.foldBatchNorm());
        for (Map.Entry<String, TensorExpressionGraph> e : cases.entrySet()) {
            TensorExpressionGraph g = e.getValue();
            Shape in = g.nodes().get(0).shape();
            Map<String, Tensor> inputs = Map.of("x", Tensor.of(in, random((int) in.size(), r)));
            List<Tensor> ref = g.evaluate(inputs);
            GreedyRewriter once = new GreedyRewriter(phases, false);
            GreedyRewriter fix = new GreedyRewriter(phases, true);
            EGraph eg = EGraph.of(g);
            int iterations = eg.saturate(EGraph.rules(), 10, 20_000);
            Map<String, TensorExpressionGraph> variants = new java.util.LinkedHashMap<>();
            variants.put("original", g.folded());
            variants.put("greedy 1 pass", once.rewrite(g).folded());
            variants.put("greedy fixpoint", fix.rewrite(g).folded());
            variants.put("e-graph", eg.extract().folded());
            for (Map.Entry<String, TensorExpressionGraph> v : variants.entrySet()) {
                TensorExpressionGraph h = v.getValue();
                double ms = KernelTimer.millis(() -> h.evaluate(inputs), 3, 9);
                List<Tensor> out = h.evaluate(inputs);
                double err = 0;
                for (int i = 0; i < out.size(); i++) {
                    err = Math.max(err, KernelTimer.relativeError(out.get(i).toArray(), ref.get(i).toArray()));
                }
                String extra = v.getKey().equals("e-graph") ? ", iters=" + iterations + ", rules=" + eg.applied()
                        : v.getKey().startsWith("greedy") ? ", fired=" + (v.getKey().endsWith("pass") ? once : fix)
                                .fired() : "";
                rows.add(new Row(e.getKey() + " " + v.getKey(), h.cost(), g.cost(), String.format(
                        "kernels=%d, flops=%.0f, %.3fms, err=%.1e%s", h.kernels(), h.flops(), ms, err, extra)));
            }
        }
        return rows;
    }

    // ------------------------------------------------------------------ 8.5.18

    /**
     * Sparse kernels against dense: {@code y = x W} with {@code x [batch, size]}.
     *
     * @param size the side of {@code W}
     * @param batch rows of {@code x}
     * @param sparsities shares of zeros
     * @param seed the seed
     * @return per sparsity and format the speed-up over the core dense matmul on the same
     *     pruned weight ({@code baseline} the relative error); 2:4 at 50% only
     */
    public static List<Row> sparse(int size, int batch, double[] sparsities, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Tensor dense = Tensor.of(Shape.of(size, size), random(size * size, r));
        Tensor x = Tensor.of(Shape.of(batch, size), random(batch * size, r));
        List<Row> rows = new ArrayList<>();
        for (double sp : sparsities) {
            Tensor unstructured = io.github.zzuegg.brainiac.experimental.nn.WeightPruning.magnitude(dense, sp);
            Tensor blocked = BlockSparseMatmul.prune(dense, 8, sp);
            double denseMs = KernelTimer.millis(() -> x.matmul(unstructured).toArray(), 5, 9);
            float[] ref = x.matmul(unstructured).toArray();
            float[] refBlocked = x.matmul(blocked).toArray();
            CsrMatmul csr = new CsrMatmul(unstructured);
            BlockSparseMatmul bsr = new BlockSparseMatmul(blocked, 8);
            SparseMatrix coo = SparseMatrix.fromDense(unstructured.transpose(), 0f);
            Tensor xt = x.transpose();
            double csrMs = KernelTimer.millis(() -> csr.multiply(x).toArray(), 5, 9);
            double bsrMs = KernelTimer.millis(() -> bsr.multiply(x).toArray(), 5, 9);
            double cooMs = KernelTimer.millis(() -> Var.noGrad(() -> coo.matmul(Var.of(xt)).value().toArray()), 5, 9);
            String pct = String.format("%.0f%%", 100 * sp);
            rows.add(new Row(pct + " csr", denseMs / csrMs,
                    KernelTimer.relativeError(csr.multiply(x).toArray(), ref), String.format("%.2fms", csrMs)));
            rows.add(new Row(pct + " bsr 8x8", denseMs / bsrMs,
                    KernelTimer.relativeError(bsr.multiply(x).toArray(), refBlocked), String.format("%.2fms", bsrMs)));
            rows.add(new Row(pct + " core coo", denseMs / cooMs, KernelTimer.relativeError(
                    Var.noGrad(() -> coo.matmul(Var.of(xt)).value()).transpose().toArray(), ref),
                    String.format("%.2fms vs dense %.2fms", cooMs, denseMs)));
            if (Math.abs(sp - 0.5) < 1e-9) {
                Tensor nm = NmSparseMatmul.prune(dense, 2, 4);
                NmSparseMatmul kernel = new NmSparseMatmul(nm, 2, 4);
                double nmDense = KernelTimer.millis(() -> x.matmul(nm).toArray(), 5, 9);
                double nmMs = KernelTimer.millis(() -> kernel.multiply(x).toArray(), 5, 9);
                rows.add(new Row(pct + " 2:4", nmDense / nmMs, KernelTimer.relativeError(kernel.multiply(x).toArray(),
                        x.matmul(nm).toArray()), String.format("%.2fms", nmMs)));
            }
        }
        return rows;
    }
}
