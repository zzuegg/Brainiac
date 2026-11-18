package io.github.zzuegg.brainiac.experiments.edge;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.edge.EdgeNet;
import io.github.zzuegg.brainiac.experimental.edge.ElasticSupernet;
import io.github.zzuegg.brainiac.experimental.edge.IntegerInference;
import io.github.zzuegg.brainiac.experimental.edge.KeywordSpotters;
import io.github.zzuegg.brainiac.experimental.edge.LiteResidual;
import io.github.zzuegg.brainiac.experimental.edge.MemoryPlanner;
import io.github.zzuegg.brainiac.experimental.edge.PatchBasedExecutor;
import io.github.zzuegg.brainiac.experimental.edge.SparseUpdateTrainer;
import io.github.zzuegg.brainiac.experimental.edge.SubnetSearch;
import io.github.zzuegg.brainiac.experimental.ml.DynamicTimeWarping;
import io.github.zzuegg.brainiac.experimental.train.BitFit;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Quantization;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Edge and TinyML deployment (wave 9, 9.6.13-9.6.17): integer-only inference, peak-memory-aware
 * patch execution and operator reordering, once-for-all supernets with a budgeted subnet search,
 * on-device training under a memory budget, and keyword spotting at microcontroller budgets. Every
 * model reports accuracy against model bytes, multiply-adds counted from the layer graph, and
 * peak activation memory.
 */
public final class EdgeDeployment {

    /** Pixel noise of the {@code ShapeImages} tasks. */
    public static final float NOISE = 0.15f;

    private EdgeDeployment() {
    }

    // ------------------------------------------------------------------ data

    /**
     * {@link ShapeImages} restricted to some classes, relabelled {@code 0..k-1}.
     *
     * @param side the image side
     * @param noise pixel noise
     * @param n samples to keep
     * @param classes the classes to keep, in label order
     * @param seed the seed
     * @return {@code {images, ids}}
     */
    public static Tensor[] shapes(int side, float noise, int n, int[] classes, long seed) {
        Tensor[] g = new ShapeImages(side, noise).generate(n * ShapeImages.CLASSES / classes.length + 32, seed);
        float[] img = g[0].toArray();
        float[] ids = g[1].toArray();
        int per = side * side;
        float[] x = new float[n * per];
        float[] y = new float[n];
        int k = 0;
        for (int i = 0; i < ids.length && k < n; i++) {
            for (int c = 0; c < classes.length; c++) {
                if ((int) ids[i] == classes[c]) {
                    System.arraycopy(img, i * per, x, k * per, per);
                    y[k++] = c;
                }
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(k, 1, side, side), Arrays.copyOf(x, k * per)),
            Tensor.of(Shape.of(k), Arrays.copyOf(y, k))};
    }

    private static int[] all(int k) {
        int[] a = new int[k];
        for (int i = 0; i < k; i++) {
            a[i] = i;
        }
        return a;
    }

    private static Tensor[] digits(int perClass, long seed, long sampleSeed) {
        ToyTasks.Data d = ToyTasks.digits(10, perClass, 8, 0.1, seed, sampleSeed);
        int n = d.x().length;
        int f = d.x()[0].length;
        float[] x = new float[n * f];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(d.x()[i], 0, x, i * f, f);
            y[i] = d.y()[i];
        }
        return new Tensor[] {Tensor.of(Shape.of(n, f), x), Tensor.of(Shape.of(n), y)};
    }

    /** Keyword classes: digits 0-5 are keywords, 6-9 are "unknown", plus "silence". */
    public static final int KEYWORDS = 6;
    /** The "unknown" class. */
    public static final int UNKNOWN = 6;
    /** The "silence" class. */
    public static final int SILENCE = 7;
    /** Classes. */
    public static final int KWS_CLASSES = 8;
    /** MFCC frames per utterance. */
    public static final int FRAMES = 16;
    /** MFCC coefficients per frame (with deltas). */
    public static final int COEFFICIENTS = 2 * SpokenDigitRecognition.CEPSTRA;

    /**
     * Keyword-spotting data.
     *
     * @param x MFCC images {@code [n, 1, FRAMES, COEFFICIENTS]}
     * @param y class ids
     * @param waves the waveforms
     * @param digits the spoken digit, or -1 for silence
     */
    public record KeywordData(Tensor x, Tensor y, List<float[]> waves, int[] digits) {
    }

    /**
     * Single-digit utterances by random speakers at 0-20 dB SNR, and background-noise silences.
     *
     * @param n samples
     * @param seed the seed
     * @return the data
     */
    public static KeywordData keywordData(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int per = FRAMES * COEFFICIENTS;
        float[] x = new float[n * per];
        float[] y = new float[n];
        int[] digit = new int[n];
        List<float[]> waves = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int c = r.nextInt(KWS_CLASSES);
            float[] w;
            if (c == SILENCE) {
                w = new float[FRAMES * SpokenDigitRecognition.HOP];
                double amp = 0.005 + 0.05 * r.nextDouble();
                for (int s = 0; s < w.length; s++) {
                    w[s] = (float) (amp * r.nextGaussian());
                }
                digit[i] = -1;
            } else {
                int d = c == UNKNOWN ? 6 + r.nextInt(4) : c;
                w = SpokenDigitRecognition.speak(new int[] {d}, SpokenDigitRecognition.speaker(r),
                        20 * r.nextDouble(), r);
                digit[i] = d;
            }
            waves.add(w);
            y[i] = c;
            float[][] m = SpokenDigitRecognition.mfcc(w, FRAMES);
            for (int t = 0; t < FRAMES; t++) {
                System.arraycopy(m[t], 0, x, i * per + t * COEFFICIENTS, COEFFICIENTS);
            }
        }
        return new KeywordData(Tensor.of(Shape.of(n, 1, FRAMES, COEFFICIENTS), x), Tensor.of(Shape.of(n), y), waves,
                digit);
    }

    // ------------------------------------------------------------------ 9.6.13

    /**
     * One model under one number format.
     *
     * @param model the task and network
     * @param method float32, post-training quantisation, QAT, ...
     * @param accuracy test accuracy
     * @param agreement for integer rows, the share of int8 outputs equal to the float fake-quant model's
     * @param bitExact for integer rows, the share of samples where the optimised and reference kernels agree exactly
     * @param bytes model bytes
     * @param macs multiply-adds per sample
     * @param microsPerSample wall time per sample on the CPU
     */
    public record IntegerRow(String model, String method, double accuracy, double agreement, double bitExact,
            long bytes, long macs, double microsPerSample) {
    }

    private static double floatMicros(EdgeNet net, Tensor x) {
        long t0 = System.nanoTime();
        Var.noGrad(() -> net.forward(Var.of(x), false));
        return (System.nanoTime() - t0) / 1e3 / x.shape().dim(0);
    }

    /**
     * Float, post-training int8, core weight-only fake quantisation, and QAT int8 for a CNN on
     * {@code ShapeImages} and an MLP on digits (9.6.13).
     *
     * @param steps float training steps (QAT fine-tuning uses a third more)
     * @param seed the seed
     * @return the rows
     */
    public static List<IntegerRow> integer(int steps, long seed) {
        List<IntegerRow> rows = new ArrayList<>();
        Tensor[] tr = shapes(8, NOISE, 512, all(6), seed);
        Tensor[] te = shapes(8, NOISE, 200, all(6), seed + 1);
        EdgeNet cnn = EdgeNet.builder(seed, 1, 8, 8).conv(8, 3, 1).conv(16, 3, 2).conv(16, 3, 2).pool()
                .dense(6, false).build();
        integerRows(rows, "ShapeImages CNN", cnn, tr, te, steps, seed);
        Tensor[] dtr = digits(60, seed, seed + 5);
        Tensor[] dte = digits(20, seed, seed + 1000);
        EdgeNet mlp = EdgeNet.builder(seed, 64).dense(64, true).dense(32, true).dense(10, false).build();
        integerRows(rows, "digits MLP", mlp, dtr, dte, steps, seed);
        return rows;
    }

    /**
     * The 9.6.13 rows for the DS-CNN keyword spotter (its int8 accuracy at every budget is also in
     * {@link #keywords}).
     *
     * @param steps float training steps
     * @param seed the seed
     * @return the rows
     */
    public static List<IntegerRow> integerKeywords(int steps, long seed) {
        List<IntegerRow> rows = new ArrayList<>();
        KeywordData ktr = keywordData(300, seed + 7);
        KeywordData kte = keywordData(120, seed + 8);
        EdgeNet kws = KeywordSpotters.sized(KeywordSpotters.Family.DS_CNN, KeywordSpotters.BUDGETS.get(0), FRAMES,
                COEFFICIENTS, KWS_CLASSES, seed).edgeNet();
        integerRows(rows, "keyword DS-CNN", kws, new Tensor[] {ktr.x(), ktr.y()}, new Tensor[] {kte.x(), kte.y()},
                steps, seed);
        return rows;
    }

    private static void integerRows(List<IntegerRow> rows, String name, EdgeNet net, Tensor[] tr, Tensor[] te,
            int steps, long seed) {
        net.train(tr[0], tr[1], steps, 32, 3e-3f, net.parameters(), seed);
        long floatBytes = 4 * net.parameterCount();
        rows.add(new IntegerRow(name, "float32", net.accuracy(te[0], te[1]), Double.NaN, Double.NaN, floatBytes,
                net.macs(), floatMicros(net, te[0])));
        Tensor calib = tr[0].slice(0, 0, Math.min(256, tr[0].shape().dim(0)));
        IntegerInference ptq = IntegerInference.convert(net, calib);
        rows.add(new IntegerRow(name, "post-training int8 (integer engine)", ptq.accuracy(te[0], te[1]),
                Double.NaN, ptq.bitExact(te[0]), ptq.modelBytes(), ptq.macs(),
                ptq.nanosPerSample(te[0], 1) / 1e3));
        // core Quantization: weight-only per-channel int8 rounding, float activations
        List<Tensor> saved = new ArrayList<>();
        Quantization q8 = Quantization.int8().perChannel();
        for (EdgeNet.Unit u : net.units()) {
            for (Var p : u.parameters()) {
                saved.add(p.value());
            }
            if (u instanceof EdgeNet.Conv c) {
                c.weight().assign(q8.quantize(c.weight().value(), 0));
            } else if (u instanceof EdgeNet.Linear l) {
                l.weight().assign(q8.quantize(l.weight().value(), 1));
            }
        }
        rows.add(new IntegerRow(name, "core Quantization weight-only int8 (float activations)",
                net.accuracy(te[0], te[1]), Double.NaN, Double.NaN, ptq.modelBytes(), net.macs(), Double.NaN));
        int k = 0;
        for (EdgeNet.Unit u : net.units()) {
            for (Var p : u.parameters()) {
                p.assign(saved.get(k++));
            }
        }
        net.quantAware(true);
        net.train(tr[0], tr[1], steps / 3 + 1, 32, 1e-3f, net.parameters(), seed + 11);
        rows.add(new IntegerRow(name, "QAT fake-quant (float simulation)", net.accuracy(te[0], te[1]), Double.NaN,
                Double.NaN, ptq.modelBytes(), net.macs(), floatMicros(net, te[0])));
        IntegerInference qat = IntegerInference.convert(net, calib);
        double[] agree = qat.agreement(net, te[0]);
        rows.add(new IntegerRow(name, "QAT int8 (integer engine)", qat.accuracy(te[0], te[1]), agree[0],
                qat.bitExact(te[0]), qat.modelBytes(), qat.macs(), qat.nanosPerSample(te[0], 1) / 1e3));
    }

    // ------------------------------------------------------------------ 9.6.14

    /**
     * Patch-based inference against layer-by-layer execution.
     *
     * @param network the network
     * @param grid patches per side (0 for layer by layer)
     * @param peakBytes peak int8 activation bytes
     * @param macs multiply-adds per sample
     * @param extraMacShare extra multiply-adds over layer by layer
     * @param maxDifference largest output difference from layer by layer
     */
    public record PatchRow(String network, int grid, long peakBytes, long macs, double extraMacShare,
            double maxDifference) {
    }

    /**
     * An MCUNet-style network: high-resolution early convolutions, then depthwise-separable
     * stages.
     *
     * @param side input side
     * @param width channel multiplier
     * @param seed the seed
     * @return the network
     */
    public static EdgeNet mcuNet(int side, int width, long seed) {
        return EdgeNet.builder(seed, 1, side, side).conv(4 * width, 3, 1).conv(8 * width, 3, 2)
                .separable(8 * width, 3, 1).conv(12 * width, 3, 2).separable(16 * width, 3, 1).pool()
                .dense(6, false).build();
    }

    /** Units of {@link #mcuNet} up to the 1/4-resolution point (run per patch). */
    public static final int MCUNET_SPLIT = 5;

    /**
     * Peak memory and extra compute of patch-based inference at several grids (9.6.14).
     *
     * @param side input side
     * @param grids patch grids
     * @param seed the seed
     * @return the rows (grid 0 is layer by layer)
     */
    public static List<PatchRow> patches(int side, int[] grids, long seed) {
        EdgeNet net = mcuNet(side, 2, seed);
        Tensor x = shapes(side, NOISE, 4, all(6), seed)[0];
        List<PatchRow> rows = new ArrayList<>();
        String name = "MCUNet-style " + side + "x" + side;
        rows.add(new PatchRow(name, 0, PatchBasedExecutor.layerByLayerPeak(net, 1), net.macs(), 0, 0));
        for (int g : grids) {
            PatchBasedExecutor p = new PatchBasedExecutor(net, MCUNET_SPLIT, g);
            rows.add(new PatchRow(name, g, p.peakBytes(1), p.macs(), p.extraMacShare(), p.maxDifference(x)));
        }
        return rows;
    }

    /**
     * Peak memory of an execution order against the best order.
     *
     * @param graph the graph
     * @param givenPeak peak of the given order, bytes
     * @param bestPeak peak of the best order, bytes
     * @param bestOrder the best order
     */
    public record OrderRow(String graph, long givenPeak, long bestPeak, String bestOrder) {
    }

    /**
     * Operator reordering on a multi-branch cell, and in-place depthwise convolution on the
     * MCUNet chain (9.6.14).
     *
     * @param seed the seed
     * @return the rows
     */
    public static List<OrderRow> reordering(long seed) {
        List<OrderRow> rows = new ArrayList<>();
        // an inception-like cell listed breadth first: three branches, each two ops, then concat
        List<MemoryPlanner.Node> cell = List.of(
                new MemoryPlanner.Node("input", new int[0], 4096, false),
                new MemoryPlanner.Node("a1 1x1 expand", new int[] {0}, 8192, false),
                new MemoryPlanner.Node("b1 1x1 expand", new int[] {0}, 8192, false),
                new MemoryPlanner.Node("c1 1x1 expand", new int[] {0}, 8192, false),
                new MemoryPlanner.Node("a2 3x3 reduce", new int[] {1}, 1024, false),
                new MemoryPlanner.Node("b2 5x5 reduce", new int[] {2}, 1024, false),
                new MemoryPlanner.Node("c2 pool reduce", new int[] {3}, 1024, false),
                new MemoryPlanner.Node("concat", new int[] {4, 5, 6}, 3072, false));
        MemoryPlanner m = new MemoryPlanner(cell);
        int[] best = m.bestOrder();
        StringBuilder sb = new StringBuilder();
        for (int i : best) {
            sb.append(cell.get(i).name().split(" ")[0]).append(' ');
        }
        rows.add(new OrderRow("3-branch cell", m.peak(m.defaultOrder()), m.peak(best), sb.toString().trim()));
        EdgeNet net = mcuNet(64, 2, seed);
        MemoryPlanner inPlace = MemoryPlanner.chain(net, 1);
        List<MemoryPlanner.Node> copy = new ArrayList<>();
        for (MemoryPlanner.Node n : inPlace.nodes()) {
            copy.add(new MemoryPlanner.Node(n.name(), n.inputs(), n.bytes(), false));
        }
        MemoryPlanner plain = new MemoryPlanner(copy);
        rows.add(new OrderRow("MCUNet-style 64x64 chain: in-place depthwise", plain.peak(plain.defaultOrder()),
                inPlace.peak(inPlace.defaultOrder()), "chain"));
        return rows;
    }

    /**
     * Accuracy at a fixed peak-memory budget.
     *
     * @param method how the budget is met
     * @param side input side
     * @param peakBytes peak int8 activation bytes
     * @param macs multiply-adds
     * @param accuracy test accuracy
     */
    public record BudgetRow(String method, int side, long peakBytes, long macs, double accuracy) {
    }

    private static Tensor downsample(Tensor x) {
        int n = x.shape().dim(0);
        int s = x.shape().dim(2);
        int h = s / 2;
        float[] v = x.toArray();
        float[] o = new float[n * h * h];
        for (int i = 0; i < n; i++) {
            for (int y = 0; y < h; y++) {
                for (int z = 0; z < h; z++) {
                    int b = i * s * s;
                    o[(i * h + y) * h + z] = (v[b + 2 * y * s + 2 * z] + v[b + 2 * y * s + 2 * z + 1]
                            + v[b + (2 * y + 1) * s + 2 * z] + v[b + (2 * y + 1) * s + 2 * z + 1]) / 4;
                }
            }
        }
        return Tensor.of(Shape.of(n, 1, h, h), o);
    }

    /**
     * At the peak memory that patch-based inference needs for an MCUNet-style network at full
     * resolution: the same network patch by patch, the network at half resolution, and a
     * narrower network at full resolution, layer by layer (9.6.14).
     *
     * @param side the full resolution
     * @param steps training steps
     * @param seed the seed
     * @return the rows
     */
    public static List<BudgetRow> budget(int side, int steps, long seed) {
        Tensor[] tr = shapes(side, NOISE, 512, all(6), seed);
        Tensor[] te = shapes(side, NOISE, 200, all(6), seed + 1);
        List<BudgetRow> rows = new ArrayList<>();
        EdgeNet full = mcuNet(side, 2, seed);
        PatchBasedExecutor patch = new PatchBasedExecutor(full, MCUNET_SPLIT, 4);
        long budget = patch.peakBytes(1);
        full.train(tr[0], tr[1], steps, 32, 3e-3f, full.parameters(), seed);
        double acc = full.accuracy(te[0], te[1]);
        rows.add(new BudgetRow("layer by layer, full resolution (over budget)", side,
                PatchBasedExecutor.layerByLayerPeak(full, 1), full.macs(), acc));
        Tensor out = patch.forward(te[0]);
        rows.add(new BudgetRow("patch-based 4x4, full resolution", side, budget, patch.macs(),
                accuracyOf(out, te[1])));
        EdgeNet half = mcuNet(side / 2, 2, seed);
        half.train(downsample(tr[0]), tr[1], steps, 32, 3e-3f, half.parameters(), seed);
        rows.add(new BudgetRow("layer by layer, half resolution", side / 2, PatchBasedExecutor.layerByLayerPeak(half, 1),
                half.macs(), half.accuracy(downsample(te[0]), te[1])));
        EdgeNet thin = mcuNet(side, 1, seed);
        thin.train(tr[0], tr[1], steps, 32, 3e-3f, thin.parameters(), seed);
        rows.add(new BudgetRow("layer by layer, half width", side, PatchBasedExecutor.layerByLayerPeak(thin, 1),
                thin.macs(), thin.accuracy(te[0], te[1])));
        return rows;
    }

    private static double accuracyOf(Tensor logits, Tensor y) {
        float[] z = logits.toArray();
        float[] ids = y.toArray();
        int k = z.length / ids.length;
        int ok = 0;
        for (int i = 0; i < ids.length; i++) {
            int best = 0;
            for (int c = 1; c < k; c++) {
                if (z[i * k + c] > z[i * k + best]) {
                    best = c;
                }
            }
            ok += best == (int) ids[i] ? 1 : 0;
        }
        return ok / (double) ids.length;
    }

    // ------------------------------------------------------------------ 9.6.15

    /**
     * A subnet under one training method.
     *
     * @param method how its weights were trained
     * @param subnet the subnet
     * @param macs multiply-adds
     * @param peakBytes peak int8 activations
     * @param accuracy test accuracy
     * @param trainingSteps total training steps the method spent for all deployments
     */
    public record OfaRow(String method, String subnet, long macs, long peakBytes, double accuracy, int trainingSteps) {
    }

    /**
     * A budgeted subnet search.
     *
     * @param budget the budget
     * @param method evolutionary search on the predictor, or random search on measured accuracy
     * @param subnet the subnet found
     * @param macs its multiply-adds
     * @param peakBytes its peak activations
     * @param predicted predicted accuracy (NaN for random search)
     * @param measured measured accuracy
     */
    public record SearchRow(String budget, String method, String subnet, long macs, long peakBytes, double predicted,
            double measured) {
    }

    /**
     * Once-for-all results.
     *
     * @param rows subnets under each training method
     * @param search budgeted searches
     * @param predictorTau Kendall tau of predicted against measured accuracy on held-out subnets
     */
    public record OfaResult(List<OfaRow> rows, List<SearchRow> search, double predictorTau) {
    }

    private static final int[] KERNELS = {3, 5};
    private static final int[] DEPTHS = {1, 2};
    private static final int[] EXPANDS = {2, 3, 4};

    private static ElasticSupernet supernet(long seed) {
        return new ElasticSupernet(1, 8, 6, 6, new int[] {8, 12}, 2, 4, seed);
    }

    /**
     * Progressive shrinking against training only the largest network, sampling everything from
     * the start, and training each deployed subnet from scratch; then a predictor-guided
     * evolutionary search under two budgets against random search (9.6.15).
     *
     * @param steps supernet training steps (each from-scratch subnet gets the same)
     * @param deployments subnets deployed
     * @param seed the seed
     * @return the results
     */
    public static OfaResult onceForAll(int steps, int deployments, long seed) {
        Tensor[] tr = shapes(8, NOISE, 512, all(6), seed);
        Tensor[] va = shapes(8, NOISE, 120, all(6), seed + 2);
        Tensor[] te = shapes(8, NOISE, 200, all(6), seed + 1);
        ElasticSupernet ps = supernet(seed);
        ps.train(tr[0], tr[1], steps, ElasticSupernet.Schedule.PROGRESSIVE, KERNELS, DEPTHS, EXPANDS, seed);
        SplittableRandom r = new SplittableRandom(seed + 3);
        List<ElasticSupernet.Config> deploy = new ArrayList<>();
        deploy.add(ps.largest());
        deploy.add(new ElasticSupernet.Config(new int[] {1, 1}, fill(4, 3), fill(4, 2)));
        while (deploy.size() < deployments) {
            deploy.add(ps.sample(r, KERNELS, DEPTHS, EXPANDS));
        }
        List<OfaRow> rows = new ArrayList<>();
        for (ElasticSupernet.Config c : deploy) {
            rows.add(new OfaRow("OFA progressive shrinking + KD", c.toString(), ps.macs(c), ps.peakBytes(c),
                    ps.accuracy(c, te[0], te[1]), steps));
        }
        ElasticSupernet big = supernet(seed);
        big.train(tr[0], tr[1], steps, ElasticSupernet.Schedule.LARGEST, KERNELS, DEPTHS, EXPANDS, seed);
        ElasticSupernet rnd = supernet(seed);
        rnd.train(tr[0], tr[1], steps, ElasticSupernet.Schedule.RANDOM, KERNELS, DEPTHS, EXPANDS, seed);
        for (ElasticSupernet.Config c : deploy) {
            rows.add(new OfaRow("largest only, subnets cut out", c.toString(), big.macs(c), big.peakBytes(c),
                    big.accuracy(c, te[0], te[1]), steps));
            rows.add(new OfaRow("random sampling, no shrinking or KD", c.toString(), rnd.macs(c), rnd.peakBytes(c),
                    rnd.accuracy(c, te[0], te[1]), steps));
            // the largest subnet trained alone is exactly the largest-only supernet
            ElasticSupernet own = big;
            if (!Arrays.equals(c.depths(), big.largest().depths()) || !Arrays.equals(c.kernels(), big.largest().kernels())
                    || !Arrays.equals(c.expands(), big.largest().expands())) {
                own = supernet(seed);
                own.trainFixed(tr[0], tr[1], steps, c, seed);
            }
            rows.add(new OfaRow("from scratch per subnet", c.toString(), own.macs(c), own.peakBytes(c),
                    own.accuracy(c, te[0], te[1]), steps * deployments));
        }
        // predictor and search
        SubnetSearch search = new SubnetSearch(ps, KERNELS, DEPTHS, EXPANDS, seed);
        List<ElasticSupernet.Config> fitSet = new ArrayList<>();
        double[] fitAcc = new double[24];
        for (int i = 0; i < fitAcc.length; i++) {
            ElasticSupernet.Config c = ps.sample(r, KERNELS, DEPTHS, EXPANDS);
            fitSet.add(c);
            fitAcc[i] = ps.accuracy(c, va[0], va[1]);
        }
        search.fit(fitSet, fitAcc, 400);
        double[] pred = new double[12];
        double[] meas = new double[12];
        for (int i = 0; i < pred.length; i++) {
            ElasticSupernet.Config c = ps.sample(r, KERNELS, DEPTHS, EXPANDS);
            pred[i] = search.predict(c);
            meas[i] = ps.accuracy(c, va[0], va[1]);
        }
        double tau = SubnetSearch.kendallTau(pred, meas);
        long bigMacs = ps.macs(ps.largest());
        long bigPeak = ps.peakBytes(ps.largest());
        List<SearchRow> found = new ArrayList<>();
        for (double share : new double[] {0.3, 0.6}) {
            long macs = (long) (share * bigMacs);
            long peak = (long) (Math.sqrt(share) * bigPeak);
            String b = String.format("%.0f%% MACs", 100 * share);
            ElasticSupernet.Config evo = search.evolve(macs, peak, 1500, seed);
            found.add(new SearchRow(b, "predictor + evolution", evo.toString(), ps.macs(evo), ps.peakBytes(evo),
                    search.predict(evo), ps.accuracy(evo, te[0], te[1])));
            ElasticSupernet.Config rs = search.randomSearch(macs, peak, 10, c -> ps.accuracy(c, va[0], va[1]),
                    seed + 9);
            if (rs != null) {
                found.add(new SearchRow(b, "random search, 10 measured", rs.toString(), ps.macs(rs), ps.peakBytes(rs),
                        Double.NaN, ps.accuracy(rs, te[0], te[1])));
            }
        }
        return new OfaResult(rows, found, tau);
    }

    private static int[] fill(int n, int v) {
        int[] a = new int[n];
        Arrays.fill(a, v);
        return a;
    }

    // ------------------------------------------------------------------ 9.6.16

    /**
     * An adaptation method under the training-memory model.
     *
     * @param method the method
     * @param trainingBytes training memory (int8 activations)
     * @param accuracy accuracy on the new task
     * @param detail what was updated
     */
    public record AdaptRow(String method, long trainingBytes, double accuracy, String detail) {
    }

    private static EdgeNet adaptNet(long seed) {
        return EdgeNet.builder(seed, 1, 8, 8).conv(8, 3, 1).conv(16, 3, 2).separable(16, 3, 1)
                .conv(24, 3, 2).pool().dense(4, false).build();
    }

    /**
     * On-device adaptation of a CNN pretrained on four glyph classes to a task with two new
     * glyphs, under a training-memory budget (9.6.16).
     *
     * @param pretrainSteps pretraining steps
     * @param adaptSteps adaptation steps
     * @param budgetShare the budget as a share of full fine-tuning's training memory
     * @param seed the seed
     * @return the rows
     */
    public static List<AdaptRow> adaptation(int pretrainSteps, int adaptSteps, double budgetShare, long seed) {
        Tensor[] pre = shapes(8, NOISE, 512, new int[] {0, 1, 2, 3}, seed);
        Tensor[] tr = shapes(8, NOISE, 256, new int[] {2, 3, 4, 5}, seed + 1);
        Tensor[] va = shapes(8, NOISE, 120, new int[] {2, 3, 4, 5}, seed + 2);
        Tensor[] te = shapes(8, NOISE, 200, new int[] {2, 3, 4, 5}, seed + 3);
        EdgeNet net = adaptNet(seed);
        net.train(pre[0], pre[1], pretrainSteps, 32, 3e-3f, net.parameters(), seed);
        net.foldBatchNorm();
        List<Tensor> pretrained = new ArrayList<>();
        net.parameters().forEach(p -> pretrained.add(p.value()));
        int n = net.units().size();
        float lr = 0.05f;
        List<AdaptRow> rows = new ArrayList<>();
        rows.add(new AdaptRow("pretrained, no adaptation", 0, net.accuracy(te[0], te[1]), "-"));

        SparseUpdateTrainer.Selection fullSel = SparseUpdateTrainer.Selection.none(n);
        for (int i = 0; i < n; i++) {
            fullSel.weightShare()[i] = net.units().get(i).parameters().isEmpty() ? 0 : 1;
            fullSel.bias()[i] = !net.units().get(i).parameters().isEmpty();
        }
        SparseUpdateTrainer fl = new SparseUpdateTrainer(net, SparseUpdateTrainer.Precision.FLOAT, lr);
        long fullBytes = fl.memory(fullSel, 0, 0).total();
        long budget = (long) (budgetShare * fullBytes);
        fl.train(tr[0], tr[1], fullSel, List.of(), adaptSteps, 32, seed);
        rows.add(new AdaptRow("full fine-tuning, float (upper bound)", fullBytes, net.accuracy(te[0], te[1]),
                "all"));

        restore(net, pretrained);
        SparseUpdateTrainer q = new SparseUpdateTrainer(net, SparseUpdateTrainer.Precision.INT8_QAS, lr);
        SparseUpdateTrainer.Selection last = SparseUpdateTrainer.Selection.none(n);
        last.weightShare()[n - 1] = 1;
        last.bias()[n - 1] = true;
        q.train(tr[0], tr[1], last, List.of(), adaptSteps, 32, seed);
        rows.add(new AdaptRow("last layer only, int8 + QAS", q.memory(last, 0, 0).total(), net.accuracy(te[0], te[1]),
                last.toString()));

        q.reset();
        SparseUpdateTrainer.Selection bitfit = last.copy();
        List<Var> biases = BitFit.biases(net.parameters());
        for (int i = 0; i < n; i++) {
            for (Var p : net.units().get(i).parameters()) {
                bitfit.bias()[i] |= biases.stream().anyMatch(b -> b == p);
            }
        }
        q.train(tr[0], tr[1], bitfit, List.of(), adaptSteps, 32, seed);
        rows.add(new AdaptRow("BitFit (biases) + head, int8 + QAS", q.memory(bitfit, 0, 0).total(),
                net.accuracy(te[0], te[1]), String.format("%.1f%% of parameters",
                        100 * BitFit.share(biases, net.parameters()))));

        q.reset();
        int[] convs = {1, 3, 4};
        LiteResidual lite = new LiteResidual(net, convs, 4, seed);
        q.train(tr[0], tr[1], bitfit, lite.parameters(), adaptSteps, 32, seed);
        rows.add(new AdaptRow("TinyTL lite residual + biases + head, int8 + QAS",
                q.memory(bitfit, lite.storedBytes(1), lite.parameterCount()).total(), net.accuracy(te[0], te[1]),
                lite.parameterCount() + " branch parameters"));
        lite.detach();

        q.reset();
        List<Var> lora = attachLora(net, 2, seed);
        long loraParams = lora.stream().mapToLong(v -> v.value().size()).sum();
        long loraStored = 0;
        List<int[]> shapes = net.shapes();
        for (int i = 0; i < n; i++) {
            if (isAdapted(net.units().get(i))) {
                loraStored += EdgeNet.size(i == 0 ? net.inputShape() : shapes.get(i - 1));
            }
        }
        q.train(tr[0], tr[1], last, lora, adaptSteps, 32, seed);
        rows.add(new AdaptRow("LoRA rank 2 on 1x1 convs + head", q.memory(last, loraStored, loraParams).total(),
                net.accuracy(te[0], te[1]), loraParams + " adapter parameters"));
        detachLora(net);

        q.reset();
        List<SparseUpdateTrainer.Candidate> cands = q.contributions(tr[0], tr[1], va[0], va[1],
                new double[] {0.25, 1.0}, Math.max(10, adaptSteps / 4), seed);
        SparseUpdateTrainer.Selection sparse = q.select(cands, budget);
        q.reset();
        q.train(tr[0], tr[1], sparse, List.of(), adaptSteps, 32, seed);
        rows.add(new AdaptRow("sparse update (contribution analysis), int8 + QAS", q.memory(sparse, 0, 0).total(),
                net.accuracy(te[0], te[1]), sparse.toString()));

        restore(net, pretrained);
        SparseUpdateTrainer noQas = new SparseUpdateTrainer(net, SparseUpdateTrainer.Precision.INT8, lr);
        noQas.train(tr[0], tr[1], sparse, List.of(), adaptSteps, 32, seed);
        rows.add(new AdaptRow("sparse update, int8 without QAS", noQas.memory(sparse, 0, 0).total(),
                net.accuracy(te[0], te[1]), sparse.toString()));

        restore(net, pretrained);
        SparseUpdateTrainer fs = new SparseUpdateTrainer(net, SparseUpdateTrainer.Precision.FLOAT, lr);
        fs.train(tr[0], tr[1], sparse, List.of(), adaptSteps, 32, seed);
        rows.add(new AdaptRow("sparse update, float", fs.memory(sparse, 0, 0).total(), net.accuracy(te[0], te[1]),
                sparse.toString()));
        rows.add(new AdaptRow("(budget)", budget, Double.NaN, String.format("%.0f%% of full fine-tuning",
                100 * budgetShare)));
        return rows;
    }

    private static void restore(EdgeNet net, List<Tensor> values) {
        List<Var> p = net.parameters();
        for (int i = 0; i < p.size(); i++) {
            p.get(i).assign(values.get(i));
        }
    }

    private static boolean isAdapted(EdgeNet.Unit u) {
        return u instanceof EdgeNet.Linear || u instanceof EdgeNet.Conv c && c.kernel() == 1;
    }

    private static List<Var> attachLora(EdgeNet net, int rank, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Var> params = new ArrayList<>();
        for (EdgeNet.Unit u : net.units()) {
            if (!isAdapted(u)) {
                continue;
            }
            Shape ws = u instanceof EdgeNet.Conv c ? c.weight().shape() : ((EdgeNet.Linear) u).weight().shape();
            boolean conv = u instanceof EdgeNet.Conv;
            int out = conv ? ws.dim(0) : ws.dim(1);
            int in = conv ? ws.dim(1) : ws.dim(0);
            Var a = Var.param(Tensor.normal(Shape.of(in, rank), 0, (float) Math.sqrt(1.0 / in), r));
            Var b = Var.param(Tensor.zeros(Shape.of(rank, out)));
            params.add(a);
            params.add(b);
            net.setWeightDelta(u, conv ? () -> a.matmul(b).transpose().reshape(out, in, 1, 1) : () -> a.matmul(b));
        }
        return params;
    }

    private static void detachLora(EdgeNet net) {
        for (EdgeNet.Unit u : net.units()) {
            net.setWeightDelta(u, null);
        }
    }

    // ------------------------------------------------------------------ 9.6.17

    /**
     * A keyword spotter.
     *
     * @param budget the budget it was sized for
     * @param family the model family
     * @param weightBytes int8 weight bytes
     * @param macs multiply-adds per inference
     * @param fits whether it fits the budget
     * @param peakBytes peak int8 activation bytes (layer by layer; 0 when not planned)
     * @param floatAccuracy float accuracy
     * @param int8Accuracy integer-engine accuracy (NaN for families not converted)
     */
    public record KeywordRow(String budget, String family, long weightBytes, long macs, boolean fits, long peakBytes,
            double floatAccuracy, double int8Accuracy) {
    }

    /**
     * Every family at every budget, trained on the same data, converted to int8 where supported
     * (9.6.17).
     *
     * @param budgets the budgets
     * @param families the families
     * @param steps training steps
     * @param seed the seed
     * @return the rows
     */
    public static List<KeywordRow> keywords(List<KeywordSpotters.Budget> budgets, KeywordSpotters.Family[] families,
            int steps, long seed) {
        KeywordData tr = keywordData(300, seed);
        KeywordData te = keywordData(120, seed + 1);
        List<KeywordRow> rows = new ArrayList<>();
        for (KeywordSpotters.Budget b : budgets) {
            for (KeywordSpotters.Family f : families) {
                KeywordSpotters.Model m = KeywordSpotters.sized(f, b, FRAMES, COEFFICIENTS, KWS_CLASSES, seed);
                KeywordSpotters.train(m.model(), tr.x(), tr.y(), steps, 3e-3f, seed);
                double fa = KeywordSpotters.accuracy(m.model(), te.x(), te.y());
                double ia = Double.NaN;
                long peak = 0;
                EdgeNet e = m.edgeNet();
                if (e != null) {
                    Tensor calib = KeywordSpotters.inputTensor(e, tr.x().slice(0, 0, 200));
                    IntegerInference ii = IntegerInference.convert(e, calib);
                    ia = ii.accuracy(KeywordSpotters.inputTensor(e, te.x()), te.y());
                    peak = PatchBasedExecutor.layerByLayerPeak(e, 1);
                }
                rows.add(new KeywordRow(b.name(), f.name(), m.weightBytes(), m.macs(), KeywordSpotters.fits(m, b), peak,
                        fa, ia));
            }
        }
        return rows;
    }

    /**
     * Unconstrained baselines: {@code SpokenDigitRecognition}'s frame-wise MLP and Conformer
     * recognisers trained with CTC (an empty transcript means silence), and MFCC + DTW templates.
     *
     * @param steps CTC steps
     * @param templates DTW templates per class
     * @param seed the seed
     * @return the rows
     */
    public static List<KeywordRow> keywordBaselines(int steps, int templates, long seed) {
        KeywordData tr = keywordData(300, seed);
        KeywordData te = keywordData(120, seed + 1);
        List<KeywordRow> rows = new ArrayList<>();
        List<float[][]> trFeats = new ArrayList<>();
        List<int[]> trDigits = new ArrayList<>();
        for (int i = 0; i < tr.waves().size(); i++) {
            if (tr.digits()[i] >= 0) {
                trFeats.add(SpokenDigitRecognition.mfcc(tr.waves().get(i), FRAMES));
                trDigits.add(new int[] {tr.digits()[i]});
            }
        }
        List<float[][]> teFeats = new ArrayList<>();
        for (float[] w : te.waves()) {
            teFeats.add(SpokenDigitRecognition.mfcc(w, FRAMES));
        }
        float[] truth = te.y().toArray();
        for (SpokenDigitRecognition.Encoder enc : new SpokenDigitRecognition.Encoder[] {
            SpokenDigitRecognition.Encoder.MLP, SpokenDigitRecognition.Encoder.CONFORMER}) {
            SpokenDigitRecognition.Recognizer rec = new SpokenDigitRecognition.Recognizer(enc, COEFFICIENTS, FRAMES,
                    seed);
            SplittableRandom r = new SplittableRandom(seed + 3);
            for (int s = 0; s < steps; s++) {
                List<float[][]> bf = new ArrayList<>();
                List<int[]> bd = new ArrayList<>();
                for (int i = 0; i < 16; i++) {
                    int j = r.nextInt(trFeats.size());
                    bf.add(trFeats.get(j));
                    bd.add(trDigits.get(j));
                }
                rec.trainCtc(bf, bd, false, r);
            }
            int[][] heard = rec.recognize(teFeats);
            int ok = 0;
            for (int i = 0; i < heard.length; i++) {
                int c = heard[i].length == 0 ? SILENCE : heard[i][0] < KEYWORDS ? heard[i][0] : UNKNOWN;
                ok += c == (int) truth[i] ? 1 : 0;
            }
            long params = rec.parameterCount();
            rows.add(new KeywordRow("unconstrained", enc.name() + " + CTC", params, 0,
                    params <= KeywordSpotters.BUDGETS.get(0).weightBytes(), 0, ok / (double) heard.length,
                    Double.NaN));
        }
        DynamicTimeWarping.TemplateClassifier dtw = new DynamicTimeWarping.TemplateClassifier(4);
        int[] count = new int[KWS_CLASSES];
        long stored = 0;
        float[] trLabels = tr.y().toArray();
        for (int i = 0; i < tr.waves().size(); i++) {
            int c = (int) trLabels[i];
            if (count[c] < templates) {
                float[][] m = SpokenDigitRecognition.mfcc(tr.waves().get(i), -1);
                dtw.add(c, m);
                stored += (long) m.length * m[0].length;
                count[c]++;
            }
        }
        int ok = 0;
        for (int i = 0; i < te.waves().size(); i++) {
            ok += dtw.classify(SpokenDigitRecognition.mfcc(te.waves().get(i), -1)) == (int) truth[i] ? 1 : 0;
        }
        rows.add(new KeywordRow("unconstrained", "MFCC + DTW (" + templates + " templates/class)", stored, 0, false,
                0, ok / (double) te.waves().size(), Double.NaN));
        return rows;
    }
}
