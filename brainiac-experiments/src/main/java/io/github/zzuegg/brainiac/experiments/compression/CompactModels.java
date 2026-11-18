package io.github.zzuegg.brainiac.experiments.compression;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.compress.DecoupledKdLoss;
import io.github.zzuegg.brainiac.experimental.compress.DoReFaQuantizer;
import io.github.zzuegg.brainiac.experimental.compress.FixedSparsityOptimizer;
import io.github.zzuegg.brainiac.experimental.compress.HessianAwareBitAllocation;
import io.github.zzuegg.brainiac.experimental.compress.PactActivation;
import io.github.zzuegg.brainiac.experimental.compress.TensorDecomposition;
import io.github.zzuegg.brainiac.experimental.compress.TensorTrainDense;
import io.github.zzuegg.brainiac.experimental.compress.TokenMerging;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.kernels.KernelTimer;
import io.github.zzuegg.brainiac.experimental.nn.LowRankDense;
import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.experimental.compress.WeightSharingQuantizer;
import io.github.zzuegg.brainiac.experimental.compress.RelationalKdLoss;
import io.github.zzuegg.brainiac.experimental.nn.DistillationLoss;
import io.github.zzuegg.brainiac.experimental.nn.FeatureDistillationLoss;
import io.github.zzuegg.brainiac.experimental.nn.Gptq;
import io.github.zzuegg.brainiac.experimental.nn.LsqDense;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.nn.WeightPruning;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Making models smaller other than by pruning, on the {@link CompressionBench} digits task and
 * the {@link ShapesCnn}:
 * <ul>
 * <li>{@link #distillation} (wave 8 item 8.5.8): decoupled and relational knowledge distillation
 *     against logit KD, feature (FitNet) distillation and the student alone; RKD also on an
 *     embedding-retrieval task;</li>
 * <li>{@link #weightSharing} (8.5.9): Deep Compression (pruning, k-means weight sharing with
 *     centroid fine-tuning, Huffman coding) against round-to-nearest and GPTQ at the same bits,
 *     and pruning alone, by accuracy against total stored bits;</li>
 * <li>{@link #mixedPrecision} (8.5.10): HAWQ-V2 bit allocation against uniform, random and
 *     weight-variance allocation at the same average bits;</li>
 * <li>{@link #quantizationAware} (8.5.11): PACT and DoReFa against LSQ and fixed-range clipping
 *     for 2 to 4-bit weights and activations;</li>
 * <li>{@link #denseDecompositions} and {@link #convDecompositions} (8.5.12): tensor-train and
 *     CP/Tucker-2 decompositions of layers against the matrix SVD at equal parameters;</li>
 * <li>{@link #tokenMerging} (8.5.13): ToMe against class-attention token pruning on a small ViT.</li>
 * </ul>
 */
public final class CompactModels {

    private CompactModels() {
    }

    // ------------------------------------------------------------------ 8.5.8

    /**
     * Distils a {@code 64-128-128-10} teacher into a {@code 64-24-10} student (about 12x fewer
     * weights), and a {@code 64-32-16} embedding student for nearest-neighbour retrieval.
     *
     * @param teacherSteps teacher training steps
     * @param studentSteps student training steps
     * @param seed the seed
     * @return classification rows ({@code metric} test accuracy): teacher, student alone, KD,
     *     DKD, FitNet, RKD; then retrieval rows ({@code metric} recall@1 on the test set):
     *     teacher, untrained student, FitNet-only student, RKD-only student
     */
    public static List<CompactRow> distillation(int teacherSteps, int studentSteps, long seed) {
        CompressionBench.Split data = CompressionBench.digits(seed);
        ToyTasks.Data train = data.train();
        SimpleMlp teacher = SimpleMlp.relu(seed, 64, 128, 128, 10);
        CompressionBench.train(teacher::forward, teacher.parameters(), Optimizer.adam(0.01f), train, teacherSteps);
        Tensor x = train.inputs();
        Tensor ids = train.ids();
        Tensor teacherLogits = teacher.predict(x);
        Tensor teacherHidden = teacher.activations(x)[2];
        List<CompactRow> rows = new ArrayList<>();
        rows.add(new CompactRow("teacher", CompressionBench.accuracy(teacher::forward, data.test()), 0,
                count(teacher), -1));
        DistillationLoss kd = new DistillationLoss(4);
        DecoupledKdLoss dkd = new DecoupledKdLoss(4, 1, 8);
        RelationalKdLoss rkd = new RelationalKdLoss(1, 2);
        for (String method : List.of("student alone", "kd", "dkd", "fitnet", "rkd")) {
            SimpleMlp student = SimpleMlp.relu(seed + 1, 64, 24, 10);
            FeatureDistillationLoss fit = new FeatureDistillationLoss(24, 128, seed);
            List<Var> params = new ArrayList<>(student.parameters());
            if (method.equals("fitnet")) {
                params.addAll(fit.parameters());
            }
            Optimizer opt = Optimizer.adam(0.01f);
            SplittableRandom r = new SplittableRandom(seed);
            for (int s = 0; s < studentSteps; s++) {
                Var hidden = student.layer(0, Var.of(x));
                Var logits = student.layer(1, hidden);
                Var ce = Functional.crossEntropy(logits, ids);
                Var loss = switch (method) {
                    case "kd" -> kd.withLabels(logits, teacherLogits, ids, 0.9f);
                    case "dkd" -> ce.add(dkd.compute(logits, teacherLogits, ids));
                    case "fitnet" -> ce.add(fit.compute(hidden, teacherHidden));
                    case "rkd" -> {
                        Tensor pick = sample(r, x.shape().dim(0), 32);
                        yield ce.add(rkd.compute(hidden.gather(0, pick), teacherHidden.gather(0, pick)));
                    }
                    default -> ce;
                };
                loss.backward();
                opt.step(params);
            }
            rows.add(new CompactRow(method, CompressionBench.accuracy(student::forward, data.test()), 0,
                    count(student), -1));
        }
        // retrieval: embeddings only, no labels
        Tensor testX = data.test().inputs();
        int[] testY = data.test().y();
        rows.add(new CompactRow("retrieval: teacher", recallAt1(teacher.activations(testX)[2], testY), 0,
                count(teacher), -1));
        for (String method : List.of("untrained", "fitnet", "rkd")) {
            SimpleMlp emb = SimpleMlp.relu(seed + 2, 64, 32, 16);
            FeatureDistillationLoss fit = new FeatureDistillationLoss(16, 128, seed);
            List<Var> params = new ArrayList<>(emb.parameters());
            params.addAll(fit.parameters());
            Optimizer opt = Optimizer.adam(0.01f);
            SplittableRandom r = new SplittableRandom(seed);
            int steps = method.equals("untrained") ? 0 : studentSteps;
            for (int s = 0; s < steps; s++) {
                Tensor pick = sample(r, x.shape().dim(0), 48);
                Var e = emb.forward(Var.of(x.gather(0, pick)));
                Tensor t = teacherHidden.gather(0, pick);
                (method.equals("rkd") ? rkd.compute(e, t) : fit.compute(e, t)).backward();
                opt.step(params);
            }
            Function<Tensor, Tensor> embed = emb::predict;
            rows.add(new CompactRow("retrieval: " + method, recallAt1(embed.apply(testX), testY), 0, count(emb), -1));
        }
        return rows;
    }

    // ------------------------------------------------------------------ 8.5.9

    /**
     * Storage compression of the reference digits MLP's weight matrices.
     *
     * @param bitsList the bits per weight to compare
     * @param sparsity the pruning share of Deep Compression and pruning alone
     * @param fineTune recovery and centroid fine-tuning steps
     * @param seed the seed
     * @return fp32 and pruning alone first, then per bits: round-to-nearest, GPTQ, weight sharing,
     *     Deep Compression without and with Huffman coding; {@code metric} is test accuracy,
     *     {@code cost} the stored bits of the weight matrices (with codebooks, scales, positions
     *     and code tables), {@code size} the compression ratio against fp32
     */
    public static List<CompactRow> weightSharing(int[] bitsList, double sparsity, int fineTune, long seed) {
        CompressionBench.Split data = CompressionBench.digits(seed);
        SimpleMlp ref = CompressionBench.trained(seed);
        long params = 0;
        for (Var w : CompressionBench.weights(ref)) {
            params += w.shape().size();
        }
        double fp32 = 32.0 * params;
        List<CompactRow> rows = new ArrayList<>();
        rows.add(new CompactRow("fp32", CompressionBench.accuracy(ref::forward, data.test()), fp32, 1, -1));
        SimpleMlp pruned = prunedReference(seed, sparsity, fineTune, data);
        long prunedBits = 0;
        for (Var w : CompressionBench.weights(pruned)) {
            prunedBits += countEntries(w.value(), 4) * (32 + 4);   // fp32 value and 4-bit gap per entry
        }
        rows.add(new CompactRow("pruned " + sparsity + " fp32", CompressionBench.accuracy(pruned::forward,
                data.test()), prunedBits, fp32 / prunedBits, -1));
        Tensor[] acts = ref.activations(data.train().inputs());
        for (int bits : bitsList) {
            SimpleMlp rtn = CompressionBench.trained(seed);
            SimpleMlp gptq = CompressionBench.trained(seed);
            long rtnBits = 0;
            for (int l = 0; l < rtn.depth(); l++) {
                Tensor w = rtn.weight(l).value();
                rtn.weight(l).assign(Gptq.roundToNearest(w, bits));
                gptq.weight(l).assign(new Gptq(bits).quantize(w, acts[l]));
                rtnBits += (long) bits * w.shape().size() + 32L * w.shape().dim(1);   // + a scale per column
            }
            rows.add(new CompactRow("rtn " + bits + "b", CompressionBench.accuracy(rtn::forward, data.test()), rtnBits,
                    fp32 / rtnBits, -1));
            rows.add(new CompactRow("gptq " + bits + "b", CompressionBench.accuracy(gptq::forward, data.test()),
                    rtnBits, fp32 / rtnBits, -1));
            for (String method : List.of("sharing", "deep compression", "deep compression+huffman")) {
                SimpleMlp net = method.equals("sharing") ? CompressionBench.trained(seed)
                        : prunedReference(seed, sparsity, fineTune, data);
                List<WeightSharingQuantizer.Shared> shared = new ArrayList<>();
                List<Var> trainable = new ArrayList<>();
                for (int l = 0; l < net.depth(); l++) {
                    WeightSharingQuantizer.Shared s = WeightSharingQuantizer.fit(net.weight(l).value(), bits, 30);
                    shared.add(s);
                    trainable.add(s.centroids());
                    trainable.add(net.bias(l));
                }
                UnaryOperator<Var> fwd = CompressionBench.forward(net, (l, w) -> shared.get(l).weight(), null);
                CompressionBench.train(fwd, trainable, Optimizer.adam(0.002f), data.train(), fineTune);
                long stored = 0;
                for (WeightSharingQuantizer.Shared s : shared) {
                    stored += s.storageBits(method.endsWith("huffman"), method.equals("sharing") ? 0 : 4);
                }
                rows.add(new CompactRow(method + " " + bits + "b", CompressionBench.accuracy(fwd, data.test()), stored,
                        fp32 / stored, -1));
            }
        }
        return rows;
    }

    // ------------------------------------------------------------------ 8.5.10

    /**
     * Mixed-precision post-training quantisation (round-to-nearest per column) of a four-layer
     * {@code 64-96-96-96-10} digits MLP.
     *
     * @param averageBits the budget in average bits per weight
     * @param steps training steps of the network
     * @param seed the seed
     * @return fp32; uniform floor(budget) and floor(budget)+1 bits; then at the budget: HAWQ, three
     *     random allocations and variance allocation; {@code metric} is test accuracy,
     *     {@code cost} the average bits, {@code size} the bits of the first layer (as a hint of
     *     the allocation)
     */
    public static List<CompactRow> mixedPrecision(double averageBits, int steps, long seed) {
        CompressionBench.Split data = CompressionBench.digits(seed);
        SimpleMlp net = SimpleMlp.relu(seed, 64, 96, 96, 96, 10);
        CompressionBench.train(net::forward, net.parameters(), Optimizer.adam(0.01f), data.train(), steps);
        List<Var> w = CompressionBench.weights(net);
        List<Tensor> saved = new ArrayList<>();
        long[] sizes = new long[w.size()];
        for (int l = 0; l < w.size(); l++) {
            saved.add(w.get(l).value());
            sizes[l] = w.get(l).shape().size();
        }
        int[] candidates = {2, 3, 4, 6, 8};
        java.util.function.BiFunction<Tensor, Integer, Tensor> rtn = Gptq::roundToNearest;
        Tensor x = data.train().inputs();
        Tensor ids = data.train().ids();
        double[] traces = HessianAwareBitAllocation.traces(
                () -> Functional.crossEntropy(net.forward(Var.of(x)), ids), w, 16, seed);
        List<CompactRow> rows = new ArrayList<>();
        rows.add(new CompactRow("fp32", CompressionBench.accuracy(net::forward, data.test()), 32, 32, -1));
        int lo = (int) Math.floor(averageBits);
        for (int b : new int[] {lo, lo + 1}) {
            int[] uniform = new int[w.size()];
            java.util.Arrays.fill(uniform, b);
            rows.add(quantized("uniform " + b + "b", net, w, saved, uniform, sizes, rtn, data));
        }
        int[] hawq = HessianAwareBitAllocation.allocate(
                HessianAwareBitAllocation.sensitivities(traces, w, candidates, rtn), candidates, sizes, averageBits);
        rows.add(quantized("hawq " + java.util.Arrays.toString(hawq), net, w, saved, hawq, sizes, rtn, data));
        for (int k = 0; k < 3; k++) {
            int[] rnd = HessianAwareBitAllocation.random(candidates, sizes, averageBits, seed + 101L * k);
            rows.add(quantized("random " + java.util.Arrays.toString(rnd), net, w, saved, rnd, sizes, rtn, data));
        }
        int[] var = HessianAwareBitAllocation.allocate(HessianAwareBitAllocation.sensitivities(
                HessianAwareBitAllocation.weightVariances(w), w, candidates, rtn), candidates, sizes, averageBits);
        rows.add(quantized("variance " + java.util.Arrays.toString(var), net, w, saved, var, sizes, rtn, data));
        return rows;
    }

    private static CompactRow quantized(String name, SimpleMlp net, List<Var> w, List<Tensor> saved, int[] bits,
            long[] sizes, java.util.function.BiFunction<Tensor, Integer, Tensor> quantizer,
            CompressionBench.Split data) {
        for (int l = 0; l < w.size(); l++) {
            w.get(l).assign(quantizer.apply(saved.get(l), bits[l]));
        }
        double acc = CompressionBench.accuracy(net::forward, data.test());
        for (int l = 0; l < w.size(); l++) {
            w.get(l).assign(saved.get(l));
        }
        return new CompactRow(name, acc, HessianAwareBitAllocation.averageBits(bits, sizes), bits[0], -1);
    }

    // ------------------------------------------------------------------ 8.5.11

    /**
     * Quantisation-aware fine-tuning of the reference digits MLP with {@code k}-bit weights (all
     * layers) and {@code k}-bit hidden activations. DoReFa weights are rescaled to each layer's
     * initial range, since the network has no batch norm to absorb the change of scale.
     *
     * @param bitsList the bit widths
     * @param steps fine-tuning steps
     * @param seed the seed
     * @return fp32, then per bit width: PACT (DoReFa weights, PACT activations), DoReFa (weights
     *     and activations clipped to {@code [0, 1]}), LSQ (learned steps for both), fixed range (round-to-nearest weights
     *     at their maximum, activations clipped at their initial maximum); {@code metric} is test
     *     accuracy, {@code cost} the bits
     */
    public static List<CompactRow> quantizationAware(int[] bitsList, int steps, long seed) {
        CompressionBench.Split data = CompressionBench.digits(seed);
        List<CompactRow> rows = new ArrayList<>();
        SimpleMlp ref = CompressionBench.trained(seed);
        rows.add(new CompactRow("fp32", CompressionBench.accuracy(ref::forward, data.test()), 32, 0, -1));
        Tensor[] acts = ref.activations(data.train().inputs());
        for (int bits : bitsList) {
            for (String method : List.of("pact", "dorefa", "lsq", "fixed range")) {
                SimpleMlp net = CompressionBench.trained(seed);
                int depth = net.depth();
                float[] range = new float[depth];
                float[] actMax = new float[depth];
                float[] actMean = new float[depth];
                for (int l = 0; l < depth; l++) {
                    range[l] = net.weight(l).value().abs().max().item();
                    actMax[l] = acts[l + 1].max().item();
                    actMean[l] = acts[l + 1].mean().item();
                }
                List<Var> trainable = new ArrayList<>(net.parameters());
                List<PactActivation> pact = new ArrayList<>();
                List<Var> wStep = new ArrayList<>();
                List<Var> aStep = new ArrayList<>();
                int qp = (1 << (bits - 1)) - 1;
                for (int l = 0; l < depth; l++) {
                    PactActivation a = new PactActivation(bits, actMax[l]);
                    pact.add(a);
                    float meanAbs = net.weight(l).value().abs().mean().item();
                    wStep.add(Var.param(Tensor.full(io.github.zzuegg.brainiac.tensor.Shape.of(1),
                            Math.max(1e-4f, 2 * meanAbs / (float) Math.sqrt(qp)))));
                    aStep.add(Var.param(Tensor.full(io.github.zzuegg.brainiac.tensor.Shape.of(1),
                            Math.max(1e-4f, 2 * actMean[l] / (float) Math.sqrt((1 << bits) - 1)))));
                }
                switch (method) {
                    case "pact" -> pact.forEach(a -> trainable.addAll(a.parameters()));
                    case "lsq" -> {
                        trainable.addAll(wStep);
                        trainable.addAll(aStep);
                    }
                    default -> {
                    }
                }
                java.util.function.BiFunction<Integer, Var, Var> weight = switch (method) {
                    case "pact", "dorefa" -> (l, w) -> DoReFaQuantizer.weight(w, bits).mul(range[l]);
                    case "lsq" -> (l, w) -> LsqDense.quantize(w, wStep.get(l), bits);
                    default -> (l, w) -> symmetric(w, bits);
                };
                java.util.function.BiFunction<Integer, Var, Var> act = switch (method) {
                    case "pact" -> (l, z) -> pact.get(l).forward(z);
                    case "dorefa" -> (l, z) -> DoReFaQuantizer.activation(z, bits);
                    // unsigned b bits = the positive half of a symmetric (b + 1)-bit grid
                    case "lsq" -> (l, z) -> LsqDense.quantize(z.relu(), aStep.get(l), bits + 1);
                    default -> (l, z) -> DoReFaQuantizer.activation(z.relu().div(actMax[l]), bits).mul(actMax[l]);
                };
                UnaryOperator<Var> fwd = CompressionBench.forward(net, weight, act);
                CompressionBench.train(fwd, trainable, Optimizer.adam(0.002f), data.train(), steps);
                rows.add(new CompactRow(method + " " + bits + "b", CompressionBench.accuracy(fwd, data.test()), bits, 0,
                        -1));
            }
        }
        return rows;
    }

    /** Symmetric per-tensor round-to-nearest at the weight's maximum, straight through. */
    private static Var symmetric(Var w, int bits) {
        int qp = Math.max(1, (1 << (bits - 1)) - 1);
        float max = Math.max(1e-8f, w.value().abs().max().item());
        Tensor q = w.value().mul(qp / max).round().mul(max / qp);
        return Functional.straightThrough(Var.of(q), w);
    }

    // ------------------------------------------------------------------ 8.5.13

    /**
     * ToMe token merging against class-attention token pruning on a small ViT (8x8 shapes,
     * 2x2 patches, 17 tokens with the class token, 3 blocks), trained on fresh batches of 32.
     *
     * @param perLayer the tokens removed per block to try
     * @param steps training steps of the dense ViT
     * @param fineTune fine-tuning steps of the reduced ViTs at the largest {@code r}
     * @param seed the seed
     * @return dense, then merge and prune at each {@code r} (training-free), then both after
     *     fine-tuning; {@code metric} is test accuracy, {@code cost} multiply-adds per image,
     *     {@code millis} the forward time of the test set
     */
    public static List<CompactRow> tokenMerging(int[] perLayer, int steps, int fineTune, long seed) {
        ShapeImages images = new ShapeImages(8, 0.1f);
        Tensor[] test = images.generate(128, seed + 1);
        TokenMerging vit = new TokenMerging(1, 8, 2, 32, 3, 4, ShapeImages.CLASSES, seed);
        trainVit(vit, images, steps, 0.003f, seed);
        List<CompactRow> rows = new ArrayList<>();
        rows.add(vitRow("dense", vit, test));
        int last = perLayer[perLayer.length - 1];
        for (int r : perLayer) {
            rows.add(vitRow("tome r=" + r, vit.reduce(TokenMerging.Mode.MERGE, r), test));
            rows.add(vitRow("prune r=" + r, vit.reduce(TokenMerging.Mode.PRUNE, r), test));
        }
        for (TokenMerging.Mode m : List.of(TokenMerging.Mode.MERGE, TokenMerging.Mode.PRUNE)) {
            TokenMerging tuned = vit.copy().reduce(m, last);
            trainVit(tuned, images, fineTune, 0.001f, seed + 7);
            rows.add(vitRow((m == TokenMerging.Mode.MERGE ? "tome" : "prune") + " r=" + last + " +ft", tuned, test));
        }
        vit.reduce(TokenMerging.Mode.NONE, 0);
        return rows;
    }

    private static void trainVit(TokenMerging vit, ShapeImages images, int steps, float lr, long seed) {
        Optimizer opt = Optimizer.adam(lr);
        List<Var> params = vit.parameters();
        for (int s = 0; s < steps; s++) {
            Tensor[] data = images.generate(32, seed * 7919 + s + 11);
            Functional.crossEntropy(vit.forward(Var.of(data[0])), data[1]).backward();
            opt.step(params);
        }
    }

    private static CompactRow vitRow(String name, TokenMerging vit, Tensor[] test) {
        double ms = KernelTimer.millis(() -> Var.noGrad(() -> vit.forward(Var.of(test[0])).value().toArray()), 2, 5);
        float[] l = Var.noGrad(() -> vit.forward(Var.of(test[0])).value()).toArray();
        float[] ids = test[1].toArray();
        int k = ShapeImages.CLASSES;
        int correct = 0;
        for (int i = 0; i < ids.length; i++) {
            int best = 0;
            for (int c = 1; c < k; c++) {
                if (l[i * k + c] > l[i * k + best]) {
                    best = c;
                }
            }
            correct += best == (int) ids[i] ? 1 : 0;
        }
        return new CompactRow(name, correct / (double) ids.length, vit.macs(), 0, ms);
    }

    // ------------------------------------------------------------------ 8.5.12

    /**
     * Decomposes the {@code 256 x 256} middle layer of a {@code 64-256-256-10} digits MLP.
     *
     * @param ttRank the largest tensor-train rank (cores over {@code 4 x 4 x 4 x 4} modes)
     * @param steps training steps of the dense network
     * @param fineTune fine-tuning steps after decomposing
     * @param seed the seed
     * @return dense; SVD at the tensor train's parameter count (before and after fine-tuning);
     *     the tensor train (before and after); an SVD-shaped bottleneck trained from scratch;
     *     {@code metric} is test accuracy, {@code size} the middle layer's weights,
     *     {@code millis} its forward time on the training batch
     */
    public static List<CompactRow> denseDecompositions(int ttRank, int steps, int fineTune, long seed) {
        CompressionBench.Split data = CompressionBench.digits(seed);
        SimpleMlp net = SimpleMlp.relu(seed, 64, 256, 256, 10);
        CompressionBench.train(net::forward, net.parameters(), Optimizer.adam(0.005f), data.train(), steps);
        Dense middle = new Dense(256, 256, Activation.RELU, net.weight(1).value(), net.bias(1).value().reshape(256));
        int[] f = {4, 4, 4, 4};
        TensorTrainDense tt = TensorTrainDense.from(middle, f, f, ttRank);
        int svdRank = Math.max(1, (int) Math.round(tt.weightCount() / 512.0));
        LowRankDense svd = LowRankDense.factorize(middle, svdRank);
        Tensor batch = data.train().inputs();
        Tensor hidden = Var.noGrad(() -> net.layer(0, Var.of(batch)).value());
        List<CompactRow> rows = new ArrayList<>();
        rows.add(mlpRow("dense", net, middle, 65536, hidden, data));
        rows.add(mlpRow("svd r=" + svdRank, net, svd, 512L * svdRank, hidden, data));
        fineTuneMiddle(net, svd, data, fineTune);
        rows.add(mlpRow("svd r=" + svdRank + " +ft", net, svd, 512L * svdRank, hidden, data));
        SimpleMlp net2 = SimpleMlp.relu(seed, 64, 256, 256, 10);
        for (int i = 0; i < net2.parameters().size(); i++) {
            net2.parameters().get(i).assign(net.parameters().get(i).value());
        }
        rows.add(mlpRow("tt r=" + ttRank, net2, tt, tt.weightCount(), hidden, data));
        fineTuneMiddle(net2, tt, data, fineTune);
        rows.add(mlpRow("tt r=" + ttRank + " +ft", net2, tt, tt.weightCount(), hidden, data));
        SimpleMlp scratch = SimpleMlp.relu(seed + 5, 64, 256, 256, 10);
        java.util.SplittableRandom r = new java.util.SplittableRandom(seed + 6);
        LowRankDense small = new LowRankDense(Tensor.normal(Shape.of(256, svdRank), 0, 0.25f, r),
                Tensor.normal(Shape.of(svdRank, 256), 0, 0.25f, r), Tensor.zeros(Shape.of(256)), Activation.RELU);
        fineTuneMiddle(scratch, small, data, steps + fineTune);
        rows.add(mlpRow("bottleneck r=" + svdRank + " from scratch", scratch, small, 512L * svdRank, hidden, data));
        return rows;
    }

    private static UnaryOperator<Var> withMiddle(SimpleMlp net, Layer middle) {
        return x -> net.layer(2, middle.forward(net.layer(0, x)));
    }

    private static void fineTuneMiddle(SimpleMlp net, Layer middle, CompressionBench.Split data, int steps) {
        List<Var> params = new ArrayList<>(List.of(net.weight(0), net.bias(0), net.weight(2), net.bias(2)));
        params.addAll(middle.parameters());
        CompressionBench.train(withMiddle(net, middle), params, Optimizer.adam(0.003f), data.train(), steps);
    }

    private static CompactRow mlpRow(String name, SimpleMlp net, Layer middle, long size, Tensor hidden,
            CompressionBench.Split data) {
        double ms = KernelTimer.millis(() -> Var.noGrad(() -> middle.forward(Var.of(hidden)).value().toArray()), 3, 9);
        return new CompactRow(name, CompressionBench.accuracy(withMiddle(net, middle), data.test()), 0, size, ms);
    }

    /**
     * Decomposes the second convolution ({@code 32 x 16 x 3 x 3}, stride 2) of the shapes CNN at
     * roughly equal parameters.
     *
     * @param tuckerRanks {@code {r_out, r_in}}
     * @param steps training steps of the CNN
     * @param fineTune fine-tuning steps after decomposing
     * @param seed the seed
     * @return dense; then SVD of the {@code [f, c k k]} unfolding, CP and Tucker-2, each before
     *     and after fine-tuning; {@code size} the layer's weights, {@code millis} the forward time
     *     of the test set
     */
    public static List<CompactRow> convDecompositions(int[] tuckerRanks, int steps, int fineTune, long seed) {
        Tensor[] train = ShapesCnn.data(192, seed);
        Tensor[] test = ShapesCnn.data(192, seed + 1);
        ShapesCnn cnn = ShapesCnn.create(16, 32, seed);
        cnn.train(train, Optimizer.adam(0.01f), steps, 0);
        Tensor w = cnn.w2().value();
        List<CompactRow> rows = new ArrayList<>();
        rows.add(cnnRow("dense", cnn, ShapesCnn.CORE, w.shape().size(), test));
        TensorDecomposition.Tucker2 tk = TensorDecomposition.tucker2(w, tuckerRanks[0], tuckerRanks[1]);
        long budget = tk.parameters();
        int svdRank = Math.max(1, (int) Math.round(budget / (32.0 + 144.0)));
        int cpRank = Math.max(1, (int) Math.round(budget / (32.0 + 16 + 3 + 3)));
        for (String method : List.of("svd r=" + svdRank, "cp r=" + cpRank,
                "tucker2 " + tuckerRanks[0] + "x" + tuckerRanks[1])) {
            ShapesCnn net = cnn.copy();
            List<Var> factors = new ArrayList<>();
            ShapesCnn.Convolution conv;
            long size;
            if (method.startsWith("svd")) {
                double[][] m = new double[32][144];
                double[] wv = w.toDoubleArray();
                for (int i = 0; i < 32; i++) {
                    System.arraycopy(wv, i * 144, m[i], 0, 144);
                }
                ThinSvd s = ThinSvd.of(m);
                Var first = Var.param(s.right(svdRank, 0.5).reshape(svdRank, 16, 3, 3));
                Var second = Var.param(s.left(svdRank, 0.5).reshape(32, svdRank, 1, 1));
                factors.addAll(List.of(first, second));
                conv = (x, ignored, stride) -> conv(conv(x, first, stride, 1, 1, 1), second, 1, 0, 0, 1);
                size = 176L * svdRank;
            } else if (method.startsWith("cp")) {
                TensorDecomposition.Cp cp = TensorDecomposition.cp(w.toDoubleArray(), new int[] {32, 16, 3, 3}, cpRank,
                        40, seed);
                Var in = Var.param(fromFactor(cp.factors().get(1), true).reshape(cpRank, 16, 1, 1));
                Var vert = Var.param(fromFactor(cp.factors().get(2), true).reshape(cpRank, 1, 3, 1));
                Var hor = Var.param(fromFactor(cp.factors().get(3), true).reshape(cpRank, 1, 1, 3));
                Var out = Var.param(fromFactor(cp.factors().get(0), false).reshape(32, cpRank, 1, 1));
                factors.addAll(List.of(in, vert, hor, out));
                int rank = cpRank;
                conv = (x, ignored, stride) -> {
                    Var a = Functional.conv(x, in, null, new int[] {1, 1}, new int[] {0, 0}, new int[] {0, 0},
                            new int[] {1, 1}, 1);
                    Var b = Functional.conv(a, vert, null, new int[] {stride, 1}, new int[] {1, 0}, new int[] {1, 0},
                            new int[] {1, 1}, rank);
                    Var c = Functional.conv(b, hor, null, new int[] {1, stride}, new int[] {0, 1}, new int[] {0, 1},
                            new int[] {1, 1}, rank);
                    return Functional.conv(c, out, null, new int[] {1, 1}, new int[] {0, 0}, new int[] {0, 0},
                            new int[] {1, 1}, 1);
                };
                size = cp.parameters();
            } else {
                Var in = Var.param(tk.in().transpose().reshape(tuckerRanks[1], 16, 1, 1));
                Var core = Var.param(tk.core());
                Var out = Var.param(tk.out().reshape(32, tuckerRanks[0], 1, 1));
                factors.addAll(List.of(in, core, out));
                conv = (x, ignored, stride) -> conv(conv(conv(x, in, 1, 0, 0, 1), core, stride, 1, 1, 1), out, 1, 0,
                        0, 1);
                size = tk.parameters();
            }
            rows.add(cnnRow(method, net, conv, size, test));
            net.train(train, Optimizer.adam(0.005f), fineTune, conv, factors);
            rows.add(cnnRow(method + " +ft", net, conv, size, test));
        }
        return rows;
    }

    private static Var conv(Var x, Var w, int stride, int padBefore, int padAfter, int groups) {
        return Functional.conv(x, w, null, new int[] {stride, stride}, new int[] {padBefore, padBefore},
                new int[] {padAfter, padAfter}, new int[] {1, 1}, groups);
    }

    /** A CP factor {@code [dim][R]} as {@code [R, dim]} (transposed) or {@code [dim, R]}. */
    private static Tensor fromFactor(double[][] f, boolean transpose) {
        int d = f.length;
        int r = f[0].length;
        float[] v = new float[d * r];
        for (int i = 0; i < d; i++) {
            for (int k = 0; k < r; k++) {
                v[transpose ? k * d + i : i * r + k] = (float) f[i][k];
            }
        }
        return transpose ? Tensor.of(Shape.of(r, d), v) : Tensor.of(Shape.of(d, r), v);
    }

    private static CompactRow cnnRow(String name, ShapesCnn net, ShapesCnn.Convolution conv2, long size,
            Tensor[] test) {
        double ms = KernelTimer.millis(() -> Var.noGrad(() -> net.forward(Var.of(test[0]), ShapesCnn.CORE, conv2)
                .value().toArray()), 2, 5);
        return new CompactRow(name, net.accuracy(test, ShapesCnn.CORE, conv2), 0, size, ms);
    }

    private static SimpleMlp prunedReference(long seed, double sparsity, int fineTune, CompressionBench.Split data) {
        SimpleMlp net = CompressionBench.trained(seed);
        List<Var> w = CompressionBench.weights(net);
        for (Var v : w) {
            v.assign(WeightPruning.magnitude(v.value(), sparsity));
        }
        CompressionBench.train(net::forward, net.parameters(), FixedSparsityOptimizer.ofZeros(Optimizer.adam(0.003f), w),
                data.train(), fineTune);
        return net;
    }

    /** Stored entries of a sparse matrix with fixed-width gaps, fillers included. */
    private static long countEntries(Tensor w, int gapBits) {
        float[] v = w.toArray();
        int maxGap = (1 << gapBits) - 1;
        long n = 0;
        int last = -1;
        for (int i = 0; i < v.length; i++) {
            if (v[i] != 0) {
                int gap = i - last - 1;
                n += gap / (maxGap + 1) + 1;
                last = i;
            }
        }
        return n;
    }

    /**
     * Leave-one-out nearest-neighbour recall@1 by cosine similarity.
     *
     * @param embeddings {@code [n, d]}
     * @param labels the classes
     * @return the share of rows whose nearest other row has the same class
     */
    public static double recallAt1(Tensor embeddings, int[] labels) {
        int n = labels.length;
        int d = embeddings.shape().dim(1);
        float[] e = embeddings.toArray();
        double[] norm = new double[n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < d; k++) {
                norm[i] += e[i * d + k] * e[i * d + k];
            }
            norm[i] = Math.sqrt(norm[i]) + 1e-12;
        }
        int hit = 0;
        for (int i = 0; i < n; i++) {
            int best = -1;
            double bestSim = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                double s = 0;
                for (int k = 0; k < d; k++) {
                    s += e[i * d + k] * e[j * d + k];
                }
                s /= norm[i] * norm[j];
                if (s > bestSim) {
                    bestSim = s;
                    best = j;
                }
            }
            hit += labels[best] == labels[i] ? 1 : 0;
        }
        return hit / (double) n;
    }

    static Tensor sample(SplittableRandom r, int n, int k) {
        float[] idx = new float[k];
        for (int i = 0; i < k; i++) {
            idx[i] = r.nextInt(n);
        }
        return Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(k), idx);
    }

    static long count(SimpleMlp net) {
        long n = 0;
        for (Var p : net.parameters()) {
            n += p.shape().size();
        }
        return n;
    }
}
