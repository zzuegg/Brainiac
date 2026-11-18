package io.github.zzuegg.brainiac.experiments.compression;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.compress.DecoupledKdLoss;
import io.github.zzuegg.brainiac.experimental.compress.FixedSparsityOptimizer;
import io.github.zzuegg.brainiac.experimental.compress.WeightSharingQuantizer;
import io.github.zzuegg.brainiac.experimental.compress.RelationalKdLoss;
import io.github.zzuegg.brainiac.experimental.nn.DistillationLoss;
import io.github.zzuegg.brainiac.experimental.nn.FeatureDistillationLoss;
import io.github.zzuegg.brainiac.experimental.nn.Gptq;
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
 *     and pruning alone, by accuracy against total stored bits.</li>
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
