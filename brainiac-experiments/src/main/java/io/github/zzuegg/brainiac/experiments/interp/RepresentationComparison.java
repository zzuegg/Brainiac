package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.interp.ModelStitcher;
import io.github.zzuegg.brainiac.experimental.interp.PermutationAlignment;
import io.github.zzuegg.brainiac.experimental.interp.RepresentationSimilarity;
import io.github.zzuegg.brainiac.experimental.nn.LinearProbe;
import io.github.zzuegg.brainiac.experimental.train.ModelMerging;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

/**
 * Wave 9 items 9.4.4 and 9.4.7: do networks trained from different seeds compute the same
 * thing, and can their weights be averaged?
 *
 * <p><b>Similarity</b> ({@link #similarity}): MLPs with four hidden layers on 8x8
 * {@link ShapeImages}, three seeds. For every pair of seeds and every similarity index
 * ({@link RepresentationSimilarity}: linear and RBF CKA, SVCCA, PWCCA; baselines: the raw
 * best-neuron correlation and the gap between per-layer {@link LinearProbe} accuracies) the
 * layer-correspondence test asks, for each layer of one network, whether the most similar
 * layer of the other is the same layer. Then {@link ModelStitcher} trains a linear stitch
 * from each layer of network A into network B and reports the accuracy penalty next to that
 * layer's CKA.
 *
 * <p><b>Connectivity</b> ({@link #connectivity}): one-hidden-layer-pair MLPs (64-w-w-6)
 * from two seeds per width. The test-loss barrier on the straight line between the weights,
 * before and after aligning B's hidden units to A's by weight matching and by activation
 * matching ({@link PermutationAlignment}); and the accuracy of the midpoint (the averaged
 * model) against naive averaging, {@link ModelMerging#slerp} and the output ensemble.
 */
public final class RepresentationComparison {

    private static final ShapeImages DATA = new ShapeImages(8, 0.15f);

    private RepresentationComparison() {
    }

    private static Model mlp(int[] widths, long seed) {
        Model.Builder b = Model.sequential().input(64);
        for (int w : widths) {
            b.dense(w, Activation.RELU);
        }
        return b.dense(ShapeImages.CLASSES, Activation.IDENTITY).seed(seed).build();
    }

    private static Model train(Model m, int steps, long seed) {
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            Tensor[] d = DATA.generate(64, seed * 7919 + s);
            Functional.crossEntropy(m.forward(Var.of(d[0].reshape(64, 64)), true), d[1]).backward();
            opt.step(m.parameters());
        }
        return m;
    }

    private static float[][] rows(Tensor t) {
        int n = t.shape().dim(0);
        int d = t.shape().dim(1);
        float[] f = t.toArray();
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(f, i * d, out[i], 0, d);
        }
        return out;
    }

    private static int[] labels(Tensor t) {
        float[] f = t.toArray();
        int[] y = new int[f.length];
        for (int i = 0; i < f.length; i++) {
            y[i] = (int) f[i];
        }
        return y;
    }

    private static double accuracy(Model m, Tensor x, Tensor y) {
        float[] p = Var.noGrad(() -> m.forward(Var.of(x))).value().argmax(1).toArray();
        float[] t = y.toArray();
        int hit = 0;
        for (int i = 0; i < t.length; i++) {
            hit += p[i] == t[i] ? 1 : 0;
        }
        return (double) hit / t.length;
    }

    private static double loss(Model m, Tensor x, Tensor y) {
        return Var.noGrad(() -> Functional.crossEntropy(m.forward(Var.of(x)), y)).value().item();
    }

    /**
     * Similarity results.
     *
     * @param correspondence per index the share of (pair, layer) whose best match is the same layer
     * @param stitchPenalty per layer the mean accuracy B loses when fed A's layer through a stitch
     * @param stitchCka per layer the mean linear CKA between A's and B's layer
     * @param accuracy the networks' mean test accuracy
     */
    public record Similarity(Map<String, Double> correspondence, double[] stitchPenalty, double[] stitchCka,
            double accuracy) {
    }

    /**
     * The similarity comparison.
     *
     * @param width the hidden width
     * @param seeds networks (at least 2)
     * @param steps training steps per network
     * @param seed the base seed
     * @return the results
     */
    public static Similarity similarity(int width, int seeds, int steps, long seed) {
        int layers = 4;
        int[] widths = {width, width, width, width};
        List<Model> nets = new ArrayList<>();
        for (int s = 0; s < seeds; s++) {
            nets.add(train(mlp(widths, seed + s), steps, seed + 31 * s));
        }
        Tensor[] probe = DATA.generate(300, seed + 1000);
        Tensor px = probe[0].reshape(300, 64);
        int[] py = labels(probe[1]);
        Tensor[] test = DATA.generate(500, seed + 2000);
        Tensor tx = test[0].reshape(500, 64);
        double acc = 0;
        List<float[][][]> acts = new ArrayList<>();
        List<double[]> probeAcc = new ArrayList<>();
        for (Model m : nets) {
            acc += accuracy(m, tx, test[1]) / seeds;
            float[][][] a = new float[layers][][];
            double[] pa = new double[layers];
            for (int l = 0; l < layers; l++) {
                a[l] = rows(ModelStitcher.activation(m, l, px));
                pa[l] = LinearProbe.fit(a[l], py, ShapeImages.CLASSES, 150, 0.05, 1e-4).accuracy(a[l], py);
            }
            acts.add(a);
            probeAcc.add(pa);
        }
        Map<String, BiFunction<float[][], float[][], Double>> indices = new LinkedHashMap<>();
        indices.put("linear CKA", RepresentationSimilarity::linearCka);
        indices.put("RBF CKA", (x, y) -> RepresentationSimilarity.rbfCka(x, y, 1.0));
        indices.put("SVCCA", (x, y) -> RepresentationSimilarity.svcca(x, y, 0.99));
        indices.put("PWCCA", (x, y) -> RepresentationSimilarity.pwcca(x, y, 0.99));
        indices.put("max neuron correlation", RepresentationSimilarity::maxCorrelation);
        Map<String, Double> corr = new LinkedHashMap<>();
        for (var e : indices.entrySet()) {
            double total = 0;
            int pairs = 0;
            for (int a = 0; a < seeds; a++) {
                for (int b = 0; b < seeds; b++) {
                    if (a == b) {
                        continue;
                    }
                    double[][] sim = new double[layers][layers];
                    for (int i = 0; i < layers; i++) {
                        for (int j = 0; j < layers; j++) {
                            sim[i][j] = e.getValue().apply(acts.get(a)[i], acts.get(b)[j]);
                        }
                    }
                    total += RepresentationSimilarity.correspondence(sim);
                    pairs++;
                }
            }
            corr.put(e.getKey(), total / pairs);
        }
        double probeTotal = 0;
        int probePairs = 0;
        for (int a = 0; a < seeds; a++) {
            for (int b = 0; b < seeds; b++) {
                if (a == b) {
                    continue;
                }
                double[][] sim = new double[layers][layers];
                for (int i = 0; i < layers; i++) {
                    for (int j = 0; j < layers; j++) {
                        sim[i][j] = -Math.abs(probeAcc.get(a)[i] - probeAcc.get(b)[j]);
                    }
                }
                probeTotal += RepresentationSimilarity.correspondence(sim);
                probePairs++;
            }
        }
        corr.put("probe accuracy gap", probeTotal / probePairs);
        double[] penalty = new double[layers];
        double[] cka = new double[layers];
        Tensor[] stitchData = DATA.generate(1000, seed + 3000);
        Tensor sx = stitchData[0].reshape(1000, 64);
        Model a = nets.get(0);
        Model b = nets.get(1);
        double accB = accuracy(b, tx, test[1]);
        for (int l = 0; l < layers; l++) {
            ModelStitcher stitch = new ModelStitcher(a, b, l, width, width, seed + l);
            stitch.train(sx, stitchData[1], 150, 64, seed + 5 + l);
            penalty[l] = accB - stitch.accuracy(tx, test[1]);
            cka[l] = RepresentationSimilarity.linearCka(acts.get(0)[l], acts.get(1)[l]);
        }
        return new Similarity(corr, penalty, cka, acc);
    }

    /**
     * Connectivity results for one width.
     *
     * @param width the hidden width
     * @param barrierNaive the test-loss barrier between A and B
     * @param barrierWeight after weight matching
     * @param barrierActivation after activation matching
     * @param accuracyA network A's accuracy
     * @param mergedNaive the accuracy of the plain average
     * @param mergedWeight the average after weight matching
     * @param mergedActivation the average after activation matching
     * @param slerp the accuracy of a SLERP merge (no alignment)
     * @param ensemble the accuracy of the output ensemble
     */
    public record Connectivity(int width, double barrierNaive, double barrierWeight, double barrierActivation,
            double accuracyA, double mergedNaive, double mergedWeight, double mergedActivation, double slerp,
            double ensemble) {
    }

    /**
     * The connectivity comparison at one width.
     *
     * @param width the hidden width
     * @param steps training steps per network
     * @param seed the seed
     * @return the results
     */
    public static Connectivity connectivity(int width, int steps, long seed) {
        int[] widths = {width, width};
        Model a = train(mlp(widths, seed), steps, seed + 1);
        Model b = train(mlp(widths, seed + 100), steps, seed + 2);
        Tensor[] test = DATA.generate(1000, seed + 5000);
        Tensor tx = test[0].reshape(1000, 64);
        Tensor[] match = DATA.generate(500, seed + 6000);
        List<Tensor> wa = PermutationAlignment.weights(a);
        List<Tensor> wb = PermutationAlignment.weights(b);
        List<Tensor> wbWeight = PermutationAlignment.permuted(b, PermutationAlignment.weightMatching(a, b, 20, seed));
        List<Tensor> wbAct = PermutationAlignment.permuted(b,
                PermutationAlignment.activationMatching(a, b, match[0].reshape(500, 64)));
        double bNaive = PermutationAlignment.barrier(a, wa, wb, 11, m -> loss(m, tx, test[1]));
        double bWeight = PermutationAlignment.barrier(a, wa, wbWeight, 11, m -> loss(m, tx, test[1]));
        double bAct = PermutationAlignment.barrier(a, wa, wbAct, 11, m -> loss(m, tx, test[1]));
        double mNaive = accuracy(PermutationAlignment.withWeights(a, PermutationAlignment.interpolate(wa, wb, 0.5)), tx,
                test[1]);
        double mWeight = accuracy(PermutationAlignment.withWeights(a,
                PermutationAlignment.interpolate(wa, wbWeight, 0.5)), tx, test[1]);
        double mAct = accuracy(PermutationAlignment.withWeights(a, PermutationAlignment.interpolate(wa, wbAct, 0.5)),
                tx, test[1]);
        double slerp = accuracy(PermutationAlignment.withWeights(a, ModelMerging.slerp(wa, wb, 0.5)), tx, test[1]);
        float[] ens = PermutationAlignment.ensembleLogits(a, b, tx).argmax(1).toArray();
        float[] y = test[1].toArray();
        int hit = 0;
        for (int i = 0; i < y.length; i++) {
            hit += ens[i] == y[i] ? 1 : 0;
        }
        return new Connectivity(width, bNaive, bWeight, bAct, accuracy(a, tx, test[1]), mNaive, mWeight, mAct, slerp,
                (double) hit / y.length);
    }
}
