package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Testing with concept activation vectors (TCAV): a concept, given only as example inputs
 * (images that show a marker, say), becomes a direction in a layer's activation space, the
 * normal of a {@link LinearProbe} that separates the concept's activations from those of
 * random inputs. The TCAV score of a class is the share of that class's inputs whose class
 * score increases along the direction ({@code d y / d a . v > 0}): near 1 when the model
 * relies on the concept for the class, near the scores of random directions when it does
 * not. Directions from several random splits give that null distribution
 * ({@link #randomCavs}).
 *
 * <p>Paper: Kim et al., "Interpretability Beyond Feature Attribution: Quantitative Testing
 * with Concept Activation Vectors (TCAV)" (2018), <a href="https://arxiv.org/abs/1711.11279">arXiv:1711.11279</a>.
 */
public final class ConceptActivationVectors {

    private ConceptActivationVectors() {
    }

    /**
     * The flattened activations of one layer.
     *
     * @param model a sequential model
     * @param layer the layer index
     * @param inputs a batch
     * @return one row per input
     */
    public static float[][] activations(Model model, int layer, Tensor inputs) {
        Tensor a = LayerHooks.activations(model, inputs).get(layer);
        int n = a.shape().dim(0);
        int d = (int) (a.shape().size() / n);
        float[] v = a.toArray();
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(v, i * d, out[i], 0, d);
        }
        return out;
    }

    /**
     * A concept activation vector: the unit normal of a linear probe separating the concept's
     * activations from the counterexamples', pointing toward the concept.
     *
     * @param model a sequential model
     * @param layer the layer index
     * @param concept inputs that show the concept
     * @param counterexamples inputs that do not (random inputs)
     * @return the unit direction, in the flattened activation space
     */
    public static float[] cav(Model model, int layer, Tensor concept, Tensor counterexamples) {
        float[][] pos = activations(model, layer, concept);
        float[][] neg = activations(model, layer, counterexamples);
        float[][] x = new float[pos.length + neg.length][];
        int[] y = new int[x.length];
        for (int i = 0; i < pos.length; i++) {
            x[i] = pos[i];
            y[i] = 1;
        }
        for (int i = 0; i < neg.length; i++) {
            x[pos.length + i] = neg[i];
        }
        LinearProbe probe = LinearProbe.fit(x, y, 2, 200, 0.05, 1e-3);
        // the probe's log-odds are linear, so their change along each axis is the normal
        int d = pos[0].length;
        float[] zero = new float[d];
        double base = logOdds(probe, zero);
        float[] v = new float[d];
        double norm = 0;
        for (int j = 0; j < d; j++) {
            zero[j] = 1;
            v[j] = (float) (logOdds(probe, zero) - base);
            zero[j] = 0;
            norm += v[j] * v[j];
        }
        float inv = (float) (1 / Math.sqrt(Math.max(norm, 1e-24)));
        for (int j = 0; j < d; j++) {
            v[j] *= inv;
        }
        return v;
    }

    private static double logOdds(LinearProbe probe, float[] x) {
        double[] p = probe.probabilities(x);
        return Math.log(Math.max(p[1], 1e-300)) - Math.log(Math.max(p[0], 1e-300));
    }

    private static float[] directional(Model model, int layer, float[] cav, Tensor inputs, int output) {
        Tensor act = LayerHooks.activations(model, inputs).get(layer);
        Var a = Var.param(act);
        Var y = LayerHooks.resume(model, layer, a).slice(1, output, output + 1).sum();
        float[] g = Var.gradients(y, List.of(a), false).get(0).value().toArray();
        int n = act.shape().dim(0);
        int d = cav.length;
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            double dot = 0;
            for (int j = 0; j < d; j++) {
                dot += g[i * d + j] * cav[j];
            }
            out[i] = (float) dot;
        }
        return out;
    }

    /**
     * The TCAV score: the share of inputs whose output increases along the concept direction.
     *
     * @param model a sequential model
     * @param layer the layer index the direction lives in
     * @param cav the direction
     * @param inputs inputs of the class being tested
     * @param output the output index (the class)
     * @return a share in {@code [0, 1]}
     */
    public static double score(Model model, int layer, float[] cav, Tensor inputs, int output) {
        float[] s = directional(model, layer, cav, inputs, output);
        int positive = 0;
        for (float v : s) {
            positive += v > 0 ? 1 : 0;
        }
        return (double) positive / s.length;
    }

    /**
     * The mean directional derivative of the output along the concept direction: the
     * magnitude the share in {@link #score} leaves out. Small networks are close to linear
     * in a layer's activations, so their scores sit near 0 or 1 for any direction, and
     * the sensitivity, compared with that of random directions, is the finer test.
     *
     * @param model a sequential model
     * @param layer the layer index the direction lives in
     * @param cav the direction
     * @param inputs inputs of the class being tested
     * @param output the output index (the class)
     * @return the mean of {@code d y / d a . v}
     */
    public static double sensitivity(Model model, int layer, float[] cav, Tensor inputs, int output) {
        float[] s = directional(model, layer, cav, inputs, output);
        double sum = 0;
        for (float v : s) {
            sum += v;
        }
        return sum / s.length;
    }

    /**
     * Directions of random concepts, the null distribution: each splits {@code pool} at
     * random into a pseudo-concept and counterexamples and fits a {@link #cav}.
     *
     * @param model a sequential model
     * @param layer the layer index
     * @param pool random inputs
     * @param runs how many random directions
     * @param seed seeds the splits
     * @return one direction per run
     */
    public static List<float[]> randomCavs(Model model, int layer, Tensor pool, int runs, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = pool.shape().dim(0);
        List<float[]> out = new ArrayList<>();
        for (int k = 0; k < runs; k++) {
            int[] order = r.ints(0, n).distinct().limit(n).toArray();
            Tensor[] a = new Tensor[n / 2];
            Tensor[] b = new Tensor[n - n / 2];
            for (int i = 0; i < n; i++) {
                Tensor row = pool.slice(0, order[i], order[i] + 1);
                if (i < n / 2) {
                    a[i] = row;
                } else {
                    b[i - n / 2] = row;
                }
            }
            out.add(cav(model, layer, Tensor.concat(0, a), Tensor.concat(0, b)));
        }
        return out;
    }
}
