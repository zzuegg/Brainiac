package io.github.zzuegg.brainiac.experimental.security;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.LayerHooks;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Repairing a backdoored classifier with a small clean set and no access to the poisoned
 * data. The model is a core {@link Model} whose hidden steps are {@link Dense} layers; every
 * method changes it in place.
 *
 * <ul>
 *   <li>{@link #neuralCleanse}: for each class, the smallest mask {@code m} and pattern
 *       {@code p} that send clean inputs {@code (1 - m) x + m p} to it, by Adam on
 *       {@code CE + λ ||m||₁}; the class whose mask norm is an outlier below the median
 *       (MAD anomaly index above 2) is the target, and {@link #unlearn} fine-tunes on clean
 *       data stamped with the recovered trigger under the true labels. As in the paper, the
 *       mask weight {@code λ} adapts: raised by 1.5 when the trigger works on 95% of a batch
 *       for most of the last ten steps, lowered otherwise, and the smallest working mask is
 *       kept (Wang et al., IEEE S&amp;P
 *       2019, <a href="https://doi.org/10.1109/SP.2019.00031">doi:10.1109/SP.2019.00031</a>).</li>
 *   <li>{@link #finePrune}: prune the hidden neurons least active on clean data, then
 *       fine-tune (Liu, Dolan-Gavitt and Garg, RAID 2018,
 *       <a href="https://arxiv.org/abs/1805.12185">arXiv:1805.12185</a>).</li>
 *   <li>{@link #anp}: adversarial neuron pruning learns a neuron mask {@code m ∈ [0,1]} under
 *       the worst multiplicative neuron perturbation {@code h (m + δ)},
 *       {@code |δ| <= ε}, minimising {@code α L(m) + (1-α) max_δ L(m + δ)}, and prunes the
 *       neurons whose mask falls below a threshold: backdoor neurons are the most sensitive
 *       to perturbation (the mask is optimised with Adam rather than the paper's SGD, to
 *       converge in a few hundred steps) (Wu and Wang, NeurIPS 2021, <a href="https://arxiv.org/abs/2110.14430">arXiv:2110.14430</a>).</li>
 *   <li>{@link #fineTune}: plain fine-tuning on the clean set, the baseline.</li>
 * </ul>
 */
public final class BackdoorRepair {

    /**
     * Neural Cleanse's reverse-engineered triggers.
     *
     * @param maskNorms the {@code ℓ1} norm of each class's mask
     * @param anomaly each class's MAD anomaly index
     * @param target the flagged class, or -1
     * @param masks each class's mask
     * @param patterns each class's pattern
     */
    public record Cleanse(double[] maskNorms, double[] anomaly, int target, float[][] masks, float[][] patterns) {
    }

    private BackdoorRepair() {
    }

    /**
     * Supervised training with Adam on minibatches.
     *
     * @param model the model
     * @param x inputs
     * @param y labels
     * @param steps steps
     * @param lr the step size
     * @param seed the seed
     */
    public static void fineTune(Model model, float[][] x, int[] y, int steps, float lr, long seed) {
        train(model, x, y, steps, lr, seed, null, null);
    }

    static void train(Model model, float[][] x, int[] y, int steps, float lr, long seed, float[] mask,
            float[] pattern) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int b = Math.min(32, x.length);
        for (int s = 0; s < steps; s++) {
            float[][] bx = new float[b][];
            float[] by = new float[b];
            for (int i = 0; i < b; i++) {
                int j = r.nextInt(x.length);
                bx[i] = x[j];
                by[i] = y[j];
                if (mask != null && r.nextDouble() < 0.2) {
                    float[] t = new float[x[j].length];
                    for (int k = 0; k < t.length; k++) {
                        t[k] = (1 - mask[k]) * x[j][k] + mask[k] * pattern[k];
                    }
                    bx[i] = t;
                }
            }
            Functional.crossEntropy(model.forward(Var.of(Tensor.of(bx)), true), Tensor.of(Shape.of(b), by))
                    .backward();
            opt.step(model.parameters());
        }
    }

    /**
     * Reverse-engineers a trigger for every class.
     *
     * @param model the suspect model
     * @param x clean inputs
     * @param classes the number of classes
     * @param steps optimisation steps per class
     * @param lambda the initial mask {@code ℓ1} weight
     * @param seed the seed
     * @return the triggers and the outlier test
     */
    public static Cleanse neuralCleanse(Model model, float[][] x, int classes, int steps, float lambda, long seed) {
        int d = x[0].length;
        double[] norms = new double[classes];
        float[][] masks = new float[classes][];
        float[][] patterns = new float[classes][];
        SplittableRandom r = new SplittableRandom(seed);
        int b = Math.min(64, x.length);
        for (int t = 0; t < classes; t++) {
            Var mRaw = Var.param(Tensor.full(Shape.of(1, d), -2f));
            Var pRaw = Var.param(Tensor.zeros(Shape.of(1, d)));
            Optimizer opt = Optimizer.adam(0.1f);
            float[] target = new float[b];
            Arrays.fill(target, t);
            Tensor ids = Tensor.of(Shape.of(b), target);
            float lam = lambda;
            double bestNorm = Double.POSITIVE_INFINITY;
            int hits = 0;
            for (int s = 0; s < steps; s++) {
                float[][] bx = new float[b][];
                for (int i = 0; i < b; i++) {
                    bx[i] = x[r.nextInt(x.length)];
                }
                Var xb = Var.of(Tensor.of(bx));
                Var m = mRaw.sigmoid();
                Var p = pRaw.sigmoid().mul(1.5f);
                Var stamped = xb.mul(m.neg().add(1f)).add(m.mul(p));
                Var out = model.forward(stamped, false);
                Var loss = Functional.crossEntropy(out, ids).add(m.sum().mul(lam));
                List<Var> g = Var.gradients(loss, List.of(mRaw, pRaw), false);
                mRaw.setGrad(g.get(0).value());
                pRaw.setGrad(g.get(1).value());
                opt.step(List.of(mRaw, pRaw));
                hits += out.value().argmax(1).equalTo(ids).mean().item() >= 0.95f ? 1 : 0;
                if ((s + 1) % 10 == 0) {
                    float[] mv = mRaw.value().sigmoid().toArray();
                    double norm = 0;
                    for (float v : mv) {
                        norm += v;
                    }
                    if (hits >= 8 && norm < bestNorm) {
                        bestNorm = norm;
                        masks[t] = mv;
                        patterns[t] = pRaw.value().sigmoid().mul(1.5f).toArray();
                    }
                    lam = hits >= 8 ? lam * 1.5f : lam / 1.5f;
                    hits = 0;
                }
            }
            if (masks[t] == null) {
                masks[t] = mRaw.value().sigmoid().toArray();
                patterns[t] = pRaw.value().sigmoid().mul(1.5f).toArray();
            }
            double n = 0;
            for (float v : masks[t]) {
                n += v;
            }
            norms[t] = n;
        }
        double med = median(norms);
        double[] dev = new double[classes];
        for (int c = 0; c < classes; c++) {
            dev[c] = Math.abs(norms[c] - med);
        }
        double mad = Math.max(1.4826 * median(dev), 1e-9);
        double[] anomaly = new double[classes];
        int flagged = -1;
        for (int c = 0; c < classes; c++) {
            anomaly[c] = Math.abs(norms[c] - med) / mad;
            if (norms[c] < med && anomaly[c] > 2 && (flagged < 0 || norms[c] < norms[flagged])) {
                flagged = c;
            }
        }
        return new Cleanse(norms, anomaly, flagged, masks, patterns);
    }

    /**
     * Neural Cleanse's unlearning: fine-tuning on clean data where a fifth of every batch
     * carries the recovered trigger with the true label (plain fine-tuning when nothing was
     * flagged).
     *
     * @param model the model
     * @param cleanse the reverse-engineered triggers
     * @param x clean inputs
     * @param y their labels
     * @param steps steps
     * @param seed the seed
     */
    public static void unlearn(Model model, Cleanse cleanse, float[][] x, int[] y, int steps, long seed) {
        int t = cleanse.target();
        train(model, x, y, steps, 1e-3f, seed, t < 0 ? null : cleanse.masks()[t], t < 0 ? null
                : cleanse.patterns()[t]);
    }

    /**
     * Fine-pruning.
     *
     * @param model the model
     * @param layer the hidden step whose neurons are pruned
     * @param x clean inputs
     * @param y their labels
     * @param fraction the share of neurons pruned, least active first
     * @param steps fine-tuning steps
     * @param seed the seed
     * @return the pruned neurons
     */
    public static boolean[] finePrune(Model model, int layer, float[][] x, int[] y, double fraction, int steps,
            long seed) {
        Tensor h = LayerHooks.activations(model, Tensor.of(x)).get(layer);
        int w = h.shape().dim(1);
        float[] v = h.toArray();
        double[] mean = new double[w];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < w; j++) {
                mean[j] += v[i * w + j] / x.length;
            }
        }
        Integer[] order = new Integer[w];
        for (int j = 0; j < w; j++) {
            order[j] = j;
        }
        Arrays.sort(order, (a, b) -> Double.compare(mean[a], mean[b]));
        boolean[] pruned = new boolean[w];
        for (int j = 0; j < (int) Math.round(fraction * w); j++) {
            pruned[order[j]] = true;
        }
        prune(model, layer, pruned);
        train(model, x, y, steps, 1e-3f, seed, null, null);
        prune(model, layer, pruned);
        return pruned;
    }

    /**
     * Adversarial neuron pruning.
     *
     * @param model the model
     * @param layer the hidden step whose neurons are masked
     * @param x clean inputs
     * @param y their labels
     * @param epsilon the perturbation radius, such as 0.4
     * @param alpha the clean-loss weight, such as 0.2
     * @param steps mask optimisation steps
     * @param threshold neurons with a mask below it are pruned, such as 0.2
     * @param seed the seed
     * @return the pruned neurons
     */
    public static boolean[] anp(Model model, int layer, float[][] x, int[] y, float epsilon, float alpha, int steps,
            double threshold, long seed) {
        int w = LayerHooks.activations(model, Tensor.of(new float[][] {x[0]})).get(layer).shape().dim(1);
        Var mask = Var.param(Tensor.ones(Shape.of(1, w)));
        Optimizer opt = Optimizer.adam(0.05f);
        SplittableRandom r = new SplittableRandom(seed);
        int b = Math.min(64, x.length);
        for (int s = 0; s < steps; s++) {
            float[][] bx = new float[b][];
            float[] by = new float[b];
            for (int i = 0; i < b; i++) {
                int j = r.nextInt(x.length);
                bx[i] = x[j];
                by[i] = y[j];
            }
            Var xb = Var.of(Tensor.of(bx));
            Tensor ids = Tensor.of(Shape.of(b), by);
            Var delta = Var.param(Tensor.uniform(Shape.of(1, w), -epsilon, epsilon, r));
            Var adv = Functional.crossEntropy(forward(model, layer, xb, Var.of(mask.value()).add(delta)), ids);
            Tensor gd = Var.gradients(adv, List.of(delta), false).get(0).value();
            float[] dv = delta.value().add(gd.sign().mul(epsilon)).toArray();
            for (int j = 0; j < w; j++) {
                dv[j] = Math.clamp(dv[j], -epsilon, epsilon);
            }
            Var worst = Var.of(Tensor.of(Shape.of(1, w), dv));
            Var loss = Functional.crossEntropy(forward(model, layer, xb, mask), ids).mul(alpha)
                    .add(Functional.crossEntropy(forward(model, layer, xb, mask.add(worst)), ids).mul(1 - alpha));
            mask.setGrad(Var.gradients(loss, List.of(mask), false).get(0).value());
            opt.step(List.of(mask));
            float[] mv = mask.value().toArray();
            for (int j = 0; j < w; j++) {
                mv[j] = Math.clamp(mv[j], 0f, 1f);
            }
            mask.assign(Tensor.of(Shape.of(1, w), mv));
        }
        float[] m = mask.value().toArray();
        boolean[] pruned = new boolean[w];
        for (int j = 0; j < w; j++) {
            pruned[j] = m[j] < threshold;
        }
        prune(model, layer, pruned);
        return pruned;
    }

    private static Var forward(Model model, int layer, Var x, Var neuronScale) {
        return model.forward(x, false, (step, out) -> step.index() == layer ? out.mul(neuronScale) : out);
    }

    /**
     * Removes hidden neurons for good by zeroing their incoming weights and biases.
     *
     * @param model the model
     * @param layer the hidden step, a {@link Dense} layer with an activation mapping 0 to 0
     * @param pruned the neurons removed
     */
    public static void prune(Model model, int layer, boolean[] pruned) {
        Dense dense = (Dense) model.layers().get(layer);
        Tensor w = dense.weight().value();
        int in = w.shape().dim(0);
        int out = w.shape().dim(1);
        float[] wv = w.toArray();
        float[] bv = dense.bias().value().toArray();
        for (int j = 0; j < out; j++) {
            if (pruned[j]) {
                bv[j] = 0;
                for (int i = 0; i < in; i++) {
                    wv[i * out + j] = 0;
                }
            }
        }
        dense.weight().assign(Tensor.of(w.shape(), wv));
        dense.bias().assign(Tensor.of(dense.bias().shape(), bv));
    }

    private static double median(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        int n = s.length;
        return n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    }
}
