package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Gradient inversion: a server that sees one client's gradient reconstructs the client's
 * training batch. The attacker optimises dummy inputs (and, for DLG, dummy soft labels) so
 * that their gradient with respect to the model's parameters matches the observed one; this
 * differentiates through a gradient, so it uses core nested autograd
 * ({@code Var.gradients(..., createGraph = true)}).
 *
 * <ul>
 *   <li>{@link #dlg}: deep leakage from gradients, squared L2 gradient distance, inputs and
 *       labels optimised jointly (Zhu, Liu and Han, NeurIPS 2019,
 *       <a href="https://arxiv.org/abs/1906.08935">arXiv:1906.08935</a>).</li>
 *   <li>{@link #cosineTv}: "inverting gradients", one minus the cosine similarity of the
 *       gradients plus a total-variation image prior, with labels read off the output bias
 *       gradient (whose entries are negative exactly at the batch's labels for softmax
 *       cross-entropy, the iDLG observation) (Geiping et al., NeurIPS 2020,
 *       <a href="https://arxiv.org/abs/2003.14053">arXiv:2003.14053</a>).</li>
 * </ul>
 *
 * <p>The model is a logits function over parameters; the observed gradient is that of the
 * batch-mean softmax cross-entropy, the usual client loss.
 */
public final class GradientInversionAttack {

    /**
     * A reconstruction.
     *
     * @param inputs the reconstructed inputs {@code [batch, features]}
     * @param labels the reconstructed labels
     * @param gradientDistance the final matching loss
     */
    public record Reconstruction(Tensor inputs, int[] labels, double gradientDistance) {
    }

    private final UnaryOperator<Var> logits;
    private final List<Var> parameters;
    private final int classes;
    private final boolean cosine;
    private float totalVariation;
    private int imageSide;
    private int iterations = 300;
    private float learningRate = 0.1f;

    private GradientInversionAttack(UnaryOperator<Var> logits, List<Var> parameters, int classes, boolean cosine) {
        this.logits = Objects.requireNonNull(logits);
        this.parameters = List.copyOf(parameters);
        this.classes = classes;
        this.cosine = cosine;
    }

    /**
     * DLG.
     *
     * @param logits the model
     * @param parameters the parameters whose gradient is observed
     * @param classes the number of classes
     * @return the attack
     */
    public static GradientInversionAttack dlg(UnaryOperator<Var> logits, List<Var> parameters, int classes) {
        return new GradientInversionAttack(logits, parameters, classes, false);
    }

    /**
     * Cosine matching with a total-variation prior.
     *
     * @param logits the model
     * @param parameters the parameters whose gradient is observed
     * @param classes the number of classes
     * @param imageSide the side of the square images (for the TV prior)
     * @param tv the TV weight
     * @return the attack
     */
    public static GradientInversionAttack cosineTv(UnaryOperator<Var> logits, List<Var> parameters, int classes,
            int imageSide, float tv) {
        GradientInversionAttack a = new GradientInversionAttack(logits, parameters, classes, true);
        a.imageSide = imageSide;
        a.totalVariation = tv;
        return a;
    }

    /**
     * @param n optimisation steps
     * @param lr Adam's step size
     * @return this
     */
    public GradientInversionAttack iterations(int n, float lr) {
        this.iterations = n;
        this.learningRate = lr;
        return this;
    }

    /**
     * The gradient a client would send: of the batch-mean cross-entropy.
     *
     * @param logits the model
     * @param parameters the parameters
     * @param x the batch
     * @param ids the labels
     * @return one gradient per parameter
     */
    public static Tensor[] clientGradient(UnaryOperator<Var> logits, List<Var> parameters, Tensor x, Tensor ids) {
        Var loss = io.github.zzuegg.brainiac.nn.Functional.crossEntropy(logits.apply(Var.of(x)), ids);
        List<Var> g = Var.gradients(loss, parameters, false);
        Tensor[] out = new Tensor[g.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = g.get(i).value();
        }
        return out;
    }

    /**
     * Labels from the output bias gradient: the {@code batch} most negative entries (with
     * repetition when a class's entry is much more negative than the others).
     *
     * @param biasGradient the observed gradient of the output layer's bias
     * @param batch the batch size
     * @return the inferred labels
     */
    public static int[] inferLabels(Tensor biasGradient, int batch) {
        float[] g = biasGradient.toArray().clone();
        int[] out = new int[batch];
        // each example of class c contributes about -(1 - p_c)/batch to entry c
        for (int i = 0; i < batch; i++) {
            int best = 0;
            for (int c = 1; c < g.length; c++) {
                best = g[c] < g[best] ? c : best;
            }
            out[i] = best;
            g[best] += 1f / batch;
        }
        return out;
    }

    /**
     * Reconstructs a batch from its gradient.
     *
     * @param observed the observed gradient, one per parameter
     * @param batch the batch size the attacker assumes
     * @param features the input width
     * @param outputBias the index of the output bias among the parameters (for label inference)
     * @param seed the seed of the dummy start
     * @return the reconstruction
     */
    public Reconstruction attack(Tensor[] observed, int batch, int features, int outputBias, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Var x = Var.param(Tensor.uniform(Shape.of(batch, features), 0, 1, r));
        Var labelLogits = Var.param(Tensor.normal(Shape.of(batch, classes), 0, 1, r));
        int[] fixedLabels = cosine ? inferLabels(observed[outputBias], batch) : null;
        Tensor fixedTargets = null;
        if (fixedLabels != null) {
            float[] t = new float[batch * classes];
            for (int i = 0; i < batch; i++) {
                t[i * classes + fixedLabels[i]] = 1;
            }
            fixedTargets = Tensor.of(Shape.of(batch, classes), t);
        }
        List<Var> optimised = cosine ? List.of(x) : List.of(x, labelLogits);
        Optimizer adam = Optimizer.adam(learningRate);
        double last = Double.NaN;
        for (int it = 0; it < iterations; it++) {
            Var out = logits.apply(x);
            Var logProbs = out.sub(out.logsumexp(1, true));
            Var targets = fixedTargets != null ? Var.of(fixedTargets)
                    : labelLogits.sub(labelLogits.logsumexp(1, true)).exp();
            Var loss = targets.mul(logProbs).sum().neg().div(batch);
            List<Var> g = Var.gradients(loss, parameters, true);
            Var match = cosine ? cosineDistance(g, observed) : l2Distance(g, observed);
            last = match.value().item();
            Var objective = match;
            if (cosine && totalVariation > 0) {
                objective = objective.add(tv(x, batch).mul(totalVariation));
            }
            for (Var v : optimised) {
                v.zeroGrad();
            }
            List<Var> grads = Var.gradients(objective, optimised, false);
            for (int i = 0; i < optimised.size(); i++) {
                optimised.get(i).setGrad(grads.get(i).value());
            }
            adam.step(optimised);
        }
        int[] labels = fixedLabels != null ? fixedLabels : labelLogits.value().argmax(1).toIntArray();
        return new Reconstruction(x.value(), labels, last);
    }

    private static Var l2Distance(List<Var> g, Tensor[] observed) {
        Var sum = null;
        for (int i = 0; i < observed.length; i++) {
            Var d = g.get(i).sub(observed[i]).square().sum();
            sum = sum == null ? d : sum.add(d);
        }
        return sum;
    }

    private static Var cosineDistance(List<Var> g, Tensor[] observed) {
        Var dot = null;
        Var norm = null;
        double observedNorm = 0;
        for (int i = 0; i < observed.length; i++) {
            Var dt = g.get(i).mul(observed[i]).sum();
            Var nt = g.get(i).square().sum();
            dot = dot == null ? dt : dot.add(dt);
            norm = norm == null ? nt : norm.add(nt);
            observedNorm += observed[i].square().sum().item();
        }
        return dot.div(norm.add(1e-12f).sqrt().mul((float) Math.sqrt(observedNorm) + 1e-12f)).neg().add(1f);
    }

    private Var tv(Var x, int batch) {
        int s = imageSide;
        Var img = x.reshape(batch, s, s);
        Var dy = img.slice(1, 1, s).sub(img.slice(1, 0, s - 1)).abs().mean();
        Var dx = img.slice(2, 1, s).sub(img.slice(2, 0, s - 1)).abs().mean();
        return dy.add(dx);
    }

    /**
     * The peak signal-to-noise ratio of each reconstruction against its best-matching
     * original (greedy one-to-one matching by squared error), peak 1.
     *
     * @param reconstructed {@code [batch, features]}
     * @param original {@code [batch, features]}
     * @return the mean PSNR in dB
     */
    public static double matchedPsnr(Tensor reconstructed, Tensor original) {
        int b = original.shape().dim(0);
        int d = Math.toIntExact(original.size() / b);
        float[] rec = reconstructed.toArray();
        float[] org = original.toArray();
        double[][] mse = new double[b][b];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < b; j++) {
                double s = 0;
                for (int k = 0; k < d; k++) {
                    double e = Math.max(0, Math.min(1, rec[i * d + k])) - Math.max(0, Math.min(1, org[j * d + k]));
                    s += e * e;
                }
                mse[i][j] = s / d;
            }
        }
        boolean[] usedR = new boolean[b];
        boolean[] usedO = new boolean[b];
        List<Double> psnr = new ArrayList<>();
        for (int round = 0; round < b; round++) {
            int bi = -1;
            int bj = -1;
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < b; j++) {
                    if (!usedR[i] && !usedO[j] && (bi < 0 || mse[i][j] < mse[bi][bj])) {
                        bi = i;
                        bj = j;
                    }
                }
            }
            usedR[bi] = true;
            usedO[bj] = true;
            psnr.add(10 * Math.log10(1 / Math.max(mse[bi][bj], 1e-10)));
        }
        return psnr.stream().mapToDouble(Double::doubleValue).average().orElse(0);
    }

    /**
     * The share of true labels recovered, as a multiset.
     *
     * @param labels the reconstructed labels
     * @param truth the true labels
     * @param classes the number of classes
     * @return the share in {@code [0, 1]}
     */
    public static double labelAccuracy(int[] labels, int[] truth, int classes) {
        int[] count = new int[classes];
        for (int t : truth) {
            count[t]++;
        }
        int hit = 0;
        for (int l : labels) {
            if (count[l] > 0) {
                count[l]--;
                hit++;
            }
        }
        return (double) hit / truth.length;
    }
}
