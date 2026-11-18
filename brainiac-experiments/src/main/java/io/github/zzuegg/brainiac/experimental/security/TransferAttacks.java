package io.github.zzuegg.brainiac.experimental.security;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.robust.AdversarialAttacks;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.function.UnaryOperator;

/**
 * Attacks that need no queries to the target:
 *
 * <ul>
 *   <li>{@link #miFgsm}: momentum iterative FGSM, {@code g ← μ g + ∇/‖∇‖₁},
 *       {@code x ← Π(x + α sign g)}, on one surrogate or on an ensemble whose logits are
 *       averaged (Dong et al., CVPR 2018, <a href="https://arxiv.org/abs/1710.06081">arXiv:1710.06081</a>).</li>
 *   <li>{@link #universal}: a universal adversarial perturbation: for every image the
 *       perturbation does not fool yet, add the minimal DeepFool step that crosses the
 *       nearest linearised boundary, and project onto the {@code ℓ∞} ball (Moosavi-Dezfooli
 *       et al., CVPR 2017, <a href="https://arxiv.org/abs/1610.08401">arXiv:1610.08401</a>);
 *       DeepFool runs on chunks of eight images at the same perturbation, whose steps are
 *       added one after another.</li>
 * </ul>
 */
public final class TransferAttacks {

    private TransferAttacks() {
    }

    /**
     * The ensemble of surrogates by averaged logits.
     *
     * @param models the surrogates
     * @return the ensemble
     */
    public static UnaryOperator<Var> ensemble(List<UnaryOperator<Var>> models) {
        List<UnaryOperator<Var>> m = List.copyOf(models);
        return x -> {
            Var sum = null;
            for (UnaryOperator<Var> f : m) {
                Var z = f.apply(x);
                sum = sum == null ? z : sum.add(z);
            }
            return sum.mul(1f / m.size());
        };
    }

    /**
     * MI-FGSM in {@code ℓ∞}.
     *
     * @param surrogate the surrogate (or {@link #ensemble})
     * @param x clean inputs
     * @param ids true classes
     * @param epsilon the budget
     * @param steps iterations ({@code α = ε / steps})
     * @param mu the momentum, 1 in the paper; 0 gives iterative FGSM
     * @return the adversarial inputs
     */
    public static Tensor miFgsm(UnaryOperator<Var> surrogate, Tensor x, Tensor ids, float epsilon, int steps,
            float mu) {
        int n = x.shape().dim(0);
        int d = Math.toIntExact(x.size() / n);
        float alpha = epsilon / steps;
        float[] g = new float[n * d];
        Tensor adv = x;
        for (int s = 0; s < steps; s++) {
            float[] grad = AdversarialAttacks.inputGradient(surrogate, adv, ids).toArray();
            for (int i = 0; i < n; i++) {
                double l1 = 0;
                for (int j = 0; j < d; j++) {
                    l1 += Math.abs(grad[i * d + j]);
                }
                for (int j = 0; j < d; j++) {
                    g[i * d + j] = (float) (mu * g[i * d + j] + grad[i * d + j] / Math.max(l1, 1e-12));
                }
            }
            adv = AdversarialAttacks.project(adv.add(Tensor.of(x.shape(), g).sign().mul(alpha)), x,
                    AdversarialAttacks.Norm.LINF, epsilon);
        }
        return adv;
    }

    /**
     * A universal perturbation.
     *
     * @param model the model
     * @param x images {@code [n][d]}
     * @param epsilon the {@code ℓ∞} radius
     * @param passes passes over the images
     * @param overshoot DeepFool's overshoot, such as 0.02
     * @return the perturbation {@code [d]}
     */
    public static float[] universal(UnaryOperator<Var> model, float[][] x, float epsilon, int passes,
            float overshoot) {
        int n = x.length;
        int d = x[0].length;
        float[] v = new float[d];
        int[] clean = predict(model, x);
        for (int p = 0; p < passes; p++) {
            boolean changed = false;
            for (int from = 0; from < n; from += 8) {
                int to = Math.min(n, from + 8);
                float[][] shifted = new float[to - from][];
                for (int i = from; i < to; i++) {
                    shifted[i - from] = BlackBoxAttacks.add(x[i], v);
                }
                int[] now = predict(model, shifted);
                int count = 0;
                for (int i = from; i < to; i++) {
                    count += now[i - from] == clean[i] ? 1 : 0;
                }
                if (count == 0) {
                    continue;
                }
                float[][] rows = new float[count][];
                int[] labels = new int[count];
                int c = 0;
                for (int i = from; i < to; i++) {
                    if (now[i - from] == clean[i]) {
                        rows[c] = shifted[i - from];
                        labels[c++] = clean[i];
                    }
                }
                for (float[] step : deepFool(model, rows, labels, overshoot)) {
                    for (int j = 0; j < d; j++) {
                        v[j] = Math.clamp(v[j] + step[j], -epsilon, epsilon);
                    }
                }
                changed = true;
            }
            if (!changed) {
                break;
            }
        }
        return v;
    }

    /**
     * One linearised DeepFool step per row: the smallest {@code ℓ2} move that crosses the
     * nearest class boundary of the linearised model, times {@code 1 + overshoot}.
     *
     * @param model the model
     * @param x the inputs
     * @param y their current classes
     * @param overshoot the overshoot
     * @return one step per row
     */
    public static float[][] deepFool(UnaryOperator<Var> model, float[][] x, int[] y, float overshoot) {
        int n = x.length;
        int d = x[0].length;
        Var in = Var.param(Tensor.of(x));
        Var z = model.apply(in);
        int k = z.shape().dim(1);
        float[] zv = z.value().toArray();
        float[][][] grads = new float[k][][];
        for (int c = 0; c < k; c++) {
            float[] sel = new float[k];
            sel[c] = 1;
            Var picked = z.mul(Tensor.of(Shape.of(1, k), sel)).sum();
            float[] g = Var.gradients(picked, List.of(in), false).get(0).value().toArray();
            grads[c] = new float[n][d];
            for (int i = 0; i < n; i++) {
                System.arraycopy(g, i * d, grads[c][i], 0, d);
            }
        }
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            double best = Double.POSITIVE_INFINITY;
            int bestC = -1;
            double bestNorm = 1;
            for (int c = 0; c < k; c++) {
                if (c == y[i]) {
                    continue;
                }
                double w2 = 0;
                for (int j = 0; j < d; j++) {
                    double w = grads[c][i][j] - grads[y[i]][i][j];
                    w2 += w * w;
                }
                double f = zv[i * k + c] - zv[i * k + y[i]];
                double dist = Math.abs(f) / Math.sqrt(Math.max(w2, 1e-12));
                if (dist < best) {
                    best = dist;
                    bestC = c;
                    bestNorm = Math.max(w2, 1e-12);
                }
            }
            double f = zv[i * k + bestC] - zv[i * k + y[i]];
            double scale = (Math.abs(f) + 1e-4) / bestNorm * (1 + overshoot);
            for (int j = 0; j < d; j++) {
                out[i][j] = (float) (scale * (grads[bestC][i][j] - grads[y[i]][i][j]));
            }
        }
        return out;
    }

    /**
     * The fooling rate of a universal perturbation: the share of images whose prediction it
     * changes.
     *
     * @param model the model
     * @param x images
     * @param v the perturbation
     * @return the fooling rate
     */
    public static double foolingRate(UnaryOperator<Var> model, float[][] x, float[] v) {
        int[] a = predict(model, x);
        float[][] s = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            s[i] = BlackBoxAttacks.add(x[i], v);
        }
        int[] b = predict(model, s);
        int f = 0;
        for (int i = 0; i < a.length; i++) {
            f += a[i] != b[i] ? 1 : 0;
        }
        return f / (double) a.length;
    }

    /**
     * Predicted classes.
     *
     * @param model the model
     * @param x inputs
     * @return the argmax classes
     */
    public static int[] predict(UnaryOperator<Var> model, float[][] x) {
        return model.apply(Var.of(Tensor.of(x))).value().argmax(1).toIntArray();
    }

    /**
     * The untargeted success rate of adversarial inputs on a target model: misclassified
     * among the examples the target classifies correctly when clean.
     *
     * @param target the target model
     * @param clean clean inputs
     * @param adversarial adversarial inputs
     * @param ids true classes
     * @return the success rate
     */
    public static double successRate(UnaryOperator<Var> target, Tensor clean, Tensor adversarial, Tensor ids) {
        int[] c = target.apply(Var.of(clean)).value().argmax(1).toIntArray();
        int[] a = target.apply(Var.of(adversarial)).value().argmax(1).toIntArray();
        int[] y = ids.toIntArray();
        int ok = 0;
        int fooled = 0;
        for (int i = 0; i < y.length; i++) {
            if (c[i] == y[i]) {
                ok++;
                fooled += a[i] != y[i] ? 1 : 0;
            }
        }
        return fooled / (double) Math.max(1, ok);
    }
}
