package io.github.zzuegg.brainiac.experimental.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.MinibatchOptimalTransport;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Linear mode connectivity and Git Re-Basin (Frankle et al., ICML 2020; Ainsworth et al.,
 * ICLR 2023) for multilayer perceptrons built from core {@link Dense} layers. Two networks
 * trained from different seeds usually have a loss barrier on the straight line between
 * their weights, much of it due to permutation symmetry: hidden units can be reordered
 * without changing the function. This class permutes network B's hidden units to line up
 * with A's, after which interpolating (or averaging) the two often has almost no barrier.
 *
 * <ul>
 *   <li>{@link #weightMatching}: maximise {@code sum_l <W_l^A, P_l W_l^B P_(l-1)^T>} by
 *       coordinate descent over layers, each step a linear assignment solved by the Hungarian
 *       algorithm ({@link MinibatchOptimalTransport#assignCost});</li>
 *   <li>{@link #activationMatching}: per layer, match units by the inner product of their
 *       activations on data;</li>
 *   <li>{@link #barrier}: the largest rise of a loss above the straight line between the
 *       end points' losses, along the interpolation path.</li>
 * </ul>
 *
 * <p>Papers: <a href="https://arxiv.org/abs/1912.05671">arXiv:1912.05671</a>,
 * <a href="https://arxiv.org/abs/2209.04836">arXiv:2209.04836</a>.
 */
public final class PermutationAlignment {

    private PermutationAlignment() {
    }

    /**
     * The dense layers of a model, in order.
     *
     * @param model a model whose weighted layers are all {@link Dense}
     * @return its dense layers
     */
    public static List<Dense> denseLayers(Model model) {
        List<Dense> out = new ArrayList<>();
        for (Layer l : model.layers()) {
            if (l instanceof Dense d) {
                out.add(d);
            }
        }
        return out;
    }

    private static double[][] w(Dense d) {
        return InterpMath.matrix(d.weight().value());
    }

    private static double[] b(Dense d) {
        return d.bias().value().toDoubleArray();
    }

    /**
     * Weight matching by coordinate descent.
     *
     * @param a network A
     * @param b network B
     * @param rounds passes over the layers (stops early when nothing changes)
     * @param seed the seed of the layer order
     * @return per hidden layer {@code l} the permutation: A's unit {@code i} is matched with
     *     B's unit {@code perm[l][i]}
     */
    public static int[][] weightMatching(Model a, Model b, int rounds, long seed) {
        List<Dense> la = denseLayers(a);
        List<Dense> lb = denseLayers(b);
        int hidden = la.size() - 1;
        int[][] perm = new int[hidden][];
        for (int l = 0; l < hidden; l++) {
            perm[l] = identity(la.get(l).bias().value().shape().dim(0));
        }
        SplittableRandom r = new SplittableRandom(seed);
        for (int round = 0; round < rounds; round++) {
            boolean changed = false;
            int[] order = r.ints(0, hidden).distinct().limit(hidden).toArray();
            for (int l : order) {
                double[][] wa = w(la.get(l));
                double[][] wb = w(lb.get(l));
                double[] ba = b(la.get(l));
                double[] bb = b(lb.get(l));
                double[][] na = w(la.get(l + 1));
                double[][] nb = w(lb.get(l + 1));
                int units = ba.length;
                int[] prev = l == 0 ? identity(wa.length) : perm[l - 1];
                int[] next = l + 1 < hidden ? perm[l + 1] : identity(na[0].length);
                double[][] cost = new double[units][units];
                for (int i = 0; i < units; i++) {
                    for (int j = 0; j < units; j++) {
                        double s = ba[i] * bb[j];
                        for (int k = 0; k < wa.length; k++) {
                            s += wa[k][i] * wb[prev[k]][j];
                        }
                        for (int m = 0; m < na[0].length; m++) {
                            s += na[i][m] * nb[j][next[m]];
                        }
                        cost[i][j] = -s;
                    }
                }
                int[] p = MinibatchOptimalTransport.assignCost(cost);
                if (!java.util.Arrays.equals(p, perm[l])) {
                    changed = true;
                    perm[l] = p;
                }
            }
            if (!changed) {
                break;
            }
        }
        return perm;
    }

    /**
     * Activation matching: per hidden layer, the assignment maximising the summed inner
     * products of matched units' activations.
     *
     * @param a network A
     * @param b network B
     * @param x data {@code [n, inputs]}
     * @return the permutations, as for {@link #weightMatching}
     */
    public static int[][] activationMatching(Model a, Model b, Tensor x) {
        int hidden = denseLayers(a).size() - 1;
        int[][] perm = new int[hidden][];
        for (int l = 0; l < hidden; l++) {
            double[][] za = InterpMath.matrix(ModelStitcher.activation(a, l, x));
            double[][] zb = InterpMath.matrix(ModelStitcher.activation(b, l, x));
            double[][] c = InterpMath.crossProduct(InterpMath.centre(za), InterpMath.centre(zb), za.length);
            for (double[] row : c) {
                for (int j = 0; j < row.length; j++) {
                    row[j] = -row[j];
                }
            }
            perm[l] = MinibatchOptimalTransport.assignCost(c);
        }
        return perm;
    }

    private static int[] identity(int n) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        return p;
    }

    /**
     * The parameters of B with its hidden units permuted, as {@code [W_0, b_0, W_1, b_1, ...]}.
     *
     * @param b network B
     * @param perm the permutations from a matching
     * @return the permuted weights and biases
     */
    public static List<Tensor> permuted(Model b, int[][] perm) {
        List<Dense> lb = denseLayers(b);
        List<Tensor> out = new ArrayList<>();
        for (int l = 0; l < lb.size(); l++) {
            double[][] wb = w(lb.get(l));
            double[] bb = b(lb.get(l));
            int[] prev = l == 0 ? identity(wb.length) : perm[l - 1];
            int[] cur = l < perm.length ? perm[l] : identity(bb.length);
            double[] wf = new double[wb.length * bb.length];
            double[] bf = new double[bb.length];
            for (int i = 0; i < bb.length; i++) {
                bf[i] = bb[cur[i]];
                for (int k = 0; k < wb.length; k++) {
                    wf[k * bb.length + i] = wb[prev[k]][cur[i]];
                }
            }
            out.add(Tensor.of(Shape.of(wb.length, bb.length), toFloat(wf)));
            out.add(Tensor.of(Shape.of(bb.length), toFloat(bf)));
        }
        return out;
    }

    private static float[] toFloat(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return f;
    }

    /**
     * The weights and biases of a model, as {@code [W_0, b_0, W_1, b_1, ...]}.
     *
     * @param model the model
     * @return copies of its dense parameters
     */
    public static List<Tensor> weights(Model model) {
        List<Tensor> out = new ArrayList<>();
        for (Dense d : denseLayers(model)) {
            out.add(d.weight().value().add(0f));
            out.add(d.bias().value().add(0f));
        }
        return out;
    }

    /**
     * A copy of a model with given dense weights.
     *
     * @param template the model whose architecture to copy
     * @param weights {@code [W_0, b_0, ...]}
     * @return the new model
     */
    public static Model withWeights(Model template, List<Tensor> weights) {
        Model m = template.copy(Device.cpu());
        List<Dense> d = denseLayers(m);
        for (int l = 0; l < d.size(); l++) {
            d.get(l).weight().assign(weights.get(2 * l));
            d.get(l).bias().assign(weights.get(2 * l + 1));
        }
        return m;
    }

    /**
     * Linear interpolation {@code (1 - t) a + t b} of two weight lists.
     *
     * @param a the first
     * @param b the second
     * @param t the position
     * @return the interpolated weights
     */
    public static List<Tensor> interpolate(List<Tensor> a, List<Tensor> b, double t) {
        List<Tensor> out = new ArrayList<>();
        for (int i = 0; i < a.size(); i++) {
            out.add(a.get(i).mul((float) (1 - t)).add(b.get(i).mul((float) t)));
        }
        return out;
    }

    /**
     * The loss barrier along the straight path from A to B.
     *
     * @param template a model of the architecture
     * @param a the weights of A
     * @param b the weights of B (permuted or not)
     * @param points interpolation points, at least 3
     * @param loss the loss (or error) of a model
     * @return {@code max_t [L((1-t) a + t b) - ((1-t) L(a) + t L(b))]}
     */
    public static double barrier(Model template, List<Tensor> a, List<Tensor> b, int points,
            ToDoubleFunction<Model> loss) {
        double la = loss.applyAsDouble(withWeights(template, a));
        double lb = loss.applyAsDouble(withWeights(template, b));
        double worst = 0;
        for (int i = 1; i < points - 1; i++) {
            double t = (double) i / (points - 1);
            double l = loss.applyAsDouble(withWeights(template, interpolate(a, b, t)));
            worst = Math.max(worst, l - ((1 - t) * la + t * lb));
        }
        return worst;
    }

    /**
     * Logits of the output ensemble (mean of two models' logits), the upper bound for merges.
     *
     * @param a the first model
     * @param b the second model
     * @param x inputs
     * @return the mean logits
     */
    public static Tensor ensembleLogits(Model a, Model b, Tensor x) {
        return Var.noGrad(() -> a.forward(Var.of(x)).add(b.forward(Var.of(x))).mul(0.5f)).value();
    }
}
