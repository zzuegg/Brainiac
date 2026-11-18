package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

/**
 * Zero-cost proxies for neural architecture search: score an untrained network from one
 * minibatch, at initialization, so thousands of candidate architectures can be ranked for the
 * price of a forward and backward pass each.
 *
 * <ul>
 *   <li>{@link #gradNorm}: the norm of the loss gradient over all weights.</li>
 *   <li>{@link #snip}: {@code sum |w * dL/dw|}, the connection sensitivity of Lee et al. 2019.</li>
 *   <li>{@link #synflow}: data-free synaptic flow (Tanaka et al. 2020): with every weight
 *       replaced by its absolute value and an all-ones input, {@code sum |w * dR/dw|} for
 *       {@code R} the sum of the outputs.</li>
 *   <li>{@link #naswot}: "NAS without training" (Mellor et al. 2021): the binary activation
 *       pattern of every unit on each input of the batch; the score is
 *       {@code log |det K|} of the kernel {@code K_ij} = the number of units on which inputs
 *       {@code i} and {@code j} agree. Networks that separate inputs well score high.</li>
 * </ul>
 *
 * <p>Survey and benchmark: Abdelfattah et al., "Zero-Cost Proxies for Lightweight NAS"
 * (2021), <a href="https://arxiv.org/abs/2101.08134">arXiv:2101.08134</a>; Mellor et al.,
 * <a href="https://arxiv.org/abs/2006.04647">arXiv:2006.04647</a>.
 */
public final class ZeroCostProxies {

    private ZeroCostProxies() {
    }

    private static List<Tensor> gradients(Model model, Var loss) {
        List<Var> g = Var.gradients(loss, model.parameters(), false);
        List<Tensor> out = new ArrayList<>();
        for (Var v : g) {
            out.add(v.value());
        }
        return out;
    }

    /**
     * The gradient norm at initialization.
     *
     * @param model the untrained model
     * @param x a batch of inputs
     * @param targets the batch's targets
     * @param loss the loss of predictions and targets
     * @return {@code ||dL/dw||}
     */
    public static double gradNorm(Model model, Tensor x, Tensor targets, BiFunction<Var, Tensor, Var> loss) {
        double s = 0;
        for (Tensor g : gradients(model, loss.apply(model.forward(Var.of(x)), targets))) {
            s += g.square().sum().item();
        }
        return Math.sqrt(s);
    }

    /**
     * SNIP's connection sensitivity.
     *
     * @param model the untrained model
     * @param x a batch of inputs
     * @param targets the batch's targets
     * @param loss the loss
     * @return {@code sum |w * dL/dw|}
     */
    public static double snip(Model model, Tensor x, Tensor targets, BiFunction<Var, Tensor, Var> loss) {
        List<Tensor> g = gradients(model, loss.apply(model.forward(Var.of(x)), targets));
        List<Var> ps = model.parameters();
        double s = 0;
        for (int i = 0; i < g.size(); i++) {
            s += ps.get(i).value().mul(g.get(i)).abs().sum().item();
        }
        return s;
    }

    /**
     * Synaptic flow, data-free; the model's weights are restored afterwards.
     *
     * @param model the untrained model
     * @param inputShape the shape of one input sample
     * @return {@code sum |w * dR/dw|}, as a logarithm because it spans many orders of magnitude
     */
    public static double synflow(Model model, Shape inputShape) {
        List<Var> ps = model.parameters();
        List<Tensor> saved = new ArrayList<>();
        for (Var p : ps) {
            saved.add(p.value());
            p.assign(p.value().abs());
        }
        try {
            int[] dims = new int[inputShape.rank() + 1];
            dims[0] = 1;
            System.arraycopy(inputShape.toArray(), 0, dims, 1, inputShape.rank());
            Var r = model.forward(Var.of(Tensor.ones(Shape.of(dims)))).sum();
            List<Tensor> g = gradients(model, r);
            double s = 0;
            for (int i = 0; i < g.size(); i++) {
                s += ps.get(i).value().mul(g.get(i)).abs().sum().item();
            }
            return Math.log(s + 1e-30);
        } finally {
            for (int i = 0; i < ps.size(); i++) {
                ps.get(i).assign(saved.get(i));
            }
        }
    }

    /**
     * NASWOT's activation-pattern score; every intermediate output of the model (all graph
     * steps but the last) contributes its units' signs.
     *
     * @param model the untrained model
     * @param x a batch of inputs, ideally a few dozen
     * @return {@code log |det K|}
     */
    public static double naswot(Model model, Tensor x) {
        int n = x.shape().dim(0);
        int last = model.graph().size() - 1;
        List<float[]> codes = new ArrayList<>();
        model.forward(Var.of(x), false, (step, out) -> {
            if (step.index() < last) {
                codes.add(out.value().toArray());
            }
            return out;
        });
        double[] k = new double[n * n];
        for (float[] c : codes) {
            int units = c.length / n;
            for (int i = 0; i < n; i++) {
                for (int j = i; j < n; j++) {
                    int agree = 0;
                    for (int u = 0; u < units; u++) {
                        agree += (c[i * units + u] > 0) == (c[j * units + u] > 0) ? 1 : 0;
                    }
                    k[i * n + j] += agree;
                    if (i != j) {
                        k[j * n + i] += agree;
                    }
                }
            }
        }
        for (int i = 0; i < n; i++) {
            k[i * n + i] += 1e-3;
        }
        double[] l = Linalg.cholesky(Tensor.of(Shape.of(n, n), k, DType.FLOAT64, Device.cpu())).toDoubleArray();
        double logDet = 0;
        for (int i = 0; i < n; i++) {
            logDet += 2 * Math.log(l[i * n + i]);
        }
        return logDet;
    }

    /**
     * Spearman's rank correlation, for judging a proxy against trained accuracies.
     *
     * @param a scores
     * @param b scores of the same items
     * @return the rank correlation in {@code [-1, 1]}
     */
    public static double spearman(double[] a, double[] b) {
        double[] ra = ranks(a);
        double[] rb = ranks(b);
        double ma = Arrays.stream(ra).average().orElse(0);
        double mb = Arrays.stream(rb).average().orElse(0);
        double cov = 0;
        double va = 0;
        double vb = 0;
        for (int i = 0; i < a.length; i++) {
            cov += (ra[i] - ma) * (rb[i] - mb);
            va += (ra[i] - ma) * (ra[i] - ma);
            vb += (rb[i] - mb) * (rb[i] - mb);
        }
        return cov / Math.sqrt(va * vb + 1e-300);
    }

    /** Ranks with ties averaged. */
    private static double[] ranks(double[] v) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (p, q) -> Double.compare(v[p], v[q]));
        double[] r = new double[v.length];
        for (int i = 0; i < idx.length;) {
            int j = i;
            while (j + 1 < idx.length && v[idx[j + 1]] == v[idx[i]]) {
                j++;
            }
            for (int k = i; k <= j; k++) {
                r[idx[k]] = (i + j) / 2.0;
            }
            i = j + 1;
        }
        return r;
    }
}
