package io.github.zzuegg.brainiac.experimental.robust;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Guaranteed output bounds of a ReLU network over an {@code ℓ∞} ball of inputs, for
 * verification and certified training. The network is a chain of affine layers
 * {@code z = h W + b} (dense layers, or convolutions unrolled into their dense matrix by
 * {@link #conv}) with ReLU between them; everything is built from {@link Var} operations, so
 * the bounds are differentiable and can be trained on.
 *
 * <ul>
 *   <li>{@link #ibp}: interval bound propagation, centre {@code c} and radius {@code r} pushed
 *       through each layer as {@code c W + b} and {@code r |W|} (Gowal et al. 2018,
 *       <a href="https://arxiv.org/abs/1810.12715">arXiv:1810.12715</a>).</li>
 *   <li>{@link #crown}: CROWN (Zhang et al., NeurIPS 2018,
 *       <a href="https://arxiv.org/abs/1811.00866">arXiv:1811.00866</a>) propagates a linear
 *       bound on the specification backwards through the network, relaxing every unstable ReLU
 *       ({@code l < 0 < u}) by the chord {@code u (z - l) / (u - l)} above and by {@code α z}
 *       below ({@code α = 1} when {@code u > -l}, else 0); intermediate bounds come from CROWN
 *       itself, layer by layer.</li>
 *   <li>{@link #crownIbp}: the same backward pass with IBP intermediate bounds, the cheap and
 *       stable variant used for certified training (Zhang et al., ICLR 2020,
 *       <a href="https://arxiv.org/abs/1906.06316">arXiv:1906.06316</a>).</li>
 * </ul>
 *
 * <p>Verification uses the margin specification: for true class {@code y} and every other
 * class {@code j}, a lower bound on {@code z_y - z_j}; the input is verified when all are
 * positive.
 */
public final class BoundPropagation {

    /**
     * One affine layer {@code z = h W + b}.
     *
     * @param weight {@code [in, out]}
     * @param bias {@code [1, out]}
     */
    public record Affine(Var weight, Var bias) {

        /** @return the input width */
        public int in() {
            return weight.shape().dim(0);
        }

        /** @return the output width */
        public int out() {
            return weight.shape().dim(1);
        }
    }

    private BoundPropagation() {
    }

    /**
     * The layers of a multi-layer perceptron (whose hidden activation must be ReLU).
     *
     * @param net the network
     * @return its affine layers
     */
    public static List<Affine> of(SimpleMlp net) {
        List<Affine> out = new ArrayList<>();
        for (int l = 0; l < net.depth(); l++) {
            out.add(new Affine(net.weight(l), net.bias(l)));
        }
        return out;
    }

    /**
     * A convolution as a dense affine layer on flattened {@code [channels, height, width]}
     * inputs: the dense matrix is the convolution of the identity basis, so it stays
     * differentiable in the kernel.
     *
     * @param kernel {@code [filters, channels, k, k]}
     * @param bias {@code [filters]}, or {@code null}
     * @param channels the input channels
     * @param height the input height
     * @param width the input width
     * @param stride the stride
     * @param pad the zero padding on every side
     * @return the affine layer, input {@code channels * height * width}
     */
    public static Affine conv(Var kernel, Var bias, int channels, int height, int width, int stride, int pad) {
        int in = channels * height * width;
        float[] eye = new float[in * in];
        for (int i = 0; i < in; i++) {
            eye[i * in + i] = 1;
        }
        Tensor basis = Tensor.of(Shape.of(in, channels, height, width), eye);
        Var y = Functional.conv(Var.of(basis), kernel, null, new int[] {stride, stride}, new int[] {pad, pad},
                new int[] {pad, pad}, new int[] {1, 1}, 1);
        int out = Math.toIntExact(y.value().size() / in);
        Var w = y.reshape(in, out);
        Var b;
        if (bias == null) {
            b = Var.of(Tensor.zeros(Shape.of(1, out)));
        } else {
            int positions = out / kernel.shape().dim(0);
            b = bias.reshape(kernel.shape().dim(0), 1).expand(Shape.of(kernel.shape().dim(0), positions))
                    .reshape(1, out);
        }
        return new Affine(w, b);
    }

    /**
     * The network's output (no ReLU after the last layer).
     *
     * @param layers the layers
     * @param x the inputs {@code [n, in]}
     * @return the logits
     */
    public static Var forward(List<Affine> layers, Var x) {
        Var h = x;
        for (int l = 0; l < layers.size(); l++) {
            h = h.matmul(layers.get(l).weight()).add(layers.get(l).bias());
            if (l < layers.size() - 1) {
                h = h.relu();
            }
        }
        return h;
    }

    /**
     * IBP bounds of every layer's pre-activation.
     *
     * @param layers the layers
     * @param x the centres {@code [n, in]}
     * @param epsilon the {@code ℓ∞} radius
     * @return {@code [layer] -> {lower, upper}}
     */
    public static List<Var[]> ibpLayers(List<Affine> layers, Var x, float epsilon) {
        List<Var[]> out = new ArrayList<>();
        Var lo = x.sub(epsilon);
        Var hi = x.add(epsilon);
        for (int l = 0; l < layers.size(); l++) {
            Affine a = layers.get(l);
            Var c = lo.add(hi).mul(0.5f);
            Var r = hi.sub(lo).mul(0.5f);
            Var zc = c.matmul(a.weight()).add(a.bias());
            Var zr = r.matmul(a.weight().abs());
            Var zl = zc.sub(zr);
            Var zu = zc.add(zr);
            out.add(new Var[] {zl, zu});
            lo = zl.relu();
            hi = zu.relu();
        }
        return out;
    }

    /**
     * IBP bounds on the logits.
     *
     * @param layers the layers
     * @param x the centres
     * @param epsilon the radius
     * @return {@code {lower, upper}} of the logits
     */
    public static Var[] ibp(List<Affine> layers, Var x, float epsilon) {
        List<Var[]> all = ibpLayers(layers, x, epsilon);
        return all.get(all.size() - 1);
    }

    /**
     * IBP lower bounds on the margins {@code z_y - z_j}, computed with the specification
     * folded into the last layer (tighter than subtracting logit bounds).
     *
     * @param layers the layers
     * @param x the centres
     * @param epsilon the radius
     * @param labels the true classes
     * @return {@code [n, classes]} margin lower bounds (0 at the true class)
     */
    public static Var ibpMargins(List<Affine> layers, Var x, float epsilon, int[] labels) {
        List<Var[]> all = ibpLayers(layers, x, epsilon);
        int n = labels.length;
        int classes = layers.get(layers.size() - 1).out();
        Affine last = layers.get(layers.size() - 1);
        if (layers.size() == 1) {
            return backward(layers, all, x, epsilon, spec(labels, classes), layers.size());
        }
        Var[] prev = all.get(all.size() - 2);
        Var lo = prev[0].relu();
        Var hi = prev[1].relu();
        Var c = lo.add(hi).mul(0.5f).reshape(n, 1, last.in());
        Var r = hi.sub(lo).mul(0.5f).reshape(n, 1, last.in());
        // per-example spec weights [n, K, in] = C [n, K, out] x W^T [out, in]
        Var cw = Var.of(spec(labels, classes)).matmul(last.weight().transpose());
        Var cb = Var.of(spec(labels, classes)).matmul(last.bias().reshape(classes, 1)).reshape(n, classes);
        Var centre = c.mul(cw).sum(2, false).add(cb);
        Var radius = r.mul(cw.abs()).sum(2, false);
        return centre.sub(radius);
    }

    /**
     * CROWN lower bounds on the margins, with CROWN intermediate bounds.
     *
     * @param layers the layers
     * @param x the centres
     * @param epsilon the radius
     * @param labels the true classes
     * @return {@code [n, classes]} margin lower bounds
     */
    public static Var crown(List<Affine> layers, Var x, float epsilon, int[] labels) {
        List<Var[]> bounds = new ArrayList<>();
        int n = x.shape().dim(0);
        for (int k = 0; k < layers.size() - 1; k++) {
            int width = layers.get(k).out();
            Tensor eye = identitySpec(n, width);
            Var lower = backward(layers, bounds, x, epsilon, eye, k + 1);
            Var upper = backward(layers, bounds, x, epsilon, eye.neg(), k + 1).neg();
            bounds.add(new Var[] {lower, upper});
        }
        int classes = layers.get(layers.size() - 1).out();
        return backward(layers, bounds, x, epsilon, spec(labels, classes), layers.size());
    }

    /**
     * CROWN-IBP lower bounds on the margins: IBP intermediate bounds, a CROWN backward pass
     * for the specification.
     *
     * @param layers the layers
     * @param x the centres
     * @param epsilon the radius
     * @param labels the true classes
     * @return {@code [n, classes]} margin lower bounds
     */
    public static Var crownIbp(List<Affine> layers, Var x, float epsilon, int[] labels) {
        List<Var[]> ibp = ibpLayers(layers, x, epsilon);
        int classes = layers.get(layers.size() - 1).out();
        return backward(layers, ibp, x, epsilon, spec(labels, classes), layers.size());
    }

    /**
     * The backward linear bound pass: a lower bound on {@code Λ z_k} where {@code z_k} is the
     * pre-activation of layer {@code k - 1} (the output of {@code k} affine layers).
     *
     * @param layers the layers
     * @param bounds pre-activation bounds of at least the first {@code k - 1} layers
     * @param x the centres {@code [n, in]}
     * @param epsilon the radius
     * @param spec {@code [n, K, width]} specification on {@code z_k}
     * @param k the number of affine layers the specification sits after
     * @return {@code [n, K]} lower bounds
     */
    static Var backward(List<Affine> layers, List<Var[]> bounds, Var x, float epsilon, Tensor spec, int k) {
        int n = spec.shape().dim(0);
        int specs = spec.shape().dim(1);
        Var lambda = Var.of(spec);
        Var constant = Var.of(Tensor.zeros(Shape.of(n, specs)));
        for (int l = k - 1; l >= 0; l--) {
            Affine a = layers.get(l);
            // the affine layer: Λ (h W + b) = (Λ W^T) h + Λ b
            constant = constant.add(lambda.mul(a.bias().reshape(1, 1, a.out())).sum(2, false));
            lambda = lambda.matmul(a.weight().transpose());
            if (l == 0) {
                break;
            }
            // the ReLU before it, relaxed with the bounds of layer l - 1's pre-activation
            Var lo = bounds.get(l - 1)[0];
            Var hi = bounds.get(l - 1)[1];
            Tensor lv = lo.value();
            Tensor hv = hi.value();
            Tensor unstable = lv.less(Tensor.zeros(lv.shape())).mul(hv.greater(Tensor.zeros(hv.shape())));
            Tensor active = lv.greater(Tensor.zeros(lv.shape())).add(lv.equalTo(Tensor.zeros(lv.shape())));
            Var safeGap = hi.sub(lo).add(Var.of(Tensor.ones(lv.shape()).sub(unstable)));
            Var upperSlope = Var.where(unstable, hi.div(safeGap), Var.of(active));
            Var upperShift = Var.where(unstable, lo.neg().mul(hi).div(safeGap), Var.of(Tensor.zeros(lv.shape())));
            Tensor bigger = hv.greater(lv.neg());
            Tensor lowerSlope = unstable.mul(bigger).add(active.mul(Tensor.ones(lv.shape()).sub(unstable)));
            int width = a.in();
            Tensor positive = lambda.value().greater(Tensor.zeros(lambda.value().shape()));
            Var lamPos = lambda.mul(Var.of(positive));
            Var lamNeg = lambda.sub(lamPos);
            Var su = upperSlope.reshape(n, 1, width);
            Var tu = upperShift.reshape(n, 1, width);
            constant = constant.add(lamNeg.mul(tu).sum(2, false));
            lambda = lamPos.mul(Var.of(lowerSlope.reshape(n, 1, width))).add(lamNeg.mul(su));
        }
        // the input box: min over ||δ||∞ <= ε of Λ (x + δ) = Λ x - ε ||Λ||_1
        Var center = lambda.mul(x.reshape(n, 1, x.shape().dim(1))).sum(2, false);
        Var radius = lambda.abs().sum(2, false).mul(epsilon);
        return center.sub(radius).add(constant);
    }

    /**
     * The margin specification: row {@code j} is {@code e_y - e_j}.
     *
     * @param labels the true classes
     * @param classes the number of classes
     * @return {@code [n, classes, classes]}
     */
    public static Tensor spec(int[] labels, int classes) {
        int n = labels.length;
        float[] c = new float[n * classes * classes];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < classes; j++) {
                if (j != labels[i]) {
                    c[(i * classes + j) * classes + labels[i]] += 1;
                    c[(i * classes + j) * classes + j] -= 1;
                }
            }
        }
        return Tensor.of(Shape.of(n, classes, classes), c);
    }

    private static Tensor identitySpec(int n, int width) {
        float[] c = new float[n * width * width];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < width; j++) {
                c[(i * width + j) * width + j] = 1;
            }
        }
        return Tensor.of(Shape.of(n, width, width), c);
    }

    /**
     * Verified accuracy: the share of inputs whose every margin lower bound (other than the
     * true class's) is positive.
     *
     * @param margins {@code [n, classes]} margin lower bounds
     * @param labels the true classes
     * @return the share verified
     */
    public static double verified(Tensor margins, int[] labels) {
        float[] m = margins.toArray();
        int classes = margins.shape().dim(1);
        int ok = 0;
        for (int i = 0; i < labels.length; i++) {
            boolean all = true;
            for (int j = 0; j < classes; j++) {
                if (j != labels[i] && m[i * classes + j] <= 0) {
                    all = false;
                }
            }
            ok += all ? 1 : 0;
        }
        return (double) ok / labels.length;
    }

    /**
     * The worst-case logits implied by margin lower bounds, {@code -lb(z_y - z_j)} (0 at the
     * true class), whose cross-entropy is the certified training loss.
     *
     * @param margins {@code [n, classes]} margin lower bounds
     * @return the worst-case logits
     */
    public static Var worstCaseLogits(Var margins) {
        return margins.neg();
    }
}
