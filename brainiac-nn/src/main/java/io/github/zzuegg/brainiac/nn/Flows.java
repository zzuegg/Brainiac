package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.function.Function;

/**
 * Normalizing flows: invertible maps with a cheap log-determinant, trained by maximum
 * likelihood under a standard normal base, for density estimation and anomaly scores.
 */
public final class Flows {

    private Flows() {
    }

    /** A step of the flow: its output and the log-determinant of its Jacobian per row. */
    public record Result(Var output, Var logDet) {
    }

    /**
     * A RealNVP affine coupling layer (Dinh et al., 2017): the features where {@code mask} is
     * 1 pass unchanged and condition a scale and shift of the others.
     *
     * @param mask {@code [features]} of 0 and 1
     * @param net maps the masked input {@code [batch, features]} to a log-scale and a shift,
     *     each {@code [batch, features]}; any network, and itself never inverted
     */
    public record Coupling(Tensor mask, Function<Var, Var[]> net) {

        /**
         * @param x {@code [batch, features]}
         * @return {@code y} and {@code log |det dy/dx|} per row
         */
        public Result forward(Var x) {
            Tensor keep = mask.to(x.value().device()).to(x.value().dtype());
            Tensor change = keep.neg().add(1);
            Var xm = x.mul(keep);
            Var[] st = net.apply(xm);
            Var s = st[0].tanh().mul(change);   // tanh keeps the scales bounded early in training
            Var t = st[1].mul(change);
            Var y = xm.add(x.mul(s.exp()).add(t).mul(change));
            return new Result(y, s.sum(1, false));
        }

        /**
         * @param y {@code [batch, features]}
         * @return {@code x}, with {@code log |det dx/dy|} per row
         */
        public Result inverse(Var y) {
            Tensor keep = mask.to(y.value().device()).to(y.value().dtype());
            Tensor change = keep.neg().add(1);
            Var ym = y.mul(keep);
            Var[] st = net.apply(ym);
            Var s = st[0].tanh().mul(change);
            Var t = st[1].mul(change);
            Var x = ym.add(y.sub(t).mul(s.neg().exp()).mul(change));
            return new Result(x, s.sum(1, false).neg());
        }
    }

    /**
     * Runs couplings in turn.
     *
     * @param steps the couplings, typically with alternating masks
     * @param x {@code [batch, features]}
     * @return the base-space values and the summed log-determinant per row
     */
    public static Result forward(List<Coupling> steps, Var x) {
        Var logDet = null;
        Var at = x;
        for (Coupling c : steps) {
            Result r = c.forward(at);
            at = r.output();
            logDet = logDet == null ? r.logDet() : logDet.add(r.logDet());
        }
        return new Result(at, logDet);
    }

    /**
     * Runs couplings backwards: samples from base-space values.
     *
     * @param steps the couplings, in their forward order
     * @param z {@code [batch, features]}
     * @return data-space values and the summed log-determinant per row
     */
    public static Result inverse(List<Coupling> steps, Var z) {
        Var logDet = null;
        Var at = z;
        for (int i = steps.size() - 1; i >= 0; i--) {
            Result r = steps.get(i).inverse(at);
            at = r.output();
            logDet = logDet == null ? r.logDet() : logDet.add(r.logDet());
        }
        return new Result(at, logDet);
    }

    /**
     * The log-likelihood of data under the flow with a standard normal base:
     * {@code log N(z; 0, I) + log |det|} per row. Its negative mean is the training loss, and
     * low values flag anomalies.
     *
     * @param mapped the result of {@link #forward(List, Var)}
     * @return {@code [batch]}
     */
    public static Var logLikelihood(Result mapped) {
        Var z = mapped.output();
        int features = z.shape().dim(1);
        Var base = z.square().sum(1, false).mul(-0.5f).sub((float) (0.5 * features * Math.log(2 * Math.PI)));
        return base.add(mapped.logDet());
    }

    /**
     * Alternating masks: the first keeps even features, the next odd ones, and so on.
     *
     * @param features the features
     * @param count how many masks
     * @return the masks
     */
    public static Tensor[] alternatingMasks(int features, int count) {
        Tensor[] masks = new Tensor[count];
        for (int k = 0; k < count; k++) {
            float[] m = new float[features];
            for (int i = 0; i < features; i++) {
                m[i] = (i + k) % 2 == 0 ? 1 : 0;
            }
            masks[k] = Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(features), m);
        }
        return masks;
    }

    /**
     * A monotonic rational-quadratic spline (Durkan et al., 2019, "Neural Spline Flows"),
     * applied element-wise: inside {@code [-tail, tail]} each value goes through its own
     * spline of {@code bins} bins, whose widths, heights and inner knot derivatives come
     * from a network; outside, the identity. VITS's stochastic duration predictor uses it.
     * Differentiable in the input and in all spline parameters.
     *
     * @param x the values {@code [...]}
     * @param widths unnormalized bin widths {@code [..., bins]} (softmax-normalized here)
     * @param heights unnormalized bin heights {@code [..., bins]}
     * @param derivatives unnormalized derivatives at the inner knots {@code [..., bins - 1]}
     *     (softplus here; the outer knots have derivative 1)
     * @param tail the bound of the spline's range
     * @param inverse whether to invert the spline
     * @return the output and the log-determinant per element (of the map applied)
     */
    public static Result rationalQuadraticSpline(Var x, Var widths, Var heights, Var derivatives, float tail,
            boolean inverse) {
        final float minW = 1e-3f;
        final float minH = 1e-3f;
        final float minD = 1e-3f;
        int bins = widths.shape().dim(-1);
        int rank = widths.shape().rank();
        int last = rank - 1;
        io.github.zzuegg.brainiac.tensor.Device device = x.value().device();
        int[] lead = java.util.Arrays.copyOf(widths.shape().toArray(), rank);
        lead[last] = 1;
        io.github.zzuegg.brainiac.tensor.Shape edgeShape = io.github.zzuegg.brainiac.tensor.Shape.of(lead);
        Var lo = Var.of(Tensor.full(edgeShape, -tail, device).to(x.dtype()));
        Var hi = Var.of(Tensor.full(edgeShape, tail, device).to(x.dtype()));
        Var cw = knots(widths, minW, tail, lo, hi, bins, last);
        Var ch = knots(heights, minH, tail, lo, hi, bins, last);
        Var w = cw.slice(last, 1, bins + 1).sub(cw.slice(last, 0, bins));
        Var h = ch.slice(last, 1, bins + 1).sub(ch.slice(last, 0, bins));
        float edge = (float) Math.log(Math.exp(1 - minD) - 1);
        Var edgeD = Var.of(Tensor.full(edgeShape, edge, device).to(x.dtype()));
        Var d = Var.concat(last, edgeD, derivatives, edgeD).softplus().add(minD);
        // the bin of each value (no gradient): knots at or below it, minus one
        float[] xs = x.value().to(io.github.zzuegg.brainiac.tensor.DType.FLOAT32).toArray();
        float[] locs = (inverse ? ch : cw).value().to(io.github.zzuegg.brainiac.tensor.DType.FLOAT32).toArray();
        float[] idx = new float[xs.length];
        for (int e = 0; e < xs.length; e++) {
            int count = 0;
            for (int k = 0; k <= bins; k++) {
                float knot = locs[e * (bins + 1) + k] + (k == bins ? 1e-6f : 0);
                if (xs[e] >= knot) {
                    count++;
                }
            }
            idx[e] = Math.max(0, Math.min(bins - 1, count - 1));
        }
        Tensor at = Tensor.of(edgeShape, idx, device);
        Tensor next = Tensor.of(edgeShape, idx, device).add(1f);
        io.github.zzuegg.brainiac.tensor.Shape xs0 = x.shape();
        Var inCw = cw.takeAlong(at).reshape(xs0.toArray());
        Var inW = w.takeAlong(at).reshape(xs0.toArray());
        Var inCh = ch.takeAlong(at).reshape(xs0.toArray());
        Var inH = h.takeAlong(at).reshape(xs0.toArray());
        Var delta = inH.div(inW);
        Var d0 = d.takeAlong(at).reshape(xs0.toArray());
        Var d1 = d.takeAlong(next).reshape(xs0.toArray());
        Tensor outside = Var.of(x.value()).abs().greater(tail);   // 1 outside the spline's range
        Var xc = x.clamp(-tail, tail);
        Var i1 = d0.add(d1).sub(delta.mul(2));
        Var out;
        Var logDet;
        if (!inverse) {
            Var theta = xc.sub(inCw).div(inW);
            Var t1 = theta.mul(theta.neg().add(1));
            Var num = inH.mul(delta.mul(theta.square()).add(d0.mul(t1)));
            Var den = delta.add(i1.mul(t1));
            out = inCh.add(num.div(den));
            Var dn = delta.square().mul(d1.mul(theta.square()).add(delta.mul(t1).mul(2))
                    .add(d0.mul(theta.neg().add(1).square())));
            logDet = dn.log().sub(den.log().mul(2));
        } else {
            Var i2 = xc.sub(inCh);
            Var i3 = i2.mul(i1);
            Var a = inH.mul(delta.sub(d0)).add(i3);
            Var b = inH.mul(d0).sub(i3);
            Var c = delta.neg().mul(i2);
            Var disc = b.square().sub(a.mul(c).mul(4)).maximum(0f);
            Var root = c.mul(2).div(b.neg().sub(disc.sqrt()));
            out = root.mul(inW).add(inCw);
            Var t1 = root.mul(root.neg().add(1));
            Var den = delta.add(i1.mul(t1));
            Var dn = delta.square().mul(d1.mul(root.square()).add(delta.mul(t1).mul(2))
                    .add(d0.mul(root.neg().add(1).square())));
            logDet = dn.log().sub(den.log().mul(2)).neg();
        }
        Tensor keep = outside.neg().add(1f).to(x.dtype());
        Var y = Var.where(outside, x, out);
        return new Result(y, logDet.mul(keep));
    }

    /** Knot positions {@code [..., bins + 1]} from {@code -tail} to {@code tail}. */
    private static Var knots(Var raw, float min, float tail, Var lo, Var hi, int bins, int last) {
        Var p = raw.sub(raw.logsumexp(last, true)).exp().mul(1 - min * bins).add(min);
        Var inner = p.cumsum(last).slice(last, 0, bins - 1).mul(2 * tail).sub(tail);
        return Var.concat(last, lo, inner, hi);
    }
}
