package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Flows;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Neural spline flows (Durkan et al., <a href="https://arxiv.org/abs/1906.04032">arXiv:1906.04032</a>):
 * coupling layers whose element-wise transform is a monotone <b>rational-quadratic spline</b>
 * on {@code [-B, B]} with {@code K} bins (identity outside), instead of RealNVP's affine map.
 * The kept half of the features predicts every bin's width, height and the derivatives at
 * the inner knots; the spline is analytically invertible (a quadratic per element) and its
 * derivative is closed-form, so the log-density stays exact. One spline layer can bend a
 * Gaussian into a multi-modal or ring-shaped marginal that affine couplings need many layers
 * for.
 */
public final class SplineFlow implements DensityFlow {
    private static final float MIN_WIDTH = 1e-3f;
    private static final float MIN_DERIVATIVE = 1e-3f;
    private final int dim;
    private final int bins;
    private final float bound;
    private final List<Tensor> masks = new ArrayList<>();
    private final List<GenerativeOps.Mlp> nets = new ArrayList<>();

    private SplineFlow(int dim, int layers, int bins, float bound, int hidden, long seed) {
        if (dim < 2 || layers < 1 || bins < 2) {
            throw new IllegalArgumentException("SplineFlow needs dim >= 2, layers >= 1 and bins >= 2");
        }
        this.dim = dim;
        this.bins = bins;
        this.bound = bound;
        SplittableRandom random = new SplittableRandom(seed);
        Tensor[] alternating = Flows.alternatingMasks(dim, 2);
        for (int l = 0; l < layers; l++) {
            masks.add(alternating[l % 2]);
            nets.add(new GenerativeOps.Mlp(random, true, dim, hidden, hidden, dim * (3 * bins - 1)));
        }
    }

    /**
     * Creates the flow; with its zero-initialised output layers it starts as the identity.
     *
     * @param dim the data dimension, at least 2
     * @param layers the number of spline couplings (alternating halves)
     * @param bins the number of spline bins K
     * @param bound the half-width B of the spline interval; identity outside it
     * @param hidden the hidden width of each coupling network
     * @param seed the initialization seed
     * @return the flow
     */
    public static SplineFlow of(int dim, int layers, int bins, float bound, int hidden, long seed) {
        return new SplineFlow(dim, layers, bins, bound, hidden, seed);
    }

    /** The spline parameters per element: knot x, knot y and knot derivatives, each {@code [b, d, K+1]}. */
    private Var[] knots(GenerativeOps.Mlp net, Var kept) {
        int b = kept.shape().dim(0);
        Var raw = net.apply(kept).reshape(b, dim, 3 * bins - 1);
        Var xs = cumulative(raw.slice(2, 0, bins), b);
        Var ys = cumulative(raw.slice(2, bins, 2 * bins), b);
        Var inner = raw.slice(2, 2 * bins, 3 * bins - 1).add((float) Math.log(Math.E - 1)).softplus()
                .add(MIN_DERIVATIVE);
        Var ones = Var.of(Tensor.ones(Shape.of(b, dim, 1)));
        Var ds = Var.concat(2, ones, inner, ones);
        return new Var[] {xs, ys, ds};
    }

    private Var cumulative(Var raw, int b) {
        Var shifted = raw.sub(Var.of(raw.value().max(2, true)));
        Var e = shifted.exp();
        Var w = e.div(e.sum(2, true)).mul(1 - MIN_WIDTH * bins).add(MIN_WIDTH);
        Var c = w.cumsum(2).mul(2 * bound).sub(bound);
        return Var.concat(2, Var.of(Tensor.full(Shape.of(b, dim, 1), -bound)), c);
    }

    @Override
    public Flows.Result forward(Var x) {
        Var h = x;
        Var logDet = null;
        for (int l = 0; l < nets.size(); l++) {
            Flows.Result r = coupling(l, h);
            h = r.output();
            logDet = logDet == null ? r.logDet() : logDet.add(r.logDet());
        }
        return new Flows.Result(h, logDet);
    }

    private Flows.Result coupling(int layer, Var x) {
        Tensor keep = masks.get(layer);
        Tensor change = keep.neg().add(1f);
        int b = x.shape().dim(0);
        Var[] k = knots(nets.get(layer), x.mul(keep));
        float[] xv = x.value().toArray();
        float[] xk = k[0].value().toArray();
        float[] index = new float[b * dim];
        float[] inside = new float[b * dim];
        for (int i = 0; i < b * dim; i++) {
            float v = xv[i];
            inside[i] = v > -bound && v < bound ? 1 : 0;
            int bin = 0;
            for (int j = 1; j < bins; j++) {
                if (v >= xk[i * (bins + 1) + j]) {
                    bin = j;
                }
            }
            index[i] = bin;
        }
        Tensor lo = Tensor.of(Shape.of(b, dim, 1), index);
        Tensor hi = lo.add(1f);
        Var xc = x.clamp(-bound, bound).reshape(b, dim, 1);
        Var x0 = k[0].takeAlong(lo);
        Var w = k[0].takeAlong(hi).sub(x0);
        Var y0 = k[1].takeAlong(lo);
        Var hgt = k[1].takeAlong(hi).sub(y0);
        Var d0 = k[2].takeAlong(lo);
        Var d1 = k[2].takeAlong(hi);
        Var s = hgt.div(w);
        Var xi = xc.sub(x0).div(w).clamp(0f, 1f);
        Var xi1 = xi.neg().add(1f);
        Var mix = xi.mul(xi1);
        Var den = s.add(d1.add(d0).sub(s.mul(2f)).mul(mix));
        Var y = y0.add(hgt.mul(s.mul(xi.square()).add(d0.mul(mix))).div(den));
        Var deriv = s.square().mul(d1.mul(xi.square()).add(s.mul(mix).mul(2f)).add(d0.mul(xi1.square())))
                .div(den.square());
        Tensor in = Tensor.of(Shape.of(b, dim), inside);
        Var ys = Var.where(in, y.reshape(b, dim), x);
        Var ld = Var.where(in, deriv.log().reshape(b, dim), Var.of(Tensor.zeros(Shape.of(b, dim))));
        Var out = x.mul(keep).add(ys.mul(change));
        return new Flows.Result(out, ld.mul(change).sum(1, false));
    }

    @Override
    public Tensor inverse(Tensor z) {
        Tensor h = z;
        for (int l = nets.size() - 1; l >= 0; l--) {
            h = invertCoupling(l, h);
        }
        return h;
    }

    private Tensor invertCoupling(int layer, Tensor y) {
        Tensor keep = masks.get(layer);
        int b = y.shape().dim(0);
        Var[] k = knots(nets.get(layer), Var.of(y.mul(keep)));
        float[] xs = k[0].value().toArray();
        float[] ys = k[1].value().toArray();
        float[] ds = k[2].value().toArray();
        float[] yv = y.toArray();
        float[] mask = keep.toArray();
        float[] out = yv.clone();
        for (int i = 0; i < b * dim; i++) {
            float v = yv[i];
            if (mask[i % dim] == 1 || v <= -bound || v >= bound) {
                continue;
            }
            int base = i * (bins + 1);
            int bin = 0;
            for (int j = 1; j < bins; j++) {
                if (v >= ys[base + j]) {
                    bin = j;
                }
            }
            double x0 = xs[base + bin];
            double w = xs[base + bin + 1] - x0;
            double y0 = ys[base + bin];
            double hgt = ys[base + bin + 1] - y0;
            double d0 = ds[base + bin];
            double d1 = ds[base + bin + 1];
            double s = hgt / w;
            double dy = v - y0;
            double a = hgt * (s - d0) + dy * (d1 + d0 - 2 * s);
            double bq = hgt * d0 - dy * (d1 + d0 - 2 * s);
            double c = -s * dy;
            double disc = Math.max(0, bq * bq - 4 * a * c);
            double xi = 2 * c / (-bq - Math.sqrt(disc));
            out[i] = (float) (x0 + Math.max(0, Math.min(1, xi)) * w);
        }
        return Tensor.of(Shape.of(b, dim), out);
    }

    @Override
    public int dimension() {
        return dim;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (GenerativeOps.Mlp n : nets) {
            p.addAll(n.parameters());
        }
        return p;
    }
}
