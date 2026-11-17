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
}
