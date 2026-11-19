package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * An AstroNet-style vetter (Shallue and Vanderburg, AJ 2018): two 1-D convolutional branches
 * over the global and the local view of a folded light curve ({@link TransitSearch#views}),
 * their max-pooled features concatenated into a logistic read-out of "planet" against "false
 * positive" (eclipsing binaries and variability).
 */
public final class TransitVetter {

    private final Cnn1d global;
    private final Cnn1d local;
    private final Var head;
    private final Var bias;

    /**
     * An untrained vetter.
     *
     * @param seed the seed
     */
    public TransitVetter(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.global = new Cnn1d(1, new int[] {8, 16, 16}, 5, seed);
        this.local = new Cnn1d(1, new int[] {8, 16}, 5, seed + 1);
        int w = global.width() + local.width();
        float[] h = new float[w];
        for (int i = 0; i < w; i++) {
            h[i] = (float) (0.1 * r.nextGaussian());
        }
        this.head = Var.param(Tensor.of(Shape.of(w, 1), h));
        this.bias = Var.param(Tensor.zeros(Shape.of(1)));
    }

    private static Tensor tensor(double[][][] views, int which, int[] idx) {
        int len = views[idx[0]][which].length;
        float[] v = new float[idx.length * len];
        for (int b = 0; b < idx.length; b++) {
            for (int i = 0; i < len; i++) {
                v[b * len + i] = (float) views[idx[b]][which][i];
            }
        }
        return Tensor.of(Shape.of(idx.length, 1, len), v);
    }

    private Var logits(Tensor g, Tensor l) {
        return Var.concat(1, global.features(Var.of(g)), local.features(Var.of(l))).matmul(head).add(bias);
    }

    /**
     * Trains by logistic loss.
     *
     * @param views {@code [example][global, local][bins]}
     * @param labels 1 for a planet
     * @param steps Adam steps
     * @param batch the batch
     * @param lr the step
     * @param seed the seed
     * @return this vetter
     */
    public TransitVetter train(double[][][] views, int[] labels, int steps, int batch, double lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Var> params = new java.util.ArrayList<>(global.featureParameters());
        params.addAll(local.featureParameters());
        params.add(head);
        params.add(bias);
        Optimizer opt = Optimizer.adam((float) lr);
        for (int s = 0; s < steps; s++) {
            int[] idx = new int[batch];
            float[] y = new float[batch];
            for (int b = 0; b < batch; b++) {
                idx[b] = r.nextInt(views.length);
                y[b] = labels[idx[b]];
            }
            Tensor g = tensor(views, 0, idx);
            Tensor l = tensor(views, 1, idx);
            Tensor yt = Tensor.of(Shape.of(batch, 1), y);
            opt.minimize(params, () -> {
                Var z = logits(g, l);
                return z.softplus().sub(z.mul(Var.of(yt))).mean();
            });
        }
        return this;
    }

    /**
     * @param views {@code [example][global, local][bins]}
     * @return planet log-odds per example
     */
    public double[] scores(double[][][] views) {
        double[] out = new double[views.length];
        int[] idx = new int[views.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Tensor g = tensor(views, 0, idx);
        Tensor l = tensor(views, 1, idx);
        float[] z = Var.noGrad(() -> logits(g, l)).value().toArray();
        for (int i = 0; i < out.length; i++) {
            out[i] = z[i];
        }
        return out;
    }
}
