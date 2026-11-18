package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * K-FAC (Martens and Grosse 2015, <a href="https://arxiv.org/abs/1503.05671">arXiv:1503.05671</a>)
 * and EKFAC (George et al. 2018, <a href="https://arxiv.org/abs/1806.03884">arXiv:1806.03884</a>):
 * natural-gradient steps with each dense layer's Fisher block approximated by the Kronecker
 * product {@code A (x) G} of the input covariance {@code A = E[a a^T]} and the pre-activation
 * gradient covariance {@code G = E[delta delta^T]}. The step for the combined weight
 * {@code [W; b]} is {@code (A + gamma_A I)^-1 grad (G + gamma_G I)^-1}, with factored
 * Tikhonov damping split by the factors' traces (the paper's pi correction), exponential
 * running averages of the factors, momentum and the KL-clipping of Ba et al. (2017) that
 * bounds the step's predicted change of the model's predictions.
 *
 * <p>EKFAC keeps the Kronecker eigenbasis {@code Q_A (x) Q_G} but refits the diagonal in it from
 * per-example gradients, {@code S = E[(Q_A^T a)^2 (Q_G^T delta)^2]}, which is the best diagonal
 * in that basis and closer to the true Fisher than {@code eig(A) (x) eig(G)}.
 *
 * <p>Use: route the dense layers through a {@link KroneckerCapture} during the forward pass,
 * call {@link #updateCurvature} with its {@link KroneckerCapture#factors factors} of a loss on
 * labels sampled from the model, then back-propagate the real loss and {@link #step}.
 * Parameters that are not a captured layer's weight or bias take plain SGD steps.
 */
public final class Kfac implements Optimizer {

    private static final class LayerState {
        Var bias;
        double[][] a;
        double[][] g;
        double[][] aInv;
        double[][] gInv;
        double[][] qa;
        double[][] qg;
        double[][] scale;
        double[][] velocity;
        int updates;
    }

    private float learningRate;
    private final float damping;
    private final float decay;
    private final int inverseEvery;
    private final float momentum;
    private final float klClip;
    private final boolean eigenvalueCorrected;
    private final Map<Var, LayerState> layers = new IdentityHashMap<>();
    private final Map<Var, Var> biasToWeight = new IdentityHashMap<>();

    /**
     * @param learningRate the step size
     * @param damping the Tikhonov damping {@code lambda}
     * @param decay the running-average decay of the factors, for example 0.95
     * @param inverseEvery curvature updates between recomputing inverses or eigenbases
     * @param momentum the momentum on the preconditioned step
     * @param klClip the largest predicted change {@code lr^2 step . grad}, 0 for none
     * @param eigenvalueCorrected whether to refit the eigenvalues (EKFAC)
     */
    public Kfac(float learningRate, float damping, float decay, int inverseEvery, float momentum, float klClip,
            boolean eigenvalueCorrected) {
        this.learningRate = learningRate;
        this.damping = damping;
        this.decay = decay;
        this.inverseEvery = inverseEvery;
        this.momentum = momentum;
        this.klClip = klClip;
        this.eigenvalueCorrected = eigenvalueCorrected;
    }

    /**
     * K-FAC with common settings (decay 0.95, inverses every 5 updates, momentum 0.9, KL clip 1e-3).
     *
     * @param learningRate the step size, for example 0.1
     * @param damping the damping, for example 1e-2
     * @return the optimizer
     */
    public static Kfac kfac(float learningRate, float damping) {
        return new Kfac(learningRate, damping, 0.95f, 5, 0.9f, 1e-3f, false);
    }

    /**
     * EKFAC with the same settings.
     *
     * @param learningRate the step size
     * @param damping the damping
     * @return the optimizer
     */
    public static Kfac ekfac(float learningRate, float damping) {
        return new Kfac(learningRate, damping, 0.95f, 5, 0.9f, 1e-3f, true);
    }

    /**
     * Folds one batch's factors into the running averages.
     *
     * @param factors from {@link KroneckerCapture#factors}
     */
    public void updateCurvature(List<KroneckerCapture.Factors> factors) {
        for (KroneckerCapture.Factors f : factors) {
            Var w = f.site().weight();
            LayerState s = layers.computeIfAbsent(w, k -> new LayerState());
            if (f.site().bias() != null) {
                s.bias = f.site().bias();
                biasToWeight.put(s.bias, w);
            }
            s.a = blend(s.a, f.inputCovariance());
            s.g = blend(s.g, f.gradientCovariance());
            boolean refresh = s.updates % inverseEvery == 0;
            s.updates++;
            if (eigenvalueCorrected) {
                if (refresh || s.qa == null) {
                    s.qa = MlMatrices.eigh(s.a).vectors();
                    s.qg = MlMatrices.eigh(s.g).vectors();
                    s.scale = null;
                }
                s.scale = blend(s.scale, projectedSquares(s.qa, s.qg, f.a(), f.delta()));
            } else if (refresh || s.aInv == null) {
                double pi = Math.sqrt((trace(s.a) / s.a.length) / Math.max(1e-12, trace(s.g) / s.g.length));
                pi = Double.isFinite(pi) && pi > 0 ? pi : 1;
                double root = Math.sqrt(damping);
                s.aInv = inverse(s.a, pi * root);
                s.gInv = inverse(s.g, root / pi);
            }
        }
    }

    private double[][] blend(double[][] old, double[][] now) {
        if (old == null) {
            return now;
        }
        for (int i = 0; i < now.length; i++) {
            for (int j = 0; j < now[i].length; j++) {
                now[i][j] = decay * old[i][j] + (1 - decay) * now[i][j];
            }
        }
        return now;
    }

    private static double[][] projectedSquares(double[][] qa, double[][] qg, double[][] a, double[][] delta) {
        int n = a.length;
        double[][] pa = MlMatrices.multiply(a, qa);
        double[][] pg = MlMatrices.multiply(delta, qg);
        double[][] s = new double[qa.length][qg.length];
        for (int i = 0; i < n; i++) {
            for (int r = 0; r < s.length; r++) {
                double ar = pa[i][r] * pa[i][r] / n;
                for (int c = 0; c < s[r].length; c++) {
                    s[r][c] += ar * pg[i][c] * pg[i][c];
                }
            }
        }
        return s;
    }

    private static double trace(double[][] m) {
        double t = 0;
        for (int i = 0; i < m.length; i++) {
            t += m[i][i];
        }
        return t;
    }

    private static double[][] inverse(double[][] m, double add) {
        double[][] d = new double[m.length][];
        for (int i = 0; i < m.length; i++) {
            d[i] = m[i].clone();
            d[i][i] += add;
        }
        return MlMatrices.choleskyInverse(MlMatrices.cholesky(d));
    }

    /**
     * The preconditioned direction for a combined weight gradient, from the current statistics.
     *
     * @param weight the captured weight
     * @param grad the {@code [in (+1)][out]} gradient of {@code [W; b]}
     * @return the natural-gradient direction, same shape
     */
    public double[][] precondition(Var weight, double[][] grad) {
        LayerState s = layers.get(weight);
        if (s == null) {
            return grad;
        }
        if (!eigenvalueCorrected) {
            return MlMatrices.multiply(MlMatrices.multiply(s.aInv, grad), s.gInv);
        }
        double[][] inner = MlMatrices.multiply(MlMatrices.multiply(MlMatrices.transpose(s.qa), grad), s.qg);
        for (int r = 0; r < inner.length; r++) {
            for (int c = 0; c < inner[r].length; c++) {
                inner[r][c] /= s.scale[r][c] + damping;
            }
        }
        return MlMatrices.multiply(MlMatrices.multiply(s.qa, inner), MlMatrices.transpose(s.qg));
    }

    @Override
    public void step(List<Var> parameters) {
        Map<Var, double[][]> grads = new IdentityHashMap<>();
        Map<Var, double[][]> dirs = new IdentityHashMap<>();
        double predicted = 0;
        for (Var p : parameters) {
            LayerState s = layers.get(p);
            if (s == null || p.grad() == null) {
                continue;
            }
            double[][] g = combined(p, s.bias);
            double[][] d = precondition(p, g);
            grads.put(p, g);
            dirs.put(p, d);
            for (int r = 0; r < g.length; r++) {
                for (int c = 0; c < g[r].length; c++) {
                    predicted += d[r][c] * g[r][c];
                }
            }
        }
        double nu = 1;
        if (klClip > 0 && predicted > 0) {
            nu = Math.min(1, Math.sqrt(klClip / (learningRate * learningRate * predicted)));
        }
        for (Map.Entry<Var, double[][]> e : dirs.entrySet()) {
            Var w = e.getKey();
            LayerState s = layers.get(w);
            double[][] d = e.getValue();
            if (s.velocity == null) {
                s.velocity = new double[d.length][d[0].length];
            }
            for (int r = 0; r < d.length; r++) {
                for (int c = 0; c < d[r].length; c++) {
                    s.velocity[r][c] = momentum * s.velocity[r][c] + nu * d[r][c];
                }
            }
            apply(w, s.bias, s.velocity);
        }
        for (Var p : parameters) {
            if (p.grad() != null && !layers.containsKey(p) && !(biasToWeight.containsKey(p)
                    && dirs.containsKey(biasToWeight.get(p)))) {
                p.assign(p.value().sub(p.grad().mul(learningRate)));
            }
        }
        for (Var p : parameters) {
            p.zeroGrad();
        }
    }

    private static double[][] combined(Var w, Var b) {
        int in = w.shape().dim(0);
        int out = w.shape().dim(1);
        float[] gw = w.grad().toArray();
        float[] gb = b == null || b.grad() == null ? new float[out] : b.grad().toArray();
        double[][] g = new double[in + (b == null ? 0 : 1)][out];
        for (int r = 0; r < in; r++) {
            for (int c = 0; c < out; c++) {
                g[r][c] = gw[r * out + c];
            }
        }
        if (b != null) {
            for (int c = 0; c < out; c++) {
                g[in][c] = gb[c];
            }
        }
        return g;
    }

    private void apply(Var w, Var b, double[][] v) {
        int in = w.shape().dim(0);
        int out = w.shape().dim(1);
        float[] dw = new float[in * out];
        for (int r = 0; r < in; r++) {
            for (int c = 0; c < out; c++) {
                dw[r * out + c] = (float) (learningRate * v[r][c]);
            }
        }
        w.assign(w.value().sub(Tensor.of(w.shape(), dw)));
        if (b != null) {
            float[] db = new float[out];
            for (int c = 0; c < out; c++) {
                db[c] = (float) (learningRate * v[in][c]);
            }
            b.assign(b.value().sub(Tensor.of(Shape.of(b.shape().toArray()), db)));
        }
    }

    @Override
    public void forget(Var parameter) {
        layers.remove(parameter);
        biasToWeight.remove(parameter);
    }

    @Override
    public float learningRate() {
        return learningRate;
    }

    @Override
    public void learningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public String toString() {
        return (eigenvalueCorrected ? "EKFAC" : "K-FAC") + "(lr=" + learningRate + ", damping=" + damping + ")";
    }
}
