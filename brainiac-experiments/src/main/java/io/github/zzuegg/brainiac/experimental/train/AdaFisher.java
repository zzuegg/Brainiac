package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * AdaFisher (Gomes et al., ICLR 2025, <a href="https://arxiv.org/abs/2405.16397">arXiv:2405.16397</a>):
 * Adam with its second moment replaced by a diagonal Kronecker-factored Fisher. For each
 * captured dense layer it keeps running averages of the <em>diagonals</em> of the K-FAC factors,
 * {@code diag(A)} and {@code diag(G)}, whose outer product {@code F = diag(A) diag(G)^T} is the
 * Fisher's diagonal under the Kronecker approximation (the follow-up
 * <a href="https://arxiv.org/abs/2504.20096">arXiv:2504.20096</a> shows the Fisher's energy is
 * concentrated there). The step is the natural-gradient one {@code m_hat / (F + lambda)}, not
 * Adam's square root, at about Adam's cost and memory.
 *
 * <p>Feed it with {@link #updateCurvature} from a {@link KroneckerCapture}, as for {@link Kfac}.
 * Parameters outside captured layers take plain Adam steps.
 */
public final class AdaFisher extends SlotOptimizer {

    private final float beta1;
    private final float beta2;
    private final float damping;
    private final float weightDecay;
    private final Map<Var, Tensor> fisher = new IdentityHashMap<>();
    private final Map<Var, double[]> diagA = new IdentityHashMap<>();
    private final Map<Var, double[]> diagG = new IdentityHashMap<>();

    /**
     * @param learningRate the step size, for example 1e-3
     * @param beta1 the momentum decay, 0.9
     * @param beta2 the decay of the Fisher factors' running averages, 0.99
     * @param damping added to the Fisher diagonal, for example 1e-3
     * @param weightDecay the decoupled weight decay
     */
    public AdaFisher(float learningRate, float beta1, float beta2, float damping, float weightDecay) {
        super(learningRate, "m", "v");
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.damping = damping;
        this.weightDecay = weightDecay;
    }

    /**
     * @param learningRate the step size
     * @return AdaFisher with {@code beta1 = 0.9, beta2 = 0.99, damping = 1e-3}
     */
    public static AdaFisher of(float learningRate) {
        return new AdaFisher(learningRate, 0.9f, 0.99f, 1e-3f, 0);
    }

    /**
     * Folds one batch's factor diagonals into the running averages.
     *
     * @param factors from {@link KroneckerCapture#factors}
     */
    public void updateCurvature(List<KroneckerCapture.Factors> factors) {
        for (KroneckerCapture.Factors f : factors) {
            Var w = f.site().weight();
            double[] a = diagonal(f.a());
            double[] g = diagonal(f.delta());
            diagA.merge(w, a, this::blend);
            diagG.merge(w, g, this::blend);
            double[] da = diagA.get(w);
            double[] dg = diagG.get(w);
            int in = w.shape().dim(0);
            int out = w.shape().dim(1);
            float[] fw = new float[in * out];
            for (int r = 0; r < in; r++) {
                for (int c = 0; c < out; c++) {
                    fw[r * out + c] = (float) (da[r] * dg[c]);
                }
            }
            fisher.put(w, Tensor.of(w.shape(), fw));
            if (f.site().bias() != null) {
                float[] fb = new float[out];
                for (int c = 0; c < out; c++) {
                    fb[c] = (float) (da[in] * dg[c]);
                }
                fisher.put(f.site().bias(), Tensor.of(f.site().bias().shape(), fb));
            }
        }
    }

    /**
     * The Fisher diagonal currently used for a parameter.
     *
     * @param parameter a captured weight or bias
     * @return its diagonal, or {@code null} if none was captured
     */
    public Tensor fisherDiagonal(Var parameter) {
        return fisher.get(parameter);
    }

    private double[] blend(double[] old, double[] now) {
        double[] out = new double[now.length];
        for (int i = 0; i < now.length; i++) {
            out[i] = beta2 * old[i] + (1 - beta2) * now[i];
        }
        return out;
    }

    private static double[] diagonal(double[][] x) {
        double[] d = new double[x[0].length];
        for (double[] row : x) {
            for (int j = 0; j < d.length; j++) {
                d[j] += row[j] * row[j] / x.length;
            }
        }
        return d;
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor m = s[0] == null ? g.mul(1 - beta1) : s[0].mul(beta1).add(g.mul(1 - beta1));
        Tensor mHat = m.div((float) (1 - Math.pow(beta1, t)));
        Tensor f = fisher.get(p);
        Tensor step;
        Tensor v = null;
        if (f != null) {
            step = mHat.div(f.add(damping));
        } else {
            v = s[1] == null ? g.square().mul(1 - beta2) : s[1].mul(beta2).add(g.square().mul(1 - beta2));
            Tensor vHat = v.div((float) (1 - Math.pow(beta2, t)));
            step = mHat.div(vHat.sqrt().add(1e-8f));
        }
        Tensor w = weightDecay > 0 ? p.value().mul(1 - learningRate() * weightDecay) : p.value();
        p.assign(w.sub(step.mul(learningRate())));
        return new Tensor[] {m, v};
    }

    @Override
    public void forget(Var parameter) {
        super.forget(parameter);
        fisher.remove(parameter);
        diagA.remove(parameter);
        diagG.remove(parameter);
    }

    /**
     * Optimizer-state bytes for a parameter list: two float slots per value, as Adam.
     *
     * @param parameters the parameters
     * @return bytes
     */
    public static long stateBytes(List<Var> parameters) {
        long n = 0;
        for (Var p : parameters) {
            n += p.shape().size();
        }
        return n * 8;
    }

    @Override
    public String toString() {
        return "AdaFisher(lr=" + learningRate() + ")";
    }
}
