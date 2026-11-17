package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.Arrays;
import java.util.List;

/**
 * Eligibility traces with the overshooting-bounded gradient step (ObGD) of Elsayed, Vasan and
 * Mahmood, "Streaming Deep Reinforcement Learning Finally Works"
 * (<a href="https://arxiv.org/abs/2410.14606">2410.14606</a>). Streaming learners update on
 * every sample without replay or batches, so one large TD error can wreck the network. ObGD
 * keeps the trace {@code z <- gamma lambda z + grad} and takes the step
 * {@code w <- w + alpha' delta z} with
 * {@code alpha' = alpha / max(1, alpha kappa max(|delta|, 1) |z|_1)}, which bounds how far
 * one update can move the prediction.
 *
 * <p>This is not an {@code Optimizer}: the step needs the TD error, which the core
 * {@code Optimizer.step(List)} cannot receive.
 */
public final class ObgdTraces {

    private final List<Var> parameters;
    private final float[][] traces;
    private float stepSize;
    private final float kappa;

    private ObgdTraces(List<Var> parameters, float stepSize, float kappa) {
        this.parameters = List.copyOf(parameters);
        this.traces = new float[parameters.size()][];
        for (int i = 0; i < traces.length; i++) {
            traces[i] = new float[(int) parameters.get(i).shape().size()];
        }
        this.stepSize = stepSize;
        this.kappa = kappa;
    }

    /**
     * @param parameters the parameters
     * @param stepSize alpha
     * @param kappa the overshooting bound (2 for critics and 3 for actors in the paper)
     * @return zeroed traces
     */
    public static ObgdTraces of(List<Var> parameters, float stepSize, float kappa) {
        return new ObgdTraces(parameters, stepSize, kappa);
    }

    /**
     * @param alpha the step size
     * @return these traces
     */
    public ObgdTraces stepSize(float alpha) {
        this.stepSize = alpha;
        return this;
    }

    /**
     * Decays the traces and adds the parameters' current gradients, which are then cleared.
     *
     * @param decay {@code gamma * lambda}
     */
    public void accumulate(float decay) {
        for (int i = 0; i < traces.length; i++) {
            Var p = parameters.get(i);
            Tensor g = p.grad();
            float[] z = traces[i];
            if (g == null) {
                for (int k = 0; k < z.length; k++) {
                    z[k] *= decay;
                }
                continue;
            }
            float[] gv = g.toArray();
            for (int k = 0; k < z.length; k++) {
                z[k] = decay * z[k] + gv[k];
            }
            p.zeroGrad();
        }
    }

    /**
     * The bounded step {@code w <- w + alpha' delta z}.
     *
     * @param delta the TD error
     * @return the effective step size used
     */
    public float step(float delta) {
        double norm = 0;
        for (float[] z : traces) {
            for (float x : z) {
                norm += Math.abs(x);
            }
        }
        double bound = stepSize * kappa * Math.max(Math.abs(delta), 1) * norm;
        float alpha = (float) (stepSize / Math.max(bound, 1));
        for (int i = 0; i < traces.length; i++) {
            Var p = parameters.get(i);
            float[] w = p.value().toArray();
            float[] z = traces[i];
            for (int k = 0; k < w.length; k++) {
                w[k] += alpha * delta * z[k];
            }
            Shape s = p.shape();
            p.assign(Tensor.of(s, w));
        }
        return alpha;
    }

    /** Zeroes the traces, as at the end of an episode. */
    public void reset() {
        for (float[] z : traces) {
            Arrays.fill(z, 0);
        }
    }
}
