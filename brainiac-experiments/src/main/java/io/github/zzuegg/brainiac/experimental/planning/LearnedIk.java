package io.github.zzuegg.brainiac.experimental.planning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Learned multi-solution inverse kinematics in the style of IKFlow (Ames et al., ICRA 2022,
 * <a href="https://arxiv.org/abs/2111.08933">arXiv:2111.08933</a>): a conditional normalising
 * flow of affine coupling layers (RealNVP) maps joint configurations to a Gaussian latent,
 * conditioned on the end-effector target, and is trained by maximum likelihood on random
 * configurations and their forward kinematics. Sampling the latent and inverting the flow gives
 * many different solutions for one target in one batched pass, covering the solution branches
 * (elbow up and down, redundant self-motions) that a regressor averages away. One damped
 * least-squares step ({@link InverseKinematics#refine}) then removes the residual error.
 *
 * <p>Joint angles are divided by pi inside the flow. Each coupling transforms the joints outside its mask: {@code y = x exp(s) + t} with
 * {@code s = scale * tanh(.)} and {@code (s, t)} from an MLP of the masked joints and the
 * target.
 */
public final class LearnedIk {

    private final InverseKinematics arm;
    private final int dims;
    private final List<SimpleMlp> nets = new ArrayList<>();
    private final List<float[]> masks = new ArrayList<>();
    private final float scale = 2f;

    /**
     * Creates an untrained flow.
     *
     * @param arm the arm (its joint limits define the training distribution)
     * @param couplings the number of coupling layers
     * @param hidden the hidden width
     * @param seed the seed
     */
    public LearnedIk(InverseKinematics arm, int couplings, int hidden, long seed) {
        this.arm = arm;
        this.dims = arm.joints();
        for (int k = 0; k < couplings; k++) {
            float[] m = new float[dims];
            for (int i = 0; i < dims; i++) {
                m[i] = (i + k) % 2 == 0 ? 1 : 0;
            }
            masks.add(m);
            nets.add(SimpleMlp.tanh(seed + k, dims + 2, hidden, hidden, 2 * dims));
        }
    }

    private List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (SimpleMlp n : nets) {
            p.addAll(n.parameters());
        }
        return p;
    }

    private Var[] st(int k, Var masked, Var cond) {
        int b = masked.shape().dim(0);
        Var out = nets.get(k).forward(Var.concat(1, masked, cond));
        Var s = out.slice(1, 0, dims).tanh().mul(scale);
        Var t = out.slice(1, dims, 2 * dims);
        Tensor inv = Tensor.of(Shape.of(1, dims), invert(masks.get(k)));
        return new Var[] {s.mul(inv.expand(Shape.of(b, dims))), t.mul(inv.expand(Shape.of(b, dims)))};
    }

    private static float[] invert(float[] m) {
        float[] o = new float[m.length];
        for (int i = 0; i < m.length; i++) {
            o[i] = 1 - m[i];
        }
        return o;
    }

    /**
     * Trains by maximum likelihood on uniformly sampled configurations.
     *
     * @param steps gradient steps
     * @param batch configurations per step
     * @param learningRate Adam's rate
     * @param seed the seed
     * @return the last negative log-likelihood per sample
     */
    public double fit(int steps, int batch, float learningRate, long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        double[] lo = arm.lower();
        double[] hi = arm.upper();
        List<Var> params = parameters();
        double last = 0;
        for (int s = 0; s < steps; s++) {
            float[] x = new float[batch * dims];
            float[] c = new float[batch * 2];
            for (int i = 0; i < batch; i++) {
                double[] q = new double[dims];
                for (int j = 0; j < dims; j++) {
                    q[j] = lo[j] + (hi[j] - lo[j]) * r.nextDouble();
                    x[i * dims + j] = (float) (q[j] / Math.PI);
                }
                double[] e = arm.endEffector(q);
                c[2 * i] = (float) e[0];
                c[2 * i + 1] = (float) e[1];
            }
            Var z = Var.of(Tensor.of(Shape.of(batch, dims), x));
            Var cond = Var.of(Tensor.of(Shape.of(batch, 2), c));
            Var logDet = null;
            for (int k = 0; k < nets.size(); k++) {
                Tensor m = Tensor.of(Shape.of(1, dims), masks.get(k)).expand(Shape.of(batch, dims));
                Var masked = z.mul(m);
                Var[] p = st(k, masked, cond);
                z = masked.add(z.mul(p[0].exp()).add(p[1]).mul(Tensor.of(Shape.of(1, dims),
                    invert(masks.get(k))).expand(Shape.of(batch, dims))));
                Var ld = p[0].sum(1, false);
                logDet = logDet == null ? ld : logDet.add(ld);
            }
            Var nll = z.square().sum(1, false).mul(0.5f).sub(logDet).mean();
            last = nll.value().item();
            nll.backward();
            opt.step(params);
        }
        return last;
    }

    /**
     * Samples joint configurations for a target.
     *
     * @param target the end-effector target
     * @param count how many
     * @param temperature the latent standard deviation (1 samples the model, below 1 sharpens)
     * @param random the random source
     * @return {@code [count][joints]}
     */
    public double[][] sample(double[] target, int count, double temperature, SplittableRandom random) {
        return Var.noGrad(() -> {
            float[] zz = new float[count * dims];
            float[] c = new float[count * 2];
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < dims; j++) {
                    zz[i * dims + j] = (float) (temperature * random.nextGaussian());
                }
                c[2 * i] = (float) target[0];
                c[2 * i + 1] = (float) target[1];
            }
            Var y = Var.of(Tensor.of(Shape.of(count, dims), zz));
            Var cond = Var.of(Tensor.of(Shape.of(count, 2), c));
            for (int k = nets.size() - 1; k >= 0; k--) {
                Tensor m = Tensor.of(Shape.of(1, dims), masks.get(k)).expand(Shape.of(count, dims));
                Tensor inv = Tensor.of(Shape.of(1, dims), invert(masks.get(k))).expand(Shape.of(count, dims));
                Var masked = y.mul(m);
                Var[] p = st(k, masked, cond);
                y = masked.add(y.sub(p[1]).mul(p[0].neg().exp()).mul(inv));
            }
            float[] v = y.value().toArray();
            double[][] out = new double[count][dims];
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < dims; j++) {
                    out[i][j] = v[i * dims + j] * Math.PI;
                }
            }
            return out;
        });
    }
}
