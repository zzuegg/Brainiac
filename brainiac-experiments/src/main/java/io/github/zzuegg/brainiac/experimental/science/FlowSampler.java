package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.DensityFlow;
import io.github.zzuegg.brainiac.nn.Flows;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A flow-based independence sampler for lattice {@code phi^4} theory (Albergo, Kanwar and
 * Shanahan, PRD 2019, <a href="https://arxiv.org/abs/1904.12072">arXiv:1904.12072</a>): a
 * RealNVP normalising flow with checkerboard masks, each affine coupling
 * {@code phi_active <- phi_active * exp(s(phi_frozen)) + t(phi_frozen)} computed by a small
 * network of the frozen sublattice, is trained by the reverse Kullback–Leibler divergence
 * {@code E_q[log q(phi) + S(phi)]} from Gaussian noise (no samples of the target needed). Its
 * samples are then proposals of an independence Metropolis–Hastings chain with acceptance
 * {@code min(1, w(phi') / w(phi))}, {@code w = exp(-S) / q}, which makes the chain exact; the
 * effective sample size of the weights measures the flow.
 *
 * <p>As a {@link DensityFlow} it maps fields to latents (the inverse of the sampling direction)
 * with the core {@link Flows.Result}, so it drops in where the other flows do.
 */
public final class FlowSampler implements DensityFlow {

    private final LatticePhi4 theory;
    private final int volume;
    private final int layers;
    private final List<Var[]> nets = new ArrayList<>();
    private final Tensor[] masks;
    private final Tensor hopping;

    /**
     * An untrained flow.
     *
     * @param theory the target theory
     * @param layers coupling layers (alternating checkerboard parity)
     * @param hidden hidden units of each coupling network
     * @param seed the seed
     */
    public FlowSampler(LatticePhi4 theory, int layers, int hidden, long seed) {
        this.theory = theory;
        this.volume = theory.volume();
        this.layers = layers;
        int l = theory.side();
        SplittableRandom r = new SplittableRandom(seed);
        masks = new Tensor[layers];
        for (int k = 0; k < layers; k++) {
            float[] m = new float[volume];
            for (int y = 0; y < l; y++) {
                for (int x = 0; x < l; x++) {
                    m[y * l + x] = ((x + y + k) & 1) == 0 ? 1 : 0;
                }
            }
            masks[k] = Tensor.of(Shape.of(1, volume), m);
            nets.add(new Var[] {
                dense(volume, hidden, 1.0 / Math.sqrt(volume), r), Var.param(Tensor.zeros(Shape.of(hidden))),
                dense(hidden, 2 * volume, 0.01, r), Var.param(Tensor.zeros(Shape.of(2 * volume)))});
        }
        float[] a = new float[volume * volume];
        for (int y = 0; y < l; y++) {
            for (int x = 0; x < l; x++) {
                int i = y * l + x;
                a[i * volume + y * l + (x + 1) % l] = 1;
                a[i * volume + ((y + 1) % l) * l + x] = 1;
            }
        }
        hopping = Tensor.of(Shape.of(volume, volume), a);
    }

    private static Var dense(int in, int out, double scale, SplittableRandom r) {
        float[] w = new float[in * out];
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) (scale * r.nextGaussian());
        }
        return Var.param(Tensor.of(Shape.of(in, out), w));
    }

    private Var[] coupling(int k, Var frozen) {
        Var[] p = nets.get(k);
        Var h = frozen.matmul(p[0]).add(p[1]).tanh().matmul(p[2]).add(p[3]);
        Var s = h.slice(1, 0, volume).tanh();
        Var t = h.slice(1, volume, 2 * volume);
        return new Var[] {s, t};
    }

    /**
     * The differentiable sampling direction: latents to fields.
     *
     * @param z latents {@code [batch, V]}
     * @return the fields and {@code log |det dphi/dz|} per row
     */
    public Flows.Result push(Var z) {
        Var x = z;
        Var logDet = null;
        for (int k = 0; k < layers; k++) {
            Tensor m = masks[k];
            Tensor active = m.mul(-1f).add(1f);
            Var[] st = coupling(k, x.mul(m));
            Var s = st[0].mul(active);
            x = x.mul(m).add(x.mul(s.exp()).add(st[1]).mul(active));
            Var ld = s.sum(1, false);
            logDet = logDet == null ? ld : logDet.add(ld);
        }
        return new Flows.Result(x, logDet);
    }

    @Override
    public Flows.Result forward(Var x) {
        Var logDet = null;
        for (int k = layers - 1; k >= 0; k--) {
            Tensor m = masks[k];
            Tensor active = m.mul(-1f).add(1f);
            Var[] st = coupling(k, x.mul(m));
            Var s = st[0].mul(active);
            x = x.mul(m).add(x.sub(st[1]).mul(s.neg().exp()).mul(active));
            Var ld = s.sum(1, false).neg();
            logDet = logDet == null ? ld : logDet.add(ld);
        }
        return new Flows.Result(x, logDet);
    }

    @Override
    public Tensor inverse(Tensor z) {
        return Var.noGrad(() -> push(Var.of(z)).output()).value();
    }

    @Override
    public int dimension() {
        return volume;
    }

    @Override
    public List<Var> parameters() {
        List<Var> out = new ArrayList<>();
        for (Var[] p : nets) {
            out.addAll(List.of(p));
        }
        return out;
    }

    private Var action(Var phi) {
        Var p2 = phi.square();
        Var quad = p2.mul((float) (4 + theory.massSquared())).add(p2.square().mul((float) theory.lambda()));
        Var hop = phi.matmul(hopping).mul(phi).mul(2f);
        return quad.sub(hop).sum(1, false);
    }

    private Tensor noise(int batch, SplittableRandom r) {
        float[] z = new float[batch * volume];
        for (int i = 0; i < z.length; i++) {
            z[i] = (float) r.nextGaussian();
        }
        return Tensor.of(Shape.of(batch, volume), z);
    }

    /**
     * Trains by reverse KL with Adam.
     *
     * @param steps gradient steps
     * @param batch samples per step
     * @param learningRate the Adam step
     * @param seed the noise seed
     * @return the final reverse-KL loss (up to {@code log Z})
     */
    public double train(int steps, int batch, double learningRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam((float) learningRate);
        double last = 0;
        for (int s = 0; s < steps; s++) {
            Tensor z = noise(batch, r);
            Var l = opt.minimize(parameters(), () -> {
                Var zv = Var.of(z);
                Flows.Result fr = push(zv);
                Var logq = zv.square().sum(1, false).mul(-0.5f).sub(fr.logDet());
                return logq.add(action(fr.output())).mean();
            });
            last = l.value().item();
        }
        return last;
    }

    /**
     * Proposals with their log importance weights {@code -S(phi) - log q(phi)} (up to a
     * constant).
     *
     * @param count how many
     * @param random the noise source
     * @return {fields [count][V], log weights [count]} packed as {@code double[count][V + 1]}
     */
    public double[][] propose(int count, SplittableRandom random) {
        Tensor z = noise(count, random);
        float[][] res = Var.noGrad(() -> {
            Flows.Result fr = push(Var.of(z));
            return new float[][] {fr.output().value().toArray(), fr.logDet().value().toArray()};
        });
        float[] zz = z.toArray();
        double[][] out = new double[count][volume + 1];
        for (int i = 0; i < count; i++) {
            double logq = 0;
            for (int j = 0; j < volume; j++) {
                out[i][j] = res[0][i * volume + j];
                logq -= 0.5 * zz[i * volume + j] * zz[i * volume + j];
            }
            logq -= res[1][i];
            out[i][volume] = -theory.action(java.util.Arrays.copyOf(out[i], volume)) - logq;
        }
        return out;
    }

    /**
     * The result of an independence Metropolis–Hastings chain.
     *
     * @param magnetizations per-site magnetisation of each chain state
     * @param acceptance the acceptance rate
     * @param essPerSample the effective sample size of the importance weights over the proposals
     */
    public record Chain(double[] magnetizations, double acceptance, double essPerSample) {
    }

    /**
     * Runs an independence Metropolis–Hastings chain on flow proposals.
     *
     * @param steps chain steps (one proposal each)
     * @param random the noise source
     * @return the chain
     */
    public Chain sample(int steps, SplittableRandom random) {
        double[][] prop = propose(steps, random);
        double[] m = new double[steps];
        double[] cur = prop[0];
        int acc = 0;
        double maxW = Double.NEGATIVE_INFINITY;
        for (double[] p : prop) {
            maxW = Math.max(maxW, p[volume]);
        }
        double sw = 0;
        double sw2 = 0;
        for (int i = 0; i < steps; i++) {
            double[] p = prop[i];
            double w = Math.exp(p[volume] - maxW);
            sw += w;
            sw2 += w * w;
            if (i > 0 && Math.log(random.nextDouble()) < p[volume] - cur[volume]) {
                cur = p;
                acc++;
            }
            m[i] = LatticePhi4.magnetization(java.util.Arrays.copyOf(cur, volume));
        }
        return new Chain(m, acc / (double) (steps - 1), sw * sw / sw2 / steps);
    }
}
