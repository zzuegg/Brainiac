package io.github.zzuegg.brainiac.experimental.comms;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ChannelCodes;
import io.github.zzuegg.brainiac.experimental.signal.DenseMatrices;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Detectors for the real-valued MIMO model {@code y = H x + n} with {@code x} in
 * {@code {-1, +1}^nt} (BPSK per antenna, or QPSK as the real-equivalent system of twice the
 * size) and noise variance {@code sigma^2} per receive dimension.
 *
 * <ul>
 *   <li>{@link Method#ZERO_FORCING}: {@code sign((H^T H)^-1 H^T y)}.</li>
 *   <li>{@link Method#MMSE}: {@code sign((H^T H + sigma^2 I)^-1 H^T y)}.</li>
 *   <li>{@link Method#SIC}: ordered MMSE successive interference cancellation (detect the
 *       stream with the best post-detection SNR, subtract it, repeat).</li>
 *   <li>{@link Method#SPHERE_DECODER}: exact maximum likelihood by Schnorr-Euchner depth-first
 *       search over the QR-triangularised lattice with a shrinking radius; counts visited
 *       nodes (Hassibi and Vikalo, IEEE TSP 2005).</li>
 *   <li>{@link Method#DETNET}: projected gradient descent on {@code ||y - Hx||^2} unfolded into
 *       a network: each layer maps {@code [H^T y, x_k, H^T H x_k, v_k]} through a ReLU layer to
 *       the next estimate (through a learned soft sign) and a carried state, trained over random
 *       channels and SNRs with a loss weighted towards later layers (Samuel, Diskin and Wiesel,
 *       "Deep MIMO detection", <a href="https://arxiv.org/abs/1706.01151">arXiv:1706.01151</a>).</li>
 * </ul>
 */
public final class MimoDetectors {

    /** A detector. */
    public enum Method {
        /** zero-forcing. */
        ZERO_FORCING,
        /** linear MMSE. */
        MMSE,
        /** ordered MMSE-SIC. */
        SIC,
        /** exact ML by sphere decoding. */
        SPHERE_DECODER,
        /** the unfolded DetNet. */
        DETNET
    }

    private final int nt;
    private final int layers;
    private final int hidden;
    private final int state;
    private final List<Var[]> net = new ArrayList<>();
    private boolean trained;
    private long nodesVisited;

    /**
     * Creates detectors for {@code nt} transmit dimensions.
     *
     * @param nt real transmit dimensions
     * @param layers DetNet layers
     * @param seed the DetNet initialisation seed
     */
    public MimoDetectors(int nt, int layers, long seed) {
        this.nt = nt;
        this.layers = layers;
        this.hidden = 4 * nt;
        this.state = 2 * nt;
        SplittableRandom rnd = new SplittableRandom(seed);
        int in = 3 * nt + state;
        for (int l = 0; l < layers; l++) {
            net.add(new Var[] {param(in, hidden, rnd), Var.param(Tensor.zeros(Shape.of(hidden))),
                param(hidden, nt, rnd), Var.param(Tensor.zeros(Shape.of(nt))), param(hidden, state, rnd),
                Var.param(Tensor.zeros(Shape.of(state))), Var.param(Tensor.of(Shape.of(1), 0.5f))});
        }
    }

    private static Var param(int in, int out, SplittableRandom rnd) {
        float[] w = new float[in * out];
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) (ChannelCodes.gaussian(rnd) / Math.sqrt(in));
        }
        return Var.param(Tensor.of(Shape.of(in, out), w));
    }

    /** @return search-tree nodes visited by the sphere decoder since the last reset */
    public long nodesVisited() {
        return nodesVisited;
    }

    /** Resets the sphere decoder's node counter. */
    public void resetNodes() {
        nodesVisited = 0;
    }

    /**
     * Detects the transmitted symbols.
     *
     * @param method the detector
     * @param h the channel {@code [rx][nt]}
     * @param y the received vector
     * @param sigma the noise standard deviation
     * @return the detected symbols in {@code {-1, +1}}
     */
    public double[] detect(Method method, double[][] h, double[] y, double sigma) {
        return switch (method) {
            case ZERO_FORCING -> sign(linear(h, y, 1e-9));
            case MMSE -> sign(linear(h, y, sigma * sigma));
            case SIC -> sic(h, y, sigma);
            case SPHERE_DECODER -> sphere(h, y);
            case DETNET -> detnet(new double[][][] {h}, new double[][] {y})[0];
        };
    }

    private static double[] sign(double[] x) {
        double[] s = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            s[i] = x[i] < 0 ? -1 : 1;
        }
        return s;
    }

    private static double[] linear(double[][] h, double[] y, double ridge) {
        return DenseMatrices.solveSpd(DenseMatrices.gram(h), DenseMatrices.multiplyTransposed(h, y), ridge);
    }

    private double[] sic(double[][] h, double[] y, double sigma) {
        int rx = h.length;
        double[] res = y.clone();
        boolean[] done = new boolean[nt];
        double[] x = new double[nt];
        for (int step = 0; step < nt; step++) {
            List<Integer> active = new ArrayList<>();
            for (int j = 0; j < nt; j++) {
                if (!done[j]) {
                    active.add(j);
                }
            }
            int k = active.size();
            double[][] hs = new double[rx][k];
            for (int i = 0; i < rx; i++) {
                for (int a = 0; a < k; a++) {
                    hs[i][a] = h[i][active.get(a)];
                }
            }
            double[][] g = DenseMatrices.gram(hs);
            // the MMSE error covariance diagonal picks the most reliable stream
            int best = 0;
            double bestErr = Double.POSITIVE_INFINITY;
            double[] est = DenseMatrices.solveSpd(g, DenseMatrices.multiplyTransposed(hs, res), sigma * sigma);
            for (int a = 0; a < k; a++) {
                double[] e = new double[k];
                e[a] = 1;
                double err = DenseMatrices.solveSpd(g, e, sigma * sigma)[a];
                if (err < bestErr) {
                    bestErr = err;
                    best = a;
                }
            }
            int j = active.get(best);
            x[j] = est[best] < 0 ? -1 : 1;
            done[j] = true;
            for (int i = 0; i < rx; i++) {
                res[i] -= h[i][j] * x[j];
            }
        }
        return x;
    }

    private double[] sphere(double[][] h, double[] y) {
        double[][][] qr = DenseMatrices.qr(h);
        double[][] q = qr[0];
        double[][] r = qr[1];
        double[] z = DenseMatrices.multiplyTransposed(q, y);
        double[] best = sign(DenseMatrices.solveSpd(DenseMatrices.gram(h), DenseMatrices.multiplyTransposed(h, y), 1e-9));
        // start the radius at the Babai (ZF) point's distance
        double radius = 0;
        for (int i = 0; i < nt; i++) {
            double s = z[i];
            for (int j = i; j < nt; j++) {
                s -= r[i][j] * best[j];
            }
            radius += s * s;
        }
        double[] x = new double[nt];
        double[] bestOut = best.clone();
        double[] radiusBox = {radius + 1e-12};
        search(r, z, nt - 1, 0, x, bestOut, radiusBox);
        return bestOut;
    }

    private void search(double[][] r, double[] z, int level, double partial, double[] x, double[] best,
            double[] radius) {
        nodesVisited++;
        double s = z[level];
        for (int j = level + 1; j < nt; j++) {
            s -= r[level][j] * x[j];
        }
        double centre = s / r[level][level];
        double first = centre < 0 ? -1 : 1;
        for (double cand : new double[] {first, -first}) {
            double d = s - r[level][level] * cand;
            double dist = partial + d * d;
            if (dist >= radius[0]) {
                continue;
            }
            x[level] = cand;
            if (level == 0) {
                radius[0] = dist;
                System.arraycopy(x, 0, best, 0, nt);
            } else {
                search(r, z, level - 1, dist, x, best, radius);
            }
        }
    }

    private Var forward(Tensor hty, Tensor hth, List<Var> outputs) {
        int batch = hty.shape().dim(0);
        Var x = Var.of(Tensor.zeros(Shape.of(batch, nt)));
        Var v = Var.of(Tensor.zeros(Shape.of(batch, state)));
        Var htyV = Var.of(hty);
        Var hthV = Var.of(hth);
        for (Var[] w : net) {
            Var hthx = hthV.mul(x.reshape(batch, 1, nt)).sum(2, false);
            Var in = Var.concat(1, htyV, x, hthx, v);
            Var zz = in.matmul(w[0]).add(w[1]).relu();
            Var pre = zz.matmul(w[2]).add(w[3]);
            // learned soft sign: -1 + relu(p + t)/|t| - relu(p - t)/|t|
            Var t = w[6].abs().add(0.05f);
            x = pre.add(t).relu().div(t).sub(pre.sub(t).relu().div(t)).sub(1f);
            v = zz.matmul(w[4]).add(w[5]);
            if (outputs != null) {
                outputs.add(x);
            }
        }
        return x;
    }

    private static Tensor[] inputs(double[][][] hs, double[][] ys) {
        int batch = hs.length;
        int nt = hs[0][0].length;
        float[] hty = new float[batch * nt];
        float[] hth = new float[batch * nt * nt];
        for (int b = 0; b < batch; b++) {
            double[] a = DenseMatrices.multiplyTransposed(hs[b], ys[b]);
            double[][] g = DenseMatrices.gram(hs[b]);
            for (int i = 0; i < nt; i++) {
                hty[b * nt + i] = (float) a[i];
                for (int j = 0; j < nt; j++) {
                    hth[(b * nt + i) * nt + j] = (float) g[i][j];
                }
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(batch, nt), hty), Tensor.of(Shape.of(batch, nt, nt), hth)};
    }

    /**
     * DetNet detection of a batch.
     *
     * @param hs channels
     * @param ys received vectors
     * @return detected symbols
     */
    public double[][] detnet(double[][][] hs, double[][] ys) {
        if (!trained) {
            throw new IllegalStateException("train DetNet first");
        }
        Tensor[] in = inputs(hs, ys);
        float[] x = Var.noGrad(() -> forward(in[0], in[1], null)).value().toArray();
        double[][] out = new double[hs.length][nt];
        for (int b = 0; b < hs.length; b++) {
            for (int i = 0; i < nt; i++) {
                out[b][i] = x[b * nt + i] < 0 ? -1 : 1;
            }
        }
        return out;
    }

    /**
     * Trains DetNet over random channels from a generator, SNR uniform in
     * {@code [loDb, hiDb]}, on {@code sum_k log(k + 1) ||x - x_k||^2}.
     *
     * @param channels draws a channel for training
     * @param steps gradient steps
     * @param batch samples per step
     * @param loDb the lowest SNR (dB, per receive dimension: {@code sigma^2 = nt / (rx SNR)})
     * @param hiDb the highest SNR
     * @param learningRate the Adam step size
     * @param seed the seed
     * @return the final loss
     */
    public double trainDetNet(java.util.function.Function<SplittableRandom, double[][]> channels, int steps, int batch,
            double loDb, double hiDb, float learningRate, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(learningRate);
        List<Var> params = new ArrayList<>();
        for (Var[] w : net) {
            params.addAll(List.of(w));
        }
        double last = 0;
        for (int s = 0; s < steps; s++) {
            double[][][] hs = new double[batch][][];
            double[][] ys = new double[batch][];
            float[] xs = new float[batch * nt];
            for (int b = 0; b < batch; b++) {
                hs[b] = channels.apply(rnd);
                double[] x = new double[nt];
                for (int i = 0; i < nt; i++) {
                    x[i] = rnd.nextBoolean() ? 1 : -1;
                    xs[b * nt + i] = (float) x[i];
                }
                double sigma = sigma(loDb + (hiDb - loDb) * rnd.nextDouble(), hs[b].length);
                ys[b] = ChannelCodes.awgn(DenseMatrices.multiply(hs[b], x), sigma, rnd);
            }
            Tensor[] in = inputs(hs, ys);
            List<Var> outs = new ArrayList<>();
            forward(in[0], in[1], outs);
            Tensor target = Tensor.of(Shape.of(batch, nt), xs);
            Var loss = null;
            for (int k = 0; k < outs.size(); k++) {
                Var term = outs.get(k).sub(target).square().mean().mul((float) Math.log(k + 2));
                loss = loss == null ? term : loss.add(term);
            }
            loss.backward();
            last = loss.value().item();
            opt.step(params);
        }
        trained = true;
        return last;
    }

    /**
     * The noise level for an SNR defined as total transmit energy over noise per receive
     * dimension, with columns of unit expected norm ({@code E ||Hx||^2 = nt}).
     *
     * @param snrDb the SNR in dB
     * @param rx receive dimensions
     * @return the noise standard deviation
     */
    public double sigma(double snrDb, int rx) {
        return Math.sqrt(nt / (rx * Math.pow(10, snrDb / 10)));
    }

    /** @return the number of DetNet layers */
    public int layers() {
        return layers;
    }

    /** @return the DetNet hidden width */
    public int hidden() {
        return hidden;
    }
}
