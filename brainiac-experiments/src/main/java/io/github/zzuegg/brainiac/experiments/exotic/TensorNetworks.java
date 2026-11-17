package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.MatrixProductState;
import io.github.zzuegg.brainiac.experimental.nn.PBitIsingMachine;
import io.github.zzuegg.brainiac.experimental.nn.PixelCnn;
import io.github.zzuegg.brainiac.experiments.generative.ToyImages;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Tensor networks as machine-learning models, with {@link MatrixProductState}:
 *
 * <ul>
 *   <li>{@link #classify}: an MPS classifier on the MNIST-like {@link ToyTasks#digits}
 *       (Stoudenmire and Schwab, <a href="https://arxiv.org/abs/1605.05775">arXiv:1605.05775</a>),
 *       accuracy against bond dimension, and {@link #mlp}, a one-hidden-layer MLP with the
 *       same number of weights;</li>
 *   <li>{@link #bornMachine}: an MPS Born machine (Han et al.,
 *       <a href="https://arxiv.org/abs/1709.01662">arXiv:1709.01662</a>) on the binarized
 *       {@link ToyImages} (77 distinct images, so the best possible NLL is {@code ln 77}),
 *       with its exact NLL and exact samples, against a {@link PixelCnn} (exact NLL,
 *       autoregressive samples), a fully visible Boltzmann machine trained by contrastive
 *       Hebbian learning with {@link PBitIsingMachine} Gibbs sampling (no tractable NLL;
 *       Markov-chain samples) and independent pixels.</li>
 * </ul>
 */
public final class TensorNetworks {

    /**
     * A classifier result.
     *
     * @param model the model
     * @param parameters trained weights
     * @param accuracy held-out accuracy
     */
    public record Classification(String model, int parameters, double accuracy) {
    }

    /**
     * A generative result.
     *
     * @param model the model
     * @param nll the exact mean NLL of the data in nats ({@code NaN} when intractable)
     * @param validShare the share of samples that are exactly one of the real images
     */
    public record Generation(String model, double nll, double validShare) {
    }

    private TensorNetworks() {
    }

    /**
     * Trains an MPS classifier on the digits.
     *
     * @param train training data
     * @param test held-out data
     * @param bond the bond dimension
     * @param steps Adam steps on minibatches of 32
     * @param seed the seed
     * @return the result
     */
    public static Classification classify(ToyTasks.Data train, ToyTasks.Data test, int bond, int steps, long seed) {
        MatrixProductState mps = MatrixProductState.classifier(train.features(), bond, train.classes(), seed);
        Optimizer opt = Optimizer.adam(0.003f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            ToyTasks.Data batch = batch(train, 32, r);
            Var loss = Functional.crossEntropy(mps.forward(Var.of(batch.inputs())), batch.ids());
            loss.backward();
            opt.step(mps.parameters());
        }
        return new Classification("mps chi=" + bond, mps.parameterCount(),
                ToyTasks.accuracy(mps.forward(Var.of(test.inputs())).value(), test.y()));
    }

    /**
     * Trains a one-hidden-layer tanh MLP with about a given number of weights.
     *
     * @param train training data
     * @param test held-out data
     * @param parameters the weight budget
     * @param steps Adam steps on minibatches of 32
     * @param seed the seed
     * @return the result
     */
    public static Classification mlp(ToyTasks.Data train, ToyTasks.Data test, int parameters, int steps, long seed) {
        int in = train.features();
        int out = train.classes();
        int hidden = Math.max(1, (parameters - out) / (in + out + 1));
        Model m = Model.sequential().input(in).dense(hidden, Activation.TANH).dense(out).seed(seed).build();
        Optimizer opt = Optimizer.adam(0.01f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            ToyTasks.Data batch = batch(train, 32, r);
            Var loss = Functional.crossEntropy(m.forward(Var.of(batch.inputs()), true), batch.ids());
            loss.backward();
            opt.step(m.parameters());
        }
        long count = 0;
        for (Var p : m.parameters()) {
            count += p.value().size();
        }
        return new Classification("mlp h=" + hidden, Math.toIntExact(count),
                ToyTasks.accuracy(m.forward(Var.of(test.inputs()), false).value(), test.y()));
    }

    private static ToyTasks.Data batch(ToyTasks.Data d, int n, SplittableRandom r) {
        float[][] x = new float[n][];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            int k = r.nextInt(d.size());
            x[i] = d.x()[k];
            y[i] = d.y()[k];
        }
        return new ToyTasks.Data(x, y, d.classes());
    }

    /** @return every binarized toy image as bits (pixel on = 1) */
    public static int[][] images() {
        List<int[]> out = new ArrayList<>();
        for (int c = 0; c < ToyImages.CLASSES; c++) {
            for (float[] img : ToyImages.all(c)) {
                int[] bits = new int[ToyImages.PIXELS];
                for (int i = 0; i < bits.length; i++) {
                    bits[i] = img[i] > 0 ? 1 : 0;
                }
                out.add(bits);
            }
        }
        return out.toArray(int[][]::new);
    }

    /**
     * Trains an MPS Born machine on the toy images.
     *
     * @param bond the bond dimension
     * @param steps two-site DMRG sweeps
     * @param samples exact samples drawn to score
     * @param seed the seed
     * @return the result
     */
    public static Generation bornMachine(int bond, int steps, int samples, long seed) {
        int[][] data = images();
        MatrixProductState mps = MatrixProductState.bornMachine(ToyImages.PIXELS, 2, bond, seed);
        for (int s = 0; s < steps; s++) {
            mps.dmrgSweep(data, 0.3, 2);
        }
        return new Generation("mps chi=" + bond, mps.nll(data).value().item(),
                validShare(mps.sample(samples, new SplittableRandom(seed))));
    }

    /**
     * Trains a PixelCNN on the toy images.
     *
     * @param steps full-batch Adam steps
     * @param samples samples drawn to score
     * @param seed the seed
     * @return the result
     */
    public static Generation pixelCnn(int steps, int samples, long seed) {
        int[][] data = images();
        PixelCnn cnn = new PixelCnn(16, 2, new SplittableRandom(seed));
        Tensor images = imageTensor(data);
        Optimizer copt = Optimizer.adam(0.01f);
        for (int s = 0; s < steps; s++) {
            Var loss = cnn.nll(images);
            loss.backward();
            copt.step(cnn.parameters());
        }
        float[] cs = cnn.sample(samples, ToyImages.SIDE, ToyImages.SIDE, new SplittableRandom(seed)).toArray();
        int[][] cnnSamples = new int[samples][ToyImages.PIXELS];
        for (int i = 0; i < samples; i++) {
            for (int p = 0; p < ToyImages.PIXELS; p++) {
                cnnSamples[i][p] = cs[i * ToyImages.PIXELS + p] > 0.5f ? 1 : 0;
            }
        }
        return new Generation("pixelcnn", cnn.nll(images).value().item(), validShare(cnnSamples));
    }

    /**
     * A fully visible Boltzmann machine trained with p-bit Gibbs chains; its NLL is intractable.
     *
     * @param updates parameter updates
     * @param samples Gibbs samples drawn to score
     * @param seed the seed
     * @return the result
     */
    public static Generation pbitBoltzmann(int updates, int samples, long seed) {
        return new Generation("p-bit boltzmann", Double.NaN,
                validShare(boltzmann(images(), updates, samples, new SplittableRandom(seed))));
    }

    /**
     * Independent Bernoulli pixels fitted by their means.
     *
     * @param samples samples drawn to score
     * @param seed the seed
     * @return the result
     */
    public static Generation independent(int samples, long seed) {
        int[][] data = images();
        SplittableRandom r = new SplittableRandom(seed);
        double nll = 0;
        double[] p = new double[ToyImages.PIXELS];
        for (int[] x : data) {
            for (int i = 0; i < p.length; i++) {
                p[i] += x[i];
            }
        }
        for (int i = 0; i < p.length; i++) {
            p[i] = Math.min(1 - 1e-6, Math.max(1e-6, p[i] / data.length));
        }
        for (int[] x : data) {
            for (int i = 0; i < p.length; i++) {
                nll -= Math.log(x[i] == 1 ? p[i] : 1 - p[i]);
            }
        }
        int[][] indep = new int[samples][ToyImages.PIXELS];
        for (int[] x : indep) {
            for (int i = 0; i < p.length; i++) {
                x[i] = r.nextDouble() < p[i] ? 1 : 0;
            }
        }
        return new Generation("independent", nll / data.length, validShare(indep));
    }

    private static Tensor imageTensor(int[][] data) {
        float[] v = new float[data.length * ToyImages.PIXELS];
        for (int i = 0; i < data.length; i++) {
            for (int p = 0; p < ToyImages.PIXELS; p++) {
                v[i * ToyImages.PIXELS + p] = data[i][p];
            }
        }
        return Tensor.of(Shape.of(data.length, 1, ToyImages.SIDE, ToyImages.SIDE), v);
    }

    /**
     * A fully visible Boltzmann machine over spins {@code 2x - 1}, trained by matching pair
     * correlations: {@code dJ = <s s>_data - <s s>_model}, with the model term from p-bit Gibbs
     * chains.
     */
    private static int[][] boltzmann(int[][] data, int updates, int samples, SplittableRandom r) {
        int n = ToyImages.PIXELS;
        double[][] j = new double[n][n];
        double[] h = new double[n];
        double[][] dataCorr = new double[n][n];
        double[] dataMean = new double[n];
        for (int[] x : data) {
            for (int a = 0; a < n; a++) {
                dataMean[a] += (2 * x[a] - 1) / (double) data.length;
                for (int b = 0; b < n; b++) {
                    dataCorr[a][b] += (2 * x[a] - 1) * (2 * x[b] - 1) / (double) data.length;
                }
            }
        }
        int chains = 16;
        int[][] state = new int[chains][];
        double lr = 0.05;
        for (int u = 0; u < updates; u++) {
            double[][] corr = new double[n][n];
            double[] mean = new double[n];
            PBitIsingMachine m = PBitIsingMachine.of(j, h);
            for (int c = 0; c < chains; c++) {
                m.randomize(r);
                if (state[c] != null) {
                    for (int i = 0; i < n; i++) {
                        m.clamp(i, state[c][i]);
                    }
                    m.release();
                }
                for (int sweep = 0; sweep < 5; sweep++) {
                    m.sweep(1, r);
                }
                int[] s = m.spins();
                state[c] = s;
                for (int a = 0; a < n; a++) {
                    mean[a] += s[a] / (double) chains;
                    for (int b = 0; b < n; b++) {
                        corr[a][b] += s[a] * s[b] / (double) chains;
                    }
                }
            }
            for (int a = 0; a < n; a++) {
                h[a] += lr * (dataMean[a] - mean[a]);
                for (int b = 0; b < n; b++) {
                    if (a != b) {
                        j[a][b] += lr * (dataCorr[a][b] - corr[a][b]);
                    }
                }
            }
        }
        PBitIsingMachine m = PBitIsingMachine.of(j, h);
        m.randomize(r);
        for (int sweep = 0; sweep < 50; sweep++) {
            m.sweep(1, r);
        }
        int[][] out = new int[samples][];
        for (int i = 0; i < samples; i++) {
            for (int sweep = 0; sweep < 10; sweep++) {
                m.sweep(1, r);
            }
            int[] s = m.spins();
            out[i] = new int[n];
            for (int a = 0; a < n; a++) {
                out[i][a] = s[a] > 0 ? 1 : 0;
            }
        }
        return out;
    }

    /**
     * The share of samples that are exactly a real image.
     *
     * @param samples bit images
     * @return the share
     */
    public static double validShare(int[][] samples) {
        int ok = 0;
        for (int[] s : samples) {
            float[] img = new float[s.length];
            for (int i = 0; i < s.length; i++) {
                img[i] = s[i] == 1 ? 1 : -1;
            }
            ok += ToyImages.nearest(img)[1] == 0 ? 1 : 0;
        }
        return ok / (double) samples.length;
    }
}
