package io.github.zzuegg.brainiac.experimental.comms;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ChannelCodes;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * An end-to-end communication system learned as an autoencoder: a dense transmitter maps one of
 * {@code M} messages (one-hot) to {@code n} real channel uses, an energy normalisation enforces
 * {@code ||x||^2 = n}, a channel layer adds noise (and fading), and a dense receiver classifies
 * the message; everything trains jointly on the cross-entropy, so constellations and codes are
 * learned together. With {@code n = 2} the transmitter output is a learned 2-D constellation.
 *
 * <p>The channels are {@link Layer}s ({@link AwgnChannel}, {@link RayleighChannel}), so they
 * drop into any model as stochastic layers. O'Shea and Hoydis, "An introduction to deep
 * learning for the physical layer" (IEEE TCCN 2017),
 * <a href="https://arxiv.org/abs/1702.00832">arXiv:1702.00832</a>.
 */
public final class CommunicationAutoencoder {

    /** Additive white Gaussian noise of a fixed standard deviation, applied in every mode. */
    public static final class AwgnChannel implements Layer {
        private double sigma;
        private final SplittableRandom rnd;

        /**
         * Creates the channel.
         *
         * @param sigma the noise standard deviation per real dimension
         * @param seed the noise seed
         */
        public AwgnChannel(double sigma, long seed) {
            this.sigma = sigma;
            this.rnd = new SplittableRandom(seed);
        }

        /**
         * Changes the noise level.
         *
         * @param sigma the new standard deviation
         */
        public void sigma(double sigma) {
            this.sigma = sigma;
        }

        @Override
        public Var forward(Var input) {
            float[] noise = new float[(int) input.value().size()];
            for (int i = 0; i < noise.length; i++) {
                noise[i] = (float) (sigma * ChannelCodes.gaussian(rnd));
            }
            return input.add(Tensor.of(input.shape(), noise));
        }
    }

    /**
     * Flat Rayleigh fading per real channel use plus AWGN; the receiver gets the faded noisy
     * samples and the gains (coherent detection), so the output is twice as wide.
     */
    public static final class RayleighChannel implements Layer {
        private double sigma;
        private final SplittableRandom rnd;

        /**
         * Creates the channel.
         *
         * @param sigma the noise standard deviation
         * @param seed the seed
         */
        public RayleighChannel(double sigma, long seed) {
            this.sigma = sigma;
            this.rnd = new SplittableRandom(seed);
        }

        /**
         * Changes the noise level.
         *
         * @param sigma the new standard deviation
         */
        public void sigma(double sigma) {
            this.sigma = sigma;
        }

        @Override
        public Var forward(Var input) {
            int size = (int) input.value().size();
            float[] h = new float[size];
            float[] noise = new float[size];
            for (int i = 0; i < size; i++) {
                double a = ChannelCodes.gaussian(rnd);
                double b = ChannelCodes.gaussian(rnd);
                h[i] = (float) Math.sqrt((a * a + b * b) / 2);
                noise[i] = (float) (sigma * ChannelCodes.gaussian(rnd));
            }
            Tensor gain = Tensor.of(input.shape(), h);
            return Var.concat(-1, input.mul(gain).add(Tensor.of(input.shape(), noise)), Var.of(gain));
        }

        @Override
        public int outputSize(int inputSize) {
            return 2 * inputSize;
        }
    }

    private final int messages;
    private final int uses;
    private final Model transmitter;
    private final Model receiver;
    private final boolean fading;
    private boolean averagePower;

    /**
     * Creates an untrained system with a per-message energy constraint.
     *
     * @param messages M
     * @param uses n real channel uses
     * @param hidden the hidden width of transmitter and receiver
     * @param fading whether the channel is Rayleigh fading (otherwise AWGN)
     * @param seed the seed
     */
    public CommunicationAutoencoder(int messages, int uses, int hidden, boolean fading, long seed) {
        this.messages = messages;
        this.uses = uses;
        this.fading = fading;
        this.transmitter = Model.sequential().input(messages).dense(hidden, Activation.RELU).dense(uses).seed(seed)
                .build();
        this.receiver = Model.sequential().input(fading ? 2 * uses : uses).dense(hidden, Activation.RELU)
                .dense(hidden, Activation.RELU).dense(messages).seed(seed + 1).build();
    }

    /**
     * Switches to an average-power constraint over the (equiprobable) constellation instead of
     * a per-message energy constraint, so learned 2-D constellations can use several rings, as
     * QAM does; per-message normalisation forces a PSK-like circle.
     *
     * @return this system
     */
    public CommunicationAutoencoder averagePower() {
        this.averagePower = true;
        return this;
    }

    /** @return the bits per message over channel uses, the rate used for {@code Eb/N0} */
    public double rate() {
        return Math.log(messages) / Math.log(2) / uses;
    }

    private Var transmit(Tensor oneHot) {
        Var x = transmitter.forward(Var.of(oneHot));
        if (averagePower) {
            int[] all = new int[messages];
            for (int i = 0; i < messages; i++) {
                all[i] = i;
            }
            Var points = transmitter.forward(Var.of(oneHot(all, messages)));
            Var meanEnergy = points.square().sum().div((float) (messages * uses)).add(1e-9f).sqrt();
            return x.div(meanEnergy);
        }
        Var norm = x.square().sum(1, true).add(1e-9f).sqrt();
        return x.div(norm).mul((float) Math.sqrt(uses));
    }

    private static Tensor oneHot(int[] msg, int m) {
        float[] v = new float[msg.length * m];
        for (int i = 0; i < msg.length; i++) {
            v[i * m + msg[i]] = 1;
        }
        return Tensor.of(Shape.of(msg.length, m), v);
    }

    /**
     * Trains end to end at one {@code Eb/N0}.
     *
     * @param ebN0Db the training {@code Eb/N0} in dB
     * @param steps gradient steps
     * @param batch messages per step
     * @param learningRate the Adam step size
     * @param seed the seed
     * @return the final cross-entropy
     */
    public double train(double ebN0Db, int steps, int batch, float learningRate, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double sigma = ChannelCodes.sigma(ebN0Db, rate());
        Layer channel = fading ? new RayleighChannel(sigma, seed) : new AwgnChannel(sigma, seed);
        Optimizer opt = Optimizer.adam(learningRate);
        List<Var> params = new ArrayList<>(transmitter.parameters());
        params.addAll(receiver.parameters());
        double last = 0;
        for (int s = 0; s < steps; s++) {
            int[] msg = new int[batch];
            float[] ids = new float[batch];
            for (int i = 0; i < batch; i++) {
                msg[i] = rnd.nextInt(messages);
                ids[i] = msg[i];
            }
            Var logits = receiver.forward(channel.forward(transmit(oneHot(msg, messages))));
            Var loss = Functional.crossEntropy(logits, Tensor.of(Shape.of(batch), ids));
            loss.backward();
            last = loss.value().item();
            opt.step(params);
        }
        return last;
    }

    /**
     * The block (message) error rate by simulation.
     *
     * @param ebN0Db the {@code Eb/N0} in dB
     * @param samples messages simulated
     * @param seed the seed
     * @return the share of messages decoded wrongly
     */
    public double blockErrorRate(double ebN0Db, int samples, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double sigma = ChannelCodes.sigma(ebN0Db, rate());
        Layer channel = fading ? new RayleighChannel(sigma, seed) : new AwgnChannel(sigma, seed);
        int errors = 0;
        int batch = 2000;
        for (int done = 0; done < samples; done += batch) {
            int[] msg = new int[batch];
            for (int i = 0; i < batch; i++) {
                msg[i] = rnd.nextInt(messages);
            }
            float[] logits = Var.noGrad(() -> receiver.forward(channel.forward(transmit(oneHot(msg, messages)))))
                    .value().toArray();
            for (int i = 0; i < batch; i++) {
                int best = 0;
                for (int c = 1; c < messages; c++) {
                    if (logits[i * messages + c] > logits[i * messages + best]) {
                        best = c;
                    }
                }
                errors += best != msg[i] ? 1 : 0;
            }
        }
        return errors / (double) (samples / batch * batch);
    }

    /**
     * The transmitted constellation: the normalised codeword of every message.
     *
     * @return {@code [message][use]}
     */
    public double[][] constellation() {
        int[] all = new int[messages];
        for (int i = 0; i < messages; i++) {
            all[i] = i;
        }
        float[] x = Var.noGrad(() -> transmit(oneHot(all, messages))).value().toArray();
        double[][] out = new double[messages][uses];
        for (int i = 0; i < messages; i++) {
            for (int j = 0; j < uses; j++) {
                out[i][j] = x[i * uses + j];
            }
        }
        return out;
    }

    /**
     * The minimum Euclidean distance between points normalised to unit average energy per
     * real dimension.
     *
     * @param points the codewords
     * @return the minimum pairwise distance
     */
    public static double minimumDistance(double[][] points) {
        double energy = 0;
        for (double[] p : points) {
            for (double v : p) {
                energy += v * v;
            }
        }
        double scale = Math.sqrt(points.length * points[0].length / energy);
        double best = Double.POSITIVE_INFINITY;
        for (int i = 0; i < points.length; i++) {
            for (int j = i + 1; j < points.length; j++) {
                double d = 0;
                for (int k = 0; k < points[i].length; k++) {
                    double t = (points[i][k] - points[j][k]) * scale;
                    d += t * t;
                }
                best = Math.min(best, Math.sqrt(d));
            }
        }
        return best;
    }
}
