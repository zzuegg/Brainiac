package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ResidualVectorQuantizer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A tiny SoundStream-style neural audio codec (<a href="https://arxiv.org/abs/2107.03312">2107.03312</a>)
 * on synthetic waveforms: an encoder turns each frame of samples into a small vector, a
 * {@code ResidualVectorQuantizer} turns that into a few code ids (the tokens an audio
 * language model would predict), and a decoder turns the codes back into samples.
 * Trained with quantizer dropout, one model serves several bitrates: each extra RVQ
 * stage adds {@code log2(codes)} bits per frame and raises the signal-to-noise ratio.
 */
public final class ToyAudioCodec {
    private final int frame;
    private final Model encoder;
    private final Model decoder;
    private final ResidualVectorQuantizer rvq;
    private final Optimizer optimizer = Optimizer.adam(2e-3f);
    private final SplittableRandom random;

    private ToyAudioCodec(int frame, int latent, int stages, int codes, long seed) {
        this.frame = frame;
        this.random = new SplittableRandom(seed);
        encoder = Model.sequential().input(frame).dense(96, Activation.SILU).dense(latent).seed(seed).build();
        decoder = Model.sequential().input(latent).dense(96, Activation.SILU).dense(frame).seed(seed + 1).build();
        rvq = ResidualVectorQuantizer.of(stages, codes, latent, 0.25f, seed + 2);
    }

    /**
     * Creates an untrained codec.
     *
     * @param frame samples per frame
     * @param latent the size of a frame's vector
     * @param stages the RVQ stages
     * @param codes codes per stage
     * @param seed the seed
     * @return the codec
     */
    public static ToyAudioCodec of(int frame, int latent, int stages, int codes, long seed) {
        return new ToyAudioCodec(frame, latent, stages, codes, seed);
    }

    /**
     * Synthesizes frames of a random "instrument": one or two sinusoids with random period,
     * amplitude and phase, plus a decaying envelope.
     *
     * @param n the number of frames
     * @param random the random source
     * @return frames of samples in about [-1, 1]
     */
    public float[][] frames(int n, SplittableRandom random) {
        float[][] out = new float[n][frame];
        for (int i = 0; i < n; i++) {
            int partials = 1 + random.nextInt(2);
            double decay = random.nextDouble() * 0.05;
            for (int p = 0; p < partials; p++) {
                double period = 4 + random.nextDouble() * 28;
                double amp = 0.3 + 0.4 * random.nextDouble();
                double phase = random.nextDouble() * 2 * Math.PI;
                for (int s = 0; s < frame; s++) {
                    out[i][s] += (float) (amp * Math.exp(-decay * s) * Math.sin(2 * Math.PI * s / period + phase));
                }
            }
        }
        return out;
    }

    /**
     * One training step with quantizer dropout.
     *
     * @param batch frames
     * @return the reconstruction loss
     */
    public float trainStep(float[][] batch) {
        Tensor x = Nets.matrix(batch);
        ResidualVectorQuantizer.Result q = rvq.quantizeWithDropout(encoder.forward(Var.of(x), true), random);
        Var rec = decoder.forward(q.output(), true).sub(x).square().mean();
        float value = rec.value().item();
        rec.add(q.loss()).backward();
        List<Var> params = new ArrayList<>(encoder.parameters());
        params.addAll(decoder.parameters());
        params.addAll(rvq.parameters());
        optimizer.step(params);
        return value;
    }

    /**
     * Seeds every RVQ stage from encoder outputs and re-seeds unused codes.
     *
     * @param batch frames
     * @return the codes restarted
     */
    public int restart(float[][] batch) {
        return rvq.restartDeadCodes(encoder.forward(Var.of(Nets.matrix(batch))).value());
    }

    /**
     * Encodes frames to tokens.
     *
     * @param batch frames
     * @param stages how many RVQ stages (the bitrate)
     * @return {@code codes[stage][frame]}
     */
    public int[][] encode(float[][] batch, int stages) {
        return rvq.quantize(encoder.forward(Var.of(Nets.matrix(batch))), stages).codes();
    }

    /**
     * Decodes tokens to frames.
     *
     * @param codes {@code codes[stage][frame]}
     * @return frames
     */
    public float[][] decode(int[][] codes) {
        return Nets.rows(decoder.forward(Var.of(rvq.decode(codes))).value());
    }

    /**
     * The signal-to-noise ratio of a round trip.
     *
     * @param batch frames
     * @param stages how many RVQ stages
     * @return the SNR in decibels
     */
    public double snr(float[][] batch, int stages) {
        float[][] y = decode(encode(batch, stages));
        double signal = 0;
        double noise = 0;
        for (int i = 0; i < batch.length; i++) {
            for (int s = 0; s < frame; s++) {
                signal += batch[i][s] * batch[i][s];
                double e = batch[i][s] - y[i][s];
                noise += e * e;
            }
        }
        return 10 * Math.log10(signal / noise);
    }
}
