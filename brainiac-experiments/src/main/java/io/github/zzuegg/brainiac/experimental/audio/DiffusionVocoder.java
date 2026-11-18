package io.github.zzuegg.brainiac.experimental.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Diffusion vocoders in the waveform: DiffWave (Kong et al., ICLR 2021,
 * <a href="https://arxiv.org/abs/2009.09761">arXiv:2009.09761</a>) and WaveGrad (Chen et al.,
 * ICLR 2021, <a href="https://arxiv.org/abs/2009.00713">arXiv:2009.00713</a>).
 *
 * <p>The network is DiffWave's: a non-causal stack of dilated convolutions (dilation doubling
 * and cycling) with gated tanh-sigmoid units, residual and skip connections, the mel upsampled to
 * the sample rate (a learned conv over frames, then repetition) and added through a 1x1 conv in
 * every layer, and the noise level embedded sinusoidally and added in every layer. It predicts
 * the noise {@code epsilon} of {@code x_t = sqrt(abar) x_0 + sqrt(1 - abar) epsilon} under a
 * linear beta schedule of {@code T} steps.
 *
 * <p>The two papers differ in the conditioning: {@link Conditioning#DISCRETE} (DiffWave) embeds
 * the step index {@code t}, so sampling must use (a subset of) the training steps;
 * {@link Conditioning#CONTINUOUS} (WaveGrad) embeds the noise scale {@code sqrt(abar)}, drawn
 * uniformly between adjacent training levels, so any schedule can be chosen after training (here
 * evenly spaced in log-SNR). Both sample with the deterministic DDIM update, except that the
 * discrete model run at all {@code T} steps samples ancestrally as DiffWave does.
 */
public final class DiffusionVocoder {

    /** How the noise level is presented to the network. */
    public enum Conditioning {
        /** DiffWave: the integer step. */
        DISCRETE,
        /** WaveGrad: the continuous noise scale. */
        CONTINUOUS
    }

    private static final int EMBED = 32;

    private final MelFeatures mel;
    private final int channels;
    private final int[] dilations;
    private final Conditioning conditioning;
    private final double[] beta;
    private final double[] alphaBar;
    private final SplittableRandom random;
    private final Var input;
    private final Var inputBias;
    private final Var melConv;
    private final Var melBias;
    private final Var embed1;
    private final Var embed1Bias;
    private final List<Var[]> layers = new ArrayList<>();
    private final Var skipOut;
    private final Var skipBias;
    private final Var out;
    private final Optimizer optimizer = Optimizer.adam(2e-3f);

    /**
     * Creates the vocoder.
     *
     * @param mel the front end
     * @param channels the residual channels
     * @param layers the dilated layers (dilations 1, 2, 4, ... cycling every 6)
     * @param steps the diffusion steps {@code T} (50 in DiffWave's base model)
     * @param conditioning discrete (DiffWave) or continuous (WaveGrad) noise conditioning
     * @param seed the seed
     */
    public DiffusionVocoder(MelFeatures mel, int channels, int layers, int steps, Conditioning conditioning,
            long seed) {
        this.mel = mel;
        this.channels = channels;
        this.conditioning = conditioning;
        this.random = new SplittableRandom(seed);
        this.beta = new double[steps];
        this.alphaBar = new double[steps];
        double a = 1;
        for (int t = 0; t < steps; t++) {
            beta[t] = 1e-4 + (0.05 - 1e-4) * t / Math.max(1, steps - 1);
            a *= 1 - beta[t];
            alphaBar[t] = a;
        }
        SplittableRandom r = new SplittableRandom(seed + 1);
        input = AudioNets.kernel(r, channels, 1, 1);
        inputBias = AudioNets.zeros(channels);
        melConv = AudioNets.kernel(r, mel.mels(), mel.mels(), 3);
        melBias = AudioNets.zeros(mel.mels());
        embed1 = AudioNets.dense(r, EMBED, channels);
        embed1Bias = AudioNets.zeros(channels);
        dilations = new int[layers];
        for (int l = 0; l < layers; l++) {
            dilations[l] = 1 << (l % 6);
            this.layers.add(new Var[] {
                AudioNets.dense(r, channels, channels),                   // step projection
                AudioNets.kernel(r, 2 * channels, channels, 3),           // dilated conv
                AudioNets.zeros(2 * channels),
                AudioNets.kernel(r, 2 * channels, mel.mels(), 1),         // conditioner
                AudioNets.kernel(r, 2 * channels, channels, 1),           // output: residual + skip
                AudioNets.zeros(2 * channels)});
        }
        skipOut = AudioNets.kernel(r, channels, channels, 1);
        skipBias = AudioNets.zeros(channels);
        out = Var.param(Tensor.zeros(Shape.of(1, channels, 1)));
    }

    /** @return the training steps {@code T} */
    public int steps() {
        return beta.length;
    }

    /** @return the conditioning */
    public Conditioning conditioning() {
        return conditioning;
    }

    /** The value the network sees for a noise level: the step index, or 100 sqrt(abar). */
    private float levelInput(double t, double abar) {
        return conditioning == Conditioning.DISCRETE ? (float) t : (float) (100 * Math.sqrt(abar));
    }

    /**
     * The noise prediction.
     *
     * @param noisy {@code [batch, samples]}
     * @param levels the level inputs per item ({@link #levelInput})
     * @param logMel {@code [batch, frames, mels]} with {@code frames * hop >= samples}
     * @return {@code [batch, samples]}
     */
    private Var epsilon(Var noisy, float[] levels, Var logMel) {
        int b = noisy.shape().dim(0);
        int n = noisy.shape().dim(1);
        Var m = AudioNets.conv(logMel.permute(0, 2, 1).add(5f).mul(0.2f), melConv, melBias, 1, 1);  // [b, mels, F]
        float[] idx = new float[n];
        for (int i = 0; i < n; i++) {
            idx[i] = Math.min(m.shape().dim(2) - 1, i / mel.hop());
        }
        Var cond = m.gather(2, Tensor.of(Shape.of(n), idx));                                        // [b, mels, n]
        Var e = Var.of(AudioNets.sinusoid(levels, 1f, EMBED)).matmul(embed1).add(embed1Bias);
        e = e.mul(e.sigmoid());                                                                     // [b, C]
        Var x = AudioNets.conv(noisy.reshape(b, 1, n), input, inputBias, 1, 1).relu();
        Var skips = null;
        float norm = (float) Math.sqrt(0.5);
        for (int l = 0; l < layers.size(); l++) {
            Var[] w = layers.get(l);
            Var h = x.add(e.matmul(w[0]).reshape(b, channels, 1));
            Var y = AudioNets.conv(h, w[1], w[2], 1, dilations[l]).add(AudioNets.conv(cond, w[3], null, 1, 1));
            Var gated = y.slice(1, 0, channels).tanh().mul(y.slice(1, channels, 2 * channels).sigmoid());
            Var o = AudioNets.conv(gated, w[4], w[5], 1, 1);
            x = x.add(o.slice(1, 0, channels)).mul(norm);
            Var s = o.slice(1, channels, 2 * channels);
            skips = skips == null ? s : skips.add(s);
        }
        Var h = AudioNets.conv(skips.mul((float) (1 / Math.sqrt(layers.size()))), skipOut, skipBias, 1, 1).relu();
        return AudioNets.conv(h, out, null, 1, 1).reshape(b, n);
    }

    /**
     * One epsilon-prediction step on waveform crops.
     *
     * @param crops {@code [batch][samples]}
     * @return the loss
     */
    public float step(float[][] crops) {
        int b = crops.length;
        int n = crops[0].length;
        Tensor x0 = AudioNets.matrix(crops);
        Tensor logMel = Var.noGrad(() -> mel.logMel(Var.of(x0))).value();
        float[] levels = new float[b];
        float[] sa = new float[b];
        float[] sn = new float[b];
        for (int i = 0; i < b; i++) {
            int t = random.nextInt(beta.length);
            double abar = alphaBar[t];
            if (conditioning == Conditioning.CONTINUOUS) {
                double hi = t == 0 ? 1 : Math.sqrt(alphaBar[t - 1]);
                double s = Math.sqrt(abar) + random.nextDouble() * (hi - Math.sqrt(abar));
                abar = s * s;
            }
            levels[i] = levelInput(t, abar);
            sa[i] = (float) Math.sqrt(abar);
            sn[i] = (float) Math.sqrt(1 - abar);
        }
        Tensor noise = Tensor.normal(Shape.of(b, n), 0, 1, random);
        Tensor noisy = x0.mul(Tensor.of(Shape.of(b, 1), sa)).add(noise.mul(Tensor.of(Shape.of(b, 1), sn)));
        Var loss = epsilon(Var.of(noisy), levels, Var.of(logMel)).sub(Var.of(noise)).square().mean();
        float v = loss.value().item();
        loss.backward();
        optimizer.step(parameters());
        return v;
    }

    /**
     * Vocodes one utterance.
     *
     * @param logMel {@code [frames][mels]}
     * @param samples the length to produce
     * @param steps sampling steps; {@code T} with discrete conditioning samples ancestrally
     * @param seed the noise seed
     * @return the waveform
     */
    public float[] vocode(float[][] logMel, int samples, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int T = beta.length;
        double[] abars = new double[steps];
        double[] tIndex = new double[steps];
        if (conditioning == Conditioning.DISCRETE || steps == T) {
            for (int k = 0; k < steps; k++) {
                int t = steps == 1 ? T - 1 : (int) Math.round((double) k * (T - 1) / (steps - 1));
                tIndex[k] = t;
                abars[k] = alphaBar[t];
            }
        } else {
            double hi = Math.log(alphaBar[0] / (1 - alphaBar[0]));
            double lo = Math.log(alphaBar[T - 1] / (1 - alphaBar[T - 1]));
            for (int k = 0; k < steps; k++) {
                double snr = steps == 1 ? lo : hi + (lo - hi) * k / (steps - 1);
                abars[k] = 1 / (1 + Math.exp(-snr));
                tIndex[k] = k;
            }
        }
        Var cond = Var.of(AudioNets.cube(new float[][][] {logMel}));
        Tensor x = Tensor.normal(Shape.of(1, samples), 0, 1, r);
        boolean ancestral = conditioning == Conditioning.DISCRETE && steps == T;
        for (int k = steps - 1; k >= 0; k--) {
            double abar = abars[k];
            float[] level = {levelInput(tIndex[k], abar)};
            Tensor xt = x;
            Tensor eps = Var.noGrad(() -> epsilon(Var.of(xt), level, cond)).value();
            double prev = k == 0 ? 1 : abars[k - 1];
            if (ancestral) {
                int t = (int) tIndex[k];
                double b = beta[t];
                x = xt.sub(eps.mul(Tensor.scalar((float) (b / Math.sqrt(1 - abar)))))
                        .mul(Tensor.scalar((float) (1 / Math.sqrt(1 - b))));
                if (k > 0) {
                    double sigma = Math.sqrt((1 - prev) / (1 - abar) * b);
                    x = x.add(Tensor.normal(Shape.of(1, samples), 0, (float) sigma, r));
                }
            } else {
                Tensor x0 = xt.sub(eps.mul(Tensor.scalar((float) Math.sqrt(1 - abar))))
                        .mul(Tensor.scalar((float) (1 / Math.sqrt(abar))));
                x0 = Var.of(x0).clamp(-1, 1).value();
                x = x0.mul(Tensor.scalar((float) Math.sqrt(prev)))
                        .add(eps.mul(Tensor.scalar((float) Math.sqrt(1 - prev))));
            }
        }
        float[] y = x.toArray();
        for (int i = 0; i < y.length; i++) {
            y[i] = Math.max(-1, Math.min(1, y[i]));
        }
        return y;
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(input, inputBias, melConv, melBias, embed1, embed1Bias));
        for (Var[] l : layers) {
            p.addAll(List.of(l));
        }
        p.addAll(List.of(skipOut, skipBias, out));
        return p;
    }
}
