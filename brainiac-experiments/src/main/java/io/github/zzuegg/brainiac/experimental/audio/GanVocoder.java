package io.github.zzuegg.brainiac.experimental.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A trainable HiFi-GAN-lite vocoder (Kong et al., NeurIPS 2020,
 * <a href="https://arxiv.org/abs/2010.05646">arXiv:2010.05646</a>), log-mels to waveform.
 *
 * <p>The generator is a 1-D conv stem, then one transposed-convolution upsampling stage per
 * factor of the hop (8, 4, 4 for hop 128), each followed by a multi-receptive-field residual block
 * (dilations 1 and 3), and a tanh output conv. It is trained with the least-squares GAN loss
 * against two discriminator families, plus feature matching (weight 2) and an L1 log-mel loss
 * (weight 45), as in the paper:
 * <ul>
 * <li>the multi-period discriminator (MPD): the waveform reshaped to {@code [samples / p, p]} for
 * periods 2, 3, 5, 7 and 11, convolved with {@code (k, 1)} kernels, so each sub-discriminator sees
 * one phase of a periodic structure;
 * <li>the multi-scale discriminator (MSD): 1-D conv stacks on the raw waveform and on its 2x and
 * 4x average-pooled versions.
 * </ul>
 * {@link #periods} with no periods drops the MPD, {@link #scales(int)} with 0 drops the MSD, and
 * {@link #adversarial(boolean)} off trains on the mel loss alone. The core {@code HifiGanBlock}
 * only runs pretrained generators; this class trains one.
 */
public final class GanVocoder {

    private static final float FEATURE_WEIGHT = 2f;
    private static final float MEL_WEIGHT = 45f;

    private final MelFeatures mel;
    private final SplittableRandom random;
    private final Var stem;
    private final Var stemBias;
    private final List<Var[]> stages = new ArrayList<>();
    private final int[] factors;
    private final Var head;
    private final Var headBias;
    private final List<Discriminator> discriminators = new ArrayList<>();
    private int[] periods = {2, 3, 5, 7, 11};
    private int scales = 3;
    private boolean adversarial = true;
    private int warmup;
    private int trained;
    private final Optimizer generatorOptimizer = Optimizer.adam(2e-3f);
    private final Optimizer discriminatorOptimizer = Optimizer.adam(2e-3f);

    /** One sub-discriminator: conv layers, each a {weight, bias, stride}; period 0 means a scale one. */
    private static final class Discriminator {
        final int period;
        final int pool;
        final List<Var[]> layers = new ArrayList<>();
        final List<Integer> strides = new ArrayList<>();

        Discriminator(int period, int pool) {
            this.period = period;
            this.pool = pool;
        }

        void add(Var w, Var b, int stride) {
            layers.add(new Var[] {w, b});
            strides.add(stride);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            for (Var[] l : layers) {
                p.addAll(List.of(l));
            }
            return p;
        }

        /** Returns the feature map of every layer; the last one is the score map. */
        List<Var> apply(Var wave) {
            int b = wave.shape().dim(0);
            int n = wave.shape().dim(1);
            Var h;
            if (period > 0) {
                int pad = (period - n % period) % period;
                Var x = pad == 0 ? wave : Var.concat(1, wave, Var.of(Tensor.zeros(Shape.of(b, pad))));
                h = x.reshape(b, 1, (n + pad) / period, period);
            } else {
                int len = n / pool * pool;
                h = wave.slice(1, 0, len).reshape(b, len / pool, pool).mean(2, false).reshape(b, 1, len / pool);
            }
            List<Var> features = new ArrayList<>();
            for (int l = 0; l < layers.size(); l++) {
                Var[] w = layers.get(l);
                h = period > 0 ? AudioNets.conv2(h, w[0], w[1], strides.get(l))
                        : AudioNets.conv(h, w[0], w[1], strides.get(l), 1);
                if (l < layers.size() - 1) {
                    h = AudioNets.leaky(h);
                }
                features.add(h);
            }
            return features;
        }
    }

    /**
     * Creates the vocoder.
     *
     * @param mel the front end; its hop must be a product of the upsampling factors
     * @param channels the stem width (halved at each upsampling stage)
     * @param factors the upsampling factors, whose product is the hop (8, 4, 4 for hop 128)
     * @param seed the seed
     */
    public GanVocoder(MelFeatures mel, int channels, int[] factors, long seed) {
        int product = 1;
        for (int f : factors) {
            product *= f;
        }
        if (product != mel.hop()) {
            throw new IllegalArgumentException("upsampling factors must multiply to the hop " + mel.hop());
        }
        this.mel = mel;
        this.factors = factors.clone();
        this.random = new SplittableRandom(seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        stem = AudioNets.kernel(r, channels, mel.mels(), 7);
        stemBias = AudioNets.zeros(channels);
        int c = channels;
        for (int f : factors) {
            int next = Math.max(4, c / 2);
            stages.add(new Var[] {
                Var.param(Tensor.normal(Shape.of(c, next, 2 * f), 0, (float) Math.sqrt(1.0 / (c * 2)), r)),
                AudioNets.zeros(next),
                AudioNets.kernel(r, next, next, 3), AudioNets.zeros(next),
                AudioNets.kernel(r, next, next, 3), AudioNets.zeros(next)});
            c = next;
        }
        head = AudioNets.kernel(r, 1, c, 7);
        headBias = AudioNets.zeros(1);
        buildDiscriminators(seed + 2);
    }

    /**
     * Creates the HiFi-GAN-lite vocoder for {@link MelFeatures#speech()} (hop 128 = 8 x 4 x 4).
     *
     * @param mel the front end
     * @param channels the stem width
     * @param seed the seed
     * @return the vocoder
     */
    public static GanVocoder of(MelFeatures mel, int channels, long seed) {
        return new GanVocoder(mel, channels, new int[] {8, 4, 4}, seed);
    }

    private void buildDiscriminators(long seed) {
        discriminators.clear();
        SplittableRandom r = new SplittableRandom(seed);
        for (int p : periods) {
            Discriminator d = new Discriminator(p, 0);
            d.add(AudioNets.kernel2(r, 8, 1, 5), AudioNets.zeros(8), 3);
            d.add(AudioNets.kernel2(r, 16, 8, 5), AudioNets.zeros(16), 3);
            d.add(AudioNets.kernel2(r, 16, 16, 5), AudioNets.zeros(16), 1);
            d.add(AudioNets.kernel2(r, 1, 16, 3), AudioNets.zeros(1), 1);
            discriminators.add(d);
        }
        for (int s = 0; s < scales; s++) {
            Discriminator d = new Discriminator(0, 1 << s);
            d.add(AudioNets.kernel(r, 8, 1, 15), AudioNets.zeros(8), 1);
            d.add(AudioNets.kernel(r, 16, 8, 11), AudioNets.zeros(16), 4);
            d.add(AudioNets.kernel(r, 16, 16, 11), AudioNets.zeros(16), 4);
            d.add(AudioNets.kernel(r, 1, 16, 3), AudioNets.zeros(1), 1);
            discriminators.add(d);
        }
    }

    /**
     * Sets the MPD periods (none drops the MPD) and rebuilds the discriminators.
     *
     * @param periods the periods
     * @return this vocoder
     */
    public GanVocoder periods(int... periods) {
        this.periods = periods.clone();
        buildDiscriminators(random.nextLong());
        return this;
    }

    /**
     * Sets the MSD scales (0 drops the MSD) and rebuilds the discriminators.
     *
     * @param scales the number of scales (raw, 2x pooled, 4x pooled, ...)
     * @return this vocoder
     */
    public GanVocoder scales(int scales) {
        this.scales = scales;
        buildDiscriminators(random.nextLong());
        return this;
    }

    /**
     * Switches the adversarial and feature-matching losses on or off (off: mel loss only).
     *
     * @param on whether to train adversarially
     * @return this vocoder
     */
    public GanVocoder adversarial(boolean on) {
        this.adversarial = on;
        return this;
    }

    /**
     * Mel-loss warm-up: the first {@code steps} calls of {@link #step} train the generator on
     * the mel-spectrogram L1 loss alone, then the discriminators switch on (0, the default, is
     * wave 9's behaviour).
     *
     * @param steps warm-up steps
     * @return this vocoder
     */
    public GanVocoder melWarmup(int steps) {
        this.warmup = steps;
        return this;
    }

    /** @return the training steps taken (including those of a loaded checkpoint) */
    public int trainedSteps() {
        return trained;
    }

    /**
     * Saves the generator and discriminator weights and the step counter (not the optimiser
     * state).
     *
     * @param file the file
     */
    public void save(java.nio.file.Path file) {
        List<Var> all = new ArrayList<>(generatorParameters());
        all.addAll(discriminatorParameters());
        ParameterFiles.write(file, all, trained);
    }

    /**
     * Loads weights saved by {@link #save} from a vocoder of the same configuration.
     *
     * @param file the file
     * @return this vocoder
     */
    public GanVocoder load(java.nio.file.Path file) {
        List<Var> all = new ArrayList<>(generatorParameters());
        all.addAll(discriminatorParameters());
        trained = ParameterFiles.read(file, all);
        return this;
    }

    /**
     * The generator.
     *
     * @param logMel {@code [batch, frames, mels]}
     * @return {@code [batch, frames * hop]}
     */
    public Var generate(Var logMel) {
        int b = logMel.shape().dim(0);
        int frames = logMel.shape().dim(1);
        Var h = AudioNets.conv(logMel.permute(0, 2, 1).add(5f).mul(0.2f), stem, stemBias, 1, 1);
        for (int s = 0; s < stages.size(); s++) {
            Var[] w = stages.get(s);
            int f = factors[s];
            h = Functional.convTranspose(AudioNets.leaky(h), w[0], w[1], new int[] {f}, new int[] {f / 2},
                    new int[] {f - f / 2}, new int[] {0}, new int[] {1}, 1);
            h = h.add(AudioNets.conv(AudioNets.leaky(h), w[2], w[3], 1, 1));
            h = h.add(AudioNets.conv(AudioNets.leaky(h), w[4], w[5], 1, 3));
        }
        return AudioNets.conv(AudioNets.leaky(h), head, headBias, 1, 1).tanh().reshape(b, frames * mel.hop());
    }

    /**
     * Vocodes one utterance.
     *
     * @param logMel {@code [frames][mels]}
     * @param samples the length to return
     * @return the waveform
     */
    public float[] vocode(float[][] logMel, int samples) {
        Tensor x = AudioNets.cube(new float[][][] {logMel});
        float[] y = Var.noGrad(() -> generate(Var.of(x))).value().toArray();
        return java.util.Arrays.copyOf(y, samples);
    }

    private static Var squaredFrom(Var x, float target) {
        return x.sub(target).square().mean();
    }

    /**
     * One training step on waveform crops: a discriminator step (when adversarial and past the
     * {@link #melWarmup}), then a generator step.
     *
     * @param crops {@code [batch][samples]}, samples a multiple of the hop
     * @return {discriminator loss, generator adversarial + feature loss, mel L1 loss}
     */
    public float[] step(float[][] crops) {
        int b = crops.length;
        int n = crops[0].length;
        Var real = Var.of(AudioNets.matrix(crops));
        Tensor realMel = Var.noGrad(() -> mel.logMel(real)).value();
        Var fake = generate(Var.of(realMel)).slice(1, 0, n);
        List<Var> dParams = discriminatorParameters();
        float dLoss = 0;
        Var gAdv = null;
        boolean adv = adversarial && trained >= warmup;
        trained++;
        if (adv && !discriminators.isEmpty()) {
            Var fakeDetached = Var.of(fake.value());
            Var d = null;
            for (Discriminator disc : discriminators) {
                List<Var> r = disc.apply(real);
                List<Var> f = disc.apply(fakeDetached);
                Var term = squaredFrom(r.get(r.size() - 1), 1).add(f.get(f.size() - 1).square().mean());
                d = d == null ? term : d.add(term);
            }
            d.backward();
            dLoss = d.value().item();
            discriminatorOptimizer.step(dParams);
            for (Var p : dParams) {
                p.zeroGrad();
            }
            for (Discriminator disc : discriminators) {
                List<Var> r = Var.noGrad(() -> disc.apply(real));
                List<Var> f = disc.apply(fake);
                Var term = squaredFrom(f.get(f.size() - 1), 1);
                for (int l = 0; l < f.size() - 1; l++) {
                    term = term.add(f.get(l).sub(Var.of(r.get(l).value())).abs().mean().mul(FEATURE_WEIGHT));
                }
                gAdv = gAdv == null ? term : gAdv.add(term);
            }
        }
        Var melLoss = mel.logMel(fake).sub(Var.of(realMel)).abs().mean();
        Var g = melLoss.mul(MEL_WEIGHT);
        if (gAdv != null) {
            g = g.add(gAdv);
        }
        g.backward();
        for (Var p : dParams) {
            p.zeroGrad();
        }
        generatorOptimizer.step(generatorParameters());
        return new float[] {dLoss, gAdv == null ? 0 : gAdv.value().item(), melLoss.value().item()};
    }

    /**
     * Random crops of whole waveforms, for {@link #step}.
     *
     * @param waves the waveforms
     * @param batch the crops
     * @param frames the crop length in hops
     * @return {@code [batch][frames * hop]}
     */
    public float[][] crops(List<float[]> waves, int batch, int frames) {
        int n = frames * mel.hop();
        float[][] out = new float[batch][n];
        for (int i = 0; i < batch; i++) {
            float[] w = waves.get(random.nextInt(waves.size()));
            int start = w.length > n ? random.nextInt(w.length - n) : 0;
            System.arraycopy(w, start, out[i], 0, Math.min(n, w.length - start));
        }
        return out;
    }

    /** @return the generator's parameters */
    public List<Var> generatorParameters() {
        List<Var> p = new ArrayList<>(List.of(stem, stemBias));
        for (Var[] s : stages) {
            p.addAll(List.of(s));
        }
        p.add(head);
        p.add(headBias);
        return p;
    }

    /** @return the discriminators' parameters */
    public List<Var> discriminatorParameters() {
        List<Var> p = new ArrayList<>();
        for (Discriminator d : discriminators) {
            p.addAll(d.parameters());
        }
        return p;
    }

    /** @return the front end */
    public MelFeatures features() {
        return mel;
    }
}
