package io.github.zzuegg.brainiac.experimental.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Class-conditional audio generation by diffusion in the latent of a mel autoencoder, in the
 * manner of AudioLDM (Liu et al., ICML 2023, <a href="https://arxiv.org/abs/2301.12503">arXiv:2301.12503</a>)
 * and Stable Audio (Evans et al., 2024, <a href="https://arxiv.org/abs/2407.14358">arXiv:2407.14358</a>):
 * a VAE compresses every group of mel frames into a small continuous vector (no quantisation),
 * a rectified-flow denoiser learns the class-conditional distribution of the latents (scaled to
 * unit variance, as latent diffusion does), and samples are decoded to mels and then vocoded.
 *
 * <p>{@link Space#MEL} is the baseline of the same denoiser directly on the normalised mel. The
 * denoiser is an MLP on {@code [x_t, class one-hot, time embedding]} with classifier-free
 * guidance (the class dropped for 10% of the training items).
 */
public final class LatentAudioDiffusion {

    /** Where diffusion runs. */
    public enum Space {
        /** In the VAE latent. */
        LATENT,
        /** On the normalised log-mel. */
        MEL
    }

    private static final float MEAN = -5f;
    private static final float SCALE = 3f;
    private static final int TIME = 16;

    private final Space space;
    private final int frames;
    private final int mels;
    private final int classes;
    private final int latent;
    private final int group;
    private final SimpleMlp encoder;
    private final SimpleMlp decoder;
    private final SimpleMlp denoiser;
    private final SplittableRandom random;
    private final Optimizer aeOptimizer = Optimizer.adam(2e-3f);
    private final Optimizer optimizer = Optimizer.adam(2e-3f);
    private float latentScale = Float.NaN;

    /**
     * Creates the model.
     *
     * @param space latent or mel diffusion
     * @param frames mel frames per clip (a multiple of {@code group})
     * @param mels mel bands
     * @param classes the classes
     * @param latent latent size per group of frames
     * @param group frames per latent vector
     * @param hidden the hidden width of the networks
     * @param seed the seed
     */
    public LatentAudioDiffusion(Space space, int frames, int mels, int classes, int latent, int group, int hidden,
            long seed) {
        if (frames % group != 0) {
            throw new IllegalArgumentException("frames must be a multiple of the group");
        }
        this.space = space;
        this.frames = frames;
        this.mels = mels;
        this.classes = classes;
        this.latent = latent;
        this.group = group;
        this.random = new SplittableRandom(seed);
        encoder = SimpleMlp.of(seed, v -> v.mul(v.sigmoid()), group * mels, hidden, 2 * latent);
        decoder = SimpleMlp.of(seed + 1, v -> v.mul(v.sigmoid()), latent, hidden, group * mels);
        int d = dimension();
        denoiser = SimpleMlp.of(seed + 2, v -> v.mul(v.sigmoid()), d + classes + TIME, 2 * hidden, 2 * hidden, d);
    }

    /** @return the size of the space diffusion runs in */
    public int dimension() {
        return space == Space.LATENT ? frames / group * latent : frames * mels;
    }

    /** @return the space */
    public Space space() {
        return space;
    }

    private Tensor normalised(List<float[][]> clips) {
        int n = clips.size();
        float[] f = new float[n * frames * mels];
        for (int i = 0; i < n; i++) {
            float[][] c = clips.get(i);
            for (int j = 0; j < frames; j++) {
                float[] row = c[Math.min(j, c.length - 1)];
                for (int k = 0; k < mels; k++) {
                    f[(i * frames + j) * mels + k] = (row[k] - MEAN) / SCALE;
                }
            }
        }
        return Tensor.of(Shape.of(n * frames / group, group * mels), f);
    }

    /**
     * One VAE step (latent space only).
     *
     * @param clips log-mels {@code [frames][mels]}
     * @return the reconstruction loss
     */
    public float trainAutoencoder(List<float[][]> clips) {
        Tensor x = normalised(clips);
        Var stats = encoder.forward(Var.of(x));
        Var mu = stats.slice(1, 0, latent);
        Var logVar = stats.slice(1, latent, 2 * latent).clamp(-8, 4);
        Tensor eps = Tensor.normal(mu.shape(), 0, 1, random);
        Var z = mu.add(logVar.mul(0.5f).exp().mul(Var.of(eps)));
        Var rec = decoder.forward(z).sub(Var.of(x)).square().mean();
        Var kl = mu.square().add(logVar.exp()).sub(logVar).sub(1f).mean().mul(0.5f);
        Var loss = rec.add(kl.mul(1e-3f));
        float v = rec.value().item();
        loss.backward();
        List<Var> p = new ArrayList<>(encoder.parameters());
        p.addAll(decoder.parameters());
        aeOptimizer.step(p);
        return v;
    }

    /** Clips to the diffusion space, {@code [n, dimension]} (latent means, unscaled). */
    private Tensor toSpace(List<float[][]> clips) {
        Tensor x = normalised(clips);
        int n = clips.size();
        if (space == Space.MEL) {
            return x.reshape(n, dimension());
        }
        return Var.noGrad(() -> encoder.forward(Var.of(x)).slice(1, 0, latent)).value().reshape(n, dimension());
    }

    private List<float[][]> fromSpace(Tensor y) {
        int n = y.shape().dim(0);
        Tensor x = space == Space.MEL ? y
                : Var.noGrad(() -> decoder.forward(Var.of(y.reshape(n * frames / group, latent)))).value();
        float[] f = x.toArray();
        List<float[][]> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            float[][] c = new float[frames][mels];
            for (int j = 0; j < frames; j++) {
                for (int k = 0; k < mels; k++) {
                    c[j][k] = f[(i * frames + j) * mels + k] * SCALE + MEAN;
                }
            }
            out.add(c);
        }
        return out;
    }

    /**
     * The autoencoder's round trip (identity for mel space).
     *
     * @param clips log-mels
     * @return their reconstructions
     */
    public List<float[][]> reconstruct(List<float[][]> clips) {
        return fromSpace(toSpace(clips));
    }

    private Tensor condition(int[] labels, float[] t, boolean dropAll, double dropRate) {
        int n = labels.length;
        float[] c = new float[n * classes];
        for (int i = 0; i < n; i++) {
            boolean drop = dropAll || random.nextDouble() < dropRate;
            if (!drop) {
                c[i * classes + labels[i]] = 1;
            }
        }
        Tensor time = AudioNets.sinusoid(t, 1000f, TIME);
        return Tensor.concat(1, Tensor.of(Shape.of(n, classes), c), time);
    }

    /**
     * One rectified-flow step of the denoiser (the first call fixes the latent scale).
     *
     * @param clips log-mels
     * @param labels their classes
     * @return the loss
     */
    public float trainDenoiser(List<float[][]> clips, int[] labels) {
        Tensor y = toSpace(clips);
        if (Float.isNaN(latentScale)) {
            float[] v = y.toArray();
            double s = 0;
            for (float a : v) {
                s += a * a;
            }
            latentScale = space == Space.LATENT ? (float) (1 / Math.sqrt(s / v.length + 1e-8)) : 1f;
        }
        int n = labels.length;
        y = y.mul(Tensor.scalar(latentScale));
        Tensor x0 = Tensor.normal(y.shape(), 0, 1, random);
        float[] t = new float[n];
        for (int i = 0; i < n; i++) {
            t[i] = (float) random.nextDouble();
        }
        Tensor tt = Tensor.of(Shape.of(n, 1), t);
        Tensor xt = x0.add(y.sub(x0).mul(tt));
        Var in = Var.of(Tensor.concat(1, xt, condition(labels, t, false, 0.1)));
        Var loss = denoiser.forward(in).sub(Var.of(y.sub(x0))).square().mean();
        float v = loss.value().item();
        loss.backward();
        optimizer.step(denoiser.parameters());
        return v;
    }

    /**
     * Generates log-mels with Euler steps and classifier-free guidance.
     *
     * @param labels the class of every sample
     * @param steps Euler steps
     * @param guidance the guidance weight (1 for none)
     * @param seed the noise seed
     * @return {@code [frames][mels]} per sample
     */
    public List<float[][]> generate(int[] labels, int steps, float guidance, long seed) {
        int n = labels.length;
        Tensor x = Tensor.normal(Shape.of(n, dimension()), 0, 1, new SplittableRandom(seed));
        for (int k = 0; k < steps; k++) {
            float[] t = new float[n];
            java.util.Arrays.fill(t, (float) k / steps);
            Tensor xk = x;
            Tensor cond = Var.noGrad(() -> denoiser.forward(Var.of(Tensor.concat(1, xk, condition(labels, t, false,
                    0))))).value();
            Tensor v = cond;
            if (guidance != 1f) {
                Tensor free = Var.noGrad(() -> denoiser.forward(Var.of(Tensor.concat(1, xk, condition(labels, t, true,
                        0))))).value();
                v = free.add(cond.sub(free).mul(Tensor.scalar(guidance)));
            }
            x = x.add(v.mul(Tensor.scalar(1f / steps)));
        }
        return fromSpace(x.mul(Tensor.scalar(1 / latentScale)));
    }

    /**
     * @param steps Euler steps
     * @param guidance the guidance weight
     * @return the network evaluations per sample
     */
    public static int evaluations(int steps, float guidance) {
        return guidance == 1f ? steps : 2 * steps;
    }
}
