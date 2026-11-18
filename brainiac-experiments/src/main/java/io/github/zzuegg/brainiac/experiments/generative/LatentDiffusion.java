package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Latent diffusion against representation-autoencoder latents and pixel diffusion (wave 6,
 * item 4.9) on 8x8 {@link ToyImages}. The same {@link MlpDenoiser} and core {@link Diffusion}
 * (cosine schedule, 100 steps, 20-step DDIM sampling) run in three spaces:
 * <ul>
 *   <li><b>pixels</b>: the 64 pixels, as {@link TinyDdpm} does;</li>
 *   <li><b>LDM</b> (Rombach et al., <a href="https://arxiv.org/abs/2112.10752">arXiv:2112.10752</a>):
 *       the 8-dimensional posterior mean of an {@link ImageAutoencoders.Vae}, standardised per
 *       dimension (LDM's scale factor), decoded by the VAE's decoder;</li>
 *   <li><b>RAE</b> (Zheng et al., <a href="https://arxiv.org/abs/2510.11690">arXiv:2510.11690</a>):
 *       the 32 features of a frozen self-supervised encoder ({@link RepaExperiment#encoder}),
 *       standardised, with only a decoder trained (on noise-augmented latents, as RAE does).</li>
 * </ul>
 * Quality ({@link GenerativeScores#images}) is recorded at diffusion-training checkpoints, so
 * the steps to a quality target can be read off; the autoencoders' own training is reported
 * separately.
 */
@SuppressWarnings("exports")
public final class LatentDiffusion {
    private static final int T = 100;

    private LatentDiffusion() {
    }

    /** Standardises latents per dimension and maps them back. */
    private record Standardizer(float[] mean, float[] std) {
        static Standardizer fit(float[][] z) {
            int d = z[0].length;
            float[] m = new float[d];
            float[] s = new float[d];
            for (float[] row : z) {
                for (int j = 0; j < d; j++) {
                    m[j] += row[j] / z.length;
                }
            }
            for (float[] row : z) {
                for (int j = 0; j < d; j++) {
                    s[j] += (row[j] - m[j]) * (row[j] - m[j]) / z.length;
                }
            }
            for (int j = 0; j < d; j++) {
                s[j] = (float) Math.sqrt(s[j] + 1e-6);
            }
            return new Standardizer(m, s);
        }

        float[][] forward(float[][] z) {
            float[][] out = new float[z.length][];
            for (int i = 0; i < z.length; i++) {
                out[i] = new float[z[i].length];
                for (int j = 0; j < z[i].length; j++) {
                    out[i][j] = (z[i][j] - mean[j]) / std[j];
                }
            }
            return out;
        }

        Tensor inverse(Tensor z) {
            float[][] rows = Nets.rows(z);
            for (float[] row : rows) {
                for (int j = 0; j < row.length; j++) {
                    row[j] = row[j] * std[j] + mean[j];
                }
            }
            return Nets.matrix(rows);
        }
    }

    /** Every toy image once: the 73 training images. */
    static float[][] images() {
        List<float[]> all = new ArrayList<>();
        for (int c = 0; c < ToyImages.CLASSES; c++) {
            all.addAll(ToyImages.all(c));
        }
        return all.toArray(new float[0][]);
    }

    /**
     * Trains a denoiser in one space and scores decoded samples at checkpoints.
     *
     * @param data the training vectors in that space
     * @param clip the clean-estimate clip of the sampler
     * @param decode maps sampled vectors to images
     * @param checkpoints ascending diffusion-training steps (of 64 vectors) to score at
     * @param seed the seed
     * @return the score at each checkpoint
     */
    static List<GenerativeScores.ImageScore> diffuse(float[][] data, float clip, Function<Tensor, float[][]> decode,
            int[] checkpoints, long seed) {
        int dim = data[0].length;
        MlpDenoiser net = MlpDenoiser.of(dim, 0, 128, 2, seed).timeScale(1f / T);
        Diffusion diffusion = new Diffusion(NoiseSchedule.cosine(T), Diffusion.Prediction.NOISE).clipSample(clip);
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        List<GenerativeScores.ImageScore> out = new ArrayList<>();
        int step = 0;
        for (int target : checkpoints) {
            for (; step < target; step++) {
                float[][] batch = new float[64][];
                for (int b = 0; b < batch.length; b++) {
                    batch[b] = data[r.nextInt(data.length)];
                }
                Var loss = diffusion.loss(Nets.unconditional(net), Nets.matrix(batch), r);
                loss.backward();
                opt.step(net.parameters());
            }
            SplittableRandom sr = new SplittableRandom(seed + 11);
            Tensor z = diffusion.sample(Nets.unconditional(net), Nets.gaussian(128, dim, sr), 20, 0, sr);
            out.add(GenerativeScores.images(decode.apply(z)));
        }
        return out;
    }

    /**
     * Runs the three spaces.
     *
     * @param autoencoderSteps training steps of the VAE, the SSL encoder and the RAE decoder
     * @param checkpoints diffusion-training steps to score at
     * @param seed the seed
     * @return per space ({@code pixels}, {@code LDM}, {@code RAE}) the scores at the checkpoints,
     *     plus {@code "<space> reconstruction"} with one entry: the autoencoder's own
     *     reconstruction of the 73 images
     */
    public static Map<String, List<GenerativeScores.ImageScore>> run(int autoencoderSteps, int[] checkpoints,
            long seed) {
        float[][] x = images();
        SplittableRandom r = new SplittableRandom(seed);
        Map<String, List<GenerativeScores.ImageScore>> out = new LinkedHashMap<>();
        out.put("pixels", diffuse(x, 1f, Nets::rows, checkpoints, seed));

        ImageAutoencoders.Vae vae = ImageAutoencoders.vae(8, 0.5f, seed);
        for (int s = 0; s < autoencoderSteps; s++) {
            vae.trainStep(PointClouds.images(64, r));
        }
        float[][] zv = Nets.rows(vae.encodeMean(x));
        Standardizer sv = Standardizer.fit(zv);
        out.put("LDM reconstruction", List.of(GenerativeScores.images(vae.decode(Nets.matrix(zv)))));
        out.put("LDM", diffuse(sv.forward(zv), 3f, z -> vae.decode(sv.inverse(z)), checkpoints, seed));

        Model encoder = RepaExperiment.encoder(autoencoderSteps, seed);
        float[][] zr = Nets.rows(encoder.forward(Var.of(Nets.matrix(x))).value());
        Standardizer sr = Standardizer.fit(zr);
        float[][] zs = sr.forward(zr);
        Model decoder = Model.sequential().input(zs[0].length).dense(128, Activation.SILU).dense(128, Activation.SILU)
                .dense(ToyImages.PIXELS).seed(seed + 2).build();
        Optimizer decOpt = Optimizer.adam(2e-3f);
        for (int s = 0; s < autoencoderSteps; s++) {
            float[][] zb = new float[64][];
            float[][] xb = new float[64][];
            for (int b = 0; b < zb.length; b++) {
                int k = r.nextInt(x.length);
                zb[b] = zs[k];
                xb[b] = x[k];
            }
            // noise-augmented latents make the decoder robust to the diffusion model's errors
            Tensor zin = Nets.matrix(zb).add(Tensor.normal(Shape.of(zb.length, zs[0].length), 0, 0.2f, r));
            Var loss = decoder.forward(Var.of(zin), true).tanh().sub(Var.of(Nets.matrix(xb))).square().mean();
            loss.backward();
            decOpt.step(decoder.parameters());
        }
        Function<Tensor, float[][]> raeDecode = z -> Nets.rows(decoder.forward(Var.of(z)).value().tanh());
        out.put("RAE reconstruction", List.of(GenerativeScores.images(raeDecode.apply(Nets.matrix(zs)))));
        out.put("RAE", diffuse(zs, 3f, raeDecode, checkpoints, seed));
        return out;
    }
}
