package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.StyleGenerator;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Adversarial;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * StyleGAN-lite against a DCGAN-style generator (wave 6, item 4.2) on 8x8 {@link ToyImages}:
 * {@link StyleGenerator} (mapping network, modulated/demodulated layers, noise injection, style
 * mixing; once without and once with lazy path-length regularisation every 4 steps) and a
 * plain generator (noise through
 * SiLU dense layers to tanh pixels, the dense analogue of DCGAN on 64 pixels), both trained by
 * core {@link Adversarial} (non-saturating loss, R1 penalty) against the same MLP
 * discriminator for the same steps. Scores: sliced Wasserstein distance to the real images,
 * {@link GenerativeScores#images}, and the perceptual path length (pixel form): the mean squared
 * pixel change per squared step along linear interpolations in the input latent (W for the
 * style generator, Z for the plain one), {@code eps = 1e-2}.
 */
@SuppressWarnings("exports")
public final class StyleGanExperiment {
    private static final int LATENT = 8;

    private StyleGanExperiment() {
    }

    /** One generator's scores. */
    public record Score(double sliced, GenerativeScores.ImageScore images, double pathLength) {
    }

    private static Model discriminator(long seed) {
        return Model.sequential().input(ToyImages.PIXELS).dense(128, Activation.SILU).dense(128, Activation.SILU)
                .dense(1).seed(seed).build();
    }

    private static Adversarial gan(Function<Var, Var> gen, List<Var> params, Optimizer opt, long seed) {
        Model d = discriminator(seed + 1);
        return new Adversarial(gen, params, opt, x -> d.forward(x, true), d.parameters(), Optimizer.adam(1e-3f),
                Adversarial.Loss.NON_SATURATING).withR1(1f);
    }

    /**
     * Trains the generators: the style generator without and with the path-length
     * regulariser, and the plain one.
     *
     * @param iterations adversarial rounds of 64 images
     * @param pathWeight the path-length penalty weight
     * @param seed the seed
     * @return {@code "StyleGAN"}, {@code "StyleGAN + PPL reg"} and {@code "DCGAN"} scores
     */
    public static Map<String, Score> run(int iterations, float pathWeight, long seed) {
        StyleGenerator style = StyleGenerator.of(LATENT, 32, 128, 2, ToyImages.PIXELS, seed).mixing(0.5f);
        StyleGenerator regular = StyleGenerator.of(LATENT, 32, 128, 2, ToyImages.PIXELS, seed).mixing(0.5f);
        Model plain = Model.sequential().input(LATENT).dense(128, Activation.SILU).dense(128, Activation.SILU)
                .dense(ToyImages.PIXELS).seed(seed).build();
        Optimizer regularOpt = Optimizer.adam(1e-3f);
        Adversarial styleGan = gan(style, style.parameters(), Optimizer.adam(1e-3f), seed);
        Adversarial regularGan = gan(regular, regular.parameters(), regularOpt, seed);
        Adversarial plainGan = gan(z -> plain.forward(z, true).tanh(), plain.parameters(), Optimizer.adam(1e-3f),
                seed);
        SplittableRandom r = new SplittableRandom(seed);
        for (int it = 0; it < iterations; it++) {
            Tensor real = Nets.matrix(PointClouds.images(64, r));
            styleGan.step(real, () -> Tensor.normal(Shape.of(64, 2 * LATENT), 0, 1, r));
            regularGan.step(real, () -> Tensor.normal(Shape.of(64, 2 * LATENT), 0, 1, r));
            plainGan.step(real, () -> Tensor.normal(Shape.of(64, LATENT), 0, 1, r));
            if (it % 4 == 3) {
                // lazy regularisation: the penalty every 4 steps, weighted by 4
                for (Var p : regular.parameters()) {
                    p.zeroGrad();
                }
                regular.pathLengthPenalty(32, r).mul(4 * pathWeight).backward();
                regularOpt.step(regular.parameters());
            }
        }
        float[][] real = PointClouds.images(256, new SplittableRandom(seed + 3));
        Map<String, Score> out = new LinkedHashMap<>();
        out.put("StyleGAN", score(style, real, seed));
        out.put("StyleGAN + PPL reg", score(regular, real, seed));
        SplittableRandom e = new SplittableRandom(seed + 4);
        Function<Tensor, Tensor> plainFromZ = z -> plain.forward(Var.of(z), false).value().tanh();
        float[][] samples = Nets.rows(plainFromZ.apply(Tensor.normal(Shape.of(256, LATENT), 0, 1, e)));
        Tensor z1 = Tensor.normal(Shape.of(256, LATENT), 0, 1, e);
        Tensor z2 = Tensor.normal(Shape.of(256, LATENT), 0, 1, e);
        out.put("DCGAN", new Score(GenerativeScores.slicedWasserstein(samples, real, 64, seed),
                GenerativeScores.images(samples), pathLength(plainFromZ, z1, z2, e)));
        return out;
    }

    private static Score score(StyleGenerator g, float[][] real, long seed) {
        SplittableRandom e = new SplittableRandom(seed + 4);
        Function<Tensor, Tensor> fromW = w -> g.synthesize(Var.of(w), null, 3, false).value();
        float[][] samples = Nets.rows(fromW.apply(g.map(Var.of(Tensor.normal(Shape.of(256, LATENT), 0, 1, e)))
                .value()));
        Tensor w1 = g.map(Var.of(Tensor.normal(Shape.of(256, LATENT), 0, 1, e))).value();
        Tensor w2 = g.map(Var.of(Tensor.normal(Shape.of(256, LATENT), 0, 1, e))).value();
        return new Score(GenerativeScores.slicedWasserstein(samples, real, 64, seed), GenerativeScores.images(samples),
                pathLength(fromW, w1, w2, e));
    }

    private static double pathLength(Function<Tensor, Tensor> g, Tensor a, Tensor b, SplittableRandom r) {
        int n = a.shape().dim(0);
        float eps = 1e-2f;
        float[] t = new float[n];
        float[] t2 = new float[n];
        for (int i = 0; i < n; i++) {
            t[i] = (float) (r.nextDouble() * (1 - eps));
            t2[i] = t[i] + eps;
        }
        Tensor d = b.sub(a);
        Tensor pa = g.apply(a.add(d.mul(Nets.column(t))));
        Tensor pb = g.apply(a.add(d.mul(Nets.column(t2))));
        return pb.sub(pa).square().sum().item() / n / (eps * eps);
    }
}
