package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.PatchDiscriminator;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Image-to-image translation (wave 6, item 4.1) on 8x8 {@link ToyImages}, with a tiny local
 * U-Net generator and a {@link PatchDiscriminator}.
 * <ul>
 *   <li><b>pix2pix</b> (Isola et al., <a href="https://arxiv.org/abs/1611.07004">arXiv:1611.07004</a>),
 *       paired: a coarse 4x4 occupancy sketch to the full shape. The sketch is ambiguous (a bar
 *       on row 2 or row 3 lights the same cells), the case where a per-pixel loss averages and
 *       an adversarial loss commits to one real answer. Trained with L1 plus a conditional
 *       patch-adversarial loss (the discriminator sees sketch and output), against the same
 *       U-Net trained with L1 only. Scores on all 73 shapes: pixel L1, the share of outputs
 *       that are real shapes ({@link GenerativeScores#images}) and the share of decisive pixels
 *       ({@code |v| > 0.8}, sharpness). (Edges to filled shapes, as first planned, turned out
 *       deterministic on these images: both models reached zero error.)</li>
 *   <li><b>CycleGAN</b> (Zhu et al., <a href="https://arxiv.org/abs/1703.10593">arXiv:1703.10593</a>),
 *       unpaired: squares to crosses and back with two generators, two patch discriminators and
 *       the cycle-consistency loss, against the same GANs without it. Scores: the share of
 *       translated squares whose nearest real image is a cross (the "shape classifier"), the
 *       distance between the input's and the output's centroids (content kept), and the
 *       cycle reconstruction error.</li>
 * </ul>
 * The U-Net is local to this experiment (two resolutions, one skip connection) to stay off the
 * shared U-Net block of another work package.
 */
@SuppressWarnings("exports")
public final class ImageTranslation {
    private static final int SIDE = ToyImages.SIDE;
    private static final int[] ONE = {1, 1};

    private ImageTranslation() {
    }

    /** A two-level U-Net: 8x8 -> 4x4 -> 8x8 with a skip connection, tanh output. */
    static final class UNet {
        private final List<Var> params = new ArrayList<>();

        UNet(long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            conv(r, 16, 1, 3);
            conv(r, 32, 16, 4);
            conv(r, 32, 32, 3);
            // transposed convolution weights are [in, out, k, k]
            params.add(Var.param(Tensor.normal(Shape.of(32, 16, 4, 4), 0, (float) Math.sqrt(2.0 / (32 * 16)), r)));
            params.add(Var.param(Tensor.zeros(Shape.of(16))));
            conv(r, 1, 32, 3);
        }

        private void conv(SplittableRandom r, int out, int in, int k) {
            params.add(Var.param(Tensor.normal(Shape.of(out, in, k, k), 0, (float) Math.sqrt(2.0 / (in * k * k)), r)));
            params.add(Var.param(Tensor.zeros(Shape.of(out))));
        }

        private Var c(Var x, int i, int stride) {
            return Functional.conv(x, params.get(2 * i), params.get(2 * i + 1), new int[] {stride, stride}, ONE, ONE,
                    ONE, 1);
        }

        private static Var act(Var x) {
            return x.mul(x.sigmoid());
        }

        Var apply(Var x) {
            int n = x.shape().dim(0);
            Var img = x.reshape(n, 1, SIDE, SIDE);
            Var e1 = act(c(img, 0, 1));
            Var e2 = act(c(e1, 1, 2));
            Var m = act(c(e2, 2, 1));
            Var u = act(Functional.convTranspose(m, params.get(6), params.get(7), new int[] {2, 2}, ONE, ONE,
                    new int[] {0, 0}, ONE, 1));
            Var out = c(Var.concat(1, u, e1), 4, 1).tanh();
            return out.reshape(n, SIDE * SIDE);
        }

        List<Var> parameters() {
            return params;
        }
    }

    private static Var image(Var x) {
        return x.reshape(x.shape().dim(0), 1, SIDE, SIDE);
    }

    /** The 4x4 occupancy map of an image, drawn at 8x8: +1 on every 2x2 cell with a lit pixel. */
    static float[] coarse(float[] image) {
        float[] out = new float[SIDE * SIDE];
        java.util.Arrays.fill(out, -1);
        for (int p = 0; p < SIDE * SIDE; p++) {
            if (image[p] > 0) {
                int cy = p / SIDE / 2 * 2;
                int cx = p % SIDE / 2 * 2;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        out[(cy + dy) * SIDE + cx + dx] = 1;
                    }
                }
            }
        }
        return out;
    }

    private static float[][] all() {
        List<float[]> a = new ArrayList<>();
        for (int c = 0; c < ToyImages.CLASSES; c++) {
            a.addAll(ToyImages.all(c));
        }
        return a.toArray(new float[0][]);
    }

    private static Var nonSaturating(Var logits, boolean real) {
        return (real ? logits.neg() : logits).softplus().mean();
    }

    /**
     * Paired translation: pix2pix against an L1-only U-Net.
     *
     * @param iterations steps of 16 pairs
     * @param seed the seed
     * @return {@code "<method> L1"}, {@code "<method> real"} and {@code "<method> sharp"} for
     *     {@code pix2pix} and {@code L1 only}
     */
    public static Map<String, Double> paired(int iterations, long seed) {
        float[][] targets = all();
        float[][] edges = new float[targets.length][];
        for (int i = 0; i < targets.length; i++) {
            edges[i] = coarse(targets[i]);
        }
        UNet l1Net = new UNet(seed);
        UNet ganNet = new UNet(seed);
        PatchDiscriminator d = PatchDiscriminator.of(2, 16, seed + 1);
        Optimizer l1Opt = Optimizer.adam(2e-3f);
        Optimizer gOpt = Optimizer.adam(2e-3f);
        Optimizer dOpt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int it = 0; it < iterations; it++) {
            float[][] xb = new float[16][];
            float[][] yb = new float[16][];
            for (int i = 0; i < xb.length; i++) {
                int k = r.nextInt(edges.length);
                xb[i] = edges[k];
                yb[i] = targets[k];
            }
            Var x = Var.of(Nets.matrix(xb));
            Var y = Var.of(Nets.matrix(yb));
            l1Net.apply(x).sub(y).abs().mean().backward();
            l1Opt.step(l1Net.parameters());
            // discriminator on (input, output) pairs
            Var fake = ganNet.apply(x);
            Var dl = nonSaturating(d.forward(Var.concat(1, image(x), image(y))), true)
                    .add(nonSaturating(d.forward(Var.concat(1, image(x), image(fake.detach()))), false));
            dl.backward();
            for (Var p : ganNet.parameters()) {
                p.zeroGrad();
            }
            dOpt.step(d.parameters());
            Var gl = nonSaturating(d.forward(Var.concat(1, image(x), image(fake))), true)
                    .add(fake.sub(y).abs().mean().mul(100f));
            gl.backward();
            for (Var p : d.parameters()) {
                p.zeroGrad();
            }
            gOpt.step(ganNet.parameters());
        }
        Map<String, Double> out = new LinkedHashMap<>();
        score(out, "pix2pix", ganNet, edges, targets);
        score(out, "L1 only", l1Net, edges, targets);
        return out;
    }

    private static void score(Map<String, Double> out, String name, UNet net, float[][] in, float[][] target) {
        float[] y = net.apply(Var.of(Nets.matrix(in))).value().toArray();
        double l1 = 0;
        int sharp = 0;
        for (int i = 0; i < in.length; i++) {
            for (int p = 0; p < SIDE * SIDE; p++) {
                float v = y[i * SIDE * SIDE + p];
                l1 += Math.abs(v - target[i][p]);
                sharp += Math.abs(v) > 0.8 ? 1 : 0;
            }
        }
        out.put(name + " L1", l1 / (in.length * SIDE * SIDE));
        out.put(name + " real", GenerativeScores.images(Nets.rows(Tensor.of(Shape.of(in.length, SIDE * SIDE), y)))
                .realShare());
        out.put(name + " sharp", (double) sharp / (in.length * SIDE * SIDE));
    }

    /**
     * Unpaired translation squares to crosses: CycleGAN against the same GANs without the cycle
     * loss.
     *
     * @param iterations steps of 16 images per domain
     * @param seed the seed
     * @return {@code "<method> target share"}, {@code "<method> centroid error"} and
     *     {@code "<method> cycle L1"} for {@code CycleGAN} and {@code no cycle}
     */
    public static Map<String, Double> unpaired(int iterations, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        cycle(out, "CycleGAN", 10f, iterations, seed);
        cycle(out, "no cycle", 0f, iterations, seed);
        return out;
    }

    private static void cycle(Map<String, Double> out, String name, float lambda, int iterations, long seed) {
        float[][] a = ToyImages.all(2).toArray(new float[0][]);
        float[][] b = ToyImages.all(3).toArray(new float[0][]);
        UNet ab = new UNet(seed);
        UNet ba = new UNet(seed + 1);
        PatchDiscriminator da = PatchDiscriminator.of(1, 16, seed + 2);
        PatchDiscriminator db = PatchDiscriminator.of(1, 16, seed + 3);
        List<Var> g = new ArrayList<>(ab.parameters());
        g.addAll(ba.parameters());
        List<Var> d = new ArrayList<>(da.parameters());
        d.addAll(db.parameters());
        Optimizer gOpt = Optimizer.adam(2e-3f);
        Optimizer dOpt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int it = 0; it < iterations; it++) {
            Var xa = Var.of(batch(a, r));
            Var xb = Var.of(batch(b, r));
            Var fb = ab.apply(xa);
            Var fa = ba.apply(xb);
            Var dl = nonSaturating(db.forward(image(xb)), true).add(nonSaturating(db.forward(image(fb.detach())), false))
                    .add(nonSaturating(da.forward(image(xa)), true))
                    .add(nonSaturating(da.forward(image(fa.detach())), false));
            dl.backward();
            for (Var p : g) {
                p.zeroGrad();
            }
            dOpt.step(d);
            Var gl = nonSaturating(db.forward(image(fb)), true).add(nonSaturating(da.forward(image(fa)), true));
            if (lambda > 0) {
                gl = gl.add(ba.apply(fb).sub(xa).abs().mean().add(ab.apply(fa).sub(xb).abs().mean()).mul(lambda));
            }
            gl.backward();
            for (Var p : d) {
                p.zeroGrad();
            }
            gOpt.step(g);
        }
        Tensor in = Nets.matrix(a);
        float[][] translated = Nets.rows(ab.apply(Var.of(in)).value());
        float[][] back = Nets.rows(ba.apply(Var.of(Nets.matrix(translated))).value());
        int target = 0;
        double centroid = 0;
        double cyc = 0;
        for (int i = 0; i < a.length; i++) {
            target += ToyImages.nearest(translated[i])[0] == 3 ? 1 : 0;
            double[] ca = centroid(a[i]);
            double[] cb = centroid(translated[i]);
            centroid += Math.hypot(ca[0] - cb[0], ca[1] - cb[1]) / a.length;
            for (int p = 0; p < SIDE * SIDE; p++) {
                cyc += Math.abs(back[i][p] - a[i][p]) / (a.length * SIDE * SIDE);
            }
        }
        out.put(name + " target share", (double) target / a.length);
        out.put(name + " centroid error", centroid);
        out.put(name + " cycle L1", cyc);
    }

    private static Tensor batch(float[][] pool, SplittableRandom r) {
        float[][] x = new float[16][];
        for (int i = 0; i < x.length; i++) {
            x[i] = pool[r.nextInt(pool.length)];
        }
        return Nets.matrix(x);
    }

    /** The centroid of the pixels above zero, {row, column}; the image centre if none. */
    private static double[] centroid(float[] img) {
        double r = 0;
        double c = 0;
        double m = 0;
        for (int p = 0; p < SIDE * SIDE; p++) {
            if (img[p] > 0) {
                r += p / SIDE;
                c += p % SIDE;
                m++;
            }
        }
        return m == 0 ? new double[] {3.5, 3.5} : new double[] {r / m, c / m};
    }
}
