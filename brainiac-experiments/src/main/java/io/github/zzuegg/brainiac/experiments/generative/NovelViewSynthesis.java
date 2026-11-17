package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.FourierFeatures;
import io.github.zzuegg.brainiac.experimental.nn.GaussianSplats2d;
import io.github.zzuegg.brainiac.experimental.nn.GaussianSplats3d;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Novel-view synthesis on a synthetic multi-view scene: {@link GaussianSplats3d} (3D Gaussian
 * Splatting, Kerbl et al., <a href="https://arxiv.org/abs/2308.04079">arXiv:2308.04079</a>)
 * against a NeRF-style radiance field (Mildenhall et al.,
 * <a href="https://arxiv.org/abs/2003.08934">arXiv:2003.08934</a>): a {@link FourierFeatures}
 * positional encoding and an MLP giving density and colour, rendered by quadrature along
 * camera rays.
 *
 * <p>The scene is three Lambert-shaded spheres, ray-traced exactly. Sixteen cameras circle it
 * above the equator; the even ones are for training and the odd ones are held out. Both
 * methods take the same number of Adam steps; the score is the held-out PSNR and the wall
 * time of training.
 */
public final class NovelViewSynthesis {
    private static final int SIDE = 16;
    private static final int VIEWS = 16;
    private static final float FOCAL = 1.2f;
    private static final float NEAR = 1.6f;
    private static final float FAR = 4.4f;
    private static final int SAMPLES = 24;
    private static final float[][] SPHERES = {
        // centre x, y, z, radius, r, g, b
        {-0.45f, 0f, 0f, 0.45f, 0.9f, 0.2f, 0.2f},
        {0.5f, 0.15f, 0.25f, 0.35f, 0.2f, 0.85f, 0.3f},
        {0.05f, -0.1f, -0.55f, 0.3f, 0.25f, 0.35f, 0.95f}};

    /**
     * The outcome of one method.
     *
     * @param method the method
     * @param psnr the mean held-out PSNR in dB
     * @param trainPsnr the mean PSNR on the training views
     * @param seconds the training wall time
     */
    public record Score(String method, double psnr, double trainPsnr, double seconds) {
    }

    private NovelViewSynthesis() {
    }

    /**
     * The camera of a view.
     *
     * @param view the view index, 0 to 15
     * @return the camera
     */
    public static GaussianSplats3d.Camera camera(int view) {
        double a = 2 * Math.PI * view / VIEWS;
        float[] eye = {(float) (2.8 * Math.cos(a)), 1f, (float) (2.8 * Math.sin(a))};
        return GaussianSplats3d.Camera.lookAt(eye, new float[3], FOCAL);
    }

    /**
     * Ray-traces the ground truth of a view.
     *
     * @param view the view index
     * @return {@code [pixels, 3]} colours in [0, 1]
     */
    public static Tensor groundTruth(int view) {
        GaussianSplats3d.Camera cam = camera(view);
        float[] grid = GaussianSplats2d.grid(SIDE, SIDE).toArray();
        float[] out = new float[SIDE * SIDE * 3];
        float[] light = {0.48f, 0.8f, 0.36f};
        for (int p = 0; p < SIDE * SIDE; p++) {
            float[] d = cam.ray(grid[2 * p], grid[2 * p + 1]);
            float[] o = cam.eye();
            double best = Double.MAX_VALUE;
            for (float[] s : SPHERES) {
                double ox = o[0] - s[0];
                double oy = o[1] - s[1];
                double oz = o[2] - s[2];
                double b = ox * d[0] + oy * d[1] + oz * d[2];
                double c = ox * ox + oy * oy + oz * oz - s[3] * s[3];
                double disc = b * b - c;
                if (disc < 0) {
                    continue;
                }
                double t = -b - Math.sqrt(disc);
                if (t > 0 && t < best) {
                    best = t;
                    double nx = (ox + t * d[0]) / s[3];
                    double ny = (oy + t * d[1]) / s[3];
                    double nz = (oz + t * d[2]) / s[3];
                    double shade = 0.35 + 0.65 * Math.max(0, nx * light[0] + ny * light[1] + nz * light[2]);
                    for (int ch = 0; ch < 3; ch++) {
                        out[3 * p + ch] = (float) (s[4 + ch] * shade);
                    }
                }
            }
        }
        return Tensor.of(Shape.of(SIDE * SIDE, 3), out);
    }

    private static double psnr(Tensor rendered, Tensor truth) {
        float[] a = rendered.toArray();
        float[] b = truth.toArray();
        double mse = 0;
        for (int i = 0; i < a.length; i++) {
            double d = Math.max(0, Math.min(1, a[i])) - b[i];
            mse += d * d;
        }
        return 10 * Math.log10(a.length / Math.max(mse, 1e-12));
    }

    /**
     * Fits Gaussian splats, one training view per step.
     *
     * @param count the number of Gaussians
     * @param steps Adam steps
     * @param seed the seed
     * @return the score
     */
    public static Score splats(int count, int steps, long seed) {
        Tensor[] truth = truths();
        Tensor pixels = GaussianSplats2d.grid(SIDE, SIDE);
        GaussianSplats3d g = GaussianSplats3d.of(count, 3, 0.8f, seed);
        Optimizer opt = Optimizer.adam(1e-2f);
        SplittableRandom r = new SplittableRandom(seed);
        long start = System.nanoTime();
        for (int s = 0; s < steps; s++) {
            int view = 2 * r.nextInt(VIEWS / 2);
            g.render(camera(view), pixels).sub(Var.of(truth[view])).square().mean().backward();
            opt.step(g.parameters());
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        double test = 0;
        double train = 0;
        for (int v = 0; v < VIEWS; v++) {
            double p = psnr(g.render(camera(v), pixels).value(), truth[v]);
            if (v % 2 == 1) {
                test += p;
            } else {
                train += p;
            }
        }
        return new Score("3D Gaussian splats (" + count + ")", test / (VIEWS / 2), train / (VIEWS / 2), seconds);
    }

    /**
     * Fits a NeRF-style field, a batch of random rays from one training view per step.
     *
     * @param rays rays per step
     * @param steps Adam steps
     * @param seed the seed
     * @return the score
     */
    public static Score nerf(int rays, int steps, long seed) {
        Tensor[] truth = truths();
        Model field = Model.sequential().input(3).add(FourierFeatures.positionalSpec(4, true))
                .dense(64, Activation.RELU).dense(64, Activation.RELU).dense(4).seed(seed).build();
        Optimizer opt = Optimizer.adam(5e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        float[] grid = GaussianSplats2d.grid(SIDE, SIDE).toArray();
        long start = System.nanoTime();
        for (int s = 0; s < steps; s++) {
            int view = 2 * r.nextInt(VIEWS / 2);
            int[] pick = new int[rays];
            float[] target = new float[rays * 3];
            float[] all = truth[view].toArray();
            for (int i = 0; i < rays; i++) {
                pick[i] = r.nextInt(SIDE * SIDE);
                System.arraycopy(all, 3 * pick[i], target, 3 * i, 3);
            }
            renderRays(field, camera(view), grid, pick, r, true).sub(Var.of(Tensor.of(Shape.of(rays, 3), target)))
                    .square().mean().backward();
            opt.step(field.parameters());
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        int[] every = new int[SIDE * SIDE];
        for (int i = 0; i < every.length; i++) {
            every[i] = i;
        }
        double test = 0;
        double train = 0;
        for (int v = 0; v < VIEWS; v++) {
            double p = psnr(renderRays(field, camera(v), grid, every, r, false).value(), truth[v]);
            if (v % 2 == 1) {
                test += p;
            } else {
                train += p;
            }
        }
        return new Score("NeRF MLP (" + SAMPLES + " samples/ray)", test / (VIEWS / 2), train / (VIEWS / 2), seconds);
    }

    private static Var renderRays(Model field, GaussianSplats3d.Camera cam, float[] grid, int[] pick,
            SplittableRandom r, boolean training) {
        int n = pick.length;
        float delta = (FAR - NEAR) / SAMPLES;
        float[] points = new float[n * SAMPLES * 3];
        float[] eye = cam.eye();
        for (int i = 0; i < n; i++) {
            float[] d = cam.ray(grid[2 * pick[i]], grid[2 * pick[i] + 1]);
            for (int k = 0; k < SAMPLES; k++) {
                float t = NEAR + delta * (k + (training ? (float) r.nextDouble() : 0.5f));
                for (int c = 0; c < 3; c++) {
                    points[(i * SAMPLES + k) * 3 + c] = eye[c] + t * d[c];
                }
            }
        }
        Var out = field.forward(Var.of(Tensor.of(Shape.of(n * SAMPLES, 3), points)), training);
        Var tau = out.slice(1, 0, 1).softplus().reshape(n, SAMPLES).mul(delta);
        Var alpha = tau.neg().exp().neg().add(1f);
        Var transmittance = tau.cumsum(1).sub(tau).neg().exp();
        Var weight = alpha.mul(transmittance);
        Var[] channels = new Var[3];
        for (int c = 0; c < 3; c++) {
            Var colour = out.slice(1, 1 + c, 2 + c).sigmoid().reshape(n, SAMPLES);
            channels[c] = weight.mul(colour).sum(1, true);
        }
        return Var.concat(1, channels);
    }

    /**
     * The held-out PSNR of an all-black render, the floor that the background alone gives.
     *
     * @return the PSNR in dB
     */
    public static double blackPsnr() {
        double sum = 0;
        for (int v = 1; v < VIEWS; v += 2) {
            sum += psnr(Tensor.zeros(Shape.of(SIDE * SIDE, 3)), groundTruth(v));
        }
        return sum / (VIEWS / 2);
    }

    private static Tensor[] truths() {
        Tensor[] t = new Tensor[VIEWS];
        for (int v = 0; v < VIEWS; v++) {
            t[v] = groundTruth(v);
        }
        return t;
    }
}
