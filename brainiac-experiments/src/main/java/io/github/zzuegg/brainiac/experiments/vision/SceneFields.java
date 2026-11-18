package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GaussianSplats2d;
import io.github.zzuegg.brainiac.experimental.nn.GaussianSplats3d;
import io.github.zzuegg.brainiac.experimental.vision.HashGridEncoding;
import io.github.zzuegg.brainiac.experimental.vision.RadianceField;
import io.github.zzuegg.brainiac.experimental.vision.SignedDistanceField;
import io.github.zzuegg.brainiac.experimental.vision.SineLayer;
import io.github.zzuegg.brainiac.experimental.vision.VolumeRenderer;
import io.github.zzuegg.brainiac.experiments.generative.NeuralFields;
import io.github.zzuegg.brainiac.experiments.generative.NovelViewSynthesis;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Coordinate networks for scenes and shapes (wave 8, items 8.1.12-8.1.14).
 *
 * <ul>
 *   <li>NeRF: a {@link RadianceField} rendered by the {@link VolumeRenderer} with hierarchical
 *       sampling, on the multi-view sphere scene of {@link NovelViewSynthesis} (16x16 renders,
 *       16 cameras, the even ones for training), against {@link GaussianSplats3d} and against
 *       a field without positional encoding; held-out PSNR, training time and render time.</li>
 *   <li>Instant-NGP: a {@link HashGridEncoding} with a tiny MLP, on the {@link NeuralFields}
 *       picture (against raw and Fourier-feature MLPs and SIREN) with a table-size sweep, and
 *       as the point encoder of the NeRF field.</li>
 *   <li>Neural SDFs: a {@link SignedDistanceField} auto-decoder fitted to six 2-D outlines
 *       (bar, square, plus, diagonal bar, two random star polygons) from 200 outline points
 *       and normals each, with SIREN, ReLU and Fourier bodies under the same eikonal losses;
 *       Chamfer distance and IoU of the zero level set and the eikonal residual.</li>
 * </ul>
 */
public final class SceneFields {

    private static final int SIDE = 16;
    private static final int VIEWS = 16;
    private static final float NEAR = 1.6f;
    private static final float FAR = 4.4f;
    private static final float BOX = 1.2f;

    /**
     * The outcome of a novel-view method.
     *
     * @param method the method
     * @param psnr the mean held-out PSNR in dB
     * @param trainPsnr the mean PSNR on the training views
     * @param trainSeconds the training wall time
     * @param renderMillis the time to render one full view
     */
    public record Score(String method, double psnr, double trainPsnr, double trainSeconds, double renderMillis) {
    }

    /** The point encoder of the NeRF field. */
    public enum Encoding {
        /** NeRF positional encoding, 4 octaves */
        POSITIONAL,
        /** raw coordinates */
        NONE,
        /** a 3-D hash grid */
        HASH_GRID
    }

    private SceneFields() {
    }

    private static Tensor[] truths() {
        Tensor[] t = new Tensor[VIEWS];
        for (int v = 0; v < VIEWS; v++) {
            t[v] = NovelViewSynthesis.groundTruth(v);
        }
        return t;
    }

    private static double psnr(float[] a, float[] b) {
        double mse = 0;
        for (int i = 0; i < a.length; i++) {
            double d = Math.max(0, Math.min(1, a[i])) - b[i];
            mse += d * d;
        }
        return 10 * Math.log10(a.length / Math.max(mse, 1e-12));
    }

    private static float[] directions(GaussianSplats3d.Camera cam, float[] grid, int[] pick) {
        float[] d = new float[pick.length * 3];
        for (int i = 0; i < pick.length; i++) {
            float[] r = cam.ray(grid[2 * pick[i]], grid[2 * pick[i] + 1]);
            System.arraycopy(r, 0, d, 3 * i, 3);
        }
        return d;
    }

    /**
     * Trains a NeRF field, a batch of random rays from one training view per step, on the
     * coarse and the fine colour.
     *
     * @param encoding the point encoder
     * @param hierarchical whether to add 16 importance samples to the 16 stratified ones
     * @param rays rays per step
     * @param steps Adam steps
     * @param seed the seed
     * @return the score
     */
    public static Score nerf(Encoding encoding, boolean hierarchical, int rays, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        RadianceField field = switch (encoding) {
            case POSITIONAL -> RadianceField.positional(4, 2, 64, r);
            case NONE -> RadianceField.positional(0, 2, 64, r);
            case HASH_GRID -> new RadianceField(new HashGridEncoding(3, 6, 2, 14, 4, 48,
                    new float[] {-BOX, -BOX, -BOX}, new float[] {BOX, BOX, BOX}, r), 12, 2, 32, r);
        };
        VolumeRenderer renderer = new VolumeRenderer(NEAR, FAR, hierarchical ? 16 : 32, hierarchical ? 16 : 0);
        Tensor[] truth = truths();
        float[] grid = GaussianSplats2d.grid(SIDE, SIDE).toArray();
        Optimizer opt = Optimizer.adam(encoding == Encoding.HASH_GRID ? 1e-2f : 2e-3f);
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
            GaussianSplats3d.Camera cam = NovelViewSynthesis.camera(view);
            VolumeRenderer.Rendering out = renderer.render(field, cam.eye(), directions(cam, grid, pick), r);
            Tensor t = Tensor.of(Shape.of(rays, 3), target);
            Var loss = out.colour().sub(t).square().mean();
            if (hierarchical) {
                loss = loss.add(out.coarse().sub(t).square().mean());
            }
            loss.backward();
            opt.step(field.parameters());
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        int[] every = new int[SIDE * SIDE];
        for (int i = 0; i < every.length; i++) {
            every[i] = i;
        }
        double test = 0;
        double train = 0;
        long renderNanos = 0;
        for (int v = 0; v < VIEWS; v++) {
            GaussianSplats3d.Camera cam = NovelViewSynthesis.camera(v);
            long t0 = System.nanoTime();
            float[] img = Var.noGrad(() -> renderer.render(field, cam.eye(), directions(cam, grid, every), null))
                    .colour().value().toArray();
            renderNanos += System.nanoTime() - t0;
            double p = psnr(img, truth[v].toArray());
            if (v % 2 == 1) {
                test += p;
            } else {
                train += p;
            }
        }
        String name = "NeRF " + encoding.name().toLowerCase(java.util.Locale.ROOT).replace('_', ' ')
                + (hierarchical ? " (16+16 hierarchical)" : " (32 stratified)");
        return new Score(name, test / (VIEWS / 2), train / (VIEWS / 2), seconds, renderNanos / 1e6 / VIEWS);
    }

    /**
     * Fits 3-D Gaussian splats, one training view per step (the 3DGS baseline, timed).
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
            g.render(NovelViewSynthesis.camera(view), pixels).sub(Var.of(truth[view])).square().mean().backward();
            opt.step(g.parameters());
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        double test = 0;
        double train = 0;
        long renderNanos = 0;
        for (int v = 0; v < VIEWS; v++) {
            int view = v;
            long t0 = System.nanoTime();
            float[] img = Var.noGrad(() -> g.render(NovelViewSynthesis.camera(view), pixels)).value().toArray();
            renderNanos += System.nanoTime() - t0;
            double p = psnr(img, truth[v].toArray());
            if (v % 2 == 1) {
                test += p;
            } else {
                train += p;
            }
        }
        return new Score("3D Gaussian splats (" + count + ")", test / (VIEWS / 2), train / (VIEWS / 2), seconds,
                renderNanos / 1e6 / VIEWS);
    }

    // ---------------------------------------------------------------- image fitting

    /** A coordinate network for image fitting. */
    public enum ImageNet {
        /** ReLU MLP on raw coordinates (64-64) */
        RAW,
        /** ReLU MLP behind 64 random Fourier features (64-64) */
        FOURIER,
        /** SIREN (64-64, omega0 30) */
        SIREN,
        /** hash grid (8 levels x 2 features) and a 32-unit MLP */
        HASH_GRID
    }

    /**
     * The outcome of an image fit.
     *
     * @param method the network
     * @param psnr the final PSNR in dB
     * @param seconds the training wall time
     * @param parameters the trainable values
     */
    public record FitScore(String method, double psnr, double seconds, long parameters) {
    }

    /**
     * Builds an image-fitting network on the unit square.
     *
     * @param net the network
     * @param log2Table the hash-table size (hash grid only)
     * @param side the picture side, the finest grid resolution
     * @param seed the seed
     * @return the model, from {@code [pixels, 2]} to {@code [pixels, 1]}
     */
    public static Model imageNet(ImageNet net, int log2Table, int side, long seed) {
        return switch (net) {
            case RAW -> NeuralFields.coordinateMlp(0, 0, seed);
            case FOURIER -> NeuralFields.coordinateMlp(64, 3f, seed);
            case SIREN -> Model.sequential().input(2).add(SineLayer.spec(64, 30f, true))
                    .add(SineLayer.spec(64, 30f, false)).dense(1).seed(seed).build();
            case HASH_GRID -> Model.sequential().input(2).add(HashGridEncoding.spec(8, 2, log2Table, 2, side))
                    .dense(32, Activation.RELU).dense(1).seed(seed).build();
        };
    }

    /**
     * Fits the {@link NeuralFields} picture (checkerboard and disc) by full-batch Adam.
     *
     * @param net the network
     * @param log2Table the hash-table size (hash grid only)
     * @param side the picture side
     * @param steps Adam steps
     * @param seed the seed
     * @return the score
     */
    public static FitScore fitImage(ImageNet net, int log2Table, int side, int steps, long seed) {
        Model m = imageNet(net, log2Table, side, seed);
        float lr = net == ImageNet.HASH_GRID ? 2e-2f : net == ImageNet.SIREN ? 1e-3f : 5e-3f;
        long start = System.nanoTime();
        double psnr = NeuralFields.fit(m, NeuralFields.picture(side, side / 4), side, steps, lr);
        double seconds = (System.nanoTime() - start) / 1e9;
        long params = 0;
        for (Var p : m.parameters()) {
            params += p.value().size();
        }
        String name = net == ImageNet.HASH_GRID ? "hash grid 2^" + log2Table : net.name().toLowerCase(java.util.Locale.ROOT);
        return new FitScore(name, psnr, seconds, params);
    }

    // ---------------------------------------------------------------- signed distance fields

    /** Outline shapes: bar, square, plus, diagonal bar and two random star polygons. */
    public static final int SHAPES = 6;
    private static final int OUTLINE_POINTS = 200;
    private static final int GRID = 48;

    /**
     * The outcome of an SDF fit, averaged over the shapes.
     *
     * @param method the body
     * @param chamfer the symmetric Chamfer distance between the zero level set and the
     *     outline, in units of the {@code [-1, 1]} box
     * @param iou the IoU of {@code f < 0} and the shape interior on a 48x48 grid
     * @param eikonal the mean {@code | |grad f| - 1 |} on that grid
     * @param seconds the training wall time
     */
    public record SdfScore(String method, double chamfer, double iou, double eikonal, double seconds) {
    }

    /**
     * The polygons, counter-clockwise, in {@code [-1, 1]^2}.
     *
     * @param seed the seed of the random polygons
     * @return {@code SHAPES} polygons as {@code [x0, y0, x1, y1, ...]}
     */
    public static List<float[]> polygons(long seed) {
        List<float[]> p = new ArrayList<>();
        p.add(new float[] {-0.7f, -0.25f, 0.7f, -0.25f, 0.7f, 0.25f, -0.7f, 0.25f});
        p.add(new float[] {-0.5f, -0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f});
        float a = 0.25f;
        float b = 0.7f;
        p.add(new float[] {-a, -b, a, -b, a, -a, b, -a, b, a, a, a, a, b, -a, b, -a, a, -b, a, -b, -a, -a, -a});
        float[] diag = new float[8];
        for (int i = 0; i < 4; i++) {
            float x = p.get(0)[2 * i];
            float y = p.get(0)[2 * i + 1];
            diag[2 * i] = (float) ((x - y) * Math.sqrt(0.5));
            diag[2 * i + 1] = (float) ((x + y) * Math.sqrt(0.5));
        }
        p.add(diag);
        SplittableRandom r = new SplittableRandom(seed);
        for (int k = 0; k < 2; k++) {
            int n = 7;
            float[] star = new float[2 * n];
            for (int i = 0; i < n; i++) {
                double ang = 2 * Math.PI * (i + 0.3 * r.nextDouble()) / n;
                double rad = 0.45 + 0.4 * r.nextDouble();
                star[2 * i] = (float) (rad * Math.cos(ang));
                star[2 * i + 1] = (float) (rad * Math.sin(ang));
            }
            p.add(star);
        }
        return p;
    }

    /**
     * Samples points evenly by arc length along a polygon outline, with outward normals.
     *
     * @param poly the counter-clockwise polygon
     * @param count the number of points
     * @param random the jitter, or {@code null} for even spacing
     * @return {@code {points [count * 2], normals [count * 2]}}
     */
    public static float[][] outline(float[] poly, int count, SplittableRandom random) {
        int n = poly.length / 2;
        double[] cum = new double[n + 1];
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            cum[i + 1] = cum[i] + Math.hypot(poly[2 * j] - poly[2 * i], poly[2 * j + 1] - poly[2 * i + 1]);
        }
        float[] pts = new float[count * 2];
        float[] nrm = new float[count * 2];
        for (int k = 0; k < count; k++) {
            double s = (k + (random == null ? 0.5 : random.nextDouble())) / count * cum[n];
            int i = 0;
            while (i < n - 1 && cum[i + 1] < s) {
                i++;
            }
            int j = (i + 1) % n;
            double dx = poly[2 * j] - poly[2 * i];
            double dy = poly[2 * j + 1] - poly[2 * i + 1];
            double len = Math.hypot(dx, dy);
            double t = (s - cum[i]) / len;
            pts[2 * k] = (float) (poly[2 * i] + t * dx);
            pts[2 * k + 1] = (float) (poly[2 * i + 1] + t * dy);
            nrm[2 * k] = (float) (dy / len);
            nrm[2 * k + 1] = (float) (-dx / len);
        }
        return new float[][] {pts, nrm};
    }

    private static boolean inside(float[] poly, double x, double y) {
        boolean in = false;
        int n = poly.length / 2;
        for (int i = 0, j = n - 1; i < n; j = i++) {
            double xi = poly[2 * i];
            double yi = poly[2 * i + 1];
            double xj = poly[2 * j];
            double yj = poly[2 * j + 1];
            if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                in = !in;
            }
        }
        return in;
    }

    private static double chamfer(float[] a, float[] b) {
        if (a.length == 0) {
            return 2;
        }
        return 0.5 * (oneWay(a, b) + oneWay(b, a));
    }

    private static double oneWay(float[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i += 2) {
            double best = Double.MAX_VALUE;
            for (int j = 0; j < b.length; j += 2) {
                best = Math.min(best, Math.hypot(a[i] - b[j], a[i + 1] - b[j + 1]));
            }
            sum += best;
        }
        return sum / (a.length / 2);
    }

    /**
     * Fits one SDF auto-decoder to all six outlines: every step sees the 200 fixed outline
     * points and normals of each shape plus 50 fresh uniform points per shape.
     *
     * @param kind the body
     * @param steps Adam steps
     * @param seed the seed
     * @return the score
     */
    public static SdfScore sdf(SignedDistanceField.Kind kind, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> polys = polygons(seed);
        int free = 50;
        int m = SHAPES * OUTLINE_POINTS;
        int total = m + SHAPES * free;
        float[] surf = new float[m * 2];
        float[] normals = new float[m * 2];
        float[] ids = new float[total];
        for (int s = 0; s < SHAPES; s++) {
            float[][] o = outline(polys.get(s), OUTLINE_POINTS, r);
            System.arraycopy(o[0], 0, surf, s * OUTLINE_POINTS * 2, OUTLINE_POINTS * 2);
            System.arraycopy(o[1], 0, normals, s * OUTLINE_POINTS * 2, OUTLINE_POINTS * 2);
            java.util.Arrays.fill(ids, s * OUTLINE_POINTS, (s + 1) * OUTLINE_POINTS, s);
            java.util.Arrays.fill(ids, m + s * free, m + (s + 1) * free, s);
        }
        Tensor idTensor = Tensor.of(Shape.of(total), ids);
        Tensor normalTensor = Tensor.of(Shape.of(m, 2), normals);
        SignedDistanceField field = new SignedDistanceField(kind, SHAPES, 8, 64, 3, r);
        Optimizer opt = Optimizer.adam(kind == SignedDistanceField.Kind.SIREN ? 1e-3f : 2e-3f);
        long start = System.nanoTime();
        for (int step = 0; step < steps; step++) {
            float[] pts = new float[total * 2];
            System.arraycopy(surf, 0, pts, 0, m * 2);
            for (int i = 2 * m; i < pts.length; i++) {
                pts[i] = (float) (2 * r.nextDouble() - 1);
            }
            field.loss(Tensor.of(Shape.of(total, 2), pts), idTensor, normalTensor).total().backward();
            opt.step(field.parameters());
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        double chamfer = 0;
        double iou = 0;
        double eik = 0;
        for (int s = 0; s < SHAPES; s++) {
            float[][] g = field.grid(s, GRID);
            float[] dense = outline(polys.get(s), 400, null)[0];
            chamfer += chamfer(SignedDistanceField.zeroCrossings(g[0], GRID), dense);
            int inter = 0;
            int union = 0;
            for (int y = 0; y < GRID; y++) {
                for (int x = 0; x < GRID; x++) {
                    boolean truth = inside(polys.get(s), -1 + (x + 0.5) * 2 / GRID, -1 + (y + 0.5) * 2 / GRID);
                    boolean pred = g[0][y * GRID + x] < 0;
                    inter += truth && pred ? 1 : 0;
                    union += truth || pred ? 1 : 0;
                }
            }
            iou += (double) inter / Math.max(1, union);
            double e = 0;
            for (float v : g[1]) {
                e += Math.abs(v - 1);
            }
            eik += e / g[1].length;
        }
        return new SdfScore(kind.name().toLowerCase(java.util.Locale.ROOT), chamfer / SHAPES, iou / SHAPES,
                eik / SHAPES, seconds);
    }
}
