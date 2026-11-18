package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.DeformableConv2d;
import io.github.zzuegg.brainiac.experimental.nn.EquivariantGraphLayer;
import io.github.zzuegg.brainiac.experimental.nn.SetEncoder;
import io.github.zzuegg.brainiac.experimental.vision.EdgeConv;
import io.github.zzuegg.brainiac.experimental.vision.GroupConv2d;
import io.github.zzuegg.brainiac.experimental.vision.PointNet;
import io.github.zzuegg.brainiac.experimental.vision.SpatialTransformerNetwork;
import io.github.zzuegg.brainiac.experiments.exotic.EquivariantPointClouds;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Geometry-aware vision (wave 8, items 8.1.15-8.1.17).
 *
 * <ul>
 *   <li>Point clouds: {@link PointNet} (with and without its T-Net) and a DGCNN of two
 *       {@link EdgeConv} layers classify 16-point clouds of five 3-D shapes (sphere, cube,
 *       cylinder, cone, torus), trained upright and tested upright and under random
 *       rotations, against EGNN ({@link EquivariantGraphLayer}), Deep Sets
 *       ({@link SetEncoder}) and an MLP on the coordinates sorted by x.</li>
 *   <li>Rotation equivariance: a p4 {@link GroupConv2d} network against the
 *       {@code VisionShootout} CNN with and without quarter-turn augmentation, on 8x8
 *       {@link ShapeImages} with rotation-related classes merged (bars, square, diagonals,
 *       plus); training sees one member of each merged pair upright (horizontal bar, main
 *       diagonal) and testing rotates by 90, 180 or 270 degrees; with fresh data and with a
 *       64-image training set.</li>
 *   <li>Spatial transformers: a {@link SpatialTransformerNetwork} in front of the CNN, on
 *       12x12 renders of five glyphs (square outline, plus, bar, L, T) under rotation of up to
 *       45 degrees, scale 0.6-1.2, translation and clutter, against the CNN, a
 *       {@link DeformableConv2d} CNN and the G-CNN.</li>
 * </ul>
 */
public final class GeometricVision {

    private GeometricVision() {
    }

    private static long count(List<Var> params) {
        long n = 0;
        for (Var p : params) {
            n += p.value().size();
        }
        return n;
    }

    private static List<Var> params(Layer... layers) {
        List<Var> p = new ArrayList<>();
        for (Layer l : layers) {
            p.addAll(l.parameters());
        }
        return p;
    }

    private static Layer asLayer(Model m) {
        return new Layer() {
            @Override
            public Var forward(Var input) {
                return m.forward(input, false);
            }

            @Override
            public List<Var> parameters() {
                return m.parameters();
            }
        };
    }

    private static double accuracy(Var logits, float[] labels) {
        float[] pred = logits.value().argmax(1).toArray();
        int ok = 0;
        for (int i = 0; i < labels.length; i++) {
            ok += (int) pred[i] == (int) labels[i] ? 1 : 0;
        }
        return (double) ok / labels.length;
    }

    // ---------------------------------------------------------------- point clouds

    /** Points per cloud. */
    public static final int POINTS = 16;
    /** Shape classes: sphere, cube, cylinder, cone, torus. */
    public static final int SHAPES = 5;

    /** A point-cloud classifier. */
    public enum PointModel {
        /** PointNet without the input transform */
        POINTNET,
        /** PointNet with the T-Net */
        POINTNET_TNET,
        /** two EdgeConv layers, k = 8 */
        DGCNN,
        /** Deep Sets, sum-pooled */
        DEEP_SETS,
        /** two EGNN layers */
        EGNN,
        /** an MLP on the points sorted by x */
        SORTED_MLP
    }

    /**
     * The outcome of a point-cloud classifier.
     *
     * @param method the model
     * @param accuracy test accuracy on upright clouds
     * @param rotatedAccuracy test accuracy on randomly rotated clouds
     * @param parameters the trainable values
     * @param seconds the training wall time
     */
    public record PointScore(String method, double accuracy, double rotatedAccuracy, long parameters,
            double seconds) {
    }

    /**
     * Samples a cloud from a shape's surface, scaled by 0.8-1.2, with 0.02 noise.
     *
     * @param shape the class
     * @param rotate whether to apply a uniformly random rotation
     * @param random the random source
     * @return {@code POINTS * 3} coordinates
     */
    public static float[] cloud(int shape, boolean rotate, SplittableRandom random) {
        double scale = random.nextDouble(0.8, 1.2);
        double[][] rot = rotate ? EquivariantPointClouds.randomRotation(random) : null;
        float[] out = new float[POINTS * 3];
        for (int i = 0; i < POINTS; i++) {
            double u = random.nextDouble();
            double v = random.nextDouble();
            double a = 2 * Math.PI * u;
            double[] p = switch (shape) {
                case 0 -> {
                    double z = 2 * v - 1;
                    double r = Math.sqrt(1 - z * z);
                    yield new double[] {r * Math.cos(a), r * Math.sin(a), z};
                }
                case 1 -> {
                    int face = random.nextInt(6);
                    double[] q = {2 * u - 1, 2 * v - 1, face % 2 == 0 ? 1 : -1};
                    yield new double[] {q[(face / 2) % 3], q[(face / 2 + 1) % 3], q[(face / 2 + 2) % 3]};
                }
                case 2 -> new double[] {0.6 * Math.cos(a), 0.6 * Math.sin(a), 2 * v - 1};
                case 3 -> {
                    double h = 1 - Math.sqrt(v);
                    yield new double[] {0.8 * (1 - h) * Math.cos(a), 0.8 * (1 - h) * Math.sin(a), 2 * h - 1};
                }
                default -> {
                    double b = 2 * Math.PI * v;
                    double r = 0.75 + 0.3 * Math.cos(b);
                    yield new double[] {r * Math.cos(a), r * Math.sin(a), 0.3 * Math.sin(b)};
                }
            };
            if (rot != null) {
                p = EquivariantPointClouds.apply(rot, p);
            }
            for (int c = 0; c < 3; c++) {
                out[3 * i + c] = (float) (scale * p[c] + 0.02 * random.nextGaussian());
            }
        }
        return out;
    }

    private static Tensor[] clouds(int n, boolean rotate, SplittableRandom random) {
        float[] x = new float[n * POINTS * 3];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            y[i] = i % SHAPES;
            System.arraycopy(cloud(i % SHAPES, rotate, random), 0, x, i * POINTS * 3, POINTS * 3);
        }
        return new Tensor[] {Tensor.of(Shape.of(n, POINTS, 3), x), Tensor.of(Shape.of(n), y)};
    }

    private static final class Dgcnn implements Layer {
        private final EdgeConv first;
        private final EdgeConv second;
        private final Dense mix;
        private final Dense head1;
        private final Dense head2;

        Dgcnn(SplittableRandom r) {
            first = new EdgeConv(3, 32, 8, r);
            second = new EdgeConv(32, 32, 8, r);
            mix = new Dense(64, 64, Activation.RELU, Init.HE, r);
            head1 = new Dense(64, 64, Activation.RELU, Init.HE, r);
            head2 = new Dense(64, SHAPES, Activation.IDENTITY, Init.XAVIER, r);
        }

        @Override
        public Var forward(Var x) {
            Var a = first.forward(x);
            Var b = second.forward(a);
            Var pooled = mix.forward(Var.concat(2, a, b)).max(1, false);
            return head2.forward(head1.forward(pooled));
        }

        @Override
        public List<Var> parameters() {
            return params(first, second, mix, head1, head2);
        }
    }

    private static final class Egnn implements Layer {
        private static final int FEATURES = 8;
        private final EquivariantGraphLayer first;
        private final EquivariantGraphLayer second;
        private final Dense head;

        Egnn(long seed, SplittableRandom r) {
            first = EquivariantGraphLayer.of(3, FEATURES, 32, seed + 1);
            second = EquivariantGraphLayer.of(3, FEATURES, 32, seed + 2);
            head = new Dense(FEATURES, SHAPES, Activation.IDENTITY, Init.XAVIER, r);
        }

        @Override
        public Var forward(Var x) {
            int b = x.shape().dim(0);
            float[] ones = new float[b * POINTS * FEATURES];
            Arrays.fill(ones, 1f);
            Var h = Var.concat(2, x, Var.of(Tensor.of(Shape.of(b, POINTS, FEATURES), ones)));
            h = second.forward(first.forward(h));
            return head.forward(h.slice(2, 3, 3 + FEATURES).mean(1, false));
        }

        @Override
        public List<Var> parameters() {
            return params(first, second, head);
        }
    }

    private static final class SortedMlp implements Layer {
        private final Dense hidden;
        private final Dense out;

        SortedMlp(SplittableRandom r) {
            hidden = new Dense(POINTS * 3, 64, Activation.RELU, Init.HE, r);
            out = new Dense(64, SHAPES, Activation.IDENTITY, Init.XAVIER, r);
        }

        @Override
        public Var forward(Var x) {
            int b = x.shape().dim(0);
            float[] v = x.value().toArray();
            float[] sorted = new float[v.length];
            Integer[] order = new Integer[POINTS];
            for (int s = 0; s < b; s++) {
                int base = s * POINTS * 3;
                for (int i = 0; i < POINTS; i++) {
                    order[i] = i;
                }
                Arrays.sort(order, (p, q) -> Float.compare(v[base + 3 * p], v[base + 3 * q]));
                for (int i = 0; i < POINTS; i++) {
                    System.arraycopy(v, base + 3 * order[i], sorted, base + 3 * i, 3);
                }
            }
            return out.forward(hidden.forward(Var.of(Tensor.of(Shape.of(b, POINTS * 3), sorted))));
        }

        @Override
        public List<Var> parameters() {
            return params(hidden, out);
        }
    }

    private static Layer pointModel(PointModel kind, long seed, SplittableRandom r) {
        return switch (kind) {
            case POINTNET -> new PointNet(SHAPES, 128, false, r);
            case POINTNET_TNET -> new PointNet(SHAPES, 128, true, r);
            case DGCNN -> new Dgcnn(r);
            case DEEP_SETS -> new SetEncoder(3, 64, SHAPES, SetEncoder.Kind.DEEP_SETS, 8, 4, r);
            case EGNN -> new Egnn(seed, r);
            case SORTED_MLP -> new SortedMlp(r);
        };
    }

    /**
     * Trains a point-cloud classifier on upright clouds (16 fresh clouds per Adam step) and
     * tests it on 200 upright and 200 rotated clouds.
     *
     * @param kind the model
     * @param steps Adam steps
     * @param seed the seed
     * @return the score
     */
    public static PointScore points(PointModel kind, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Layer net = pointModel(kind, seed, r);
        Optimizer opt = Optimizer.adam(3e-3f);
        long start = System.nanoTime();
        for (int s = 0; s < steps; s++) {
            Tensor[] batch = clouds(16, false, r);
            Functional.crossEntropy(net.forward(Var.of(batch[0])), batch[1]).backward();
            opt.step(net.parameters());
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        SplittableRandom test = new SplittableRandom(seed + 1000);
        Tensor[] upright = clouds(200, false, test);
        Tensor[] rotated = clouds(200, true, test);
        double acc = Var.noGrad(() -> accuracy(net.forward(Var.of(upright[0])), upright[1].toArray()));
        double rot = Var.noGrad(() -> accuracy(net.forward(Var.of(rotated[0])), rotated[1].toArray()));
        return new PointScore(kind.name().toLowerCase(java.util.Locale.ROOT), acc, rot, count(net.parameters()),
                seconds);
    }

    // ---------------------------------------------------------------- group convolution

    /** Merged classes: bar, square, diagonal, plus. */
    public static final int MERGED = 4;
    private static final int[] MERGE = {0, 0, 1, 2, 2, 3};
    private static final ShapeImages IMAGES = new ShapeImages(8, 0.1f);

    /** A classifier for the rotation study. */
    public enum RotationModel {
        /** the VisionShootout CNN */
        CNN,
        /** the CNN trained with random quarter turns */
        CNN_AUGMENTED,
        /** lifting and group convolution, rotation pooling */
        GCNN
    }

    /**
     * The outcome of the rotation study.
     *
     * @param method the model
     * @param upright accuracy on upright test images of the training orientations
     * @param rotated accuracy on test images rotated by 90, 180 or 270 degrees
     * @param parameters the trainable values
     */
    public record RotationScore(String method, double upright, double rotated, long parameters) {
    }

    private static final class GroupCnn implements Layer {
        private final GroupConv2d lift;
        private final GroupConv2d group;
        private final Dense head;

        GroupCnn(int channels, int classes, SplittableRandom r) {
            lift = new GroupConv2d(GroupConv2d.Kind.LIFT, channels, 8, 3, r);
            group = new GroupConv2d(GroupConv2d.Kind.GROUP, 8, 16, 3, r);
            head = new Dense(16, classes, Activation.IDENTITY, Init.XAVIER, r);
        }

        @Override
        public Var forward(Var x) {
            Var h = group.forward(lift.forward(x).relu()).relu();
            h = GroupConv2d.poolRotations(h);
            return head.forward(Functional.globalPool(h, Functional.PoolKind.AVG, false));
        }

        @Override
        public List<Var> parameters() {
            return params(lift, group, head);
        }
    }

    private static Model cnn(int side, int classes, boolean deformable, long seed) {
        Model.Builder b = Model.sequential().input(1, side, side).seed(seed).conv2d(16, 3, Activation.RELU);
        if (deformable) {
            b.add(DeformableConv2d.spec(32, 3)).activation(Activation.RELU);
        } else {
            b.conv2d(32, 3, Activation.RELU);
        }
        return b.globalAvgPool().dense(classes, Activation.IDENTITY).build();
    }

    /**
     * Upright images of the training orientations (horizontal bar, square, main diagonal,
     * plus), with merged labels.
     *
     * @param n the number of images
     * @param seed the seed
     * @return {@code {images [n, 1, 8, 8], labels [n]}}
     */
    public static Tensor[] uprightImages(int n, long seed) {
        float[] img = new float[n * 64];
        float[] lab = new float[n];
        int filled = 0;
        long s = seed;
        while (filled < n) {
            Tensor[] g = IMAGES.generate(64, s++);
            float[] x = g[0].toArray();
            float[] y = g[1].toArray();
            for (int i = 0; i < 64 && filled < n; i++) {
                int c = (int) y[i];
                if (c == 1 || c == 4) {
                    continue;
                }
                System.arraycopy(x, i * 64, img, filled * 64, 64);
                lab[filled++] = MERGE[c];
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(n, 1, 8, 8), img), Tensor.of(Shape.of(n), lab)};
    }

    private static Tensor quarterTurns(Tensor images, SplittableRandom r, boolean nonZero) {
        int n = images.shape().dim(0);
        Var[] parts = new Var[n];
        for (int i = 0; i < n; i++) {
            Tensor one = Var.of(images).slice(0, i, i + 1).value();
            parts[i] = Var.of(GroupConv2d.rotateImages(one, nonZero ? 1 + r.nextInt(3) : r.nextInt(4)));
        }
        return Var.concat(0, parts).value();
    }

    /**
     * Trains a classifier on upright images and tests it on 400 upright and 400 rotated
     * images.
     *
     * @param kind the model
     * @param trainImages the size of a fixed training set, or 0 for fresh images every step
     * @param steps AdamW steps on batches of 32
     * @param seed the seed
     * @return the score
     */
    public static RotationScore rotation(RotationModel kind, int trainImages, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Layer net;
        UnaryOperator<Var> forward;
        if (kind == RotationModel.GCNN) {
            GroupCnn g = new GroupCnn(1, MERGED, r);
            net = g;
            forward = g::forward;
        } else {
            Model m = cnn(8, MERGED, false, seed);
            net = asLayer(m);
            forward = x -> m.forward(x, true);
        }
        Optimizer opt = Optimizer.adamW(3e-3f, 1e-4f);
        Tensor[] fixed = trainImages > 0 ? uprightImages(trainImages, seed + 7) : null;
        for (int s = 0; s < steps; s++) {
            Tensor[] batch;
            if (fixed == null) {
                batch = uprightImages(32, r.nextLong());
            } else {
                int[] pick = new int[32];
                for (int i = 0; i < 32; i++) {
                    pick[i] = r.nextInt(trainImages);
                }
                Tensor idx = Tensor.of(Shape.of(32), toFloats(pick));
                batch = new Tensor[] {fixed[0].gather(0, idx), fixed[1].gather(0, idx)};
            }
            Tensor x = kind == RotationModel.CNN_AUGMENTED ? quarterTurns(batch[0], r, false) : batch[0];
            Functional.crossEntropy(forward.apply(Var.of(x)), batch[1]).backward();
            opt.step(net.parameters());
        }
        Tensor[] test = uprightImages(400, seed + 99);
        Tensor turned = quarterTurns(test[0], new SplittableRandom(seed + 5), true);
        double up = Var.noGrad(() -> accuracy(net.forward(Var.of(test[0])), test[1].toArray()));
        double rot = Var.noGrad(() -> accuracy(net.forward(Var.of(turned)), test[1].toArray()));
        return new RotationScore(kind.name().toLowerCase(java.util.Locale.ROOT), up, rot, count(net.parameters()));
    }

    private static float[] toFloats(int[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = v[i];
        }
        return f;
    }

    // ---------------------------------------------------------------- spatial transformers

    /** Glyph classes: square outline, plus, bar, L, T. */
    public static final int GLYPHS = 5;
    /** The side of the distorted glyph images. */
    public static final int GLYPH_SIDE = 12;
    private static final String[][] GLYPH = {
        {"#####", "#...#", "#...#", "#...#", "#####"},
        {"..#..", "..#..", "#####", "..#..", "..#.."},
        {".....", ".....", "#####", ".....", "....."},
        {"#....", "#....", "#....", "#....", "#####"},
        {"#####", "..#..", "..#..", "..#..", "..#.."}};

    /** A classifier for the distortion study. */
    public enum DistortionModel {
        /** the CNN */
        CNN,
        /** the CNN with a deformable second convolution */
        DEFORMABLE_CNN,
        /** the p4 G-CNN */
        GCNN,
        /** a spatial transformer in front of the CNN */
        STN_CNN
    }

    /**
     * The outcome of the distortion study.
     *
     * @param method the model
     * @param accuracy test accuracy on distorted glyphs
     * @param parameters the trainable values
     * @param transformSpread for the STN, the mean over classes of the per-class variance of
     *     the six transform parameters on test images (low means a consistent canonical
     *     pose); {@code NaN} otherwise
     */
    public record DistortionScore(String method, double accuracy, long parameters, double transformSpread) {
    }

    /**
     * Renders glyphs under random rotation (up to 45 degrees either way), scale (0.6-1.2),
     * translation (up to 2 px) and clutter (three faint blobs), with pixel noise.
     *
     * @param n the number of images
     * @param random the random source
     * @return {@code {images [n, 1, 12, 12], labels [n]}}
     */
    public static Tensor[] distortedGlyphs(int n, SplittableRandom random) {
        int s = GLYPH_SIDE;
        float[] img = new float[n * s * s];
        float[] lab = new float[n];
        for (int i = 0; i < n; i++) {
            int c = i % GLYPHS;
            lab[i] = c;
            double angle = Math.toRadians(random.nextDouble(-45, 45));
            double scale = random.nextDouble(0.6, 1.2) * 1.6;
            double tx = random.nextDouble(-2, 2);
            double ty = random.nextDouble(-2, 2);
            double cos = Math.cos(angle);
            double sin = Math.sin(angle);
            float bright = (float) random.nextDouble(0.7, 1.1);
            int o = i * s * s;
            for (int y = 0; y < s; y++) {
                for (int x = 0; x < s; x++) {
                    double px = x + 0.5 - s / 2.0 - tx;
                    double py = y + 0.5 - s / 2.0 - ty;
                    double u = (cos * px + sin * py) / scale + 2.5;
                    double v = (-sin * px + cos * py) / scale + 2.5;
                    img[o + y * s + x] = bright * glyph(c, u - 0.5, v - 0.5);
                }
            }
            for (int b = 0; b < 3; b++) {
                int bx = random.nextInt(s - 1);
                int by = random.nextInt(s - 1);
                float level = (float) random.nextDouble(0.2, 0.5);
                for (int k = 0; k < 4; k++) {
                    img[o + (by + k / 2) * s + bx + k % 2] += level;
                }
            }
            for (int p = 0; p < s * s; p++) {
                img[o + p] += (float) (0.05 * random.nextGaussian());
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(n, 1, s, s), img), Tensor.of(Shape.of(n), lab)};
    }

    private static float glyph(int c, double u, double v) {
        int x0 = (int) Math.floor(u);
        int y0 = (int) Math.floor(v);
        double fx = u - x0;
        double fy = v - y0;
        return (float) ((1 - fx) * (1 - fy) * cell(c, x0, y0) + fx * (1 - fy) * cell(c, x0 + 1, y0)
                + (1 - fx) * fy * cell(c, x0, y0 + 1) + fx * fy * cell(c, x0 + 1, y0 + 1));
    }

    private static double cell(int c, int x, int y) {
        return x < 0 || y < 0 || x > 4 || y > 4 ? 0 : GLYPH[c][y].charAt(x) == '#' ? 1 : 0;
    }

    /**
     * Trains a classifier on distorted glyphs (16 fresh images per AdamW step) and tests it
     * on 400 more.
     *
     * @param kind the model
     * @param steps steps
     * @param seed the seed
     * @return the score
     */
    public static DistortionScore distortion(DistortionModel kind, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Layer net;
        UnaryOperator<Var> forward;
        SpatialTransformerNetwork stn = null;
        switch (kind) {
            case GCNN -> {
                GroupCnn g = new GroupCnn(1, GLYPHS, r);
                net = g;
                forward = g::forward;
            }
            case STN_CNN -> {
                SpatialTransformerNetwork t = new SpatialTransformerNetwork(1, 8, r);
                Model m = cnn(GLYPH_SIDE, GLYPHS, false, seed);
                stn = t;
                forward = x -> m.forward(t.forward(x), true);
                net = new Layer() {
                    @Override
                    public Var forward(Var input) {
                        return m.forward(t.forward(input), false);
                    }

                    @Override
                    public List<Var> parameters() {
                        List<Var> p = new ArrayList<>(t.parameters());
                        p.addAll(m.parameters());
                        return p;
                    }
                };
            }
            default -> {
                Model m = cnn(GLYPH_SIDE, GLYPHS, kind == DistortionModel.DEFORMABLE_CNN, seed);
                net = asLayer(m);
                forward = x -> m.forward(x, true);
            }
        }
        Optimizer opt = Optimizer.adamW(3e-3f, 1e-4f);
        for (int s = 0; s < steps; s++) {
            Tensor[] batch = distortedGlyphs(16, r);
            Functional.crossEntropy(forward.apply(Var.of(batch[0])), batch[1]).backward();
            opt.step(net.parameters());
        }
        Tensor[] test = distortedGlyphs(400, new SplittableRandom(seed + 99));
        double acc = Var.noGrad(() -> accuracy(net.forward(Var.of(test[0])), test[1].toArray()));
        double spread = Double.NaN;
        if (stn != null) {
            SpatialTransformerNetwork t = stn;
            float[] theta = Var.noGrad(() -> t.transform(Var.of(test[0]))).value().toArray();
            float[] labels = test[1].toArray();
            spread = 0;
            for (int c = 0; c < GLYPHS; c++) {
                for (int q = 0; q < 6; q++) {
                    double sum = 0;
                    double sq = 0;
                    int n = 0;
                    for (int i = 0; i < labels.length; i++) {
                        if ((int) labels[i] == c) {
                            double v = theta[6 * i + q];
                            sum += v;
                            sq += v * v;
                            n++;
                        }
                    }
                    spread += sq / n - (sum / n) * (sum / n);
                }
            }
            spread /= GLYPHS * 6;
        }
        return new DistortionScore(kind.name().toLowerCase(java.util.Locale.ROOT), acc, count(net.parameters()),
                spread);
    }
}
