package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.anomaly.DeepSvdd;
import io.github.zzuegg.brainiac.experimental.anomaly.Dinomaly;
import io.github.zzuegg.brainiac.experimental.anomaly.Ecod;
import io.github.zzuegg.brainiac.experimental.anomaly.ImageAnomalyMetrics;
import io.github.zzuegg.brainiac.experimental.anomaly.PatchCore;
import io.github.zzuegg.brainiac.experimental.anomaly.PatchDistribution;
import io.github.zzuegg.brainiac.experimental.data.DefectImages;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.vision.VisionOps;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 11 items 11.5.11-11.5.12 on {@link DefectImages} (24 px, five categories).
 *
 * <p>Features come from a small CNN pre-trained to classify {@link ShapeImages} strokes and then
 * frozen: three layers on a {@code 12 x 12} token grid (16, 24 and 24 channels).
 *
 * <ul>
 *   <li>per category: PatchCore and PaDiM against Deep SVDD, ECOD and nearest-neighbour
 *       distance on mean-pooled image embeddings and a dense autoencoder's pixel
 *       reconstruction error;
 *   <li>multi-class (one model for all categories): pooled PatchCore and PaDiM, Dinomaly and
 *       its ablation (softmax attention, no dropout, layer-wise loss).
 * </ul>
 *
 * <p>Image AUROC, pixel AUROC and PRO (FPR up to 0.3), and the image AUROC on misplacements only.
 */
public final class ImageAnomalyShootout {

    private ImageAnomalyShootout() {
    }

    /**
     * A row.
     *
     * @param setting "per category" (means over categories) or "multi-class"
     * @param method the method
     * @param image image AUROC
     * @param pixel pixel AUROC ({@code NaN} for image-level methods)
     * @param pro PRO area ({@code NaN} for image-level methods)
     * @param misplaced image AUROC on normal against misplaced images
     */
    public record Row(String setting, String method, double image, double pixel, double pro, double misplaced) {
    }

    private static final int SIDE = 24;
    private static final int GRID = 12;
    private static final int[] LAYERS = {16, 24, 24};

    /** The frozen feature extractor. */
    private static final class Extractor {
        private final Var w1;
        private final Var w2;
        private final Var w3;
        private final Var out;
        private final Var b1 = VisionOps.zeros(LAYERS[0]);
        private final Var b2 = VisionOps.zeros(LAYERS[1]);
        private final Var b3 = VisionOps.zeros(LAYERS[2]);

        Extractor(int steps, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            w1 = VisionOps.convWeight(r, LAYERS[0], 1, 3);
            w2 = VisionOps.convWeight(r, LAYERS[1], LAYERS[0], 3);
            w3 = VisionOps.convWeight(r, LAYERS[2], LAYERS[1], 3);
            out = Var.param(Tensor.normal(Shape.of(LAYERS[2], ShapeImages.CLASSES), 0, 0.2f, r));
            ShapeImages gen = new ShapeImages(SIDE, 0.1f);
            List<Var> p = List.of(w1, w2, w3, b1, b2, b3, out);
            Optimizer opt = Optimizer.adam(3e-3f);
            for (int s = 0; s < steps; s++) {
                Tensor[] batch = gen.generate(16, seed + s);
                Var[] f = layers(Var.of(batch[0]));
                Var logits = f[2].mean(3, false).mean(2, false).matmul(out);
                p.forEach(Var::zeroGrad);
                Functional.crossEntropy(logits, batch[1]).backward();
                opt.step(p);
            }
            p.forEach(Var::zeroGrad);
        }

        Var[] layers(Var x) {
            int b = x.shape().dim(0);
            Var a = VisionOps.conv(x, w1, b1).relu();
            Var a1 = a.reshape(b, LAYERS[0], GRID, 2, GRID, 2).mean(5, false).mean(3, false);
            Var a2 = VisionOps.conv(a1, w2, b2).relu();
            Var a3 = VisionOps.conv(a2, w3, b3, 1, 2).relu();
            return new Var[] {a1, a2, a3};
        }

        /** Per layer, per image its {@code [tokens][dims]}. */
        List<List<float[][]>> features(List<DefectImages.Sample> images) {
            List<List<float[][]>> out = new ArrayList<>();
            for (int l = 0; l < LAYERS.length; l++) {
                out.add(new ArrayList<>());
            }
            for (int from = 0; from < images.size(); from += 32) {
                int to = Math.min(images.size(), from + 32);
                float[] px = new float[(to - from) * SIDE * SIDE];
                for (int i = from; i < to; i++) {
                    System.arraycopy(images.get(i).pixels(), 0, px, (i - from) * SIDE * SIDE, SIDE * SIDE);
                }
                Tensor batch = Tensor.of(Shape.of(to - from, 1, SIDE, SIDE), px);
                Var[] f = Var.noGrad(() -> layers(Var.of(batch)));
                for (int l = 0; l < LAYERS.length; l++) {
                    float[] v = f[l].value().toArray();
                    int c = LAYERS[l];
                    for (int i = 0; i < to - from; i++) {
                        float[][] tok = new float[GRID * GRID][c];
                        for (int ch = 0; ch < c; ch++) {
                            for (int t = 0; t < GRID * GRID; t++) {
                                tok[t][ch] = v[(i * c + ch) * GRID * GRID + t];
                            }
                        }
                        out.get(l).add(tok);
                    }
                }
            }
            return out;
        }
    }

    private static List<float[][]> concat(List<List<float[][]>> layers, boolean aggregate) {
        List<float[][]> out = new ArrayList<>();
        int d = Arrays.stream(LAYERS).sum();
        for (int i = 0; i < layers.get(0).size(); i++) {
            float[][] tok = new float[GRID * GRID][d];
            int off = 0;
            for (int l = 0; l < LAYERS.length; l++) {
                for (int t = 0; t < GRID * GRID; t++) {
                    System.arraycopy(layers.get(l).get(i)[t], 0, tok[t], off, LAYERS[l]);
                }
                off += LAYERS[l];
            }
            out.add(aggregate ? PatchCore.aggregate(tok, GRID) : tok);
        }
        return out;
    }

    private static double[] upsample(double[] map) {
        double[] out = new double[SIDE * SIDE];
        int f = SIDE / GRID;
        for (int y = 0; y < SIDE; y++) {
            for (int x = 0; x < SIDE; x++) {
                out[y * SIDE + x] = map[(y / f) * GRID + x / f];
            }
        }
        return out;
    }

    private static double[][] embeddings(List<float[][]> tokens) {
        double[][] out = new double[tokens.size()][];
        for (int i = 0; i < tokens.size(); i++) {
            float[][] t = tokens.get(i);
            double[] e = new double[t[0].length];
            for (float[] row : t) {
                for (int k = 0; k < e.length; k++) {
                    e[k] += row[k] / t.length;
                }
            }
            out[i] = e;
        }
        return out;
    }

    /** Scores for one test set. */
    private record Result(double[] image, List<double[]> maps) {
    }

    private static Row row(String setting, String name, List<DefectImages.Sample> test, Result r) {
        boolean[] y = new boolean[test.size()];
        List<boolean[]> masks = new ArrayList<>();
        List<Double> ms = new ArrayList<>();
        List<Boolean> my = new ArrayList<>();
        for (int i = 0; i < y.length; i++) {
            DefectImages.Sample s = test.get(i);
            y[i] = s.anomalous();
            masks.add(s.mask());
            if (!s.anomalous() || s.defect() == DefectImages.Defect.MISPLACED) {
                ms.add(r.image()[i]);
                my.add(s.anomalous());
            }
        }
        boolean[] myb = new boolean[my.size()];
        for (int i = 0; i < myb.length; i++) {
            myb[i] = my.get(i);
        }
        double pixel = r.maps() == null ? Double.NaN : ImageAnomalyMetrics.pixelAuroc(r.maps(), masks);
        double pro = r.maps() == null ? Double.NaN : ImageAnomalyMetrics.pro(r.maps(), masks, 0.3);
        return new Row(setting, name, ImageAnomalyMetrics.auroc(r.image(), y), pixel, pro,
                ImageAnomalyMetrics.auroc(ms.stream().mapToDouble(Double::doubleValue).toArray(), myb));
    }

    private static Result patchCore(List<float[][]> train, List<float[][]> test, double ratio, long seed) {
        PatchCore pc = PatchCore.fit(train, ratio, seed);
        double[] img = new double[test.size()];
        List<double[]> maps = new ArrayList<>();
        for (int i = 0; i < test.size(); i++) {
            double[] m = pc.map(test.get(i));
            img[i] = Arrays.stream(m).max().orElse(0);
            maps.add(upsample(m));
        }
        return new Result(img, maps);
    }

    private static Result padim(List<float[][]> train, List<float[][]> test, long seed) {
        PatchDistribution pd = PatchDistribution.fit(train, 0, 0.01, seed);
        double[] img = new double[test.size()];
        List<double[]> maps = new ArrayList<>();
        for (int i = 0; i < test.size(); i++) {
            double[] m = pd.map(test.get(i));
            img[i] = Arrays.stream(m).max().orElse(0);
            maps.add(upsample(m));
        }
        return new Result(img, maps);
    }

    private static Result knn(double[][] train, double[][] test) {
        double[] s = new double[test.length];
        for (int i = 0; i < test.length; i++) {
            double best = Double.MAX_VALUE;
            for (double[] t : train) {
                double d = 0;
                for (int k = 0; k < t.length; k++) {
                    d += (t[k] - test[i][k]) * (t[k] - test[i][k]);
                }
                best = Math.min(best, d);
            }
            s[i] = best;
        }
        return new Result(s, null);
    }

    private static Result autoencoder(List<DefectImages.Sample> train, List<DefectImages.Sample> test, int steps,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = SIDE * SIDE;
        int[] widths = {d, 128, 24, 128, d};
        List<Var> w = new ArrayList<>();
        for (int i = 0; i + 1 < widths.length; i++) {
            w.add(Var.param(Tensor.normal(Shape.of(widths[i], widths[i + 1]), 0,
                    (float) Math.sqrt(1.0 / widths[i]), r)));
        }
        Optimizer opt = Optimizer.adam(2e-3f);
        for (int s = 0; s < steps; s++) {
            float[] x = new float[32 * d];
            for (int i = 0; i < 32; i++) {
                System.arraycopy(train.get(r.nextInt(train.size())).pixels(), 0, x, i * d, d);
            }
            Var in = Var.of(Tensor.of(Shape.of(32, d), x));
            w.forEach(Var::zeroGrad);
            forward(in, w).sub(in).square().mean().backward();
            opt.step(w);
        }
        w.forEach(Var::zeroGrad);
        double[] img = new double[test.size()];
        List<double[]> maps = new ArrayList<>();
        for (int i = 0; i < test.size(); i++) {
            float[] x = test.get(i).pixels();
            Var in = Var.of(Tensor.of(Shape.of(1, d), x));
            float[] y = Var.noGrad(() -> forward(in, w)).value().toArray();
            double[] m = new double[d];
            for (int p = 0; p < d; p++) {
                m[p] = (y[p] - x[p]) * (y[p] - x[p]);
            }
            // image score: the largest 3 x 3 mean error
            double best = 0;
            for (int yy = 1; yy + 1 < SIDE; yy++) {
                for (int xx = 1; xx + 1 < SIDE; xx++) {
                    double s = 0;
                    for (int a = -1; a <= 1; a++) {
                        for (int b = -1; b <= 1; b++) {
                            s += m[(yy + a) * SIDE + xx + b];
                        }
                    }
                    best = Math.max(best, s);
                }
            }
            img[i] = best;
            maps.add(m);
        }
        return new Result(img, maps);
    }

    private static Var forward(Var x, List<Var> w) {
        Var h = x;
        for (int i = 0; i < w.size(); i++) {
            h = h.matmul(w.get(i));
            if (i + 1 < w.size()) {
                h = h.tanh();
            }
        }
        return h;
    }

    private static Result dinomaly(Dinomaly.Config c, List<List<float[][]>> train, List<List<float[][]>> test,
            int steps, long seed) {
        Dinomaly d = new Dinomaly(LAYERS, 32, 2, 4, c, seed);
        d.train(train, steps, 4, 3e-3f, seed);
        int n = test.get(0).size();
        double[] img = new double[n];
        List<double[]> maps = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            double[] m = d.map(test, i);
            double[] sorted = m.clone();
            Arrays.sort(sorted);
            img[i] = (sorted[sorted.length - 1] + sorted[sorted.length - 2] + sorted[sorted.length - 3]) / 3;
            maps.add(upsample(m));
        }
        return new Result(img, maps);
    }

    /** Per layer and channel, z-scores with the training statistics (layer-norm-like inputs). */
    private static List<List<float[][]>> standardise(List<List<float[][]>> train, List<List<float[][]>> x) {
        List<List<float[][]>> out = new ArrayList<>();
        for (int l = 0; l < LAYERS.length; l++) {
            int c = LAYERS[l];
            double[] mean = new double[c];
            double[] sq = new double[c];
            long n = 0;
            for (float[][] im : train.get(l)) {
                for (float[] t : im) {
                    for (int k = 0; k < c; k++) {
                        mean[k] += t[k];
                        sq[k] += t[k] * t[k];
                    }
                    n++;
                }
            }
            float[] mu = new float[c];
            float[] sd = new float[c];
            for (int k = 0; k < c; k++) {
                mu[k] = (float) (mean[k] / n);
                sd[k] = (float) Math.sqrt(Math.max(1e-6, sq[k] / n - mu[k] * mu[k]));
            }
            List<float[][]> layer = new ArrayList<>();
            for (float[][] im : x.get(l)) {
                float[][] z = new float[im.length][c];
                for (int t = 0; t < im.length; t++) {
                    for (int k = 0; k < c; k++) {
                        z[t][k] = (im[t][k] - mu[k]) / sd[k];
                    }
                }
                layer.add(z);
            }
            out.add(layer);
        }
        return out;
    }

    private static List<List<float[][]>> merge(List<List<List<float[][]>>> parts) {
        List<List<float[][]>> out = new ArrayList<>();
        for (int l = 0; l < LAYERS.length; l++) {
            List<float[][]> all = new ArrayList<>();
            for (List<List<float[][]>> p : parts) {
                all.addAll(p.get(l));
            }
            out.add(all);
        }
        return out;
    }

    /**
     * Runs the shoot-out.
     *
     * @param normals normal training images per category
     * @param testNormals normal test images per category
     * @param perDefect test images per defect type per category
     * @param steps training steps of each learned model
     * @param seed the seed
     * @return rows
     */
    public static List<Row> run(int normals, int testNormals, int perDefect, int steps, long seed) {
        Extractor ex = new Extractor(Math.min(steps, 80), seed);
        DefectImages gen = new DefectImages(SIDE, 0.05f);
        String[] names = {"PatchCore", "PaDiM", "Deep SVDD", "ECOD", "kNN embedding", "autoencoder"};
        double[][] sums = new double[names.length][4];
        List<List<List<float[][]>>> trainParts = new ArrayList<>();
        List<List<List<float[][]>>> testParts = new ArrayList<>();
        List<DefectImages.Sample> allTest = new ArrayList<>();
        for (int c = 0; c < DefectImages.CATEGORIES; c++) {
            List<DefectImages.Sample> tr = gen.normal(c, normals, seed);
            List<DefectImages.Sample> te = gen.test(c, testNormals, perDefect, seed);
            List<List<float[][]>> ftr = ex.features(tr);
            List<List<float[][]>> fte = ex.features(te);
            trainParts.add(ftr);
            testParts.add(fte);
            allTest.addAll(te);
            List<float[][]> atr = concat(ftr, true);
            List<float[][]> ate = concat(fte, true);
            double[][] etr = embeddings(concat(ftr, false));
            double[][] ete = embeddings(concat(fte, false));
            DeepSvdd svdd = new DeepSvdd(etr[0].length, new int[] {32, 16}, seed);
            svdd.fit(etr, null, 1f, steps, 1e-3f, seed);
            Result[] res = {patchCore(atr, ate, 0.1, seed), padim(atr, ate, seed), new Result(svdd.score(ete), null),
                    new Result(Ecod.fit(etr).score(ete), null), knn(etr, ete), autoencoder(tr, te, steps, seed)};
            for (int m = 0; m < names.length; m++) {
                Row r = row("per category", names[m], te, res[m]);
                sums[m][0] += r.image() / DefectImages.CATEGORIES;
                sums[m][1] += r.pixel() / DefectImages.CATEGORIES;
                sums[m][2] += r.pro() / DefectImages.CATEGORIES;
                sums[m][3] += r.misplaced() / DefectImages.CATEGORIES;
            }
        }
        List<Row> rows = new ArrayList<>();
        for (int m = 0; m < names.length; m++) {
            rows.add(new Row("per category", names[m], sums[m][0], sums[m][1], sums[m][2], sums[m][3]));
        }
        List<List<float[][]>> mtr = merge(trainParts);
        List<List<float[][]>> mte = merge(testParts);
        List<float[][]> atr = concat(mtr, true);
        List<float[][]> ate = concat(mte, true);
        rows.add(row("multi-class", "PatchCore pooled", allTest, patchCore(atr, ate, 0.02, seed)));
        rows.add(row("multi-class", "PaDiM pooled", allTest, padim(atr, ate, seed)));
        List<List<float[][]>> ztr = standardise(mtr, mtr);
        List<List<float[][]>> zte = standardise(mtr, mte);
        int ds = steps / 2;
        rows.add(row("multi-class", "Dinomaly", allTest, dinomaly(Dinomaly.Config.dinomaly(), ztr, zte, ds, seed)));
        rows.add(row("multi-class", "plain reconstruction (ablation)", allTest,
                dinomaly(Dinomaly.Config.plain(), ztr, zte, ds, seed)));
        rows.add(row("multi-class", "Dinomaly softmax attention", allTest,
                dinomaly(new Dinomaly.Config(false, 0.2f, true, 0.1), ztr, zte, ds, seed)));
        return rows;
    }
}
