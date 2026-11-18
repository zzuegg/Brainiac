package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.data.VectorIndex;
import io.github.zzuegg.brainiac.experimental.data.GlyphDocuments;
import io.github.zzuegg.brainiac.experimental.recsys.IvfPqIndex;
import io.github.zzuegg.brainiac.experimental.vision.DeepHashing;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 11 item 11.5.16: image retrieval with binary codes on noisy 12 x 12 renderings of the
 * ten digit glyphs. Hashers are trained on seven seen classes; the database and queries hold
 * all ten, so the three unseen classes test generalisation. DSH and HashNet codes against
 * random-hyperplane LSH and ITQ (PCA plus learned rotation) codes of metric-learned float
 * embeddings (64 real DSH outputs before binarisation), product quantisation ({@link IvfPqIndex}) of those embeddings at the same bytes,
 * and the float embeddings themselves (cosine, the uncompressed ceiling). mAP over the whole
 * database, precision at 50 and microseconds per query.
 */
public final class HashingRetrieval {

    private HashingRetrieval() {
    }

    /**
     * A row.
     *
     * @param bits code length (0 for float embeddings)
     * @param method the method
     * @param seenMap mAP of seen-class queries
     * @param unseenMap mAP of unseen-class queries
     * @param precision precision at 50 over all queries
     * @param micros microseconds per query
     */
    public record Row(int bits, String method, double seenMap, double unseenMap, double precision, double micros) {
    }

    private static final int SIDE = 12;
    private static final int SEEN = 7;

    private static float[] digit(int c, SplittableRandom r) {
        float[] g = GlyphDocuments.glyph((char) ('0' + c));
        float[] img = new float[SIDE * SIDE];
        int ox = r.nextInt(SIDE - 6 + 1);
        int oy = r.nextInt(SIDE - 10 + 1);
        float ink = 0.7f + 0.3f * r.nextFloat();
        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 6; x++) {
                if (g[(y / 2) * GlyphDocuments.GLYPH_WIDTH + x / 2] > 0 && r.nextDouble() > 0.1) {
                    img[(oy + y) * SIDE + ox + x] = ink;
                }
            }
        }
        for (int i = 0; i < img.length; i++) {
            img[i] += (float) (0.2 * r.nextGaussian());
        }
        return img;
    }

    private static float[][] unit(float[][] x) {
        for (float[] row : x) {
            double n = 0;
            for (float v : row) {
                n += v * v;
            }
            float inv = (float) (1 / Math.sqrt(n + 1e-12));
            for (int j = 0; j < row.length; j++) {
                row[j] *= inv;
            }
        }
        return x;
    }

    // ------------------------------------------------------------ ITQ helpers

    private static double[][] mul(double[][] a, double[][] b) {
        int n = a.length;
        int m = b[0].length;
        int k = b.length;
        double[][] c = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < k; t++) {
                double v = a[i][t];
                for (int j = 0; j < m; j++) {
                    c[i][j] += v * b[t][j];
                }
            }
        }
        return c;
    }

    private static double[][] transpose(double[][] a) {
        double[][] t = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                t[j][i] = a[i][j];
            }
        }
        return t;
    }

    /** Gram-Schmidt on the columns. */
    private static double[][] orthonormalise(double[][] a) {
        int n = a.length;
        int m = a[0].length;
        for (int j = 0; j < m; j++) {
            for (int p = 0; p < j; p++) {
                double dot = 0;
                for (int i = 0; i < n; i++) {
                    dot += a[i][j] * a[i][p];
                }
                for (int i = 0; i < n; i++) {
                    a[i][j] -= dot * a[i][p];
                }
            }
            double norm = 0;
            for (int i = 0; i < n; i++) {
                norm += a[i][j] * a[i][j];
            }
            norm = Math.sqrt(norm + 1e-18);
            for (int i = 0; i < n; i++) {
                a[i][j] /= norm;
            }
        }
        return a;
    }

    private static double[][] inverse(double[][] m) {
        int n = m.length;
        double[][] a = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], 0, a[i], 0, n);
            a[i][n + i] = 1;
        }
        for (int c = 0; c < n; c++) {
            int piv = c;
            for (int i = c + 1; i < n; i++) {
                if (Math.abs(a[i][c]) > Math.abs(a[piv][c])) {
                    piv = i;
                }
            }
            double[] t = a[c];
            a[c] = a[piv];
            a[piv] = t;
            double v = a[c][c];
            if (Math.abs(v) < 1e-12) {
                v = 1e-12;
            }
            for (int j = 0; j < 2 * n; j++) {
                a[c][j] /= v;
            }
            for (int i = 0; i < n; i++) {
                if (i != c) {
                    double f = a[i][c];
                    for (int j = 0; j < 2 * n; j++) {
                        a[i][j] -= f * a[c][j];
                    }
                }
            }
        }
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], n, out[i], 0, n);
        }
        return out;
    }

    /** The orthogonal polar factor by Newton iteration. */
    private static double[][] polar(double[][] m) {
        double norm = 0;
        for (double[] row : m) {
            for (double v : row) {
                norm += v * v;
            }
        }
        double s = 1 / Math.sqrt(norm / m.length + 1e-18);
        double[][] x = new double[m.length][m.length];
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m.length; j++) {
                x[i][j] = m[i][j] * s;
            }
        }
        for (int it = 0; it < 20; it++) {
            double[][] inv = transpose(inverse(x));
            for (int i = 0; i < m.length; i++) {
                for (int j = 0; j < m.length; j++) {
                    x[i][j] = 0.5 * (x[i][j] + inv[i][j]);
                }
            }
        }
        return x;
    }

    /** ITQ: {@code {mean, projection [dim][bits] (PCA times rotation)}}. */
    private static double[][][] itq(float[][] x, int bits, long seed) {
        int n = x.length;
        int d = x[0].length;
        double[][] mean = new double[1][d];
        for (float[] row : x) {
            for (int j = 0; j < d; j++) {
                mean[0][j] += row[j] / n;
            }
        }
        double[][] c = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                c[i][j] = x[i][j] - mean[0][j];
            }
        }
        double[][] cov = mul(transpose(c), c);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] v = new double[d][bits];
        for (double[] row : v) {
            for (int j = 0; j < bits; j++) {
                row[j] = r.nextGaussian();
            }
        }
        v = orthonormalise(v);
        for (int it = 0; it < 40; it++) {
            v = orthonormalise(mul(cov, v));
        }
        double[][] p = mul(c, v);
        double[][] rot = new double[bits][bits];
        for (double[] row : rot) {
            for (int j = 0; j < bits; j++) {
                row[j] = r.nextGaussian();
            }
        }
        rot = orthonormalise(rot);
        for (int it = 0; it < 30; it++) {
            double[][] b = mul(p, rot);
            for (double[] row : b) {
                for (int j = 0; j < bits; j++) {
                    row[j] = row[j] >= 0 ? 1 : -1;
                }
            }
            rot = polar(mul(transpose(p), b));
        }
        return new double[][][] {mean, mul(v, rot)};
    }

    private static float[][] project(float[][] x, double[] mean, double[][] w) {
        float[][] out = new float[x.length][w[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int k = 0; k < w[0].length; k++) {
                double s = 0;
                for (int j = 0; j < w.length; j++) {
                    s += (x[i][j] - mean[j]) * w[j][k];
                }
                out[i][k] = (float) s;
            }
        }
        return out;
    }

    // ------------------------------------------------------------ evaluation

    private interface Ranker {
        int[] rank(int query);
    }

    private static Row evaluate(int bits, String name, Ranker ranker, int[] dbLabels, int[] qLabels) {
        double seen = 0;
        double unseen = 0;
        int ns = 0;
        int nu = 0;
        double prec = 0;
        long t0 = System.nanoTime();
        List<int[]> ranks = new ArrayList<>();
        for (int q = 0; q < qLabels.length; q++) {
            ranks.add(ranker.rank(q));
        }
        double micros = (System.nanoTime() - t0) / 1e3 / qLabels.length;
        for (int q = 0; q < qLabels.length; q++) {
            int[] rank = ranks.get(q);
            int hits = 0;
            double ap = 0;
            int at50 = 0;
            for (int k = 0; k < rank.length; k++) {
                if (dbLabels[rank[k]] == qLabels[q]) {
                    hits++;
                    ap += hits / (double) (k + 1);
                    if (k < 50) {
                        at50++;
                    }
                }
            }
            ap /= Math.max(1, hits);
            prec += at50 / 50.0 / qLabels.length;
            if (qLabels[q] < SEEN) {
                seen += ap;
                ns++;
            } else {
                unseen += ap;
                nu++;
            }
        }
        return new Row(bits, name, seen / ns, unseen / nu, prec, micros);
    }

    private static int[] cosineRank(float[][] db, float[] q) {
        Integer[] idx = new Integer[db.length];
        double[] s = new double[db.length];
        for (int i = 0; i < db.length; i++) {
            idx[i] = i;
            for (int j = 0; j < q.length; j++) {
                s[i] -= db[i][j] * q[j];
            }
        }
        java.util.Arrays.sort(idx, (a, b) -> Double.compare(s[a], s[b]));
        return java.util.Arrays.stream(idx).mapToInt(Integer::intValue).toArray();
    }

    /**
     * Runs the comparison.
     *
     * @param trainPerClass training images per seen class
     * @param dbPerClass database images per class
     * @param queriesPerClass queries per class
     * @param bitsList code lengths (multiples of 8 dividing 1024)
     * @param steps training steps
     * @param seed the seed
     * @return rows
     */
    public static List<Row> run(int trainPerClass, int dbPerClass, int queriesPerClass, int[] bitsList, int steps,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] tx = new float[SEEN * trainPerClass][];
        int[] ty = new int[tx.length];
        for (int i = 0; i < tx.length; i++) {
            ty[i] = i % SEEN;
            tx[i] = digit(ty[i], r);
        }
        float[][] db = new float[10 * dbPerClass][];
        int[] dy = new int[db.length];
        for (int i = 0; i < db.length; i++) {
            dy[i] = i % 10;
            db[i] = digit(dy[i], r);
        }
        float[][] qx = new float[10 * queriesPerClass][];
        int[] qy = new int[qx.length];
        for (int i = 0; i < qx.length; i++) {
            qy[i] = i % 10;
            qx[i] = digit(qy[i], r);
        }
        DeepHashing metric = new DeepHashing(DeepHashing.Loss.DSH, SIDE * SIDE, 128, 64, seed + 1);
        metric.train(tx, ty, steps, 64, 2e-3f, seed + 1);
        float[][] fdb = unit(metric.embed(db));
        float[][] fq = unit(metric.embed(qx));
        float[][] ftr = unit(metric.embed(tx));
        List<Row> rows = new ArrayList<>();
        rows.add(evaluate(0, "float embedding (cosine)", q -> cosineRank(fdb, fq[q]), dy, qy));
        int dim = fdb[0].length;
        for (int bits : bitsList) {
            SplittableRandom lr = new SplittableRandom(seed + bits);
            double[][] planes = new double[dim][bits];
            for (double[] row : planes) {
                for (int k = 0; k < bits; k++) {
                    row[k] = lr.nextGaussian();
                }
            }
            double[] mean = itq(ftr, bits, seed)[0][0];
            DeepHashing.Codes lshDb = DeepHashing.Codes.ofSigns(project(fdb, mean, planes));
            DeepHashing.Codes lshQ = DeepHashing.Codes.ofSigns(project(fq, mean, planes));
            rows.add(evaluate(bits, "LSH", q -> lshDb.rank(lshQ, q), dy, qy));
            double[][][] it = itq(ftr, bits, seed);
            DeepHashing.Codes itqDb = DeepHashing.Codes.ofSigns(project(fdb, it[0][0], it[1]));
            DeepHashing.Codes itqQ = DeepHashing.Codes.ofSigns(project(fq, it[0][0], it[1]));
            rows.add(evaluate(bits, "ITQ", q -> itqDb.rank(itqQ, q), dy, qy));
            for (DeepHashing.Loss l : DeepHashing.Loss.values()) {
                DeepHashing h = new DeepHashing(l, SIDE * SIDE, 128, bits, seed);
                h.train(tx, ty, steps, 64, 2e-3f, seed);
                DeepHashing.Codes cdb = h.encode(db);
                DeepHashing.Codes cq = h.encode(qx);
                rows.add(evaluate(bits, l.name(), q -> cdb.rank(cq, q), dy, qy));
            }
            IvfPqIndex pq = IvfPqIndex.of(dim, VectorIndex.Metric.DOT, 1, bits / 8, 256, 1, 1, seed);
            float[] all = new float[fdb.length * dim];
            for (int i = 0; i < fdb.length; i++) {
                System.arraycopy(fdb[i], 0, all, i * dim, dim);
            }
            pq.add(Tensor.of(Shape.of(fdb.length, dim), all));
            rows.add(evaluate(bits, "PQ (" + pq.bytesPerVector() + " B)", q -> {
                List<VectorIndex.Hit> hits = pq.search(Tensor.of(Shape.of(dim), fq[q]), fdb.length);
                return hits.stream().mapToInt(VectorIndex.Hit::id).toArray();
            }, dy, qy));
        }
        return rows;
    }
}
