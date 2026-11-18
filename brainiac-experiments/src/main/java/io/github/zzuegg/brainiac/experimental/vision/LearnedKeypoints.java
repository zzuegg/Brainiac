package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * SuperPoint-style learned keypoints: a small shared convolutional trunk with a detector head
 * (a per-pixel corner logit) and a descriptor head (a unit-length per-pixel vector). Training is
 * MagicPoint-style detector supervision from the synthetic renderer's corners plus homographic
 * adaptation: each step warps the image by a random homography with photometric changes, the
 * detector is trained on both views (class-balanced cross-entropy) and the descriptors of
 * corresponding corners in the two views by an InfoNCE loss against the other corners.
 *
 * <p>Paper: DeTone, Malisiewicz, Rabinovich, "SuperPoint: Self-Supervised Interest Point
 * Detection and Description", CVPR workshops 2018,
 * <a href="https://arxiv.org/abs/1712.07629">arXiv:1712.07629</a>.
 */
public final class LearnedKeypoints {

    private final Var c1;
    private final Var b1;
    private final Var c2;
    private final Var b2;
    private final Var det;
    private final Var desc;
    private final int dim;
    private final List<Var> params;

    /**
     * Creates a network.
     *
     * @param channels the trunk width
     * @param dim the descriptor width
     * @param seed the seed
     */
    public LearnedKeypoints(int channels, int dim, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.dim = dim;
        c1 = VisionOps.convWeight(r, channels / 2, 1, 3);
        b1 = VisionOps.zeros(channels / 2);
        c2 = VisionOps.convWeight(r, channels, channels / 2, 3);
        b2 = VisionOps.zeros(channels);
        det = VisionOps.convWeight(r, 1, channels, 3);
        desc = VisionOps.convWeight(r, dim, channels, 3);
        params = List.of(c1, b1, c2, b2, det, desc);
    }

    private Var[] forward(float[] img, int h, int w) {
        Var x = Var.of(Tensor.of(Shape.of(1, 1, h, w), img));
        Var a = VisionOps.conv(x, c1, b1).relu();
        a = VisionOps.conv(a, c2, b2, 1, 2).relu();
        Var logit = VisionOps.conv(a, det, null).reshape(h * w);
        Var d = VisionOps.conv(a, desc, null).reshape(dim, h * w).transpose();
        d = d.div(d.square().sum(1, true).add(1e-8f).sqrt());
        return new Var[] {logit, d};
    }

    private static int pixel(double[] p, int h, int w) {
        int x = (int) Math.round(p[0]);
        int y = (int) Math.round(p[1]);
        return x < 0 || y < 0 || x >= w || y >= h ? -1 : y * w + x;
    }

    private static Var detectionLoss(Var logit, List<double[]> corners, int h, int w) {
        float[] t = new float[h * w];
        for (double[] c : corners) {
            int p = pixel(c, h, w);
            if (p >= 0) {
                t[p] = 1;
            }
        }
        float pos = 0;
        for (float v : t) {
            pos += v;
        }
        float beta = 1 - pos / t.length;
        float[] wt = new float[t.length];
        for (int i = 0; i < t.length; i++) {
            wt[i] = t[i] > 0 ? beta : 1 - beta;
        }
        Var p = logit.sigmoid();
        Tensor y = Tensor.of(Shape.of(t.length), t);
        Var l = p.add(1e-6f).log().mul(y).add(p.mul(-1f).add(1f + 1e-6f).log().mul(Var.of(y).mul(-1f).add(1f)));
        return l.mul(Tensor.of(Shape.of(t.length), wt)).sum().mul(-1f / Math.max(1, pos));
    }

    /**
     * Random homography: the four image corners moved by up to {@code shift} pixels.
     *
     * @param h rows
     * @param w columns
     * @param shift the largest corner displacement
     * @param r the random source
     * @return the homography
     */
    public static double[] randomHomography(int h, int w, double shift, SplittableRandom r) {
        List<double[]> src = List.of(new double[] {0, 0}, new double[] {w - 1, 0}, new double[] {0, h - 1},
                new double[] {w - 1, h - 1});
        List<double[]> dst = new ArrayList<>();
        for (double[] p : src) {
            dst.add(new double[] {p[0] + shift * (2 * r.nextDouble() - 1), p[1] + shift * (2 * r.nextDouble() - 1)});
        }
        return FeatureMatching.dlt(src, dst);
    }

    /**
     * Trains.
     *
     * @param images training images
     * @param corners per image its true corners
     * @param h rows
     * @param w columns
     * @param steps steps (one image pair each)
     * @param shift the homography corner displacement
     * @param lr the learning rate
     * @param seed the seed
     */
    public void train(List<float[]> images, List<List<double[]>> corners, int h, int w, int steps, double shift,
            float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(lr);
        for (int s = 0; s < steps; s++) {
            int i = r.nextInt(images.size());
            double[] hm = randomHomography(h, w, shift, r);
            float[] warped = FeatureMatching.warp(images.get(i), h, w, hm, 0.3f);
            float gain = (float) (0.7 + 0.6 * r.nextDouble());
            double sigma = r.nextDouble() < 0.5 ? 0 : 0.5 + r.nextDouble();
            if (sigma > 0) {
                warped = Conv2dOps.gaussianBlur(warped, h, w, sigma);
            }
            for (int p = 0; p < warped.length; p++) {
                warped[p] = warped[p] * gain + (float) (0.03 * r.nextGaussian());
            }
            List<double[]> c1s = corners.get(i);
            List<double[]> c2s = new ArrayList<>();
            List<Integer> a = new ArrayList<>();
            List<Integer> b = new ArrayList<>();
            for (double[] c : c1s) {
                double[] q = FeatureMatching.apply(hm, c);
                c2s.add(q);
                int pa = pixel(c, h, w);
                int pb = pixel(q, h, w);
                if (pa >= 0 && pb >= 0) {
                    a.add(pa);
                    b.add(pb);
                }
            }
            Var[] f1 = forward(images.get(i), h, w);
            Var[] f2 = forward(warped, h, w);
            Var loss = detectionLoss(f1[0], c1s, h, w).add(detectionLoss(f2[0], c2s, h, w));
            if (a.size() >= 2) {
                Var da = f1[1].gather(0, ids(a));
                Var db = f2[1].gather(0, ids(b));
                float[] diag = new float[a.size()];
                for (int k = 0; k < diag.length; k++) {
                    diag[k] = k;
                }
                loss = loss.add(Functional.crossEntropy(da.matmul(db.transpose()).div(0.1f),
                        Tensor.of(Shape.of(diag.length), diag)));
            }
            params.forEach(Var::zeroGrad);
            loss.backward();
            opt.step(params);
        }
        params.forEach(Var::zeroGrad);
    }

    private static Tensor ids(List<Integer> v) {
        float[] f = new float[v.size()];
        for (int i = 0; i < f.length; i++) {
            f[i] = v.get(i);
        }
        return Tensor.of(Shape.of(f.length), f);
    }

    /**
     * Keypoints and their descriptors.
     *
     * @param points {@code {x, y}} per keypoint
     * @param descriptors per keypoint its unit descriptor
     */
    public record Detection(List<double[]> points, float[][] descriptors) {
    }

    /**
     * Detects and describes.
     *
     * @param img row-major {@code h x w}
     * @param h rows
     * @param w columns
     * @param count the most keypoints
     * @param radius the suppression radius
     * @return the detection
     */
    public Detection detect(float[] img, int h, int w, int count, int radius) {
        Var[] f = Var.noGrad(() -> forward(img, h, w));
        float[] score = f[0].sigmoid().value().toArray();
        float[] d = f[1].value().toArray();
        List<double[]> pts = FeatureMatching.keypoints(score, h, w, count, radius, 3);
        float[][] desc = new float[pts.size()][dim];
        for (int k = 0; k < pts.size(); k++) {
            int p = (int) pts.get(k)[1] * w + (int) pts.get(k)[0];
            System.arraycopy(d, p * dim, desc[k], 0, dim);
        }
        return new Detection(pts, desc);
    }
}
