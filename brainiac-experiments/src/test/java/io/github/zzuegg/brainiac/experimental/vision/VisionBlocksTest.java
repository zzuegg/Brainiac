package io.github.zzuegg.brainiac.experimental.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeScenes;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class VisionBlocksTest {

    private static float[] texture(int w, int h, double dx, double dy) {
        float[] img = new float[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double u = x - dx;
                double v = y - dy;
                img[y * w + x] = (float) (0.5 + 0.2 * Math.sin(0.7 * u + 0.3 * v) + 0.2 * Math.cos(0.5 * v - 0.4 * u));
            }
        }
        return img;
    }

    @Test
    void identityTransformReproducesTheImage() {
        Tensor img = Tensor.normal(Shape.of(2, 3, 5, 6), 0, 1, new SplittableRandom(1));
        Var grid = AffineSampler.affineGrid(Var.of(AffineSampler.identity(2)), 5, 6);
        assertArrayEquals(img.toArray(), AffineSampler.sample(Var.of(img), grid).value().toArray(), 1e-5f);
    }

    @Test
    void bilinearSamplingGradientsMatchFiniteDifferences() {
        SplittableRandom r = new SplittableRandom(2);
        Tensor img = Tensor.normal(Shape.of(1, 2, 4, 4), 0, 1, r);
        float[] g = new float[3 * 3 * 2];
        for (int i = 0; i < g.length; i++) {
            g[i] = -0.8f + 0.37f * (i % 5) + 0.05f;
        }
        Tensor grid = Tensor.of(Shape.of(1, 3, 3, 2), g);
        GradCheck.Result res = GradCheck.check(v -> AffineSampler.sample(v.get(0), v.get(1)).square().sum(),
                List.of(img, grid), 1e-3f, 2e-2f);
        assertTrue(res.passed(), res.toString());
    }

    @Test
    void classicalFlowRecoversATranslation() {
        int w = 24;
        float[] a = texture(w, w, 0, 0);
        float[] b = texture(w, w, 1.3, -0.7);
        double[] epe = new double[2];
        float[][] flows = {ClassicalFlow.hornSchunck(a, b, w, w, 0.3f, 100, 2),
            ClassicalFlow.lucasKanade(a, b, w, w, 2, 2, 3)};
        for (int m = 0; m < 2; m++) {
            double s = 0;
            int n = 0;
            for (int y = 4; y < w - 4; y++) {
                for (int x = 4; x < w - 4; x++) {
                    int i = y * w + x;
                    s += Math.hypot(flows[m][i] - 1.3, flows[m][w * w + i] + 0.7);
                    n++;
                }
            }
            epe[m] = s / n;
        }
        System.out.printf("translation EPE: Horn-Schunck %.3f, Lucas-Kanade %.3f%n", epe[0], epe[1]);
        assertTrue(epe[0] < 0.3 && epe[1] < 0.3, "both recover a smooth translation");
    }

    @Test
    void videoFlowWarpsTheNextFrameOntoThisOne() {
        ShapeScenes.Video v = new ShapeScenes(24, 0f).video(2, 2, 8, 1, 3, false, new SplittableRandom(3));
        float[] warped = FlowMetrics.warp(v.frames()[1], v.flow()[0], 24, 24);
        float[] zero = FlowMetrics.warp(v.frames()[1], new float[2 * 24 * 24], 24, 24);
        double withFlow = 0;
        double without = 0;
        for (int i = 0; i < warped.length; i++) {
            if (v.flow()[0][i] != 0) {
                withFlow += Math.abs(warped[i] - v.frames()[0][i]);
                without += Math.abs(zero[i] - v.frames()[0][i]);
            }
        }
        assertTrue(withFlow < 0.5 * without, "the true flow explains the motion: " + withFlow + " vs " + without);
    }

    @Test
    void perfectDetectionsScoreOne() {
        ShapeScenes.Scene[] s = new ShapeScenes(24, 0.1f).scenes(4, 1, 5, ShapeScenes.Layout.SPARSE, 4);
        List<DetectionMetrics.Detection> dets = new java.util.ArrayList<>();
        float[][] boxes = new float[4][];
        int[][] classes = new int[4][];
        for (int i = 0; i < 4; i++) {
            boxes[i] = s[i].boxes();
            classes[i] = s[i].classes();
            for (int k = 0; k < s[i].count(); k++) {
                dets.add(new DetectionMetrics.Detection(i, s[i].classes()[k], 0.9f, s[i].box(k)));
            }
        }
        assertEquals(1.0, DetectionMetrics.meanAp(dets, boxes, classes, 6, 0.5), 1e-9);
        assertEquals(1.0, DetectionMetrics.meanApCoco(dets, boxes, classes, 6), 1e-9);
        assertEquals(1.0, DetectionMetrics.foregroundAri(s[0].instances(), s[0].instances()), 1e-9);
    }

    @Test
    void separatedEmbeddingsCostNothingAndCluster() {
        int side = 4;
        float[] ids = {1, 1, 0, 2, 1, 1, 0, 2, 0, 0, 0, 2, 3, 3, 0, 0};
        float[] emb = new float[2 * 16];
        float[][] centres = {{0, 0}, {3, 0}, {-3, 0}, {0, 3}};
        for (int p = 0; p < 16; p++) {
            emb[p] = centres[(int) ids[p]][0];
            emb[16 + p] = centres[(int) ids[p]][1];
        }
        DiscriminativeEmbeddingLoss loss = new DiscriminativeEmbeddingLoss(0.5f, 1.5f, 0f);
        float l = loss.compute(Var.of(Tensor.of(Shape.of(1, 2, side, side), emb)),
                Tensor.of(Shape.of(1, side, side), ids)).value().item();
        assertEquals(0f, l, 1e-6f);
        boolean[] fg = new boolean[16];
        for (int p = 0; p < 16; p++) {
            fg[p] = ids[p] > 0;
        }
        int[] inst = EmbeddingClustering.seeds(emb, 2, fg, 0.5, 1.5, 1);
        int[] truth = new int[16];
        for (int p = 0; p < 16; p++) {
            truth[p] = (int) ids[p];
        }
        assertEquals(1.0, DetectionMetrics.foregroundAri(inst, truth), 1e-9);
        assertEquals(1.0, DetectionMetrics.foregroundAri(EmbeddingClustering.meanShift(emb, 2, fg, 0.5, 3), truth),
                1e-9);
    }

    @Test
    void blockMatchingFindsTextureDisparity() {
        ShapeScenes.Stereo s = new ShapeScenes(24, 0f).stereo(6, new SplittableRandom(5));
        float[] d = StereoCostVolume.blockMatching(s.left(), s.right(), 24, 24, 6, 2);
        assertTrue(FlowMetrics.badShare(d, s.disparity(), 1) < 0.2, "block matching is mostly right");
    }

    @Test
    void softArgmaxFindsASharpPeak() {
        float[] heat = new float[36];
        java.util.Arrays.fill(heat, -50f);
        heat[2 * 6 + 4] = 50f;
        float[] xy = KeypointHeads.softArgmax(Var.of(Tensor.of(Shape.of(1, 1, 6, 6), heat)), 2).value().toArray();
        assertEquals(9f, xy[0], 1e-3f);
        assertEquals(5f, xy[1], 1e-3f);
    }

    @Test
    void correlationLookupPeaksAtTheTrueMatch() {
        SplittableRandom r = new SplittableRandom(6);
        Tensor f = Tensor.normal(Shape.of(1, 8, 6, 6), 0, 1, r);
        CorrelationVolume cv = new CorrelationVolume(Var.of(f), Var.of(f), 1);
        float[] targets = new float[2 * 36];
        for (int p = 0; p < 36; p++) {
            targets[p] = p % 6;
            targets[36 + p] = p / 6;
        }
        float[] c = cv.lookup(targets, 1).value().toArray();
        int centre = 0;
        for (int p = 0; p < 36; p++) {
            int best = 0;
            for (int k = 1; k < 9; k++) {
                best = c[k * 36 + p] > c[best * 36 + p] ? k : best;
            }
            centre += best == 4 ? 1 : 0;
        }
        assertTrue(centre >= 34, "self-correlation peaks at zero displacement: " + centre);
    }

    @Test
    void connectedComponentsMergeTouchingShapes() {
        int[] labels = {1, 1, 0, 1, 1, 1, 0, 2, 0, 0, 0, 2};
        int[] cc4 = EmbeddingClustering.connectedComponents(labels, 4, false, 1);
        int max = 0;
        for (int v : cc4) {
            max = Math.max(max, v);
        }
        assertEquals(3, max);
    }
}
