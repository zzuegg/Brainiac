package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class DetectionAndFlowsTest {

    private static Tensor random(long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, new SplittableRandom(seed));
    }

    @Test
    void boxOverlapsByHand() {
        Tensor a = Tensor.of(Shape.of(2, 4), 0, 0, 2, 2, 10, 10, 11, 11);
        Tensor b = Tensor.of(Shape.of(2, 4), 1, 1, 3, 3, 0, 0, 2, 2);
        // [0,0,2,2] vs [1,1,3,3]: intersection 1, union 7; hull 9
        float[] iou = Boxes.iou(Var.of(a), Var.of(b)).value().toArray();
        assertArrayEquals(new float[] {1 / 7f, 1, 0, 0}, iou, 1e-6f);
        float[] giou = Boxes.giou(Var.of(a), Var.of(b)).value().toArray();
        assertEquals(1 / 7f - 2 / 9f, giou[0], 1e-6f);
        assertEquals(1, giou[1], 1e-6f);
        // [10,10,11,11] vs [0,0,2,2]: no overlap, hull 121, union 5
        assertEquals(0 - (121 - 5) / 121f, giou[3], 1e-6f);
        assertArrayEquals(new float[] {4, 1}, Boxes.area(Var.of(a)).value().toArray(), 0);
    }

    @Test
    void encodeAndDecodeAreInverse() {
        Tensor anchors = Boxes.anchors(2, 3, 16, new float[] {32}, new float[] {0.5f, 1, 2});
        assertArrayEquals(new int[] {18, 4}, anchors.shape().toArray());
        // the first anchor: cell (0, 0) center (8, 8), square 32 at ratio 1
        assertArrayEquals(new float[] {-8, -8, 24, 24}, anchors.slice(0, 1, 2).toArray(), 1e-4f);
        Tensor boxes = Boxes.decode(Var.of(random(3, 18, 4).mul(0.3f)), Var.of(anchors)).value();
        Tensor deltas = Boxes.encode(Var.of(boxes), Var.of(anchors)).value();
        assertArrayEquals(boxes.toArray(), Boxes.decode(Var.of(deltas), Var.of(anchors)).value().toArray(), 1e-3f);
        assertArrayEquals(boxes.toArray(), Boxes.toCorners(Boxes.toCenters(Var.of(boxes))).value().toArray(), 1e-4f);
    }

    @Test
    void giouLossHasGradients() {
        Tensor pred = Tensor.of(Shape.of(2, 4), 0, 0.1f, 2, 2.5f, 1, 1, 4, 3.4f);   // no ties: min and max have kinks there
        Tensor target = Tensor.of(Shape.of(2, 4), 0.5f, 0.2f, 2.2f, 2, 0, 0, 3, 3);
        GradCheck.Result r = GradCheck.check(v -> Boxes.giouLoss(v.get(0), Var.of(target)), List.of(pred));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void nmsKeepsTheBestOfEachCluster() {
        float[] boxes = {0, 0, 10, 10, 1, 1, 11, 11, 50, 50, 60, 60, 0, 0, 10, 10};
        float[] scores = {0.9f, 0.8f, 0.7f, 0.95f};
        assertArrayEquals(new int[] {3, 2}, Boxes.nms(boxes, scores, null, 0.5f, 0, 10));
        // per class: box 0 is another class from box 3, so it stays
        assertArrayEquals(new int[] {3, 0, 2}, Boxes.nms(boxes, scores, new int[] {1, 0, 0, 0}, 0.5f, 0, 10));
        assertArrayEquals(new int[] {3}, Boxes.nms(boxes, scores, null, 0.5f, 0.75f, 10));
        assertArrayEquals(new int[] {3}, Boxes.nms(boxes, scores, null, 0.5f, 0, 1));
    }

    @Test
    void quantileLossIsThePinball() {
        Tensor p = Tensor.of(Shape.of(2, 2), 1, 1, 3, 3);
        Tensor t = Tensor.of(Shape.of(2, 1), 2, 2);
        // row 1 under by 1: 0.1 and 0.9; row 2 over by 1: 0.9 and 0.1
        assertEquals((0.1f + 0.9f + 0.9f + 0.1f) / 4, Loss.quantile(0.1f, 0.9f).compute(Var.of(p), t).value().item(),
                1e-6f);
    }

    @Test
    void sampledSoftmaxWithEveryClassIsCrossEntropy() {
        Tensor h = random(1, 3, 4);
        Tensor w = random(2, 4, 6);
        Tensor ids = Tensor.of(new float[] {2, 5, 2});
        float full = Functional.crossEntropy(Var.of(h).matmul(Var.of(w)), ids).value().item();
        // enough draws to pick every class: the same classes, in another order
        float sampled = Functional.sampledSoftmax(Var.of(h), Var.of(w), ids, 500, new SplittableRandom(4)).value()
                .item();
        assertEquals(full, sampled, 1e-5f);
        Var wp = Var.param(w);
        Functional.sampledSoftmax(Var.of(h), wp, ids, 2, new SplittableRandom(5)).backward();
        assertTrue(wp.grad() != null);
    }

    @Test
    void spectralNormBringsTheLargestSingularValueToOne() {
        Tensor w = Tensor.of(Shape.of(2, 2), 3, 0, 0, 1);
        float[] u = new float[2];
        Var normed = null;
        for (int i = 0; i < 5; i++) {
            normed = Functional.spectralNorm(Var.of(w), u);
        }
        assertArrayEquals(new float[] {1, 0, 0, 1 / 3f}, normed.value().toArray(), 1e-4f);
    }

    @Test
    void r1PenaltyOfALinearCriticIsItsWeightNorm() {
        Tensor w = Tensor.of(Shape.of(3, 1), 1, 2, 2);
        Var wp = Var.param(w);
        Var penalty = Functional.r1Penalty(x -> x.matmul(wp), random(6, 4, 3), 2);
        assertEquals(9, penalty.value().item(), 1e-5f);   // gamma/2 * |w|^2
        penalty.backward();
        assertArrayEquals(new float[] {2, 4, 4}, wp.grad().toArray(), 1e-4f);   // gamma * w
    }

    @Test
    void couplingsInvertAndTheirLogDetMatchesTheJacobian() {
        Tensor[] masks = Flows.alternatingMasks(2, 2);
        Tensor w1 = random(7, 2, 2);
        Tensor w2 = random(8, 2, 2);
        List<Flows.Coupling> flow = List.of(
                new Flows.Coupling(masks[0], x -> new Var[] {x.matmul(Var.of(w1)), x.matmul(Var.of(w2))}),
                new Flows.Coupling(masks[1], x -> new Var[] {x.matmul(Var.of(w2)), x.matmul(Var.of(w1))}));
        Tensor x = random(9, 1, 2);
        Flows.Result z = Flows.forward(flow, Var.of(x));
        Flows.Result back = Flows.inverse(flow, z.output());
        assertArrayEquals(x.toArray(), back.output().value().toArray(), 1e-5f);
        assertEquals(-z.logDet().value().item(), back.logDet().value().item(), 1e-5f);
        // log |det J| by finite differences
        float e = 1e-3f;
        float[] j = new float[4];
        for (int c = 0; c < 2; c++) {
            float[] plus = x.toArray();
            float[] minus = x.toArray();
            plus[c] += e;
            minus[c] -= e;
            float[] zp = Flows.forward(flow, Var.of(Tensor.of(Shape.of(1, 2), plus))).output().value().toArray();
            float[] zm = Flows.forward(flow, Var.of(Tensor.of(Shape.of(1, 2), minus))).output().value().toArray();
            for (int r = 0; r < 2; r++) {
                j[r * 2 + c] = (zp[r] - zm[r]) / (2 * e);
            }
        }
        double det = Math.log(Math.abs(j[0] * j[3] - j[1] * j[2]));
        assertEquals(det, z.logDet().value().item(), 2e-3);
        float ll = Flows.logLikelihood(z).value().item();
        float[] zv = z.output().value().toArray();
        assertEquals(-0.5 * (zv[0] * zv[0] + zv[1] * zv[1]) - Math.log(2 * Math.PI) + det, ll, 2e-3);
    }
}
