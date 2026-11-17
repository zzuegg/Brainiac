package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.autograd.DualTensor;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

class LocalRulesTest {

    /** Correlated 2-D Gaussian data whose first principal component is (1, 1)/sqrt(2). */
    private static Tensor correlated(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] v = new float[n * 2];
        for (int i = 0; i < n; i++) {
            double a = r.nextGaussian() * 2;
            double b = r.nextGaussian() * 0.8;
            v[2 * i] = (float) ((a + b) / Math.sqrt(2));
            v[2 * i + 1] = (float) ((a - b) / Math.sqrt(2));
        }
        return Tensor.of(Shape.of(n, 2), v);
    }

    @Test
    void ojaFindsTheFirstPrincipalComponentWithUnitNorm() {
        Var w = Var.param(Tensor.of(Shape.of(2, 1), 0.3f, -0.8f));
        PlasticityRule oja = PlasticityRule.oja(0.02f);
        for (int i = 0; i < 300; i++) {
            Tensor x = correlated(32, i);
            oja.apply(w, x, x.matmul(w.value()));
        }
        float[] v = w.value().toArray();
        assertEquals(1, Math.hypot(v[0], v[1]), 0.05);
        assertEquals(1, Math.abs(v[0] + v[1]) / Math.sqrt(2), 0.05);
    }

    @Test
    void sangerOrdersTheComponentsAndHebbGrowsWithoutBound() {
        Var w = Var.param(Tensor.of(Shape.of(2, 2), 0.5f, 0.1f, -0.2f, 0.7f));
        PlasticityRule sanger = PlasticityRule.sanger(0.02f);
        for (int i = 0; i < 600; i++) {
            Tensor x = correlated(32, 1000 + i);
            sanger.apply(w, x, x.matmul(w.value()));
        }
        float[] v = w.value().toArray();   // [in, out]: column 0 = first PC, column 1 = second
        assertEquals(1, Math.abs(v[0] + v[2]) / Math.sqrt(2), 0.06);
        assertEquals(1, Math.abs(v[1] - v[3]) / Math.sqrt(2), 0.1);

        Var h = Var.param(Tensor.of(Shape.of(2, 1), 0.1f, 0.1f));
        PlasticityRule hebb = PlasticityRule.hebb(0.05f);
        for (int i = 0; i < 50; i++) {
            Tensor x = correlated(32, i);
            hebb.apply(h, x, x.matmul(h.value()));
        }
        assertTrue(LearningMath.norm(h.value()) > 10, "plain Hebb is unstable");
    }

    @Test
    void bcmMakesAUnitSelectiveForOnePattern() {
        // two input patterns; a BCM unit ends up responding to one much more than the other
        Tensor patterns = Tensor.of(Shape.of(2, 2), 1, 0, 0, 1);
        Var w = Var.param(Tensor.of(Shape.of(2, 1), 0.55f, 0.45f));
        PlasticityRule bcm = PlasticityRule.bcm(0.05f, 0.2f);
        SplittableRandom r = new SplittableRandom(3);
        for (int i = 0; i < 2000; i++) {
            int k = r.nextInt(2);
            Tensor x = patterns.slice(0, k, k + 1);
            bcm.apply(w, x, x.matmul(w.value()).relu());
        }
        float[] v = w.value().toArray();
        assertTrue(Math.max(v[0], v[1]) > 4 * Math.max(0.01f, Math.min(v[0], v[1])), "selective: " + v[0] + ", " + v[1]);
        PlasticityRule cov = PlasticityRule.covariance(0.1f, 0.5f);
        Tensor d = cov.update(patterns, patterns, Tensor.zeros(Shape.of(2, 2)));
        assertTrue(d.toArray()[1] < 0, "anti-correlated inputs weaken: " + d);
    }

    @Test
    void feedbackAlignmentAlignsForwardWeightsWithTheFeedback() {
        SimpleMlp net = SimpleMlp.tanh(2, 2, 32, 32, 3);
        FeedbackAlignment fa = new FeedbackAlignment(net, Optimizer.sgd(0.2f), FeedbackAlignment.Mode.FEEDBACK_ALIGNMENT, 9);
        double before = fa.alignment(2);
        SplittableRandom r = new SplittableRandom(1);
        for (int i = 0; i < 200; i++) {
            Tensor x = Tensor.normal(Shape.of(32, 2), 0, 1, r);
            Tensor y = x.slice(1, 0, 1).greater(Tensor.full(Shape.of(), 0f)).reshape(32);
            fa.trainBatch(x, y);
        }
        assertTrue(fa.alignment(2) > before + 0.1, "alignment " + before + " -> " + fa.alignment(2));
    }

    // ---------------------------------------------------------------- forward mode

    private static Function<List<Var>, Var> mlpLoss(Tensor x) {
        return p -> Var.of(x).matmul(p.get(0)).add(p.get(1)).tanh().matmul(p.get(2)).add(p.get(3));
    }

    private static DualTensor dualLoss(List<DualTensor> p, Tensor x, Tensor ids) {
        return DualTensor.constant(x).matmul(p.get(0)).add(p.get(1)).tanh().matmul(p.get(2)).add(p.get(3))
                .crossEntropy(ids);
    }

    private static List<Var> params(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return List.of(Var.param(Tensor.normal(Shape.of(2, 8), 0, 1, r)), Var.param(Tensor.normal(Shape.of(1, 8), 0, 0.1f, r)),
                Var.param(Tensor.normal(Shape.of(8, 3), 0, 0.5f, r)), Var.param(Tensor.zeros(Shape.of(1, 3))));
    }

    @Test
    void dualNumbersGiveTheSameDirectionalDerivativeAsBackprop() {
        SplittableRandom r = new SplittableRandom(4);
        Tensor x = Tensor.normal(Shape.of(16, 2), 0, 1, r);
        Tensor ids = Tensor.of(Shape.of(16), 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0);
        List<Var> p = params(5);
        Var loss = io.github.zzuegg.brainiac.nn.Functional.crossEntropy(mlpLoss(x).apply(p), ids);
        loss.backward();
        List<Tensor> v = new ArrayList<>();
        double expected = 0;
        for (Var q : p) {
            Tensor t = Tensor.normal(q.shape(), 0, 1, r);
            v.add(t);
            expected += q.grad().mul(t).sum().item();
        }
        DualTensor out = ForwardGradient.jvp(p, v, d -> dualLoss(d, x, ids));
        assertEquals(loss.value().item(), out.primal().item(), 1e-5);
        assertEquals(expected, out.tangent().item(), 1e-3 * Math.max(1, Math.abs(expected)));
    }

    @Test
    void forwardGradientIsUnbiasedAndTrains() {
        SplittableRandom r = new SplittableRandom(6);
        Tensor x = Tensor.normal(Shape.of(32, 2), 0, 1, r);
        float[] lab = new float[32];
        float[] xv = x.toArray();
        for (int i = 0; i < 32; i++) {
            lab[i] = xv[2 * i] > 0 ? (xv[2 * i + 1] > 0 ? 0 : 1) : 2;
        }
        Tensor ids = Tensor.of(Shape.of(32), lab);
        List<Var> p = params(7);
        io.github.zzuegg.brainiac.nn.Functional.crossEntropy(mlpLoss(x).apply(p), ids).backward();
        List<Tensor> exact = p.stream().map(Var::grad).toList();
        p.forEach(Var::zeroGrad);
        new ForwardGradient(8, 3000).estimate(p, d -> dualLoss(d, x, ids));
        double cos = LearningMath.cosine(LearningMath.flatten(exact),
                LearningMath.flatten(p.stream().map(Var::grad).toList()));
        assertTrue(cos > 0.85, "mean forward gradient aligns with the true gradient: " + cos);
        p.forEach(Var::zeroGrad);

        ForwardGradient fg = new ForwardGradient(9, 4);
        Optimizer opt = Optimizer.adam(0.02f);
        float first = fg.estimate(p, d -> dualLoss(d, x, ids));
        opt.step(p);
        float last = first;
        for (int i = 0; i < 300; i++) {
            last = fg.estimate(p, d -> dualLoss(d, x, ids));
            opt.step(p);
        }
        assertTrue(last < 0.6 * first, "loss " + first + " -> " + last);
    }

    // ---------------------------------------------------------------- e-prop

    /** Target: the input bit from {@code delay} steps ago, smoothed. */
    private static Tensor[][] delayTask(int batch, int steps, int delay, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] bits = new float[steps][batch];
        Tensor[] xs = new Tensor[steps];
        Tensor[] ys = new Tensor[steps];
        for (int t = 0; t < steps; t++) {
            for (int b = 0; b < batch; b++) {
                bits[t][b] = r.nextBoolean() ? 1 : -1;
            }
            xs[t] = Tensor.of(Shape.of(batch, 1), bits[t]);
            ys[t] = t >= delay ? Tensor.of(Shape.of(batch, 1), bits[t - delay]) : Tensor.zeros(Shape.of(batch, 1));
        }
        return new Tensor[][] {xs, ys};
    }

    @Test
    void epropLearnsATemporalTaskCloseToBptt() {
        float[] finalLoss = new float[3];
        for (EProp.Mode mode : EProp.Mode.values()) {
            EProp net = new EProp(1, 24, 1, 0.5f, mode, Optimizer.adam(0.01f), 1);
            Tensor[][] eval = delayTask(64, 20, 1, 999);
            float before = net.loss(eval[0], eval[1]);
            for (int i = 0; i < 150; i++) {
                Tensor[][] d = delayTask(32, 20, 1, i);
                net.trainSequence(d[0], d[1]);
            }
            finalLoss[mode.ordinal()] = net.loss(eval[0], eval[1]);
            System.out.println(mode + " loss " + before + " -> " + finalLoss[mode.ordinal()]);
            assertTrue(finalLoss[mode.ordinal()] < 0.6 * before, mode + " learns");
        }
        assertTrue(finalLoss[EProp.Mode.SYMMETRIC.ordinal()] < 2 * finalLoss[EProp.Mode.BPTT.ordinal()] + 0.05);
    }
}
