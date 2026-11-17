package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;

class OptimizerZooTest {

    static final Tensor[] DATA = Fixtures.spiral(60, 1);

    @Test
    void everyOptimizerTrainsTheSpiral() {
        Map<String, Supplier<Optimizer>> zoo = new LinkedHashMap<>();
        zoo.put("adam (core baseline)", () -> Optimizer.adam(0.02f));
        zoo.put("schedule-free adamw", () -> ScheduleFree.adamW(0.02f, 0, 10));
        zoo.put("schedule-free sgd", () -> ScheduleFree.sgd(1f, 0.9f, 10));
        zoo.put("prodigy", Prodigy::of);
        zoo.put("d-adaptation sgd", DAdaptation::sgd);
        zoo.put("lamb", () -> Lamb.of(0.02f, 0));
        zoo.put("lars", () -> new Lars(0.5f, 0.9f, 0, 0.02f));
        zoo.put("adan", () -> Adan.of(0.02f, 0));
        zoo.put("ademamix", () -> new AdEMAMix(0.01f, 0.9f, 0.999f, 0.999f, 2, 50, 0));
        zoo.put("nesterov", () -> new Nesterov(0.3f, 0.9f));
        zoo.put("sophia", () -> Sophia.of(0.02f));
        zoo.put("shampoo", () -> new Shampoo(0.05f, 0.9f, 10, 64));
        zoo.put("soap", () -> Soap.of(0.02f));
        zoo.put("galore", () -> new GaLore(0.02f, 4, 50, 1, 0));
        zoo.put("lookahead(adam)", () -> new Lookahead(Optimizer.adam(0.03f), 5, 0.5f));
        zoo.put("agc+gc(adam)", () -> GradientTransform.wrap(Optimizer.adam(0.02f),
                GradientTransform.adaptiveClipping(0.1f, 1e-3f), GradientTransform.centralization()));
        float initial = Fixtures.loss(SimpleMlp.tanh(3, 2, 32, 32, 3), DATA);
        for (var e : zoo.entrySet()) {
            SimpleMlp net = SimpleMlp.tanh(3, 2, 32, 32, 3);
            float loss = Fixtures.train(net, e.getValue().get(), DATA, 200);
            double acc = LearningMath.accuracy(net.predict(DATA[0]), DATA[1]);
            System.out.printf("%-22s loss %.3f -> %.3f  acc %.3f%n", e.getKey(), initial, loss, acc);
            assertTrue(loss < 0.6 * initial && acc > 0.6, e.getKey() + " trains: " + loss + ", " + acc);
        }
    }

    @Test
    void checkpointStateRestoresIdenticalTrajectories() {
        for (Supplier<SlotOptimizer> make : List.<Supplier<SlotOptimizer>>of(Prodigy::of,
                () -> ScheduleFree.adamW(0.02f, 0.01f, 5), () -> Soap.of(0.02f), () -> Adan.of(0.02f, 0))) {
            SimpleMlp a = SimpleMlp.tanh(4, 2, 8, 3);
            SlotOptimizer oa = make.get();
            Fixtures.train(a, oa, DATA, 12);
            SimpleMlp b = SimpleMlp.tanh(99, 2, 8, 3);
            LearningMath.assign(b.parameters(), LearningMath.values(a.parameters()));
            SlotOptimizer ob = make.get();
            ob.loadState(b.parameters(), oa.state(a.parameters()));
            Fixtures.train(a, oa, DATA, 5);
            Fixtures.train(b, ob, DATA, 5);
            assertArrayEquals(LearningMath.flatten(LearningMath.values(a.parameters())),
                    LearningMath.flatten(LearningMath.values(b.parameters())), 1e-5f, oa.toString());
        }
    }

    @Test
    void scheduleFreeEvaluatesAtTheAverageAndSwitchesBack() {
        SimpleMlp net = SimpleMlp.tanh(5, 2, 32, 3);
        ScheduleFree opt = ScheduleFree.adamW(0.02f, 0, 10);
        Fixtures.train(net, opt, DATA, 150);
        float[] y = LearningMath.flatten(LearningMath.values(net.parameters()));
        opt.evalMode(net.parameters());
        float evalLoss = Fixtures.loss(net, DATA);
        opt.trainMode(net.parameters());
        assertArrayEquals(y, LearningMath.flatten(LearningMath.values(net.parameters())), 1e-5f);
        assertTrue(evalLoss < 0.7f, "the averaged weights are good: " + evalLoss);
    }

    @Test
    void parameterFreeOptimizersGrowTheirStepSize() {
        Prodigy p = Prodigy.of();
        Fixtures.train(SimpleMlp.tanh(6, 2, 16, 3), p, DATA, 60);
        assertTrue(p.distance() > 1e-3, "prodigy d grew from 1e-6 to " + p.distance());
        DAdaptation d = DAdaptation.sgd();
        Fixtures.train(SimpleMlp.tanh(6, 2, 16, 3), d, DATA, 60);
        assertTrue(d.distance() > 1e-4, "d-adaptation d grew to " + d.distance());
    }

    @Test
    void samStepsFromTheOriginalWeightsWithThePerturbedGradient() {
        SimpleMlp net = SimpleMlp.tanh(7, 2, 8, 3);
        List<Var> params = net.parameters();
        List<Tensor> start = LearningMath.values(params);
        float[] before = LearningMath.flatten(start);
        Supplier<Var> loss = () -> {
            Var l = Functional.crossEntropy(net.forward(Var.of(DATA[0])), DATA[1]);
            l.backward();
            return l;
        };
        // reference: the gradient at the perturbed point
        loss.get();
        SharpnessAware.sam(Optimizer.sgd(0.1f), 0.05f).perturb(params);
        loss.get();
        float[] perturbedGrad = LearningMath.flatten(params.stream().map(Var::grad).toList());
        params.forEach(Var::zeroGrad);
        LearningMath.assign(params, start);
        // the full SAM step from the original weights applies exactly that gradient
        SharpnessAware.sam(Optimizer.sgd(0.1f), 0.05f).step(params, loss);
        float[] after = LearningMath.flatten(LearningMath.values(params));
        for (int i = 0; i < after.length; i++) {
            assertEquals(before[i] - 0.1f * perturbedGrad[i], after[i], 1e-4f);
        }
        SimpleMlp asam = SimpleMlp.tanh(8, 2, 32, 32, 3);
        SharpnessAware opt = SharpnessAware.asam(Optimizer.adam(0.02f), 0.2f, 0.01f);
        Supplier<Var> l2 = () -> {
            Var l = Functional.crossEntropy(asam.forward(Var.of(DATA[0])), DATA[1]);
            l.backward();
            return l;
        };
        float first = opt.step(asam.parameters(), l2);
        float last = first;
        for (int i = 0; i < 150; i++) {
            last = opt.step(asam.parameters(), l2);
        }
        assertTrue(last < 0.5f * first, "ASAM trains: " + first + " -> " + last);
    }

    @Test
    void sophiaHutchinsonRecoversADiagonalHessianAndGnbTrains() {
        Var w = Var.param(Tensor.of(1, 2, 3));
        Tensor a = Tensor.of(2, 5, 0.5f);
        Sophia s = new Sophia(0.1f, 0.9f, 0f, 1, 0);   // beta2 = 0: the estimate itself
        s.updateHessianHutchinson(List.of(w), () -> w.square().mul(a).sum().mul(0.5f), 1);
        // a quadratic with diagonal Hessian a: u * (H u) = a exactly
        w.mul(a).sum().backward();
        s.step(List.of(w));
        float[] v = w.value().toArray();
        // step = clip(m / h): m = 0.1 * grad(=a*w direction) ... check it moved toward 0 by at most lr
        assertTrue(v[0] < 1 && v[0] >= 0.9f && v[1] < 2 && v[2] < 3);

        SimpleMlp net = SimpleMlp.tanh(9, 2, 32, 32, 3);
        Sophia opt = Sophia.of(0.02f);
        SplittableRandom r = new SplittableRandom(1);
        float first = Fixtures.loss(net, DATA);
        for (int i = 0; i < 200; i++) {
            if (i % 10 == 0) {
                Tensor sampled = Sophia.sampleLabels(net.predict(DATA[0]), r);
                Functional.crossEntropy(net.forward(Var.of(DATA[0])), sampled).backward();
                opt.updateHessianGnb(net.parameters(), DATA[1].shape().dim(0));
            }
            Functional.crossEntropy(net.forward(Var.of(DATA[0])), DATA[1]).backward();
            opt.step(net.parameters());
        }
        assertTrue(Fixtures.loss(net, DATA) < 0.5f * first);
    }

    @Test
    void mezoTrainsWithoutABackwardPass() {
        SplittableRandom r = new SplittableRandom(3);
        Tensor x = Tensor.normal(Shape.of(64, 4), 0, 1, r);
        Tensor trueW = Tensor.of(Shape.of(4, 1), 1, -2, 0.5f, 3);
        Tensor y = x.matmul(trueW);
        Var w = Var.param(Tensor.zeros(Shape.of(4, 1)));
        Supplier<Float> loss = () -> x.matmul(w.value()).sub(y).square().mean().item();
        float first = loss.get();
        MeZo mezo = new MeZo(0.02f, 1e-3f, 0, 5);
        for (int i = 0; i < 400; i++) {
            mezo.step(List.of(w), loss);
        }
        assertTrue(loss.get() < 0.05f * first, "MeZO: " + first + " -> " + loss.get());
        // the same seed regenerates the same direction
        assertArrayEquals(MeZo.direction(w, 0, 42).toArray(), MeZo.direction(w, 0, 42).toArray());
    }

    @Test
    void forgettingAParameterDropsItsSlotsInWrappersToo() {
        Var a = Var.param(Tensor.of(1, 2));
        Var b = Var.param(Tensor.of(3, 4));
        List<Var> params = List.of(a, b);
        Lamb inner = Lamb.of(0.01f, 0);
        Optimizer wrapped = new Lookahead(inner, 5, 0.5f);
        a.setGrad(Tensor.of(1, 1));
        b.setGrad(Tensor.of(1, 1));
        wrapped.step(params);
        assertTrue(inner.state(params).containsKey("0.m"));
        wrapped.forget(a);
        assertTrue(!inner.state(params).containsKey("0.m") && inner.state(params).containsKey("1.m"),
                "the core Optimizer.forget hook reaches the slots: " + inner.state(params).keySet());
    }

    @Test
    void galoreKeepsLowRankStateAndPreconditionersAreCorrect() {
        GaLore g = new GaLore(0.01f, 4, 50, 1, 0);
        assertTrue(g.stateSize(256, 512) < 2L * 256 * 512 / 10);
        // the preconditioners are core matrix functions (Linalg), as Shampoo and SOAP use them
        Tensor a = Tensor.of(new float[][] {{4, 1, 0.5f}, {1, 3, 0.2f}, {0.5f, 0.2f, 2}});
        Tensor prod = a.matmul(Linalg.symmetricPower(a, -1, 0));
        assertArrayEquals(new float[] {1, 0, 0, 0, 1, 0, 0, 0, 1}, prod.toArray(), 1e-4f);
        float[] values = Linalg.eigh(a)[0].toArray();
        assertTrue(values[0] <= values[1] && values[1] <= values[2]);
    }

    @Test
    void gradientTransformsClipAndCentre() {
        Var w = Var.param(Tensor.of(new float[][] {{1, 0}, {0, 1}}));
        Tensor g = Tensor.of(new float[][] {{10, 0.001f}, {10, 0.002f}});
        Tensor clipped = GradientTransform.adaptiveClipping(0.1f, 1e-3f).apply(w, g);
        float[] c = clipped.toArray();
        assertEquals(0.1, Math.hypot(c[0], c[2]), 1e-4);   // column 0 clipped to 0.1 * |w_col|
        assertEquals(0.001f, c[1], 1e-6f);                    // small column untouched
        Tensor centred = GradientTransform.centralization().apply(w, g);
        assertArrayEquals(new float[] {0, 0}, centred.sum(0, false).toArray(), 1e-6f);
    }

    @Test
    void lrRangeTestSuggestsAWorkingRateAndRestoresWeights() {
        SimpleMlp net = SimpleMlp.tanh(10, 2, 32, 3);
        float[] before = LearningMath.flatten(LearningMath.values(net.parameters()));
        LrRangeTest.Result r = LrRangeTest.run(net.parameters(), Optimizer.adam(1e-3f),
                () -> Functional.crossEntropy(net.forward(Var.of(DATA[0])), DATA[1]), 1e-5f, 10f, 120);
        assertArrayEquals(before, LearningMath.flatten(LearningMath.values(net.parameters())));
        System.out.println("suggested " + r.suggested() + ", minimum at " + r.minimumLossRate());
        assertTrue(r.suggested() > 1e-3f && r.suggested() < 1f, "suggestion " + r.suggested());
        float loss = Fixtures.train(net, Optimizer.adam(r.suggested()), DATA, 100);
        assertTrue(loss < 0.8f, "the suggested rate trains: " + loss);
    }
}
