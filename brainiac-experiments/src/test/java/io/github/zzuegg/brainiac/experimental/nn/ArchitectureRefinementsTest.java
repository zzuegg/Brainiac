package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class ArchitectureRefinementsTest {

    private static Var input(int b, int t, int d, long seed) {
        return Var.of(Tensor.normal(Shape.of(b, t, d), 0, 1, new SplittableRandom(seed)));
    }

    /** Output rows before {@code cut} must not change when inputs from {@code cut} on change. */
    private static void assertCausal(Layer layer, int t, int d, int cut) {
        Var a = input(2, t, d, 5);
        float[] other = a.value().toArray();
        for (int b = 0; b < 2; b++) {
            for (int i = cut * d; i < t * d; i++) {
                other[b * t * d + i] += 3f;
            }
        }
        float[] ya = layer.forward(a).value().toArray();
        float[] yb = layer.forward(Var.of(Tensor.of(Shape.of(2, t, d), other))).value().toArray();
        for (int b = 0; b < 2; b++) {
            for (int i = 0; i < cut * d; i++) {
                assertEquals(ya[b * t * d + i], yb[b * t * d + i], 1e-4, "leak at " + i);
            }
        }
    }

    @Test
    void softpickIsRectifiedAndCanAttendToNothing() {
        Var scores = Var.of(Tensor.of(Shape.of(2, 3), 1f, -2f, 0.5f, -1f, -0.5f, -3f));
        float[] w = Softpick.apply(scores, null).value().toArray();
        double den = Math.abs(Math.exp(1) - 1) + Math.abs(Math.exp(-2) - 1) + Math.abs(Math.exp(0.5) - 1);
        assertEquals((Math.exp(1) - 1) / den, w[0], 1e-5);
        assertEquals(0, w[1], 1e-7);
        assertEquals((Math.exp(0.5) - 1) / den, w[2], 1e-5);
        assertArrayEquals(new float[] {0, 0, 0}, new float[] {w[3], w[4], w[5]}, 1e-7f);
        float[] masked = Softpick.apply(scores, Tensor.of(Shape.of(3), 1f, 1f, 0f)).value().toArray();
        assertEquals(0, masked[2], 0);
        assertEquals((Math.exp(1) - 1) / (Math.exp(1) - 1 + 1 - Math.exp(-2)), masked[0], 1e-5);   // masked key not counted
    }

    @Test
    void stickBreakingMatchesTheDefinition() {
        int t = 4;
        Tensor z = Tensor.normal(Shape.of(1, 1, t, t), 0, 1.5f, new SplittableRandom(3));
        Tensor keep = RefinementOps.keep(t, t, 0, 0, true, 0, z.device());
        float[] a = StickBreakingAttention.weights(Var.of(z), keep).value().toArray();
        float[] zz = z.toArray();
        for (int i = 0; i < t; i++) {
            double rowSum = 0;
            for (int j = 0; j < t; j++) {
                double expected = 0;
                if (j <= i) {
                    expected = 1 / (1 + Math.exp(-zz[i * t + j]));
                    for (int k = j + 1; k <= i; k++) {
                        expected *= 1 - 1 / (1 + Math.exp(-zz[i * t + k]));
                    }
                }
                assertEquals(expected, a[i * t + j], 1e-5, i + "," + j);
                rowSum += a[i * t + j];
            }
            assertTrue(rowSum <= 1 + 1e-6);
        }
    }

    @Test
    void selectivePenaltyAccumulatesEarlierSelections() {
        int t = 4;
        float[] l = new float[t * t];
        l[2 * t + 1] = 2f;    // token 2 selects token 1
        l[3 * t + 1] = 1f;    // token 3 selects token 1 again
        l[2 * t + 0] = 5f;    // the first token is never masked
        l[2 * t + 2] = 5f;    // nor does a token mask itself
        float[] f = SelectiveAttention.penalty(Var.of(Tensor.of(Shape.of(1, 1, t, t), l))).value().toArray();
        assertEquals(0, f[2 * t + 1], 0);           // query 2 does not see its own selection
        assertEquals(2, f[3 * t + 1], 0);           // query 3 sees token 2's selection
        assertEquals(0, f[3 * t + 0], 0);
        assertEquals(0, f[3 * t + 2], 0);
        assertEquals(1.0, SelectiveAttention.keptContext(Tensor.zeros(Shape.of(1, 1, t, t)), 1f), 1e-9);
    }

    @Test
    void everyRefinedAttentionOptionIsCausalAndTrainable() {
        Map<String, RefinedAttention.Spec> specs = new LinkedHashMap<>();
        RefinedAttention.Spec base = RefinedAttention.of(2).causal(true).rotary(10000f);
        specs.put("softmax", base);
        specs.put("sigmoid", base.weighting(RefinedAttention.Weighting.SIGMOID));
        specs.put("softpick", base.weighting(RefinedAttention.Weighting.SOFTPICK));
        specs.put("stick-breaking", RefinedAttention.of(2).causal(true)
                .weighting(RefinedAttention.Weighting.STICK_BREAKING));
        specs.put("ssmax", base.scalableSoftmax(true));
        specs.put("sink", base.sinkLogit(true));
        specs.put("registers", base.registers(2));
        specs.put("talking heads", base.talkingHeads(true));
        specs.put("selective", base.selective(true));
        specs.put("window", base.window(3));
        specs.put("everything", base.scalableSoftmax(true).sinkLogit(true).registers(1).talkingHeads(true)
                .selective(true).weighting(RefinedAttention.Weighting.SOFTPICK));
        for (Map.Entry<String, RefinedAttention.Spec> e : specs.entrySet()) {
            RefinedAttention a = e.getValue().create(Shape.of(6, 8), new SplittableRandom(1));
            assertCausal(a, 6, 8, 3);
            Var y = a.forward(input(2, 6, 8, 2));
            assertEquals(Shape.of(2, 6, 8), y.shape(), e.getKey());
            y.square().mean().backward();
            for (Var p : a.parameters()) {
                assertNotNull(p.grad(), e.getKey());
                for (float g : p.grad().toArray()) {
                    assertFalse(Float.isNaN(g), e.getKey());
                }
            }
            assertNotNull(a.lastWeights());
        }
    }

    @Test
    void restrictionHidesKeys() {
        RefinedAttention a = RefinedAttention.of(1).causal(true).create(Shape.of(4, 4), new SplittableRandom(2));
        float[] keep = new float[16];
        for (int i = 0; i < 4; i++) {
            keep[i * 4 + i] = 1;    // each query sees only itself
        }
        a.restrict(Tensor.of(Shape.of(4, 4), keep));
        a.forward(input(1, 4, 4, 1));
        float[] w = a.lastWeights().toArray();
        for (int i = 0; i < 4; i++) {
            assertEquals(1, w[i * 4 + i], 1e-5);
        }
    }

    @Test
    void latentAttentionCacheReproducesTheFullPass() {
        LatentAttention mla = new LatentAttention(16, 4, 6, 4, 10000f, new SplittableRandom(4));
        Var x = input(2, 5, 16, 3);
        float[] full = mla.forward(x).value().toArray();
        LatentAttention.Cache cache = new LatentAttention.Cache();
        float[] first = mla.forward(x.slice(1, 0, 3), cache).value().toArray();
        float[] rest = mla.forward(x.slice(1, 3, 5), cache).value().toArray();
        assertEquals(5, cache.length());
        assertEquals(10, cache.valuesPerStep());
        assertEquals(10, mla.cacheValuesPerToken());
        for (int b = 0; b < 2; b++) {
            for (int i = 0; i < 3 * 16; i++) {
                assertEquals(full[b * 80 + i], first[b * 48 + i], 1e-4);
            }
            for (int i = 0; i < 2 * 16; i++) {
                assertEquals(full[b * 80 + 48 + i], rest[b * 32 + i], 1e-4);
            }
        }
        assertCausal(mla, 5, 16, 2);
    }

    @Test
    void stacksAreCausalAndCountTheirCaches() {
        SplittableRandom r = new SplittableRandom(6);
        CrossLayerKvBlock none = new CrossLayerKvBlock(8, 2, 4, CrossLayerKvBlock.Sharing.NONE, 1, 1, 10000f, r);
        CrossLayerKvBlock cla = new CrossLayerKvBlock(8, 2, 4, CrossLayerKvBlock.Sharing.CLA, 2, 1, 10000f, r);
        CrossLayerKvBlock yoco = new CrossLayerKvBlock(8, 2, 4, CrossLayerKvBlock.Sharing.YOCO, 1, 2, 10000f, r);
        assertEquals(64, none.cachedValuesPerToken());
        assertEquals(32, cla.cachedValuesPerToken());
        assertEquals(16, yoco.cachedValuesPerToken());
        assertTrue(cla.parameters().size() < none.parameters().size());
        List<Layer> layers = List.of(none, cla, yoco, new ValueResidualBlock(8, 2, 3, true, 10000f, r),
                new InterleavedRopeStack(8, 2, 4, 2, true, 3, r), new ParallelHybridBlock(8, 2, 2, 4, 2, r),
                ResidualMixerBlock.of(RefinedAttention.of(2).causal(true)).canon(3).create(Shape.of(6, 8), r),
                new CanonLayer(8, 3, r));
        for (Layer l : layers) {
            assertCausal(l, 6, 8, 3);
            assertEquals(Shape.of(2, 6, 8), l.forward(input(2, 6, 8, 1)).shape(), l.toString());
        }
        assertArrayEquals(new boolean[] {false, true, false, true},
                new InterleavedRopeStack(8, 2, 4, 2, true, 0, r).nopeLayers());
    }

    @Test
    void valueResidualLearnsItsMixingWeight() {
        ValueResidualBlock vr = new ValueResidualBlock(8, 2, 3, true, 10000f, new SplittableRandom(2));
        vr.forward(input(2, 5, 8, 1)).square().mean().backward();
        assertNotNull(vr.lambdas());
        assertEquals(3 * 9 + 2, vr.parameters().size());   // 6 attention, 3 ffn per layer + 2 lambdas
    }

    @Test
    void expertChoiceBalancesPerfectlyAndTokenChoiceDropsOverflow() {
        SplittableRandom r = new SplittableRandom(8);
        Var x = Var.of(Tensor.normal(Shape.of(32, 8), 0, 1, r));
        ExpertChoiceMoe ec = ExpertChoiceMoe.of(8, 4, 16, ExpertChoiceMoe.Routing.EXPERT_CHOICE, 2, 2f, r)
                .zLossWeight(1e-3f);
        Var y = ec.forward(x);
        assertEquals(Shape.of(32, 8), y.shape());
        assertArrayEquals(new int[] {16, 16, 16, 16}, ec.expertLoad());
        assertEquals(0, ExpertChoiceMoe.loadImbalance(ec.expertLoad()), 0);
        assertTrue(ec.zLoss().value().item() > 0);
        y.sum().add(ec.auxiliaryLoss()).backward();
        assertNotNull(ec.router().grad());
        ExpertChoiceMoe tc = ExpertChoiceMoe.of(8, 4, 16, ExpertChoiceMoe.Routing.TOKEN_CHOICE, 1, 0.5f, r);
        tc.forward(x);
        for (int load : tc.expertLoad()) {
            assertTrue(load <= 4, "capacity 0.5 * 32 / 4 = 4");
        }
        assertTrue(tc.droppedFraction() >= 0.5);
    }

    @Test
    void upcyclingStartsAsTheDenseNetwork() {
        SplittableRandom r = new SplittableRandom(9);
        SiluMlpExpert dense = new SiluMlpExpert(8, 16, 8, r);
        Var x = Var.of(Tensor.normal(Shape.of(10, 8), 0, 1, r));
        float[] want = dense.forward(x).value().toArray();
        float[] got = SparseUpcycling.upcycle(dense, 4, 2, 0f, r).forward(x).value().toArray();
        for (int i = 0; i < want.length; i++) {
            assertEquals(want[i], got[i], 2e-2 * (1 + Math.abs(want[i])));
        }
        List<SiluMlpExpert> dropped = SparseUpcycling.copies(dense, 2, 0.5f, 1f, r);
        float[] w0 = dropped.get(0).w1().value().toArray();
        float[] w1 = dropped.get(1).w1().value().toArray();
        float[] wd = dense.w1().value().toArray();
        int same = 0;
        int differ = 0;
        for (int i = 0; i < wd.length; i++) {
            same += w0[i] == wd[i] ? 1 : 0;
            differ += w0[i] != w1[i] ? 1 : 0;
        }
        assertEquals(wd.length / 2, same);
        assertTrue(differ > 0);
    }

    @Test
    void sparseDecodeMatchesTheMaskedForward() {
        SplittableRandom r = new SplittableRandom(10);
        Dense dense = new Dense(32, 16, Activation.RELU, Init.HE, r);
        Tensor calibration = Tensor.normal(Shape.of(64, 32), 0, 1, r);
        SparseActivationDense sparse = SparseActivationDense.calibrated(dense, 0.4f, calibration);
        Tensor probe = Tensor.normal(Shape.of(1, 32), 0, 1, r);
        float[] masked = sparse.forward(Var.of(probe)).value().toArray();
        float[] fast = sparse.decode(probe.toArray());
        assertArrayEquals(masked, fast, 1e-4f);
        sparse.resetCounts();
        sparse.forward(Var.of(calibration));
        assertEquals(0.6, sparse.density(), 0.02);
        assertArrayEquals(dense.forward(Var.of(probe)).value().toArray(),
                new SparseActivationDense(dense, 0).decode(probe.toArray()), 1e-4f);
    }

    @Test
    void specsDropIntoAChain() {
        LayerSpec chain = LayerChain.of(CanonLayer.of(4), Softpick.of(2, true, 10000f), SigmoidAttention.of(2, true, 0),
                StickBreakingAttention.of(2), TalkingHeadsAttention.of(2, true, 0), SelectiveAttention.of(2, 0),
                LatentAttention.of(2, 4, 2), ExpertChoiceMoe.spec(2, 8, ExpertChoiceMoe.Routing.TOKEN_CHOICE, 1, 1.5f,
                        1e-3f), SiluMlpExpert.spec(2f));
        Layer l = chain.create(Shape.of(5, 8), new SplittableRandom(1));
        Var y = l.forward(input(2, 5, 8, 1), true);
        assertEquals(Shape.of(2, 5, 8), y.shape());
        assertNotNull(l.auxiliaryLoss());
    }
}
