package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.MomentumTeacher;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class SelfSupervisedLossesTest {

    private static Tensor randn(long seed, int... dims) {
        return Tensor.normal(Shape.of(dims), 0, 1, new SplittableRandom(seed));
    }

    @Test
    void ntXentPrefersMatchingPairs() {
        Tensor a = randn(1, 6, 4);
        float matched = SelfSupervisedLosses.ntXent(Var.of(a), Var.of(a.add(randn(2, 6, 4).mul(0.01f))), 0.2f)
                .value().item();
        float random = SelfSupervisedLosses.ntXent(Var.of(a), Var.of(randn(3, 6, 4)), 0.2f).value().item();
        assertTrue(matched < random - 1, matched + " vs " + random);
        GradCheck.Result r = GradCheck.check(v -> SelfSupervisedLosses.ntXent(v.get(0), v.get(1), 0.5f),
                List.of(randn(4, 3, 4), randn(5, 3, 4)));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void byolIsZeroForAlignedViewsAndIgnoresTheTargetGradient() {
        Var p = Var.param(randn(6, 4, 3));
        Var z = Var.param(p.value().mul(2f));
        assertEquals(0f, SelfSupervisedLosses.byol(p, z).value().item(), 1e-5f);
        SelfSupervisedLosses.byol(p, Var.param(randn(7, 4, 3))).backward();
        assertTrue(p.grad() != null);
    }

    @Test
    void dinoCenteringAndGradient() {
        SelfSupervisedLosses.Center c = new SelfSupervisedLosses.Center(3, 0.5f);
        c.update(Tensor.of(Shape.of(2, 3), 1f, 2f, 3f, 3f, 2f, 1f));
        assertArrayEquals(new float[] {1f, 1f, 1f}, c.value().toArray(), 1e-6f);
        Tensor teacher = randn(8, 4, 3);
        GradCheck.Result r = GradCheck.check(v -> SelfSupervisedLosses.dino(v.get(0), Var.of(teacher), c.value(),
                0.1f, 0.04f), List.of(randn(9, 4, 3).mul(0.1f)), 1e-3f, 1e-2f);
        assertTrue(r.passed(), r.toString());
        // the student's best response to a sharpened teacher puts its mass on the teacher's argmax
        Var s = Var.param(Tensor.zeros(Shape.of(1, 3)));
        for (int i = 0; i < 200; i++) {
            SelfSupervisedLosses.dino(s, Var.of(Tensor.of(Shape.of(1, 3), 0f, 1f, 0f)), Tensor.zeros(Shape.of(3)),
                    0.1f, 0.04f).backward();
            s.assign(s.value().sub(s.grad().mul(0.01f)));
            s.zeroGrad();
        }
        float[] sv = s.value().toArray();
        assertTrue(sv[1] > sv[0] && sv[1] > sv[2]);
    }

    @Test
    void barlowTwinsAndVicregAreLowForDecorrelatedInvariantViews() {
        // identical views of nearly uncorrelated features: the correlation matrix is close to the identity
        Tensor z = randn(10, 64, 4);
        float same = SelfSupervisedLosses.barlowTwins(Var.of(z), Var.of(z), 0.25f).value().item();
        float other = SelfSupervisedLosses.barlowTwins(Var.of(z), Var.of(randn(11, 64, 4)), 0.25f).value().item();
        assertTrue(same < 0.5f && other > 2f, same + " vs " + other);
        float collapsed = SelfSupervisedLosses.vicreg(Var.of(Tensor.zeros(Shape.of(64, 4))),
                Var.of(Tensor.zeros(Shape.of(64, 4))), 25, 25, 1).value().item();
        float spread = SelfSupervisedLosses.vicreg(Var.of(z.mul(2f)), Var.of(z.mul(2f)), 25, 25, 1).value().item();
        assertTrue(collapsed > 40 && spread < 2, collapsed + " vs " + spread);
        GradCheck.Result r = GradCheck.check(v -> SelfSupervisedLosses.vicreg(v.get(0), v.get(1), 1, 1, 1)
                .add(SelfSupervisedLosses.barlowTwins(v.get(0), v.get(1), 0.1f)),
                List.of(randn(12, 6, 3), randn(13, 6, 3)), 1e-3f, 2e-2f);
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void momentumTeacherFollowsTheStudent() {
        Var s = Var.param(Tensor.of(1f, 1f));
        Var t = Var.param(Tensor.of(0f, 0f));
        MomentumTeacher teacher = new MomentumTeacher(List.of(t), List.of(s), 0.5f);
        assertArrayEquals(new float[] {1f, 1f}, t.value().toArray());
        s.assign(Tensor.of(3f, 5f));
        teacher.update();
        assertArrayEquals(new float[] {2f, 3f}, t.value().toArray());
    }

    @Test
    void patchMaskingRoundTrips() {
        PatchMasking.Mask m = PatchMasking.random(2, 8, 0.75f, new SplittableRandom(14));
        assertEquals(2, m.keep()[0].length);
        assertEquals(6, m.masked()[1].length);
        Tensor x = randn(15, 2, 8, 3);
        Var kept = PatchMasking.gather(Var.of(x), m.keep());
        Var hidden = PatchMasking.gather(Var.of(x), m.masked());
        assertArrayEquals(x.toArray(), PatchMasking.unshuffle(kept, hidden, m).value().toArray());
        assertEquals(12f, PatchMasking.indicator(m, 8).sum().item());
    }

    @Test
    void prototypesClassifyByNearestMean() {
        Tensor support = Tensor.of(Shape.of(4, 2), 0f, 0f, 0f, 2f, 10f, 10f, 10f, 12f);
        Var protos = PrototypicalNetworks.prototypes(Var.of(support), new int[] {0, 0, 1, 1}, 2);
        assertArrayEquals(new float[] {0f, 1f, 10f, 11f}, protos.value().toArray());
        int[] pred = PrototypicalNetworks.classify(Var.of(Tensor.of(Shape.of(2, 2), 1f, 1f, 9f, 9f)), protos);
        assertArrayEquals(new int[] {0, 1}, pred);
        GradCheck.Result r = GradCheck.check(v -> PrototypicalNetworks.loss(v.get(0), new int[] {0, 1, 0, 1},
                v.get(1), new int[] {1, 0}, 2), List.of(randn(16, 4, 3), randn(17, 2, 3)));
        assertTrue(r.passed(), r.toString());
    }
}
