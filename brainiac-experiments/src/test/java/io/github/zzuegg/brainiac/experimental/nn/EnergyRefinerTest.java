package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class EnergyRefinerTest {

    private static Tensor[] data(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[n * 2];
        float[] y = new float[n * 2];
        for (int i = 0; i < n; i++) {
            float a = (float) r.nextDouble(-1, 1);
            float b = (float) r.nextDouble(-1, 1);
            x[2 * i] = a;
            x[2 * i + 1] = b;
            y[2 * i] = (float) Math.sin(2 * a);
            y[2 * i + 1] = a * b;
        }
        return new Tensor[] {Tensor.of(Shape.of(n, 2), x), Tensor.of(Shape.of(n, 2), y)};
    }

    private static Var noise(int n, SplittableRandom r) {
        float[] v = new float[n * 2];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (r.nextGaussian() * 0.5);
        }
        return Var.of(Tensor.of(Shape.of(n, 2), v));
    }

    @Test
    void handWrittenEnergyGradientMatchesAutograd() {
        EnergyRefiner e = EnergyRefiner.of(2, 2, 8, 3, 0.5f, 1);
        Tensor[] d = data(4, 2);
        Var y = Var.param(d[1].add(0.3f));
        e.energy(Var.of(d[0]), y).sum().backward();
        float[] auto = y.grad().toArray();
        float[] hand = e.energyGradient(Var.of(d[0]), Var.of(d[1].add(0.3f))).value().toArray();
        assertArrayEquals(auto, hand, 1e-5f);
    }

    @Test
    void thinkingLongerLowersTheErrorAndTheEnergyVerifies() {
        EnergyRefiner e = EnergyRefiner.of(2, 2, 48, 6, 0.5f, 3);
        Tensor[] train = data(128, 4);
        SplittableRandom r = new SplittableRandom(5);
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < 300; s++) {
            Var loss = e.loss(Var.of(train[0]), train[1], noise(128, r), 2 + r.nextInt(6));
            loss.backward();
            opt.step(e.parameters());
        }
        Tensor[] test = data(200, 6);
        Var start = noise(200, new SplittableRandom(7));
        double startErr = start.value().sub(test[1]).square().mean().item();
        List<Var> ys = e.refine(Var.of(test[0]), start, 8);
        double[] err = new double[ys.size()];
        for (int k = 0; k < err.length; k++) {
            err[k] = ys.get(k).value().sub(test[1]).square().mean().item();
        }
        System.out.println("EBT error start " + startErr + " by step " + java.util.Arrays.toString(err));
        assertTrue(err[0] < startErr && err[5] < err[0] && err[5] < 0.05, java.util.Arrays.toString(err));

        // verification: the right answer has lower energy than a perturbed one
        float[] good = e.energy(Var.of(test[0]), Var.of(test[1])).value().toArray();
        float[] bad = e.energy(Var.of(test[0]), Var.of(test[1].add(start.value()))).value().toArray();
        int lower = 0;
        for (int i = 0; i < good.length; i++) {
            lower += good[i] < bad[i] ? 1 : 0;
        }
        assertTrue(lower > 0.8 * good.length, lower + "/" + good.length);
    }

    @Test
    void isALayer() {
        Model m = Model.sequential().input(3).add(EnergyRefiner.spec(2, 8, 4, 0.5f)).seed(1).build();
        assertEquals(2, m.predict(new float[3]).length);
    }
}
