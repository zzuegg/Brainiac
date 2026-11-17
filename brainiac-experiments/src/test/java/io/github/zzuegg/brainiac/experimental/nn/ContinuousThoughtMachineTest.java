package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class ContinuousThoughtMachineTest {

    @Test
    void learnsParityByThinkingOverTicks() {
        int bits = 5;
        int n = 1 << bits;
        float[] x = new float[n * bits];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            int ones = 0;
            for (int b = 0; b < bits; b++) {
                int v = (i >> b) & 1;
                x[i * bits + b] = v * 2 - 1;
                ones += v;
            }
            y[i] = ones % 2;
        }
        Tensor xs = Tensor.of(Shape.of(n, bits), x);
        Tensor ys = Tensor.of(Shape.of(n), y);
        ContinuousThoughtMachine ctm = ContinuousThoughtMachine.of(bits, 24, 4, 4, 6, 32, 2, 1);
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < 250; s++) {
            Var loss = ContinuousThoughtMachine.tickLoss(ctm.ticks(Var.of(xs)), ys);
            loss.backward();
            opt.step(ctm.parameters());
        }
        List<Var> ticks = ctm.ticks(Var.of(xs));
        int best = 0;
        for (Var t : ticks) {
            float[] v = t.value().toArray();
            int right = 0;
            for (int i = 0; i < n; i++) {
                right += (v[2 * i + 1] > v[2 * i] ? 1 : 0) == (int) y[i] ? 1 : 0;
            }
            best = Math.max(best, right);
        }
        System.out.println("CTM parity right " + best + "/" + n);
        assertTrue(best >= 0.9 * n, best + "/" + n);
    }

    @Test
    void gradientsThroughTicksMatchNumerical() {
        ContinuousThoughtMachine ctm = ContinuousThoughtMachine.of(3, 4, 3, 3, 3, 5, 2, 7);
        ctm.parameters().forEach(p -> p.assign(p.value().to(DType.FLOAT64)));
        SplittableRandom r = new SplittableRandom(1);
        double[] in = new double[6];
        for (int i = 0; i < in.length; i++) {
            in[i] = r.nextGaussian();
        }
        Tensor x = Tensor.of(Shape.of(2, 3), in, DType.FLOAT64, io.github.zzuegg.brainiac.tensor.Device.getDefault());
        GradCheck.Result res = GradCheck.check(v -> {
            List<Var> t = ctm.ticks(v.get(0));
            return t.get(t.size() - 1).square().sum();
        }, List.of(x));
        assertTrue(res.passed(), res.toString());
    }

    @Test
    void isALayer() {
        Model m = Model.sequential().input(4).add(ContinuousThoughtMachine.spec(8, 3, 4, 3, 6, 2)).seed(1).build();
        assertEquals(2, m.predict(new float[4]).length);
    }
}
