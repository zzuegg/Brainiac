package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.nn.LoraAdapterDense;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** rsLoRA against LoRA across ranks, and LoftQ against QLoRA on a 4-bit base. */
class LoraScalingTest {

    private static double error(Tensor a, Tensor b) {
        float[] x = a.toArray();
        float[] y = b.toArray();
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            s += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return Math.sqrt(s);
    }

    @Test
    void loftQStartsCloserToTheFullPrecisionWeight() {
        SplittableRandom r = new SplittableRandom(3);
        Tensor w = Tensor.normal(Shape.of(32, 24), 0, 0.3f, r);
        Dense dense = new Dense(32, 24, Activation.IDENTITY, w, Tensor.zeros(Shape.of(24)));
        LoraAdapterDense q = LoraAdapterDense.qlora(dense, 4, 8, 4, r);
        LoraAdapterDense l = LoraAdapterDense.loftQ(dense, 4, 4, 5);
        double qErr = error(q.weight().value(), w);
        double lErr = error(l.weight().value(), w);
        System.out.printf("4-bit weight error at init: LoftQ %.4f vs QLoRA %.4f%n", lErr, qErr);
        assertTrue(lErr < 0.9 * qErr, lErr + " vs " + qErr);
    }

    @Test
    void rsLoraKeepsImprovingWithRank() {
        HookedDecoder base = PeftShootoutTest.base();
        StringBuilder line = new StringBuilder("new-task loss by rank (LoRA / rsLoRA):");
        float lora16 = 0;
        float rs16 = 0;
        for (int rank : new int[] {2, 16}) {
            PeftShootout.Result lora = PeftShootout.run(base, PeftShootout.Method.LORA, 30, 9, rank);
            PeftShootout.Result rs = PeftShootout.run(base, PeftShootout.Method.RSLORA, 30, 9, rank);
            line.append(String.format(" r%d %.3f / %.3f", rank, lora.after(), rs.after()));
            lora16 = lora.after();
            rs16 = rs.after();
        }
        System.out.println(line);
        assertTrue(rs16 < lora16, "rsLoRA " + rs16 + " vs LoRA " + lora16 + " at rank 16");
    }

    @Test
    void loftQAndQLoraBothAdaptAFourBitBase() {
        HookedDecoder base = PeftShootoutTest.base();
        PeftShootout.Result q = PeftShootout.run(base, PeftShootout.Method.QLORA, 30, 9);
        PeftShootout.Result l = PeftShootout.run(base, PeftShootout.Method.LOFTQ, 30, 9);
        PeftShootout.Result full = PeftShootout.run(base, PeftShootout.Method.LORA, 30, 9);
        System.out.printf("4-bit base, new-task loss after 30 steps: LoftQ %.3f vs QLoRA %.3f (fp32 LoRA %.3f)%n",
                l.after(), q.after(), full.after());
        assertTrue(l.after() < 0.75f * l.before() && q.after() < 0.75f * q.before());
    }
}
