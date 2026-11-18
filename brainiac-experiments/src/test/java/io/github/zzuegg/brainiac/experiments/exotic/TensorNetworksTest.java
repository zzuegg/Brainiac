package io.github.zzuegg.brainiac.experiments.exotic;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.MatrixProductState;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** Wave 6, 6.2: MPS classifier and Born machine. */
class TensorNetworksTest {

    @Test
    void bornMachineIsNormalizedAndSamplesExactly() {
        MatrixProductState mps = MatrixProductState.bornMachine(4, 2, 3, 1);
        int[][] all = new int[16][4];
        for (int i = 0; i < 16; i++) {
            for (int b = 0; b < 4; b++) {
                all[i][b] = (i >> b) & 1;
            }
        }
        mps.dmrgSweep(new int[][] {{1, 0, 1, 1}, {0, 0, 1, 0}}, 0.3, 2);
        float[] lp = mps.logProbability(all).value().toArray();
        double total = 0;
        for (float v : lp) {
            total += Math.exp(v);
        }
        assertEquals(1, total, 1e-4);
        int[][] s = mps.sample(4000, new SplittableRandom(2));
        int hits = 0;
        for (int[] x : s) {
            hits += x[0] == 0 && x[1] == 0 && x[2] == 1 && x[3] == 0 ? 1 : 0;
        }
        assertEquals(Math.exp(lp[4]), hits / 4000.0, 0.03, "exact sampling matches p(0010)");
    }

    @Test
    void mpsClassifiesDigitsAndModelsShapes() {
        ToyTasks.Data train = ToyTasks.digits(10, 30, 8, 0.1, 1);
        ToyTasks.Data test = ToyTasks.digits(10, 20, 8, 0.1, 1, 999);
        TensorNetworks.Classification chi4 = TensorNetworks.classify(train, test, 4, 300, 1);
        TensorNetworks.Classification chi10 = TensorNetworks.classify(train, test, 10, 300, 1);
        TensorNetworks.Classification mlp = TensorNetworks.mlp(train, test, chi10.parameters(), 300, 1);
        TensorNetworks.Generation born8 = TensorNetworks.bornMachine(8, 6, 200, 1);
        TensorNetworks.Generation born32 = TensorNetworks.bornMachine(32, 6, 200, 1);
        TensorNetworks.Generation cnn = TensorNetworks.pixelCnn(60, 200, 1);
        TensorNetworks.Generation bm = TensorNetworks.pbitBoltzmann(100, 200, 1);
        TensorNetworks.Generation indep = TensorNetworks.independent(200, 1);
        System.out.printf("MPS digits acc chi=4 %.2f, chi=10 %.2f (%d params) vs MLP %.2f (%d params);"
                + " shapes NLL nats (ln77=%.2f) / exact-image share: Born chi=8 %.2f/%.2f, chi=32 %.2f/%.2f,"
                + " PixelCNN %.2f/%.2f, p-bit BM -/%.2f, independent %.2f/%.2f%n", chi4.accuracy(), chi10.accuracy(),
                chi10.parameters(), mlp.accuracy(), mlp.parameters(), Math.log(77), born8.nll(), born8.validShare(),
                born32.nll(), born32.validShare(), cnn.nll(), cnn.validShare(), bm.validShare(), indep.nll(),
                indep.validShare());
        assertTrue(chi10.accuracy() > 0.5 && chi10.accuracy() > chi4.accuracy(), "mps " + chi10.accuracy());
        assertTrue(born32.nll() < born8.nll() && born32.nll() < 5, "bond dimension buys likelihood " + born32.nll());
        assertTrue(born32.validShare() > 0.7, "exact samples are real shapes: " + born32.validShare());
        assertTrue(born32.nll() < indep.nll() - 10);
    }
}
