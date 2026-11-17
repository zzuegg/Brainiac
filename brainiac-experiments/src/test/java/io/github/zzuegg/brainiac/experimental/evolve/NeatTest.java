package io.github.zzuegg.brainiac.experimental.evolve;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class NeatTest {

    static double xor(Neat.Network net) {
        double[][] x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[] y = {0, 1, 1, 0};
        double err = 0;
        for (int i = 0; i < 4; i++) {
            err += Math.abs(net.activate(x[i])[0] - y[i]);
        }
        return (4 - err) * (4 - err);
    }

    @Test
    void solvesXorByGrowingHiddenNodes() {
        Neat neat = Neat.of(2, 1).population(150).seed(3);
        Neat.Genome best = neat.run(NeatTest::xor, 150, 15.0);
        Neat.Network net = best.network();
        assertTrue(best.fitness() >= 15.0, "fitness " + best.fitness() + " after " + neat.history().size());
        assertTrue(net.activate(0, 1)[0] > 0.5 && net.activate(1, 0)[0] > 0.5);
        assertTrue(net.activate(0, 0)[0] < 0.5 && net.activate(1, 1)[0] < 0.5);
        assertTrue(best.hiddenNodes() >= 1, "xor needs a hidden node");
    }

    @Test
    void initialGenomesShareInnovationsAndHaveZeroDistanceWithoutWeights() {
        Neat neat = Neat.of(3, 2).population(10).seed(1);
        neat.step(n -> 1);
        List<Neat.Genome> pop = neat.population();
        for (Neat.Genome g : pop) {
            // innovations 0..7 for (3 inputs + bias) x 2 outputs, maybe more after mutation
            assertTrue(g.connections().stream().anyMatch(c -> c.innovation() == 7));
        }
        assertEquals(0, neat.distance(pop.get(0), pop.get(0)), 1e-12);
    }

    @Test
    void hyperNeatPaintsAMirrorConnectivity() {
        int n = 7;
        HyperNeat.Substrate substrate = new HyperNeat.Substrate(
                List.of(HyperNeat.Substrate.line(n, -1), HyperNeat.Substrate.line(n, 1)),
                NeatActivation.TANH, NeatActivation.SIGMOID);
        HyperNeat hyper = HyperNeat.of(substrate);
        hyper.neat().seed(5).population(80);
        // light at position i must light up output n-1-i (and only it)
        java.util.function.ToDoubleFunction<HyperNeat.SubstrateNetwork> fitness = net -> {
            double score = 0;
            for (int i = 0; i < n; i++) {
                double[] in = new double[n];
                in[i] = 1;
                double[] out = net.activate(in);
                int arg = 0;
                for (int j = 1; j < n; j++) {
                    if (out[j] > out[arg]) {
                        arg = j;
                    }
                }
                score += arg == n - 1 - i ? 1 : 0;
                score += 0.5 * (out[n - 1 - i] - (sum(out) - out[n - 1 - i]) / (n - 1));
            }
            return Math.max(0, score);
        };
        HyperNeat.SubstrateNetwork best = hyper.run(fitness, 300, 1.4 * n);
        int correct = 0;
        for (int i = 0; i < n; i++) {
            double[] in = new double[n];
            in[i] = 1;
            double[] out = best.activate(in);
            int arg = 0;
            for (int j = 1; j < n; j++) {
                if (out[j] > out[arg]) {
                    arg = j;
                }
            }
            correct += arg == n - 1 - i ? 1 : 0;
        }
        assertEquals(n, correct, "generations " + hyper.history().size());
        assertTrue(hyper.history().getFirst().bestFitness() < hyper.history().getLast().bestFitness());
    }

    private static double sum(double[] a) {
        double s = 0;
        for (double x : a) {
            s += x;
        }
        return s;
    }
}
