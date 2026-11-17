package io.github.zzuegg.brainiac.experimental.evolve;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;

import java.util.List;

import org.junit.jupiter.api.Test;

class EvolutionStrategyTest {

    private static double sphere(double[] x, double shift) {
        double s = 0;
        for (double v : x) {
            s += (v - shift) * (v - shift);
        }
        return s;
    }

    @Test
    void noiseIsRegeneratedFromItsSeed() {
        assertArrayEquals(ParameterVector.noise(42, 50), ParameterVector.noise(42, 50));
        double[] a = new double[50];
        ParameterVector.addNoise(a, 42, 1);
        assertArrayEquals(ParameterVector.noise(42, 50), a, 1e-12);
    }

    @Test
    void centredRanksAreScaleFree() {
        assertArrayEquals(new double[] {0.5, -0.5, 0}, EvolutionStrategy.centredRanks(new double[] {1e9, -3, 2}));
    }

    @Test
    void climbsAShiftedSphere() {
        EvolutionStrategy es = EvolutionStrategy.of(new double[20]).sigma(0.1).learningRate(0.05).pairs(20).seed(3);
        double start = sphere(es.center(), 1);
        es.run(x -> -sphere(x, 1), 300);
        double end = sphere(es.center(), 1);
        assertTrue(end < start * 0.02, "sphere " + start + " -> " + end);
        assertEquals(300, es.history().size());
    }

    @Test
    void antitheticSamplingHasLowerVarianceGradient() {
        // estimate the gradient direction of a linear reward several times and compare spread
        double[] w = ParameterVector.noise(9, 30);
        double[] spread = new double[2];
        for (int mode = 0; mode < 2; mode++) {
            double total = 0;
            for (int rep = 0; rep < 10; rep++) {
                EvolutionStrategy es = EvolutionStrategy.of(new double[30]).antithetic(mode == 0).rankShaping(false)
                        .adam(false).learningRate(1).sigma(0.1).pairs(8).seed(rep + 100);
                es.step(x -> {
                    double s = 5;   // a large offset: antithetic pairs cancel it, single samples do not
                    for (int i = 0; i < x.length; i++) {
                        s += w[i] * x[i];
                    }
                    return s + 0.5 * sphere(x, 0);
                });
                double[] g = es.center();
                double dot = 0;
                double norm = 0;
                double wn = 0;
                for (int i = 0; i < 30; i++) {
                    dot += g[i] * w[i];
                    norm += g[i] * g[i];
                    wn += w[i] * w[i];
                }
                total += dot / Math.sqrt(norm * wn);
            }
            spread[mode] = total / 10;
        }
        assertTrue(spread[0] > spread[1], "antithetic cosine " + spread[0] + " vs plain " + spread[1]);
    }

    @Test
    void trainsABrainiacModelWithoutGradients() {
        Model model = Model.sequential().input(2).dense(6, Activation.TANH).dense(1, Activation.SIGMOID).seed(5).build();
        List<Var> params = model.parameters();
        float[][] x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        float[] y = {0, 1, 1, 0};
        java.util.function.ToDoubleFunction<double[]> reward = theta -> {
            ParameterVector.assign(params, theta);
            float[][] out = model.predict(x);
            double loss = 0;
            for (int i = 0; i < 4; i++) {
                loss += (out[i][0] - y[i]) * (out[i][0] - y[i]);
            }
            return -loss;
        };
        double[] start = ParameterVector.flatten(params);
        double before = -reward.applyAsDouble(start);
        EvolutionStrategy es = EvolutionStrategy.of(start).sigma(0.1).learningRate(0.05).pairs(16).seed(1);
        double after = -reward.applyAsDouble(es.run(reward, 250));
        assertTrue(after < 0.1 && after < before, "xor loss " + before + " -> " + after);
    }

    @Test
    void cmaEsSolvesRosenbrock() {
        CmaEs cma = CmaEs.of(new double[] {0, 0, 0, 0}, 0.5).seed(7);
        double[] best = cma.minimize(x -> {
            double s = 0;
            for (int i = 0; i < x.length - 1; i++) {
                s += 100 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(1 - x[i], 2);
            }
            return s;
        }, 6_000);
        assertTrue(cma.bestCost() < 1e-6, "cost " + cma.bestCost());
        for (double v : best) {
            assertEquals(1, v, 1e-2);
        }
    }

    @Test
    void cmaEsLearnsAnIllConditionedEllipse() {
        CmaEs cma = CmaEs.of(new double[] {3, 3, 3, 3, 3, 3}, 1).seed(2);
        cma.minimize(x -> {
            double s = 0;
            for (int i = 0; i < x.length; i++) {
                s += Math.pow(1e4, i / 5.0) * x[i] * x[i];
            }
            return s;
        }, 5_000);
        assertTrue(cma.bestCost() < 1e-8, "cost " + cma.bestCost());
    }
}
