package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.nn.DartsCell;
import io.github.zzuegg.brainiac.experimental.nn.DartsCell.Candidate;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class DartsSearchTest {

    /** y = sum of squares of a random projection: the SQUARE operation fits it exactly. */
    private static Dataset data(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        SplittableRandom w = new SplittableRandom(99);
        double[][] proj = new double[4][4];
        for (double[] row : proj) {
            for (int j = 0; j < 4; j++) {
                row[j] = w.nextGaussian() * 0.5;
            }
        }
        float[][] x = new float[n][4];
        float[][] y = new float[n][1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 4; j++) {
                x[i][j] = (float) r.nextGaussian();
            }
            for (int k = 0; k < 4; k++) {
                double p = 0;
                for (int j = 0; j < 4; j++) {
                    p += x[i][j] * proj[j][k];
                }
                y[i][0] += (float) (p * p);
            }
        }
        return Dataset.of(x, y);
    }

    @Test
    void findsTheOperationThatMatchesTheTask() {
        DartsCell cell = DartsCell.of(4, 1, new SplittableRandom(1), Candidate.ZERO, Candidate.IDENTITY,
                Candidate.LINEAR, Candidate.RELU, Candidate.TANH, Candidate.SQUARE);
        Model model = Model.sequential().input(4).layer(cell).dense(1).seed(2).build();
        List<DartsSearch.Epoch> h = DartsSearch.search(model, cell.architecture(), data(256, 1), data(256, 2),
                Loss.MSE, Optimizer.adam(0.01f), Optimizer.adam(0.03f), 15, 32);
        assertEquals(List.of(Candidate.SQUARE), cell.derive());
        assertTrue(h.getLast().validLoss() < 0.3 * h.getFirst().validLoss(),
                h.getFirst().validLoss() + " -> " + h.getLast().validLoss());
        float[] mix = cell.mixture(0);
        assertTrue(mix[5] > 0.4, "square weight " + mix[5]);
    }

    @Test
    void cellIsALayerSpecForSequentialModels() {
        Model model = Model.sequential().input(3).add(DartsCell.spec(2)).dense(1).seed(1).build();
        assertEquals(1, model.predict(1, 2, 3).length);
        DartsCell cell = (DartsCell) model.layers().getFirst();
        assertEquals(3, cell.edges());
    }
}
