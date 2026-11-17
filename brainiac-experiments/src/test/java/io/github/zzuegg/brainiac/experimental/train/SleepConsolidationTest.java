package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SleepConsolidationTest {

    private static Dataset data(int n) {
        float[][] x = new float[n][2];
        float[][] y = new float[n][3];
        for (int i = 0; i < n; i++) {
            int c = i % 3;
            x[i][0] = c;
            x[i][1] = -c;
            y[i][c] = 1;
        }
        return Dataset.of(x, y);
    }

    @Test
    void sleepPrunesTheStoreToItsBudget() {
        Model model = Model.sequential().input(2).dense(3, Activation.SOFTMAX).seed(1).build();
        for (SleepConsolidation.Pruning rule : SleepConsolidation.Pruning.values()) {
            SleepConsolidation s = SleepConsolidation.of(model, Loss.CROSS_ENTROPY, Optimizer.adam(0.01f), 2)
                    .store(9, rule);
            s.wake(data(30));
            assertEquals(30, s.storeSize());
            s.sleep(3, 8);
            assertEquals(9, s.storeSize(), rule.toString());
            assertEquals(1, s.sleeps());
        }
    }

    @Test
    void sleepsAsATrainerListener() {
        Model model = Model.sequential().input(2).dense(3, Activation.SOFTMAX).seed(1).build();
        SleepConsolidation s = SleepConsolidation.of(model, Loss.CROSS_ENTROPY, Optimizer.adam(0.01f), 2).store(6,
                SleepConsolidation.Pruning.RANDOM);
        Dataset d = data(12);
        s.remember(d);
        Trainer.of(model).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.01f)).epochs(4).batchSize(4)
                .listener(s.listener(2, 2, 4)).fit(d);
        assertEquals(2, s.sleeps());
        assertEquals(6, s.storeSize());
    }
}
