package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.io.Jsonl;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RunLogTest {

    @Test
    void recordsEpochsBatchesAndWritesADashboard(@TempDir Path dir) throws IOException {
        Dataset xor = Dataset.of(new float[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}}, new float[][] {{0}, {1}, {1}, {0}});
        Model model = Model.sequential().input(2).dense(4, Activation.TANH).dense(1).seed(1).build();
        Path run = dir.resolve("xor");
        try (RunLog log = RunLog.create(run, Map.of("lr", 0.05, "hidden", 4)).everyBatches(2)) {
            Trainer.of(model).loss(Loss.MSE).optimizer(Optimizer.adam(0.05f)).batchSize(2).epochs(3).seed(1)
                    .listener(log).fit(xor);
        }
        List<Map<String, Object>> lines = Jsonl.read(run.resolve("metrics.jsonl"));
        assertEquals(3 + 3, lines.size());   // 3 epochs, and 6 steps of which every second
        assertTrue(Files.readString(run.resolve("config.json")).contains("\"hidden\""));
        List<String> csv = Files.readAllLines(run.resolve("metrics.csv"));
        assertEquals("step,seconds,batch_loss,loss,lr", csv.getFirst());
        String html = Files.readString(run.resolve("dashboard.html"));
        assertTrue(html.contains("<polyline") && html.contains("batch_loss"));
    }
}
