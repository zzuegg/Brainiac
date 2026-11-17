package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.vision.ClassicCnnShootout.Architecture;
import io.github.zzuegg.brainiac.experiments.vision.ClassicCnnShootout.Result;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class ClassicCnnShootoutTest {

    static Map<Architecture, Result> runAll(String title, Architecture... architectures) {
        Map<Architecture, Result> results = new EnumMap<>(Architecture.class);
        StringBuilder line = new StringBuilder(title + ", 256 train images (acc, params, MACs):");
        for (Architecture a : architectures) {
            Result r = ClassicCnnShootout.run(a, 150, 3);
            results.put(a, r);
            line.append(String.format(" %s %.3f/%d/%d;", a, r.accuracy(), r.parameters(), r.multiplyAdds()));
        }
        System.out.println(line);
        for (Result r : results.values()) {
            assertTrue(r.accuracy() > 0.4f, "every model learns: " + results);
        }
        return results;
    }

    @Test
    void classicBlocksLearnShapesFromFewImages() {
        Map<Architecture, Result> results = runAll("classic CNNs", Architecture.CNN, Architecture.CNN_SPATIAL_DROPOUT,
                Architecture.CNN_DROPBLOCK, Architecture.CNN_SE, Architecture.DENSENET, Architecture.MBCONV);
        Result cnn = results.get(Architecture.CNN);
        assertTrue(results.get(Architecture.CNN_DROPBLOCK).accuracy() > cnn.accuracy() - 0.05f,
                "DropBlock does not hurt the small-data CNN");
        assertTrue(results.get(Architecture.DENSENET).parameters() < cnn.parameters(),
                "DenseNet is smaller than the plain CNN");
        assertTrue(results.get(Architecture.MBCONV).accuracy() > cnn.accuracy(), "MBConv beats the plain CNN");
    }
}
