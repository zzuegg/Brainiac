package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.VariationalGru;
import io.github.zzuegg.brainiac.experiments.techniques.ClassicRegularizers.ImageResult;
import io.github.zzuegg.brainiac.experiments.techniques.ClassicRegularizers.Regularizer;

import org.junit.jupiter.api.Test;

class ClassicRegularizersTest {

    @Test
    void dropConnectAndConcreteDropoutRegulariseFewShotShapes() {
        ImageResult none = ClassicRegularizers.images(Regularizer.NONE, 400, 1);
        ImageResult dropout = ClassicRegularizers.images(Regularizer.DROPOUT, 400, 1);
        ImageResult connect = ClassicRegularizers.images(Regularizer.DROPCONNECT, 400, 1);
        ImageResult concrete = ClassicRegularizers.images(Regularizer.CONCRETE, 400, 1);
        System.out.printf("few-shot shapes acc/ECE: none %.3f/%.3f, dropout %.3f/%.3f, DropConnect %.3f/%.3f, "
                + "concrete %.3f/%.3f (learned p %.2f)%n", none.accuracy(), none.ece(), dropout.accuracy(),
                dropout.ece(), connect.accuracy(), connect.ece(), concrete.accuracy(), concrete.ece(),
                concrete.learnedRate());
        assertTrue(connect.accuracy() > none.accuracy() && concrete.accuracy() > none.accuracy(),
                "DropConnect and concrete dropout beat no regulariser");
        assertTrue(connect.ece() < none.ece() && dropout.ece() < none.ece() && concrete.ece() < none.ece(),
                "every regulariser improves calibration");
        assertTrue(concrete.learnedRate() > 0.01 && concrete.learnedRate() < 0.9, "a rate was learned");
    }

    @Test
    void tiedMasksMatchPerStepMasksInAGruLm() {
        double tied = ClassicRegularizers.gruLanguageModel(VariationalGru.Masks.TIED, 0.3f, 48, 150, 2);
        double perStep = ClassicRegularizers.gruLanguageModel(VariationalGru.Masks.PER_STEP, 0.3f, 48, 150, 2);
        double none = ClassicRegularizers.gruLanguageModel(VariationalGru.Masks.NONE, 0f, 48, 150, 2);
        System.out.printf("GRU LM next-token accuracy: variational (tied) %.3f, per-step masks %.3f, no dropout %.3f%n",
                tied, perStep, none);
        assertTrue(tied > perStep - 0.02, "tied masks are not worse than per-step masks");
        assertTrue(tied > none, "variational dropout helps the small-data GRU");
    }
}
