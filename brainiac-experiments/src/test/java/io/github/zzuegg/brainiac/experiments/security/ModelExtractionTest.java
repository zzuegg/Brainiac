package io.github.zzuegg.brainiac.experiments.security;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class ModelExtractionTest {

    @Test
    void equationSolvingRecoversASoftmaxRegression() {
        ModelExtraction.Solved s = ModelExtraction.equationSolving(1);
        System.out.printf("Equation solving %d queries: agreement digits %.3f, noise %.3f%n", s.queries(),
                s.agreement(), s.noiseAgreement());
        assertTrue(s.agreement() > 0.99 && s.noiseAgreement() > 0.99, s.toString());
    }

    @Test
    void knockoffStealsWithProbabilitiesAndDefencesKeepHonestAccuracy() {
        List<ModelExtraction.Row> rows = ModelExtraction.extraction(new int[] {200, 1500}, 1);
        Map<String, ModelExtraction.Row> big = new HashMap<>();
        for (ModelExtraction.Row r : rows) {
            System.out.printf("Extraction %-22s budget %4d agreement %.3f copy accuracy %.3f%n", r.attack(), r.budget(),
                    r.agreement(), r.accuracy());
            if (r.budget() == 1500 || r.budget() == 0) {
                big.put(r.attack(), r);
            }
        }
        double probs = big.get("KNOCKOFF_RANDOM").agreement();
        double labels = big.get("KNOCKOFF_LABELS").agreement();
        double top1 = big.get("KNOCKOFF_TOP1_DEFENDED").agreement();
        System.out.printf("Extraction 1500 queries: Knockoff probs %.3f labels %.3f top-1 defended %.3f rounded %.3f "
                + "Jacobian %.3f true labels %.3f; honest acc %.3f defended %.3f%n", probs, labels, top1,
                big.get("KNOCKOFF_ROUNDED").agreement(), big.get("JACOBIAN").agreement(),
                big.get("TRUE_LABELS").agreement(), big.get("HONEST_UNDEFENDED").accuracy(),
                big.get("HONEST_TOP1_DEFENDED").accuracy());
        assertTrue(probs > 0.7, "Knockoff agreement " + probs);
        assertTrue(big.get("HONEST_TOP1_DEFENDED").accuracy() == big.get("HONEST_UNDEFENDED").accuracy(),
                "top-1 defence is free for honest users");
        assertTrue(top1 <= probs + 0.02, "defence does not help the attacker");
    }
}
