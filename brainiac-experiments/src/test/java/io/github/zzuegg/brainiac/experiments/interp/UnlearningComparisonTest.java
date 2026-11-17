package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class UnlearningComparisonTest {

    @Test
    void unlearningMethodsAgainstRetraining() {
        List<UnlearningComparison.Result> out = UnlearningComparison.run(400, 40, 1);
        for (UnlearningComparison.Result r : out) {
            System.out.printf("Unlearn %-15s forget %.2f  test %.2f  MIA AUC %.2f  relearned %.2f%n", r.method(),
                    r.forget(), r.test(), r.membershipAuc(), r.relearned());
        }
        UnlearningComparison.Result original = out.get(1);
        UnlearningComparison.Result ascent = out.get(2);
        UnlearningComparison.Result npo = out.get(4);
        assertTrue(original.forget() > 0.9 && original.membershipAuc() > 0.8, original.toString());
        assertTrue(npo.forget() < 0.2 && npo.test() > 0.75, npo.toString());
        assertTrue(npo.relearned() < ascent.relearned() && npo.membershipAuc() < original.membershipAuc(),
                npo + " vs " + ascent);
    }
}
