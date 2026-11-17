package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Wave 5, 3.14: GenRM (direct and CoT) against the discriminative verifier and the PRM, best-of-16. */
class GenerativeVerificationTest {

    @Test
    void generativeVerifierRanksCandidates() {
        GenerativeVerification.Result r = GenerativeVerification.run(new ChainTask(3), 40, 600, 16, 4, 1);
        System.out.printf("best-of-16 accuracy: one sample %.3f, discriminative ORM %.3f, PRM %.3f, GenRM direct %.3f,"
                + " GenRM-CoT (4 rationales) %.3f, oracle %.3f%n", r.single(), r.discriminative(), r.prm(),
                r.genrmDirect(), r.genrmCot(), r.oracle());
        assertTrue(r.genrmCot() > r.single() + 0.1, "GenRM-CoT lifts best-of-N well above one sample");
        assertTrue(r.genrmCot() >= r.discriminative(), "and matches or beats the discriminative verifier");
    }
}
