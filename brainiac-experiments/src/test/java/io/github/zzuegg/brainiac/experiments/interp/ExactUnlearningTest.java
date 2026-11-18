package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ExactUnlearningTest {

    @Test
    void sisaAndS3tForgetExactlyAndScrubApproximately() {
        List<UnlearningComparison.Exact> rows = UnlearningComparison.exact(400, 40, 4, 1);
        for (UnlearningComparison.Exact r : rows) {
            System.out.printf("Exact unlearning %-16s forget %.2f test %.2f MIA AUC %.2f cost %.0f%n", r.method(),
                    r.forget(), r.test(), r.membershipAuc(), r.cost());
        }
        UnlearningComparison.Exact retrain = rows.get(0);
        UnlearningComparison.Exact original = rows.get(1);
        UnlearningComparison.Exact sisa = rows.get(3);
        UnlearningComparison.Exact s3t = rows.get(5);
        UnlearningComparison.Exact scrub = rows.get(6);
        System.out.printf("Exact unlearning summary: forget acc retrain %.2f original %.2f SISA %.2f S3T %.2f SCRUB "
                + "%.2f; SISA cost per deletion %.0f vs retrain %.0f%n", retrain.forget(), original.forget(),
                sisa.forget(), s3t.forget(), scrub.forget(), sisa.cost() / 40, retrain.cost());
        assertTrue(original.forget() > retrain.forget() + 0.3, "original memorised");
        assertTrue(sisa.forget() < original.forget() && s3t.forget() < original.forget(), "exact methods forget");
        assertTrue(scrub.forget() < original.forget(), "SCRUB forgets");
    }

    @Test
    void sisaDeletionStreamCostsAFractionOfRetraining() {
        List<UnlearningComparison.Stream> rows = UnlearningComparison.deletionStream(400, new double[] {0.01, 0.05,
            0.1}, 4, 1);
        for (UnlearningComparison.Stream s : rows) {
            System.out.printf("Deletion stream %.0f%%: SISA test %.3f cost share %.3f, S3T test %.3f depth %d, retrain "
                    + "test %.3f%n", 100 * s.deleted(), s.sisaTest(), s.sisaCostShare(), s.s3tTest(), s.s3tDepth(),
                    s.retrainTest());
            assertTrue(s.sisaCostShare() < 0.6, "SISA cheaper than retraining " + s.sisaCostShare());
        }
    }
}
