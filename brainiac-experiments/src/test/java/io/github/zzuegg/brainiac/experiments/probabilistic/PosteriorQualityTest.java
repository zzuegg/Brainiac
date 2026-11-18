package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class PosteriorQualityTest {

    @Test
    void gradientSamplersBeatTheRandomWalk() {
        for (String target : new String[] {"banana", "funnel"}) {
            Map<String, double[]> r = PosteriorQuality.samplers(target, 1);
            System.out.printf("Samplers on %s {ESS/1000 evals, R-hat, sliced W}: %s%n", target,
                    ProbabilisticProgramsTest.line(r));
            assertTrue(r.get("NUTS")[1] < 1.1, target + " NUTS R-hat");
            assertTrue(r.get("NUTS")[2] < r.get("random-walk MH")[2] + 0.05, target + " NUTS sliced W");
        }
    }

    @Test
    void steinParticlesCoverEveryMode() {
        Map<String, double[]> r = PosteriorQuality.steinMixture(2);
        System.out.printf("Four-mode mixture {MMD^2, modes covered}: %s%n", ProbabilisticProgramsTest.line(r));
        assertTrue(r.get("SVGD")[1] == 4, "SVGD covers all modes");
        assertTrue(r.get("SVGD")[0] < r.get("gradient-ascent ensemble")[0], "SVGD beats the collapsed ensemble");
    }

    @Test
    void flowsFitCurvedPosteriorsBetterThanMeanField() {
        for (String target : new String[] {"banana", "two moons"}) {
            Map<String, double[]> r = PosteriorQuality.variational(target, 300, 3);
            System.out.printf("VI on %s {ELBO, KL, s}: %s%n", target, ProbabilisticProgramsTest.line(r));
            double best = Math.max(r.get("PLANAR")[0], Math.max(r.get("RADIAL")[0], r.get("IAF")[0]));
            assertTrue(best > r.get("MEAN_FIELD")[0], target + " a flow beats mean field");
        }
    }

    @Test
    void annealingEstimatesTheEvidence() {
        Map<String, double[]> r = PosteriorQuality.evidence(4);
        System.out.printf("log Z (true 3) {|bias|, std, evals}: %s%n", ProbabilisticProgramsTest.line(r));
        Map<String, double[]> v = PosteriorQuality.linearGaussianVae(5);
        System.out.printf("Linear-Gaussian VAE log p(x) {estimate, gap to exact}: %s%n",
                ProbabilisticProgramsTest.line(v));
        assertTrue(r.get("AIS")[0] < 0.3, "AIS bias");
        assertTrue(Math.abs(v.get("AIS from encoder")[1]) < v.get("ELBO (k=1)")[1], "AIS tighter than the ELBO");
    }
}
