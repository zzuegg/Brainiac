package io.github.zzuegg.brainiac.experiments.imaging;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class PosteriorSamplingTest {

    @Test
    void exactPosteriors() {
        Map<String, double[]> r = PosteriorSampling.exact(120, 60, new double[] {0.05, 0.5}, 3);
        System.out.printf("13.4.7 posterior sampling on GMM priors {SW2, MMD, mode coverage, evals/sample}: %s%n",
                TomographyTest.line(r, "%.3f"));
        String nl = "2-D magnitude, sigma_y 0.05";
        assertTrue(r.get(nl + " / DAPS")[0] < r.get(nl + " / DPS")[0], TomographyTest.line(r));
        String lin = "2-D inpaint, sigma_y 0.5";
        assertTrue(r.get(lin + " / DDRM")[0] < r.get(lin + " / exact (second draw)")[0] + 0.3, TomographyTest.line(r));
        String hi = "8-D inpaint, sigma_y 0.05";
        assertTrue(r.get(hi + " / MAP (multi-start)")[2] < r.get(hi + " / DAPS")[2], TomographyTest.line(r));
    }

    @Test
    void flowPrior() {
        Map<String, double[]> r = PosteriorSampling.images(16, 256, 2, 150, 30, 0.01, 5);
        System.out.printf("13.4.8 flow prior 16x16 shapes {PSNR box, 50%% pixels, deblur, 2x SR, SSIM, ms}: %s%n",
                TomographyTest.line(r, "%.2f"));
        double[] obs = r.get("observation (A^T y / |A|^2)");
        double[] flow = r.get("PnP-Flow");
        assertTrue(flow[0] > obs[0] && flow[1] > obs[1], TomographyTest.line(r));
    }
}
