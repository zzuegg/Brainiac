package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.optim.TrainingStabilityExperiment.ClipRun;
import io.github.zzuegg.brainiac.experiments.optim.TrainingStabilityExperiment.Clipping;
import io.github.zzuegg.brainiac.experiments.optim.TrainingStabilityExperiment.DecayRun;
import io.github.zzuegg.brainiac.experiments.optim.TrainingStabilityExperiment.GradNormRun;
import io.github.zzuegg.brainiac.experiments.optim.TrainingStabilityExperiment.Schedule;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class TrainingStabilityExperimentTest {

    @Test
    void cautiousWeightDecayReachesLowerLoss() {
        double plain = 0;
        double cautious = 0;
        double plainNorm = 0;
        double cautiousNorm = 0;
        for (long seed = 1; seed <= 2; seed++) {
            DecayRun p = TrainingStabilityExperiment.cautiousDecay(false, 1e-2f, 0.5f, 400, seed);
            DecayRun c = TrainingStabilityExperiment.cautiousDecay(true, 1e-2f, 0.5f, 400, seed);
            plain += p.loss() / 2;
            cautious += c.loss() / 2;
            plainNorm += p.weightNorm() / 2;
            cautiousNorm += c.weightNorm() / 2;
        }
        System.out.printf("Cautious weight decay (wd 0.5): loss %.3f, weight norm %.1f vs decoupled decay %.3f / "
                + "%.1f%n", cautious, cautiousNorm, plain, plainNorm);
        assertTrue(cautious < plain, "decaying only where the update agrees reaches a lower loss");
        assertTrue(cautiousNorm < 1.5 * plainNorm, "while still keeping the weights small");
    }

    @Test
    void adamCKeepsTheGradientNormFlatAtTheEnd() {
        GradNormRun adamW = TrainingStabilityExperiment.correctedDecay(false, 2e-2f, 1f, 600, 1);
        GradNormRun adamC = TrainingStabilityExperiment.correctedDecay(true, 2e-2f, 1f, 600, 1);
        System.out.printf("AdamC (cosine, wd 1): end/min grad-norm ratio %.2f, loss %.3f vs AdamW %.2f / %.3f; "
                + "curves %s vs %s%n", adamC.endRise(), adamC.loss(), adamW.endRise(), adamW.loss(),
                java.util.Arrays.toString(round(adamC.gradNorm())), java.util.Arrays.toString(round(adamW.gradNorm())));
        assertTrue(adamW.endRise() > 1.3 && adamC.endRise() < adamW.endRise(), "AdamW's gradient norm climbs, AdamC's less");
        assertTrue(adamC.loss() < 1.03 * adamW.loss(), "at no cost in loss");
    }

    static double[] round(double[] x) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = Math.round(x[i] * 100) / 100.0;
        }
        return y;
    }

    @Test
    void zClipCutsSpikesAtHighLearningRate() {
        Map<Clipping, ClipRun> runs = new EnumMap<>(Clipping.class);
        StringBuilder line = new StringBuilder();
        for (Clipping c : Clipping.values()) {
            int spikes = 0;
            double mse = 0;
            for (long seed = 1; seed <= 2; seed++) {
                ClipRun run = TrainingStabilityExperiment.clipping(c, 1e-2f, 600, 25, seed);
                spikes += run.spikes();
                mse += run.finalMse() / 2;
            }
            runs.put(c, new ClipRun((float) mse, spikes));
            line.append(String.format(" %s %.4f/%d", c, mse, spikes));
        }
        System.out.println("Clipping at lr 1e-2 with outlier batches (final MSE/spikes):" + line);
        ClipRun z = runs.get(Clipping.ZCLIP);
        assertTrue(z.spikes() < runs.get(Clipping.NONE).spikes(), "ZClip cuts spikes");
        assertTrue(z.finalMse() < runs.get(Clipping.FIXED_NORM).finalMse(), "and ends below fixed-norm clipping");
    }

    @Test
    void warmupStableMergeMatchesDecayWithoutDecaying() {
        Map<Schedule, Double> loss = new EnumMap<>(Schedule.class);
        StringBuilder line = new StringBuilder();
        for (Schedule s : Schedule.values()) {
            double l = 0;
            for (long seed = 1; seed <= 2; seed++) {
                l += TrainingStabilityExperiment.schedule(s, 2e-2f, 500, seed) / 2;
            }
            loss.put(s, l);
            line.append(String.format(" %s %.3f", s, l));
        }
        System.out.println("Schedules at 500 steps, final loss:" + line);
        assertTrue(loss.get(Schedule.WSM) < loss.get(Schedule.CONSTANT), "merging replaces the decay");
        assertTrue(loss.get(Schedule.WSM) < 1.02 * loss.get(Schedule.WSD), "about as well as WSD's decay");
    }
}
