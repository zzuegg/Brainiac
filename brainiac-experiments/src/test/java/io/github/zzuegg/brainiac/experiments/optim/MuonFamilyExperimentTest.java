package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.train.AdamMini;
import io.github.zzuegg.brainiac.experimental.train.AdaptiveMuon;
import io.github.zzuegg.brainiac.experimental.train.Apollo;
import io.github.zzuegg.brainiac.experimental.train.GaLore;
import io.github.zzuegg.brainiac.experimental.train.NorMuon;
import io.github.zzuegg.brainiac.experimental.train.OrthogonalMuon;
import io.github.zzuegg.brainiac.experimental.train.Orthogonalizer;
import io.github.zzuegg.brainiac.experimental.train.PolarExpress;
import io.github.zzuegg.brainiac.experiments.optim.ModernOptimizerExperiment.Run;
import io.github.zzuegg.brainiac.train.Optimizer;

import org.junit.jupiter.api.Test;

class MuonFamilyExperimentTest {

    static final float[] MUON_RATES = {0.01f, 0.03f};
    static final int STEPS = 300;

    static double[] muon(Orthogonalizer o) {
        return MuonFamilyExperiment.sweep(
                lr -> MuonFamilyExperiment.withAdamForVectors(new OrthogonalMuon(lr, 0, 0.95f, o, false)), MUON_RATES,
                STEPS, 1);
    }

    /** The better of two learning rates: {loss, spread}. */
    static double[] tunedSpread(java.util.function.Function<Float, Optimizer> make, float a, float b) {
        double[] x = MuonFamilyExperiment.neuronSpread(make.apply(a), STEPS, 100, 1);
        double[] y = MuonFamilyExperiment.neuronSpread(make.apply(b), STEPS, 100, 1);
        return x[0] <= y[0] ? x : y;
    }

    @Test
    void polarExpressMatchesNewtonSchulzWithFewerIterations() {
        double ns3 = MuonFamilyExperiment.best(muon(Orthogonalizer.newtonSchulz(3)));
        double pe3 = MuonFamilyExperiment.best(muon(PolarExpress.of(3)));
        double ns5 = MuonFamilyExperiment.best(muon(Orthogonalizer.newtonSchulz(5)));
        double pe5 = MuonFamilyExperiment.best(muon(PolarExpress.of(5)));
        System.out.printf("Muon + Polar Express: loss %.3f (3 it) / %.3f (5 it) vs Newton-Schulz %.3f / %.3f "
                + "(entropy %.3f)%n", pe3, pe5, ns3, ns5, MuonFamilyExperiment.SOURCE.entropy());
        assertTrue(pe3 < 1.03 * ns3 && pe5 < 1.03 * ns5, "Polar Express trains at least as well");
        assertTrue(pe3 < 1.03 * ns5, "and three of its iterations do the work of five Newton-Schulz steps");
    }

    @Test
    void norMuonEvensOutNeuronUpdates() {
        double[] adamW = tunedSpread(lr -> Optimizer.adamW(lr, 0.01f), 3e-3f, 1e-2f);
        double[] muon = tunedSpread(lr -> MuonFamilyExperiment.withAdamForVectors(OrthogonalMuon.polarExpress(lr, 5)),
                0.01f, 0.03f);
        double[] nor = tunedSpread(lr -> MuonFamilyExperiment.withAdamForVectors(NorMuon.of(lr)), 0.01f, 0.03f);
        System.out.printf("NorMuon: loss %.3f, neuron update spread (max/median) %.2f vs Muon %.3f / %.2f, "
                + "AdamW %.3f / %.2f%n", nor[0], nor[1], muon[0], muon[1], adamW[0], adamW[1]);
        assertTrue(nor[1] < muon[1], "neuron-wise normalisation narrows the spread of update norms");
        assertTrue(nor[0] < 1.03 * muon[0] && nor[0] < 1.03 * adamW[0], "and trains as well");
    }

    @Test
    void adaptiveMuonsTrainAndAreLessSensitiveToTheLearningRate() {
        float[] rates = {0.01f, 0.03f, 0.1f, 0.3f};
        double[] muon = MuonFamilyExperiment.sweep(
                lr -> MuonFamilyExperiment.withAdamForVectors(OrthogonalMuon.polarExpress(lr, 5)), rates, 150, 1);
        double[] nor = MuonFamilyExperiment.sweep(
                lr -> MuonFamilyExperiment.withAdamForVectors(NorMuon.of(lr)), rates, 150, 1);
        StringBuilder line = new StringBuilder();
        double bestAdaptive = Double.POSITIVE_INFINITY;
        double leastSensitive = Double.POSITIVE_INFINITY;
        for (AdaptiveMuon.Mode mode : AdaptiveMuon.Mode.values()) {
            double[] l = MuonFamilyExperiment.sweep(
                    lr -> MuonFamilyExperiment.withAdamForVectors(AdaptiveMuon.of(mode, lr)), rates, 150, 1);
            line.append(String.format(" %s %.3f/%.3f", mode, MuonFamilyExperiment.best(l),
                    MuonFamilyExperiment.sensitivity(l)));
            bestAdaptive = Math.min(bestAdaptive, MuonFamilyExperiment.best(l));
            leastSensitive = Math.min(leastSensitive, MuonFamilyExperiment.sensitivity(l));
        }
        System.out.printf("Adaptive Muon (best loss/lr sensitivity):%s vs Muon %.3f/%.3f, NorMuon %.3f/%.3f%n", line,
                MuonFamilyExperiment.best(muon), MuonFamilyExperiment.sensitivity(muon),
                MuonFamilyExperiment.best(nor), MuonFamilyExperiment.sensitivity(nor));
        assertTrue(bestAdaptive < 1.03 * MuonFamilyExperiment.best(muon), "an adaptive variant keeps up with Muon");
        assertTrue(leastSensitive < MuonFamilyExperiment.sensitivity(muon), "and one is less sensitive to the rate");
    }

    /** The run with the lowest final loss over two learning rates. */
    static Run tunedRun(java.util.function.Function<Float, Optimizer> make) {
        Run best = null;
        for (float lr : new float[] {1e-2f, 3e-2f}) {
            Run run = ModernOptimizerExperiment.train(() -> make.apply(lr), MuonFamilyExperiment.SOURCE, 64, STEPS, 32,
                    25, 1);
            best = best == null || run.finalLoss() < best.finalLoss() ? run : best;
        }
        return best;
    }

    @Test
    void apolloTrainsWithSgdLikeState() {
        Run adamW = tunedRun(lr -> Optimizer.adamW(lr, 0.01f));
        Run apollo = tunedRun(lr -> Apollo.of(lr, 8));
        Run mini = tunedRun(Apollo::mini);
        Run galore = tunedRun(lr -> new GaLore(lr, 8, 100, 1, 0.01f));
        Run eight = tunedRun(lr -> Optimizer.adam8bit(lr, 0.9f, 0.999f, 1e-8f, 0.01f));
        Run adamMini = tunedRun(lr -> AdamMini.of(lr, 0.01f));
        System.out.printf("APOLLO: loss %.3f / %d state bytes, APOLLO-Mini %.3f / %d B vs GaLore %.3f / %d B, "
                + "Adam8bit %.3f / %d B, Adam-mini %.3f / %d B, AdamW %.3f / %d B%n", apollo.finalLoss(),
                apollo.stateBytes(), mini.finalLoss(), mini.stateBytes(), galore.finalLoss(), galore.stateBytes(),
                eight.finalLoss(), eight.stateBytes(), adamMini.finalLoss(), adamMini.stateBytes(), adamW.finalLoss(),
                adamW.stateBytes());
        assertTrue(apollo.stateBytes() < 0.5 * adamW.stateBytes() && mini.stateBytes() < apollo.stateBytes(),
                "a fraction of AdamW's state");
        assertTrue(apollo.finalLoss() < galore.finalLoss() && mini.finalLoss() < galore.finalLoss(),
                "below GaLore at less memory");
        assertTrue(apollo.finalLoss() < 1.1 * adamW.finalLoss() && mini.finalLoss() < 1.1 * adamW.finalLoss(),
                "and within 10% of AdamW");
    }
}
