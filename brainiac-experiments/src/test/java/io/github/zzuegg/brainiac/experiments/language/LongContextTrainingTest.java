package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.SequenceLengthWarmup;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;

import org.junit.jupiter.api.Test;

class LongContextTrainingTest {

    @Test
    void positionInterpolationExtendsTheContext() {
        HookedDecoder base = LongContextTraining.shortContextModel(120, 8, 5);
        LongContextTraining.Extension extra = LongContextTraining.extend(base, 8, 24, false, 20, 3);
        LongContextTraining.Extension pi = LongContextTraining.extend(base, 8, 24, true, 20, 3);
        // both recipes recover with a short fine-tune at the new length; interpolation keeps
        // every rotary angle inside the trained range while doing so
        System.out.printf("length 8 -> 24: interpolation loss %.3f -> %.3f, extrapolation loss %.3f -> %.3f%n",
                pi.lossBeforeTuning(), pi.lossAfterTuning(), extra.lossBeforeTuning(), extra.lossAfterTuning());
        assertTrue(pi.lossAfterTuning() < pi.lossBeforeTuning() && pi.lossAfterTuning() < 0.3f, pi.toString());
        assertTrue(extra.lossAfterTuning() < 0.3f, extra.toString());
    }

    @Test
    void sequenceLengthWarmupReachesTheSameLossWithFewerTokens() {
        LongContextTraining.Run full = LongContextTraining.train(100, 16, null, 2);
        LongContextTraining.Run warm = LongContextTraining.train(100, 16, new SequenceLengthWarmup(4, 16, 50, 4), 2);
        System.out.printf("sequence-length warmup: loss %.3f with %d tokens vs full length %.3f with %d tokens%n",
                warm.finalLoss(), warm.tokens(), full.finalLoss(), full.tokens());
        assertTrue(warm.tokens() < full.tokens());
        assertTrue(warm.finalLoss() < full.finalLoss() + 0.3f, full + " " + warm);
    }
}
