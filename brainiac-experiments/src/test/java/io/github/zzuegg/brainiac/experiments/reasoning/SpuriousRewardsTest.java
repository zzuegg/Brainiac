package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.reasoning.SpuriousRewards.Signal;
import org.junit.jupiter.api.Test;

/** Wave 5, 3.8 and 3.7: GRPO from the same weak base under true, format-only, random and label-free rewards. */
class SpuriousRewardsTest {

    private static SpuriousRewards.Result run(Signal s) {
        SpuriousRewards.Result r = SpuriousRewards.run(ArithmeticTask.addition(6), s, 200, 1);
        System.out.printf("%-14s greedy %.2f -> %.2f (gain %+.2f), pass@1 %.2f -> %.2f%n", s, r.before(), r.after(),
                r.gain(), r.passBefore(), r.passAfter());
        return r;
    }

    @Test
    void trueRewardBeatsSpuriousControls() {
        SpuriousRewards.Result truth = run(Signal.TRUE);
        SpuriousRewards.Result format = run(Signal.FORMAT);
        SpuriousRewards.Result random = run(Signal.RANDOM);
        assertTrue(truth.gain() > 0, "the true reward helps");
        assertTrue(truth.gain() > random.gain() && truth.gain() > format.gain(), "and more than the controls");
    }

    /** A barely trained base (20 steps on every puzzle), Adam 1e-3, 150 steps: where label-free signals do best here. */
    @Test
    void labelFreeRewardsOnABarelyTrainedBase() {
        SpuriousRewards.Result[] rs = new SpuriousRewards.Result[3];
        Signal[] signals = {Signal.SELF_CERTAINTY, Signal.MAJORITY, Signal.RANDOM};
        for (int i = 0; i < 3; i++) {
            rs[i] = SpuriousRewards.run(ArithmeticTask.addition(6), signals[i], 150, 1, 20, 1.0, 1e-3f);
            System.out.printf("weak base, %-14s greedy %.2f -> %.2f, pass@1 %.2f -> %.2f%n", signals[i], rs[i].before(),
                    rs[i].after(), rs[i].passBefore(), rs[i].passAfter());
        }
        assertTrue(rs[0].after() >= rs[0].before(), "Intuitor does not lose accuracy here");
        assertTrue(rs[1].after() >= rs[1].before(), "TTRL does not lose accuracy here");
    }
}
