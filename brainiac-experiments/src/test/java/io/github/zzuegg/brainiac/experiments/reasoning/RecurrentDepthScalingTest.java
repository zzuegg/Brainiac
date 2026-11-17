package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.sequence.MixerShootout;

import java.util.List;

import org.junit.jupiter.api.Test;

class RecurrentDepthScalingTest {

    @Test
    void recurrentDepthLearnsTwoHopsAndStaysStableBeyondTrainingDepth() {
        MixerShootout.Task task = RecurrentDepthScaling.pointerChasing(4, 2);
        RecurrentDepthScaling.Row huginn = RecurrentDepthScaling.recurrentDepth(task, 3, new int[] {1, 2, 4, 8}, 600, 1);
        RecurrentDepthScaling.Row fixed = RecurrentDepthScaling.fixed(task, 600, 1);
        System.out.print(RecurrentDepthScaling.table(List.of(huginn, fixed)));
        double[] a = huginn.accuracy();
        System.out.printf("2-hop pointer chasing: recurrent depth r=1/2/4/8 %.3f/%.3f/%.3f/%.3f vs fixed 3 attention "
                + "blocks %.3f (chance 0.25)%n", a[0], a[1], a[2], a[3], fixed.accuracy()[0]);
        assertTrue(a[3] > 0.25 + 0.1, "above chance");
        assertTrue(a[3] >= a[0] - 0.05, "extra iterations do not diverge");
    }
}
