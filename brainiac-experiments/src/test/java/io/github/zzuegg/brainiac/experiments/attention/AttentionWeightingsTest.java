package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class AttentionWeightingsTest {

    @Test
    void sigmoidAndSoftpickTrainAsWellAsSoftmaxWithoutTheSink() {
        List<AttentionWeightings.Row> rows = AttentionWeightings.run(List.of("softmax", "sigmoid", "softpick",
                "gated"), 1, 250, 1);
        System.out.print(AttentionWeightings.table(rows));
        AttentionWeightings.Row softmax = rows.get(0);
        AttentionWeightings.Row sigmoid = rows.get(1);
        AttentionWeightings.Row softpick = rows.get(2);
        System.out.printf("val loss: sigmoid %.4f, softpick %.4f (sink %.3f, mass %.3f), gated %.4f vs softmax %.4f "
                + "(sink %.3f)%n", sigmoid.validationLoss(), softpick.validationLoss(), softpick.sinkShare(),
                softpick.mass(), rows.get(3).validationLoss(), softmax.validationLoss(), softmax.sinkShare());
        assertTrue(sigmoid.validationLoss() < softmax.validationLoss() + 0.1);
        assertTrue(softpick.validationLoss() < softmax.validationLoss() + 0.15);
        assertTrue(softpick.sinkShare() < softmax.sinkShare());
        assertTrue(softpick.mass() < 1);
    }
}
