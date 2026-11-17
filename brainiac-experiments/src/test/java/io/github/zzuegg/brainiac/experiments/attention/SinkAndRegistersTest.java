package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SinkAndRegistersTest {

    @Test
    void sinkLogitsRegistersAndTalkingHeadsAtLeastMatchSoftmax() {
        List<AttentionWeightings.Row> rows = AttentionWeightings.run(List.of("softmax", "sink logit", "registers 2",
                "talking heads"), 1, 250, 1);
        System.out.print(AttentionWeightings.table(rows));
        AttentionWeightings.Row softmax = rows.get(0);
        AttentionWeightings.Row sink = rows.get(1);
        AttentionWeightings.Row registers = rows.get(2);
        AttentionWeightings.Row talking = rows.get(3);
        System.out.printf("val loss: sink logit %.4f (first-token share %.3f), registers %.4f (%.3f), "
                + "talking heads %.4f vs softmax %.4f (%.3f)%n", sink.validationLoss(), sink.sinkShare(),
                registers.validationLoss(), registers.sinkShare(), talking.validationLoss(),
                softmax.validationLoss(), softmax.sinkShare());
        assertTrue(sink.sinkShare() < softmax.sinkShare());
        assertTrue(registers.sinkShare() < softmax.sinkShare());
        assertTrue(sink.mass() < 1);
        assertTrue(sink.validationLoss() < softmax.validationLoss() + 0.1);
        assertTrue(registers.validationLoss() < softmax.validationLoss() + 0.1);
        assertTrue(talking.validationLoss() < softmax.validationLoss() + 0.1);
    }
}
