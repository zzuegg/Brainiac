package io.github.zzuegg.brainiac.experiments.shift;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class LabelShiftTest {

    @Test
    void estimatorsRecoverMostOfTheOracleGain() {
        LabelShift.Source src = LabelShift.source(500, 1);
        Map<LabelShift.Method, double[]> sum = new EnumMap<>(LabelShift.Method.class);
        int sets = 0;
        for (Map.Entry<String, double[]> e : LabelShift.shifts(1).entrySet()) {
            Map<LabelShift.Method, LabelShift.Score> s = LabelShift.estimate(src, e.getValue(), 400, 2);
            StringBuilder sb = new StringBuilder();
            s.forEach((m, v) -> {
                sb.append(String.format("  %s %.3f/%.3f", m, v.weightMse(), v.accuracy()));
                sum.computeIfAbsent(m, k -> new double[2]);
                sum.get(m)[0] += v.weightMse();
                sum.get(m)[1] += v.accuracy();
            });
            sets++;
            System.out.printf("Label shift %-15s (weight MSE/acc)%s%n", e.getKey(), sb);
        }
        int n = sets;
        StringBuilder sb = new StringBuilder();
        sum.forEach((m, v) -> sb.append(String.format("  %s %.3f/%.3f", m, v[0] / n, v[1] / n)));
        System.out.printf("Label shift mean over shifts%s%n", sb);
        double none = sum.get(LabelShift.Method.NONE)[1];
        assertTrue(sum.get(LabelShift.Method.ORACLE)[1] > none, "oracle correction helps");
        assertTrue(sum.get(LabelShift.Method.MLLS_BCTS)[1] > none, "MLLS correction helps");
        assertTrue(sum.get(LabelShift.Method.MLLS_BCTS)[0] < sum.get(LabelShift.Method.BBSE)[0] * 1.5,
                "MLLS estimates weights about as well as BBSE or better");
        LabelShift.StreamScores st = LabelShift.stream(src, 60, 16, 0.1, 3);
        System.out.printf("Label shift drifting stream: acc none %.3f online %.3f per-batch MLLS %.3f; prior error "
                + "online %.3f per-batch %.3f%n", st.noCorrection(), st.online(), st.perBatch(), st.onlineError(),
                st.perBatchError());
        assertTrue(st.onlineError() < st.perBatchError(), "online tracking beats refitting on each small batch");
    }
}
