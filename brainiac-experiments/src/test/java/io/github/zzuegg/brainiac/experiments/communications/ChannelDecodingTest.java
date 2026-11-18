package io.github.zzuegg.brainiac.experiments.communications;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.comms.ChannelDecoders.Method;
import io.github.zzuegg.brainiac.experimental.data.ChannelCodes;
import io.github.zzuegg.brainiac.experimental.data.ChannelCodes.LinearCode;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

class ChannelDecodingTest {

    static String line(Map<String, double[]> r) {
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> {
            b.append(k).append(" {");
            for (int i = 0; i < v.length; i++) {
                b.append(String.format(i == 0 ? "%.4g" : ", %.4g", v[i]));
            }
            b.append("} | ");
        });
        return b.toString();
    }

    @Test
    void codesAreConsistent() {
        for (LinearCode c : List.of(ChannelCodes.hamming74(), ChannelCodes.bch31x16(), ChannelCodes.bch63x45(),
                ChannelCodes.regularLdpc(96, 3, 6, 1))) {
            for (int[] row : c.g()) {
                for (int s : c.syndrome(row)) {
                    assertEquals(0, s, c.name());
                }
            }
        }
        assertEquals(16, ChannelCodes.bch31x16().k());
        assertEquals(45, ChannelCodes.bch63x45().k());
        assertEquals(7, ChannelCodes.minimumDistance(ChannelCodes.bch31x16()));
    }

    @Test
    void beliefPropagationAndNeuralBpOnBch() {
        double[] snrs = {2, 4, 6};
        List<Method> all = List.of(Method.HARD_SYNDROME, Method.BP, Method.MIN_SUM, Method.NORMALIZED_MIN_SUM,
                Method.NEURAL_BP, Method.NEURAL_MIN_SUM, Method.MAXIMUM_LIKELIHOOD);
        Map<String, double[]> h = ChannelDecoding.blockCodes(ChannelCodes.hamming74(), all, snrs, 400, 5, 1);
        System.out.printf("12.4.7 Hamming(7,4) {BER at 2/4/6 dB, FER at 2/4/6 dB}: %s%n", line(h));
        Map<String, double[]> b = ChannelDecoding.blockCodes(ChannelCodes.bch31x16(), all, snrs, 150, 5, 2);
        System.out.printf("12.4.7 BCH(31,16) {BER at 2/4/6 dB, FER at 2/4/6 dB}: %s%n", line(b));
        assertTrue(b.get("MAXIMUM_LIKELIHOOD")[4] <= b.get("BP")[4], line(b));
        assertTrue(b.get("BP")[1] < b.get("uncoded BPSK (BER only)")[1], line(b));
        assertTrue(b.get("NEURAL_BP")[1] <= b.get("BP")[1] * 1.3, line(b));
        assertTrue(h.get("MAXIMUM_LIKELIHOOD")[4] <= h.get("HARD_SYNDROME")[4], line(h));
    }

    @Test
    void ldpcAndConvolutionalCodes() {
        double[] snrs = {1, 2, 3};
        Map<String, double[]> l = ChannelDecoding.blockCodes(ChannelCodes.regularLdpc(96, 3, 6, 3),
                List.of(Method.BP, Method.MIN_SUM, Method.NORMALIZED_MIN_SUM), snrs, 200, 20, 3);
        System.out.printf("12.4.7 LDPC(96,48) {BER at 1/2/3 dB, FER at 1/2/3 dB}: %s%n", line(l));
        Map<String, double[]> b63 = ChannelDecoding.blockCodes(ChannelCodes.bch63x45(),
                List.of(Method.HARD_SYNDROME, Method.BP, Method.NORMALIZED_MIN_SUM), new double[] {4, 6}, 150, 5, 4);
        System.out.printf("12.4.7 BCH(63,45) {BER at 4/6 dB, FER at 4/6 dB}: %s%n", line(b63));
        double[] v = ChannelDecoding.convolutional(new double[] {2, 4, 6}, 20000, 5);
        System.out.printf("12.4.7 (7,5) convolutional BER at 2/4/6 dB: Viterbi %.2e %.2e %.2e, uncoded %.2e %.2e %.2e%n",
                v[0], v[1], v[2], v[3], v[4], v[5]);
        assertTrue(l.get("BP")[2] < l.get("uncoded BPSK (BER only)")[2], line(l));
        assertTrue(v[1] < v[4] / 3, "Viterbi " + v[1] + " uncoded " + v[4]);
    }

    @Test
    void transformerDecodersAgainstBp() {
        Map<String, double[]> t = ChannelDecoding.transformers(ChannelCodes.hamming74(), new double[] {4, 5, 6}, 1000,
                100, 1);
        System.out.printf("12.4.8 Hamming(7,4) {-ln BER at 4/5/6 dB, parameters}: %s%n", line(t));
        assertTrue(t.get("ECCT (d 32, 2 layers)")[2] > 2, line(t));
        assertTrue(t.get("CrossMPT (d 32, 2 layers)")[2] > 2, line(t));
    }
}
