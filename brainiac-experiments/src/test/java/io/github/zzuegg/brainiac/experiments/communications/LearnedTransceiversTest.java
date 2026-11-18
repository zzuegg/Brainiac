package io.github.zzuegg.brainiac.experiments.communications;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class LearnedTransceiversTest {

    @Test
    void autoencodersMatchClassicalSchemes() {
        double[] snrs = {2, 4, 6};
        Map<String, double[]> a = LearnedTransceivers.autoencoder(snrs, 6000, 400, 2);
        System.out.printf("12.4.9 AWGN block error rate at 2/4/6 dB: %s%n", ChannelDecodingTest.line(a));
        double[] f = LearnedTransceivers.fading(new double[] {10, 15}, 6000, 300, 3);
        System.out.printf("12.4.9 Rayleigh (4,2) BLER at 10/15 dB: AE %.4f %.4f, QPSK %.4f %.4f%n", f[0], f[1], f[2],
                f[3]);
        assertTrue(a.get("AE (16,7)")[2] < a.get("Hamming(7,4) hard")[2] * 1.5, ChannelDecodingTest.line(a));
        assertTrue(a.get("AE (4,2)")[1] < a.get("QPSK")[1] * 1.5, ChannelDecodingTest.line(a));
        assertTrue(f[1] < f[3], "fading AE " + f[1] + " QPSK " + f[3]);
    }

    @Test
    void mimoDetectors() {
        double[] snrs = {5, 10, 15};
        Map<String, double[]> r = LearnedTransceivers.mimo(8, new double[] {0, 0.7}, snrs, 250, 500, 4);
        System.out.printf("12.4.10 MIMO 8x8 real (4x4 QPSK) SER at 5/10/15 dB: %s%n", ChannelDecodingTest.line(r));
        assertTrue(r.get("rho 0.0: SPHERE_DECODER")[2] <= r.get("rho 0.0: MMSE")[2], ChannelDecodingTest.line(r));
        assertTrue(r.get("rho 0.0: MMSE")[2] <= r.get("rho 0.0: ZERO_FORCING")[2] + 0.01, ChannelDecodingTest.line(r));
        assertTrue(r.get("rho 0.0: DETNET")[2] < 0.2, ChannelDecodingTest.line(r));
    }
}
