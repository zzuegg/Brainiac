package io.github.zzuegg.brainiac.experiments.communications;

import io.github.zzuegg.brainiac.experimental.comms.ChannelDecoders;
import io.github.zzuegg.brainiac.experimental.comms.CommunicationAutoencoder;
import io.github.zzuegg.brainiac.experimental.comms.MimoDetectors;
import io.github.zzuegg.brainiac.experimental.data.ChannelCodes;
import io.github.zzuegg.brainiac.experimental.signal.DenseMatrices;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Learned transceivers and MIMO detection (wave 12, items 12.4.9-12.4.10).
 *
 * <ul>
 *   <li>{@link #autoencoder}: block error rate of end-to-end autoencoders for {@code (M, n)} in
 *       {(4, 2), (16, 2), (16, 7)} over AWGN (the 16-point 2-D constellation under an average
 *       power constraint, the others per message) against QPSK, 16-QAM and Hamming(7,4) with BPSK
 *       (hard syndrome and soft ML decoding), and the minimum distance of the learned 16-point
 *       2-D constellation against 16-QAM's.</li>
 *   <li>{@link #fading}: (4, 2) over Rayleigh fading with coherent detection against QPSK: the
 *       autoencoder can spread each bit over both channel uses (diversity).</li>
 *   <li>{@link #mimo}: symbol error rate of ZF, MMSE, MMSE-SIC, the sphere decoder (exact ML)
 *       and DetNet on real 8 x 8 Rayleigh channels (4 x 4 complex with QPSK), i.i.d. and
 *       correlated, with the sphere decoder's mean visited nodes.</li>
 * </ul>
 */
public final class LearnedTransceivers {

    private LearnedTransceivers() {
    }

    private static double qpskBler(double ebN0, int samples, SplittableRandom rnd) {
        double sigma = ChannelCodes.sigma(ebN0, 1);
        int errors = 0;
        for (int i = 0; i < samples; i++) {
            boolean wrong = false;
            for (int u = 0; u < 2; u++) {
                wrong |= 1 + sigma * ChannelCodes.gaussian(rnd) < 0;
            }
            errors += wrong ? 1 : 0;
        }
        return errors / (double) samples;
    }

    private static double qamBler(double ebN0, int samples, SplittableRandom rnd) {
        double sigma = ChannelCodes.sigma(ebN0, 2);
        double scale = 1 / Math.sqrt(5);
        int[] levels = {-3, -1, 1, 3};
        int errors = 0;
        for (int i = 0; i < samples; i++) {
            boolean wrong = false;
            for (int u = 0; u < 2; u++) {
                int sent = levels[rnd.nextInt(4)];
                double y = sent * scale + sigma * ChannelCodes.gaussian(rnd);
                int best = levels[0];
                for (int l : levels) {
                    if (Math.abs(y - l * scale) < Math.abs(y - best * scale)) {
                        best = l;
                    }
                }
                wrong |= best != sent;
            }
            errors += wrong ? 1 : 0;
        }
        return errors / (double) samples;
    }

    private static double[] hammingBler(double ebN0, int samples, SplittableRandom rnd) {
        ChannelCodes.LinearCode code = ChannelCodes.hamming74();
        ChannelDecoders dec = new ChannelDecoders(code, 5);
        double sigma = ChannelCodes.sigma(ebN0, code.rate());
        int hard = 0;
        int soft = 0;
        for (int i = 0; i < samples; i++) {
            int[] cw = code.encode(ChannelCodes.randomBits(4, rnd));
            double[] llr = ChannelCodes.llr(ChannelCodes.awgn(ChannelCodes.bpsk(cw), sigma, rnd), sigma);
            hard += ChannelCodes.errors(dec.decode(ChannelDecoders.Method.HARD_SYNDROME, llr), cw) > 0 ? 1 : 0;
            soft += ChannelCodes.errors(dec.decode(ChannelDecoders.Method.MAXIMUM_LIKELIHOOD, llr), cw) > 0 ? 1 : 0;
        }
        return new double[] {hard / (double) samples, soft / (double) samples};
    }

    /**
     * Autoencoders over AWGN against classical modulation and coding.
     *
     * @param snrs {@code Eb/N0} values in dB
     * @param samples test messages per SNR
     * @param steps training steps per autoencoder
     * @param seed the seed
     * @return system to block error rate per SNR; key "min distance {AE 16x2, 16-QAM}" holds the
     *     two minimum distances
     */
    public static Map<String, double[]> autoencoder(double[] snrs, int samples, int steps, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        CommunicationAutoencoder ae42 = new CommunicationAutoencoder(4, 2, 32, false, seed);
        ae42.train(4, steps, 256, 0.01f, seed);
        CommunicationAutoencoder ae162 = new CommunicationAutoencoder(16, 2, 64, false, seed + 1).averagePower();
        ae162.train(10, 2 * steps, 256, 0.01f, seed + 1);
        CommunicationAutoencoder ae167 = new CommunicationAutoencoder(16, 7, 64, false, seed + 2);
        ae167.train(6, steps, 256, 0.005f, seed + 2);
        Map<String, double[]> out = new LinkedHashMap<>();
        String[] names = {"AE (4,2)", "QPSK", "AE (16,2)", "16-QAM", "AE (16,7)", "Hamming(7,4) hard",
            "Hamming(7,4) soft ML"};
        for (String n : names) {
            out.put(n, new double[snrs.length]);
        }
        for (int s = 0; s < snrs.length; s++) {
            out.get(names[0])[s] = ae42.blockErrorRate(snrs[s], samples, seed + 10);
            out.get(names[1])[s] = qpskBler(snrs[s], samples, rnd);
            out.get(names[2])[s] = ae162.blockErrorRate(snrs[s], samples, seed + 11);
            out.get(names[3])[s] = qamBler(snrs[s], samples, rnd);
            out.get(names[4])[s] = ae167.blockErrorRate(snrs[s], samples, seed + 12);
            double[] h = hammingBler(snrs[s], samples / 4, rnd);
            out.get(names[5])[s] = h[0];
            out.get(names[6])[s] = h[1];
        }
        double[][] qam = new double[16][2];
        int[] levels = {-3, -1, 1, 3};
        for (int i = 0; i < 16; i++) {
            qam[i][0] = levels[i % 4];
            qam[i][1] = levels[i / 4];
        }
        out.put("min distance {AE 16x2, 16-QAM}", new double[] {
            CommunicationAutoencoder.minimumDistance(ae162.constellation()),
            CommunicationAutoencoder.minimumDistance(qam)});
        return out;
    }

    /**
     * A (4, 2) autoencoder over Rayleigh fading against QPSK with coherent detection.
     *
     * @param snrs {@code Eb/N0} values in dB
     * @param samples test messages per SNR
     * @param steps training steps
     * @param seed the seed
     * @return {@code {AE BLER per SNR..., QPSK BLER per SNR...}}
     */
    public static double[] fading(double[] snrs, int samples, int steps, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        CommunicationAutoencoder ae = new CommunicationAutoencoder(4, 2, 64, true, seed);
        ae.train(12, steps, 256, 0.005f, seed);
        double[] out = new double[2 * snrs.length];
        for (int s = 0; s < snrs.length; s++) {
            out[s] = ae.blockErrorRate(snrs[s], samples, seed + 5);
            double sigma = ChannelCodes.sigma(snrs[s], 1);
            int errors = 0;
            for (int i = 0; i < samples; i++) {
                boolean wrong = false;
                for (int u = 0; u < 2; u++) {
                    double a = ChannelCodes.gaussian(rnd);
                    double b = ChannelCodes.gaussian(rnd);
                    double h = Math.sqrt((a * a + b * b) / 2);
                    wrong |= h + sigma * ChannelCodes.gaussian(rnd) < 0;
                }
                errors += wrong ? 1 : 0;
            }
            out[snrs.length + s] = errors / (double) samples;
        }
        return out;
    }

    /**
     * MIMO detection on real {@code nt x nt} Rayleigh channels; DetNet is trained once on
     * i.i.d. channels at 5-20 dB and tested on every correlation level.
     *
     * @param nt real antennas on each side
     * @param rhos Kronecker correlations of the test channels (0 for i.i.d.)
     * @param snrs SNRs in dB
     * @param samples test vectors per SNR
     * @param steps DetNet training steps
     * @param seed the seed
     * @return {@code "rho <r>: <detector>"} to {@code {SER per SNR...}}, plus
     *     {@code "rho <r>: sphere nodes per detection"}
     */
    public static Map<String, double[]> mimo(int nt, double[] rhos, double[] snrs, int samples, int steps, long seed) {
        MimoDetectors det = new MimoDetectors(nt, 6, seed);
        det.trainDetNet(r -> ChannelCodes.mimoChannel(nt, nt, 0, r), steps, 128, 5, 20, 3e-3f, seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        MimoDetectors.Method[] methods = MimoDetectors.Method.values();
        for (double rho : rhos) {
            String prefix = "rho " + rho + ": ";
            for (MimoDetectors.Method m : methods) {
                out.put(prefix + m.name(), new double[snrs.length]);
            }
            double[] nodes = new double[snrs.length];
            SplittableRandom rnd = new SplittableRandom(seed + 3);
            for (int s = 0; s < snrs.length; s++) {
                double sigma = det.sigma(snrs[s], nt);
                double[][][] hs = new double[samples][][];
                double[][] ys = new double[samples][];
                double[][] xs = new double[samples][nt];
                for (int i = 0; i < samples; i++) {
                    hs[i] = ChannelCodes.mimoChannel(nt, nt, rho, rnd);
                    for (int j = 0; j < nt; j++) {
                        xs[i][j] = rnd.nextBoolean() ? 1 : -1;
                    }
                    ys[i] = ChannelCodes.awgn(DenseMatrices.multiply(hs[i], xs[i]), sigma, rnd);
                }
                det.resetNodes();
                double[][] detnet = det.detnet(hs, ys);
                for (MimoDetectors.Method m : methods) {
                    int errors = 0;
                    for (int i = 0; i < samples; i++) {
                        double[] x = m == MimoDetectors.Method.DETNET ? detnet[i] : det.detect(m, hs[i], ys[i], sigma);
                        for (int j = 0; j < nt; j++) {
                            errors += x[j] != xs[i][j] ? 1 : 0;
                        }
                    }
                    out.get(prefix + m.name())[s] = errors / (double) (samples * nt);
                }
                nodes[s] = det.nodesVisited() / (double) samples;
            }
            out.put(prefix + "sphere nodes per detection", nodes);
        }
        return out;
    }
}
