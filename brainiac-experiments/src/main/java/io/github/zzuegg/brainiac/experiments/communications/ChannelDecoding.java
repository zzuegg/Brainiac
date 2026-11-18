package io.github.zzuegg.brainiac.experiments.communications;

import io.github.zzuegg.brainiac.experimental.comms.ChannelDecoders;
import io.github.zzuegg.brainiac.experimental.comms.TransformerDecoders;
import io.github.zzuegg.brainiac.experimental.data.ChannelCodes;
import io.github.zzuegg.brainiac.experimental.data.ChannelCodes.LinearCode;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Channel decoding over BPSK/AWGN (wave 12, items 12.4.7-12.4.8).
 *
 * <ul>
 *   <li>{@link #blockCodes}: bit and frame error rates against {@code Eb/N0} for syndrome
 *       decoding, BP, min-sum, normalised min-sum, neural BP / neural min-sum (trained on
 *       all-zero words at 1-6 dB) and, for small codes, exhaustive ML; plus uncoded BPSK.</li>
 *   <li>{@link #convolutional}: Viterbi decoding of the (7, 5) convolutional code against
 *       uncoded BPSK (BER), the coding gain.</li>
 *   <li>{@link #transformers}: ECCT and CrossMPT against BP and neural BP, in the papers' unit
 *       {@code -ln BER}, with parameter counts.</li>
 * </ul>
 */
public final class ChannelDecoding {

    private ChannelDecoding() {
    }

    /**
     * Simulates block decoders.
     *
     * @param code the code
     * @param methods the decoders (neural ones are trained first)
     * @param snrs {@code Eb/N0} values in dB
     * @param frames frames per SNR
     * @param iterations message-passing iterations
     * @param seed the seed
     * @return method to {@code {BER per SNR..., FER per SNR...}}, plus "uncoded BPSK" BERs
     */
    public static Map<String, double[]> blockCodes(LinearCode code, List<ChannelDecoders.Method> methods, double[] snrs,
            int frames, int iterations, long seed) {
        ChannelDecoders dec = new ChannelDecoders(code, iterations);
        for (ChannelDecoders.Method m : methods) {
            if (m == ChannelDecoders.Method.NEURAL_BP || m == ChannelDecoders.Method.NEURAL_MIN_SUM) {
                dec.train(m, 150, 64, 1, 6, 0.01f, seed);
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        for (ChannelDecoders.Method m : methods) {
            out.put(m.name(), new double[2 * snrs.length]);
        }
        double[] uncoded = new double[snrs.length];
        SplittableRandom rnd = new SplittableRandom(seed + 1);
        for (int s = 0; s < snrs.length; s++) {
            double sigma = ChannelCodes.sigma(snrs[s], code.rate());
            double sigmaUncoded = ChannelCodes.sigma(snrs[s], 1);
            for (int f = 0; f < frames; f++) {
                int[] msg = ChannelCodes.randomBits(code.k(), rnd);
                int[] cw = code.encode(msg);
                double[] llr = ChannelCodes.llr(ChannelCodes.awgn(ChannelCodes.bpsk(cw), sigma, rnd), sigma);
                for (ChannelDecoders.Method m : methods) {
                    int e = ChannelCodes.errors(dec.decode(m, llr), cw);
                    double[] row = out.get(m.name());
                    row[s] += e / (double) (code.n() * frames);
                    row[snrs.length + s] += e > 0 ? 1.0 / frames : 0;
                }
                double[] y = ChannelCodes.awgn(ChannelCodes.bpsk(msg), sigmaUncoded, rnd);
                for (int i = 0; i < msg.length; i++) {
                    uncoded[s] += ((y[i] < 0 ? 1 : 0) != msg[i] ? 1.0 : 0.0) / (code.k() * frames);
                }
            }
        }
        out.put("uncoded BPSK (BER only)", uncoded);
        return out;
    }

    /**
     * The (7, 5) convolutional code with Viterbi decoding against uncoded BPSK.
     *
     * @param snrs {@code Eb/N0} values in dB
     * @param bits message bits per SNR
     * @param seed the seed
     * @return {@code {Viterbi BER per SNR..., uncoded BER per SNR...}}
     */
    public static double[] convolutional(double[] snrs, int bits, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double[] out = new double[2 * snrs.length];
        int block = 200;
        for (int s = 0; s < snrs.length; s++) {
            double rate = block / (2.0 * (block + 2));
            double sigma = ChannelCodes.sigma(snrs[s], rate);
            double sigmaUncoded = ChannelCodes.sigma(snrs[s], 1);
            for (int done = 0; done < bits; done += block) {
                int[] msg = ChannelCodes.randomBits(block, rnd);
                double[] llr = ChannelCodes.llr(
                        ChannelCodes.awgn(ChannelCodes.bpsk(ChannelCodes.convolutionalEncode(msg)), sigma, rnd), sigma);
                out[s] += ChannelCodes.errors(ChannelDecoders.viterbi(llr), msg) / (double) bits;
                double[] y = ChannelCodes.awgn(ChannelCodes.bpsk(msg), sigmaUncoded, rnd);
                for (int i = 0; i < block; i++) {
                    out[snrs.length + s] += ((y[i] < 0 ? 1 : 0) != msg[i] ? 1.0 : 0.0) / bits;
                }
            }
        }
        return out;
    }

    /**
     * Learned transformer decoders against (neural) BP.
     *
     * @param code the code
     * @param snrs test {@code Eb/N0} values
     * @param frames test frames per SNR
     * @param steps training steps per transformer
     * @param seed the seed
     * @return method to {@code {-ln BER per SNR..., parameter count}}
     */
    public static Map<String, double[]> transformers(LinearCode code, double[] snrs, int frames, int steps, long seed) {
        ChannelDecoders dec = new ChannelDecoders(code, 5);
        dec.train(ChannelDecoders.Method.NEURAL_BP, 150, 64, 1, 6, 0.01f, seed);
        TransformerDecoders ecct = new TransformerDecoders(TransformerDecoders.Kind.ECCT, code, 32, 2, seed);
        ecct.train(steps, 32, 3, 6, 3e-3f, seed);
        TransformerDecoders cross = new TransformerDecoders(TransformerDecoders.Kind.CROSS_MPT, code, 32, 2, seed);
        cross.train(steps, 32, 3, 6, 3e-3f, seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        String[] names = {"BP (5 iterations)", "neural BP (5 iterations)", "ECCT (d 32, 2 layers)",
            "CrossMPT (d 32, 2 layers)"};
        for (String n : names) {
            out.put(n, new double[snrs.length + 1]);
        }
        out.get(names[0])[snrs.length] = 0;
        out.get(names[1])[snrs.length] = 5.0 * (dec.edges() + code.n());
        out.get(names[2])[snrs.length] = ecct.parameterCount();
        out.get(names[3])[snrs.length] = cross.parameterCount();
        SplittableRandom rnd = new SplittableRandom(seed + 7);
        for (int s = 0; s < snrs.length; s++) {
            double sigma = ChannelCodes.sigma(snrs[s], code.rate());
            double[][] y = new double[frames][];
            int[][] cws = new int[frames][];
            long[] errors = new long[names.length];
            for (int f = 0; f < frames; f++) {
                cws[f] = code.encode(ChannelCodes.randomBits(code.k(), rnd));
                y[f] = ChannelCodes.awgn(ChannelCodes.bpsk(cws[f]), sigma, rnd);
                double[] llr = ChannelCodes.llr(y[f], sigma);
                errors[0] += ChannelCodes.errors(dec.decode(ChannelDecoders.Method.BP, llr), cws[f]);
                errors[1] += ChannelCodes.errors(dec.decode(ChannelDecoders.Method.NEURAL_BP, llr), cws[f]);
            }
            int[][] e1 = ecct.decode(y);
            int[][] e2 = cross.decode(y);
            for (int f = 0; f < frames; f++) {
                errors[2] += ChannelCodes.errors(e1[f], cws[f]);
                errors[3] += ChannelCodes.errors(e2[f], cws[f]);
            }
            for (int m = 0; m < names.length; m++) {
                double ber = Math.max(errors[m], 0.5) / ((double) frames * code.n());
                out.get(names[m])[s] = -Math.log(ber);
            }
        }
        return out;
    }
}
