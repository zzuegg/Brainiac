package io.github.zzuegg.brainiac.experiments.coding;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.codec.ArithmeticCoder;
import io.github.zzuegg.brainiac.experimental.codec.BitsBackCoder;
import io.github.zzuegg.brainiac.experimental.codec.FrequencyTable;
import io.github.zzuegg.brainiac.experimental.codec.ModelCodec;
import io.github.zzuegg.brainiac.experimental.compress.HuffmanCoder;
import io.github.zzuegg.brainiac.experimental.data.ByteNgramModel;
import io.github.zzuegg.brainiac.experimental.data.KeyStreams;
import io.github.zzuegg.brainiac.experimental.data.SyntheticCorpus;
import io.github.zzuegg.brainiac.experimental.llm.KneserNeyLm;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;
import java.util.zip.DataFormatException;
import java.util.zip.Deflater;
import java.util.zip.Inflater;

/**
 * Wave 12 items 12.2.14 and 12.2.16: models as compressors.
 *
 * <ul>
 *   <li>{@link #lossless}: {@link ModelCodec} with adaptive order-0 to order-3 context models
 *       (arithmetic coding, and rANS for one order), a static {@link ByteNgramModel}, a
 *       {@link KneserNeyLm} and a small neural {@link WindowLm} against {@link HuffmanCoder},
 *       Deflate at level 9 and the empirical order-0 entropy, on synthetic text, a Zipf symbol
 *       stream and the quantised weights of a trained MLP. Every coded stream is decoded and
 *       compared, and code lengths are reported without and with the model's description (for
 *       static models trained on separate text: the Deflate size of that text, or 16 bits per
 *       neural weight).</li>
 *   <li>{@link #bitsBack}: BB-ANS ({@link BitsBackCoder}) on a chain of binarised 8×8 digits
 *       against the VAE's negative ELBO (the target), an independent per-pixel Bernoulli model
 *       coded arithmetically and Deflate.</li>
 * </ul>
 */
public final class EntropyCoding {

    private EntropyCoding() {
    }

    /**
     * One coder on one data set.
     *
     * @param data the data set
     * @param method the coder
     * @param bitsPerSymbol the code length per symbol
     * @param withModel the code length per symbol including the model's description
     * @param logLoss the model's log loss per symbol ({@code NaN} for non-model coders)
     * @param roundTrip whether decoding gave back the input exactly
     * @param symbolsPerSecond encode plus decode throughput
     */
    public record CodeRow(String data, String method, double bitsPerSymbol, double withModel, double logLoss,
            boolean roundTrip, double symbolsPerSecond) {
    }

    // ------------------------------------------------------------------ data

    private static final String[] SYLLABLES = {"ka", "lo", "mi", "ne", "ru", "sa", "ti", "vo", "pe", "da", "gu",
        "ha", "zi", "bo", "fe", "ju"};

    private static String word(int token) {
        SplittableRandom r = new SplittableRandom(token * 7919L + 13);
        int n = 1 + r.nextInt(3);
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < n; i++) {
            b.append(SYLLABLES[r.nextInt(SYLLABLES.length)]);
        }
        return b.toString();
    }

    /**
     * Synthetic text: documents of a {@link SyntheticCorpus} drawn with replacement, every token
     * written as a pseudo-word, words separated by spaces and documents by full stops.
     *
     * @param bytes the length
     * @param seed the seed of the document draws
     * @return ASCII bytes
     */
    public static byte[] text(int bytes, long seed) {
        SyntheticCorpus corpus = SyntheticCorpus.generate(20, 8, 6, 0.5, 5);
        SplittableRandom r = new SplittableRandom(seed);
        StringBuilder b = new StringBuilder();
        while (b.length() < bytes) {
            int[] doc = corpus.document(r.nextInt(corpus.size()));
            for (int i = 0; i < doc.length; i++) {
                if (doc[i] == SyntheticCorpus.PAD) {
                    continue;
                }
                b.append(word(doc[i])).append(i == doc.length - 1 ? ". " : " ");
            }
        }
        return Arrays.copyOf(b.toString().getBytes(java.nio.charset.StandardCharsets.US_ASCII), bytes);
    }

    private static int[] unsigned(byte[] b) {
        int[] s = new int[b.length];
        for (int i = 0; i < b.length; i++) {
            s[i] = b[i] & 0xff;
        }
        return s;
    }

    /** The weights of an MLP trained briefly on digits, quantised uniformly to 8 bits. */
    static int[] quantisedWeights(long seed) {
        ToyTasks.Data d = ToyTasks.digits(10, 60, 8, 0.1, seed);
        SimpleMlp net = SimpleMlp.tanh(seed, 64, 96, 10);
        Optimizer opt = Optimizer.adam(3e-3f);
        Tensor x = d.inputs();
        float[] ids = new float[d.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = d.y()[i];
        }
        Tensor y = Tensor.of(Shape.of(ids.length), ids);
        for (int s = 0; s < 200; s++) {
            Functional.crossEntropy(net.forward(Var.of(x)), y).backward();
            opt.step(net.parameters());
        }
        List<Integer> out = new ArrayList<>();
        for (Var p : net.parameters()) {
            float[] w = p.value().toArray();
            float lo = Float.POSITIVE_INFINITY;
            float hi = Float.NEGATIVE_INFINITY;
            for (float v : w) {
                lo = Math.min(lo, v);
                hi = Math.max(hi, v);
            }
            for (float v : w) {
                out.add(hi > lo ? Math.round((v - lo) / (hi - lo) * 255) : 0);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    // ------------------------------------------------------------------ baselines

    static byte[] deflate(byte[] data) {
        Deflater d = new Deflater(9);
        d.setInput(data);
        d.finish();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[1 << 14];
        while (!d.finished()) {
            out.write(buf, 0, d.deflate(buf));
        }
        d.end();
        return out.toByteArray();
    }

    static byte[] inflate(byte[] data, int length) {
        Inflater i = new Inflater();
        i.setInput(data);
        byte[] out = new byte[length];
        try {
            int n = 0;
            while (n < length && !i.finished()) {
                n += i.inflate(out, n, length - n);
            }
        } catch (DataFormatException e) {
            throw new IllegalStateException(e);
        } finally {
            i.end();
        }
        return out;
    }

    private static byte[] bytes(int[] s) {
        byte[] b = new byte[s.length];
        for (int i = 0; i < s.length; i++) {
            b[i] = (byte) s[i];
        }
        return b;
    }

    private static double entropy(int[] s, int alphabet) {
        double[] c = new double[alphabet];
        for (int v : s) {
            c[v]++;
        }
        double h = 0;
        for (double v : c) {
            if (v > 0) {
                double p = v / s.length;
                h -= p * Math.log(p) / Math.log(2);
            }
        }
        return h;
    }

    // ------------------------------------------------------------------ 12.2.14

    /**
     * 12.2.14: lossless compression with model-driven arithmetic coding and rANS.
     *
     * @param length symbols coded per data set
     * @param seed the seed
     * @return one row per data set and coder
     */
    public static List<CodeRow> lossless(int length, long seed) {
        List<CodeRow> rows = new ArrayList<>();
        byte[] trainText = text(4 * length, seed);
        byte[] testText = text(length, seed + 1);
        Map<String, int[]> data = new LinkedHashMap<>();
        data.put("synthetic text", unsigned(testText));
        data.put("Zipf symbols (s 1.1)", KeyStreams.zipf(length, 256, 1.1, seed + 2));
        int[] w = quantisedWeights(seed + 3);
        data.put("8-bit MLP weights", Arrays.copyOf(w, Math.min(w.length, length)));
        for (Map.Entry<String, int[]> e : data.entrySet()) {
            int[] s = e.getValue();
            int n = s.length;
            String name = e.getKey();
            rows.add(new CodeRow(name, "empirical order-0 entropy", entropy(s, 256), entropy(s, 256), Double.NaN, true,
                    Double.NaN));
            long t0 = System.nanoTime();
            HuffmanCoder h = HuffmanCoder.fit(s, 256);
            long[] hc = h.encode(s);
            boolean ok = Arrays.equals(h.decode(hc, n), s);
            double rate = 2.0 * n / ((System.nanoTime() - t0) / 1e9);
            rows.add(new CodeRow(name, "Huffman (order 0)", (double) h.encodedBits(s) / n,
                    (double) (h.encodedBits(s) + h.tableBits()) / n, Double.NaN, ok, rate));
            t0 = System.nanoTime();
            byte[] raw = bytes(s);
            byte[] z = deflate(raw);
            ok = Arrays.equals(inflate(z, n), raw);
            rate = 2.0 * n / ((System.nanoTime() - t0) / 1e9);
            rows.add(new CodeRow(name, "Deflate level 9", 8.0 * z.length / n, 8.0 * z.length / n, Double.NaN, ok,
                    rate));
            for (int order = 0; order <= 3; order++) {
                int k = order;
                rows.add(model(name, "adaptive order-" + k + " context, arithmetic", s,
                        () -> new ModelCodec.ContextModel(256, k), ModelCodec.Coder.ARITHMETIC, 0));
            }
            rows.add(model(name, "adaptive order-2 context, rANS", s, () -> new ModelCodec.ContextModel(256, 2),
                    ModelCodec.Coder.RANS, 0));
            if (name.startsWith("synthetic")) {
                long modelBits = 8L * deflate(trainText).length;
                ByteNgramModel ngram = ByteNgramModel.train(trainText, 4);
                rows.add(model(name, "static byte 4-gram (trained on 4x text)", s, () -> ModelCodec.ngram(ngram),
                        ModelCodec.Coder.ARITHMETIC, modelBits));
                KneserNeyLm kn = KneserNeyLm.train(new int[][] {unsigned(trainText)}, 4, 256, 0.75);
                rows.add(model(name, "Kneser-Ney 4-gram (trained on 4x text)", s, () -> ModelCodec.kneserNey(kn),
                        ModelCodec.Coder.ARITHMETIC, modelBits));
                WindowLm lm = trainWindowLm(unsigned(trainText), seed);
                long params = 0;
                for (Var p : lm.parameters()) {
                    params += p.value().size();
                }
                rows.add(model(name, "window neural LM (6 bytes, 64 hidden)", s, () -> ModelCodec.window(lm),
                        ModelCodec.Coder.ARITHMETIC, 16 * params));
            }
        }
        return rows;
    }

    private static CodeRow model(String data, String method, int[] s, Supplier<ModelCodec.Predictor> predictor,
            ModelCodec.Coder coder, long modelBits) {
        ModelCodec codec = ModelCodec.of(predictor, coder, 16);
        long t0 = System.nanoTime();
        byte[] code = codec.encode(s);
        boolean ok = Arrays.equals(codec.decode(code, s.length), s);
        double rate = 2.0 * s.length / ((System.nanoTime() - t0) / 1e9);
        double loss = ModelCodec.logLossBits(predictor.get(), s) / s.length;
        double bits = 8.0 * code.length / s.length;
        return new CodeRow(data, method, bits, bits + (double) modelBits / s.length, loss, ok, rate);
    }

    private static WindowLm trainWindowLm(int[] text, long seed) {
        int window = 6;
        WindowLm lm = WindowLm.of(256, window, 64, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int step = 0; step < 400; step++) {
            List<int[]> prompts = new ArrayList<>();
            List<int[]> completions = new ArrayList<>();
            for (int b = 0; b < 64; b++) {
                int t = window + r.nextInt(text.length - window);
                prompts.add(Arrays.copyOfRange(text, t - window, t));
                completions.add(new int[] {text[t]});
            }
            lm.supervisedStep(prompts, completions, opt);
        }
        return lm;
    }

    // ------------------------------------------------------------------ 12.2.16

    /**
     * One coder of a chain of binary images.
     *
     * @param method the coder
     * @param bitsPerImage the code length per image over the chain
     * @param firstImage the bits spent on the first image ({@code NaN} if not separable)
     * @param laterImages the bits per image over the second half of the chain
     * @param roundTrip whether decoding gave back the chain
     */
    public record ChainRow(String method, double bitsPerImage, double firstImage, double laterImages,
            boolean roundTrip) {
    }

    private static int[][] binarise(ToyTasks.Data d) {
        int[][] out = new int[d.size()][];
        for (int i = 0; i < out.length; i++) {
            out[i] = new int[d.x()[i].length];
            for (int j = 0; j < out[i].length; j++) {
                out[i][j] = d.x()[i][j] > 0.5f ? 1 : 0;
            }
        }
        return out;
    }

    /**
     * 12.2.16: bits-back coding of a chain of binarised digits.
     *
     * @param trainImages VAE training images
     * @param chain images in the coded chain
     * @param steps VAE training steps
     * @param seed the seed
     * @return negative ELBO, BB-ANS, independent Bernoulli, Deflate
     */
    public static List<ChainRow> bitsBack(int trainImages, int chain, int steps, long seed) {
        int[][] train = binarise(ToyTasks.digits(10, trainImages / 10, 8, 0.1, seed, seed + 1));
        int[][] test = Arrays.copyOf(binarise(ToyTasks.digits(10, (chain + 9) / 10, 8, 0.1, seed, seed + 2)), chain);
        float[][] tf = new float[train.length][];
        for (int i = 0; i < tf.length; i++) {
            tf[i] = new float[train[i].length];
            for (int j = 0; j < tf[i].length; j++) {
                tf[i][j] = train[i][j];
            }
        }
        float[][] testF = new float[chain][];
        for (int i = 0; i < chain; i++) {
            testF[i] = new float[test[i].length];
            for (int j = 0; j < test[i].length; j++) {
                testF[i][j] = test[i][j];
            }
        }
        List<ChainRow> rows = new ArrayList<>();
        BitsBackCoder vae = BitsBackCoder.train(tf, 8, 64, 8, steps, 64, 3e-3f, seed);
        double elbo = vae.negElboBits(testF, 16, seed + 5);
        rows.add(new ChainRow("VAE negative ELBO (target)", elbo, Double.NaN, Double.NaN, true));
        BitsBackCoder.Encoded enc = vae.encode(test, seed + 6);
        int[][] back = vae.decode(enc.message(), chain);
        boolean ok = Arrays.deepEquals(back, test);
        long[] after = enc.bitsAfter();
        int half = chain / 2;
        rows.add(new ChainRow("BB-ANS (8 latents, 8-bit buckets)", (double) enc.bits() / chain, after[0],
                (double) (after[chain - 1] - after[half - 1]) / (chain - half), ok));
        // independent Bernoulli pixels, fitted on the training images, coded arithmetically
        int pixels = train[0].length;
        FrequencyTable[] tables = new FrequencyTable[pixels];
        for (int j = 0; j < pixels; j++) {
            double c = 1;
            for (int[] img : train) {
                c += img[j];
            }
            double p = c / (train.length + 2);
            tables[j] = FrequencyTable.of(new double[] {1 - p, p}, 12);
        }
        ArithmeticCoder.Encoder ae = new ArithmeticCoder.Encoder();
        long[] bitsAfter = new long[chain];
        for (int i = 0; i < chain; i++) {
            for (int j = 0; j < pixels; j++) {
                ae.encode(tables[j], test[i][j]);
            }
            bitsAfter[i] = ae.bits();
        }
        byte[] code = ae.finish();
        ArithmeticCoder.Decoder ad = new ArithmeticCoder.Decoder(code);
        boolean same = true;
        for (int i = 0; i < chain && same; i++) {
            for (int j = 0; j < pixels; j++) {
                if (ad.decode(tables[j]) != test[i][j]) {
                    same = false;
                    break;
                }
            }
        }
        rows.add(new ChainRow("independent Bernoulli pixels, arithmetic", 8.0 * code.length / chain, bitsAfter[0],
                (double) (bitsAfter[chain - 1] - bitsAfter[half - 1]) / (chain - half), same));
        // Deflate of the packed pixels
        byte[] packed = new byte[(chain * pixels + 7) / 8];
        for (int i = 0; i < chain; i++) {
            for (int j = 0; j < pixels; j++) {
                int b = i * pixels + j;
                if (test[i][j] == 1) {
                    packed[b >> 3] |= (byte) (1 << (b & 7));
                }
            }
        }
        byte[] z = deflate(packed);
        rows.add(new ChainRow("Deflate level 9 (packed bits)", 8.0 * z.length / chain, Double.NaN, Double.NaN,
                Arrays.equals(inflate(z, packed.length), packed)));
        return rows;
    }
}
