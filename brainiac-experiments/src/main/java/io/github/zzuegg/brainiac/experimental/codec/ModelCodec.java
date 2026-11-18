package io.github.zzuegg.brainiac.experimental.codec;

import io.github.zzuegg.brainiac.experimental.data.ByteNgramModel;
import io.github.zzuegg.brainiac.experimental.llm.KneserNeyLm;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Lossless compression with any next-symbol model: the model's distribution at every position
 * is quantised to a {@link FrequencyTable} and the symbol is coded with {@link ArithmeticCoder}
 * or {@link Rans}. The decoder runs a fresh copy of the same model on the symbols it has decoded
 * so far, so adaptive models need no side information; static models (a trained n-gram or
 * neural LM) must be shipped separately, which is why "language modeling is compression"
 * (Delétang et al., ICLR 2024, <a href="https://arxiv.org/abs/2309.10668">arXiv:2309.10668</a>)
 * reports code lengths with and without the model's size.
 *
 * <p>Adapters: {@link ContextModel} (adaptive, order k), {@link #ngram} for
 * {@link ByteNgramModel}, {@link #kneserNey} for {@link KneserNeyLm} and {@link #window} for
 * {@link WindowLm}; {@link #ofPrefix} wraps any function of the recent history.
 */
public final class ModelCodec {

    /** Which entropy coder to use. */
    public enum Coder {
        /** {@link ArithmeticCoder}, streaming in order. */
        ARITHMETIC,
        /** {@link Rans}, pushed in reverse and popped in order. */
        RANS
    }

    /**
     * A sequential next-symbol model: {@link #next()} gives the distribution of the coming
     * symbol, {@link #update} tells it the actual symbol.
     */
    public interface Predictor {

        /** @return the alphabet size */
        int alphabet();

        /** @return the distribution of the next symbol */
        double[] next();

        /**
         * Observes the symbol that came.
         *
         * @param symbol the symbol
         */
        void update(int symbol);
    }

    private final Supplier<Predictor> model;
    private final Coder coder;
    private final int precision;

    private ModelCodec(Supplier<Predictor> model, Coder coder, int precision) {
        this.model = Objects.requireNonNull(model);
        this.coder = Objects.requireNonNull(coder);
        this.precision = precision;
    }

    /**
     * A codec.
     *
     * @param model makes a fresh predictor (one for the encoder, one for the decoder)
     * @param coder the entropy coder
     * @param precision frequency bits, at most 16 (rANS) or 24 (arithmetic)
     * @return the codec
     */
    public static ModelCodec of(Supplier<Predictor> model, Coder coder, int precision) {
        return new ModelCodec(model, coder, precision);
    }

    /**
     * Compresses a sequence.
     *
     * @param symbols the sequence
     * @return the code
     */
    public byte[] encode(int[] symbols) {
        Predictor p = model.get();
        if (coder == Coder.ARITHMETIC) {
            ArithmeticCoder.Encoder enc = new ArithmeticCoder.Encoder();
            for (int s : symbols) {
                enc.encode(FrequencyTable.of(p.next(), precision), s);
                p.update(s);
            }
            return enc.finish();
        }
        int[] start = new int[symbols.length];
        int[] freq = new int[symbols.length];
        for (int i = 0; i < symbols.length; i++) {
            FrequencyTable t = FrequencyTable.of(p.next(), precision);
            start[i] = t.start(symbols[i]);
            freq[i] = t.frequency(symbols[i]);
            p.update(symbols[i]);
        }
        Rans r = new Rans();
        for (int i = symbols.length - 1; i >= 0; i--) {
            r.push(start[i], freq[i], precision);
        }
        return r.toBytes();
    }

    /**
     * Decompresses.
     *
     * @param code the code
     * @param count the number of symbols
     * @return the sequence
     */
    public int[] decode(byte[] code, int count) {
        Predictor p = model.get();
        int[] out = new int[count];
        if (coder == Coder.ARITHMETIC) {
            ArithmeticCoder.Decoder dec = new ArithmeticCoder.Decoder(code);
            for (int i = 0; i < count; i++) {
                out[i] = dec.decode(FrequencyTable.of(p.next(), precision));
                p.update(out[i]);
            }
            return out;
        }
        Rans r = Rans.fromBytes(code);
        for (int i = 0; i < count; i++) {
            out[i] = r.pop(FrequencyTable.of(p.next(), precision));
            p.update(out[i]);
        }
        return out;
    }

    /**
     * The model's log loss on a sequence, the ideal code length.
     *
     * @param model the model
     * @param symbols the sequence
     * @return bits
     */
    public static double logLossBits(Predictor model, int[] symbols) {
        double bits = 0;
        for (int s : symbols) {
            double[] d = model.next();
            double sum = 0;
            for (double v : d) {
                sum += v;
            }
            bits -= Math.log(Math.max(1e-300, d[s] / sum)) / Math.log(2);
            model.update(s);
        }
        return bits;
    }

    /**
     * Wraps a function of the most recent symbols.
     *
     * @param alphabet the alphabet size
     * @param context how many recent symbols the function sees (fewer at the start)
     * @param distribution the next-symbol distribution given the recent symbols
     * @return a predictor
     */
    public static Predictor ofPrefix(int alphabet, int context, Function<int[], double[]> distribution) {
        return new Predictor() {
            private int[] history = new int[16];
            private int length;

            @Override
            public int alphabet() {
                return alphabet;
            }

            @Override
            public double[] next() {
                int from = Math.max(0, length - context);
                return distribution.apply(Arrays.copyOfRange(history, from, length));
            }

            @Override
            public void update(int symbol) {
                if (length == history.length) {
                    history = Arrays.copyOf(history, length * 2);
                }
                history[length++] = symbol;
            }
        };
    }

    /**
     * A static byte n-gram model as a predictor over bytes.
     *
     * @param m the trained model
     * @return a predictor
     */
    public static Predictor ngram(ByteNgramModel m) {
        int k = m.order();
        return ofPrefix(256, k, ctx -> {
            byte[] b = new byte[ctx.length + 1];
            for (int i = 0; i < ctx.length; i++) {
                b[i] = (byte) ctx[i];
            }
            return m.distribution(b, ctx.length);
        });
    }

    /**
     * A Kneser–Ney model as a predictor.
     *
     * @param m the trained model
     * @return a predictor
     */
    public static Predictor kneserNey(KneserNeyLm m) {
        return ofPrefix(m.vocabSize(), Math.max(0, m.order() - 1), m::distribution);
    }

    /**
     * A window neural LM as a predictor (softmax of its logits).
     *
     * @param m the trained model
     * @return a predictor
     */
    public static Predictor window(WindowLm m) {
        return ofPrefix(m.vocabSize(), m.window(), ctx -> {
            float[] logits = m.logits(ctx);
            double max = Double.NEGATIVE_INFINITY;
            for (float v : logits) {
                max = Math.max(max, v);
            }
            double[] p = new double[logits.length];
            for (int i = 0; i < p.length; i++) {
                p[i] = Math.exp(logits[i] - max);
            }
            return p;
        });
    }

    /**
     * An adaptive order-k context model with Witten–Bell blending: the estimate of order
     * {@code k} is {@code (c(s) + u P_k-1(s)) / (n + u)}, where {@code n} counts the symbols seen
     * after this context and {@code u} how many distinct ones; order -1 is uniform. Counts start
     * empty and grow as symbols are coded, so encoder and decoder stay in step without any
     * side information.
     */
    public static final class ContextModel implements Predictor {
        private final int alphabet;
        private final int order;
        private final java.util.List<java.util.HashMap<Long, int[]>> counts = new java.util.ArrayList<>();
        private final int[] history;
        private int seen;

        /**
         * Creates an empty model.
         *
         * @param alphabet the alphabet size
         * @param order the longest context
         */
        public ContextModel(int alphabet, int order) {
            this.alphabet = alphabet;
            this.order = order;
            for (int k = 0; k <= order; k++) {
                counts.add(new java.util.HashMap<>());
            }
            this.history = new int[Math.max(1, order)];
        }

        private long key(int k) {
            long h = k;
            for (int i = 0; i < k; i++) {
                h = h * 1_000_003L + history[(seen - 1 - i) % history.length] + 1;
            }
            return h;
        }

        @Override
        public int alphabet() {
            return alphabet;
        }

        @Override
        public double[] next() {
            double[] p = new double[alphabet];
            Arrays.fill(p, 1.0 / alphabet);
            for (int k = 0; k <= Math.min(order, seen); k++) {
                int[] c = counts.get(k).get(key(k));
                if (c == null) {
                    continue;
                }
                int n = c[alphabet];
                int u = c[alphabet + 1];
                for (int s = 0; s < alphabet; s++) {
                    p[s] = (c[s] + u * p[s]) / (n + u);
                }
            }
            return p;
        }

        @Override
        public void update(int symbol) {
            for (int k = 0; k <= Math.min(order, seen); k++) {
                int[] c = counts.get(k).computeIfAbsent(key(k), x -> new int[alphabet + 2]);
                if (c[symbol] == 0) {
                    c[alphabet + 1]++;
                }
                c[symbol]++;
                c[alphabet]++;
            }
            history[seen % history.length] = symbol;
            seen++;
        }
    }
}
