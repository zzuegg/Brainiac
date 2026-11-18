package io.github.zzuegg.brainiac.experimental.codec;

import java.util.Arrays;

/**
 * A 32-bit binary arithmetic coder in the style of Witten, Neal and Cleary ("Arithmetic coding
 * for data compression", CACM 1987): the interval {@code [low, high]} narrows to each symbol's
 * share of the frequency total; leading bits that agree are emitted, and the near-half
 * underflow case (E3) is deferred as pending bits. Coding a sequence costs its log loss under the
 * model plus at most two bits, so any probabilistic model becomes a compressor ("Language
 * modeling is compression", Delétang et al., ICLR 2024,
 * <a href="https://arxiv.org/abs/2309.10668">arXiv:2309.10668</a>).
 *
 * <p>Use {@link Encoder} and {@link Decoder} with the same sequence of {@link FrequencyTable}s
 * (frequency totals up to {@code 2^24}); the decoder may build each table from the symbols it
 * has already decoded, which is how adaptive models stay in sync.
 */
public final class ArithmeticCoder {

    private static final long TOP = 0xFFFFFFFFL;
    private static final long HALF = 0x80000000L;
    private static final long QUARTER = 0x40000000L;

    private ArithmeticCoder() {
    }

    /** The encoding side: feed symbols, then {@link #finish()}. */
    public static final class Encoder {
        private long low;
        private long high = TOP;
        private int pending;
        private byte[] out = new byte[64];
        private long bitCount;

        /** Creates an encoder with an empty output. */
        public Encoder() {
        }

        private void bit(int b) {
            int byteIndex = (int) (bitCount >>> 3);
            if (byteIndex >= out.length) {
                out = Arrays.copyOf(out, out.length * 2);
            }
            if (b != 0) {
                out[byteIndex] |= (byte) (0x80 >>> (bitCount & 7));
            }
            bitCount++;
        }

        private void bitPlusPending(int b) {
            bit(b);
            for (; pending > 0; pending--) {
                bit(1 - b);
            }
        }

        /**
         * Encodes a symbol given its cumulative interval.
         *
         * @param cumLow the summed frequency before the symbol
         * @param cumHigh {@code cumLow} plus the symbol's frequency
         * @param total the frequency total, at most {@code 2^24}
         */
        public void encode(int cumLow, int cumHigh, int total) {
            if (cumLow < 0 || cumHigh <= cumLow || cumHigh > total || total > (1 << 24)) {
                throw new IllegalArgumentException("bad interval [" + cumLow + ", " + cumHigh + ") of " + total);
            }
            long range = high - low + 1;
            high = low + range * cumHigh / total - 1;
            low = low + range * cumLow / total;
            while (true) {
                if (high < HALF) {
                    bitPlusPending(0);
                } else if (low >= HALF) {
                    bitPlusPending(1);
                    low -= HALF;
                    high -= HALF;
                } else if (low >= QUARTER && high < HALF + QUARTER) {
                    pending++;
                    low -= QUARTER;
                    high -= QUARTER;
                } else {
                    break;
                }
                low <<= 1;
                high = (high << 1) | 1;
            }
        }

        /**
         * Encodes a symbol with a table.
         *
         * @param table the table
         * @param symbol the symbol
         */
        public void encode(FrequencyTable table, int symbol) {
            encode(table.start(symbol), table.start(symbol) + table.frequency(symbol), table.total());
        }

        /**
         * Flushes the final bits.
         *
         * @return the code, {@code ceil(bits / 8)} bytes
         */
        public byte[] finish() {
            pending++;
            bitPlusPending(low < QUARTER ? 0 : 1);
            return Arrays.copyOf(out, (int) ((bitCount + 7) >>> 3));
        }

        /** @return bits written so far */
        public long bits() {
            return bitCount;
        }
    }

    /** The decoding side, reading a code produced by {@link Encoder}. */
    public static final class Decoder {
        private final byte[] in;
        private long position;
        private long low;
        private long high = TOP;
        private long value;

        /**
         * Starts decoding.
         *
         * @param code the code
         */
        public Decoder(byte[] code) {
            this.in = code.clone();
            for (int i = 0; i < 32; i++) {
                value = (value << 1) | nextBit();
            }
        }

        private int nextBit() {
            int byteIndex = (int) (position >>> 3);
            int b = byteIndex < in.length ? (in[byteIndex] >>> (7 - (position & 7))) & 1 : 0;
            position++;
            return b;
        }

        /**
         * The cumulative value of the next symbol; find the symbol whose interval holds it, then
         * call {@link #consume}.
         *
         * @param total the frequency total
         * @return a value in {@code [0, total)}
         */
        public int target(int total) {
            long range = high - low + 1;
            return (int) (((value - low + 1) * total - 1) / range);
        }

        /**
         * Removes a decoded symbol from the interval.
         *
         * @param cumLow the symbol's cumulative start
         * @param cumHigh its cumulative end
         * @param total the frequency total
         */
        public void consume(int cumLow, int cumHigh, int total) {
            long range = high - low + 1;
            high = low + range * cumHigh / total - 1;
            low = low + range * cumLow / total;
            while (true) {
                if (high >= HALF) {
                    if (low >= HALF) {
                        low -= HALF;
                        high -= HALF;
                        value -= HALF;
                    } else if (low >= QUARTER && high < HALF + QUARTER) {
                        low -= QUARTER;
                        high -= QUARTER;
                        value -= QUARTER;
                    } else {
                        break;
                    }
                }
                low <<= 1;
                high = (high << 1) | 1;
                value = (value << 1) | nextBit();
            }
        }

        /**
         * Decodes one symbol with a table.
         *
         * @param table the table the encoder used
         * @return the symbol
         */
        public int decode(FrequencyTable table) {
            int s = table.symbol(target(table.total()));
            consume(table.start(s), table.start(s) + table.frequency(s), table.total());
            return s;
        }
    }
}
