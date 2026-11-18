package io.github.zzuegg.brainiac.experimental.codec;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Range asymmetric numeral systems (rANS; Duda, "Asymmetric numeral systems: entropy coding
 * combining speed of Huffman coding with compression rate of arithmetic coding",
 * <a href="https://arxiv.org/abs/1311.2540">arXiv:1311.2540</a>) as a <em>stack</em>: the state
 * {@code x} lives in {@code [2^23, 2^31)} and spills whole bytes onto a byte stack. {@link #push}
 * encodes a symbol and {@link #pop} decodes the most recently pushed one, so symbols come back
 * in reverse order. That last-in-first-out property is what bits-back coding needs
 * ({@link BitsBackCoder}): popping a symbol under a distribution it was never pushed with
 * "decodes" fresh information from the bits already on the stack.
 *
 * <p>Popping from an empty stack draws seeded random bytes instead ("clean bits"), counted by
 * {@link #borrowedBits()}; the message length {@link #bits()} is the stack plus the 32-bit
 * state. {@link #encode} and {@link #decode} code a whole sequence with one static table.
 */
public final class Rans {

    private static final long L = 1L << 23;

    private long state = L;
    private byte[] stack = new byte[64];
    private int size;
    private long borrowed;
    private final SplittableRandom clean;

    /** Creates an empty coder whose clean bits come from seed 0. */
    public Rans() {
        this(0);
    }

    /**
     * Creates an empty coder.
     *
     * @param cleanSeed the seed of the random bytes drawn when popping from an empty stack
     */
    public Rans(long cleanSeed) {
        this.clean = new SplittableRandom(cleanSeed);
    }

    private void emit(int b) {
        if (size == stack.length) {
            stack = Arrays.copyOf(stack, size * 2);
        }
        stack[size++] = (byte) b;
    }

    private int take() {
        if (size == 0) {
            borrowed += 8;
            return clean.nextInt(256);
        }
        return stack[--size] & 0xff;
    }

    /**
     * Encodes a symbol that occupies {@code [start, start + freq)} of {@code 2^scaleBits}.
     *
     * @param start its cumulative start
     * @param freq its frequency, positive
     * @param scaleBits the precision, at most 16
     */
    public void push(int start, int freq, int scaleBits) {
        long xMax = ((L >> scaleBits) << 8) * freq;
        while (state >= xMax) {
            emit((int) (state & 0xff));
            state >>>= 8;
        }
        state = ((state / freq) << scaleBits) + (state % freq) + start;
    }

    /**
     * Encodes a symbol with a table.
     *
     * @param table the table (at most 16 bits)
     * @param symbol the symbol
     */
    public void push(FrequencyTable table, int symbol) {
        push(table.start(symbol), table.frequency(symbol), table.bits());
    }

    /**
     * The cumulative value of the symbol on top, without removing it.
     *
     * @param scaleBits the precision it was pushed with
     * @return a value in {@code [0, 2^scaleBits)}
     */
    public int peek(int scaleBits) {
        return (int) (state & ((1L << scaleBits) - 1));
    }

    /**
     * Removes the symbol on top, identified by its interval (see {@link #peek}).
     *
     * @param start its cumulative start
     * @param freq its frequency
     * @param scaleBits the precision
     */
    public void pop(int start, int freq, int scaleBits) {
        long mask = (1L << scaleBits) - 1;
        state = freq * (state >>> scaleBits) + (state & mask) - start;
        while (state < L) {
            state = (state << 8) | take();
        }
    }

    /**
     * Decodes the symbol on top with a table.
     *
     * @param table the table
     * @return the symbol
     */
    public int pop(FrequencyTable table) {
        int s = table.symbol(peek(table.bits()));
        pop(table.start(s), table.frequency(s), table.bits());
        return s;
    }

    /** @return the message length in bits: the byte stack plus the 32-bit state */
    public long bits() {
        return 8L * size + 32;
    }

    /** @return the random bits drawn by pops from an empty stack */
    public long borrowedBits() {
        return borrowed;
    }

    /** @return the message: the stack bytes followed by the four state bytes */
    public byte[] toBytes() {
        byte[] out = Arrays.copyOf(stack, size + 4);
        for (int i = 0; i < 4; i++) {
            out[size + i] = (byte) (state >>> (8 * i));
        }
        return out;
    }

    /**
     * Restores a coder from {@link #toBytes()}.
     *
     * @param message the bytes
     * @return the coder
     */
    public static Rans fromBytes(byte[] message) {
        Rans r = new Rans();
        int n = message.length - 4;
        r.stack = Arrays.copyOf(message, Math.max(64, n));
        r.size = n;
        long s = 0;
        for (int i = 3; i >= 0; i--) {
            s = (s << 8) | (message[n + i] & 0xff);
        }
        r.state = s;
        return r;
    }

    /**
     * Static rANS: codes a sequence with one table (symbols are pushed in reverse, so that
     * decoding returns them in order).
     *
     * @param symbols the sequence
     * @param table the table (at most 16 bits)
     * @return the message
     */
    public static byte[] encode(int[] symbols, FrequencyTable table) {
        Rans r = new Rans();
        for (int i = symbols.length - 1; i >= 0; i--) {
            r.push(table, symbols[i]);
        }
        return r.toBytes();
    }

    /**
     * Decodes {@link #encode}.
     *
     * @param message the message
     * @param count the number of symbols
     * @param table the table
     * @return the sequence
     */
    public static int[] decode(byte[] message, int count, FrequencyTable table) {
        Rans r = fromBytes(message);
        int[] out = new int[count];
        for (int i = 0; i < count; i++) {
            out[i] = r.pop(table);
        }
        return out;
    }
}
