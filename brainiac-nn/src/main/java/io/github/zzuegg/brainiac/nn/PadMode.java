package io.github.zzuegg.brainiac.nn;

/** How cells outside an image are filled when it is padded. */
public enum PadMode {
    /** zeros (or a chosen constant) */
    CONSTANT,
    /** mirror without repeating the edge: {@code c b | a b c | b a} */
    REFLECT,
    /** mirror repeating the edge: {@code b a | a b c | c b} */
    SYMMETRIC,
    /** repeat the edge value: {@code a a | a b c | c c} */
    REPLICATE,
    /** wrap around: {@code b c | a b c | a b} */
    CIRCULAR;

    /**
     * Maps a position outside {@code [0, length)} to the cell it reads.
     *
     * @param i the position, possibly negative or past the end
     * @param length the size of the dimension
     * @return the cell index, or -1 for {@link #CONSTANT}
     */
    int source(int i, int length) {
        if (i >= 0 && i < length) {
            return i;
        }
        return switch (this) {
            case CONSTANT -> -1;
            case REPLICATE -> Math.clamp(i, 0, length - 1);
            case CIRCULAR -> Math.floorMod(i, length);
            case REFLECT -> {
                if (length == 1) {
                    yield 0;
                }
                int period = 2 * (length - 1);
                int j = Math.floorMod(i, period);
                yield j >= length ? period - j : j;
            }
            case SYMMETRIC -> {
                int period = 2 * length;
                int j = Math.floorMod(i, period);
                yield j >= length ? period - 1 - j : j;
            }
        };
    }
}
