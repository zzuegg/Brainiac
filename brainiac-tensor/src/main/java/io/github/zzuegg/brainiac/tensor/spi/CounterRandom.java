package io.github.zzuegg.brainiac.tensor.spi;

/**
 * Random numbers computed from a seed and an element index alone (SplitMix64's mixing
 * function), so every element can be generated independently and in any order: on the CPU,
 * or inside a device kernel, with the same values everywhere.
 */
public final class CounterRandom {

    private CounterRandom() {
    }

    /**
     * The value of element {@code index} for {@code seed}, uniform in {@code [0, 1)} with 24 bits.
     *
     * @param seed the seed
     * @param index the element index
     * @return the value
     */
    public static float uniform(long seed, long index) {
        long z = seed + (index + 1) * 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z ^= z >>> 31;
        return (z >>> 40) * 0x1.0p-24f;
    }
}
