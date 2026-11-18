package io.github.zzuegg.brainiac.backend.vulkan;

/**
 * A set of nonzero device addresses for the hazard tracking between barriers: an open-addressing
 * table of primitive longs whose clear costs only the entries added, so the barrier recorded
 * before nearly every decode dispatch doesn't sweep a table or box addresses.
 */
final class AddressSet {

    private long[] keys = new long[64];
    private int[] used = new int[32];
    private int size;

    boolean contains(long address) {
        if (address == 0) {
            return false;
        }
        int mask = keys.length - 1;
        for (int i = hash(address) & mask; ; i = (i + 1) & mask) {
            long k = keys[i];
            if (k == address) {
                return true;
            }
            if (k == 0) {
                return false;
            }
        }
    }

    void add(long address) {
        if (address == 0) {
            return;
        }
        if (2 * (size + 1) > keys.length) {
            grow();
        }
        int mask = keys.length - 1;
        for (int i = hash(address) & mask; ; i = (i + 1) & mask) {
            long k = keys[i];
            if (k == address) {
                return;
            }
            if (k == 0) {
                keys[i] = address;
                if (size == used.length) {
                    used = java.util.Arrays.copyOf(used, 2 * size);
                }
                used[size++] = i;
                return;
            }
        }
    }

    void clear() {
        for (int j = 0; j < size; j++) {
            keys[used[j]] = 0;
        }
        size = 0;
    }

    private void grow() {
        long[] old = keys;
        int count = size;
        int[] slots = used.clone();
        keys = new long[2 * old.length];
        size = 0;
        for (int j = 0; j < count; j++) {
            add(old[slots[j]]);
        }
    }

    private static int hash(long address) {
        long h = address * 0x9E3779B97F4A7C15L;
        return (int) (h >>> 32);
    }
}
