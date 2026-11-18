package io.github.zzuegg.brainiac.rl;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.tensor.Device;
import org.junit.jupiter.api.Test;

/** The buffer grows as it fills, overwrites the oldest, and relabels rewards. */
class ReplayBufferStorageTest {

    private static Transition t(int i) {
        return new Transition(new float[] {i, -i}, new float[] {i % 3}, i, new float[] {i + 1, -i - 1}, i % 5 == 0, 0.99f);
    }

    @Test
    void aLargeCapacityCostsNothingUntilUsed() {
        // a million pixel observations would be gigabytes if allocated up front
        ReplayBuffer big = new ReplayBuffer(1_000_000, 84 * 84 * 4, 1);
        big.add(new Transition(new float[84 * 84 * 4], new float[1], 1, new float[84 * 84 * 4], false, 1));
        assertEquals(1, big.size());
    }

    @Test
    void growsPastItsFirstBlockAndWrapsAround() {
        ReplayBuffer b = new ReplayBuffer(3000, 2, 1);
        for (int i = 0; i < 3500; i++) {
            b.add(t(i));
        }
        assertEquals(3000, b.size());
        // slot 0 was overwritten by transition 3000, slot 2999 still holds 2999
        assertArrayEquals(new float[] {3000, -3000}, b.transition(0).observation());
        assertEquals(2999f, b.transition(2999).reward());
        assertEquals(true, b.transition(0).terminated());
        ReplayBuffer.Batch batch = b.gather(new int[] {5, 1500}, new float[] {1, 1}, Device.cpu());
        assertArrayEquals(new float[] {3005, 1500}, batch.rewards().toArray());
    }

    @Test
    void relabelsRewards() {
        ReplayBuffer b = new ReplayBuffer(10, 2, 1);
        for (int i = 0; i < 4; i++) {
            b.add(t(i));
        }
        b.relabel((o, a, n) -> n[0] * 10);
        assertEquals(30f, b.transition(2).reward());
        b.setReward(1, -1);
        assertEquals(-1f, b.transition(1).reward());
    }
}
