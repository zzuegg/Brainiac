package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.PositionalAttention.Position;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class PositionShootoutTest {

    @Test
    void relativeEncodingsSolveLaggedCopyBetterThanNoPositions() {
        Map<Position, Float> acc = new EnumMap<>(Position.class);
        for (Position p : new Position[] {Position.NONE, Position.ROTARY, Position.XPOS, Position.ALIBI, Position.T5,
            Position.COPE}) {
            acc.put(p, PositionShootout.run(p, 2, 150, 1));
        }
        System.out.println(acc);
        float nope = acc.get(Position.NONE);
        for (Position p : new Position[] {Position.ROTARY, Position.XPOS, Position.T5, Position.COPE}) {
            assertTrue(acc.get(p) > 0.9f, p + " solves the task: " + acc);
            assertTrue(acc.get(p) > nope + 0.2f, p + " beats NoPE: " + acc);
        }
        // ALiBi only decays with distance, so it cannot single out the token exactly two back
        assertTrue(acc.get(Position.ALIBI) > nope, "ALiBi beats NoPE: " + acc);
    }
}
