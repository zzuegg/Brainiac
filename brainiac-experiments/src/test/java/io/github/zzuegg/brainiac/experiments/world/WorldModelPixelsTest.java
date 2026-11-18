package io.github.zzuegg.brainiac.experiments.world;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertTrue;

class WorldModelPixelsTest {

    @Test
    void pixelCartPole() {
        List<WorldModelShootout.PixelRow> rows = WorldModelShootout.pixels(new String[] {"cartpole"}, 1_000, 1, 12);
        System.out.printf("11.6.17 cart-pole return at 1k steps, 12x12 renders: %s%n", rows.stream()
                .map(r -> String.format(Locale.ROOT, "%s %s=%.1f", r.task(), r.agent(), r.meanReturn()))
                .collect(Collectors.joining(", ")));
        assertTrue(rows.stream().allMatch(r -> r.meanReturn() >= 1));
    }
}
