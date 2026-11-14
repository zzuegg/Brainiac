package io.github.zzuegg.brainiac;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class BrainiacTest {

    @Test
    void exposesVersion() {
        assertEquals("0.1.0-SNAPSHOT", Brainiac.VERSION);
    }
}
