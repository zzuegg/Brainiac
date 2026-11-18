package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.vision.ClassicCnnShootout.Result;

import org.junit.jupiter.api.Test;

class CompoundScalingTest {

    @Test
    void compoundScalingAgainstWidthOnly() {
        Result base = ClassicCnnShootout.compoundScaling(0, true, 150, 5);
        Result compound = ClassicCnnShootout.compoundScaling(2, true, 150, 5);
        Result width = ClassicCnnShootout.compoundScaling(2, false, 150, 5);
        System.out.printf("EfficientNet-lite scaling phi=2: compound %.3f (%d MACs), width-only %.3f (%d MACs); "
                + "base %.3f (%d MACs)%n", compound.accuracy(), compound.multiplyAdds(), width.accuracy(),
                width.multiplyAdds(), base.accuracy(), base.multiplyAdds());
        assertTrue(compound.multiplyAdds() > base.multiplyAdds() && width.multiplyAdds() > base.multiplyAdds());
        assertTrue(compound.accuracy() > width.accuracy() - 0.05f, "compound scaling keeps up with width-only");
    }
}
