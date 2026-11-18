package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class LengthCurriculaTest {

    @Test
    void progressTeachersAgainstFixedPacing() {
        Map<DataCuration.LengthTeacher, DataCuration.LengthRun> runs = new EnumMap<>(
                DataCuration.LengthTeacher.class);
        for (DataCuration.LengthTeacher t : DataCuration.LengthTeacher.values()) {
            runs.put(t, DataCuration.lengthCurriculum(t, 300, 3));
            System.out.println("length curriculum, 300 steps: " + runs.get(t));
        }
        DataCuration.LengthRun exp3 = runs.get(DataCuration.LengthTeacher.EXP3S);
        DataCuration.LengthRun uniform = runs.get(DataCuration.LengthTeacher.UNIFORM);
        System.out.printf("progress curricula: wasted on the random task EXP3.S %.2f, slope greedy %.2f, Thompson %.2f "
                + "vs uniform %.2f; final 4-digit slot accuracy EXP3.S %.3f vs fixed pacing %.3f%n", exp3.wasted(),
                runs.get(DataCuration.LengthTeacher.SLOPE_GREEDY).wasted(),
                runs.get(DataCuration.LengthTeacher.SLOPE_THOMPSON).wasted(), uniform.wasted(),
                exp3.finalAccuracy(), runs.get(DataCuration.LengthTeacher.FIXED_PACING).finalAccuracy());
        assertTrue(uniform.wasted() > 0.1 && uniform.wasted() < 0.3);
        assertTrue(runs.get(DataCuration.LengthTeacher.FIXED_PACING).finalAccuracy() > 0.2);
        assertTrue(exp3.finalAccuracy() > 0.2);
    }
}
