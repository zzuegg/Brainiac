package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ClassicalSegmentationTest {

    @Test
    void superpixels() {
        StringBuilder line = new StringBuilder("12.3.3 segments/BR/USE/ASA:");
        List<ClassicalSegmentation.SuperpixelRow> rows = null;
        for (int target : new int[] {100, 200}) {
            rows = ClassicalSegmentation.superpixels(10, target, 1233);
            line.append(" target ").append(target).append(':');
            for (ClassicalSegmentation.SuperpixelRow r : rows) {
                line.append(String.format(" %s %.0f/%.3f/%.3f/%.3f;", r.method(), r.segments(), r.boundaryRecall(),
                    r.undersegmentation(), r.asa()));
            }
        }
        System.out.println(line);
        assertTrue(rows.get(0).boundaryRecall() > 0.6, line.toString());
    }

    @Test
    void markovRandomFields() {
        List<ClassicalSegmentation.MrfRow> rows = ClassicalSegmentation.mrf(8, 0.6, 1234);
        StringBuilder line = new StringBuilder("12.3.4 error/energy/ms:");
        for (ClassicalSegmentation.MrfRow r : rows) {
            line.append(String.format(" %s %s %.4f/%.1f/%.1f;", r.task(), r.method(), r.pixelError(), r.energy(),
                r.millis()));
        }
        System.out.println(line);
        assertTrue(rows.get(4).energy() <= rows.get(2).energy() + 1e-6, line.toString());
        assertTrue(rows.get(4).pixelError() < rows.get(0).pixelError(), line.toString());
        assertTrue(rows.get(8).energy() <= rows.get(6).energy(), line.toString());
    }

    @Test
    void houghTransforms() {
        List<ClassicalSegmentation.HoughRow> rows = ClassicalSegmentation.hough(8, 1235);
        StringBuilder line = new StringBuilder("12.3.5 F1/param-error:");
        for (ClassicalSegmentation.HoughRow r : rows) {
            line.append(String.format(" %s n=%d c=%.1f %s %.2f/%.2f;", r.primitive(), r.count(), r.clutter(),
                r.method(), r.f1(), r.parameterError()));
        }
        System.out.println(line);
        assertTrue(rows.get(8).f1() > 0.5, line.toString());
    }
}
