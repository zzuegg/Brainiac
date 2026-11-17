package io.github.zzuegg.brainiac.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class AnnotationsTest {

    @TempDir
    Path dir;

    @Test
    void readsCoco() throws IOException {
        Path f = dir.resolve("instances.json");
        Files.writeString(f, """
                {"images": [{"id": 7, "file_name": "a.jpg", "width": 640, "height": 480},
                            {"id": 9, "file_name": "b.jpg", "width": 100, "height": 50}],
                 "annotations": [{"image_id": 7, "category_id": 18, "bbox": [10, 20, 30, 40], "iscrowd": 0},
                                 {"image_id": 7, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": 1},
                                 {"image_id": 7, "category_id": 1, "bbox": [1.5, 2, 3, 4]}],
                 "categories": [{"id": 1, "name": "person"}, {"id": 18, "name": "dog"}]}
                """);
        Annotations.Dataset d = Annotations.coco(f, dir.resolve("img"));
        assertEquals(List.of("person", "dog"), d.classes());
        assertEquals(2, d.images().size());
        Annotations.Image a = d.images().getFirst();
        assertEquals(dir.resolve("img/a.jpg"), a.file());
        assertEquals(640, a.width());
        assertEquals(List.of(new Annotations.Box(10, 20, 40, 60, 1), new Annotations.Box(1.5f, 2, 4.5f, 6, 0)),
                a.boxes());
        assertEquals(List.of(), d.images().get(1).boxes());
    }

    @Test
    void readsVoc() throws IOException {
        Path ann = Files.createDirectories(dir.resolve("Annotations"));
        Files.writeString(ann.resolve("000001.xml"), """
                <annotation><filename>000001.jpg</filename>
                  <size><width>353</width><height>500</height><depth>3</depth></size>
                  <object><name>dog</name><difficult>0</difficult>
                    <bndbox><xmin>48</xmin><ymin>240</ymin><xmax>195</xmax><ymax>371</ymax></bndbox></object>
                  <object><name>person</name><difficult>1</difficult>
                    <bndbox><xmin>8</xmin><ymin>12</ymin><xmax>352</xmax><ymax>498</ymax></bndbox></object>
                </annotation>
                """);
        Annotations.Dataset all = Annotations.voc(ann, dir, null, false);
        assertEquals(List.of("dog", "person"), all.classes());
        Annotations.Image im = all.images().getFirst();
        assertEquals(353, im.width());
        assertEquals(new Annotations.Box(47, 239, 195, 371, 0), im.boxes().getFirst());
        Annotations.Dataset easy = Annotations.voc(ann, dir, List.of("person", "dog"), true);
        assertEquals(List.of(new Annotations.Box(47, 239, 195, 371, 1)), easy.images().getFirst().boxes());
        assertThrows(IOException.class, () -> Annotations.voc(ann, dir, List.of("cat"), false));
    }
}
