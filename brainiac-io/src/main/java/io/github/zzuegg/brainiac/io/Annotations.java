package io.github.zzuegg.brainiac.io;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 * Object-detection annotations in the COCO (JSON) and Pascal VOC (XML) formats: per image its
 * file, size and boxes, with boxes as corners {@code [x1, y1, x2, y2]} in pixels, as
 * {@code io.github.zzuegg.brainiac.nn.Boxes} uses them.
 */
public final class Annotations {

    private Annotations() {
    }

    /**
     * One object.
     *
     * @param x1 left
     * @param y1 top
     * @param x2 right
     * @param y2 bottom
     * @param label the class id, an index into {@link Dataset#classes()}
     */
    public record Box(float x1, float y1, float x2, float y2, int label) {
    }

    /**
     * One annotated image.
     *
     * @param file the image file
     * @param width its width in pixels
     * @param height its height in pixels
     * @param boxes its objects
     */
    public record Image(Path file, int width, int height, List<Box> boxes) {
    }

    /**
     * Annotated images and the names of their classes.
     *
     * @param images the images
     * @param classes the class names; a box's label indexes this list
     */
    public record Dataset(List<Image> images, List<String> classes) {
    }

    /**
     * Reads a COCO {@code instances_*.json}. Category ids become labels 0, 1, ... in the
     * order of the file's categories; crowd annotations are left out.
     *
     * @param json the annotation file
     * @param images the folder holding the images
     * @return the images in the file's order, also those without boxes
     * @throws IOException if the file cannot be read
     */
    public static Dataset coco(Path json, Path images) throws IOException {
        Map<String, Object> root = Json.object(Json.read(json));
        Map<Long, Integer> label = new HashMap<>();
        List<String> classes = new ArrayList<>();
        for (Object o : Json.array(root.get("categories"))) {
            Map<String, Object> c = Json.object(o);
            label.put(Json.integer(c, "id"), classes.size());
            classes.add(Json.string(c, "name", "class" + classes.size()));
        }
        Map<Long, List<Box>> boxes = new HashMap<>();
        for (Object o : Json.array(root.getOrDefault("annotations", List.of()))) {
            Map<String, Object> a = Json.object(o);
            if (Json.integer(a, "iscrowd", 0) != 0) {
                continue;
            }
            List<Object> b = Json.array(a.get("bbox"));
            float x = ((Number) b.get(0)).floatValue();
            float y = ((Number) b.get(1)).floatValue();
            float w = ((Number) b.get(2)).floatValue();
            float h = ((Number) b.get(3)).floatValue();
            Integer l = label.get(Json.integer(a, "category_id"));
            if (l == null) {
                throw new IllegalArgumentException("annotation with unknown category " + a.get("category_id"));
            }
            boxes.computeIfAbsent(Json.integer(a, "image_id"), k -> new ArrayList<>()).add(new Box(x, y, x + w, y + h, l));
        }
        List<Image> out = new ArrayList<>();
        for (Object o : Json.array(root.get("images"))) {
            Map<String, Object> im = Json.object(o);
            out.add(new Image(images.resolve(Json.string(im, "file_name", "")), (int) Json.integer(im, "width"),
                    (int) Json.integer(im, "height"), List.copyOf(boxes.getOrDefault(Json.integer(im, "id"), List.of()))));
        }
        return new Dataset(List.copyOf(out), List.copyOf(classes));
    }

    /**
     * Reads Pascal VOC annotations: one XML file per image. VOC's corners count from 1 and
     * include the last pixel; they are shifted to count from 0. Classes are named in the
     * order they first appear unless given.
     *
     * @param annotations the folder of {@code .xml} files, read in name order
     * @param images the folder holding the images
     * @param classes the class names in label order, or {@code null} to collect them
     * @param skipDifficult whether to leave out objects marked difficult
     * @return the images and classes
     * @throws IOException if a file cannot be read or parsed
     */
    public static Dataset voc(Path annotations, Path images, List<String> classes, boolean skipDifficult)
            throws IOException {
        Map<String, Integer> label = new LinkedHashMap<>();
        if (classes != null) {
            for (String c : classes) {
                label.put(c, label.size());
            }
        }
        List<Path> files;
        try (Stream<Path> s = Files.list(annotations)) {
            files = s.filter(p -> p.getFileName().toString().endsWith(".xml")).sorted().toList();
        }
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        try {
            factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
            factory.setExpandEntityReferences(false);
        } catch (ParserConfigurationException e) {
            throw new IllegalStateException(e);
        }
        List<Image> out = new ArrayList<>();
        for (Path f : files) {
            Document doc;
            try {
                doc = factory.newDocumentBuilder().parse(f.toFile());
            } catch (ParserConfigurationException | SAXException e) {
                throw new IOException("cannot parse " + f + ": " + e.getMessage(), e);
            }
            Element root = doc.getDocumentElement();
            Element size = child(root, "size");
            List<Box> boxes = new ArrayList<>();
            NodeList objects = root.getElementsByTagName("object");
            for (int i = 0; i < objects.getLength(); i++) {
                Element o = (Element) objects.item(i);
                if (skipDifficult && "1".equals(text(o, "difficult", "0"))) {
                    continue;
                }
                String name = text(o, "name", "");
                Integer l = label.get(name);
                if (l == null) {
                    if (classes != null) {
                        throw new IOException(f + ": unknown class " + name);
                    }
                    l = label.size();
                    label.put(name, l);
                }
                Element b = child(o, "bndbox");
                boxes.add(new Box(number(b, "xmin") - 1, number(b, "ymin") - 1, number(b, "xmax"), number(b, "ymax"), l));
            }
            String name = text(root, "filename", f.getFileName().toString().replace(".xml", ".jpg"));
            out.add(new Image(images.resolve(name), (int) number(size, "width"), (int) number(size, "height"),
                    List.copyOf(boxes)));
        }
        return new Dataset(List.copyOf(out), List.copyOf(label.keySet()));
    }

    private static Element child(Element e, String tag) throws IOException {
        NodeList l = e.getElementsByTagName(tag);
        if (l.getLength() == 0) {
            throw new IOException("missing <" + tag + ">");
        }
        return (Element) l.item(0);
    }

    private static String text(Element e, String tag, String otherwise) {
        NodeList l = e.getElementsByTagName(tag);
        return l.getLength() == 0 ? otherwise : l.item(0).getTextContent().strip();
    }

    private static float number(Element e, String tag) throws IOException {
        return Float.parseFloat(child(e, tag).getTextContent().strip());
    }
}
