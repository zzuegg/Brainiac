package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelFileTest {

    private static final Tensor X = Tensor.uniform(Shape.of(6, 4), -1, 1, new SplittableRandom(3));

    @TempDir
    Path dir;

    @Test
    void sequentialModelRoundTrips() throws IOException {
        Model m = Model.sequential().input(4).dense(8, Activation.RELU).batchNorm().dropout(0.2f)
                .dense(3).activation(Activation.SOFTMAX).seed(1).build();
        m.forward(Var.of(X), true); // move the batch-norm running statistics
        Path file = dir.resolve("model.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        assertEquals(m.summary(), loaded.summary());
        assertArrayEquals(m.predict(X).toArray(), loaded.predict(X).toArray());
        BatchNorm original = (BatchNorm) m.layers().get(1);
        BatchNorm restored = (BatchNorm) loaded.layers().get(1);
        assertArrayEquals(original.runningMean().toArray(), restored.runningMean().toArray());
    }

    @Test
    void copiesAreIndependent() {
        Model m = Model.sequential().input(4).dense(8, Activation.RELU).dense(3).seed(1).build();
        Model copy = m.copy(io.github.zzuegg.brainiac.tensor.Device.cpu());
        assertEquals(m.summary(), copy.summary());
        assertArrayEquals(m.predict(X).toArray(), copy.predict(X).toArray());
        copy.parameters().get(0).assign(copy.parameters().get(0).value().mul(2f));
        org.junit.jupiter.api.Assertions.assertFalse(java.util.Arrays.equals(m.predict(X).toArray(),
                copy.predict(X).toArray()));
    }

    @Test
    void graphModelWithSharedLayerRoundTrips() throws IOException {
        Node in = Node.input(4);
        Dense shared = new Dense(4, 4, Activation.TANH, Init.XAVIER, new SplittableRandom(2));
        Node h = in.layer(shared);
        Node out = h.layer(shared).add(h).concat(in).dense(2, Activation.SIGMOID);
        Model m = Model.graph(in, out).seed(5).build();
        Path file = dir.resolve("graph.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        assertEquals(m.summary(), loaded.summary());
        assertEquals(m.layers().size(), loaded.layers().size());
        assertArrayEquals(m.predict(X).toArray(), loaded.predict(X).toArray());
    }

    @Test
    void unregisteredCustomLayersNeedLoadWeights() throws IOException {
        Activation mine = Activation.named("unregistered-swish", x -> x.mul(x.sigmoid()));
        Model m = Model.sequential().input(4).dense(5, mine).dense(1).seed(7).build();
        Path file = dir.resolve("custom.brainiac");
        m.save(file);
        IOException e = assertThrows(IOException.class, () -> Model.load(file));
        assertTrue(e.getMessage().contains("unregistered-swish") && e.getMessage().contains("loadWeights"),
                e.getMessage());

        Model fresh = Model.sequential().input(4).dense(5, mine).dense(1).seed(8).build();
        fresh.loadWeights(file);
        assertArrayEquals(m.predict(X).toArray(), fresh.predict(X).toArray());

        Model withLayer = Model.sequential().input(4).layer(new Unregistered()).seed(1).build();
        withLayer.save(file);
        e = assertThrows(IOException.class, () -> Model.load(file));
        assertTrue(e.getMessage().contains(Unregistered.class.getName()) && e.getMessage().contains("Layer.register"),
                e.getMessage());
    }

    private static final class Unregistered implements Layer {
        @Override
        public Var forward(Var input) {
            return input;
        }
    }

    @Test
    void learnableActivationsRoundTrip() throws IOException {
        Node in = Node.input(4);
        Activation shared = Activation.prelu().bind(6);
        Node h = in.dense(6, shared).activation(Activation.swish()).dense(6, shared);
        Model m = Model.graph(in, h.dense(2, Activation.prelu())).seed(9).build();
        randomize(m);
        Path file = dir.resolve("learnable.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        assertEquals(m.summary(), loaded.summary());
        assertEquals(m.parameterCount(), loaded.parameterCount());
        assertArrayEquals(m.predict(X).toArray(), loaded.predict(X).toArray());
        // the shared slopes are still one parameter
        Dense first = (Dense) loaded.layers().get(0);
        Dense second = (Dense) loaded.layers().get(2);
        assertSame(first.activation(), second.activation());
    }

    @Test
    void registeredCustomActivationsLoad() throws IOException {
        Activation cube = Activation.named("test-cube", x -> x.mul(x).mul(x));
        Activation.register(cube);
        Activation scaledTanh = Activation.withParameters("test-scaled-tanh",
                width -> List.of(Tensor.ones(Shape.of(width))), (x, p) -> x.tanh().mul(p.get(0)));
        Activation.register(scaledTanh);
        Activation.register("test-shift", c -> new Shift(c.getFloat("by")));

        Model m = Model.sequential().input(4).dense(5, cube).dense(5, scaledTanh)
                .activation(new Shift(0.5f)).dense(2).seed(10).build();
        randomize(m);
        Path file = dir.resolve("registered.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        assertEquals(m.summary(), loaded.summary());
        assertArrayEquals(m.predict(X).toArray(), loaded.predict(X).toArray());
        assertEquals(0.5f, ((Shift) loaded.layers().get(2)).by);
    }

    /** An activation with a setting: adds a constant. */
    private static final class Shift implements Activation {
        final float by;

        Shift(float by) {
            this.by = by;
        }

        @Override
        public Var forward(Var input) {
            return input.add(by);
        }

        @Override
        public Config config() {
            return Config.builder().put("by", by).build();
        }

        @Override
        public String toString() {
            return "test-shift";
        }
    }

    @Test
    void registeredCustomLayersLoadWithTheirSettingsAndParameters() throws IOException {
        Layer.register("test-gated", Gated.class, new LayerCodec<>() {
            @Override
            public Config write(Gated g) {
                return Config.builder()
                        .put("width", g.width)
                        .put("gate", g.gate)
                        .put("all", ALL_TYPES)
                        .build();
            }

            @Override
            public Gated read(Config c) {
                assertEquals(ALL_TYPES, c.getConfig("all"));
                return new Gated(c.getInt("width"), c.getActivation("gate"));
            }
        });
        Model m = Model.sequential().input(4).layer(new Gated(4, Activation.prelu())).dense(2).seed(11).build();
        randomize(m);
        Path file = dir.resolve("gated.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        assertEquals(m.parameterCount(), loaded.parameterCount());
        assertArrayEquals(m.predict(X).toArray(), loaded.predict(X).toArray());
    }

    private static final Config ALL_TYPES = Config.builder()
            .put("b", true).put("i", 3).put("l", 4L).put("f", 1.5f).put("d", 2.5).put("s", "text")
            .put("ints", new int[] {1, 2}).put("floats", new float[] {0.5f})
            .put("nested", Config.builder().put("act", Activation.TANH).build())
            .build();

    /** {@code x * gate(scale)}, with a learnable scale and a learnable gate activation. */
    private static final class Gated implements Layer {
        final int width;
        final Activation gate;
        final Var scale;

        Gated(int width, Activation gate) {
            this.width = width;
            this.gate = gate.bind(width);
            this.scale = Var.param(Tensor.ones(Shape.of(width)));
        }

        @Override
        public Var forward(Var input) {
            return input.mul(gate.forward(input.mul(scale)));
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new java.util.ArrayList<>(List.of(scale));
            p.addAll(gate.parameters());
            return p;
        }
    }

    @Test
    void readsVersion1Files() throws IOException {
        Path file = dir.resolve("v1.brainiac");
        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(file))) {
            out.writeInt(0x42524E43);
            out.writeInt(1);
            out.writeInt(2);
            out.writeInt(1);
            out.writeInt(1); // one layer
            out.writeUTF("dense");
            out.writeInt(2);
            out.writeInt(1);
            out.writeUTF("relu");
            out.writeInt(1); // one step
            out.writeByte(Node.Kind.LAYER.ordinal());
            out.writeInt(1);
            out.writeInt(0);
            out.writeInt(0);
            out.writeInt(2); // weight [2, 1] = (1, 2), bias [1] = 0.5
            writeTensor(out, new int[] {2, 1}, 1, 2);
            writeTensor(out, new int[] {1}, 0.5f);
        }
        Model m = Model.load(file);
        assertArrayEquals(new float[] {3.5f}, m.predict(1, 1), 1e-6f);
        Model fresh = Model.sequential().input(2).dense(1, Activation.RELU).build();
        assertArrayEquals(new float[] {3.5f}, fresh.loadWeights(file).predict(1, 1), 1e-6f);
    }

    private static void writeTensor(DataOutputStream out, int[] dims, float... values) throws IOException {
        out.writeInt(dims.length);
        for (int d : dims) {
            out.writeInt(d);
        }
        ByteBuffer bytes = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().put(values);
        out.write(bytes.array());
    }

    private static void randomize(Model m) {
        SplittableRandom r = new SplittableRandom(42);
        for (Var p : m.parameters()) {
            p.assign(Tensor.uniform(p.shape(), 0.1f, 0.9f, r));
        }
    }

    @Test
    void rejectsMismatchedOrInvalidFiles() throws IOException {
        Path file = dir.resolve("small.brainiac");
        Model.sequential().input(4).dense(2).build().save(file);
        Model other = Model.sequential().input(4).dense(3).build();
        assertThrows(IOException.class, () -> other.loadWeights(file));

        Path junk = dir.resolve("junk.bin");
        Files.write(junk, new byte[] {1, 2, 3, 4, 5, 6, 7, 8});
        assertThrows(IOException.class, () -> Model.load(junk));

        byte[] bytes = Files.readAllBytes(file);
        Path truncated = dir.resolve("truncated.brainiac");
        Files.write(truncated, java.util.Arrays.copyOf(bytes, bytes.length - 3));
        assertThrows(IOException.class, () -> Model.load(truncated));
    }
}
