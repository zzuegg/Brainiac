package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
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
    void customLayersNeedLoadWeights() throws IOException {
        Activation swish = Activation.named("swish", x -> x.mul(x.sigmoid()));
        Model m = Model.sequential().input(4).dense(5, swish).dense(1).seed(7).build();
        Path file = dir.resolve("custom.brainiac");
        m.save(file);
        IOException e = assertThrows(IOException.class, () -> Model.load(file));
        assertTrue(e.getMessage().contains("loadWeights"), e.getMessage());

        Model fresh = Model.sequential().input(4).dense(5, swish).dense(1).seed(8).build();
        fresh.loadWeights(file);
        assertArrayEquals(m.predict(X).toArray(), fresh.predict(X).toArray());
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
