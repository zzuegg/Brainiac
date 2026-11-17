package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Plasticity;
import io.github.zzuegg.brainiac.train.Trainer;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

import org.junit.jupiter.api.Test;

class BrainMovieTest {

    static Model brainModel() {
        return Model.sequential().input(2)
                .add(Brain.builder()
                        .region("hidden", 36).initialNeurons(6).activation(Activation.RELU)
                        .region("deep", 25).initialNeurons(6).activation(Activation.TANH)
                        .connect("input", "hidden", 1.0)
                        .connect("hidden", "deep", 0.5)
                        .connect("deep", "hidden", 0.2)
                        .output("deep")
                        .ticks(2)
                        .build())
                .dense(3, Activation.SOFTMAX).seed(3).build();
    }

    @Test
    void activityIsTheMeanAbsoluteOutputOfEachRegion() {
        Model m = brainModel();
        Brain brain = (Brain) m.layers().getFirst();
        int[] y = new int[20];
        float[][] probe = EnsembleUncertaintyTest.blobs(20, y, 2);
        double[][] act = BrainMovie.of(m, probe).activity();
        float[] out = brain.forward(Var.of(Tensor.of(probe))).value().toArray();
        double[] expected = new double[25];
        for (int s = 0; s < 20; s++) {
            for (int k = 0; k < 25; k++) {
                expected[k] += Math.abs(out[s * 25 + k]) / 20;
            }
        }
        for (int k = 0; k < 25; k++) {
            assertEquals(expected[k], act[1][k], 1e-5);
        }
    }

    @Test
    void filmsActivityRewiringAndLossAsAGif() throws IOException {
        Model m = brainModel();
        int[] y = new int[300];
        float[][] x = EnsembleUncertaintyTest.blobs(300, y, 1);
        BrainMovie movie = BrainMovie.of(m, Arrays.copyOf(x, 30));
        movie.capture();
        Plasticity plasticity = Plasticity.of(m).neurogenesis(2, 0.02, 4).rewire(5, 0.2).seed(1);
        Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.02f)).batchSize(32).epochs(12)
                .listener(plasticity).listener(movie).seed(1)
                .fit(Dataset.of(x, EnsembleUncertaintyTest.oneHot(y)));
        assertEquals(13, movie.frames());
        int added = 0;
        int removed = 0;
        int heat = 0;
        for (int f = 1; f < movie.frames(); f++) {
            for (byte b : movie.frame(f)) {
                added += b == 3 ? 1 : 0;
                removed += b == 4 ? 1 : 0;
                heat += (b & 0xFF) >= 64 ? 1 : 0;
            }
        }
        assertTrue(added > 0 && removed > 0, "added " + added + " removed " + removed);
        assertTrue(heat > 0);
        byte[] gif = movie.toGif(20);
        ImageReader reader = ImageIO.getImageReadersByFormatName("gif").next();
        try (ImageInputStream in = ImageIO.createImageInputStream(new ByteArrayInputStream(gif))) {
            reader.setInput(in);
            assertEquals(13, reader.getNumImages(true));
            BufferedImage last = reader.read(12);
            assertEquals(movie.width(), last.getWidth());
            assertEquals(movie.height(), last.getHeight());
            byte[] px = movie.frame(12);
            int[] palette = BrainMovie.palette();
            for (int i = 0; i < px.length; i += 7) {
                assertEquals(palette[px[i] & 0xFF], last.getRGB(i % movie.width(), i / movie.width()) & 0xFFFFFF);
            }
        } finally {
            reader.dispose();
        }
    }
}
