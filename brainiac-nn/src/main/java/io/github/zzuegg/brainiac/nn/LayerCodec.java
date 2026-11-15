package io.github.zzuegg.brainiac.nn;

/**
 * Turns a layer into a {@link Config} and back, so models using it can be saved and
 * loaded with {@link Model#save} and {@link Model#load}. Only settings go in the config:
 * the layer's parameters and buffers are saved and restored automatically, in the order
 * {@link Layer#parameters()} and {@link Layer#buffers()} list them.
 *
 * <pre>{@code
 * Layer.register("scale", Scale.class, new LayerCodec<>() {
 *     public Config write(Scale s) { return Config.builder().put("width", s.width()).build(); }
 *     public Scale read(Config c) { return new Scale(c.getInt("width")); }
 * });
 * }</pre>
 *
 * @param <L> the layer class
 */
public interface LayerCodec<L extends Layer> {

    /**
     * Describes a layer's settings.
     *
     * @param layer the layer
     * @return its settings
     */
    Config write(L layer);

    /**
     * Rebuilds a layer from its settings. Parameter values do not matter; they are
     * overwritten from the file afterwards.
     *
     * @param config the settings {@link #write} returned
     * @return a new layer
     */
    L read(Config config);
}
