/**
 * Brainiac Studio: a local web app for chatting with, generating with, building and training
 * models. The server is the JDK's HTTP server bound to this machine only; the page is plain
 * HTML, CSS and JavaScript served from this module.
 *
 * <p>Capabilities are found at startup: built-in ones from the Brainiac modules present, and
 * more from any module that provides {@link io.github.zzuegg.brainiac.studio.spi.CapabilityProvider}.
 */
module io.github.zzuegg.brainiac.studio {
    requires io.github.zzuegg.brainiac;
    requires java.net.http;
    requires jdk.httpserver;

    exports io.github.zzuegg.brainiac.studio;
    exports io.github.zzuegg.brainiac.studio.spi;

    uses io.github.zzuegg.brainiac.studio.spi.CapabilityProvider;
    provides io.github.zzuegg.brainiac.studio.spi.CapabilityProvider
            with io.github.zzuegg.brainiac.studio.server.CoreCapabilities;
}
