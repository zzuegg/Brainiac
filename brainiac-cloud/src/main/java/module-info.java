/**
 * Renting GPUs in the cloud to train and run Brainiac: a client for Vast.ai's REST API and a
 * command line ({@link io.github.zzuegg.brainiac.cloud.VastCli}) that rents one instance,
 * runs a job on it over SSH and destroys it again.
 */
module io.github.zzuegg.brainiac.cloud {
    requires transitive io.github.zzuegg.brainiac.io;
    requires transitive java.net.http;

    exports io.github.zzuegg.brainiac.cloud;
}
