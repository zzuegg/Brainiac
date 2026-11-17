# Running Brainiac on rented Vast.ai GPUs

Brainiac trains on one GPU per run through its Vulkan backend. `brainiac-cloud` rents one
machine on [Vast.ai](https://vast.ai), copies this checkout to it, runs `setup.sh` there and
destroys the machine again.

## Once

1. Add your SSH public key at https://cloud.vast.ai/manage-keys/ (new instances get it).
2. Give the launcher your API key, either as the `VAST_API_KEY` environment variable or
   saved in the studio (Training → Vast.ai), which keeps it in
   `~/.brainiac/studio/keys.properties`. The launcher never prints it.

## Use

```sh
./gradlew :brainiac-cloud:run --args="offers"                 # cheapest single RTX 4090/5090, A100, H100
./gradlew :brainiac-cloud:run --args="run --offer 123456"     # asks, rents, smoke-tests, destroys
./gradlew :brainiac-cloud:run --args="run --offer 123456 --job tests"
./gradlew :brainiac-cloud:run --args="run --offer 123456 --job run -- ./gradlew :brainiac-examples:run --args=iris"
./gradlew :brainiac-cloud:run --args="worker --offer 123456"  # studio worker behind an SSH tunnel
./gradlew :brainiac-cloud:run --args="instances"              # anything still rented (and billing)?
./gradlew :brainiac-cloud:run --args="destroy --all"
```

Renting always asks first with the offer and its hourly price (`--yes` skips the question).
The instance is destroyed when the job ends or fails, after `--max-minutes` (default 120),
or on Ctrl+C. `--keep` keeps it; `--instance ID` reuses one. Logs go to `build/vast/`.
Only the `HEAD` commit is copied, so commit first.

## setup.sh jobs

| Job | What it does |
| --- | --- |
| `check` | Installs the Vulkan loader and checks that an NVIDIA Vulkan device shows up |
| `smoke` | `check`, JDK 21 + 27, build, Vulkan backend tests, benchmarks, 1 epoch of MNIST training |
| `tests` | `check`, JDKs, `./gradlew build` |
| `bench` | `check`, JDKs, benchmarks only |
| `worker` | A studio worker (`--worker --listen 127.0.0.1 --port 7870`) for chat and training |
| `run CMD...` | `check`, JDKs, then any command |

## Vulkan on NVIDIA containers

NVIDIA's container runtime mounts the driver's Vulkan libraries only when the container asks
for the `graphics` capability, so instances are rented with `NVIDIA_DRIVER_CAPABILITIES=all`.
`setup.sh` writes the missing `nvidia_icd.json` when the libraries are there but the file is
not. Some hosts still expose CUDA only; then `check` fails with exit code 2 and the instance
is destroyed. Pick another offer.

## The worker

`worker` starts the studio in worker mode bound to the instance's `127.0.0.1`, reachable only
through the SSH tunnel the launcher holds open (`127.0.0.1:7870` on your PC). The launcher
generates the bearer token and sends it over the SSH connection, then prints the URL and token
to add under Training → Add a host. `VastLaunch.exposePorts(7870)` maps the port publicly
instead, but that sends the token and every prompt unencrypted.
