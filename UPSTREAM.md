# Upstream dependency policy

This repository is a thin integration layer around four upstream projects. The
submodule revisions below were refreshed on 2026-08-22:

| Dependency | Revision | Update policy |
| --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `e85caa81ea2b65797396018c179b87ad61fa38ab` | Track `master`; the current mtmd/common APIs are adapted by `zz_compat.patch`. |
| [stable-diffusion.cpp](https://github.com/thxcode/stable-diffusion.cpp) | `adbed8f22496342410ef565476fddc156b20e7f7` | Track `dev-dcf91f9e-5`; this is newer than the repository's stale `master` branch and is the branch configured in `.gitmodules`. |
| [concurrentqueue](https://github.com/cameron314/concurrentqueue) | `683b9e31ea15eb69f1b81cc1defc7850d5f20b71` | Track `master`. |
| [readerwriterqueue](https://github.com/cameron314/readerwriterqueue) | `131af2c479c6ba36142ee9805e9f60fc4adfa22c` | Track `master`. |

## Compatibility layer

`llama-box/patches/llama.cpp/zz_compat.patch` contains only the integration
surface that does not belong in the upstream projects:

- current mtmd image/audio value types and audio-buffer decoding;
- compatibility getters for causal attention, model architecture and unknown vocabulary tokens;
- chat-template capability queries and the legacy template alias helper;
- the current sampler and chat-template API calls used by llama-box;
- the deprecated `defrag_thold` command-line compatibility passthrough (accepted and ignored by current upstream);
- the remaining target/API compatibility shims.

The old patch set was written against an earlier llama.cpp layout. Patches that
no longer apply were removed instead of leaving build-time warnings and false
confidence. The remaining patches are applied by `llama-box/scripts/build-patch.cmake`
from a clean submodule checkout and are cleaned after the build.

## Refresh procedure

1. Update each submodule to a reviewed upstream revision.
2. Run the CPU configure/build and verify that every required patch applies.
3. Port only the integration breakages into `zz_compat.patch` or the llama-box sources.
4. Run `git diff --check`, `build/bin/llama-box --help`, and the available API tests.

The stable-diffusion branch should not be changed to `master` automatically:
the current pinned development branch is the newer functional revision for this
integration.
