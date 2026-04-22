# Installation

Full build-from-source guide.

## Requirements

The stack targets Strix Halo specifically. Radeon 8060S iGPU, gfx1151, wave32 WMMA.
Linux kernel 6.18.22-lts is the reference. Rust 1.86 stable.

## Distro policy

CachyOS with Btrfs plus snapper plus limine is the reference host.
Other Arch-family distros work. Ubuntu is not tested.

## Build steps

Clone the repo, run `cargo build --release --workspace`, then `cargo install --path crates/1bit-cli`.
