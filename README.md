# Red Sea Parting Simulation

A real-time 3D simulation of the parting of the Red Sea, powered by shallow water equations and rendered with OpenGL.

![Water simulation using a Lax-Friedrichs solver on a 200×200 grid]

## How it works

| Layer | What it does |
|-------|-------------|
| **Shallow Water Equations** | A 2D grid (200×200) is evolved each frame using the Lax-Friedrichs finite-difference scheme. The solver tracks water depth *h*, x-momentum *hu*, and z-momentum *hv* per cell. |
| **Parting mechanism** | A seabed ridge rises along the centre line while a lateral force pushes water sideways. Together they drain the path and pile water into dramatic walls. |
| **OpenGL 3.3 Core** | Two meshes (water surface + seabed) are rebuilt from the simulation grid every frame. Phong lighting, depth-based colouring, and a Fresnel-like rim give the water a realistic look. |

## Prerequisites

* **CMake ≥ 3.16**
* A C++17 compiler (Clang, GCC, or MSVC)
* **OpenGL 3.3** capable GPU + drivers
* On macOS: Xcode Command Line Tools (`xcode-select --install`)

> GLFW and GLM are fetched automatically by CMake — no manual installation needed.

## Build & run

```bash
cd RedSea
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
./build/redsea
```

## Controls

| Key / Input | Action |
|-------------|--------|
| **SPACE** | Start the parting animation |
| **R** | Reset the simulation |
| **Left-drag** | Orbit the camera |
| **Scroll** | Zoom in / out |
| **ESC** | Quit |

## Project structure

```
RedSea/
├── CMakeLists.txt          # Build system (FetchContent for deps)
├── README.md
└── src/
    ├── main.cpp            # Window, OpenGL setup, render loop
    └── shallow_water.h     # SWE solver (header-only)
```

## Tuning tips

* **Grid resolution**: change `GRID` in `main.cpp` (higher = prettier but slower).
* **Parting speed**: `app.partingSpeed` controls how fast the sea parts (default 0.1 = ~10 s).
* **Water depth**: `WATER_H` sets the initial depth.
* **Damping**: the `damping` constant in `ShallowWater::step()` controls numerical viscosity.
