// ============================================================================
// shallow_water.h — 2D Shallow Water Equations solver (Lax-Friedrichs scheme)
//
// The SWE describe how a thin layer of fluid flows under gravity.
// Variables per grid cell:
//   h  – water depth (above bottom)
//   hu – x-momentum (h * velocity_x)
//   hv – z-momentum (h * velocity_z)
//   B  – bottom elevation (topography)
//
// The surface elevation seen by the renderer is:  η = B + h
// ============================================================================
#pragma once

#include <vector>
#include <algorithm>
#include <array>
#include <cmath>

class ShallowWater {
public:
    // Grid dimensions and physical extents
    int   Nx, Nz;        // number of cells in x and z
    float Lx, Lz;        // physical domain size (metres)
    float dx, dz;        // cell spacing
    float g;             // gravitational acceleration

    // Per-cell quantities  (stored in row-major order: index = j*Nx + i)
    std::vector<float> h, hu, hv;   // conserved variables
    std::vector<float> B;           // bottom topography

    // Scratch arrays for time-stepping
    std::vector<float> h_new, hu_new, hv_new;

    // ------------------------------------------------------------------
    ShallowWater(int nx, int nz, float lx, float lz, float gravity = 9.81f)
        : Nx(nx), Nz(nz), Lx(lx), Lz(lz), g(gravity)
    {
        dx = Lx / (Nx - 1);
        dz = Lz / (Nz - 1);
        int n = Nx * Nz;
        h.assign(n, 0.f);   hu.assign(n, 0.f);   hv.assign(n, 0.f);
        B.assign(n, 0.f);
        h_new.assign(n, 0.f);  hu_new.assign(n, 0.f);  hv_new.assign(n, 0.f);
    }

    // Row-major index
    int idx(int i, int j) const { return j * Nx + i; }

    // Map grid indices → world coordinates (centred at origin)
    float worldX(int i) const { return (i - Nx * 0.5f) * dx; }
    float worldZ(int j) const { return (j - Nz * 0.5f) * dz; }

    // Smooth Hermite interpolation  (clamped to [0, 1])
    static float smoothstep(float t) {
        t = std::clamp(t, 0.f, 1.f);
        return t * t * (3.f - 2.f * t);
    }

    // ------------------------------------------------------------------
    // Initialise a calm sea with small sinusoidal perturbations
    // ------------------------------------------------------------------
    void initialize(float waterHeight) {
        constexpr float PI = 3.14159265358979f;
        for (int j = 0; j < Nz; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int id = idx(i, j);
                B[id]  = 0.f;
                float x = i * dx, z = j * dz;
                float perturbation =
                    0.03f * std::sin(2.f * PI * x / 4.f) *
                            std::sin(2.f * PI * z / 5.f)
                  + 0.02f * std::sin(2.f * PI * x / 3.f + 1.f) *
                            std::cos(2.f * PI * z / 7.f);
                h[id]  = waterHeight + perturbation;
                hu[id] = 0.f;
                hv[id] = 0.f;
            }
        }
    }

    // ------------------------------------------------------------------
    // Drive the parting animation.
    //   progress : 0 → 1   (how far along the animation we are)
    //   dt       : frame delta time (used for force integration)
    //
    // The seabed stays flat (B = 0).  Water is pushed sideways by a
    // lateral body force and directly drained from the corridor.
    // ------------------------------------------------------------------
    void applyParting(float progress, float dt) {
        if (progress <= 0.f) return;
        progress = std::min(progress, 1.f);

        int   centerI   = Nx / 2;
        float halfWidth = 1.5f + 0.5f * progress;          // half-width of the dry corridor (m)
        float maxForce  = 50.f * progress;                  // lateral push strength

        // The path "unzips" from front to back over the domain
        float pathFront = 0.05f * Lz;
        float pathBack  = pathFront + 0.9f * Lz * smoothstep(progress * 1.2f);

        for (int j = 0; j < Nz; ++j) {
            float z = j * dz;
            if (z < pathFront - 1.f || z > pathBack + 1.f) continue;

            // Taper the effect at the leading / trailing edges
            float zFactor = 1.f;
            float taper   = 2.f;
            if (z < pathFront)
                zFactor = smoothstep((z - pathFront + 1.f));
            else if (z - pathFront < taper)
                zFactor = smoothstep((z - pathFront) / taper);
            if (z > pathBack)
                zFactor *= smoothstep((pathBack + 1.f - z));
            else if (pathBack - z < taper)
                zFactor *= smoothstep((pathBack - z) / taper);

            for (int i = 0; i < Nx; ++i) {
                int   id    = idx(i, j);
                float xDist = std::abs((i - centerI) * dx);

                // Push water sideways (acts over a region wider than the corridor)
                if (xDist < halfWidth + 2.f && h[id] > 0.01f) {
                    float sign = (i < centerI) ? -1.f : 1.f;
                    float frc  = smoothstep(1.f - xDist / (halfWidth + 1.5f));
                    hu[id] += sign * maxForce * frc * zFactor * h[id] * dt;
                }

                // Drain water inside the corridor
                if (xDist < halfWidth) {
                    float inner  = smoothstep(1.f - xDist / halfWidth);
                    float drain  = 4.f * inner * zFactor * smoothstep(progress) * dt;
                    h[id] = std::max(0.f, h[id] - drain);
                    if (h[id] < 0.02f) { hu[id] = 0.f; hv[id] = 0.f; }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Compute a stable Δt from the CFL condition
    // ------------------------------------------------------------------
    float computeTimeStep() const {
        float maxSpeed = 1.f;
        for (int j = 1; j < Nz - 1; ++j)
            for (int i = 1; i < Nx - 1; ++i) {
                int id = idx(i, j);
                if (h[id] > 0.01f) {
                    float c = std::sqrt(g * h[id]);
                    float u = std::abs(hu[id] / h[id]);
                    float v = std::abs(hv[id] / h[id]);
                    maxSpeed = std::max(maxSpeed, u + c);
                    maxSpeed = std::max(maxSpeed, v + c);
                }
            }
        return 0.25f * std::min(dx, dz) / maxSpeed;
    }

    // ------------------------------------------------------------------
    // Advance one time-step using the Lax-Friedrichs finite-difference
    // scheme.  This is first-order accurate and unconditionally stable
    // when the CFL condition is respected.
    // ------------------------------------------------------------------
    void step(float dt) {
        const float damping = 0.999f;      // very light numerical viscosity

        // Helper: compute x-direction flux vector [F_h, F_hu, F_hv]
        auto fluxX = [&](int k) -> std::array<float,3> {
            if (h[k] < 0.001f) return {0.f, 0.f, 0.f};
            float uk = hu[k] / h[k];
            float vk = hv[k] / h[k];
            return { hu[k],
                     hu[k] * uk + 0.5f * g * h[k] * h[k],
                     hu[k] * vk };
        };
        // Helper: compute z-direction flux vector [G_h, G_hu, G_hv]
        auto fluxZ = [&](int k) -> std::array<float,3> {
            if (h[k] < 0.001f) return {0.f, 0.f, 0.f};
            float uk = hu[k] / h[k];
            float vk = hv[k] / h[k];
            return { hv[k],
                     hv[k] * uk,
                     hv[k] * vk + 0.5f * g * h[k] * h[k] };
        };

        // Interior cells  (boundaries handled separately below)
        for (int j = 1; j < Nz - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                int id = idx(i, j);
                int il = idx(i-1, j), ir = idx(i+1, j);
                int jd = idx(i, j-1), ju = idx(i, j+1);

                // Lax-Friedrichs averaging of neighbours
                float ha  = 0.25f * (h[il]  + h[ir]  + h[jd]  + h[ju]);
                float hua = 0.25f * (hu[il] + hu[ir] + hu[jd] + hu[ju]);
                float hva = 0.25f * (hv[il] + hv[ir] + hv[jd] + hv[ju]);

                // Centred flux differences
                auto fR = fluxX(ir), fL = fluxX(il);
                auto gU = fluxZ(ju), gD = fluxZ(jd);

                h_new[id]  = ha  - 0.5f*dt/dx*(fR[0]-fL[0]) - 0.5f*dt/dz*(gU[0]-gD[0]);
                hu_new[id] = hua - 0.5f*dt/dx*(fR[1]-fL[1]) - 0.5f*dt/dz*(gU[1]-gD[1]);
                hv_new[id] = hva - 0.5f*dt/dx*(fR[2]-fL[2]) - 0.5f*dt/dz*(gU[2]-gD[2]);

                // Source term: pressure force from bottom slope
                if (ha > 0.01f) {
                    float dBdx = (B[ir] - B[il]) / (2.f * dx);
                    float dBdz = (B[ju] - B[jd]) / (2.f * dz);
                    hu_new[id] -= dt * g * ha * dBdx;
                    hv_new[id] -= dt * g * ha * dBdz;
                }

                // Enforce non-negative depth & apply damping
                h_new[id]  = std::max(h_new[id], 0.f);
                hu_new[id] *= damping;
                hv_new[id] *= damping;

                // Dry-cell handling: kill momentum in very shallow cells
                if (h_new[id] < 0.005f) {
                    hu_new[id] = 0.f;
                    hv_new[id] = 0.f;
                }

                // Velocity clamping for robustness
                if (h_new[id] > 0.01f) {
                    constexpr float maxVel = 20.f;
                    float u = hu_new[id] / h_new[id];
                    float v = hv_new[id] / h_new[id];
                    if (std::abs(u) > maxVel)
                        hu_new[id] = (u > 0 ? maxVel : -maxVel) * h_new[id];
                    if (std::abs(v) > maxVel)
                        hv_new[id] = (v > 0 ? maxVel : -maxVel) * h_new[id];
                }
            }
        }

        // Reflective boundary conditions (walls on all four sides)
        for (int j = 0; j < Nz; ++j) {
            h_new[idx(0,j)]      =  h_new[idx(1,j)];
            hu_new[idx(0,j)]     = -hu_new[idx(1,j)];
            hv_new[idx(0,j)]     =  hv_new[idx(1,j)];
            h_new[idx(Nx-1,j)]   =  h_new[idx(Nx-2,j)];
            hu_new[idx(Nx-1,j)]  = -hu_new[idx(Nx-2,j)];
            hv_new[idx(Nx-1,j)]  =  hv_new[idx(Nx-2,j)];
        }
        for (int i = 0; i < Nx; ++i) {
            h_new[idx(i,0)]      =  h_new[idx(i,1)];
            hu_new[idx(i,0)]     =  hu_new[idx(i,1)];
            hv_new[idx(i,0)]     = -hv_new[idx(i,1)];
            h_new[idx(i,Nz-1)]   =  h_new[idx(i,Nz-2)];
            hu_new[idx(i,Nz-1)]  =  hu_new[idx(i,Nz-2)];
            hv_new[idx(i,Nz-1)]  = -hv_new[idx(i,Nz-2)];
        }

        // Swap old ↔ new
        std::swap(h,  h_new);
        std::swap(hu, hu_new);
        std::swap(hv, hv_new);
    }

    // ------------------------------------------------------------------
    // Accessors for the renderer
    // ------------------------------------------------------------------
    float surfaceHeight(int i, int j) const { return B[idx(i,j)] + h[idx(i,j)]; }
    float waterDepth   (int i, int j) const { return h[idx(i,j)]; }
    float bottomHeight (int i, int j) const { return B[idx(i,j)]; }
};
