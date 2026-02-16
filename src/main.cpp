// ============================================================================
// main.cpp — Red Sea Parting Simulation
//
// Renders a 3D water surface driven by a shallow-water-equations solver.
// Press SPACE to trigger the parting animation.
//
// Dependencies (fetched automatically by CMake):
//   • GLFW  — windowing & input
//   • GLM   — linear-algebra / matrix math
//   • OpenGL 3.3 Core
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// --- OpenGL headers ---------------------------------------------------------
#ifdef __APPLE__
    #include <OpenGL/gl3.h>          // macOS ships GL 3.3–4.1 in the framework
#else
    // On Linux / Windows you would include GLAD here instead:
    // #include <glad/glad.h>
    #include <GL/gl.h>
#endif

#define GLFW_INCLUDE_NONE            // we supply our own GL header above
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shallow_water.h"

// ============================================================================
//  GLSL Shader Sources  (embedded as raw string literals)
// ============================================================================

// ---- Water surface ---------------------------------------------------------
static const char* waterVertSrc = R"(
#version 330 core
layout(location = 0) in vec3  aPos;
layout(location = 1) in vec3  aNormal;
layout(location = 2) in float aDepth;
layout(location = 3) in vec2  aVelocity;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3  FragPos;
out vec3  Normal;
out float WaterDepth;
out vec2  Velocity;

void main() {
    vec4 wp  = model * vec4(aPos, 1.0);
    FragPos  = wp.xyz;
    Normal   = mat3(transpose(inverse(model))) * aNormal;
    WaterDepth = aDepth;
    Velocity   = aVelocity;
    gl_Position = projection * view * wp;
}
)";

static const char* waterFragSrc = R"(
#version 330 core
in vec3  FragPos;
in vec3  Normal;
in float WaterDepth;
in vec2  Velocity;

out vec4 FragColor;

uniform vec3  lightDir;
uniform vec3  viewPos;
uniform float uTime;

// ---- Procedural noise (hash-based) ----------------------------------------
vec2 hash22(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453);
}
float gnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(dot(hash22(i + vec2(0,0)), f - vec2(0,0)),
                   dot(hash22(i + vec2(1,0)), f - vec2(1,0)), u.x),
               mix(dot(hash22(i + vec2(0,1)), f - vec2(0,1)),
                   dot(hash22(i + vec2(1,1)), f - vec2(1,1)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
    for (int i = 0; i < 4; i++) {
        v += a * gnoise(p);
        p  = rot * p * 2.0;
        a *= 0.5;
    }
    return v;
}

// ---- Schlick Fresnel -------------------------------------------------------
float fresnelSchlick(float cosTheta) {
    float F0 = 0.02;   // water IOR ≈ 1.33  →  F0 ≈ 0.02
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    if (WaterDepth < 0.03) discard;

    vec3 N = normalize(Normal);
    vec3 L = normalize(lightDir);
    vec3 V = normalize(viewPos - FragPos);
    vec3 H = normalize(L + V);

    // --- Detail normals (small animated ripples) --------------------------
    vec2 uv1 = FragPos.xz * 1.8 + uTime * vec2( 0.04,  0.03);
    vec2 uv2 = FragPos.xz * 3.2 + uTime * vec2(-0.03,  0.05);
    vec3 detailN = vec3(
        gnoise(uv1) * 0.12 + gnoise(uv2) * 0.06,
        1.0,
        gnoise(uv1.yx) * 0.12 + gnoise(uv2.yx) * 0.06
    );
    N = normalize(N + detailN * 0.25);

    // --- Beer's-law absorption (exponential depth-based colouring) --------
    vec3 absorb   = vec3(0.45, 0.08, 0.04);   // absorption coefficients (RGB)
    vec3 scatter   = vec3(0.0, 0.02, 0.03);    // in-scattered light
    vec3 deepColor = exp(-absorb * WaterDepth) + scatter * WaterDepth;

    vec3 shallow = vec3(0.22, 0.65, 0.72);
    vec3 deep    = vec3(0.01, 0.06, 0.22);
    float df     = clamp(WaterDepth / 2.5, 0.0, 1.0);
    vec3 bodyColor = mix(shallow, deep, df) * deepColor;

    // --- Fresnel ------------------------------------------------------------
    float NdotV = max(dot(N, V), 0.0);
    float F     = fresnelSchlick(NdotV);

    // --- Reflection (environment approx) ------------------------------------
    vec3 skyZenith  = vec3(0.30, 0.50, 0.85);
    vec3 skyHorizon = vec3(0.60, 0.75, 0.95);
    vec3 R_dir      = reflect(-V, N);
    float skyFactor = clamp(R_dir.y, 0.0, 1.0);
    vec3 envRefl    = mix(skyHorizon, skyZenith, skyFactor);

    // Sun reflection highlight
    float sunAngle = max(dot(R_dir, L), 0.0);
    vec3  sunSpec  = vec3(1.0, 0.95, 0.85) * pow(sunAngle, 256.0) * 1.5;
    envRefl += sunSpec;

    // --- Blinn-Phong specular -----------------------------------------------
    float NdotH = max(dot(N, H), 0.0);
    float spec  = pow(NdotH, 256.0);
    vec3 specular = vec3(1.0, 0.97, 0.90) * spec * 0.8;

    // --- Subsurface scattering approximation --------------------------------
    float sss = pow(clamp(dot(V, -L), 0.0, 1.0), 4.0);
    vec3 sssColor = vec3(0.1, 0.5, 0.4) * sss * 0.3 * clamp(WaterDepth / 1.5, 0.0, 1.0);

    // --- Diffuse lighting ----------------------------------------------------
    float diff = max(dot(N, L), 0.0);
    vec3 ambient = bodyColor * 0.2;

    // --- Foam (at high-velocity regions: parting edges, wave crests) --------
    float spd   = length(Velocity);
    float foamT = smoothstep(2.5, 6.0, spd);
    // Add noise-based foam breakup
    float foamNoise = fbm(FragPos.xz * 3.0 + uTime * 0.5);
    foamT *= smoothstep(-0.1, 0.4, foamNoise);
    vec3 foamColor = vec3(0.90, 0.95, 1.0);

    // --- Compose -------------------------------------------------------------
    vec3 refracted = ambient + bodyColor * diff * 0.6 + sssColor;
    vec3 waterSurf = mix(refracted, envRefl, F) + specular;
    vec3 result    = mix(waterSurf, foamColor, foamT * 0.7);

    // Alpha: opaque-ish for deep water; foam is always opaque
    float alpha = clamp(WaterDepth / 0.2, 0.45, 0.93);
    alpha = mix(alpha, 1.0, foamT * 0.6);

    FragColor = vec4(result, alpha);
}
)";

// ---- Seabed / ground -------------------------------------------------------
static const char* groundVertSrc = R"(
#version 330 core
layout(location = 0) in vec3  aPos;
layout(location = 1) in vec3  aNormal;
layout(location = 2) in float aHeight;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3  FragPos;
out vec3  Normal;
out float Height;

void main() {
    vec4 wp = model * vec4(aPos, 1.0);
    FragPos = wp.xyz;
    Normal  = mat3(transpose(inverse(model))) * aNormal;
    Height  = aHeight;
    gl_Position = projection * view * wp;
}
)";

static const char* groundFragSrc = R"(
#version 330 core
in vec3  FragPos;
in vec3  Normal;
in float Height;

out vec4 FragColor;

uniform vec3  lightDir;
uniform float uTime;
uniform float uWaterLevel;   // resting water surface Y

// Simple hash for caustics
vec2 hash22(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453);
}
float gnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(dot(hash22(i + vec2(0,0)), f - vec2(0,0)),
                   dot(hash22(i + vec2(1,0)), f - vec2(1,0)), u.x),
               mix(dot(hash22(i + vec2(0,1)), f - vec2(0,1)),
                   dot(hash22(i + vec2(1,1)), f - vec2(1,1)), u.x), u.y);
}
float caustics(vec2 p, float t) {
    float c = 0.0;
    vec2 uv1 = p * 2.5 + t * vec2(0.08, 0.06);
    vec2 uv2 = p * 3.7 + t * vec2(-0.06, 0.09);
    vec2 uv3 = p * 5.1 + t * vec2(0.04, -0.07);
    c += abs(gnoise(uv1));
    c += abs(gnoise(uv2)) * 0.5;
    c += abs(gnoise(uv3)) * 0.25;
    // Sharpen into bright caustic lines
    c = pow(c, 1.5) * 1.5;
    return clamp(c, 0.0, 1.0);
}

void main() {
    // Sandy palette
    vec3 sand    = vec3(0.76, 0.60, 0.42);
    vec3 wetSand = vec3(0.50, 0.38, 0.25);
    float hf     = clamp(Height / 2.0, 0.0, 1.0);
    vec3 base    = mix(wetSand, sand, hf);

    vec3  N   = normalize(Normal);
    vec3  L   = normalize(lightDir);
    float amb = 0.30;
    float dif = max(dot(N, L), 0.0);
    vec3 col = (amb + dif * 0.65) * base;

    // Caustics — only visible when underwater (below water level)
    if (FragPos.y < uWaterLevel - 0.05) {
        float c = caustics(FragPos.xz, uTime);
        // Stronger caustics in shallower water
        float depthBelow = uWaterLevel - FragPos.y;
        float cStrength  = clamp(1.0 - depthBelow / 3.0, 0.0, 1.0) * 0.35;
        col += vec3(0.7, 0.85, 1.0) * c * cStrength * dif;
    }

    FragColor = vec4(col, 1.0);
}
)";

// ============================================================================
//  Shader helpers
// ============================================================================

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    int ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::cerr << "Shader compile error:\n" << log << std::endl;
    }
    return s;
}

static GLuint createProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER,   vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    int ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::cerr << "Program link error:\n" << log << std::endl;
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// ============================================================================
//  Mesh generation from simulation grid
// ============================================================================

// Shared index buffer for an Nx × Nz quad grid (2 triangles per quad)
static void generateIndices(int Nx, int Nz, std::vector<unsigned int>& idx) {
    idx.clear();
    idx.reserve((Nx - 1) * (Nz - 1) * 6);
    for (int j = 0; j < Nz - 1; ++j)
        for (int i = 0; i < Nx - 1; ++i) {
            unsigned tl = j * Nx + i;
            unsigned tr = tl + 1;
            unsigned bl = (j + 1) * Nx + i;
            unsigned br = bl + 1;
            idx.push_back(tl); idx.push_back(bl); idx.push_back(tr);
            idx.push_back(tr); idx.push_back(bl); idx.push_back(br);
        }
}

// Water vertex: pos(3) + normal(3) + depth(1) + velocity(2) = 9 floats
static const int WATER_STRIDE = 9;

static void updateWaterMesh(const ShallowWater& sim, std::vector<float>& v) {
    v.resize(sim.Nx * sim.Nz * WATER_STRIDE);
    for (int j = 0; j < sim.Nz; ++j)
        for (int i = 0; i < sim.Nx; ++i) {
            int base = (j * sim.Nx + i) * WATER_STRIDE;
            float x = sim.worldX(i);
            float y = sim.surfaceHeight(i, j);
            float z = sim.worldZ(j);
            float depth = sim.waterDepth(i, j);

            // Central-difference normal from the height field
            float hL = (i > 0)            ? sim.surfaceHeight(i-1, j) : y;
            float hR = (i < sim.Nx - 1)   ? sim.surfaceHeight(i+1, j) : y;
            float hD = (j > 0)            ? sim.surfaceHeight(i, j-1) : y;
            float hU = (j < sim.Nz - 1)   ? sim.surfaceHeight(i, j+1) : y;
            float nx = (hL - hR) / (2.f * sim.dx);
            float nz = (hD - hU) / (2.f * sim.dz);
            float ny = 1.f;
            float len = std::sqrt(nx*nx + ny*ny + nz*nz);
            nx /= len; ny /= len; nz /= len;

            v[base+0]=x;  v[base+1]=y;  v[base+2]=z;
            v[base+3]=nx; v[base+4]=ny; v[base+5]=nz;
            v[base+6]=depth;
            v[base+7]=sim.velocityX(i, j);
            v[base+8]=sim.velocityZ(i, j);
        }
}

// Ground vertex: pos(3) + normal(3) + height(1)  = 7 floats
static void updateGroundMesh(const ShallowWater& sim, std::vector<float>& v) {
    const int S = 7;
    v.resize(sim.Nx * sim.Nz * S);
    for (int j = 0; j < sim.Nz; ++j)
        for (int i = 0; i < sim.Nx; ++i) {
            int base = (j * sim.Nx + i) * S;
            float x = sim.worldX(i);
            float y = sim.bottomHeight(i, j);
            float z = sim.worldZ(j);

            float bL = (i > 0)            ? sim.bottomHeight(i-1, j) : y;
            float bR = (i < sim.Nx - 1)   ? sim.bottomHeight(i+1, j) : y;
            float bD = (j > 0)            ? sim.bottomHeight(i, j-1) : y;
            float bU = (j < sim.Nz - 1)   ? sim.bottomHeight(i, j+1) : y;
            float nx = (bL - bR) / (2.f * sim.dx);
            float nz = (bD - bU) / (2.f * sim.dz);
            float ny = 1.f;
            float len = std::sqrt(nx*nx + ny*ny + nz*nz);
            nx /= len; ny /= len; nz /= len;

            v[base+0]=x;  v[base+1]=y;  v[base+2]=z;
            v[base+3]=nx; v[base+4]=ny; v[base+5]=nz;
            v[base+6]=y;  // passed to fragment shader for colouring
        }
}

// ============================================================================
//  Water wall / skirt mesh
//
//  Generates vertical quads so the water looks like a solid volume:
//    1) Domain boundary skirts  (4 edges of the grid)
//    2) Interior wet/dry edges  (the parting walls)
//
//  Each vertex: pos(3) + normal(3) + depth(1) + velocity(2) = 9 floats  (same as water)
// ============================================================================

static void pushVert(std::vector<float>& v,
                     float x, float y, float z,
                     float nx, float ny, float nz, float depth) {
    v.push_back(x);  v.push_back(y);  v.push_back(z);
    v.push_back(nx); v.push_back(ny); v.push_back(nz);
    v.push_back(depth);
    v.push_back(0.f); v.push_back(0.f);   // velocity = 0 for wall faces
}

// Add a vertical quad (two triangles).
//   (x0,y_bot,z0)-(x1,y_bot,z1) is the bottom edge
//   (x0,y_top0,z0)-(x1,y_top1,z1) is the top edge
static void addWallQuad(std::vector<float>& v, std::vector<unsigned int>& idx,
                        float x0, float z0, float y_top0, float d0,
                        float x1, float z1, float y_top1, float d1,
                        float nx, float ny, float nz, float y_bot) {
    unsigned base = (unsigned)(v.size() / WATER_STRIDE);
    pushVert(v, x0, y_bot,   z0, nx, ny, nz, d0);  // 0  bottom-left
    pushVert(v, x1, y_bot,   z1, nx, ny, nz, d1);  // 1  bottom-right
    pushVert(v, x1, y_top1,  z1, nx, ny, nz, d1);  // 2  top-right
    pushVert(v, x0, y_top0,  z0, nx, ny, nz, d0);  // 3  top-left
    idx.push_back(base);   idx.push_back(base+1); idx.push_back(base+2);
    idx.push_back(base);   idx.push_back(base+2); idx.push_back(base+3);
}

static void updateWallMesh(const ShallowWater& sim,
                           std::vector<float>& v,
                           std::vector<unsigned int>& idx) {
    v.clear();
    idx.clear();
    const float depthThresh = 0.05f;
    const float baseY = -0.02f;   // slightly below seabed

    // --- 1) Domain boundary skirts ------------------------------------------

    // Left edge (i = 0), normal pointing -X
    for (int j = 0; j < sim.Nz - 1; ++j) {
        float d0 = sim.waterDepth(0, j), d1 = sim.waterDepth(0, j+1);
        if (d0 < depthThresh && d1 < depthThresh) continue;
        addWallQuad(v, idx,
            sim.worldX(0), sim.worldZ(j),   sim.surfaceHeight(0, j),   d0,
            sim.worldX(0), sim.worldZ(j+1), sim.surfaceHeight(0, j+1), d1,
            -1, 0, 0, baseY);
    }
    // Right edge (i = Nx-1), normal pointing +X
    for (int j = 0; j < sim.Nz - 1; ++j) {
        int i = sim.Nx - 1;
        float d0 = sim.waterDepth(i, j), d1 = sim.waterDepth(i, j+1);
        if (d0 < depthThresh && d1 < depthThresh) continue;
        addWallQuad(v, idx,
            sim.worldX(i), sim.worldZ(j+1), sim.surfaceHeight(i, j+1), d1,
            sim.worldX(i), sim.worldZ(j),   sim.surfaceHeight(i, j),   d0,
            1, 0, 0, baseY);
    }
    // Front edge (j = 0), normal pointing -Z
    for (int i = 0; i < sim.Nx - 1; ++i) {
        float d0 = sim.waterDepth(i, 0), d1 = sim.waterDepth(i+1, 0);
        if (d0 < depthThresh && d1 < depthThresh) continue;
        addWallQuad(v, idx,
            sim.worldX(i+1), sim.worldZ(0), sim.surfaceHeight(i+1, 0), d1,
            sim.worldX(i),   sim.worldZ(0), sim.surfaceHeight(i, 0),   d0,
            0, 0, -1, baseY);
    }
    // Back edge (j = Nz-1), normal pointing +Z
    for (int j2 = sim.Nz - 1, i = 0; i < sim.Nx - 1; ++i) {
        float d0 = sim.waterDepth(i, j2), d1 = sim.waterDepth(i+1, j2);
        if (d0 < depthThresh && d1 < depthThresh) continue;
        addWallQuad(v, idx,
            sim.worldX(i),   sim.worldZ(j2), sim.surfaceHeight(i, j2),   d0,
            sim.worldX(i+1), sim.worldZ(j2), sim.surfaceHeight(i+1, j2), d1,
            0, 0, 1, baseY);
    }

    // --- 2) Interior wet ↔ dry transitions (the parting walls) --------------
    // X-direction edges: check neighbouring cells across i
    for (int j = 0; j < sim.Nz; ++j) {
        // Determine z-span for this row's quads
        float zA = sim.worldZ(j) - sim.dz * 0.5f;
        float zB = sim.worldZ(j) + sim.dz * 0.5f;
        if (j == 0)            zA = sim.worldZ(0);
        if (j == sim.Nz - 1)   zB = sim.worldZ(sim.Nz - 1);

        for (int i = 0; i < sim.Nx - 1; ++i) {
            float dL = sim.waterDepth(i, j);
            float dR = sim.waterDepth(i+1, j);
            bool wetL = dL > depthThresh;
            bool wetR = dR > depthThresh;
            if (wetL == wetR) continue;

            // The boundary sits between cell i and cell i+1
            float x = (sim.worldX(i) + sim.worldX(i+1)) * 0.5f;
            if (wetL && !wetR) {
                // Water on the left, dry on the right → wall faces +X
                float h0 = sim.surfaceHeight(i, j);
                addWallQuad(v, idx,
                    x, zA, h0, dL,
                    x, zB, h0, dL,
                    1, 0, 0, baseY);
            } else {
                // Water on the right, dry on the left → wall faces -X
                float h1 = sim.surfaceHeight(i+1, j);
                addWallQuad(v, idx,
                    x, zB, h1, dR,
                    x, zA, h1, dR,
                    -1, 0, 0, baseY);
            }
        }
    }
    // Z-direction edges: check neighbouring cells across j
    for (int i = 0; i < sim.Nx; ++i) {
        float xA = sim.worldX(i) - sim.dx * 0.5f;
        float xB = sim.worldX(i) + sim.dx * 0.5f;
        if (i == 0)            xA = sim.worldX(0);
        if (i == sim.Nx - 1)   xB = sim.worldX(sim.Nx - 1);

        for (int j = 0; j < sim.Nz - 1; ++j) {
            float dD = sim.waterDepth(i, j);
            float dU = sim.waterDepth(i, j+1);
            bool wetD = dD > depthThresh;
            bool wetU = dU > depthThresh;
            if (wetD == wetU) continue;

            float z = (sim.worldZ(j) + sim.worldZ(j+1)) * 0.5f;
            if (wetD && !wetU) {
                float h0 = sim.surfaceHeight(i, j);
                addWallQuad(v, idx,
                    xB, z, h0, dD,
                    xA, z, h0, dD,
                    0, 0, 1, baseY);
            } else {
                float h1 = sim.surfaceHeight(i, j+1);
                addWallQuad(v, idx,
                    xA, z, h1, dU,
                    xB, z, h1, dU,
                    0, 0, -1, baseY);
            }
        }
    }
}

// ============================================================================
//  Application state
// ============================================================================
struct AppState {
    // Camera (orbit)
    float camYaw   = -30.f;     // degrees
    float camPitch =  30.f;
    float camDist  =  18.f;
    glm::vec3 camTarget{0.f, 0.5f, 0.f};

    // Mouse
    bool   mouseDown = false;
    double lastMX = 0, lastMY = 0;

    // Window
    int winW = 1280, winH = 800;

    // Simulation control
    bool  partingActive   = false;
    bool  resetRequested  = false;
    float partingProgress = 0.f;
    float partingSpeed    = 0.1f;   // full parting in ~10 s
    float totalTime       = 0.f;
};
static AppState app;

// ============================================================================
//  GLFW callbacks
// ============================================================================
static void fbCallback(GLFWwindow*, int w, int h) {
    glViewport(0, 0, w, h);
    app.winW = w; app.winH = h;
}
static void keyCallback(GLFWwindow* win, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(win, true);
    if (key == GLFW_KEY_SPACE && !app.partingActive) {
        app.partingActive = true;
        std::cout << "Parting the Red Sea...\n";
    }
    if (key == GLFW_KEY_R) {
        app.resetRequested = true;
    }
}
static void mbCallback(GLFWwindow* win, int btn, int act, int) {
    if (btn == GLFW_MOUSE_BUTTON_LEFT) {
        app.mouseDown = (act == GLFW_PRESS);
        if (app.mouseDown) glfwGetCursorPos(win, &app.lastMX, &app.lastMY);
    }
}
static void cursorCallback(GLFWwindow*, double x, double y) {
    if (!app.mouseDown) return;
    app.camYaw   += (float)(x - app.lastMX) * 0.3f;
    app.camPitch += (float)(y - app.lastMY) * 0.3f;
    app.camPitch  = glm::clamp(app.camPitch, 5.f, 85.f);
    app.lastMX = x; app.lastMY = y;
}
static void scrollCallback(GLFWwindow*, double, double dy) {
    app.camDist -= (float)dy * 1.f;
    app.camDist  = glm::clamp(app.camDist, 5.f, 40.f);
}

// ============================================================================
//  main
// ============================================================================
int main() {
    // --- GLFW init ----------------------------------------------------------
    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return 1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);   // required on macOS
#endif
    glfwWindowHint(GLFW_SAMPLES, 4);                        // 4× MSAA

    GLFWwindow* window = glfwCreateWindow(
        app.winW, app.winH, "Red Sea Parting — Shallow Water Simulation",
        nullptr, nullptr);
    if (!window) { std::cerr << "Window creation failed\n"; glfwTerminate(); return 1; }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);                                    // VSync

    glfwSetFramebufferSizeCallback(window, fbCallback);
    glfwSetKeyCallback            (window, keyCallback);
    glfwSetMouseButtonCallback    (window, mbCallback);
    glfwSetCursorPosCallback      (window, cursorCallback);
    glfwSetScrollCallback         (window, scrollCallback);

    // Handle Retina / HiDPI
    int fbW, fbH;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    glViewport(0, 0, fbW, fbH);
    app.winW = fbW; app.winH = fbH;

    // --- OpenGL global state ------------------------------------------------
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.52f, 0.71f, 0.90f, 1.f);                // sky blue

    std::cout << "GL Version : " << glGetString(GL_VERSION)  << '\n';
    std::cout << "GL Renderer: " << glGetString(GL_RENDERER) << '\n';

    // --- Compile shaders ----------------------------------------------------
    GLuint waterProg  = createProgram(waterVertSrc,  waterFragSrc);
    GLuint groundProg = createProgram(groundVertSrc, groundFragSrc);

    // --- Simulation ---------------------------------------------------------
    constexpr int   GRID = 200;
    constexpr float DOMAIN_SIZE = 20.f;
    constexpr float WATER_H = 2.f;

    ShallowWater sim(GRID, GRID, DOMAIN_SIZE, DOMAIN_SIZE);
    sim.initialize(WATER_H);

    // Shared index buffer
    std::vector<unsigned int> indices;
    generateIndices(GRID, GRID, indices);

    // --- Water mesh (VAO / VBO / EBO) ---------------------------------------
    GLuint wVAO, wVBO, wEBO;
    glGenVertexArrays(1, &wVAO);
    glGenBuffers(1, &wVBO);
    glGenBuffers(1, &wEBO);

    std::vector<float> waterVerts;
    updateWaterMesh(sim, waterVerts);

    glBindVertexArray(wVAO);
    glBindBuffer(GL_ARRAY_BUFFER, wVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(waterVerts.size() * sizeof(float)),
                 waterVerts.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (GLsizeiptr)(indices.size() * sizeof(unsigned)),
                 indices.data(), GL_STATIC_DRAW);

    // layout(location=0) vec3 aPos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, WATER_STRIDE*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // layout(location=1) vec3 aNormal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, WATER_STRIDE*sizeof(float),
                          (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    // layout(location=2) float aDepth
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, WATER_STRIDE*sizeof(float),
                          (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);
    // layout(location=3) vec2 aVelocity
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, WATER_STRIDE*sizeof(float),
                          (void*)(7*sizeof(float)));
    glEnableVertexAttribArray(3);

    // --- Ground mesh (VAO / VBO / EBO) --------------------------------------
    GLuint gVAO, gVBO, gEBO;
    glGenVertexArrays(1, &gVAO);
    glGenBuffers(1, &gVBO);
    glGenBuffers(1, &gEBO);

    std::vector<float> groundVerts;
    updateGroundMesh(sim, groundVerts);

    glBindVertexArray(gVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(groundVerts.size() * sizeof(float)),
                 groundVerts.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (GLsizeiptr)(indices.size() * sizeof(unsigned)),
                 indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7*sizeof(float),
                          (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7*sizeof(float),
                          (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);

    // --- Water wall/skirt mesh (VAO / VBO / EBO) ----------------------------
    GLuint wallVAO, wallVBO, wallEBO;
    glGenVertexArrays(1, &wallVAO);
    glGenBuffers(1, &wallVBO);
    glGenBuffers(1, &wallEBO);

    std::vector<float> wallVerts;
    std::vector<unsigned int> wallIndices;
    updateWallMesh(sim, wallVerts, wallIndices);

    glBindVertexArray(wallVAO);
    glBindBuffer(GL_ARRAY_BUFFER, wallVBO);
    // Pre-allocate generous buffer (will grow as parting creates more walls)
    size_t wallVertCap  = std::max(wallVerts.size(), (size_t)(GRID * 80 * WATER_STRIDE));
    size_t wallIdxCap   = std::max(wallIndices.size(), (size_t)(GRID * 80 * 6));
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(wallVertCap * sizeof(float)),
                 nullptr, GL_DYNAMIC_DRAW);
    if (!wallVerts.empty())
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        (GLsizeiptr)(wallVerts.size() * sizeof(float)),
                        wallVerts.data());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wallEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (GLsizeiptr)(wallIdxCap * sizeof(unsigned)),
                 nullptr, GL_DYNAMIC_DRAW);
    if (!wallIndices.empty())
        glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0,
                        (GLsizeiptr)(wallIndices.size() * sizeof(unsigned)),
                        wallIndices.data());

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, WATER_STRIDE*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, WATER_STRIDE*sizeof(float),
                          (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, WATER_STRIDE*sizeof(float),
                          (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, WATER_STRIDE*sizeof(float),
                          (void*)(7*sizeof(float)));
    glEnableVertexAttribArray(3);

    glBindVertexArray(0);

    // --- Print controls -----------------------------------------------------
    std::cout << "\n=== Red Sea Parting Simulation ===\n"
              << "  SPACE  Start parting\n"
              << "  R      Reset\n"
              << "  Mouse  Drag to orbit camera\n"
              << "  Scroll Zoom in / out\n"
              << "  ESC    Quit\n"
              << "==================================\n\n";

    // --- Main loop ----------------------------------------------------------
    float lastT = (float)glfwGetTime();
    int   frames = 0;
    float fpsAcc = 0.f;

    while (!glfwWindowShouldClose(window)) {
        float now = (float)glfwGetTime();
        float dt  = std::min(now - lastT, 0.05f);   // cap to avoid spiral
        lastT = now;
        app.totalTime += dt;

        // FPS display
        ++frames; fpsAcc += dt;
        if (fpsAcc >= 1.f) {
            std::cout << "FPS: " << frames
                      << "  |  Parting: " << (int)(app.partingProgress * 100) << "%\n";
            frames = 0; fpsAcc = 0.f;
        }

        glfwPollEvents();

        // Handle reset
        if (app.resetRequested) {
            app.resetRequested  = false;
            app.partingActive   = false;
            app.partingProgress = 0.f;
            sim.initialize(WATER_H);
            std::cout << "Reset.  Press SPACE to begin parting.\n";
        }

        // Advance parting
        if (app.partingActive && app.partingProgress < 1.f) {
            app.partingProgress += dt * app.partingSpeed;
            app.partingProgress  = std::min(app.partingProgress, 1.f);
            sim.applyParting(app.partingProgress, dt);
        }

        // Simulation sub-steps (adaptive Δt from CFL condition)
        {
            float remaining = dt;
            for (int s = 0; s < 15 && remaining > 1e-5f; ++s) {
                float sdt = std::min(sim.computeTimeStep(), remaining);
                sim.step(sdt);
                remaining -= sdt;
            }
        }

        // Upload updated meshes to GPU
        updateWaterMesh(sim, waterVerts);
        glBindBuffer(GL_ARRAY_BUFFER, wVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        (GLsizeiptr)(waterVerts.size() * sizeof(float)),
                        waterVerts.data());

        updateGroundMesh(sim, groundVerts);
        glBindBuffer(GL_ARRAY_BUFFER, gVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        (GLsizeiptr)(groundVerts.size() * sizeof(float)),
                        groundVerts.data());

        // Update wall mesh (may grow as parting creates more wet/dry edges)
        updateWallMesh(sim, wallVerts, wallIndices);
        if (!wallVerts.empty()) {
            glBindBuffer(GL_ARRAY_BUFFER, wallVBO);
            if (wallVerts.size() > wallVertCap) {
                wallVertCap = wallVerts.size() * 2;
                glBufferData(GL_ARRAY_BUFFER,
                             (GLsizeiptr)(wallVertCap * sizeof(float)),
                             wallVerts.data(), GL_DYNAMIC_DRAW);
            } else {
                glBufferSubData(GL_ARRAY_BUFFER, 0,
                                (GLsizeiptr)(wallVerts.size() * sizeof(float)),
                                wallVerts.data());
            }
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wallEBO);
            if (wallIndices.size() > wallIdxCap) {
                wallIdxCap = wallIndices.size() * 2;
                glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                             (GLsizeiptr)(wallIdxCap * sizeof(unsigned)),
                             wallIndices.data(), GL_DYNAMIC_DRAW);
            } else {
                glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0,
                                (GLsizeiptr)(wallIndices.size() * sizeof(unsigned)),
                                wallIndices.data());
            }
        }

        // --- Camera ---------------------------------------------------------
        if (!app.mouseDown) app.camYaw += dt * 3.f;  // slow auto-orbit

        float yr = glm::radians(app.camYaw);
        float pr = glm::radians(app.camPitch);
        glm::vec3 camPos = app.camTarget + glm::vec3(
            app.camDist * cosf(pr) * sinf(yr),
            app.camDist * sinf(pr),
            app.camDist * cosf(pr) * cosf(yr));

        glfwGetFramebufferSize(window, &fbW, &fbH);
        float aspect = (float)fbW / std::max(fbH, 1);

        glm::mat4 view  = glm::lookAt(camPos, app.camTarget, {0,1,0});
        glm::mat4 proj  = glm::perspective(glm::radians(45.f), aspect, 0.1f, 100.f);
        glm::mat4 model(1.f);
        glm::vec3 lightDir = glm::normalize(glm::vec3(0.5f, 1.f, 0.3f));

        // --- Render ---------------------------------------------------------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 1) Ground (opaque, drawn first)
        glUseProgram(groundProg);
        glUniformMatrix4fv(glGetUniformLocation(groundProg,"model"),1,GL_FALSE,glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(groundProg,"view"), 1,GL_FALSE,glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(groundProg,"projection"),1,GL_FALSE,glm::value_ptr(proj));
        glUniform3fv(glGetUniformLocation(groundProg,"lightDir"),1,glm::value_ptr(lightDir));
        glUniform1f(glGetUniformLocation(groundProg,"uTime"), app.totalTime);
        glUniform1f(glGetUniformLocation(groundProg,"uWaterLevel"), WATER_H);

        glBindVertexArray(gVAO);
        glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);

        // 2) Water (semi-transparent, drawn second with blending)
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(waterProg);
        glUniformMatrix4fv(glGetUniformLocation(waterProg,"model"),1,GL_FALSE,glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(waterProg,"view"), 1,GL_FALSE,glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(waterProg,"projection"),1,GL_FALSE,glm::value_ptr(proj));
        glUniform3fv(glGetUniformLocation(waterProg,"lightDir"),1,glm::value_ptr(lightDir));
        glUniform3fv(glGetUniformLocation(waterProg,"viewPos"), 1,glm::value_ptr(camPos));
        glUniform1f(glGetUniformLocation(waterProg,"uTime"), app.totalTime);

        glBindVertexArray(wVAO);
        glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);

        // 3) Water walls (same shader, same blending state)
        if (!wallIndices.empty()) {
            glBindVertexArray(wallVAO);
            glDrawElements(GL_TRIANGLES, (GLsizei)wallIndices.size(), GL_UNSIGNED_INT, 0);
        }

        glDisable(GL_BLEND);

        glfwSwapBuffers(window);
    }

    // --- Cleanup ------------------------------------------------------------
    glDeleteVertexArrays(1, &wVAO);
    glDeleteBuffers(1, &wVBO);
    glDeleteBuffers(1, &wEBO);
    glDeleteVertexArrays(1, &gVAO);
    glDeleteBuffers(1, &gVBO);
    glDeleteBuffers(1, &gEBO);
    glDeleteVertexArrays(1, &wallVAO);
    glDeleteBuffers(1, &wallVBO);
    glDeleteBuffers(1, &wallEBO);
    glDeleteProgram(waterProg);
    glDeleteProgram(groundProg);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
