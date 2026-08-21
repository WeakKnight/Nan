import common;

static const uint SURFACE_PROBE_SH_L2_COEFFICIENT_COUNT = 9u;

float surface_probe_sh_l2_basis(uint coefficient, float3 direction)
{
    float3 d = normalize(direction);
    float x = d.x;
    float y = d.y;
    float z = d.z;
    if (coefficient == 0u) return 0.2820947918;
    if (coefficient == 1u) return 0.4886025119 * y;
    if (coefficient == 2u) return 0.4886025119 * z;
    if (coefficient == 3u) return 0.4886025119 * x;
    if (coefficient == 4u) return 1.0925484306 * x * y;
    if (coefficient == 5u) return 1.0925484306 * y * z;
    if (coefficient == 6u)
        return 0.3153915653 * (3.0 * y * y - 1.0);
    if (coefficient == 7u) return 1.0925484306 * x * z;
    return 0.5462742153 * (x * x - z * z);
}

