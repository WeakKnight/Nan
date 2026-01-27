static const float PI = 3.14159265358;

// Units are in megameters.
static const float groundRadiusMM = 6.360;
static const float atmosphereRadiusMM = 6.460;

// 200M above the ground.
static const float3 viewPos = float3(0.0, groundRadiusMM + 0.0002, 0.0);

static const float2 tLUTRes = float2(256.0, 64.0);
static const float2 msLUTRes = float2(32.0, 32.0);
// Doubled the vertical skyLUT res from the paper, looks way
// better for sunrise.
static const float2 skyLUTRes = float2(200.0, 200.0);

static const float3 groundAlbedo = float3(0.3);

// These are per megameter.
static const float3 rayleighScatteringBase = float3(5.802, 13.558, 33.1);
static const float rayleighAbsorptionBase = 0.0;

static const float mieScatteringBase = 3.996;
static const float mieAbsorptionBase = 4.4;

static const float3 ozoneAbsorptionBase = float3(0.650, 1.881, .085);



float getMiePhase(float cosTheta) {
    const float g = 0.8;
    const float scale = 3.0/(8.0*PI);
    
    float num = (1.0-g*g)*(1.0+cosTheta*cosTheta);
    float denom = (2.0+g*g)*pow((1.0 + g*g - 2.0*g*cosTheta), 1.5);
    
    return scale*num/denom;
}

float getRayleighPhase(float cosTheta) {
    const float k = 3.0/(16.0*PI);
    return k*(1.0+cosTheta*cosTheta);
}

void getScatteringValues(float3 pos, 
                         out float3 rayleighScattering, 
                         out float mieScattering,
                         out float3 extinction) {
    float altitudeKM = (length(pos) - groundRadiusMM) * 1000.0;
    // Note: Paper gets these switched up.
    float rayleighDensity = exp(-altitudeKM / 8.0);
    float mieDensity = exp(-altitudeKM / 1.2);
    
    rayleighScattering = rayleighScatteringBase * rayleighDensity;
    float rayleighAbsorption = rayleighAbsorptionBase * rayleighDensity;
    
    mieScattering = mieScatteringBase * mieDensity;
    float mieAbsorption = mieAbsorptionBase * mieDensity;
    
    float3 ozoneAbsorption = ozoneAbsorptionBase * max(0.0, 1.0 - abs(altitudeKM - 25.0) / 15.0);
    
    extinction = rayleighScattering + rayleighAbsorption + mieScattering + mieAbsorption + ozoneAbsorption;
}

float safeacos(const float x) {
    return acos(clamp(x, -1.0, 1.0));
}

// From https://gamedev.stackexchange.com/questions/96459/fast-ray-sphere-collision-code.
float rayIntersectSphere(float3 ro, float3 rd, float rad) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - rad * rad;
    if (c > 0.0f && b > 0.0) return -1.0;
    float discr = b * b - c;
    if (discr < 0.0) return -1.0;
    // Special case: inside sphere, use far discriminant
    if (discr > b * b) return (-b + sqrt(discr));
    return -b - sqrt(discr);
}

/*
 * Same parameterization here.
 */
// Sample from transmittance LUT (assumes texture is exactly tLUTRes size)
float3 getValFromTLUT(Texture2D tex, SamplerState samp, float3 pos, float3 sunDir) {
    float height = length(pos);
    float3 up = pos / height;
    float sunCosZenithAngle = dot(sunDir, up);
    
    // Calculate UV in [0,1] range directly
    float2 uv = float2(
        clamp(0.5 + 0.5 * sunCosZenithAngle, 0.0, 1.0),
        clamp((height - groundRadiusMM) / (atmosphereRadiusMM - groundRadiusMM), 0.0, 1.0)
    );
    
    return tex.Sample(samp, uv).rgb;
}

// Sample from multiple scattering LUT (assumes texture is exactly msLUTRes size)
float3 getValFromMultiScattLUT(Texture2D tex, SamplerState samp, float3 pos, float3 sunDir) {
    float height = length(pos);
    float3 up = pos / height;
    float sunCosZenithAngle = dot(sunDir, up);
    
    // Calculate UV in [0,1] range directly
    float2 uv = float2(
        clamp(0.5 + 0.5 * sunCosZenithAngle, 0.0, 1.0),
        clamp((height - groundRadiusMM) / (atmosphereRadiusMM - groundRadiusMM), 0.0, 1.0)
    );
    
    return tex.Sample(samp, uv).rgb;
}

// Buffer A generates the Transmittance LUT. Each pixel coordinate corresponds to a height and sun zenith angle, and
// the value is the transmittance from that point to sun, through the atmosphere.
static const float sunTransmittanceSteps = 40.0;

float3 getSunTransmittance(float3 pos, float3 sunDir) {
    if (rayIntersectSphere(pos, sunDir, groundRadiusMM) > 0.0) {
        return float3(0.0);
    }

    float atmoDist = rayIntersectSphere(pos, sunDir, atmosphereRadiusMM);
    float t = 0.0;

    float3 transmittance = float3(1.0);
    for (float i = 0.0; i < sunTransmittanceSteps; i += 1.0) {
        float newT = ((i + 0.3) / sunTransmittanceSteps) * atmoDist;
        float dt = newT - t;
        t = newT;

        float3 newPos = pos + t * sunDir;

        float3 rayleighScattering, extinction;
        float mieScattering;
        getScatteringValues(newPos, rayleighScattering, mieScattering, extinction);

        transmittance *= exp(-dt * extinction);
    }
    return transmittance;
}

// 256 x 64, float4
RWTexture2D<float4> g_transmittance_lut;

[[shader("compute")]]
[[numthreads(8, 8, 1)]]
void transmittance_lut_compute(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (any(pixel >= uint2((uint)tLUTRes.x, (uint)tLUTRes.y)))
    {
        return;
    }

    float2 uv = (pixel + 0.5) / tLUTRes;
    float u = uv.x;
    float v = uv.y;
    float sunCosTheta = 2.0 * u - 1.0;
    float sunTheta = safeacos(sunCosTheta);
    float height = lerp(groundRadiusMM, atmosphereRadiusMM, v);
    float3 pos = float3(0.0, height, 0.0);
    float3 sunDir = normalize(float3(0.0, sunCosTheta, -sin(sunTheta)));
    float3 transmittance = getSunTransmittance(pos, sunDir);
    g_transmittance_lut[pixel] = float4(transmittance, 1.0);
}

// Buffer B is the multiple-scattering LUT. Each pixel coordinate corresponds to a height and sun zenith angle, and
// the value is the multiple scattering approximation (Psi_ms from the paper, Eq. 10).
static const float mulScattSteps = 20.0;
static const int sqrtSamples = 8;

float3 getSphericalDir(float theta, float phi) {
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    return float3(sinPhi * sinTheta, cosPhi, sinPhi * cosTheta);
}

// Calculates Equation (5) and (7) from the paper.
void getMulScattValues(float3 pos, float3 sunDir, Texture2D transmittanceLUT, SamplerState samp, out float3 lumTotal, out float3 fms) {
    lumTotal = float3(0.0);
    fms = float3(0.0);

    float invSamples = 1.0 / float(sqrtSamples * sqrtSamples);
    for (int i = 0; i < sqrtSamples; i++) {
        for (int j = 0; j < sqrtSamples; j++) {
            // This integral is symmetric about theta = 0 (or theta = PI), so we
            // only need to integrate from zero to PI, not zero to 2*PI.
            float theta = PI * (float(i) + 0.5) / float(sqrtSamples);
            float phi = safeacos(1.0 - 2.0 * (float(j) + 0.5) / float(sqrtSamples));
            float3 rayDir = getSphericalDir(theta, phi);

            float atmoDist = rayIntersectSphere(pos, rayDir, atmosphereRadiusMM);
            float groundDist = rayIntersectSphere(pos, rayDir, groundRadiusMM);
            float tMax = atmoDist;
            if (groundDist > 0.0) {
                tMax = groundDist;
            }

            float cosTheta = dot(rayDir, sunDir);

            float miePhaseValue = getMiePhase(cosTheta);
            float rayleighPhaseValue = getRayleighPhase(-cosTheta);

            float3 lum = float3(0.0), lumFactor = float3(0.0), transmittance = float3(1.0);
            float t = 0.0;
            for (float stepI = 0.0; stepI < mulScattSteps; stepI += 1.0) {
                float newT = ((stepI + 0.3) / mulScattSteps) * tMax;
                float dt = newT - t;
                t = newT;

                float3 newPos = pos + t * rayDir;

                float3 rayleighScattering, extinction;
                float mieScattering;
                getScatteringValues(newPos, rayleighScattering, mieScattering, extinction);

                float3 sampleTransmittance = exp(-dt * extinction);

                // Integrate within each segment.
                float3 scatteringNoPhase = rayleighScattering + mieScattering;
                float3 scatteringF = (scatteringNoPhase - scatteringNoPhase * sampleTransmittance) / extinction;
                lumFactor += transmittance * scatteringF;

                // This is slightly different from the paper, but I think the paper has a mistake?
                // In equation (6), I think S(x,w_s) should be S(x-tv,w_s).
                float3 sunTransmittance = getValFromTLUT(transmittanceLUT, samp, newPos, sunDir);

                float3 rayleighInScattering = rayleighScattering * rayleighPhaseValue;
                float mieInScattering = mieScattering * miePhaseValue;
                float3 inScattering = (rayleighInScattering + mieInScattering) * sunTransmittance;

                // Integrated scattering within path segment.
                float3 scatteringIntegral = (inScattering - inScattering * sampleTransmittance) / extinction;

                lum += scatteringIntegral * transmittance;
                transmittance *= sampleTransmittance;
            }

            if (groundDist > 0.0) {
                float3 hitPos = pos + groundDist * rayDir;
                if (dot(pos, sunDir) > 0.0) {
                    hitPos = normalize(hitPos) * groundRadiusMM;
                    lum += transmittance * groundAlbedo * getValFromTLUT(transmittanceLUT, samp, hitPos, sunDir);
                }
            }

            fms += lumFactor * invSamples;
            lumTotal += lum * invSamples;
        }
    }
}

// 32 x 32, float4
Texture2D g_transmittance_lut_input;
SamplerState g_sampler;
RWTexture2D<float4> g_multiscatt_lut;

[[shader("compute")]]
[[numthreads(8, 8, 1)]]
void multiscatt_lut_compute(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (any(pixel >= uint2((uint)msLUTRes.x, (uint)msLUTRes.y)))
    {
        return;
    }

    float2 uv = (pixel + 0.5) / msLUTRes;
    float u = uv.x;
    float v = uv.y;
    
    float sunCosTheta = 2.0 * u - 1.0;
    float sunTheta = safeacos(sunCosTheta);
    float height = lerp(groundRadiusMM, atmosphereRadiusMM, v);
    
    float3 pos = float3(0.0, height, 0.0);
    float3 sunDir = normalize(float3(0.0, sunCosTheta, -sin(sunTheta)));
    
    float3 lum, f_ms;
    getMulScattValues(pos, sunDir, g_transmittance_lut_input, g_sampler, lum, f_ms);
    
    // Equation 10 from the paper.
    float3 psi = lum / (1.0 - f_ms);
    g_multiscatt_lut[pixel] = float4(psi, 1.0);
}

// Buffer C calculates the actual sky-view! It's a lat-long map (or maybe altitude-azimuth is the better term),
// but the latitude/altitude is non-linear to get more resolution near the horizon.
static const int numScatteringSteps = 32;

float3 raymarchScattering(
    float3 pos,
    float3 rayDir,
    float3 sunDir,
    float tMax,
    float numSteps,
    Texture2D transmittanceLUT,
    Texture2D multiscattLUT,
    SamplerState samp)
{
    float cosTheta = dot(rayDir, sunDir);

    float miePhaseValue = getMiePhase(cosTheta);
    float rayleighPhaseValue = getRayleighPhase(-cosTheta);

    float3 lum = float3(0.0);
    float3 transmittance = float3(1.0);
    float t = 0.0;
    for (float i = 0.0; i < numSteps; i += 1.0) {
        float newT = ((i + 0.3) / numSteps) * tMax;
        float dt = newT - t;
        t = newT;

        float3 newPos = pos + t * rayDir;

        float3 rayleighScattering, extinction;
        float mieScattering;
        getScatteringValues(newPos, rayleighScattering, mieScattering, extinction);

        float3 sampleTransmittance = exp(-dt * extinction);

        float3 sunTransmittance = getValFromTLUT(transmittanceLUT, samp, newPos, sunDir);
        float3 psiMS = getValFromMultiScattLUT(multiscattLUT, samp, newPos, sunDir);

        float3 rayleighInScattering = rayleighScattering * (rayleighPhaseValue * sunTransmittance + psiMS);
        float3 mieInScattering = mieScattering * (miePhaseValue * sunTransmittance + psiMS);
        float3 inScattering = (rayleighInScattering + mieInScattering);

        // Integrated scattering within path segment.
        float3 scatteringIntegral = (inScattering - inScattering * sampleTransmittance) / extinction;

        lum += scatteringIntegral * transmittance;

        transmittance *= sampleTransmittance;
    }
    return lum;
}

// 200 x 200, float4
Texture2D g_transmittance_lut_input_sky;
Texture2D g_multiscatt_lut_input;
SamplerState g_sampler_sky;
RWTexture2D<float4> g_sky_view_lut;

// Uniform parameters - sun direction directly passed from CPU
uniform float3 g_sun_direction;

[[shader("compute")]]
[[numthreads(8, 8, 1)]]
void sky_view_lut_compute(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (any(pixel >= uint2((uint)skyLUTRes.x, (uint)skyLUTRes.y)))
    {
        return;
    }

    float2 uv = (pixel + 0.5) / skyLUTRes;
    float u = uv.x;
    float v = uv.y;

    float azimuthAngle = (u - 0.5) * 2.0 * PI;
    
    // Non-linear mapping of altitude. See Section 5.3 of the paper.
    float adjV;
    if (v < 0.5) {
        float coord = 1.0 - 2.0 * v;
        adjV = -coord * coord;
    } else {
        float coord = v * 2.0 - 1.0;
        adjV = coord * coord;
    }

    float height = length(viewPos);
    float3 up = viewPos / height;
    float horizonAngle = safeacos(sqrt(height * height - groundRadiusMM * groundRadiusMM) / height) - 0.5 * PI;
    float altitudeAngle = adjV * 0.5 * PI - horizonAngle;

    float cosAltitude = cos(altitudeAngle);
    float3 rayDir = float3(cosAltitude * sin(azimuthAngle), sin(altitudeAngle), -cosAltitude * cos(azimuthAngle));

    // Use sun direction directly from uniform (already in Y-up coordinate system)
    float sunAltitude = (0.5 * PI) - acos(dot(g_sun_direction, up));
    float3 sunDir = float3(0.0, sin(sunAltitude), -cos(sunAltitude));

    float atmoDist = rayIntersectSphere(viewPos, rayDir, atmosphereRadiusMM);
    float groundDist = rayIntersectSphere(viewPos, rayDir, groundRadiusMM);
    float tMax = (groundDist < 0.0) ? atmoDist : groundDist;
    
    float3 lum = raymarchScattering(
        viewPos, rayDir, sunDir, tMax, float(numScatteringSteps),
        g_transmittance_lut_input_sky, g_multiscatt_lut_input, g_sampler_sky
    );
    
    g_sky_view_lut[pixel] = float4(lum, 1.0);
}