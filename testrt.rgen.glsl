#version 460
#extension GL_EXT_ray_tracing : enable

layout(binding = 0, set = 0) uniform accelerationStructureEXT TLAS;
layout(binding = 1, set = 0, rgba8) uniform image2D image;

layout(binding = 2, set = 0) uniform Camera
{
    mat4 viewInverse;
    mat4 projInverse;
} cam;

layout(location = 0) rayPayloadEXT vec3 hitValue;

// Adapted from Sascha's rgen
void main()
{
    const vec2 pixel_center = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 uv = pixel_center / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = uv * 2.0 - 1.0;

#if 1
    vec4 origin = cam.viewInverse * vec4(0,0,0,1);
    vec4 target = cam.projInverse * vec4(d.x, d.y, 1, 1) ;
    vec4 direction = cam.viewInverse * vec4(normalize(target.xyz), 0) ;
#elif 0
    vec4 origin = vec4(0, 0, 5, 1);
    vec4 direction = vec4(normalize(vec3(d.x, d.y, -1)), 0);
#elif 0
    float a = (uv.y * 2.0 - 1.0) * 3.14159 / 2;
    float b = uv.x * 3.14159 * 2;
    vec4 origin = cam.viewInverse * vec4(0, 0, 5, 1);
    vec4 direction = cam.viewInverse * vec4(normalize(vec3(cos(a) * sin(b), sin(a), cos(a) * cos(b))), 0);
#else
    vec4 origin = vec4(d * 10, 100, 1);
    vec4 direction = vec4(0, 0, -1, 0);
#endif

    float tmin = 0.001;
    float tmax = 10000.0;

    hitValue = vec3(1, 0, 0); // vec3(0.0);

    traceRayEXT(TLAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(origin), tmin, vec3(direction), tmax, 0);

    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(hitValue, 0.0));
}
