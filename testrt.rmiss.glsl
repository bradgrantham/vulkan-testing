#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT vec3 hitValue;

void main()
{
    vec3 result_color = vec3(.1, .1, .2);

    hitValue = result_color;
}


