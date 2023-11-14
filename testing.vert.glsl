#version 450

layout (location = 0) in vec3 input_position;
layout (location = 1) in vec3 input_color;

layout (location = 0) out vec4 outColor;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main() 
{
   gl_Position = /* ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * */ vec4(input_position.xyz, 1.0);
   outColor = vec4(input_color.xyz, 1.0);
}
