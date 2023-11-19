#version 450

vec3 positions[3] = vec3[](
    vec3(-0.5, 0.5, 0.5),
    vec3(-0.5, -0.5, 0.5),
    vec3(0.5, -0.5, 0.5)
);

layout (location = 0) in vec3 input_position;
layout (location = 1) in vec3 input_color;

layout (location = 0) out vec4 outColor;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main() 
{
   gl_Position = vec4(positions[gl_VertexIndex], 1.0); 
   // gl_Position =  /* ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * */ vec4(input_position.xyz, 1.0);
   outColor = vec4(input_color.xyz, 1.0);
}
