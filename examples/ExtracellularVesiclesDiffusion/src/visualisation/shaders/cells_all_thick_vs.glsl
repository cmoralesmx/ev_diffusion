#version 330 core
layout (location=0) in vec2 position;
layout (location=1) in vec2 unit_normal;

out vData {
   vec2 unit_normal;
} vertex;

void main()
{					
   gl_Position = vec4(position, 0.0, 1.0);
   vertex.unit_normal = unit_normal;
}