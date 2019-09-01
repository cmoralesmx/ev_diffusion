#version 330 core
layout (location=0) in vec2 position;
out vec4 colour;
uniform mat4 mvp;
void main()
{
   gl_Position = mvp * vec4(position, 0.0, 1.0);
	colour = vec4(0., 0., 1., 1.);
}