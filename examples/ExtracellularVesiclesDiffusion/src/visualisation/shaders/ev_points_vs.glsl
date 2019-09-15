#version 330 core
layout (location=0) in vec2 position;
layout (location=1) in float radius;
out vec4 colour;
uniform mat4 mvp;
void main()
{
	// radius values in range 0.04-0.16 um
	gl_PointSize = radius * 80;
	if (radius > 0.14)
		colour = vec4(0., 0.59, 0.53, 1.0);
	else if (radius > 0.12)
		colour = vec4(0.9, 0.12, 0.39, 1.0);
	else if (radius > 0.1)
		colour = vec4(0.61, 0.15, 0.69, 1.0);
	else if (radius > 0.08)
		colour = vec4(0.1, 0.66, 0.96, 1.0);
	else if (radius > 0.06)
		colour = vec4(1.0, 0.92, 0.23, 1.0);
	else //if (radius > 0.04)
		colour = vec4(0.5, 0.2, 0.0, 1.0);
	/*else
		colour = vec4(0.37, 0.49, 0.55, 1.0);*/

    gl_Position = mvp * vec4(position, 0, 1);
}