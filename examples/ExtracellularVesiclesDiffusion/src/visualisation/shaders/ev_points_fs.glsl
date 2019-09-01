#version 330 core
in vec4 colour;
out vec4 FragColor;
void main ()
{
	vec4 ambientColor = colour;
	vec4 diffuseColor = vec4(0.0, 0.0, 0.25, 1.0);

	FragColor = ambientColor + diffuseColor;
}