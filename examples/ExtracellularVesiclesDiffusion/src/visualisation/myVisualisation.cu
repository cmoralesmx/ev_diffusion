/*
* FLAME GPU v 1.5.X for CUDA 9
* Copyright University of Sheffield.
* Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
* Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
*
* University of Sheffield retain all intellectual property and
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from
* University of Sheffield is strictly prohibited.
*
* For terms of licence agreement please attached licence or view licence
* on www.flamegpu.com website.
*
*/

// includes, project
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>

#include <GL/glew.h>
//#include <GL/glut.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include "header.h"
#include "visualisation.h"
#define __MYVISUAL
#include "myVisualisation.h"

extern float *h_xs;
extern float *h_ys;
extern float *h_angles;
#define FOVY 45.0

// bo variables
GLuint sphereVerts;
GLuint sphereNormals;
GLuint wallVerts;
GLuint wallNormals;

//Simulation output buffers/textures
cudaGraphicsResource_t EV_default_cgr;
GLuint EV_default_tbo;
GLuint EV_default_displacementTex;

cudaGraphicsResource_t CiliaryCell_c_default_cgr;
GLuint CiliaryCell_default_tbo;
GLuint CiliaryCell_default_displacementTex;

cudaGraphicsResource_t SecretoryCell_s_default_cgr;
GLuint SecretoryCell_default_tbo;
GLuint SecretoryCell_default_displacementTex;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -VIEW_DISTANCE;  // This parameter comes from the visualisation.h file

// camera movement
float offset_x = 0, offset_y = 0;

// keyboard controls
#if defined(PAUSE_ON_START)
bool paused = true;
#else
bool paused = false;
#endif

// vertex Shader
GLuint vertexShader;
GLuint fragmentShader;
GLuint shaderProgram;
//GLuint vs_displacementMap;
GLuint vs_mapIndex;

GLuint walls_vertexShader;
GLuint walls_fragmentShader;
GLuint walls_shaderProgram;
//GLuint walls_vs_displacementMap;
GLuint walls_vs_mapIndex;

//timer
cudaEvent_t start, stop;
const int display_rate = 50;
int frame_count;
float frame_time = 0.0;

#ifdef SIMULATION_DELAY
//delay
int delay_count = 0;
#endif

// prototypes
int initGL();
void initShader();
void createVBO(GLuint* vbo, GLuint size);
void deleteVBO(GLuint* vbo);
//void createTBO(GLuint* tbo, GLuint* tex, GLuint size);
//void deleteTBO(GLuint* tbo);
void createTBO(cudaGraphicsResource_t* cudaResource, GLuint* tbo, GLuint* tex, GLuint size);
void deleteTBO(cudaGraphicsResource_t* cudaResource, GLuint* tbo);
void setVertexBufferData();
void reshape(int width, int height);
void display();
void close();
void keyboard(unsigned char key, int x, int y);
void special(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void runCuda();
void checkGLError();

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort = true)
{
	gpuAssert(cudaPeekAtLastError(), file, line);
#ifdef _DEBUG
	gpuAssert(cudaDeviceSynchronize(), file, line);
#endif

}

/**
This is an OpenGL 3

uses an uniform of type samplerBuffer, which is defined by the GL_EXT_gpu_shader4 extension
Inputs:
float mapIndex - specifies the target element to read from the buffer
Ouputs:
vec3 normal
vec3 lightDir
vec4 colour

This shader creates a sampler type (samplerBuffer) for buffer textures
and adds a lookup function (texelFetchBuffer) to explicitly access the
buffer texture ussing texture hardware

store in 'position' the value of gl_Vertex
store in 'lookup' the value at 'mapIndex' stored in 'displacementMap'

Then, the value in lookup.w determines the colour to use for the diffuse colour,
which is the colour reflected from the object

The first 3 elements (lookup.xyz) affect the positional data obtained from gl_Vertex
*/
const char vertexShaderSource[] =
{
	"#extension GL_EXT_gpu_shader4 : enable										\n"
	"uniform samplerBuffer displacementMap;										\n"
	"attribute in float mapIndex;												\n"
	"varying vec3 normal, lightDir;												\n"
	"varying vec4 colour;														\n"
	"void main()																\n"
	"{																			\n"
	"	vec4 position = gl_Vertex;											    \n"
	"	vec4 lookup = texelFetchBuffer(displacementMap, (int)mapIndex);		    \n"
	"	if (lookup.w > 0.0664)	                									\n"
	"		colour = vec4(0., 0.59, 0.53, 1.0);						    		\n" // teal (0,150,133)
	"	else if (lookup.w > 0.0579)	               								\n"
	"		colour = vec4(0.9, 0.12, 0.39, 1.0);							    \n" // pink (229,30,99)
	"	else if (lookup.w > 0.0493)	                							\n"
	"		colour = vec4(0.61, 0.15, 0.69, 1.0);							    \n" // purple (155,38,175)
	"	else if (lookup.w > 0.0407)	                							\n"
	"		colour = vec4(0.1, 0.66, 0.96, 1.0);							    \n"	// light blue (25,168,244)
	"	else if (lookup.w > 0.0321)	                							\n"
	"		colour = vec4(1.0, 0.92, 0.23, 1.0);								\n" // yellow (255,234,58)
	"	else if (lookup.w > 0.0236)	                							\n"
	"		colour = vec4(0.5, 0.2, 0.0, 1.0);								\n" // brown (128, 51, 0)
	"	else                      	                							\n"
	"		colour = vec4(0.37, 0.49, 0.55, 1.0);								\n" // blue grey (91, 124, 140)
	"																    		\n"
	"	lookup.w = 1.0;												    		\n"
	"	position += lookup;											    		\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * position;		    		\n" // transform the scene coordinates in position -> screen coordinates in gl_Position
	"																			\n"
	"	vec3 mvVertex = vec3(gl_ModelViewMatrix * position);			    	\n" // intersects the gl_ModelViewMatrix and position
	"	lightDir = vec3(gl_LightSource[0].position.xyz - mvVertex);				\n"	// obtains lightning directional info for the coordinates obtained by the previous intersection
	"	normal = gl_NormalMatrix * gl_Normal;									\n" // intersects both normal matrices??
	"}																			\n"
};

const char vertexShaderSource2[] =
{
	"#extension GL_EXT_gpu_shader4 : enable										\n"
	"uniform samplerBuffer displacementMap;										\n"
	"attribute in float mapIndex;												\n"
	"varying vec3 normal, lightDir;												\n"
	"varying vec4 colour;														\n"
	"void main()																\n"
	"{																			\n"
	"	vec4 lookup = texelFetchBuffer(displacementMap, (int)mapIndex);		    \n"
	"   mat4 A = mat4(mat3(mat2(cos(lookup.w), sin(lookup.w), -sin(lookup.w), cos(lookup.w)))); \n"
	"   colour = vec4(0., 0., 0., 1.0);										\n" // ciliary cells - black
	"	vec4 position = A* gl_Vertex;											\n"
	"	lookup.w = 1.0;												    		\n"
	"	position += lookup;									    		\n"
	"																			\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * position;		    		\n" // transform the scene coordinates in position -> screen coordinates in gl_Position
	"																			\n"
	"	vec3 mvVertex = vec3(gl_ModelViewMatrix * position);			    	\n" // intersects the gl_ModelViewMatrix and position
	"	lightDir = vec3(gl_LightSource[0].position.xyz - mvVertex);				\n"	// obtains lightning directional info for the coordinates obtained by the previous intersection
	"	normal = gl_NormalMatrix * gl_Normal;									\n" // intersects both normal matrices??
	"}																			\n"
};
/*
A nice explanation of how shadding works is available at
https://learnopengl.com/Lighting/Basic-Lighting
http://www.opengl-tutorial.org/beginners-tutorials/tutorial-8-basic-shading/
*/
const char fragmentShaderSource[] =
{
	"varying vec3 normal, lightDir;												\n"
	"varying vec4 colour;														\n"
	"void main (void)															\n"
	"{																			\n"
	"	// Defining The Material Colors											\n"
	"	vec4 AmbientColor = colour;												\n"
	"	vec4 DiffuseColor = vec4(0.0, 0.0, 0.25, 1.0);	           		    	\n"
	"																			\n"
	"	// Scaling The Input Vector To Length 1									\n"
	"	vec3 n_normal = normalize(normal);							        	\n"
	"	vec3 n_lightDir = normalize(lightDir);	                                \n"
	"																			\n"
	"	// Calculating The Diffuse Term And Clamping It To [0;1]				\n"
	"	float DiffuseTerm = clamp(dot(n_normal, n_lightDir), 0.0, 1.0);\n"
	"																			\n"
	"	// Calculating The Final Color											\n"
	"	gl_FragColor = AmbientColor + DiffuseColor * DiffuseTerm;				\n"
	"																			\n"
	"}																			\n"
};
const char fragmentShaderSource2[] =
{
	"varying vec3 normal, lightDir;												\n"
	"varying vec4 colour;														\n"
	"void main (void)															\n"
	"{																			\n"
	"	// Defining The Material Colors											\n"
	"	vec4 AmbientColor = colour;												\n"
	"	vec4 DiffuseColor = vec4(0.0, 0.0, 0.25, 1.0);	           		    	\n"
	"																			\n"
	"	// Scaling The Input Vector To Length 1									\n"
	"	vec3 n_normal = normalize(normal);							        	\n"
	"	vec3 n_lightDir = normalize(lightDir);	                                \n"
	"																			\n"
	"	// Calculating The Diffuse Term And Clamping It To [0;1]				\n"
	"	float DiffuseTerm = clamp(dot(n_normal, n_lightDir), 0.0, 1.0);\n"
	"																			\n"
	"	// Calculating The Final Color											\n"
	"	gl_FragColor = AmbientColor + DiffuseColor * DiffuseTerm;				\n"
	"																			\n"
	"}																			\n"
};

//GPU Kernels
/**
The screen X,Y values are computer here for each agent
float3 centralize provides the pre-computed coordinates for the absolute center of the screen.
Then, each agent's value is set in relation to the computed centre.

*/
__global__ void output_EV_agent_to_VBO(xmachine_memory_EV_list* agents, glm::vec4* vbo, glm::vec3 centralise, float offset_x, float offset_y){

	//global thread index
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	vbo[index].x = offset_x; //0.0;
	vbo[index].y = offset_y; // 0.0;
	vbo[index].z = 0;

	vbo[index].x += agents->x[index] - centralise.x;
	vbo[index].y += agents->y[index] - centralise.y;
	vbo[index].z = agents->z[index] - centralise.z;
	//vbo[index].w = ((agents->radius_um[index] * 2000.f) / 300.f) * 8.f;  // agents->colour[index];
	vbo[index].w = agents->radius_um[index];// *53.333333333333336f;
}

__global__ void output_CiliaryCell_agent_to_VBO(xmachine_memory_CiliaryCell_list* agents,
	glm::vec4* vbo, glm::vec3 centralise, float offset_x, float offset_y) {

	//global thread index
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	//int index_2 = index * 2;

	vbo[index].x = offset_x; //0.0;
	vbo[index].y = offset_y; // 0.0;
	vbo[index].z = 0; // 0.0;
	


	vbo[index].x += agents->x[index] - centralise.x;
	vbo[index].y += agents->y[index] - centralise.y;
	vbo[index].z = agents->z[index] - centralise.z;
	vbo[index].w = agents->angle[index]; // agents->ra[index];
}

__global__ void output_SecretoryCell_agent_to_VBO(xmachine_memory_SecretoryCell_list* agents,
	glm::vec4* vbo, glm::vec3 centralise, float offset_x, float offset_y) {

	//global thread index
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	//int index_2 = index * 2;

	vbo[index].x = offset_x; //0.0;
	vbo[index].y = offset_y; // 0.0;
	vbo[index].z = 0; // 0.0;

	vbo[index].x += agents->x[index] - centralise.x;
	vbo[index].y += agents->y[index] - centralise.y;
	vbo[index].z = agents->z[index] - centralise.z;
	vbo[index].w = agents->angle[index]; // agents->ra[index];
}

void initVisualisation()
{
	// Create GL context
	int   argc = 1;
	char glutString[] = "GLUT application";
	char *argv[] = { glutString, NULL };
	//char *argv[] = {"GLUT application", NULL};
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow("FLAME GPU Visualiser");

	// initialize GL
	if (!initGL()) {
		return;
	}
	initShader();

	// register callbacks
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutCloseFunc(close);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	// create VBO's
	// CAMG allocate enough buffer space to contain the vertices of a single sphere only
	createVBO(&sphereVerts, SPHERE_SLICES* (SPHERE_STACKS + 1) * sizeof(glm::vec3));
	// CAMG allocate enough buffer space to contain the normals of a single sphere only
	createVBO(&sphereNormals, SPHERE_SLICES* (SPHERE_STACKS + 1) * sizeof(glm::vec3));
	// CAMG write the vertices and normals for a sinlge sphere
	// Should this data be written per agent? because we want to represent the size of each agent.
	// It seems 'cloning' is offloaded to the geometry shader, which would draw an instance of the sphere per point particle???
	createVBO(&wallVerts, 12 * sizeof(glm::vec3));
	createVBO(&wallNormals, 12 * sizeof(glm::vec3));
	

	// write the default vertices and normals for one sphere in sphereVerts and sphereNormals respectively
	setVertexBufferData();

	// create TBO - This does the CUDA - 
	createTBO(&EV_default_cgr, &EV_default_tbo, &EV_default_displacementTex, xmachine_memory_EV_MAX * sizeof(glm::vec4));
	createTBO(&CiliaryCell_c_default_cgr, &CiliaryCell_default_tbo, &CiliaryCell_default_displacementTex, xmachine_memory_CiliaryCell_MAX * sizeof(glm::vec4));
	createTBO(&SecretoryCell_s_default_cgr, &SecretoryCell_default_tbo, &SecretoryCell_default_displacementTex, xmachine_memory_SecretoryCell_MAX * sizeof(glm::vec4));

	//set shader uniforms
	glUseProgram(shaderProgram);

	//create a events for timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

void runVisualisation(){
	// start rendering mainloop
	glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
	if (!paused){
#ifdef SIMULATION_DELAY
		delay_count++;
		if (delay_count == SIMULATION_DELAY){
			delay_count = 0;
			singleIteration();
		}
#else
		singleIteration();
#endif
	}

	//kernals sizes
	int threads_per_tile = 256;
	int tile_size;
	dim3 grid;
	dim3 threads;
	glm::vec3 centralise;

	//pointer
	glm::vec4 *dptr, *dptr2, *dptr3;


	if (get_agent_EV_default_count() > 0)
	{
		size_t accessibleBufferSize = 0;
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGraphicsMapResources(1, &EV_default_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &accessibleBufferSize, EV_default_cgr));
		//cuda block size
		tile_size = (int)ceil((float)get_agent_EV_default_count() / threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

		centralise = getMaximumBounds() + getMinimumBounds();

		centralise /= 2;
		//printf("centralise %f %f %f", centralise.x, centralise.y, centralise.z);


		output_EV_agent_to_VBO << < grid, threads >> >(get_device_EV_default_agents(), dptr, centralise, offset_x, offset_y);
		gpuErrchkLaunch();
		// unmap buffer object
		gpuErrchk(cudaGraphicsUnmapResources(1, &EV_default_cgr));
	}

	if (get_agent_CiliaryCell_c_default_count() > 0)
	{
		size_t accessibleBufferSize = 0;
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGraphicsMapResources(1, &CiliaryCell_c_default_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr2, &accessibleBufferSize, CiliaryCell_c_default_cgr));
		//cuda block size
		tile_size = (int)ceil((float)get_agent_CiliaryCell_c_default_count() / threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

		centralise = getMaximumBounds() + getMinimumBounds();

		centralise /= 2;
		//printf("centralise %f %f %f", centralise.x, centralise.y, centralise.z);


		output_CiliaryCell_agent_to_VBO << < grid, threads >> >(get_device_CiliaryCell_c_default_agents(), dptr2, centralise, offset_x, offset_y);
		gpuErrchkLaunch();
		// unmap buffer object
		gpuErrchk(cudaGraphicsUnmapResources(1, &CiliaryCell_c_default_cgr));
	}

	if (get_agent_SecretoryCell_s_default_count() > 0)
	{
		size_t accessibleBufferSize = 0;
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGraphicsMapResources(1, &SecretoryCell_s_default_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr2, &accessibleBufferSize, SecretoryCell_s_default_cgr));
		//cuda block size
		tile_size = (int)ceil((float)get_agent_SecretoryCell_s_default_count() / threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

		centralise = getMaximumBounds() + getMinimumBounds();

		centralise /= 2;
		//printf("centralise %f %f %f", centralise.x, centralise.y, centralise.z);


		output_SecretoryCell_agent_to_VBO << < grid, threads >> >(get_device_SecretoryCell_s_default_agents(), dptr2, centralise, offset_x, offset_y);
		gpuErrchkLaunch();
		// unmap buffer object
		gpuErrchk(cudaGraphicsUnmapResources(1, &SecretoryCell_s_default_cgr));
	}

}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
int initGL()
{
	// initialize necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 "
		"GL_ARB_pixel_buffer_object")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
		fflush(stderr);
		return 1;
	}

	// default initialization
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glEnable(GL_DEPTH_TEST);

	reshape(WINDOW_WIDTH, WINDOW_HEIGHT);
	checkGLError();

	//lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	return 1;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GLSL Vertex Shader
////////////////////////////////////////////////////////////////////////////////
void initShader()
{
	const char* v = vertexShaderSource;
	const char* f = fragmentShaderSource;

	//vertex shader
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &v, 0);
	glCompileShader(vertexShader);

	//fragment shader
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &f, 0);
	glCompileShader(fragmentShader);

	//program
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// check for errors
	GLint status;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(vertexShader, 262144, &len, data);
		printf("%s", data);
	}
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(fragmentShader, 262144, &len, data);
		printf("%s", data);
	}
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Program Link Error\n");
	}

	// get shader variables
	//vs_displacementMap = glGetUniformLocation(shaderProgram, "displacementMap");
	vs_mapIndex = glGetAttribLocation(shaderProgram, "mapIndex");

	const char* v2 = vertexShaderSource2;
	const char* f2 = fragmentShaderSource2;

	//vertex shader
	walls_vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(walls_vertexShader, 1, &v2, 0);
	glCompileShader(walls_vertexShader);

	//fragment shader
	walls_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(walls_fragmentShader, 1, &f2, 0);
	glCompileShader(walls_fragmentShader);

	//program
	walls_shaderProgram = glCreateProgram();
	glAttachShader(walls_shaderProgram, walls_vertexShader);
	glAttachShader(walls_shaderProgram, walls_fragmentShader);
	glLinkProgram(walls_shaderProgram);

	// check for errors
	//GLint status;
	glGetShaderiv(walls_vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(walls_vertexShader, 262144, &len, data);
		printf("%s", data);
	}
	glGetShaderiv(walls_fragmentShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(walls_fragmentShader, 262144, &len, data);
		printf("%s", data);
	}
	glGetProgramiv(walls_shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Shader Program Link Error\n");
	}

	// get shader variables
	//walls_vs_displacementMap = glGetUniformLocation(walls_shaderProgram, "displacementMap");
	walls_vs_mapIndex = glGetAttribLocation(walls_shaderProgram, "mapIndex");
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, GLuint size)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkGLError();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint* vbo)
{
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Create TBO
////////////////////////////////////////////////////////////////////////////////
void createTBO(cudaGraphicsResource_t* cudaResource, GLuint* tbo, GLuint* tex, GLuint size)
{
	// &EV_default_tbo, &EV_default_displacementTex, 
	// generates one buffer object name in tbo
	glGenBuffers(1, tbo);
	// binds a GL_TEXTURE_BUFFER_EXT to the buffer object named -> tbo
	glBindBuffer(GL_TEXTURE_BUFFER_EXT, *tbo);

	// initialize the buffer object to the size -> xmachine_memory_EV_MAX * sizeof(glm::vec4)
	glBufferData(GL_TEXTURE_BUFFER_EXT, size, 0, GL_DYNAMIC_DRAW);

	//tex - generates one texture name and stores it in -> tex
	glGenTextures(1, tex);
	// binds a GL_TEXTURE_BUFFER_EXT to the buffer object named -> tex
	glBindTexture(GL_TEXTURE_BUFFER_EXT, *tex);
	// map the TBO data to a texture
	// Two Vertex Buffer Objects (VBOs) are used to store vertex data for some vertex point data and 
	// some vertex attributes respectively. A vertex shader then fetches the CUDA position data using 
	// the per vertex attribute data as an index in the texture data.
	glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, *tbo);
	// binds the same texture buffer object to 0-NOTHING?
	glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);

	// register buffer object with CUDA
	gpuErrchk(cudaGraphicsGLRegisterBuffer(cudaResource, *tbo, cudaGraphicsMapFlagsWriteDiscard));

	checkGLError();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete TBO
////////////////////////////////////////////////////////////////////////////////
void deleteTBO(cudaGraphicsResource_t* cudaResource, GLuint* tbo)
{
	gpuErrchk(cudaGraphicsUnregisterResource(*cudaResource));
	*cudaResource = 0;

	glBindBuffer(1, *tbo);
	glDeleteBuffers(1, tbo);

	*tbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Set Sphere Vertex Data
////////////////////////////////////////////////////////////////////////////////
static void setSphereVertex(glm::vec3* data, int slice, int stack) {
	float PI = 3.14159265358;

	double sl = 2 * PI*slice / SPHERE_SLICES;
	double st = 2 * PI*stack / SPHERE_STACKS;

	data->x = cos(st)*sin(sl) * SPHERE_RADIUS;
	data->y = sin(st)*sin(sl) * SPHERE_RADIUS;
	data->z = cos(sl) * SPHERE_RADIUS;
}

////////////////////////////////////////////////////////////////////////////////
//! Set Sphere Normal Data
////////////////////////////////////////////////////////////////////////////////
static void setSphereNormal(glm::vec3* data, int slice, int stack) {
	float PI = 3.14159265358;

	double sl = 2 * PI*slice / SPHERE_SLICES;
	double st = 2 * PI*stack / SPHERE_STACKS;

	data->x = cos(st)*sin(sl);
	data->y = sin(st)*sin(sl);
	data->z = cos(sl);
}

static void setWallVertex(glm::vec3* data, int index) {
	GLfloat vertices[] = {
		0.5f, 0.1f, 0.0f, // Top-right
		-0.5f, 0.1f, 0.0f, // Top-left
		0.5f, -0.1f, 0.0f, // Bottom-right
		-0.5f, -0.1f, 0.0f };  // Bottom-left
	data->x = vertices[index * 3];
	data->y = vertices[index * 3 + 1];
	data->z = vertices[index * 3 + 2];
}

////////////////////////////////////////////////////////////////////////////////
//! Set Vertex Buffer Data
////////////////////////////////////////////////////////////////////////////////
void setVertexBufferData()
{
	int slice, stack;
	int i;

	// upload vertex points data
	// Binds the 'sphereVerts' buffer object to the binding target GL_ARRAY_BUFFER
	glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
	// map a buffer object data store. Target: GL_ARRAY_BUFFER, access: GL_WRITE_ONLY
	// Therefore, the buffer GL_ARRAY_BUFFER will be the target of the data writen in this function. Our reference to this buffer/data is spheresVerts
	glm::vec3* verts = (glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice = 0; slice<SPHERE_SLICES / 2; slice++) {
		for (stack = 0; stack <= SPHERE_STACKS; stack++) {
			// write the vertices data to the 'sphereVerts' buffer object
			setSphereVertex(&verts[i++], slice, stack);
			setSphereVertex(&verts[i++], slice + 1, stack);
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);

	// upload vertex normal data
	glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
	glm::vec3* normals = (glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice = 0; slice<SPHERE_SLICES / 2; slice++) {
		for (stack = 0; stack <= SPHERE_STACKS; stack++) {
			setSphereNormal(&normals[i++], slice, stack);
			setSphereNormal(&normals[i++], slice + 1, stack);
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
	

	glBindBuffer(GL_ARRAY_BUFFER, wallVerts);
	glm::vec3* wall_verts = (glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	for(int i=0; i<12; i++)
		setWallVertex(&wall_verts[i], i);
	glUnmapBuffer(GL_ARRAY_BUFFER);

	/*glBindBuffer(GL_ARRAY_BUFFER, wallNormals);
	glm::vec3* wall_normals = (glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	for (int i = 0; i<36; i++)
		setWallNormals(&wall_normals[i], i);
	glUnmapBuffer(GL_ARRAY_BUFFER);*/
}




////////////////////////////////////////////////////////////////////////////////
//! Reshape callback
////////////////////////////////////////////////////////////////////////////////

void reshape(int width, int height){
	// viewport
	glViewport(0, 0, width, height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(FOVY, (GLfloat)width / (GLfloat)height, NEAR_CLIP, FAR_CLIP);

	checkGLError();
}


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	float millis;

	//CUDA start Timing
	cudaEventRecord(start);

	// run CUDA kernel to generate vertex positions
	runCuda();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//zoom
	glTranslatef(0.0, 0.0, translate_z);

	//move
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 0.0, 1.0);


	//Set light position
	glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION);

	//Draw EV Agents in default state
	/*
	CAMG texture unit 0 us used for drawing the displacement map???
	glBindTexture - binds the named texture EV_default_displacementTex to the
	texturing target GL_TEXTURE_BUFFER_EXT
	https://www.khronos.org/registry/OpenGL-Refpages/es2.0/xhtml/glBindTexture.xml
	*/
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, EV_default_displacementTex);

	//loop
	for (int i = 0; i< get_agent_EV_default_count(); i++){
		// specify the value (i) of a generic vertex attribute to be modified. 
		// Here vs_mapIndex is referencing the attribute "mapIndex from the gl program shaderProgam
		glVertexAttrib1f(vs_mapIndex, (float)i);

		// glEnableClientState - enables the specified capabilities
		// GL_VERTEX_ARRAY - enables the vertex array for writing, it is also used during rendering when 
		// glArrayElement, glDrawArrays, glDrawElements, glDrawRangeElements glMultiDrawArrays,
		// or glMultiDrawElements is called
		glEnableClientState(GL_VERTEX_ARRAY);
		// GL_NORMAL_ARRAY - If enabled, the normal array is enabled for writing and used during rendering when 
		// glArrayElement, glDrawArrays, glDrawElements, glDrawRangeElements glMultiDrawArrays, or glMultiDrawElements is called.
		glEnableClientState(GL_NORMAL_ARRAY);

		// We specify the source for the data we will be rendering

		// activates the buffer type GL_ARRAY_BUFFER - Here sphereVerts contains the vertices for a sphere at default position???
		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		// specifies the location and data format of an array of vertex coordinates to use when rendering. (size, type, stride, pointer to the first vertex in the arrar)
		glVertexPointer(3, GL_FLOAT, 0, 0);

		// activates the buffer type GL_ARRAY_BUFFER
		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		// define an array of normals (type, stride, first coordinate of the first normal in the array
		glNormalPointer(GL_FLOAT, 0, 0);

		// render N primitives by reading the parameters from the buffers previously setup
		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS + 1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}

	glUseProgram(walls_shaderProgram);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, CiliaryCell_default_displacementTex);
	
	//loop
	for (int i = 0; i< get_agent_CiliaryCell_c_default_count(); i++) {
		glVertexAttrib1f(walls_vs_mapIndex, (float)i);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		// We specify the source for the data we will be rendering

		glBindBuffer(GL_ARRAY_BUFFER, wallVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	for (int i = 0; i< get_agent_SecretoryCell_s_default_count(); i++) {
		glVertexAttrib1f(walls_vs_mapIndex, (float)i);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		// We specify the source for the data we will be rendering

		glBindBuffer(GL_ARRAY_BUFFER, wallVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	glUseProgram(shaderProgram);

	//CUDA stop timing
	cudaEventRecord(stop);
	glFlush();
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);
	frame_time += millis;

	if (frame_count == display_rate){
		char title[100];
		sprintf(title, "Execution & Rendering Total: %f (FPS), %f milliseconds per frame", display_rate / (frame_time / 1000.0f), frame_time / display_rate);
		glutSetWindowTitle(title);

		//reset
		frame_count = 0;
		frame_time = 0.0;
	}
	else{
		frame_count++;
	}


	glutSwapBuffers();
	glutPostRedisplay();

}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key) {

	case(87) :
	case(119) :
			  // W == 87, w == 119
			  offset_y -= 10;
		//printf("offset_y=%f", offset_y);
		break;
	case(65) :
	case(97) :
			 // A == 65, a == 97
			 offset_x += 10;
		//printf("offset_x=%f", offset_x);
		break;
	case(83) :
	case(115) :
			  // S == 83, s == 115
			  offset_y += 10;
		//printf("offset_y=%f", offset_y);
		break;

	case(68) :
	case(100) :
			  // D == 68, d == 100
			  offset_x -= 10;
		//printf("offset_x=%f", offset_x);
		break;

	case(75) :
	case(107): // xoom in x 10 - k or K
		translate_z += 10.0;
		printf("zoom in: translate_z=%f\n", translate_z);
		break;
	case(73) :
	case(105) :
			  // zoom in - I or i
			  translate_z += 1.0;
			  printf("zoom in: translate_z=%f\n", translate_z);
		break;

	case(76) :
	case(108): // zoom out x10 - l or L
		translate_z -= 10.0;
		printf("zoom out: translate_z=%f\n", translate_z);
		break;
	case(79) :
	case(111) :
			  // zoom out - O or o
			  translate_z -= 1.0;
			  printf("zoom out: translate_z=%f\n", translate_z);
		break;

		// P == 80, p == 112
	case(80) :
	case(112) :
			  if (paused)
				  printf("un-paused\n");
			  else
				  printf("paused\n");
		paused = !paused;
		break;
		// Esc == 27
	case(27) :
		
		exit(EXIT_SUCCESS);
		// Space == 32
	case(GLUT_KEY_RIGHT) :
		singleIteration();
		fflush(stdout);
		break;
	}
}


void close() {
	deleteVBO(&sphereVerts);
	deleteVBO(&sphereNormals);

	deleteTBO(&EV_default_cgr, &EV_default_tbo);
	deleteTBO(&SecretoryCell_s_default_cgr, &SecretoryCell_default_tbo);
	deleteTBO(&CiliaryCell_c_default_cgr, &CiliaryCell_default_tbo);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cleanup();
}

void special(int key, int x, int y){
}


////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx, dy;
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4) {
		translate_z += dy * VIEW_DISTANCE * 0.001;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void checkGLError(){
	int Error;
	if ((Error = glGetError()) != GL_NO_ERROR)
	{
		const char* Message = (const char*)gluErrorString(Error);
		fprintf(stderr, "OpenGL Error : %s\n", Message);
	}
}
