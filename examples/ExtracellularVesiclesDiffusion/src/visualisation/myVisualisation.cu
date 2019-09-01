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
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include "header.h"
#include "visualisation.h"
#define __MYVISUAL
#include "myVisualisation.h"
#include "shader_s.h"

extern float *h_xs;
extern float *h_ys;
#define FOVY 45.0

GLuint vboIdCiliary;
GLuint vboIdSecretory;
GLuint vaoIdSecretory;
GLuint vaoIdCiliary;
cudaGraphicsResource *CGRsecretory, *CGRciliary;
bool display_secretory = true;

//Simulation output buffers/textures
cudaGraphicsResource_t EV_default_cgr;
// vertex Shader
GLuint vertexShader;
GLuint fragmentShader;
GLuint shaderProgram;
GLuint vs_mapIndex;

GLuint secretory_vertexShader;
GLuint secretory_shaderProgram;
GLuint boundaries_fragmentShader;
GLuint ciliary_vertexShader;
GLuint ciliary_shaderProgram;

// bo variables
GLuint sphereVerts;

Shader *evShader;
Shader *secretoryShader;Shader *ciliaryShader;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -VIEW_DISTANCE;  // This parameter comes from the visualisation.h file

// camera movement
const float DEFAULT_OFFSET_X = 0.0f, DEFAULT_OFFSET_Y = 0.0f, DEFAULT_ZOOM_LEVEL = -VIEW_DISTANCE;
float offset_x = DEFAULT_OFFSET_X, offset_y = DEFAULT_OFFSET_Y, zoom_level = DEFAULT_ZOOM_LEVEL;
glm::mat4 model, view, projection;

// keyboard controls
#if defined(PAUSE_ON_START)
bool paused = true;
#else
bool paused = false;
#endif

//timer
cudaEvent_t start, stop;
const int display_rate = 50;
int frame_count;
float frame_time = 0.0;
unsigned long iterations = 0;

#ifdef SIMULATION_DELAY
//delay
int delay_count = 0;
#endif

// prototypes
int initGL();
void createVBO(GLuint* vbo, GLuint size);
void deleteVBO(GLuint* vbo);
void reshape(int width, int height);
void display();
void close();
void keyboard(unsigned char key, int x, int y);
void special(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void runCuda();
void checkGLError(char * caller);

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

//GPU Kernels
__global__ void output_EV_agent_to_VBO(xmachine_memory_EV_list* agents, glm::vec3* vbo){

	//global thread index
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	vbo[index].x = agents->x[index];
	vbo[index].y = agents->y[index];
	vbo[index].z = agents->radius_um[index];
}

__device__ int copyVertexToVBO(glm::vec2* vbo, int index, float p1_x, float p1_y, float p2_x, float p2_y, float type) {
	int index2 = index * 2;

	vbo[index2].x = p1_x;
	vbo[index2].y = p1_y;
	vbo[index2 + 1].x = p2_x;
	vbo[index2 + 1].y = p2_y;
	return 0;
}

__global__ void output_SecretoryCell_agent_to_VBO(xmachine_memory_SecretoryCell_list* agents, glm::vec2* vbo) {

	//global thread index
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	copyVertexToVBO(vbo, index, agents->p1_x[index], agents->p1_y[index], agents->p2_x[index], agents->p2_y[index], 1.0);
}

__global__ void output_CiliaryCell_agent_to_VBO(xmachine_memory_CiliaryCell_list* agents, glm::vec2* vbo) {

	//global thread index
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	copyVertexToVBO(vbo, index, agents->p1_x[index], agents->p1_y[index],  agents->p2_x[index], agents->p2_y[index], 0.0);
}

void initVisualisation()
{
	printf("Initialising visualisation\n");
	// Create GL context
	int argc = 1;
	char glutString[] = "GLUT application";
	char *argv[] = { glutString, NULL };
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow("FLAME GPU Visualiser");

	// initialize GL
	if (!initGL()) {
		return;
	}

	// register callbacks
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutCloseFunc(close);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
		evShader = new Shader("./shaders/ev_vs.glsl", "./shaders/ev_fs.glsl");	secretoryShader = new Shader("./shaders/secretory_vs.glsl", "./shaders/cells_fs.glsl");	ciliaryShader = new Shader("./shaders/ciliary_vs.glsl", "./shaders/cells_fs.glsl");
	
	// create VBO's
	createVBO(&sphereVerts, get_agent_EV_MAX_count() * sizeof(glm::vec3));
	
	// registers a GraphicsGLResource with CUDA for interop access
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&EV_default_cgr, sphereVerts, cudaGraphicsMapFlagsNone));

	//create a events for timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	printf("Visualisation initialised\n");
}

void runVisualisation(){
	printf(">>>>> Allocating OpenGL resources for: [Secretory cells: %d, Ciliary cells: %d]\n", get_agent_SecretoryCell_s_default_count(), get_agent_CiliaryCell_c_default_count());

	// prepare the containers
	unsigned int size;
	int threads_per_tile = 256;
	int tile_size;
	dim3 grid;
	dim3 threads;

	if (get_agent_SecretoryCell_s_default_count() > 0) {
		size = get_agent_SecretoryCell_MAX_count() * 4 * sizeof(float);
		createVBO(&vboIdSecretory, size);
		gpuErrchk(cudaGraphicsGLRegisterBuffer(&CGRsecretory, vboIdSecretory, cudaGraphicsMapFlagsReadOnly));
		checkGLError("createVBO - secretory");

		glm::vec2 *secretory_dptr;
		size_t size_secretory;
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGraphicsMapResources(1, &CGRsecretory, 0));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&secretory_dptr, &size_secretory, CGRsecretory));

		//cuda block size
		tile_size = (int)ceil((float)get_agent_SecretoryCell_s_default_count() / threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

		output_SecretoryCell_agent_to_VBO << < grid, threads >> > (get_device_SecretoryCell_s_default_agents(), secretory_dptr);
		cudaDeviceSynchronize();
		gpuErrchkLaunch();

		gpuErrchk(cudaGraphicsUnmapResources(1, &CGRsecretory, 0));
	}

	if (get_agent_CiliaryCell_c_default_count() > 0) {
		size = get_agent_CiliaryCell_MAX_count() * 4 * sizeof(float);
		createVBO(&vboIdCiliary, size);
		gpuErrchk(cudaGraphicsGLRegisterBuffer(&CGRciliary, vboIdCiliary, cudaGraphicsMapFlagsReadOnly));
		checkGLError("createVBO - ciliary");

		glm::vec2 *ciliary_dptr;
		size_t size_ciliary;
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGraphicsMapResources(1, &CGRciliary, 0));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&ciliary_dptr, &size_ciliary, CGRciliary));

		//cuda block size
		tile_size = (int)ceil((float)get_agent_CiliaryCell_c_default_count() / threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

		output_CiliaryCell_agent_to_VBO << < grid, threads >> > (get_device_CiliaryCell_c_default_agents(), ciliary_dptr);
		cudaDeviceSynchronize();
		gpuErrchkLaunch();

		gpuErrchk(cudaGraphicsUnmapResources(1, &CGRciliary, 0));
	}

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
		++iterations;
#endif
	}

	//kernals sizes
	int threads_per_tile = 256;
	int tile_size;
	dim3 grid;
	dim3 threads;

	glm::vec3 *dptr;

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
		
		output_EV_agent_to_VBO << < grid, threads >> >(get_device_EV_default_agents(), dptr);
		gpuErrchkLaunch();
		// unmap buffer object
		gpuErrchk(cudaGraphicsUnmapResources(1, &EV_default_cgr));
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL - method called by initVisualisation
////////////////////////////////////////////////////////////////////////////////
int initGL()
{
	// initialize necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported("GL_VERSION_3_3")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
		fflush(stderr);
		return 1;
	}

	// default initialization
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);

	reshape(WINDOW_WIDTH, WINDOW_HEIGHT);
	checkGLError("initGL");

	//lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	return 1;
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

	checkGLError("createVBO");
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
//! Reshape callback
////////////////////////////////////////////////////////////////////////////////
void reshape(int width, int height){
	// viewport
	glViewport(0, 0, width, height);
	projection = glm::perspective(glm::radians((float)FOVY), (float)width / (float)height, (float)NEAR_CLIP, (float)FAR_CLIP);
	checkGLError("reshape");
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

	//Set light position
	glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION);
	checkGLError("display 0");

	model = glm::mat4(1);
	view = glm::translate(glm::mat4(1), glm::vec3(offset_x, offset_y, zoom_level));
	glm::mat4 mv = view * model;
	glm::mat4 mvp = projection * view * model;

	//Draw EV Agents in default state
	if (get_agent_EV_default_count() > 0) {
		evShader->use();
		evShader->setMat4("mvp", mvp);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glDrawArrays(GL_POINTS, 0, get_agent_EV_default_count());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	checkGLError("display 1b");
		
	if (get_agent_SecretoryCell_s_default_count() > 0) {
		secretoryShader->use();
		secretoryShader->setMat4("mvp", mvp);

		glBindBuffer(GL_ARRAY_BUFFER, vboIdSecretory);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
		glEnableVertexAttribArray(0);

		glDrawArrays(GL_LINES, 0, 4 * get_agent_SecretoryCell_s_default_count());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	checkGLError("display 2");
		
	if (get_agent_CiliaryCell_c_default_count() > 0) {
		ciliaryShader->use();
		ciliaryShader->setMat4("mvp", mvp);

		glBindBuffer(GL_ARRAY_BUFFER, vboIdCiliary);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
		glEnableVertexAttribArray(0);

		glDrawArrays(GL_LINES, 0, 4 * get_agent_CiliaryCell_c_default_count());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	checkGLError("display 3");
		
	//CUDA stop timing
	cudaEventRecord(stop);
	glFlush();
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);
	frame_time += millis;

	if (frame_count == display_rate){
		char title[100];
		sprintf(title, "Exec & Render Total: %6.3f (FPS), %6.3f millis p/frame. | iters: %lu", display_rate / (frame_time / 1000.0f), frame_time / display_rate, iterations);
		glutSetWindowTitle(title);

		//reset
		frame_count = 0;
		frame_time = 0.0;
	}
	else{
		frame_count++;
	}
	checkGLError("display 4");

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
			  offset_y -= 1;
		//printf("offset_y=%f", offset_y);
		break;
	case(65) :
	case(97) :
			 // A == 65, a == 97
			 offset_x += 1;
		//printf("offset_x=%f", offset_x);
		break;
	case(83) :
	case(115) :
			  // S == 83, s == 115
			  offset_y += 1;
		//printf("offset_y=%f", offset_y);
		break;

	case(68) :
	case(100) :
			  // D == 68, d == 100
			  offset_x -= 1;
		//printf("offset_x=%f", offset_x);
		break;

	case(75) :
	case(107): // xoom in x 10 - k or K
		translate_z += 10.0;
		zoom_level += 10.0;
		printf("zoom in: translate_z=%f\n", translate_z);
		break;
	case(73) :
	case(105) :
			  // zoom in - I or i
			  translate_z += 1.0;
			  zoom_level += 1.0;
			  printf("zoom in: translate_z=%f\n", translate_z);
		break;

	case(76) :
	case(108): // zoom out x10 - l or L
		translate_z -= 10.0;
		zoom_level -= 10.0;
		printf("zoom out: translate_z=%f\n", translate_z);
		break;
	case(79) :
	case(111) :
			  // zoom out - O or o
			  translate_z -= 1.0;
			  zoom_level -= 1.0;
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
	checkGLError("keyboard");
}


void close() {
	deleteVBO(&sphereVerts);
	deleteVBO(&vboIdSecretory);
	deleteVBO(&vboIdCiliary);

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
	checkGLError("mouse");
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4) {
		translate_z += (float)(dy * VIEW_DISTANCE * 0.001);
	}

	mouse_old_x = x;
	mouse_old_y = y;
	//checkGLError("motion");
}

void checkGLError(char* caller){
	int Error;
	if ((Error = glGetError()) != GL_NO_ERROR)
	{
		const char* Message = (const char*)gluErrorString(Error);
		fprintf(stderr, "OpenGL Error (caller:%s) [%d] : %s \n", caller, Error, Message);
	}
}
