
/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.s
 * 
 */

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#ifndef VISUALISATION
//#define INSTRUMENT_ITERATIONS 1
//#define INSTRUMENT_AGENT_FUNCTIONS 1
//#define INSTRUMENT_INIT_FUNCTIONS 1
//#define INSTRUMENT_STEP_FUNCTIONS 1
//#define INSTRUMENT_EXIT_FUNCTIONS 1
//#define OUTPUT_POPULATION_PER_ITERATION 1
#endif

#include <header.h>
#define _USE_MATH_DEFINES
#include <math_constants.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "myVisualisation.h"
//#include "textureAccess.h"

//#define radius 2.0

// Declare global scope variables for host-based agent creation, so allocation of host data is only performed once.
xmachine_memory_EV ** h_EV_AoS;
unsigned int h_EV_AoS_MAX;

__device__ __host__ float degrees(float radians){
	return radians * (180 / M_PI);
}

__device__ __host__ float radians(float degrees){
	return degrees * (M_PI / 180.0);
}

/**
* Compute the distance between two agents using the coordinates of their centers
*/
__device__ float euclidean_distance(xmachine_memory_EV* agent, float message_x, float message_y)
{
	return sqrt((agent->x - message_x) * (agent->x - message_x) +
		(agent->y - message_y) * (agent->y - message_y));
}


__FLAME_GPU_STEP_FUNC__ void exitFunction(){
	//float x = reduce_EV_default_x_variable();
	//float y = reduce_EV_default_y_variable();
	//float z = reduce_EV_default_z_variable();
	//printf("FlameGPU exit function, avg position is (%f, %f, %f)\n", x, y);
}

__FLAME_GPU_STEP_FUNC__ void print_iteration_no() {
	//float x = reduce_EV_default_x_variable();
	//float y = reduce_EV_default_y_variable();
	//float z = reduce_EV_default_z_variable();
	//printf("FlameGPU exit function, avg position is (%f, %f, %f)\n", x, y);
	//printf("iter: %d\n", ++iteration_number);
}

__FLAME_GPU_INIT_FUNC__ void initialize_new_ev_structures(){
	srand(time(NULL));

	if (*get_const_ev_collisions() > 0) {
		printf("Ev-Ev collisions are enabled\n");
	}
	else {
		printf("Ev-Ev collisions are not enabled\n");
	}
	iteration_number = 0;
	if(*get_introduce_n_new_evs() > 0){
		printf("Attempting introduction of new EVs every %f seconds with a probability threshold of %f\n",
		 *get_seconds_before_introducing_new_evs(), *get_new_evs_threshold());

		//h_EV_AoS_MAX = *get_introduce_n_new_evs();
		//h_EV_AoS = h_allocate_agent_EV_array(*get_introduce_n_new_evs());
	} else {
		printf("No new EVs will be introduced during simulation\n");
	}
}

__FLAME_GPU_INIT_FUNC__ void precompute_values() {
	float val = 6 * M_PI * *get_const_water_viscosity();
	set_const_6_pi_viscosity(&val);
	printf("set_const_6_pi_viscosity: %f\n", val);

	float val2 = *get_const_Boltzmann() * *get_const_Temperature_K();
	printf("boltzmann: %.35f\n", *get_const_Boltzmann());
	set_const_boltzmann_x_temp(&val2);
	printf("set_const_boltzmann_x_temp: %.30f\n", val2);
}

/**
 * output_data FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_EV. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
 */
__FLAME_GPU_FUNC__ int output_data(xmachine_memory_EV* agent, xmachine_message_location_list* location_messages){
    
	add_location_message(location_messages, agent->id, agent->x, agent->y,
		agent->z, agent->radius_um, agent->mass_kg, agent->vx, agent->vy);
	agent->closest_ev_id = -1;
	agent->closest_ev_distance = -1.0f;
	agent->closest_cell_id = -1;
	agent->closest_cell_distance = -1.0f;
    return 0;
}

/**
* output_data FLAMEGPU Agent Function
* Automatically generated using functions.xslt
* @param agent Pointer to an agent structure of type xmachine_memory_EV. This represents a single agent instance and can be modified directly.
* @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
*/
__FLAME_GPU_FUNC__ int output_ciliary_cell_location(xmachine_memory_CiliaryCell* agent, xmachine_message_ciliary_cell_location_list* location_messages) {

	add_ciliary_cell_location_message(location_messages, agent->id, agent->x, agent->y, 0,
		agent->p1_x, agent->p1_y, agent->p2_x, agent->p2_y, agent->direction_x, agent->direction_y
	);
	return 0;
}

__FLAME_GPU_FUNC__ int output_secretory_cell_location(xmachine_memory_SecretoryCell* agent, xmachine_message_secretory_cell_location_list* location_messages) {

	add_secretory_cell_location_message(location_messages, agent->id, agent->x, agent->y, 0,
		agent->p1_x, agent->p1_y, agent->p2_x, agent->p2_y, agent->direction_x, agent->direction_y
	);
	return 0;
}


__device__ float perpendicular(float vec_x, float vec_y, float lenght_vec, int component=0) {
	float x, y;
	if (lenght_vec > 0) {
		x = vec_y * (1 / lenght_vec);
		y = -vec_x * (1 / lenght_vec);
	}
	else {
		x = 0;// = make_float2(0, 0);
		y = 0;
	}
	return component == 0 ? x : y;
}

__device__ float dotprod(float x1, float y1, float x2, float y2) {
	return x1*x2 + y1*y2;
}

__device__ float vlength(float x, float y) {
	return sqrt(x*x + y*y);
}

__device__ float projection(float a_x, float a_y, float b_x, float b_y) {
	float lengthVec = vlength(b_x, b_y);
	if (vlength(a_x, a_y) == 0 || lengthVec == 0)
		return 0;
	return dotprod(a_x, a_y, b_x, b_y) / lengthVec;
}

__device__ float parallel(float vec_x, float vec_y, float u, int component = 0) {
	if (component == 0)
		return vec_x * (u / vlength(vec_x, vec_y));
	else
		return vec_y * (u / vlength(vec_x, vec_y));
}


/**
Collision resolution algorithm modified from Physics for Javascript Games, Animation, and Simulation Ch, 11
*/
__FLAME_GPU_FUNC__ int cell_collision_resolution(xmachine_memory_EV* agent, xmachine_message_cell_collision_list* cell_collision_messages){

	xmachine_message_cell_collision* message = get_first_cell_collision_message(cell_collision_messages);

	//float acceleration_x = 0, acceleration_y = 0;
	
	while (message){
		// we verify the ev_id in the message matches this agent's id
		if (message->ev_id == agent->id) {
			//printf("================ collision checking for agent %d and wall %d =================\n", agent->id, message->cell_id);
			// vector along wall/cell
			float cellDir_x = message->p2_x - message->p1_x;
			float cellDir_y = message->p2_y - message->p1_y;
			float cellDir_len = vlength(cellDir_x, cellDir_y);
			float cellDir_x_unit = cellDir_x / cellDir_len;
			float cellDir_y_unit = cellDir_y / cellDir_len;

			//printf("cell dir x:%f y:%f\n", cellDir_x, cellDir_y);
			// vectors from ball to endpoints of wall
			// var ballp1 = wall.p1.subtract(obj.pos2D);
			float ballp1_x = message->p1_x - agent->x;
			float ballp1_y = message->p1_y - agent->y;
			float ballp2_x = message->p2_x - agent->x;
			float ballp2_y = message->p2_y - agent->y;
			//printf("ballp1 x:%f y:%f message->p1 x:%f y:%f\n", ballp1_x, ballp1_y, message->p1_x, message->p1_y);

			// projection of above vectors onto wall vector
			float proj1 = projection(ballp1_x, ballp1_y, cellDir_x, cellDir_y);
			float proj2 = projection(ballp2_x, ballp2_y, cellDir_x, cellDir_y);
			//printf("projection1:%f\n", proj1);

			// Perpendicular distance vector from the object to the wall
			// var dist = ballp1.addScaled(wdir.unit(), proj1*(-1));
			// needs the cell direction unit vector
			// float2 dist = ballp1 + (cellDir / length(cellDir)) * proj1 * -1;
			
			//printf("cellDir_unit x:%f, y:%f\n", cellDir_x_unit, cellDir_y_unit);
			float dist_x = ballp1_x + (proj1 * -1) * cellDir_x_unit;  // <- distance to compensate
			float dist_y = ballp1_y + (proj1 * -1) * cellDir_y_unit;

			// collision detection
			//var test = ((Math.abs(proj1) < wdir.length()) && (Math.abs(proj2) < wdir.length()));
			
			bool test = (abs(proj1) < cellDir_len) && (abs(proj2) < cellDir_len);
			if ((vlength(dist_x, dist_y) < agent->radius_um) && test)
			{
				//printf("Current values vx:%f vy:%f x:%f y:%f\n", agent->vx, agent->vy, agent->x, agent->y);
				// 1. angle between velocity and wall 
				float angle = acosf(dotprod(agent->vx, agent->vy, cellDir_x, cellDir_y) / (vlength(agent->vx, agent->vy) * cellDir_len));
				//printf("angle: %f\n", angle);

				// 2. reposition object
				// var normal = wall.normal;
				// 2a compute the perpendicular to the cell
				float normal_x = perpendicular(cellDir_x, cellDir_y, cellDir_len, 0);
				float normal_y = perpendicular(cellDir_x, cellDir_y, cellDir_len, 1);
				//printf("wall normal x:%f y:%f\n", normal_x, normal_y);
				// 2b scale the normal by -1 if needed
				float d = dotprod(normal_x, normal_y, agent->vx, agent->vy);
				if (d > 0) {
					normal_x *= -1;
					normal_y *= -1;
					//printf("d > 0 wall with wrong normal?? \n");
				}

				// 3. compute deltaS
				// var deltaS = (obj.radius + dist.dotProduct(normal)) / Math.sin(angle);
				float deltaS = (agent->radius_um + dotprod(dist_x, dist_y, normal_x, normal_y)) / sin(angle);
				//printf("distx:%f disty:%f, deltaS:%f\n", dist_x, dist_y, deltaS);

				// 4. estimate what was the displacement = velocity parallel to delta
				// var displ = obj.velo2D.para(deltaS);
				// float2 displ = parallel(velocity, deltaS); <- parallel: return obj * (u / length(obj));
				float displ_x = parallel(agent->vx, agent->vy, deltaS, 0);
				float displ_y = parallel(agent->vx, agent->vy, deltaS, 1);

				// 5. update position by subtracting the displacement
				// obj.pos2D = obj.pos2D.subtract(displ);
				agent->x -= displ_x;
				agent->y -= displ_y;

				// velocity correction factor
				//var vcor = 1-acc.dotProduct(displ)/obj.velo2D.lengthSquared();
				//float numerator = dotprod(acceleration_x, acceleration_y, displ_x, displ_y);
				//float sqr_vel = (agent->vx * agent->vx + agent->vy*agent->vy);
				//float vfrac = numerator / sqr_vel;
				//float vcor = 1 - vfrac;

				//printf("displx:%f disply:%f numerator:%f sqr_vel:%f vfrac:%f vcor:%f\n", displ_x, displ_y, numerator, sqr_vel, vfrac, vcor);
				// corrected velocity vector just before impact 
				// var Velo = obj.velo2D.multiply(vcor)
				float new_velocity_x = agent->vx;// *vcor;
				float new_velocity_y = agent->vy;// *vcor;
				//printf("newVelocity x:%f y:%f\n", new_velocity_x, new_velocity_y);

				// 6. decompose the velocity
				// velocity vector component perpendicular to wall just before impact
				// var normalVelo = dist.para(Velo.projection(dist));
				// float2 normalVelocity = parallel(dist, projection(velocity, dist));
				float velocityProjection = projection(new_velocity_x, new_velocity_y, dist_x, dist_y);
				float normalVelocity_x = parallel(dist_x, dist_y, velocityProjection, 0);
				float normalVelocity_y = parallel(dist_x, dist_y, velocityProjection, 1);
				//printf("velocityProjection:%f normalVelocity_x:%f normalVelocity_y:%f\n", velocityProjection, normalVelocity_x, normalVelocity_y);

				// velocity vector component parallel to wall; unchanged by impact
				// var tangentVelo = Velo.subtract(normalVelo);
				// float2 tangentVelocity = velocity - normalVelocity;
				float tangentVelocity_x = new_velocity_x - normalVelocity_x;
				float tangentVelocity_y = new_velocity_y - normalVelocity_y;
				//printf("tangentVelocity_x:%f tangentVelocity_y:%f\n", tangentVelocity_x, tangentVelocity_y);

				// velocity vector component perpendicular to wall just after impact
				// obj.velo2D = tangentVelo.addScaled(normalVelo, -vfac);
				//float2 newVelocity = tangentVelocity + normalVelocity * -1;

				agent->vx = tangentVelocity_x + normalVelocity_x * -1;
				agent->vy = tangentVelocity_y + normalVelocity_y * -1;
				agent->velocity_ums = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
				//printf("New values vx:%f vy:%f x:%f y:%f\n", agent->vx, agent->vy, agent->x, agent->y);
			}
			// collision at the boundaries
			else if (abs(vlength(ballp1_x, ballp1_y)) < agent->radius_um) {
				// bouce off endpoint wall_p1
				float distp_x = agent->x - ballp1_x;
				float distp_y = agent->y - ballp1_y;
				//var distp = obj.pos2D.subtract(pEndpoint);
				//// move particle so that it just touches the endpoint			
				//var L = obj.radius - distp.length();
				//var vrel = obj.velo2D.length();
				//obj.pos2D = obj.pos2D.addScaled(obj.velo2D, -L / vrel);
				float L = agent->radius_um - vlength(distp_x, distp_y);
				float vrel = vlength(agent->vx, agent->vy);
				agent->x += (-L / vrel) * agent->vx;
				agent->y += (-L / vrel) * agent->vy;

				//// normal velocity vector just before the impact
				//var normalVelo = obj.velo2D.project(distp);
				float normalVelo = projection(agent->vx, agent->vy, distp_x, distp_y);
				//// tangential velocity vector
				//var tangentVelo = obj.velo2D.subtract(normalVelo);
				float tangentVelo_x = agent->vx - normalVelo;
				float tangentVelo_y = agent->vy - normalVelo;
				//// normal velocity vector after collision
				//normalVelo.scaleBy(-vfac);

				//// final velocity vector after collision
				//obj.velo2D = normalVelo.add(tangentVelo);
				agent->vx = normalVelo + tangentVelo_x;
				agent->vy = normalVelo + tangentVelo_y;
				agent->velocity_ums = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
				//printf("Collision with boundary p1\n");
			}
			else if (abs(vlength(ballp2_x, ballp2_y)) < agent->radius_um) {
				// bouce off endpoint wall_p2

				float distp_x = agent->x - ballp2_x;
				float distp_y = agent->y - ballp2_y;
				//var distp = obj.pos2D.subtract(pEndpoint);
				//// move particle so that it just touches the endpoint			
				//var L = obj.radius - distp.length();
				//var vrel = obj.velo2D.length();
				//obj.pos2D = obj.pos2D.addScaled(obj.velo2D, -L / vrel);
				float L = agent->radius_um - vlength(distp_x, distp_y);
				float vrel = vlength(agent->vx, agent->vy);
				agent->x += (-L / vrel) * agent->vx;
				agent->y += (-L / vrel) * agent->vy;

				//// normal velocity vector just before the impact
				//var normalVelo = obj.velo2D.project(distp);
				float normalVelo = projection(agent->vx, agent->vy, distp_x, distp_y);
				//// tangential velocity vector
				//var tangentVelo = obj.velo2D.subtract(normalVelo);
				float tangentVelo_x = agent->vx - normalVelo;
				float tangentVelo_y = agent->vy - normalVelo;
				//// normal velocity vector after collision
				//normalVelo.scaleBy(-vfac);

				//// final velocity vector after collision
				//obj.velo2D = normalVelo.add(tangentVelo);
				agent->vx = normalVelo + tangentVelo_x;
				agent->vy = normalVelo + tangentVelo_y;
				agent->velocity_ums = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
				//printf("Collision with boundary p2\n");
			}
			// no collision occurred
		}
		message = get_next_cell_collision_message(message, cell_collision_messages);
	}
	return 0;
}

/**
This function resolves a single collision between an EV and a cell
After this, the EV has new values for x, y, vx, and vy, compensated for the collision
*/
__FLAME_GPU_FUNC__ int ciliary_cell_collision_resolution(xmachine_memory_EV* agent, xmachine_message_ciliary_cell_collision_list* ciliary_cell_collision_messages) {

	xmachine_message_ciliary_cell_collision* message = get_first_ciliary_cell_collision_message(ciliary_cell_collision_messages);
	//float acceleration_x = 0, acceleration_y = 0;
	while (message) {
		// we verify the ev_id in the message matches this agent's id
		if (message->ev_id == agent->id) {
			//printf("================ collision checking for agent %d and wall %d =================\n", agent->id, message->cell_id);
			// vector along wall/cell
			float cellDir_x = message->p2_x - message->p1_x;
			float cellDir_y = message->p2_y - message->p1_y;
			float cellDir_len = vlength(cellDir_x, cellDir_y);
			float cellDir_x_unit = cellDir_x / cellDir_len;
			float cellDir_y_unit = cellDir_y / cellDir_len;


			//printf("cell dir x:%f y:%f\n", cellDir_x, cellDir_y);
			// vectors from ball to endpoints of wall
			// var ballp1 = wall.p1.subtract(obj.pos2D);
			float ballp1_x = message->p1_x - agent->x;
			float ballp1_y = message->p1_y - agent->y;
			float ballp2_x = message->p2_x - agent->x;
			float ballp2_y = message->p2_y - agent->y;
			//printf("ballp1 x:%f y:%f message->p1 x:%f y:%f\n", ballp1_x, ballp1_y, message->p1_x, message->p1_y);

			// projection of above vectors onto wall vector
			float proj1 = projection(ballp1_x, ballp1_y, cellDir_x, cellDir_y);
			float proj2 = projection(ballp2_x, ballp2_y, cellDir_x, cellDir_y);
			//printf("projection1:%f\n", proj1);

			// Perpendicular distance vector from the object to the wall
			// var dist = ballp1.addScaled(wdir.unit(), proj1*(-1));
			// needs the cell direction unit vector
			// float2 dist = ballp1 + (cellDir / length(cellDir)) * proj1 * -1;
			
			//printf("cellDir_unit x:%f, y:%f\n", cellDir_x_unit, cellDir_y_unit);
			float dist_x = ballp1_x + (proj1 * -1) * cellDir_x_unit;  // <- distance to compensate
			float dist_y = ballp1_y + (proj1 * -1) * cellDir_y_unit;

			// collision detection
			//var test = ((Math.abs(proj1) < wdir.length()) && (Math.abs(proj2) < wdir.length()));
			
			bool test = (abs(proj1) < cellDir_len) && (abs(proj2) < cellDir_len);
			if ((vlength(dist_x, dist_y) < agent->radius_um) && test)
			{
				//printf("Current values vx:%f vy:%f x:%f y:%f\n", agent->vx, agent->vy, agent->x, agent->y);
				// 1. angle between velocity and wall 
				float angle = acosf(dotprod(agent->vx, agent->vy, cellDir_x, cellDir_y) / (vlength(agent->vx, agent->vy) * cellDir_len));
				//printf("angle: %f\n", angle);

				// 2. reposition object
				// var normal = wall.normal;
				// 2a compute the perpendicular to the cell
				float normal_x = perpendicular(cellDir_x, cellDir_y, cellDir_len, 0);
				float normal_y = perpendicular(cellDir_x, cellDir_y, cellDir_len, 1);
				//printf("wall normal x:%f y:%f\n", normal_x, normal_y);
				// 2b scale the normal by -1 if needed
				float d = dotprod(normal_x, normal_y, agent->vx, agent->vy);
				if (d > 0) {
					normal_x *= -1;
					normal_y *= -1;
					//printf("d > 0 wall with wrong normal?? \n");
				}

				// 3. compute deltaS
				// var deltaS = (obj.radius + dist.dotProduct(normal)) / Math.sin(angle);
				float deltaS = (agent->radius_um + dotprod(dist_x, dist_y, normal_x, normal_y)) / sin(angle);
				//printf("distx:%f disty:%f, deltaS:%f\n", dist_x, dist_y, deltaS);

				// 4. estimate what was the displacement = velocity parallel to delta
				// var displ = obj.velo2D.para(deltaS);
				// float2 displ = parallel(velocity, deltaS); <- parallel: return obj * (u / length(obj));
				float displ_x = parallel(agent->vx, agent->vy, deltaS, 0);
				float displ_y = parallel(agent->vx, agent->vy, deltaS, 1);

				// 5. update position by subtracting the displacement
				// obj.pos2D = obj.pos2D.subtract(displ);
				agent->x -= displ_x;
				agent->y -= displ_y;

				// velocity correction factor
				//var vcor = 1-acc.dotProduct(displ)/obj.velo2D.lengthSquared();
				// our simulation uses zero acceleration
				//float numerator = dotprod(0, 0, displ_x, displ_y);
				//float sqr_vel = (agent->vx * agent->vx + agent->vy*agent->vy);
				//float vfrac = numerator / sqr_vel;
				//float vcor = 1 - vfrac;

				//printf("displx:%f disply:%f numerator:%f sqr_vel:%f vfrac:%f vcor:%f\n", displ_x, displ_y, numerator, sqr_vel, vfrac, vcor);
				// corrected velocity vector just before impact 
				// var Velo = obj.velo2D.multiply(vcor)
				float new_velocity_x = agent->vx;// *1;
				float new_velocity_y = agent->vy;// *1;
				//printf("newVelocity x:%f y:%f\n", new_velocity_x, new_velocity_y);

				// 6. decompose the velocity
				// velocity vector component perpendicular to wall just before impact
				// var normalVelo = dist.para(Velo.projection(dist));
				// float2 normalVelocity = parallel(dist, projection(velocity, dist));
				float velocityProjection = projection(new_velocity_x, new_velocity_y, dist_x, dist_y);
				float normalVelocity_x = parallel(dist_x, dist_y, velocityProjection, 0);
				float normalVelocity_y = parallel(dist_x, dist_y, velocityProjection, 1);
				//printf("velocityProjection:%f normalVelocity_x:%f normalVelocity_y:%f\n", velocityProjection, normalVelocity_x, normalVelocity_y);

				// velocity vector component parallel to wall; unchanged by impact
				// var tangentVelo = Velo.subtract(normalVelo);
				// float2 tangentVelocity = velocity - normalVelocity;
				float tangentVelocity_x = new_velocity_x - normalVelocity_x;
				float tangentVelocity_y = new_velocity_y - normalVelocity_y;
				//printf("tangentVelocity_x:%f tangentVelocity_y:%f\n", tangentVelocity_x, tangentVelocity_y);

				// velocity vector component perpendicular to wall just after impact
				// obj.velo2D = tangentVelo.addScaled(normalVelo, -vfac);
				//float2 newVelocity = tangentVelocity + normalVelocity * -1;

				agent->vx = tangentVelocity_x + normalVelocity_x * -1;
				agent->vy = tangentVelocity_y + normalVelocity_y * -1;
				agent->velocity_ums = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
				//printf("New values vx:%f vy:%f x:%f y:%f\n", agent->vx, agent->vy, agent->x, agent->y);
			}
			// collision at the boundaries
			else if (abs(vlength(ballp1_x, ballp1_y)) < agent->radius_um) {
				// bouce off endpoint wall_p1
				float distp_x = agent->x - ballp1_x;
				float distp_y = agent->y - ballp1_y;
				//var distp = obj.pos2D.subtract(pEndpoint);
				//// move particle so that it just touches the endpoint			
				//var L = obj.radius - distp.length();
				//var vrel = obj.velo2D.length();
				//obj.pos2D = obj.pos2D.addScaled(obj.velo2D, -L / vrel);
				float L = agent->radius_um - vlength(distp_x, distp_y);
				float vrel = vlength(agent->vx, agent->vy);
				agent->x += (-L / vrel) * agent->vx;
				agent->y += (-L / vrel) * agent->vy;

				//// normal velocity vector just before the impact
				//var normalVelo = obj.velo2D.project(distp);
				float normalVelo = projection(agent->vx, agent->vy, distp_x, distp_y);
				//// tangential velocity vector
				//var tangentVelo = obj.velo2D.subtract(normalVelo);
				float tangentVelo_x = agent->vx - normalVelo;
				float tangentVelo_y = agent->vy - normalVelo;
				//// normal velocity vector after collision
				//normalVelo.scaleBy(-vfac);

				//// final velocity vector after collision
				//obj.velo2D = normalVelo.add(tangentVelo);
				agent->vx = normalVelo + tangentVelo_x;
				agent->vy = normalVelo + tangentVelo_y;
				agent->velocity_ums = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
				//printf("Collision with boundary p1\n");
			}
			else if (abs(vlength(ballp2_x, ballp2_y)) < agent->radius_um) {
				// bouce off endpoint wall_p2

				float distp_x = agent->x - ballp2_x;
				float distp_y = agent->y - ballp2_y;
				//var distp = obj.pos2D.subtract(pEndpoint);
				//// move particle so that it just touches the endpoint			
				//var L = obj.radius - distp.length();
				//var vrel = obj.velo2D.length();
				//obj.pos2D = obj.pos2D.addScaled(obj.velo2D, -L / vrel);
				float L = agent->radius_um - vlength(distp_x, distp_y);
				float vrel = vlength(agent->vx, agent->vy);
				agent->x += (-L / vrel) * agent->vx;
				agent->y += (-L / vrel) * agent->vy;

				//// normal velocity vector just before the impact
				//var normalVelo = obj.velo2D.project(distp);
				float normalVelo = projection(agent->vx, agent->vy, distp_x, distp_y);
				//// tangential velocity vector
				//var tangentVelo = obj.velo2D.subtract(normalVelo);
				float tangentVelo_x = agent->vx - normalVelo;
				float tangentVelo_y = agent->vy - normalVelo;
				//// normal velocity vector after collision
				//normalVelo.scaleBy(-vfac);

				//// final velocity vector after collision
				//obj.velo2D = normalVelo.add(tangentVelo);
				agent->vx = normalVelo + tangentVelo_x;
				agent->vy = normalVelo + tangentVelo_y;
				agent->velocity_ums = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
				//printf("Collision with boundary p2\n");
			}
			// no collision occurred
		}
		message = get_next_ciliary_cell_collision_message(message, ciliary_cell_collision_messages);
	}
	return 0;
}

/**
let's call our point p0 and the points that define the line as p1 and p2.
Then you get the vectors A = p0 - p1 and B = p2 - p1.Param is the scalar value
that when multiplied to B gives you the point on the line closest to p0.
Modified from https ://stackoverflow.com/a/6853926/3830240
returns
would_intersect
distance b / w intersection point and p0
coordinates of intersection point
*/
__device__ float4 point_to_line_segment_distance(float x, float y, float x1, float y1, float x2, float y2)
{
	float A = x - x1;
	float B = y - y1;
	float C = x2 - x1;
	float D = y2 - y1;

	float dot = A * C + B * D;
	float len_sq = C * C + D * D;
	float param = -1;
	bool intersecting = false;
	float xx = 0.f, yy = 0.f, dx, dy;

	if (len_sq != 0){ // in case of 0 length line
		param = dot / len_sq;
	}
	
	if (param < 0){ // the closest point is p1, point not projecting on segment
		xx = x1;
		yy = y1;
	} 
	else if (param > 1){ // the closest point is p2, point not projecting on segment
		xx = x2;
		yy = y2;
	}
	else{   // interpolate to find the closes point on the line segment
		intersecting = true;
		xx = x1 + param * C;
		yy = y1 + param * D;
	}

	// dx / dy is the vector from p0 to the closest point on the segment
	dx = x - xx;
	dy = y - yy;
	float4 result;
	result.x = (intersecting ? 1.f : 0.f);
	result.y = sqrt(dx * dx + dy * dy);
	result.z = xx;
	result.w = yy;
	
	return result;
}

/**
Checks any possible collision with the secretory cells on the boundaries. 
If there is an occurrence, a collision_message is generated with the following parameters:
	ev_id = the id of the current EV
	closest_cell - id of the cell the collision occurs with
	closest_cell_distance
	collision_point_x
	collision_point_y
	angle of the wall
When no collision is detected, the ev agent will have the values:
	agent->closest_cell_id = -1;
	agent->closest_cell_distance = -1.0f;
*/
__FLAME_GPU_FUNC__ int test_secretory_cell_collision(xmachine_memory_EV* agent, xmachine_message_secretory_cell_location_list* location_messages,
	xmachine_message_secretory_cell_location_PBM* partition_matrix, xmachine_message_cell_collision_list* cell_collision_messages)
{
	int closest_cell = -1;
	float closest_cell_distance = 10000.f;
	float3 closest;
	float4 res;
	float2 direction;
	float4 cell_pts;

	xmachine_message_secretory_cell_location* message = get_first_secretory_cell_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	while (message) {
		// get the distance between agent and cell
		res = point_to_line_segment_distance(agent->x, agent->y, message->p1_x, message->p1_y, message->p2_x, message->p2_y);

		// if the ev is in collision route and the distance to segment < radius
		if (res.x > 0.f && res.y < agent->radius_um)
		{
			//printf("secretory res.x: %f res.y:%f radius_um:%f\n", res.x, res.y, agent->radius_um);
			if (closest_cell == -1) // save the reference if this is the first candidate collision
			{
				closest_cell = message->id;
				closest_cell_distance = res.y;
				closest.x = res.z;
				closest.y = res.w;
				//closest.z = message->angle; //get_boundary_angle(offset_angles[agent->sec] + i);
				direction.x = message->direction_x;
				direction.y = message->direction_y;

				cell_pts.x = message->p1_x;
				cell_pts.y = message->p1_y;
				cell_pts.z = message->p2_x;
				cell_pts.w = message->p2_y;
			}
			else if (res.y < closest_cell_distance) {
				//		// otherwise only save the reference if this collision takes precedence
				closest_cell = message->id;
				closest_cell_distance = res.y;
				closest.x = res.z;
				closest.y = res.w;
				//closest.z = message->angle; //get_boundary_angle(offset_angles[agent->sec] + i);
				direction.x = message->direction_x;
				direction.y = message->direction_y;

				cell_pts.x = message->p1_x;
				cell_pts.y = message->p1_y;
				cell_pts.z = message->p2_x;
				cell_pts.w = message->p2_y;
			}
		}

		// get the next message
		message = get_next_secretory_cell_location_message(message, location_messages, partition_matrix);
	}

	if (closest_cell > -1){
		//printf("collision detected with secretory cell:%d at:%f\n", closest_cell, closest_cell_distance);
		agent->closest_cell_id = closest_cell;
		agent->closest_cell_distance = closest_cell_distance;
		// write the corresponding collision_message
		add_cell_collision_message(cell_collision_messages, agent->id, closest_cell, closest_cell_distance,
			cell_pts.x, cell_pts.y, cell_pts.z, cell_pts.w, direction.x, direction.y);
	}
	return 0;
}

/**
Checks any possible collision with the secretory cells on the boundaries.
If there is an occurrence, a collision_message is generated with the following parameters:
	ev_id = the id of the current EV
	closest_cell - id of the cell the collision occurs with
	closest_cell_distance
	collision_point_x
	collision_point_y
	angle of the wall
When no collision is detected, the ev agent will not have its values overwriten
*/
__FLAME_GPU_FUNC__ int test_ciliary_cell_collision(xmachine_memory_EV* agent, xmachine_message_ciliary_cell_location_list* location_messages,
	xmachine_message_ciliary_cell_location_PBM* partition_matrix, xmachine_message_ciliary_cell_collision_list* ciliary_cell_collision_messages)
{
	int closest_cell = -1;
	float closest_cell_distance = 10000.f;
	float3 closest;
	float4 res;
	float2 direction;
	float4 cell_pts;

	xmachine_message_ciliary_cell_location* message = get_first_ciliary_cell_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	while (message) {
		// get the distance between agent and cell
		res = point_to_line_segment_distance(agent->x, agent->y, message->p1_x, message->p1_y, message->p2_x, message->p2_y);

		// if the ev is in collision route and the distance to segment < radius
		if (res.x > 0.f && res.y < agent->radius_um)
		{
			//printf("ciliary res.x: %f res.y:%f radius_um:%f\n", res.x, res.y, agent->radius_um);
			if (closest_cell == -1) // save the reference if this is the first candidate collision
			{
				closest_cell = message->id;
				closest_cell_distance = res.y;
				closest.x = res.z;
				closest.y = res.w;
				//closest.z = message->angle; //get_boundary_angle(offset_angles[agent->sec] + i);
				direction.x = message->direction_x;
				direction.y = message->direction_y;

				cell_pts.x = message->p1_x;
				cell_pts.y = message->p1_y;
				cell_pts.z = message->p2_x;
				cell_pts.w = message->p2_y;
			}
			else if (res.y < closest_cell_distance) {
				//		// otherwise only save the reference if this collision takes precedence
				closest_cell = message->id;
				closest_cell_distance = res.y;
				closest.x = res.z;
				closest.y = res.w;
				//closest.z = message->angle; //get_boundary_angle(offset_angles[agent->sec] + i);
				direction.x = message->direction_x;
				direction.y = message->direction_y;

				cell_pts.x = message->p1_x;
				cell_pts.y = message->p1_y;
				cell_pts.z = message->p2_x;
				cell_pts.w = message->p2_y;
			}
		}
		message = get_next_ciliary_cell_location_message(message, location_messages, partition_matrix);
	}
	
	if (closest_cell > -1) {
		if (agent->closest_cell_id > -1.f) {
			if (closest_cell_distance < agent->closest_cell_distance) {
				//printf("collision detected with ciliary cell:%d at:%f occurs before the collision with the secretory cell\n", closest_cell, closest_cell_distance);
				// agent is closest to this ciliary cell than any secretory cell
				agent->closest_cell_id = closest_cell;
				agent->closest_cell_distance = closest_cell_distance;

				// write the corresponding collision_message
				add_ciliary_cell_collision_message(ciliary_cell_collision_messages, agent->id, closest_cell, closest_cell_distance,
					cell_pts.x, cell_pts.y, cell_pts.z, cell_pts.w, direction.x, direction.y);
			}
			else {
				// agent is closest to a secretory cell
			}
		}
		else {
			//printf("collision detected with ciliary cell:%d at:%f with no previous collition with a secretory cell\n", closest_cell, closest_cell_distance);
			// agent is closest to the detected ciliary cell
			agent->closest_cell_id = closest_cell;
			agent->closest_cell_distance = closest_cell_distance;
			// write the corresponding collision_message
			add_ciliary_cell_collision_message(ciliary_cell_collision_messages, agent->id, closest_cell, closest_cell_distance,
				cell_pts.x, cell_pts.y, cell_pts.z, cell_pts.w, direction.x, direction.y);
		}
	}
	return 0;
}


/**
This function has two cases.
Current Ev is the EV1 in collision message.
Current Ev in the EV2 in collision message.
As a single message is generated per collisison, we must use it to compensate both EVs involved
*/
__FLAME_GPU_FUNC__ int ev_collision_resolution(xmachine_memory_EV* agent, xmachine_message_ev_collision_list* ev_collision_messages){
	float dmin, cs, sc, vp1, vp2, ddt, dx, dy, ax, ay, distance, va1, va2, vb, vaP;

	// fetch the message where this EV is involved in the collision
	xmachine_message_ev_collision* message = get_first_ev_collision_message(ev_collision_messages);
	while (message){
		if (message->ev1_id == agent->id && message->ev2_id == agent->closest_ev_id){
			// agent is ev1
			dmin = message->ev1_r_um + message->ev2_r_um;

			// compute the direction resulting of the collision
			cs = (message->ev2_x - message->ev1_x) / (message->distance + FLT_MIN);
			sc = (message->ev2_y - message->ev1_y) / (message->distance + FLT_MIN);

			// calculate the component of velocity in direction cs, sc for each particle
			vp1 = message->ev1_vx * cs + message->ev1_vy * sc; // ev1
			vp2 = message->ev2_vx * cs + message->ev2_vy * sc; // ev2

			// back to collision time
			ddt = (dmin - message->distance) / ((vp1 - vp2) + FLT_MIN);

			// time of collision occuring after the time step ? don't affect current velocities
			// otherwise, get the next position back to the moment of collision
			if (ddt > dt)
				ddt = 0;
			else if (ddt < 0)
				ddt = dt;

			agent->x -= message->ev1_vx * ddt; // ev1 - agent->x
			agent->y -= message->ev1_vy * ddt;

			// calculate components of velocity
			dx = (message->ev2_x - message->ev1_x);
			dy = (message->ev2_y - message->ev1_y);

			// where x1, y1 are center of ball1, and x2, y2 are center of ball2
			distance = sqrt(dx*dx + dy*dy);
			// Unit vector in the direction of the collision
			ax = dx / (distance + FLT_MIN);
			ay = dy / (distance + FLT_MIN);
			// Projection of the velocities in these axes
			// vector sum for velocity p1, p2
			va1 = message->ev1_vx * ax + message->ev1_vy * ay;
			vb = -(message->ev1_vx) * ay + message->ev1_vy * ax;
			va2 = message->ev2_vx * ax + message->ev2_vy * ay;
			// New velocities in these axes(after collision)

			float vaP = va1 + 2 * (va2 - va1) / ((1 + message->ev1_mass / message->ev2_mass) + FLT_MIN);

			// undo the projections, get back to the original coordinates
			agent->vx = vaP * ax - vb * ay;
			agent->vy = vaP * ay + vb * ax; // new vx, vy for ball 1 after collision;
			agent->velocity_ums = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
			// Because we have moved back the time by dt, we need to move the time forward by dt.
			agent->x += agent->vx * (dt - ddt);
			agent->y += agent->vy * (dt - ddt);
		}
		else if (message->ev2_id == agent->id && message->ev1_id == agent->closest_ev_id){
			// agent is ev2
			dmin = message->ev1_r_um + message->ev2_r_um;

			// compute the direction resulting of the collision
			cs = (message->ev2_x - message->ev1_x) / (message->distance + FLT_MIN);
			sc = (message->ev2_y - message->ev1_y) / (message->distance + FLT_MIN);

			// calculate the component of velocity in direction cs, sc for each particle
			vp1 = message->ev1_vx * cs + message->ev1_vy * sc; // ev1
			vp2 = message->ev2_vx * cs + message->ev2_vy * sc; // ev2

			// back to collision time
			ddt = (dmin - message->distance) / ((vp1 - vp2) + FLT_MIN);

			// time of collision occuring after the time step ? don't affect current velocities
			// otherwise, get the next position back to the moment of collision
			if (ddt > dt)
				ddt = 0;
			else if (ddt < 0)
				ddt = dt;

			agent->x -= message->ev2_vx * ddt; // ev2 - agent
			agent->y -= message->ev2_vx * ddt;

			// calculate component of velocity
			dx = (message->ev2_x - message->ev1_x);
			dy = (message->ev2_y - message->ev1_y);
			// where x1, y1 are center of ball1, and x2, y2 are center of ball2
			distance = sqrt(dx*dx + dy*dy);
			// Unit vector in the direction of the collision
			ax = dx / (distance + FLT_MIN);
			ay = dy / (distance + FLT_MIN);
			// Projection of the velocities in these axes
			// vector sum for velocity p1, p2
			va1 = message->ev1_vx * ax + message->ev1_vy * ay;
			//float vb1 = -message->ev1_vx * ay + message->ev1_vy * ax;
			va2 = message->ev2_vx * ax + message->ev2_vy * ay;
			vb = -message->ev2_vx * ay + message->ev2_vy * ax;
			// New velocities in these axes(after collision)
			//float vaP1 = va1 + 2 * (va2 - va1) / (1 + message->ev1_mass / message->ev2_mass);
			vaP = va2 + 2 * (va1 - va2) / ((1 + message->ev2_mass / message->ev1_mass) + FLT_MIN);

			// undo the projections, get back to the original coordinates
			agent->vx = vaP * ax - vb * ay;
			agent->vy = vaP * ay + vb * ax; // new vx, vy for ball 2 after collision
			agent->velocity_ums = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
			// Because we have moved back the time by dt, we need to move the time forward by dt.
			agent->x += agent->vx * (dt - ddt);
			agent->y += agent->vy * (dt - ddt);
		}
		message = get_next_ev_collision_message(message, ev_collision_messages);
	}

	
	return 0;
}

/**
Identifies the first collision occuring with another EV and records in the current EV the id of the colliding EV the collision distance
*/	
__FLAME_GPU_FUNC__ int test_ev_collision(xmachine_memory_EV* agent, xmachine_message_location_list* location_messages, 
	xmachine_message_location_PBM* partition_matrix, xmachine_message_ev_collision_list* ev_collision_messages){
	int closest_ev_id = -1;
	float closest_ev_distance = 10000.f;
	float distance;
	float radii = 0.0;
	float ev2_mass, ev2_r_um, ev2_x, ev2_y, ev2_vx, ev2_vy;

	xmachine_message_location* message = get_first_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	// Identify the closest EV colliding with this EV
	while (message)
	{
		if (message->id != agent->id)
		{
			// check for collision
			distance = euclidean_distance(agent, message->x, message->y);
			radii = agent->radius_um + message->radius_um;

			if (distance <= radii)
			{
				if(closest_ev_id == -1){
					closest_ev_id = message->id;
					closest_ev_distance = distance;

					ev2_mass = message->mass_kg;
					ev2_r_um = message->radius_um;
					ev2_x = message->x;
					ev2_y = message->y;
					ev2_vx = message->vx;
					ev2_vy = message->vy;
				}
				else if (distance < closest_ev_distance){
					closest_ev_id = message->id;
					closest_ev_distance = distance;

					ev2_mass = message->mass_kg;
					ev2_r_um = message->radius_um;
					ev2_x = message->x;
					ev2_y = message->y;
					ev2_vx = message->vx;
					ev2_vy = message->vy;
				}
			}
		}
		message = get_next_location_message(message, location_messages, partition_matrix);
	}
	// record the relevant data
	
	if (closest_ev_id != -1){
		agent->closest_ev_id = closest_ev_id;
		agent->closest_ev_distance = closest_ev_distance;
		// write the corresponding collision_message
		add_ev_collision_message(ev_collision_messages, agent->id, closest_ev_id, closest_ev_distance, 
			agent->mass_kg, agent->radius_um, agent->x, agent->y, agent->vx, agent->vy, 
			ev2_mass, ev2_r_um, ev2_x, ev2_y, ev2_vx, ev2_vy);
	} else {
		agent->closest_ev_id = -1;
		agent->closest_ev_distance = -1.0f;
	}
	return 0;
}

__FLAME_GPU_FUNC__ int reset_state(xmachine_memory_EV* agent) {
	return 0;
}

__FLAME_GPU_FUNC__ int brownian_movement_1d(xmachine_memory_EV* agent, RNG_rand48* rand48) {
	float u1, u2, r, theta;

	u1 = rnd<CONTINUOUS>(rand48);
	u2 = rnd<CONTINUOUS>(rand48);
	r = sqrt(-2.0 * log(u1));
	theta = 2 * M_PI * u2;

	// the product of r * (cos|sin)(theta) becomes the displacement factor to use in this iteration
	agent->vx = agent->velocity_ums * r * cos(theta);
	//agent->vx += agent->velocity_ums * r * sin(theta);

	//agent->velocity_ums = sqrtf(agent->vx * agent->vx);
	return 0;
}

/*
Uses the Box-Mueller transform to generate a pair of normally distributed random numbers
by sampling from two uniformly distributed RNG.
Then, the values are transformed into cartesian coordinates.
Due to limitations in FlameGPU, we sample both continuos numbers from the same RNG.
*/
__FLAME_GPU_FUNC__ int brownian_movement_2d(xmachine_memory_EV* agent, RNG_rand48* rand48) {
	float u1, u2, fac, rsq, r, theta; 
	
	u1 = rnd<CONTINUOUS>(rand48);
	u2 = rnd<CONTINUOUS>(rand48);
	r = sqrt(-2.0 * log(u1));
	theta = 2 * M_PI * u2;

	agent->vx = agent->velocity_ums * r * cos(theta);
	agent->vy = agent->velocity_ums * r * sin(theta);

	return 0;
}
/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_EV. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_EV* agent){
	
	agent->x_1 = agent->x;
	agent->y_1 = agent->y;

	agent->x += agent->vx;
	agent->y += agent->vy;
	agent->age += dt;

	agent->velocity_ums = sqrt(agent->diff_rate_um_x_twice_dof * agent->age) / agent->age;

	agent->closest_cell_id = -1;
	agent->closest_cell_distance = -1.0f;
	agent->closest_ev_id = -1;
	agent->closest_ev_distance = -1.0f;

    return 0;
}

// This function can add up tu N agents between steps;
__FLAME_GPU_FUNC__ int secrete_ev(xmachine_memory_SecretoryCell* agent, xmachine_memory_EV_list* EVs, RNG_rand48* rand48){
	if (introduce_n_new_evs > 0)
	{
		// introduce new agents if the last introduction was longer ago than 
		// get_seconds_before_introducing_new_evs() and 
		// a RNG goes above get_new_evs_threshold() and
		//  there is space in the environment
		// (xmachine_memory_EV_MAX - get_agent_EV_default_count() > 0
		if(agent->time_since_last_secreted > seconds_before_introducing_new_evs){
			
			float rn = rnd<CONTINUOUS>(rand48);
			
			if( rn > new_evs_threshold)
			{
				int rand_i = (int)(rnd<CONTINUOUS>(rand48) * 120);
				// setup the new agent
				// our simulations use 30-150 nm EVs
				float diameter_nm = (rand_i % 120) + 30;
				float diameter_um = diameter_nm * 0.001;// / 1000;

				float radius_nm = diameter_nm * .5;
				float radius_um = diameter_um * .5; // ((30-150 nm) * 0.001/2) // (15 to 75) 
				float radius_m = radius_um * 1E-6; // / 1e+6;

				// compute the volume
				float volume = const_pi_4div3 * radius_nm * radius_nm * radius_nm;

				// to convert:
				// N square metres to square micrometre: multiply N * 1e+12
				// N metres to micrometres: multiply N * 1e+6
				float diffusion_rate_m = const_Boltzmann_x_Temp_K / (const_6_pi_viscosity * radius_m); // square metre
				float diffusion_rate_um = diffusion_rate_m * 1e+12; // square micrometre

				float velocity_um = sqrt(4 * diffusion_rate_um);
				float velocity_m = sqrt(4 * diffusion_rate_m);

				// decompose velocity
				float vx = 0;
				float vy = 0; 
				if (new_evs_random_direction > 0) {
					float r = 2 * M_PI * rnd<CONTINUOUS>(rand48);
					vx = velocity_um * cos(r);
					vy = velocity_um * sin(r);
				}
				else {
					vx = velocity_um * agent->direction_x;
					vy = velocity_um * agent->direction_y;
				}

				// displace the ev by a sigle step
				float x = agent->x + vx * dt;
				float y = agent->y + vy * dt;
				unsigned int id = generate_EV_id();
				
				add_EV_agent(EVs, id, x, y, 0,
					agent->x, agent->y, vx, vy, volume * const_mass_per_volume_unit, 0, // <- colour
					radius_um, radius_m, diffusion_rate_m, diffusion_rate_um, 
					diffusion_rate_um * 2 * dof,
					-1, -1, -1, -1, 0, 
					agent->direction_x, agent->direction_y, velocity_um, velocity_m);

				agent->time_since_last_secreted = 0;
			}
			else
				agent->time_since_last_secreted += dt;
		}
		else
			agent->time_since_last_secreted += dt;
	}
	return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
