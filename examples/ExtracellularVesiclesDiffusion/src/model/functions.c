
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

int exiting_early = 0;

__device__ __host__ float degrees(float radians){
	return radians * (180 / M_PI);
}

__device__ __host__ float radians(float degrees){
	return degrees * (M_PI / 180.0);
}

/**
* Compute the distance between two agents using the coordinates of their centers
*/
__device__ float euclidean_distance(xmachine_memory_EV* agent, 
	float message_x, float message_y)
{
	return sqrt((agent->x - message_x) * (agent->x - message_x) +
		(agent->y - message_y) * (agent->y - message_y));
}

__FLAME_GPU_INIT_FUNC__ void precompute_values() {
	srand(time(NULL));
	printf("Simulation timestep, dt=%f\n", *get_dt());
	if (*get_ev_collisions() > 0) {
		printf("Ev-Ev collisions are enabled\n");
	}
	else {
		printf("Ev-Ev collisions are disabled\n");
	}
	if (*get_boundaries() > 0) {
		printf("Collision with the boundaries are enabled\n");
	}
	else {
		printf("Collision with the boundaries are disabled\n");
	}
	if (*get_drag() > 0) {
		printf("Drag is enabled\n");
	}
	else {
		printf("Drag is disabled\n");
	}
	if (*get_brownian_motion_1d() > 0) {
		printf("brownian_motion_1d is enabled\n");
	}
	else {
		printf("brownian_motion_1d is disabled\n");
	}
	if (*get_brownian_motion_2d() > 0) {
		printf("brownian_motion_2d is enabled\n");
	}
	else {
		printf("brownian_motion_2d is disabled\n");
	}
	if (*get_apoptosis() > 0) {
		printf("Apoptosis is enabled, threshold: %f\n", *get_apoptosis_threshold());
	} else {
		printf("Apoptosis is disabled\n");
	}
	iteration_number = 0;
	if (*get_ev_secretion() > 0) {
		printf("EV secretion is enabled, interval %f seconds, p. threshold of %f\n",
			*get_ev_secretion_interval(), *get_ev_secretion_threshold());
		printf("Min radius: %d, Max radius: %d\n", *get_min_ev_radius(), *get_max_ev_radius());
	}
	else {
		printf("EV secretion is disabled\n");
	}
	float val0 = 6 * M_PI * *get_const_water_dynamic_viscosity_pas();
	set_const_6_pi_dynamic_viscosity(&val0);
	printf("set_const_6_pi_dynamic_viscosity: %f\n", val0);
	float val1 = 6 * M_PI * *get_const_water_kinematic_viscosity_kg_ums();
	set_const_6_pi_kinematic_viscosity(&val1);
	printf("set_const_6_pi_kinematic_viscosity: %f\n", val1);
	float val2 = *get_const_Boltzmann() * *get_const_Temperature_K();
	printf("boltzmann: %.35f\n", *get_const_Boltzmann());
	set_const_boltzmann_x_temp(&val2);
	printf("set_const_boltzmann_x_temp: %.30f\n", val2);
}

/**
 * output_data FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_EV. Represents an agent instance, can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list.
 */
__FLAME_GPU_FUNC__ int output_location_ev_default(xmachine_memory_EV* agent, xmachine_message_location_ev_default_list* location_messages){
	add_location_ev_default_message(location_messages, agent->id, agent->x, agent->y,
		agent->z, agent->radius_um, agent->mass_ag, agent->vx, agent->vy);
    return 0;
}
__FLAME_GPU_FUNC__ int output_location_ev_initial(xmachine_memory_EV* agent, xmachine_message_location_ev_initial_list* location_messages) {
	add_location_ev_initial_message(location_messages, agent->id, agent->x, agent->y,
			agent->z, agent->radius_um, agent->mass_ag, agent->vx, agent->vy);
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
		agent->p1_x, agent->p1_y, agent->p2_x, agent->p2_y, agent->direction_x, agent->direction_y,
		agent->direction_x_unit, agent->direction_y_unit, agent->direction_length, 
		agent->normal_x, agent->normal_y, agent->unit_normal_x, agent->unit_normal_y, agent->normal_length
	);
	return 0;
}

__FLAME_GPU_FUNC__ int output_secretory_cell_location(xmachine_memory_SecretoryCell* agent, xmachine_message_secretory_cell_location_list* location_messages) {
	add_secretory_cell_location_message(location_messages, agent->id, agent->x, agent->y, 0,
		agent->p1_x, agent->p1_y, agent->p2_x, agent->p2_y, agent->direction_x, agent->direction_y,
		agent->direction_x_unit, agent->direction_y_unit, agent->direction_length, 
		agent->normal_x, agent->normal_y, agent->unit_normal_x, agent->unit_normal_y, agent->normal_length
	);
	return 0;
}

__device__ float dotprod(float x1, float y1, float x2, float y2) {
	return x1*x2 + y1*y2;
}

__device__ __host__ float vlength(float x, float y) {
	return sqrt(x*x + y*y);
}

// https://github.com/takagi/cl-cuda/blob/master/include/float4.h
__device__ float2 float2_add(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
__device__ float4 float4_add(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ float2 float2_sub(float2 a, float2 b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
__device__ float4 float4_sub(float4 a, float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ float2 float2_scale(float2 a, float k)
{
	return make_float2(a.x * k, a.y * k);
}
__device__ float4 float4_scale(float4 a, float k)
{
	return make_float4(a.x * k, a.y * k, a.z * k, a.w * k);
}
__device__ float float2_dot(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}
__device__ float float4_dot(float4 a, float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/*
Scales b by scale and adds it to a
*/
__device__ float2 add_scaled(float2 a, float2 b, float scale) {
	return float2_add(a, float2_scale(b, scale));
}

/*
Creates a vector of lenght u parallel to vec
*/
__device__ float2 parallel(float vec_x, float vec_y, float u) {
	float factor = (u / vlength(vec_x, vec_y));
	return make_float2(vec_x * factor, vec_y * factor);
}
/*
Returns the length of the projection of vector A in the direction of vector B
*/
__device__ float projection(float a_x, float a_y, float b_x, float b_y) {
	float length = vlength(a_x, a_y);
	float lengthVec = vlength(b_x, b_y);
	if (length == 0 || lengthVec == 0)
		return 0;
	return (a_x * b_x + a_y * b_y) / lengthVec;
}

__device__ float2 float2_project(float2 v1, float2 v2) {
	float mag = projection(v1.x, v1.y, v2.x, v2.y);
	return parallel(v2.x, v2.y, mag);
}
__device__ float2 project(float v1x, float v1y, float v2x, float v2y) {
	return float2_project(make_float2(v1x, v1y), make_float2(v2x, v2y));
}

__FLAME_GPU_STEP_FUNC__  void checkStopConditions() {
	if(exiting_early > 0){
		set_exit_early();
	}
	float dt = *get_dt();
	for(int i=0; i < get_agent_EV_default_count(); i++) {
		float x = get_EV_default_variable_x(i);
		float y = get_EV_default_variable_y(i);
		int id = get_EV_default_variable_id(i);
		int nanInfVel = get_EV_default_variable_debugNanInfVelocity(i);
		int nanInfLoc = get_EV_default_variable_debugNanInfLocation(i);
		int nanInfValue = get_EV_default_variable_debugNanInfValue(i);
		int dragVel0At = get_EV_default_variable_debugDragVel0At(i);

		float div0 = get_EV_default_variable_debugDiv0(i);

		if(!isfinite(x) || !isfinite(y)) {
			printf("EV id:%d has NaN or Inf coordinates.\n", id);
			++exiting_early;
		}
		if(nanInfVel != 0){
			printf("EV id:%d produced NaN or Inf velocity at %d.\n", id, nanInfVel);
			++exiting_early;
		}
		if(nanInfLoc != 0){
			printf("EV id:%d produced NaN or Inf location at %d.\n", id, nanInfLoc);
			++exiting_early;
		}
		if(div0 != 0){
			printf("EV id:%d did DIV/0 in %f.\n", id, div0);
			++exiting_early;
		}
		if(nanInfValue != 0){
			printf("NaN or Inf value produced for EV id:%d at %d.\n", id, nanInfValue);
			++exiting_early;
		}
		float corrAppX = get_EV_default_variable_colInfCorrectionAppliedX(i);
		float corrAppY = get_EV_default_variable_colInfCorrectionAppliedY(i);
		float vx = get_EV_default_variable_vx(i);
		float vy = get_EV_default_variable_vy(i);
		float bm_vx = get_EV_default_variable_bm_vx(i);
		float bm_vy = get_EV_default_variable_bm_vy(i);
		float correctionApplied = vlength(corrAppX, corrAppY);
		float velocity_plus_bm = (vlength(vx, vy) + vlength(bm_vx, bm_vy)) * (*get_dt() * 2);
		if(correctionApplied > velocity_plus_bm) {
			printf("EV id:%d correction applied > 2*dt*(|v|+|bm|) (%f > %f). Will be leaving early\n",
				id, correctionApplied, velocity_plus_bm);
			++exiting_early;
		}
	}
}

__device__ int checkNanInfVelocity(xmachine_memory_EV* agent, float caller_id){
    if(agent->debugNanInfVelocity == 0){
		if(!isfinite(agent->vx)){
			agent->debugNanInfVelocity = caller_id;
		}
	}
	if(agent->debugNanInfVelocity == 0){
        if(!isfinite(agent->vy)){
            agent->debugNanInfVelocity = caller_id + 1;
        }
    }
    return 0;
}

__device__ int checkNanInfLocation(xmachine_memory_EV* agent, float caller_id){
    if(agent->debugNanInfLocation == 0){
        if(!isfinite(agent->x)){
            agent->debugNanInfLocation = caller_id;
        }
	}
	if(agent->debugNanInfLocation == 0){
        if(!isfinite(agent->y)){
            agent->debugNanInfLocation = caller_id + 1;
        }
    }
    return 0;
}

__device__ int checkDivZero(xmachine_memory_EV* agent, float value, float caller_id) {
	if(agent->debugDiv0 == 0) {
		if(abs(value) < 1e-6) {
			agent->debugDiv0 = caller_id;
		}
	}
	return 0;
}

__device__ int checkDivZero(xmachine_memory_EV* agent, float2 value, float caller_id){
	if(agent->debugDiv0 == 0) {
		if(abs(value.x) < 1e-6 || abs(value.y) < 1e-6) {
			agent->debugDiv0 = caller_id;
		}
	}
	return 0;
}

__device__ int checkValueNanInf(xmachine_memory_EV* agent, float value, float caller_id){
	if(agent->debugNanInfValue == 0){
		if(!isfinite(value)) {
			agent->debugNanInfValue = caller_id;
		}
	}
	return 0;
}

__device__ int checkValueNanInf(xmachine_memory_EV* agent, float2 value, float caller_id){
	if(agent->debugNanInfValue == 0){
		if(!isfinite(value.x)) {
			agent->debugNanInfValue = caller_id;
		} else if(!isfinite(value.y)) {
			agent->debugNanInfValue = caller_id + 1;
		}
	}
	return 0;
}

__device__ float4 vectors_from_ev_to_wall_endpoints(xmachine_memory_EV* agent, const float & p1_x, const float & p1_y,
	const float & p2_x, const float & p2_y) {
	return make_float4(p1_x - agent->x, p1_y - agent->y, p2_x - agent->x, p2_y - agent->y);
}

__device__ float2 project_vectors_onto_wall(float4 ev_wall, float cell_direction_x, float cell_direction_y) {
	return make_float2(
		projection(ev_wall.x, ev_wall.y, cell_direction_x, cell_direction_y),
		projection(ev_wall.z, ev_wall.w, cell_direction_x, cell_direction_y));
}

__device__ float2 perpendicular_distance_from_ev_to_wall(float4 ev_wall, float proj1, float cell_direction_x_unit, float cell_direction_y_unit) {
	// add_scaled(ballp1, wdir.unit(), proj1*-1)
	return make_float2(
		ev_wall.x + cell_direction_x_unit * -proj1,
		ev_wall.y + cell_direction_y_unit * -proj1
	);
}

__device__ int drag_force(xmachine_memory_EV* agent) {
	float vx = agent->vx + agent->bm_vx;
	float vy = agent->vy + agent->bm_vy;
	float vel_mag_ums = vlength(vx, vy); // in (um/s)
	if(vel_mag_ums < 1e-8){
		// Drag cannot produce negative velocity values
		agent->vx = 0;
		agent->vy = 0;
		if (agent->debugDragVel0At != -1){
			agent->debugDragVel0At = agent->age;
		}
	} else {
		float2 vel_unit = make_float2(vx / vel_mag_ums, vy / vel_mag_ums);
		// kinematic viscosity in kg/um.s
		float factor = const_6_pi_kinematic_viscosity * agent->radius_um * vel_mag_ums;
		float2 vel = float2_scale(vel_unit, factor);
		agent->vx = vel.x;
		agent->vy = vel.y;
		checkNanInfVelocity(agent, 1550);
	}
	return 0;
}

// solves direct collisions between EV and segments
__device__  int solve_segment_collision(xmachine_memory_EV* agent, float cell_dir_x, 
	float cell_dir_y, float cell_direction_length, float perp_dist_x, float perp_dist_y,
	float cell_unit_normal_x, float cell_unit_normal_y) {
	// 1. angle between velocity and wall 
	float angle = acosf(dotprod(agent->vx, agent->vy, cell_dir_x, cell_dir_y) 
		/ (vlength(agent->vx, agent->vy) * cell_direction_length));
	// 2. reposition object
	float normal_x = cell_unit_normal_x;
	float normal_y = cell_unit_normal_y;

	// 3. compute deltaS
	float dist_dp_normal = dotprod(perp_dist_x, perp_dist_y, normal_x, normal_y);
	float deltaS = (agent->radius_um + dist_dp_normal) / sin(angle);

	// 4. estimate what was the displacement = velocity parallel to delta
	float2 displ = parallel(agent->vx, agent->vy, deltaS);

	// 5. update position by subtracting the displacement
	agent->x -= displ.x;
	agent->y -= displ.y;

	// velocity correction factor
	//var vcor = 1-acc.dotProduct(displ)/obj.velo2D.lengthSquared();
	//float numerator = dotprod(acceleration_x, acceleration_y, displ_x, displ_y);
	//float sqr_vel = (agent->vx * agent->vx + agent->vy*agent->vy);
	//float vfrac = numerator / sqr_vel;
	//float vcor = 1 - vfrac;

	//printf("displx:%f disply:%f numerator:%f sqr_vel:%f vfrac:%f vcor:%f\n", displ_x, displ_y, numerator, sqr_vel, vfrac, vcor);
	// corrected velocity vector just before impact 
	// var Velo = obj.velo2D.multiply(vcor)
	//float new_velocity_x = agent->vx;// *vcor;
	//float new_velocity_y = agent->vy;// *vcor;

	// 6. decompose the velocity
	// velocity vector component perpendicular to wall just before impact
	float velocityProjection = projection(agent->vx, agent->vy, perp_dist_x, perp_dist_y);
	float2 normalVelocity = parallel(perp_dist_x, perp_dist_y, velocityProjection);

	// velocity vector component parallel to wall; unchanged by impact
	float tangentVelocity_x = agent->vx - normalVelocity.x;
	float tangentVelocity_y = agent->vy - normalVelocity.y;

	// velocity vector component perpendicular to wall just after impact
	float new_vx = tangentVelocity_x + (-normalVelocity.x);
	float new_vy = tangentVelocity_y + (-normalVelocity.y);
	if (isfinite(new_vx) && isfinite(new_vy)){
		agent->vx = new_vx;
		agent->vy = new_vy;
	} else {
		checkValueNanInf(agent, new_vx, caller_id + 70);
		checkValueNanInf(agent, new_vy, caller_id + 75);
	}
	checkNanInfVelocity(agent, caller_id + 70);
	return 0;
}

__device__  int solve_segment_end_point_collision(xmachine_memory_EV* agent, 
	float wall_pt_x, float wall_pt_y, float caller_id
) {
	// bounce off endpoint wall_p
	float distp_x = agent->x - wall_pt_x;
	float distp_y = agent->y - wall_pt_y;

	// move particle so that it just touches the endpoint			
	float L = agent->radius_um - vlength(distp_x, distp_y);
	float vrel = vlength(agent->vx, agent->vy);

	checkNanInfLocation(agent, caller_id + 1);
	checkDivZero(agent, vrel, caller_id + 5);
	agent->x += agent->vx * (-L / vrel);
	agent->y += agent->vy * (-L / vrel);

	checkNanInfVelocity(agent, caller_id + 6);

	// normal velocity vector just before the impact
	float2 normalVelo = project(agent->vx, agent->vy, distp_x, distp_y);
	// tangential velocity vector
	float tangentVelo_x = agent->vx - normalVelo.x;
	float tangentVelo_y = agent->vy - normalVelo.y;
	// normal velocity vector after collision

	//// final velocity vector after collision
	agent->vx = -1.0 * normalVelo.x + tangentVelo_x;
	agent->vy = -1.0 * normalVelo.y + tangentVelo_y;
	checkNanInfVelocity(agent, caller_id + 10);

	return 0;
}

/**
Collision resolution algorithm modified from Physics for Javascript Games, Animation, and Simulation Ch, 11
ID 3000
*/
__FLAME_GPU_FUNC__ int secretory_cell_collision_resolution(xmachine_memory_EV* agent,
	xmachine_message_secretory_cell_collision_list* secretory_cell_collision_messages,
	xmachine_message_secretory_cell_collision_PBM* partition_matrix){

	xmachine_message_secretory_cell_collision* message = get_first_secretory_cell_collision_message(secretory_cell_collision_messages, partition_matrix, agent->x, agent->y, agent->z);

	while (message){
		// we verify the ev_id in the message matches this agent's id
		if (message->ev_id == agent->id) {
			float4 ev_wall_vectors = vectors_from_ev_to_wall_endpoints(agent, message->p1_x, message->p1_y, message->p2_x, message->p2_y);
			float2 projections = project_vectors_onto_wall(ev_wall_vectors, message->cell_direction_x, message->cell_direction_y);
			float2 perp_dist = perpendicular_distance_from_ev_to_wall(ev_wall_vectors, projections.x, message->cell_direction_x_unit, message->cell_direction_y_unit);

			bool test_needed = ((abs(projections.x) < message->cell_direction_length)
					&& (abs(projections.y) < message->cell_direction_length));

			if ((vlength(perp_dist.x, perp_dist.y) < agent->radius_um) && test_needed) {
				agent->debugExternPerpDistX = perp_dist.x;
				agent->debugExternPerpDistY = perp_dist.y;
				agent->debugExternProjX = projections.x;
				agent->debugExternProjY = projections.y;
				agent->debugExternCellDirLen = message->cell_direction_length;
				agent->debugExternCellDirX = message->cell_direction_x;
				agent->debugExternCellDirY = message->cell_direction_y;

				solve_segment_collision(agent, message->cell_direction_x,
					message->cell_direction_y, message->cell_direction_length,
					perp_dist.x, perp_dist.y,
					message->unit_normal_x, message->unit_normal_y, 3100.0);
			}
			else if (abs(vlength(ev_wall_vectors.x, ev_wall_vectors.y)) < agent->radius_um) {
				// collision with 1st end point
				solve_segment_end_point_collision(agent, message->p1_x, message->p1_y, 3200.0);
			}
			else if (abs(vlength(ev_wall_vectors.z, ev_wall_vectors.w)) < agent->radius_um) {
				// collision with 2nd end point
				solve_segment_end_point_collision(agent, message->p2_x, message->p2_y, 3300.0);
			}
		}
		message = get_next_secretory_cell_collision_message(message, secretory_cell_collision_messages, partition_matrix);
	}
	return 0;
}

/**
This function resolves a single collision between an EV and a ciliary cell
After this, the EV has new values for x, y, vx, and vy, compensated for the collision
ID 4000
*/
__FLAME_GPU_FUNC__ int ciliary_cell_collision_resolution(xmachine_memory_EV* agent, 
	xmachine_message_ciliary_cell_collision_list* ciliary_cell_collision_messages,
	xmachine_message_ciliary_cell_collision_PBM* partition_matrix) {

	xmachine_message_ciliary_cell_collision* message = get_first_ciliary_cell_collision_message(ciliary_cell_collision_messages, partition_matrix, agent->x, agent->y, agent->z);
	//float acceleration_x = 0, acceleration_y = 0;
	while (message) {
		// we verify the ev_id in the message matches this agent's id
		if (message->ev_id == agent->id) {
			float4 ev_wall_vectors = vectors_from_ev_to_wall_endpoints(agent, message->p1_x, message->p1_y, message->p2_x, message->p2_y);
			float2 projections = project_vectors_onto_wall(ev_wall_vectors, message->cell_direction_x, message->cell_direction_y);
			float2 perp_dist = perpendicular_distance_from_ev_to_wall(ev_wall_vectors, projections.x, message->cell_direction_x_unit, message->cell_direction_y_unit);

			bool test_needed = (abs(projections.x) < message->cell_direction_length) && (abs(projections.y) < message->cell_direction_length);
			bool tunneling = false;

			if ((vlength(perp_dist.x, perp_dist.y) < agent->radius_um || tunneling) && test_needed) {
				agent->debugExternPerpDistX = perp_dist.x;
				agent->debugExternPerpDistY = perp_dist.y;
				agent->debugExternProjX = projections.x;
				agent->debugExternProjY = projections.y;
				agent->debugExternCellDirLen = message->cell_direction_length;
				agent->debugExternCellDirX = message->cell_direction_x;
				agent->debugExternCellDirY = message->cell_direction_y;

				solve_segment_collision(agent, message->cell_direction_x,
					message->cell_direction_y, message->cell_direction_length,
					perp_dist.x, perp_dist.y, message->unit_normal_x, message->unit_normal_y, 4100.0);
			}
			else if (abs(vlength(ev_wall_vectors.x, ev_wall_vectors.y)) < agent->radius_um) {
				// collision with 1st end point
				solve_segment_end_point_collision(agent, message->p1_x, message->p1_y, 4200.0);
			}
			else if (abs(vlength(ev_wall_vectors.z, ev_wall_vectors.w)) < agent->radius_um) {
				// collision with 2nd end point
				solve_segment_end_point_collision(agent, message->p2_x, message->p2_y, 4300.0);
			}
		}
		message = get_next_ciliary_cell_collision_message(message, ciliary_cell_collision_messages, partition_matrix);
	}
	return 0;
}

/**
Computes the distance between a point (p0) and the segment of a line with end points (p1) and (p2).
We get the vectors A = p0 - p1 and B = p2 - p1. 
-param- is the scalar value that when multiplied by B gives you the point on the line closest to p0.
A collision is possible if the point:
	- is projecting on the segment, 
	- is not yet proyecting on the segment but displacing towards the segment and within a shorter distance from the endpoints than its radius
Modified from https://stackoverflow.com/a/6853926/3830240

Returns
float4 where:
	x - 1 if the point is projecting on the segment, 0 otherwise
	y - distance from the closest point on the segment to the EV position
	z - x coordinate of the closest point from the segment to the EV
	w - y coordinate of the closest point from the segment to the EV
*/
__device__ float4 point_to_line_segment_distance(float x, float y, float x1, float y1, float x2, float y2)
{
	float4 result;

	// vector components from point to one of the end points of the line segment
	float A = x - x1;
	float B = y - y1;
	// vector components of the line segment
	float C = x2 - x1;
	float D = y2 - y1;

	float dot = A * C + B * D; // vector projection
	float len_sq = C * C + D * D; // sqrd magnitude of the line segment
	float param = -1;
	float xx = 0.f, yy = 0.f, dx, dy;
	//float projecting_on_segment = 0.0f;

	if (len_sq != 0){ // in case of 0 length line
		param = dot / len_sq; // ratio of the projected vectors over the length of the line segment
	}
	
	if (param < 0){ // the closest point is p1, point not projecting on segment
		xx = x1;
		yy = y1;
	} 
	else if (param > 1){ // the closest point is p2, point not projecting on segment
		xx = x2;
		yy = y2;
	}
	else {   // the point is projecting on the segment
		// interpolate to find the closes point on the line segment
		//projecting_on_segment = 1.0f;
		xx = x1 + param * C;
		yy = y1 + param * D;
	}

	// dx / dy is the vector from p0 to the closest point on the segment
	dx = x - xx;
	dy = y - yy;

	result.x = param;
	result.y = sqrt(dx * dx + dy * dy); // distance between p
	result.z = xx;
	result.w = yy;
	
	return result;
}

/**
Checks any possible collision with the secretory cells on the boundaries. 
If a collision occurred, a collision_message is generated and the agent will have the values:
	agent->closest_secretory_cell_id = closest_cell_id;
	agent->closest_secretory_cell_distance = closest_cell_distance;
Otherwise, no message is produced and the ev agent will have the default values unchanged:
	agent->closest_secretory_cell_id = -1;
	agent->closest_secretory_cell_distance = -1.0f;
*/
__FLAME_GPU_FUNC__ int test_secretory_cell_collision(xmachine_memory_EV* agent, xmachine_message_secretory_cell_location_list* location_messages,
	xmachine_message_secretory_cell_location_PBM* partition_matrix, xmachine_message_secretory_cell_collision_list* secretory_cell_collision_messages)
{
	int closest_cell = -2;
	float closest_cell_distance = 100.f;
	float separation_x_1k = 1000.f;
	float4 res;
	float4 wall_direction;
	float3 wall_normal;
	float2 wall_unit_normal;
	float4 wall_pts;
	float wall_direction_length;

	xmachine_message_secretory_cell_location* message = get_first_secretory_cell_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	while (message) {
		// we only check for collisions if the EV: -is displacing in opposite or perpendicular direction to the wall
		float dotProduct = dotprod(agent->vx, agent->vy, message->unit_normal_x, message->unit_normal_y);

		// the dot product is:
		//	<0 for vectors in opposing directions
		//  =0 for those perpendicular to each other
		// If unormal and direction are perpendicular to each other
		// the displacement may be parallel or collinear to the wall
		if (dotProduct <= 0) {
			// for each agent's new position and a cell compute:
			// x - ratio of the segment where the position is projecting or 0,1 if outside
			// y - distance between position and closest point on the segment
			// z,w - coordinates where the position is projecting on the segment
			res = point_to_line_segment_distance(agent->x, agent->y,
				message->p1_x, message->p1_y, message->p2_x, message->p2_y);

			// if the ev new position is projecting over the segment
			// and the distance to the segment < 1.5 x radius
			if (res.y < agent->radius_um * 1.5)
			{
				if (res.y < closest_cell_distance)
				{
					closest_cell = message->id;
					closest_cell_distance = res.y;
					separation_x_1k = (res.y - agent->radius_um) * 1000.f;
					
					// wall direction values
					wall_direction.x = message->direction_x;
					wall_direction.y = message->direction_y;
					// wall direction unit values
					wall_direction.z = message->direction_x_unit;
					wall_direction.w = message->direction_y_unit;
					wall_direction_length = message->direction_length;
					// wall normal values
					wall_normal.x = message->normal_x;
					wall_normal.y = message->normal_y;
					wall_normal.z = message->normal_length;
					// wall unit normal values
					wall_unit_normal.x = message->unit_normal_x;
					wall_unit_normal.y = message->unit_normal_y;
					
					wall_pts.x = message->p1_x;
					wall_pts.y = message->p1_y;
					wall_pts.z = message->p2_x;
					wall_pts.w = message->p2_y;
				}
			}
		}

		// get the next message
		message = get_next_secretory_cell_location_message(message, location_messages, partition_matrix);
	}

	if (closest_cell > -1) { // a potential collision was detected
		// we store the reference in the agent for future comparisons
		agent->closest_secretory_cell_id = closest_cell;
		agent->closest_secretory_cell_distance = closest_cell_distance;
		agent->colInfSeparation_x_1k = separation_x_1k;
		// write the corresponding collision_message
		add_secretory_cell_collision_message(secretory_cell_collision_messages, agent->id,
			agent->x, agent->y, agent->z,
			agent->closest_secretory_cell_id, agent->closest_secretory_cell_distance,
			wall_pts.x, wall_pts.y, wall_pts.z, wall_pts.w, 
			wall_direction.x, wall_direction.y, wall_direction.z, wall_direction.w, wall_direction_length,
			wall_normal.x, wall_normal.y, wall_unit_normal.x, wall_unit_normal.y, wall_normal.z);
	}
	return 0;
}

__FLAME_GPU_FUNC__ int test_ciliary_cell_collision(xmachine_memory_EV* agent, xmachine_message_ciliary_cell_location_list* location_messages,
	xmachine_message_ciliary_cell_location_PBM* partition_matrix, xmachine_message_ciliary_cell_collision_list* ciliary_cell_collision_messages)
{
	int closest_cell = -2;
	float closest_cell_distance = 100.f;
	float separation_x_1k = 1000.f;
	float4 res;
	float4 wall_direction;
	float3 wall_normal;
	float2 wall_unit_normal;
	float4 wall_pts;
	float wall_direction_length;

	xmachine_message_ciliary_cell_location* message = get_first_ciliary_cell_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	while (message) {
		// we only check for collisions if the EV: -is displacing in opposite or perpendicular direction to the wall
		float dotProduct = dotprod(agent->vx, agent->vy, message->unit_normal_x, message->unit_normal_y);

		// the dot product will be negative for vectors in opposite directions and zero for those perpendicular to each other
		if (dotProduct <= 0) {
			// get the distance between agent and cell
			res = point_to_line_segment_distance(agent->x, agent->y,
				message->p1_x, message->p1_y, message->p2_x, message->p2_y);

			// if the distance to segment < radius
			if (res.y < agent->radius_um * 1.5)
			{
				//printf("ciliary res.x: %f res.y:%f radius_um:%f\n", res.x, res.y, agent->radius_um);
				// save the reference if this is the first candidate collision
				// or if the agent is closer to the wall than the previous collision
				if (res.y < closest_cell_distance)
				{
					closest_cell = message->id;
					closest_cell_distance = res.y;
					separation_x_1k = (res.y - agent->radius_um) * 1000.f;
					// wall direction values
					wall_direction.x = message->direction_x;
					wall_direction.y = message->direction_y;
					// wall direction unit values
					wall_direction.z = message->direction_x_unit;
					wall_direction.w = message->direction_y_unit;
					wall_direction_length = message->direction_length;
					// wall normal values
					wall_normal.x = message->normal_x;
					wall_normal.y = message->normal_y;
					wall_normal.z = message->normal_length;
					// wall unit normal values
					wall_unit_normal.x = message->unit_normal_x;
					wall_unit_normal.y = message->unit_normal_y;

					wall_pts.x = message->p1_x;
					wall_pts.y = message->p1_y;
					wall_pts.z = message->p2_x;
					wall_pts.w = message->p2_y;
				}
			}
		}
		message = get_next_ciliary_cell_location_message(message, location_messages, partition_matrix);
	}
	
	if (closest_cell > -1) { // a collision was detected
		// agent is closest to the detected ciliary cell
		agent->closest_ciliary_cell_id = closest_cell;
		agent->closest_ciliary_cell_distance = closest_cell_distance;
		agent->colInfSeparation_x_1k = separation_x_1k;
		// write the corresponding collision_message
		add_ciliary_cell_collision_message(ciliary_cell_collision_messages, agent->id,
			agent->x, agent->y, agent->z,
			agent->closest_ciliary_cell_id, agent->closest_ciliary_cell_distance,
			wall_pts.x, wall_pts.y, wall_pts.z, wall_pts.w, wall_direction.x, wall_direction.y,
			wall_direction.z, wall_direction.w, wall_direction_length,
			wall_normal.x, wall_normal.y, wall_unit_normal.x, wall_unit_normal.y, wall_normal.z);
	}
	return 0;
}

struct PostCollisionData {
	float2 correctedLocation;
	float2 correctionApplied;
	float2 correctedVelocity;
	float correctionFactor;
	float overlap;
	float vrel;
};
/*
Computes the new position and velocity after a collision for an agent.
It can compute the values for two agents involved, however, we can only update one
active agent at a time
*/
__device__ struct PostCollisionData solve_collision_ev_default_ev_default(float2 ev1_loc, float2 ev1_velo, float ev1_mass_ag,
	float2 ev2_loc, float2 ev2_velo, float ev2_mass_ag, float min_distance, float2 dist, float dist_length) {
	struct PostCollisionData pcd;
	// normal velocity vectors just before the impact
	float2 normal_velocity1 = float2_project(ev1_velo, dist);
	float2 normal_velocity2 = float2_project(ev2_velo, dist);
	// tangential velocity vectors
	float2 tangent_velocity1 = float2_sub(ev1_velo, normal_velocity1);

	// move particles so that they just touch
	pcd.overlap = min_distance - dist_length;
	float2 normal_velo_subtracted = float2_sub(normal_velocity1, normal_velocity2);
	pcd.vrel = vlength(normal_velo_subtracted.x, normal_velo_subtracted.y);
	pcd.correctionFactor = pcd.overlap / pcd.vrel;
	pcd.correctedLocation = add_scaled(ev1_loc, normal_velocity1, -pcd.correctionFactor);
	pcd.correctionApplied = float2_scale(normal_velocity1, -pcd.correctionFactor);
	
	// normal velocity components after the impact
	float u1 = projection(normal_velocity1.x, normal_velocity1.y, dist.x, dist.y);
	float u2 = projection(normal_velocity2.x, normal_velocity2.y, dist.x, dist.y);
	float v1 = ((ev1_mass_ag - ev2_mass_ag)*u1 + 2 * ev2_mass_ag*u2) / (ev1_mass_ag + ev2_mass_ag);

	normal_velocity1 = parallel(dist.x, dist.y, v1);

	pcd.correctedVelocity = float2_add(normal_velocity1, tangent_velocity1);
	return pcd;
}

struct PostCollisionDataInitial {
	float2 correctedPosition;
	float2 correctionApplied;
	float2 correctedVelocity;
	float correctionFactor;
	float overlap;
	float vrel;
	float vnorm;
	float2 auxiliaryFactors;
};
__device__ struct PostCollisionDataInitial solve_collision_ev_default_ev_initial(float2 ev1_loc, float2 ev1_velo, float ev1_mass_ag,
	float2 ev2_loc, float2 ev2_velo, float ev2_mass_ag, float min_distance, float2 dist, float dist_length) {
	struct PostCollisionDataInitial pcdi;

	// normal velocity vectors just before the impact
	float2 normal_velocity1 = float2_project(ev1_velo, dist);
	float2 normal_velocity2 = float2_project(ev2_velo, dist);
	// tangential velocity vectors
	float2 tangent_velocity1 = float2_sub(ev1_velo, normal_velocity1);

	// move particles so that they just touch
	pcdi.overlap = min_distance - dist_length;
	float2 normal_velo_subtracted = float2_sub(normal_velocity1, normal_velocity2);
	pcdi.vrel = vlength(normal_velo_subtracted.x, normal_velo_subtracted.y);
	pcdi.vnorm = vlength(ev1_velo.x, ev1_velo.y);
	float dp = ev1_velo.x*ev2_velo.x + ev1_velo.y*ev2_velo.y;
	pcdi.auxiliaryFactors.x = pcdi.overlap / pcdi.vrel;
	pcdi.auxiliaryFactors.y = pcdi.auxiliaryFactors.x > 100 ? pcdi.auxiliaryFactors.x /10 : pcdi.auxiliaryFactors.x;
	pcdi.correctionFactor = (pcdi.overlap * 4) / pcdi.vnorm;
	if(dp < 0){
		pcdi.auxiliaryFactors.y = -pcdi.auxiliaryFactors.y;
		pcdi.correctionFactor = -pcdi.correctionFactor;
	}
	//float2 new_ev1_loc = add_scaled(ev1_loc, normal_velocity1, pcdi.correctionFactors.y);
	pcdi.correctedPosition = add_scaled(ev1_loc, normal_velocity1, pcdi.correctionFactor);
	pcdi.correctionApplied = float2_scale(normal_velocity1, -pcdi.correctionFactor);

	// normal velocity components after the impact
	float u1 = projection(normal_velocity1.x, normal_velocity1.y, dist.x, dist.y);
	float u2 = projection(normal_velocity2.x, normal_velocity2.y, dist.x, dist.y);
	float v1 = ((ev1_mass_ag - ev2_mass_ag)*u1 + 2 * ev2_mass_ag*u2) / (ev1_mass_ag + ev2_mass_ag);
	normal_velocity1 = parallel(dist.x, dist.y, v1);
	pcdi.correctedVelocity = float2_add(normal_velocity1, tangent_velocity1);

	return pcdi;
}

/*
 *  This function is WIP.
 *  Collision scenarios:
 *  - default, same direction - cannot occur
 *  - default, oposite direction - ev_initial unchanged
 *  - default, orthogonal directions - ev_initial unchanged
 *  - initial, same direction - cannot occur
 *  - initial, oposite direction - They must deviate to a side
 *  - initial, orthogonal directions - Std collision resolution?
 */
__device__ float4 solve_collision_ev_initial_ev_any(float2 ev1_loc, float2 ev1_velo, float ev1_mass_ag, 
	float2 ev2_loc, float2 ev2_velo, float ev2_mass_ag, float min_distance, float2 dist, float dist_length) {
	// normal velocity vectors just before the impact
	float2 normal_velocity1 = float2_project(ev1_velo, dist);
	float2 normal_velocity2 = float2_project(ev2_velo, dist);
	// tangential velocity vectors
	float2 tangent_velocity1 = float2_sub(ev1_velo, normal_velocity1);
	//float2 tangent_velocity2 = ev2_velo - normal_velocity2;

	// move particles so that they just touch
	float L = min_distance - dist_length;
	float2 normal_velo_subtracted = float2_sub(normal_velocity1, normal_velocity2);
	float vrel = vlength(normal_velo_subtracted.x, normal_velo_subtracted.y);
	//float2 new_ev1_loc = add_scaled(make_float2(0, 0), normal_velocity1, -L / vrel);
	float dp = float2_dot(normal_velocity1, normal_velocity2);
	float2 new_ev1_loc = add_scaled(ev1_loc, normal_velocity1, -L / vrel);
	return make_float4(new_ev1_loc.x, new_ev1_loc.y, ev1_velo.x, ev1_velo.y);
}

/**
Checks any possible collision with another EV
If a collision occurred, a collision_message is generated and the agent will have the values:
	agent->closest_ev_id = closest_ev_id;
	agent->closest_ev_distance = closest_ev_distance;
Otherwise, no message is produced and the ev agent will have the default values unchanged:
	agent->closest_ev_id = -1;
	agent->closest_ev_distance = -1.0f;
*/	
__FLAME_GPU_FUNC__ int test_collision_ev_default_ev_default(xmachine_memory_EV* agent, xmachine_message_location_ev_default_list* location_messages, 
	xmachine_message_location_ev_default_PBM* partition_matrix){
	
	float distance, closest_ev_distance = 100.f;
	float radii = 0.0, radius_to_radius;
	float max_overlap = 0, overlap;

	int ev2_id = -1;
	float ev2_x, ev2_y, ev2_vx, ev2_vy, ev2_mass_ag;
	float2 distance_vector;
	//float4 new_values;
	struct PostCollisionData pcd;

	xmachine_message_location_ev_default* message = get_first_location_ev_default_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	// This check could be improved by identifying and solving only the first collision involving this agent.
	while (message)	{
		if (message->id != agent->id){
			// check for collision
			distance = euclidean_distance(agent, message->x, message->y);
			radii = agent->radius_um + message->radius_um;

			if (distance < radii){
				overlap = radii - distance;
				if (ev2_id == -1 || overlap > max_overlap){
					ev2_id = message->id;
					closest_ev_distance = distance;
					radius_to_radius = radii;
					max_overlap = overlap;
					distance_vector = make_float2(agent->x - message->x, agent->y - message->y);
					ev2_x = message->x;
					ev2_y = message->y;
					ev2_vx = message->vx;
					ev2_vy = message->vy;
					ev2_mass_ag = message->mass_ag;
				}
			}
		}
		message = get_next_location_ev_default_message(message, location_messages, partition_matrix);
	}
	if (ev2_id != -1) {
		pcd = solve_collision_ev_default_ev_default(
			make_float2(agent->x, agent->y), make_float2(agent->vx, agent->vy), agent->mass_ag,
			make_float2(ev2_x, ev2_y), make_float2(ev2_vx, ev2_vy), ev2_mass_ag,
			radius_to_radius, distance_vector, vlength(distance_vector.x, distance_vector.y));
		// we store the current position as the previous and update the values accordingly
		agent->x_1 = agent->x;
		agent->y_1 = agent->y;

		agent->x = pcd.correctedLocation.x;
		agent->y = pcd.correctedLocation.y;
		agent->vx = pcd.correctedVelocity.x;
		agent->vy = pcd.correctedVelocity.y;
		agent->colInfCorrectionFactor = pcd.correctionFactor; // fac1
		agent->colInfOverlap = pcd.overlap;
		/*
		Vnorm could produce a Div0 error but this cannot be checked
		inside the function where its value is computed. We do so here.
		*/
		agent->colInfVnorm = pcd.vrel;  // L
		agent->colInfCorrectionAppliedX = pcd.correctionApplied.x;
		agent->colInfCorrectionAppliedY = pcd.correctionApplied.y;

		if(pcd.vrel < 1e-6){
			agent->debugDiv0 = 6001;
		}
		checkNanInfVelocity(agent, 6002);
		checkNanInfLocation(agent, 6006);

		agent->closest_ev_id = ev2_id;
		agent->closest_ev_distance = closest_ev_distance;
	}

	return 0;
}

__FLAME_GPU_FUNC__ int test_collision_ev_default_ev_initial(xmachine_memory_EV* agent, xmachine_message_location_ev_initial_list* location_messages, 
	xmachine_message_location_ev_initial_PBM* partition_matrix){
	
	float distance, closest_ev_distance = 100.f;
	float radii = 0.0, radius_to_radius;
	float max_overlap = 0, overlap;

	int ev2_id = -1;
	float ev2_x, ev2_y, ev2_vx, ev2_vy, ev2_mass_ag;
	float2 distance_vector;
	struct PostCollisionDataInitial pcdi;

	xmachine_message_location_ev_initial* message = get_first_location_ev_initial_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	// This check could be improved by identifying and solving only the first collision involving this agent.
	while (message)	{
		if (message->id != agent->id){
			// check for collision
			distance = euclidean_distance(agent, message->x, message->y);
			radii = agent->radius_um + message->radius_um;

			if (distance < radii){
				overlap = radii - distance;
				if (ev2_id == -1 || overlap > max_overlap){
					ev2_id = message->id;
					closest_ev_distance = distance;
					radius_to_radius = radii;
					max_overlap = overlap;
					distance_vector = make_float2(agent->x - message->x, agent->y - message->y);
					ev2_x = message->x;
					ev2_y = message->y;
					ev2_vx = message->vx;
					ev2_vy = message->vy;
					ev2_mass_ag = message->mass_ag;
					//ev2_radius_um = message->radius_um;
				}
			}
		}
		message = get_next_location_ev_initial_message(message, location_messages, partition_matrix);
	}
	if (ev2_id != -1) {
		pcdi = solve_collision_ev_default_ev_initial(
			make_float2(agent->x, agent->y), make_float2(agent->vx, agent->vy), agent->mass_ag,
			make_float2(ev2_x, ev2_y), make_float2(ev2_vx, ev2_vy), ev2_mass_ag,
			radius_to_radius, distance_vector, vlength(distance_vector.x, distance_vector.y));
		// we store the current position as the previous and update the values accordingly
		agent->x_1 = agent->x;
		agent->y_1 = agent->y;

		agent->x = pcdi.correctedPosition.x;
		agent->y = pcdi.correctedPosition.y;
		agent->vx = pcdi.correctedVelocity.x;
		agent->vy = pcdi.correctedVelocity.y;
		agent->colInfCorrectionFactor = pcdi.correctionFactor;
		agent->colInfOverlap = pcdi.overlap;
		/*
		both Vnorm and Vrel could produce a Div0 error but they cannot be checked
		inside the function where their values are computed.
		For debugging, we check their values here.
		*/
		agent->colInfVnorm = pcdi.vnorm;
		agent->colInfVrel = pcdi.vrel; // auxiliary
		agent->colInfAuxFactor1 = pcdi.auxiliaryFactors.x; // factor 1
		agent->colInfAuxFactor2 = pcdi.auxiliaryFactors.y; // factor 2
		agent->colInfCorrectionAppliedX = pcdi.correctionApplied.x;
		agent->colInfCorrectionAppliedY = pcdi.correctionApplied.y;

		agent->closest_ev_id = ev2_id;
		agent->closest_ev_distance = closest_ev_distance;

		checkDivZero(agent, pcdi.vrel, 5001);
		checkDivZero(agent, pcdi.vnorm, 5002);

		checkNanInfVelocity(agent, 5003);
		checkNanInfLocation(agent, 5007);
	}

	return 0;
}

__FLAME_GPU_FUNC__ int reset_state(xmachine_memory_EV* agent) {
	agent->closest_ciliary_cell_id = -1;
	agent->closest_ciliary_cell_distance = 100;
	agent->closest_secretory_cell_id = -1;
	agent->closest_secretory_cell_distance = 100;
	agent->closest_ev_id = -1;
	agent->closest_ev_distance = 100;

	// debugging values
	agent->colInfVnorm = 0;
	agent->colInfVrel = 0;
	agent->colInfOverlap = 0;
	agent->colInfCorrectionFactor = 0;
	agent->colInfAuxFactor1 = 0;
	agent->colInfAuxFactor2 = 0;
	agent->colInfCorrectionAppliedX = 0;
	agent->colInfCorrectionAppliedY = 0;
	agent->colInfSeparation_x_1k = -1;

	agent->debugNanInfVelocity = 0;
	agent->debugNanInfLocation = 0;

	agent->debugDragVel0At = 0;
	agent->debugDiv0 = 0;
	agent->debugNanInfValue = 0;

	agent->debugOriginalX = 0;
	agent->debugOriginalY = 0;
	agent->debugOriginalVX = 0;
	agent->debugOriginalVY = 0;
	agent->debugDistDpNormal = 0;
	agent->debugDotProduct = -997.7;
	agent->debugLengthsProduct = -998.8;

	agent->debugQuotient = -999.9;
	agent->debugAngle = 0;
	agent->debugSinAngle = 0;
    agent->debugDeltaS = 0;
	agent->debugDisplX = 0;
	agent->debugDisplY = 0;

	agent->debugNormalVelX = 0;
	agent->debugNormalVelY = 0;
	agent->debugTangVelX = 0;
	agent->debugTangVelY = 0;

	agent->debugVelocityProjection = 1002;

	agent->debugExternPerpDistX = 2000;
	agent->debugExternPerpDistY = 2001;
	agent->debugExternProjX = 2002;
	agent->debugExternProjY = 2003;
	agent->debugExternCellDirLen = 2004;
	agent->debugExternCellDirX = 2005;
	agent->debugExternCellDirY = 2006;

	agent->debugNonFiniteBmAt = -1;

	return 0;
}

__FLAME_GPU_FUNC__ int initial_to_default(xmachine_memory_EV* agent) {
	return 0;
}

/*
Uses the Box-Mueller transform to generate a pair of normally distributed random numbers
by sampling from two uniformly distributed RNG.
In the original algorithm, the sampled values are transformed into cartesian coordinates,
here they become the new velocity for the next step
Due to limitations in FlameGPU, we sample both continuos numbers from the same RNG.
*/
__FLAME_GPU_FUNC__ int brownian_movement_1d(xmachine_memory_EV* agent, RNG_rand48* rand48) {
	float u1, u2, r, theta;
	
	u1 = rnd<CONTINUOUS>(rand48);
	u2 = rnd<CONTINUOUS>(rand48);
	r = sqrt(-2.0 * log(u1));
	theta = 2 * M_PI * u2;
	// the product of r * (cos|sin)(theta) becomes the displacement factor to use in this iteration
	agent->vx = agent->velocity_ums * r * cos(theta) * dt;
	
	return 0;
}

__FLAME_GPU_FUNC__ int brownian_movement_2d(xmachine_memory_EV* agent, RNG_rand48* rand48) {
	float u, r, theta;
	if(isfinite(agent->bm_vx) && isfinite(agent->bm_vy)){
		agent->debugNonFiniteBmAt = -2;
		u = rnd<CONTINUOUS>(rand48);
		r = sqrt(agent->vx * agent->vx + agent->vy * agent->vy) * dt;
		theta = 2 * M_PI * u; // angle of rotation
		float bm_vx = (r * cos(theta));
		float bm_vy = (r * sin(theta));
		if(isfinite(bm_vx) && isfinite(bm_vy)){
			agent->bm_vx = bm_vx;
			agent->bm_vy = bm_vy;
		} else {
			agent->bm_vx = 0;
			agent->bm_vy = 0;
			agent->debugNonFiniteBmAt = agent->age;
		}

		checkValueNanInf(agent, agent->bm_vx, 1000);
		checkValueNanInf(agent, agent->bm_vy, 1005);
		checkNanInfVelocity(agent, 1010);
		agent->debugOriginalVX = agent->vx;
		agent->debugOriginalVY = agent->vy;
		if(agent->debugNonFiniteBmAt < 0)
			agent->debugNonFiniteBmAt = -4;
	} else {
		agent->bm_vx = 0;
		agent->bm_vy = 0;
		if(agent->debugNonFiniteBmAt < 0){
			agent->debugNonFiniteBmAt = agent->age;
		}
	}
	return 0;
}

__FLAME_GPU_FUNC__ int ev_apoptosis(xmachine_memory_EV* agent, RNG_rand48* rand48){
	float rn = rnd<CONTINUOUS>(rand48);
	if(rn > apoptosis_threshold){
		return 1;
	}
	return 0;
}

/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_EV. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int move_bm(xmachine_memory_EV* agent){

	checkNanInfLocation(agent, 2000);

	agent->x_1 = agent->x;
	agent->y_1 = agent->y;

	checkNanInfVelocity(agent, 2005);

	agent->debugOriginalVX = agent->vx;
	agent->debugOriginalVY = agent->vy;
	agent->debugOriginalX = agent->x + agent->vx * dt;
	agent->debugOriginalY = agent->y + agent->vy * dt;
	checkValueNanInf(agent, agent->debugOriginalX, 2010);
	checkValueNanInf(agent, agent->debugOriginalY, 2015);

	agent->x += (agent->vx + agent->bm_vx) * dt;
	agent->y += (agent->vy + agent->bm_vy) * dt;
	checkNanInfLocation(agent, 2020);

	agent->age += dt;
    return 0;
}
__FLAME_GPU_FUNC__ int move(xmachine_memory_EV* agent){

	checkNanInfLocation(agent, 2000);

	agent->x_1 = agent->x;
	agent->y_1 = agent->y;

	checkNanInfVelocity(agent, 2005);

	agent->debugOriginalVX = agent->vx;
	agent->debugOriginalVY = agent->vy;
	agent->debugOriginalX = agent->x + agent->vx * dt;
	agent->debugOriginalY = agent->y + agent->vy * dt;
	checkValueNanInf(agent, agent->debugOriginalX, 2010);
	checkValueNanInf(agent, agent->debugOriginalY, 2015);

	agent->x = agent->x + (agent->vx * 2) * dt;
	agent->y = agent->y + agent->vy * dt;
	checkNanInfLocation(agent, 2020);

	agent->age += dt;
    return 0;
}
__FLAME_GPU_FUNC__ int moveInitial(xmachine_memory_EV* agent) {

	agent->x_1 = agent->x;
	agent->y_1 = agent->y;

	agent->x += agent->vx * dt;
	agent->y += agent->vy * dt;

	agent->age += dt;

	agent->closest_ev_id = -1;
	agent->closest_ev_distance = 100;

	return 0;
}

__FLAME_GPU_FUNC__ int secrete_ev(xmachine_memory_SecretoryCell* secretoryCell, xmachine_memory_EV_list* EVs, RNG_rand48* rand48){
	// compute the next interval of secretion
	if(secretoryCell->time_to_next_secretion_attempt < 0)
	{
		secretoryCell->time_to_next_secretion_attempt = (ev_secretion_interval/2 + 
				(rnd<CONTINUOUS>(rand48) * ev_secretion_interval)) + dt;

		// value saved in the secretoryCell for debugging
		secretoryCell->probability_of_secretion = rnd<CONTINUOUS>(rand48);
				
		if( secretoryCell->probability_of_secretion > ev_secretion_threshold)
		{
			int rand_i = (int)(rnd<CONTINUOUS>(rand48) * max_ev_radius);
			// our simulations work with EVs with diameters in the range 80-320 nm
			// therefore, the radius must be in the range 40-160 nm
			float radius_nm = (rand_i % max_ev_radius) + min_ev_radius;
			float radius_um = (radius_nm) / 1000; // faster than doing /1000
			float radius_m = radius_um / 1E6;

			// compute the volume
			//float radius_nm_3 = radius_nm * radius_nm * radius_nm;
			//float volume = const_pi_4div3 * radius_nm_3;
			//float mass_g = const_mass_per_volume_unit * volume;
			float mass_g = const_mass_p_vol_u_x_4div3_pi * radius_nm * radius_nm * radius_nm;
			float mass_ag = mass_g / 1e-18;

			// to convert:
			// N square metres to square micrometre: multiply N * 1e+12
			// N metres to micrometres: multiply N * 1e+6
			float diffusion_rate_ms = const_Boltzmann_x_Temp_K / (const_6_pi_dynamic_viscosity * radius_m); // square metre
			float diffusion_rate_ums = diffusion_rate_ms * 1e+12; // square micrometre

			// compute the diffusion-rate dependant Mean Square Displacement
			float velocity_ums = sqrt(2 * dof * diffusion_rate_ums * dt);

			// decompose velocity
			float vx = velocity_ums * secretoryCell->unit_normal_x;
			float vy = velocity_ums * secretoryCell->unit_normal_y;

			// choose a random starting point
			int rand_i2 = (int)(rnd<CONTINUOUS>(rand48) * secretoryCell->source_points) - 1;
			if(rand_i2 < 0)
				rand_i2++;
			else if (rand_i2 == secretoryCell->last_source_point_secreting)
				rand_i2--;

			secretoryCell->last_source_point_secreting = rand_i2;
			int ssp = rand_i2 * xmachine_memory_SecretoryCell_MAX;
			float x = secretoryCell->source_points_xs[ssp];
			float y = secretoryCell->source_points_ys[ssp];
			unsigned int id = generate_EV_id();
			// displace the startgint coordinates backwards by the diameter of the EV
			x -= secretoryCell->unit_normal_x * radius_um;
			y -= secretoryCell->unit_normal_y * radius_um;

			float time_in_initial = ((radius_um * 2.) / (velocity_ums * dt)) * dt;

			// EV_secretoryCell_list, id, x, y, z, x_1, y_1, vx, vy, bm_vx, bm_vy,
			add_EV_agent(EVs, id, x, y, secretoryCell->id, x - vx * dt, y - vy * dt, vx, vy, 0, 0,
				// mass_ag, radius_um, diffusion_rate_um, diff_rate_um_x_twice_dof
				mass_ag, radius_um, diffusion_rate_ums, diffusion_rate_ums * 2 * dof,
				// closest: ev_id, dist, secretory, dist, ciliary, diste, age, velocity_um, velocity_m
				-1, 100, -1, 100, -1, 100, 0, time_in_initial, velocity_ums,
				// colInfVnorm - colInfSeparation_x_1k
				0, 0, 0, 0, 0, 0, 0, 0, -1,
				// debug
				// debugNanInfVelocity - debugSinAngle
				0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -997.7, -998.8, -999.9, 0, 0,
				// debugDeltaS - debugTangVelY
				0, 0, 0, 0, 0, 0, 0,
				// debugInternPerpDistX - debugNonFiniteBmAt
				//1000, 1001,
				1002, 2000, 2001, 2002, 2003, 2004, 2005, 2006, -1
				);
		}
	}
	secretoryCell->time_to_next_secretion_attempt -= dt;
	return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
