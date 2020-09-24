#pragma once

#include <curand_kernel.h>

class AntGPU
{
	int routeIndex{};
	int position{};
	int* possibleLocations{};
	int possibleLocationsLastIndex{};
	int possibleLocationsNextIndex{};
	double* goodnessNumerators{};

	curandState_t m_randomState{};

public:
	__device__
		AntGPU(int initialLocation, int matrixDim, curandState_t randState);

	AntGPU() = default;

	__device__
		void Venture(int* route, const double* distanceMatrix, const double* pheromoneMatrix, int matrixDim, double alpha, double beta);

	__device__
		void Reset(int* route, int initialLocation, int matrixDim);

	double distance = 0;
private:
	__device__
		int SelectNextHop(const double* distance_matrix, const double* pheromoneMatrix, int matrixDim, double alpha, double beta);

	__device__
		void GoTo(int next, int* route, const double* distanceMatrix, int matrixDim);
};

