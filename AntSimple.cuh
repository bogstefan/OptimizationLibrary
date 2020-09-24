#pragma once

#include <curand_kernel.h>

namespace SIMPLE
{
	class Ant
	{
		int visitedIndex{};
		bool* isVisited{};
		int position{};

		double* goodnessNumerators{};

		curandState_t m_randomState{};

	public:
		__device__
			Ant(int initialLocation, int matrixDim, curandState_t randState);

		Ant() = default;

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

}


