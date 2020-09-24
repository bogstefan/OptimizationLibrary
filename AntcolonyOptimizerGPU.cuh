#pragma once

#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "AntGPU.cuh"

//using namespace SIMPLE;

namespace INTERNAL {
	namespace PSO {
		struct InitFunctor
		{
			int initialLocation;
			int matrixDim;

			InitFunctor(int initialLocation, int matrixDim);

			__device__
				void operator()(const thrust::tuple<AntGPU&, int>& antIndexTuple) const;
		};

		struct VentureFunctor
		{
			double* distanceMatrix;
			double* pheromoneMatrix;
			int matrixDim;
			int initialLocation;

			VentureFunctor(double* distanceMatrix, double* pheromoneMatrix, int matrixDim, int initialLocation,
				double alpha, double beta);

			double alpha;
			double beta;

			__device__
				void operator()(const thrust::tuple<AntGPU&, int*&>& antRouteTuple) const;
		};

		struct SetPheromoneChangesFunctor
		{
			double* pheromoneMatrixChanges;
			int matrixDim;

			SetPheromoneChangesFunctor(double* pheromoneMatrixChanges, int matrixDim);

			__device__
				void operator()(const thrust::tuple<AntGPU&, int*&>& antRouteTuple) const;
		};

		struct EvaporatePheromoneFunctor
		{
			double evaporationRate;

			explicit EvaporatePheromoneFunctor(double evaporationRate);

			__device__
				void operator()(double& x) const;

		};

		struct UpdatePheromonesFunctor
		{
			__device__
				void operator()(const thrust::tuple<double&, double&>& pheromonePheromoneChangesTuple) const;
		};
	}
}

class AntcolonyOptimizerGPU
{
	thrust::device_vector<AntGPU> mdv_ants;
	std::vector<thrust::device_vector<int>> mdv_routes;

	thrust::device_vector<double> mdv_distanceMatrix;
	thrust::device_vector<double> mdv_pheromoneMatrix;
	thrust::device_vector<double> mdv_pheromoneMatrixChanges;

	thrust::device_vector<int*> mdv_routePointers;

	int mh_numAnts;
	int m_matrixDim;
	int m_initialLocation;


public:
	double m_bestDistance;

	thrust::host_vector<int> mh_bestRoute;

public:
	AntcolonyOptimizerGPU(int numAnts, int initialLocation, thrust::host_vector<double> distanceMatrix, thrust::host_vector<double> pheromoneMatrix, int matrixDim);
	void Run(int numIterations, double alpha, double beta, double evaporationRate);
	void Report();
};

