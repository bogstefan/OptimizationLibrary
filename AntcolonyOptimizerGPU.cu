#include "AntcolonyOptimizerGPU.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <time.h>
#include <iostream>


#ifdef __INTELLISENSE__
double atomicAdd(double* address, double val) {};
#endif

//using namespace SIMPLE;


namespace INTERNAL {
	namespace PSO 
	{

		__device__
			void InitFunctor::operator()(const thrust::tuple<AntGPU&, int>& antIndexTuple) const
		{
			AntGPU& ant = antIndexTuple.get<0>();
			int index = antIndexTuple.get<1>();
			curandState_t randomState;
			curand_init(clock() + index, 0, 0, &randomState);
			//curand_init(clock(), index, 0, &randomState);
			ant = AntGPU(initialLocation, matrixDim, randomState);
		}

		InitFunctor::InitFunctor(int initialLocation, int matrixDim) :
			initialLocation(initialLocation),
			matrixDim(matrixDim)
		{}

		__device__
			void VentureFunctor::operator()(const thrust::tuple<AntGPU&, int*&>& antRouteTuple) const
		{
			AntGPU& ant = antRouteTuple.get<0>();
			int* route = antRouteTuple.get<1>();

			ant.Reset(route, initialLocation, matrixDim);
			ant.Venture(route, distanceMatrix, pheromoneMatrix, matrixDim, alpha, beta);
		}

		VentureFunctor::VentureFunctor(double* distanceMatrix, double* pheromoneMatrix, int matrixDim,
			int initialLocation, double alpha, double beta) :
			distanceMatrix(distanceMatrix),
			pheromoneMatrix(pheromoneMatrix),
			matrixDim(matrixDim),
			initialLocation(initialLocation),
			alpha(alpha),
			beta(beta)
		{}

		__device__
			void SetPheromoneChangesFunctor::operator()(const thrust::tuple<AntGPU&, int*&>& antRouteTuple) const
		{
			AntGPU& ant = antRouteTuple.get<0>();
			int* route = antRouteTuple.get<1>();

			for (int i = 0; i <= matrixDim - 1; ++i)
			{
				int currentNode = route[i];
				int nextNode = route[i + 1];
				double pheromoneAddition = 1.0 / ant.distance;

				atomicAdd(&pheromoneMatrixChanges[matrixDim * currentNode + nextNode], pheromoneAddition);
				atomicAdd(&pheromoneMatrixChanges[matrixDim * nextNode + currentNode], pheromoneAddition);
			}
		}

		SetPheromoneChangesFunctor::SetPheromoneChangesFunctor(double* pheromoneMatrixChanges, int matrixDim) :
			pheromoneMatrixChanges(pheromoneMatrixChanges),
			matrixDim(matrixDim)
		{}

		__device__
			void EvaporatePheromoneFunctor::operator()(double& x) const
		{
			x = (1.0 - evaporationRate) * x;
		}

		EvaporatePheromoneFunctor::EvaporatePheromoneFunctor(double evaporationRate) :
			evaporationRate(evaporationRate)
		{}

		void UpdatePheromonesFunctor::operator()(const thrust::tuple<double&, double&>& pheromonePheromoneChangesTuple) const
		{
			double& pheromone = pheromonePheromoneChangesTuple.get<0>();
			double pheromoneChange = pheromonePheromoneChangesTuple.get<1>();
			pheromone += pheromoneChange;
		}

	}
}

AntcolonyOptimizerGPU::AntcolonyOptimizerGPU(int numAnts, int initialLocation, thrust::host_vector<double> distanceMatrix, thrust::host_vector<double> pheromoneMatrix, int matrixDim) :
	mh_numAnts(numAnts),
	m_matrixDim(matrixDim),
	mdv_ants(numAnts),
	mdv_distanceMatrix(distanceMatrix),
	mdv_pheromoneMatrix(pheromoneMatrix),
	mdv_pheromoneMatrixChanges(matrixDim* matrixDim, 0.0),
	mh_bestRoute(matrixDim),
	mdv_routes(numAnts, thrust::device_vector<int>(matrixDim + 1, 9)),
	m_initialLocation(initialLocation)
{
	m_bestDistance = 999999999;
	thrust::host_vector<int*> deviceRoutePointersH(mh_numAnts);

	for (int i = 0; i < mh_numAnts; ++i)
	{
		deviceRoutePointersH[i] = mdv_routes[i].data().get();
	}

	mdv_routePointers = deviceRoutePointersH;

	thrust::counting_iterator<int> counter = thrust::make_counting_iterator(0);

	auto start = thrust::make_zip_iterator(thrust::make_tuple(mdv_ants.begin(), counter));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(mdv_ants.end(), counter + numAnts));

	INTERNAL::PSO::InitFunctor initFunctor(initialLocation, matrixDim);
	thrust::for_each(start, end, initFunctor);
}


void AntcolonyOptimizerGPU::Run(int numIterations, double alpha, double beta, double evaporationRate)
{
	auto antRouteBegin = thrust::make_zip_iterator(thrust::make_tuple(mdv_ants.begin(), mdv_routePointers.begin()));
	auto antRouteEnd = thrust::make_zip_iterator(thrust::make_tuple(mdv_ants.end(), mdv_routePointers.end()));

	INTERNAL::PSO::VentureFunctor ventureFunctor(mdv_distanceMatrix.data().get(), mdv_pheromoneMatrix.data().get(), m_matrixDim, m_initialLocation, alpha, beta);
	INTERNAL::PSO::SetPheromoneChangesFunctor setPheromoneChangesFunctor(mdv_pheromoneMatrixChanges.data().get(), m_matrixDim);

	INTERNAL::PSO::EvaporatePheromoneFunctor evaporatePheromoneFunctor(evaporationRate);

	INTERNAL::PSO::UpdatePheromonesFunctor updatePheromonesFunctor;
	auto pheromonePheromoneChangesBegin = thrust::make_zip_iterator(thrust::make_tuple(mdv_pheromoneMatrix.begin(), mdv_pheromoneMatrixChanges.begin()));
	auto pheromonePheromoneChangesEnd = thrust::make_zip_iterator(thrust::make_tuple(mdv_pheromoneMatrix.end(), mdv_pheromoneMatrixChanges.end()));

	for (int i = 0; i < numIterations; ++i)
	{
		std::cout << "Iteration Nr: " << i << '\n';

		std::cout << "Venturing\n";
		thrust::for_each(antRouteBegin, antRouteEnd, ventureFunctor);

		std::cout << "Set Pheromone Delta\n";
		thrust::for_each(antRouteBegin, antRouteEnd, setPheromoneChangesFunctor);
	
		thrust::for_each(mdv_pheromoneMatrix.begin(), mdv_pheromoneMatrix.begin() + m_matrixDim * m_matrixDim, evaporatePheromoneFunctor);

		//thrust::for_each(pheromonePheromoneChangesBegin, pheromonePheromoneChangesEnd, updatePheromonesFunctor);
		thrust::transform(mdv_pheromoneMatrix.begin(), mdv_pheromoneMatrix.end(), mdv_pheromoneMatrixChanges.begin(), mdv_pheromoneMatrix.begin(), thrust::plus<double>());
		//thrust::host_vector<double> after = mdv_pheromoneMatrix;

		auto minDistanceIter = thrust::min_element(mdv_ants.begin(), mdv_ants.end(), [] __device__(const AntGPU & lhs, const AntGPU & rhs)
		{
			return lhs.distance < rhs.distance;
		});

		int minDistanceIndex = minDistanceIter - mdv_ants.begin();
		AntGPU minDistanceAnt = *minDistanceIter;
		double minDistance = minDistanceAnt.distance;


		if (minDistance < m_bestDistance)
		{
			m_bestDistance = minDistance;
			mh_bestRoute = mdv_routes[minDistanceIndex];
		}

		thrust::fill(thrust::device, mdv_pheromoneMatrixChanges.begin(), mdv_pheromoneMatrixChanges.end(), 0.0);
	}
}

void AntcolonyOptimizerGPU::Report()
{
	std::cout << "The Optimal Length is: " << m_bestDistance << std::endl;
}
