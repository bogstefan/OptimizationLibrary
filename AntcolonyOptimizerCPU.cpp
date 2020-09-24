#include "AntcolonyOptimizerCPU.h"
#include "ThreadPool.h"
#include <iostream>
#include <algorithm>

AntcolonyOptimizerCPU::AntcolonyOptimizerCPU(size_t numAnts, int initialLocation, std::vector<double>& distanceMatrix, std::vector<double>& pheromoneMatrix, size_t matrixDim) :
	m_bestRouteLength(9999999.0),
	m_bestRoute(numAnts + 1),
	m_distanceMatrix(distanceMatrix),
	m_pheromoneMatrix(pheromoneMatrix),
	m_pheromoneMatrixChanges(matrixDim* matrixDim, 0.0),
	m_matrixDim(matrixDim)
{
	m_ants.reserve(numAnts);
	for (int i = 0; i < numAnts; ++i) { m_ants.emplace_back(m_distanceMatrix, m_pheromoneMatrix, initialLocation, m_matrixDim); }

}

void AntcolonyOptimizerCPU::Run(int numIterations, double alpha, double beta, double evaporationRate)
{
	ThreadPool pool(8);
	for (int iteration = 0; iteration < numIterations; ++iteration)
	{
		//std::cout << "Iteration Nr: " << iteration << '\n';
		//Venture
		//std::cout << "Venturing\n";
		for (auto& ant : m_ants)
		{
			
			pool.enqueue_task([&]() {
				ant.Reset();
				ant.Venture(alpha, beta);
			});
			
			
			//ant.Reset();
			//ant.Venture(alpha, beta);
		}

		pool.wait();

		//Evaporate
		//std::cout << "Evaporation\n";
		for (double& pheromone : m_pheromoneMatrix) { pheromone *= (1 - evaporationRate); }

		//Set pheromone Changes
		//std::cout << "Setting Pheromne Delta\n";
		for (auto& ant : m_ants)
		{
			for (int i = 0; i <= m_matrixDim - 1; ++i)
			{
				int currentNode = ant.m_route[i];
				int nextNode = ant.m_route[i + 1];
				double pheromoneAddition = 1.0 / ant.distance;

				m_pheromoneMatrixChanges[m_matrixDim * currentNode + nextNode] += pheromoneAddition;
				m_pheromoneMatrixChanges[m_matrixDim * nextNode + currentNode] += pheromoneAddition;
			}
		}

		//Update pheromones
		//std::cout << "Updating Pheromone\n";
		rsize_t numElements = m_matrixDim * m_matrixDim;
		for (size_t i = 0; i < numElements; ++i) { m_pheromoneMatrix[i] += m_pheromoneMatrixChanges[i]; }


		//Select best ant
		//std::cout << "Selecting best Ant\n";
		auto bestAnt = *std::min_element(m_ants.begin(), m_ants.end(), [](const AntCPU& left, const AntCPU& right) { return left.distance < right.distance; });

		if (bestAnt.distance < m_bestRouteLength)
		{
			m_bestRouteLength = bestAnt.distance;
			m_bestRoute = bestAnt.m_route;
		}

		//Clear pheromone Changes
		//std::cout << "Clearing Pheromone Delta\n";
		for (size_t i = 0; i < numElements; ++i) { m_pheromoneMatrixChanges[i] = 0; }
	}
	
}

void AntcolonyOptimizerCPU::Report()
{
	std::cout << "The Optimal Length is: " << m_bestRouteLength << std::endl;
}


