#include "AntCPU.h"

AntCPU::AntCPU(const std::vector<double>& distanceMatrix, const std::vector<double>& pheromoneMatrix, int initialLocation, int matrixDim) :
	position(initialLocation),
	possibleLocations(matrixDim),
	goodnessNumerators(matrixDim),
	possibleLocationsLastIndex(matrixDim - 1),
	rng(std::random_device{}()),
	m_route(matrixDim + 1),
	m_distanceMatrix(distanceMatrix),
	m_pheromoneMatrix(pheromoneMatrix),
	m_matrixDim(matrixDim)
{
}

void AntCPU::Venture(double alpha, double beta)
{
	while (possibleLocationsLastIndex >= 0)
	{
		int nextHop = SelectNextHop(alpha, beta);
		GoTo(nextHop);
	}
	m_route[m_matrixDim] = m_route[0];
	distance += m_distanceMatrix[m_route[m_matrixDim - 1] * m_matrixDim + m_route[0]];
}

void AntCPU::Reset()
{
	routeIndex = 0;
	distance = 0;
	position = m_initialLocation;
	possibleLocationsLastIndex = m_matrixDim - 1;
	for (int i = 0; i < m_matrixDim; ++i) { possibleLocations[i] = i; }
	m_route[routeIndex++] = m_initialLocation;
	possibleLocations[m_initialLocation] = possibleLocations[possibleLocationsLastIndex--];
}

int AntCPU::SelectNextHop(double alpha, double beta)
{
	double denominator = 0;
	for (int i = 0; i <= possibleLocationsLastIndex; ++i)
	{
		int possiblePosition = possibleLocations[i];
		double nextPheromone = m_pheromoneMatrix[position * m_matrixDim + possiblePosition];
		double nextDistance = m_distanceMatrix[position * m_matrixDim + possiblePosition];
		double goodnessNumerator = std::pow(nextPheromone, alpha) * std::pow(1.0 / nextDistance, beta);

		goodnessNumerators[possiblePosition] = goodnessNumerator;
		denominator += goodnessNumerator;
	}

	double sum = 0;
	double random = dist(rng);

	for (int i = 0; i <= possibleLocationsLastIndex; ++i)
	{
		int possiblePosition = possibleLocations[i];
		double numerator = goodnessNumerators[possiblePosition];
		double probability = numerator / denominator;
		if (random <= sum + probability)
		{
			possibleLocationsNextIndex = i;
			return possiblePosition;
		}
		sum += probability;
	}
	return -1;
}

void AntCPU::GoTo(size_t next)
{
	m_route[routeIndex++] = next;
	possibleLocations[possibleLocationsNextIndex] = possibleLocations[possibleLocationsLastIndex--];
	distance += m_distanceMatrix[position * m_matrixDim + next];
	position = next;
}
