#pragma once
#include <vector>
#include <random>


class AntCPU
{
private:
	int routeIndex{};
	int position{};
	std::vector<int> possibleLocations;
	int possibleLocationsLastIndex{};
	int possibleLocationsNextIndex{};
	std::vector<double> goodnessNumerators;

	int m_initialLocation{};
	const std::vector<double>& m_distanceMatrix;
	const std::vector<double>& m_pheromoneMatrix;
	const size_t m_matrixDim{};

	std::uniform_real_distribution<double> dist;
	std::mt19937 rng;

public:
	std::vector<int> m_route;
	double distance{};

public:
	AntCPU(const std::vector<double>& distanceMatrix, const std::vector<double>& pheromoneMatrix, int initialLocation, int matrixDim);

	void Venture(double alpha, double beta);

	void Reset();

	
private:
	int SelectNextHop(double alpha, double beta);

	void GoTo(size_t next);
};

