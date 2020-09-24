#include "Particle.cuh"

__device__
Particle::Particle(int dimSize, double (*function)(double*, int), const curandState_t& randomState) :
	m_dimSize(dimSize),
	m_function(function),
	m_randState(randomState),
	m_currentVelocity(new double[dimSize]),
	m_currentPosition(new double[dimSize])
{
	memset(m_currentVelocity, 0, m_dimSize * sizeof(double));
}

__device__
Particle::Particle()
{}

__device__
void Particle::Run(double aInd, double aGrp, double w, double* currentPersonalBest, const double* currentGlobalBest, const double* boundsVector, const double* boundsSize)
{

	//Initialize random location
	for (int i = 0; i < m_dimSize; ++i) { m_currentPosition[i] = boundsVector[i] + boundsSize[i] * curand_uniform_double(&m_randState); }

	for (int i = 0; i < m_dimSize; ++i)
	{
		double rInd = curand_uniform_double(&m_randState);
		double rGrp = curand_uniform_double(&m_randState);

		m_currentVelocity[i] = w * m_currentVelocity[i] + aInd * rInd * (currentPersonalBest[i] - m_currentPosition[i]) + aGrp * rGrp * (currentGlobalBest[i] - m_currentPosition[i]);
		m_currentPosition[i] = m_currentPosition[i] * m_currentVelocity[i];
	}
	auto currentValue = m_function(m_currentPosition, m_dimSize);
	if (currentValue < m_currentMinimum)
	{
		m_currentMinimum = currentValue;
		for (int i = 0; i < m_dimSize; ++i) { currentPersonalBest[i] = m_currentPosition[i]; }
	}
}

__device__
double Particle::GetMinimum() const { return m_currentMinimum; }

__device__
Particle::~Particle()
{
	delete[] m_currentVelocity;
	delete[] m_currentPosition;
}