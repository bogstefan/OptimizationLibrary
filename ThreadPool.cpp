#include "ThreadPool.h"

ThreadPool::ThreadPool(int no_threads)
{

	for (int i = 0; i < no_threads; ++i)
	{
		m_threads.emplace_back([this]
		{
			while (true)
			{
				task current_task;
				{
					std::unique_lock<std::mutex> lock(m_mutex);
					m_hasWork.wait(lock, [=] {return !m_running || !m_tasks.empty(); });

					if (!m_running && m_tasks.empty()) { break; }
					current_task = std::move(m_tasks.front());
					m_tasks.pop();
				}
				{
					std::unique_lock<std::mutex> lock(m_mutex);
					++m_execNo;
				}
				current_task();
				{
					std::unique_lock<std::mutex> lock(m_mutex);
					--m_execNo;
					m_finished.notify_one();
				}
			}
		});
	}

}

ThreadPool::~ThreadPool() noexcept
{

	{
		std::unique_lock<std::mutex> lock(m_mutex);
		m_running = false;
	}

	m_hasWork.notify_all();
	for (auto& thread : m_threads) { thread.join(); }
}

void ThreadPool::wait()
{
	std::unique_lock<std::mutex> lock(m_mutex);
	m_finished.wait(lock, [this] {return m_execNo == 0 && m_tasks.empty(); });
}
