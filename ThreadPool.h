#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <future>
#include <iostream>



class ThreadPool
{
public:
	using task = std::function<void()>;

	explicit ThreadPool(int no_threads);

	~ThreadPool() noexcept;

	explicit ThreadPool(const ThreadPool& other) = delete;
	explicit ThreadPool(ThreadPool&& other) = delete;
	ThreadPool& operator= (const ThreadPool& other) = delete;
	ThreadPool& operator= (ThreadPool&& other) = delete;

	void wait();

	template<typename T>
	auto enqueue_task(T&& task)->std::future<decltype(task())>
	{
		auto wrapper = std::make_shared<std::packaged_task<decltype(task())()>>(std::forward<T>(task));
		{
			std::unique_lock<std::mutex> mutex(m_mutex);
			m_tasks.emplace([=] {(*wrapper)(); });
		}
		m_hasWork.notify_one();
		return wrapper->get_future();
	}

private:
	std::vector<std::thread> m_threads;
	std::mutex m_mutex;
	std::condition_variable m_hasWork;
	std::condition_variable m_finished;
	std::queue<task> m_tasks;
	bool m_running = true;
	int m_execNo = 0;
};
