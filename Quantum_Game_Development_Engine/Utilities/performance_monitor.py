# performance_monitor.py
# Quantum_Game_Development_Engine/Utilities/performance_monitor.py

import time
import psutil
import logging

class PerformanceMonitor:
    def __init__(self, log_file="performance.log"):
        """
        Initializes the PerformanceMonitor with the specified log file.
        :param log_file: Path to the log file
        """
        self.log_file = log_file
        self.logger = logging.getLogger('PerformanceMonitor')
        self.logger.setLevel(logging.INFO)
        self._setup_file_handler()
        self._setup_console_handler()
        self.start_time = time.time()

    def _setup_file_handler(self):
        """
        Sets up a file handler for logging performance data to a file.
        """
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(file_handler)

    def _setup_console_handler(self):
        """
        Sets up a console handler for logging performance data to the console.
        """
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(console_handler)

    def _get_formatter(self):
        """
        Returns the log message format.
        :return: A logging.Formatter instance
        """
        return logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def monitor_function(self, func):
        """
        Decorator to monitor the performance of a function.
        :param func: The function to monitor
        :return: Wrapper function
        """
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.logger.info(f"Function {func.__name__} took {elapsed_time:.4f} seconds")
            return result
        return wrapper

    def log_memory_usage(self):
        """
        Logs the current memory usage of the game.
        """
        memory_info = psutil.virtual_memory()
        self.logger.info(f"Memory Usage: {memory_info.percent}%")
        self.logger.info(f"Available Memory: {memory_info.available / (1024 ** 2):.2f} MB")

    def log_performance_summary(self):
        """
        Logs a summary of the performance metrics.
        """
        elapsed_time = time.time() - self.start_time
        self.logger.info(f"Total Elapsed Time: {elapsed_time:.2f} seconds")

# Example usage
if __name__ == "__main__":
    monitor = PerformanceMonitor()

    @monitor.monitor_function
    def sample_function(n):
        """
        Sample function to demonstrate performance monitoring.
        :param n: Number of iterations
        """
        total = 0
        for i in range(n):
            total += i
        return total

    sample_function(10000)
    monitor.log_memory_usage()
    monitor.log_performance_summary()
