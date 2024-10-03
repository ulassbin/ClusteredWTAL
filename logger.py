import logging
import torch
from colorama import Fore, init

# Initialize colorama (for colored console output)
init(autoreset=True)

class CustomLogger:
    def __init__(self, name="CustomLogger", log_file="clustering.log", level=logging.INFO):
        # Create a logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create handlers (console and file)
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log(self, message, level=logging.INFO):
        """Log a message at the given level."""
        if level == logging.DEBUG:
            self.logger.debug(message)
        elif level == logging.INFO:
            self.logger.info(message)
        elif level == logging.WARNING:
            self.logger.warning(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.CRITICAL:
            self.logger.critical(message)

    def log_memory_usage(self, message="", level=logging.INFO):
        """Log GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            memory_message = f"{message} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB"
            self.log(memory_message, level)
        else:
            self.log("CUDA not available.", level)

    def log(self, message, level=logging.INFO):
        """Log a colored message to the console."""
        if level == logging.DEBUG:
            colored_message = Fore.GREEN + message
            self.logger.debug(colored_message)
        elif level == logging.INFO:
            colored_message = Fore.WHITE + message
            self.logger.info(colored_message)
        elif level == logging.WARNING:
            colored_message = Fore.YELLOW + message
            self.logger.warning(colored_message)
        elif level == logging.ERROR:
            colored_message = Fore.RED + message
            self.logger.error(colored_message)
        elif level == logging.CRITICAL:
            colored_message = Fore.LIGHTRED_EX + message  # Use light red for critical
            self.logger.critical(colored_message)


# Example usage
if __name__ == "__main__":
    # Create a logger instance
    logger = CustomLogger(level=logging.DEBUG)

    # Log some messages
    logger.log("This is an info message.", level=logging.INFO)
    logger.log("This is a debug message.", level=logging.DEBUG)
    
    # Log GPU memory usage
    logger.log_memory_usage("GPU status", level=logging.INFO)
    
