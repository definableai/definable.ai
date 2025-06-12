"""
AgentLogger: A flexible, colorful logging library for agent development frameworks

Features:
- Color-coded log levels
- Context-aware logging
- Hierarchical loggers
- Log retention and rotation
- Multiple output destinations
- Customizable formatting
"""

import atexit
import datetime
import glob
import json
import os
import sys
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass
from enum import IntEnum
from threading import Lock
from typing import Any, Dict, List, Optional, Set, TextIO


# ----- Color Support -----
class Colors:
  RESET = "\033[0m"
  BOLD = "\033[1m"
  DIM = "\033[2m"
  UNDERLINE = "\033[4m"

  # Foreground Colors
  BLACK = "\033[30m"
  RED = "\033[31m"
  GREEN = "\033[32m"
  YELLOW = "\033[33m"
  BLUE = "\033[34m"
  MAGENTA = "\033[35m"
  CYAN = "\033[36m"
  WHITE = "\033[37m"

  # Bright Foreground Colors
  BRIGHT_BLACK = "\033[90m"
  BRIGHT_RED = "\033[91m"
  BRIGHT_GREEN = "\033[92m"
  BRIGHT_YELLOW = "\033[93m"
  BRIGHT_BLUE = "\033[94m"
  BRIGHT_MAGENTA = "\033[95m"
  BRIGHT_CYAN = "\033[96m"
  BRIGHT_WHITE = "\033[97m"

  # Background Colors
  BG_BLACK = "\033[40m"
  BG_RED = "\033[41m"
  BG_GREEN = "\033[42m"
  BG_YELLOW = "\033[43m"
  BG_BLUE = "\033[44m"
  BG_MAGENTA = "\033[45m"
  BG_CYAN = "\033[46m"
  BG_WHITE = "\033[47m"

  @staticmethod
  def colorize(text: str, color: str) -> str:
    """Apply color to text"""
    return f"{color}{text}{Colors.RESET}"

  @staticmethod
  def strip_colors(text: str) -> str:
    """Remove ANSI color codes from text"""
    import re

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


class LogLevel(IntEnum):
  """Log level definitions"""

  TRACE = 5
  DEBUG = 10
  INFO = 20
  WARN = 30
  ERROR = 40
  FATAL = 50
  SILENT = 100


# Maps log levels to names and colors
LOG_LEVEL_NAMES: Dict[LogLevel, str] = {
  LogLevel.TRACE: "TRACE",
  LogLevel.DEBUG: "DEBUG",
  LogLevel.INFO: "INFO",
  LogLevel.WARN: "WARN",
  LogLevel.ERROR: "ERROR",
  LogLevel.FATAL: "FATAL",
  LogLevel.SILENT: "SILENT",
}

LOG_LEVEL_COLORS: Dict[LogLevel, str] = {
  LogLevel.TRACE: Colors.BRIGHT_BLACK,
  LogLevel.DEBUG: Colors.CYAN,
  LogLevel.INFO: Colors.GREEN,
  LogLevel.WARN: Colors.YELLOW,
  LogLevel.ERROR: Colors.RED,
  LogLevel.FATAL: f"{Colors.BG_RED}{Colors.WHITE}",
  LogLevel.SILENT: Colors.WHITE,
}

# Component type colors
COMPONENT_COLORS: Dict[str, str] = {
  "agent": Colors.MAGENTA,
  "tool": Colors.BLUE,
  "workflow": Colors.GREEN,
  "knowledgeBase": Colors.CYAN,
  "default": Colors.WHITE,
}


@dataclass
class LogEntry:
  """Log entry representation"""

  timestamp: datetime.datetime
  level: LogLevel
  component: str
  namespace: str
  message: str
  data: Optional[Any] = None
  error: Optional[Exception] = None

  def to_dict(self) -> Dict[str, Any]:
    """Convert log entry to dictionary format"""
    result = {
      "timestamp": self.timestamp.isoformat(),
      "level": int(self.level),
      "level_name": LOG_LEVEL_NAMES[self.level],
      "component": self.component,
      "namespace": self.namespace,
      "message": self.message,
    }

    if self.data is not None:
      try:
        # Attempt to convert data to a JSON-serializable format
        result["data"] = self.data
      except (TypeError, ValueError):
        # If it can't be directly serialized, convert to string
        result["data"] = str(self.data)

    if self.error is not None:
      result["error"] = {
        "type": self.error.__class__.__name__,
        "message": str(self.error),
        "traceback": traceback.format_exception(type(self.error), self.error, self.error.__traceback__),
      }

    return result


class LogTransport:
  """Base class for log transports (output destinations)"""

  def write(self, entry: LogEntry) -> None:
    """Write a log entry to the destination"""
    raise NotImplementedError("Subclasses must implement write()")

  def flush(self) -> None:
    """Flush any buffered output"""
    pass

  def close(self) -> None:
    """Close the transport"""
    self.flush()


class ConsoleTransport(LogTransport):
  """Transport that writes logs to the console with colors"""

  def __init__(self, show_timestamp: bool = True, show_namespace: bool = True, use_colors: bool = True):
    """
    Initialize console transport

    Args:
        show_timestamp: Whether to show timestamps in the output
        show_namespace: Whether to show namespace in the output
        use_colors: Whether to use colors in the output
    """
    self.show_timestamp = show_timestamp
    self.show_namespace = show_namespace
    self.use_colors = use_colors
    self._determine_color_support()

  def _determine_color_support(self) -> None:
    """Determine if the terminal supports colors"""
    if not self.use_colors:
      return

    # Disable colors if the output is being redirected or piped
    if not sys.stdout.isatty():
      self.use_colors = False
      return

    # Check for NO_COLOR environment variable
    if os.environ.get("NO_COLOR", ""):
      self.use_colors = False
      return

    # Check for FORCE_COLOR environment variable
    if os.environ.get("FORCE_COLOR", ""):
      self.use_colors = True
      return

  def _colorize(self, text: str, color: str) -> str:
    """Apply color to text if colors are enabled"""
    if self.use_colors:
      return Colors.colorize(text, color)
    return text

  def write(self, entry: LogEntry) -> None:
    """Write log entry to console"""
    # Get level color and name
    level_color = LOG_LEVEL_COLORS[entry.level]
    level_name = LOG_LEVEL_NAMES[entry.level].ljust(5)

    # Get component color
    component_type = entry.component or "default"
    component_color = COMPONENT_COLORS.get(component_type, COMPONENT_COLORS["default"])

    output_parts = []

    # Add timestamp if configured
    if self.show_timestamp:
      timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
      output_parts.append(self._colorize(f"[{timestamp}]", Colors.DIM))

    # Add level with color
    output_parts.append(self._colorize(level_name, level_color))

    # Add component with color
    namespace_part = ""
    if entry.namespace and self.show_namespace:
      namespace_part = f":{entry.namespace}"

    output_parts.append(self._colorize(f"[{component_type}{namespace_part}]", component_color))

    # Add message
    output_parts.append(entry.message)

    # Join all parts
    output = " ".join(output_parts)

    # Determine output stream based on log level
    stream = sys.stderr if entry.level >= LogLevel.ERROR else sys.stdout

    # Write to console
    print(output, file=stream)

    # If there's data, format and print it
    if entry.data is not None:
      try:
        # Try to format as JSON
        if isinstance(entry.data, (dict, list)):
          formatted_data = json.dumps(entry.data, indent=2, default=str)
        else:
          formatted_data = str(entry.data)

        # Print with indentation
        for line in formatted_data.split("\n"):
          print(self._colorize(f"    {line}", Colors.DIM), file=stream)
      except Exception:
        print(self._colorize(f"    {entry.data}", Colors.DIM), file=stream)

    # If there's an error, print its traceback
    if entry.error is not None:
      error_lines = traceback.format_exception(type(entry.error), entry.error, entry.error.__traceback__)
      for line in error_lines:
        print(self._colorize(f"    {line.rstrip()}", Colors.RED), file=stream)

  def flush(self) -> None:
    """Flush console output"""
    sys.stdout.flush()
    sys.stderr.flush()


@dataclass
class RetentionOptions:
  """Options for log file retention"""

  max_files: int = 10  # Maximum number of log files to keep
  max_size: int = 10_485_760  # Maximum size per log file (10MB default)
  max_age: int = 604_800_000  # Maximum age of log files in milliseconds (7 days default)


class FileTransport(LogTransport):
  """Transport that writes logs to rotating files with retention policies"""

  def __init__(
    self,
    directory: str,
    filename: Optional[str] = None,
    retention: Optional[RetentionOptions] = None,
    include_colors: bool = False,
    format_json: bool = False,
  ):
    """
    Initialize file transport

    Args:
        directory: Directory to store log files
        filename: Base filename for log files (default: "agent")
        retention: Retention policy options
        include_colors: Whether to include ANSI color codes in the log files
        format_json: Whether to format log entries as JSON
    """
    self.base_path = os.path.join(directory, filename or "agent")
    self.retention = retention or RetentionOptions()
    self.include_colors = include_colors
    self.format_json = format_json

    self.current_file: str = ""  # Initialize with empty string
    self.current_file_handle: Optional[TextIO] = None  # Proper type hint for file handle
    self.current_file_size: int = 0
    self.file_index: int = 0
    self.lock = Lock()

    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Set up the initial log file
    self._rotate_log_file()

    # Schedule cleanup for old logs
    self._cleanup_old_logs()

    # Register cleanup on program exit
    atexit.register(self.close)

  def _rotate_log_file(self) -> None:
    """Rotate to a new log file"""
    with self.lock:
      # Close existing file if open
      if self.current_file_handle is not None:
        self.current_file_handle.close()
        self.current_file_handle = None

      # Generate new filename with timestamp
      now = datetime.datetime.now()
      timestamp = now.strftime("%Y-%m-%dT%H-%M-%S")

      self.file_index += 1
      new_file = f"{self.base_path}.{timestamp}.{self.file_index}.log"
      self.current_file = new_file
      self.current_file_size = 0

      # Open new file
      self.current_file_handle = open(self.current_file, "a", encoding="utf-8")

  def _cleanup_old_logs(self) -> None:
    """Clean up old log files according to retention policy"""
    with suppress(Exception):
      # Get all log files matching our base name
      log_files = glob.glob(f"{self.base_path}.*.*.*")

      # Skip if no files to clean up
      if not log_files:
        return

      # Sort by modification time (oldest first)
      log_files.sort(key=os.path.getmtime)

      now = time.time() * 1000  # Current time in milliseconds

      # Remove files exceeding max count (keep newest)
      if len(log_files) > self.retention.max_files:
        for old_file in log_files[: -self.retention.max_files]:
          with suppress(OSError):
            os.remove(old_file)
            log_files.remove(old_file)

      # Remove files exceeding max age
      for log_file in log_files[:]:
        file_mtime = os.path.getmtime(log_file) * 1000  # Convert to milliseconds
        file_age = now - file_mtime

        if file_age > self.retention.max_age:
          with suppress(OSError):
            os.remove(log_file)
            log_files.remove(log_file)

  def write(self, entry: LogEntry) -> None:
    """Write log entry to file"""
    with self.lock:
      if self.current_file_handle is None:
        self._rotate_log_file()

      if self.current_file_handle is None:  # Double check after rotation
        raise RuntimeError("Failed to open log file")

      # Format the log entry
      if self.format_json:
        log_line = json.dumps(entry.to_dict(), default=str) + "\n"
      else:
        # Text format similar to console output but without colors
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level_name = LOG_LEVEL_NAMES[entry.level].ljust(5)
        component = f"[{entry.component}{':' + entry.namespace if entry.namespace else ''}]"

        parts = [f"[{timestamp}]", level_name, component, entry.message]
        log_line = " ".join(parts)

        # Add data if present
        if entry.data is not None:
          with suppress(Exception):
            if isinstance(entry.data, (dict, list)):
              data_str = json.dumps(entry.data, indent=2, default=str)
            else:
              data_str = str(entry.data)

            # Indent each line
            data_lines = [f"    {line}" for line in data_str.split("\n")]
            log_line += "\n" + "\n".join(data_lines)
          if not data_str:  # If suppress caught an exception
            log_line += f"\n    {entry.data}"

        # Add error if present
        if entry.error is not None:
          error_lines = traceback.format_exception(type(entry.error), entry.error, entry.error.__traceback__)
          error_text = "".join(f"    {line}" for line in error_lines)
          log_line += f"\n{error_text}"

        log_line += "\n"

        # Remove color codes if not including colors
        if not self.include_colors:
          log_line = Colors.strip_colors(log_line)

      # Write to file
      self.current_file_handle.write(log_line)
      self.current_file_handle.flush()

      # Update file size and rotate if needed
      self.current_file_size += len(log_line.encode("utf-8"))
      if self.current_file_size >= self.retention.max_size:
        self._rotate_log_file()

  def flush(self) -> None:
    """Flush file output"""
    with self.lock:
      if self.current_file_handle is not None:
        self.current_file_handle.flush()

  def close(self) -> None:
    """Close the file transport"""
    with self.lock:
      if self.current_file_handle is not None:
        self.current_file_handle.close()
        self.current_file_handle = None


class MemoryTransport(LogTransport):
  """Transport that keeps logs in memory for testing or replay"""

  def __init__(self, max_entries: int = 1000):
    """Initialize memory transport with a maximum number of entries to keep"""
    self.entries: List[LogEntry] = []
    self.max_entries = max_entries

  def write(self, entry: LogEntry) -> None:
    """Write log entry to memory"""
    self.entries.append(entry)

    # Trim if needed
    if len(self.entries) > self.max_entries:
      self.entries = self.entries[-self.max_entries :]

  def get_entries(self, level: Optional[LogLevel] = None, component: Optional[str] = None, namespace: Optional[str] = None) -> List[LogEntry]:
    """
    Get filtered log entries

    Args:
        level: Filter by minimum log level
        component: Filter by component name
        namespace: Filter by namespace

    Returns:
        List of matching log entries
    """
    result = self.entries.copy()

    if level is not None:
      result = [e for e in result if e.level >= level]

    if component is not None:
      result = [e for e in result if e.component == component]

    if namespace is not None:
      result = [e for e in result if e.namespace == namespace]

    return result

  def clear(self) -> None:
    """Clear all stored entries"""
    self.entries.clear()


class Logger:
  """Main logger class with hierarchical support"""

  _loggers: Dict[str, "Logger"] = {}
  _root_logger: Optional["Logger"] = None

  @classmethod
  def get_root_logger(cls) -> "Logger":
    """Get or create the root logger"""
    if cls._root_logger is None:
      cls._root_logger = Logger(level=LogLevel.INFO, component="default", namespace="root", transports=[ConsoleTransport()])
    return cls._root_logger

  @classmethod
  def get_logger(cls, component: str, namespace: Optional[str] = None) -> "Logger":
    """
    Get a logger by component and namespace, creating it if needed

    Args:
        component: Component name (e.g., 'agent', 'tool')
        namespace: Optional namespace within the component

    Returns:
        Logger instance
    """
    key = f"{component}:{namespace}" if namespace else component

    if key not in cls._loggers:
      parent = cls.get_root_logger()
      cls._loggers[key] = Logger(level=parent.level, component=component, namespace=namespace or "", parent=parent)

    return cls._loggers[key]

  def __init__(
    self,
    level: LogLevel = LogLevel.INFO,
    component: str = "default",
    namespace: str = "",
    parent: Optional["Logger"] = None,
    transports: Optional[List[LogTransport]] = None,
  ):
    """
    Initialize a new logger

    Args:
        level: Minimum log level to record
        component: Component name (e.g., 'agent', 'tool')
        namespace: Namespace within the component
        parent: Parent logger for hierarchy
        transports: List of log transports
    """
    self.level = level
    self.component = component
    self.namespace = namespace
    self.parent = parent
    self.transports: List[LogTransport] = transports or []
    self._child_loggers: Set["Logger"] = set()

    # Register as child if we have a parent
    if parent:
      parent._child_loggers.add(self)

  def _log(self, level: LogLevel, message: str, data: Any = None, error: Optional[Exception] = None) -> None:
    """Internal logging implementation"""
    # Skip if level is too low
    if level < self.level:
      return

    # Create log entry
    entry = LogEntry(
      timestamp=datetime.datetime.now(), level=level, component=self.component, namespace=self.namespace, message=message, data=data, error=error
    )

    # Write to all transports
    for transport in self.transports:
      with suppress(Exception):
        transport.write(entry)

    # Also log to parent if available
    if self.parent and not self.transports:
      self.parent._log(level, message, data, error)

  def set_level(self, level: LogLevel, propagate: bool = False) -> None:
    """
    Set the log level for this logger

    Args:
        level: New log level
        propagate: Whether to propagate to child loggers
    """
    self.level = level

    # Propagate to children if requested
    if propagate and self._child_loggers:
      for child in self._child_loggers:
        child.set_level(level, propagate=True)

  def add_transport(self, transport: LogTransport) -> None:
    """Add a transport to this logger"""
    self.transports.append(transport)

  def remove_transport(self, transport: LogTransport) -> None:
    """Remove a transport from this logger"""
    if transport in self.transports:
      self.transports.remove(transport)

  def create_child(self, namespace: str) -> "Logger":
    """
    Create a child logger with the same component but different namespace

    Args:
        namespace: Child namespace

    Returns:
        Child logger instance
    """
    full_namespace = f"{self.namespace}.{namespace}" if self.namespace else namespace

    child = Logger(level=self.level, component=self.component, namespace=full_namespace, parent=self)

    return child

  # Log level methods
  def trace(self, message: str, data: Any = None) -> None:
    """Log a trace message"""
    self._log(LogLevel.TRACE, message, data)

  def debug(self, message: str, data: Any = None) -> None:
    """Log a debug message"""
    self._log(LogLevel.DEBUG, message, data)

  def info(self, message: str, data: Any = None) -> None:
    """Log an info message"""
    self._log(LogLevel.INFO, message, data)

  def warn(self, message: str, data: Any = None) -> None:
    """Log a warning message"""
    self._log(LogLevel.WARN, message, data)

  def error(self, message: str, error: Optional[Exception] = None, data: Any = None) -> None:
    """Log an error message"""
    self._log(LogLevel.ERROR, message, data, error)

  def fatal(self, message: str, error: Optional[Exception] = None, data: Any = None) -> None:
    """Log a fatal error message"""
    self._log(LogLevel.FATAL, message, data, error)

  def flush(self) -> None:
    """Flush all transports"""
    for transport in self.transports:
      with suppress(Exception):
        transport.flush()

    # Also flush parent if available
    if self.parent:
      self.parent.flush()


# ---- Utility functions for easy setup ----


def setup_console_logger(level: LogLevel = LogLevel.INFO, component: str = "default", show_timestamp: bool = True) -> Logger:
  """
  Create and configure a logger with console output

  Args:
      level: Minimum log level
      component: Component name
      show_timestamp: Whether to show timestamps

  Returns:
      Configured logger instance
  """
  logger = Logger(level=level, component=component, transports=[ConsoleTransport(show_timestamp=show_timestamp)])
  return logger


def setup_file_logger(
  directory: str,
  filename: Optional[str] = None,
  level: LogLevel = LogLevel.DEBUG,
  component: str = "default",
  format_json: bool = False,
  retention: Optional[RetentionOptions] = None,
) -> Logger:
  """
  Create and configure a logger with file output

  Args:
      directory: Directory to store log files
      filename: Base filename (default uses component name)
      level: Minimum log level
      component: Component name
      format_json: Whether to format logs as JSON
      retention: Retention policy options

  Returns:
      Configured logger instance
  """
  if filename is None:
    filename = f"{component}-logs"

  logger = Logger(
    level=level, component=component, transports=[FileTransport(directory=directory, filename=filename, format_json=format_json, retention=retention)]
  )
  return logger


def setup_multi_output_logger(
  directory: str, component: str, console_level: LogLevel = LogLevel.INFO, file_level: LogLevel = LogLevel.DEBUG, format_json: bool = False
) -> Logger:
  """
  Create and configure a logger with both console and file output

  Args:
      directory: Directory to store log files
      component: Component name
      console_level: Minimum level for console output
      file_level: Minimum level for file output
      format_json: Whether to format file logs as JSON

  Returns:
      Configured logger instance
  """
  # Create console transport
  console = ConsoleTransport()

  # Create file transport
  file = FileTransport(directory=directory, filename=f"{component}-logs", format_json=format_json)

  # Create logger with both transports
  # Use the lower of the two levels
  min_level = min(console_level, file_level)

  logger = Logger(level=min_level, component=component, transports=[console, file])

  return logger


# ----- Example Usage -----

if __name__ == "__main__":
  """
    Full-fledged example demonstrating all major features of the AgentLogger library
    """
  import os
  import random
  import tempfile
  import time
  from threading import Thread

  # Create a temporary directory for log files
  log_dir = tempfile.mkdtemp()
  print(f"Storing logs in: {log_dir}")

  # ===== BASIC SETUP =====

  # 1. Setup a root logger with console output
  root_logger = Logger.get_root_logger()
  root_logger.set_level(LogLevel.INFO)
  root_logger.info("AgentLogger Demo Starting", {"version": "1.0.0"})

  # 2. Create a multi-output logger for the agent component
  agent_logger = setup_multi_output_logger(
    directory=log_dir, component="agent", console_level=LogLevel.INFO, file_level=LogLevel.DEBUG, format_json=True
  )

  # 3. Create loggers for other components
  tool_logger = setup_console_logger(level=LogLevel.DEBUG, component="tool")

  kb_logger = setup_console_logger(level=LogLevel.INFO, component="knowledgeBase")

  workflow_logger = setup_file_logger(
    directory=log_dir,
    component="workflow",
    level=LogLevel.DEBUG,
    format_json=False,
    retention=RetentionOptions(
      max_files=5,
      max_size=1024 * 100,  # Small size for demo
      max_age=24 * 60 * 60 * 1000,  # 1 day
    ),
  )
  workflow_logger.add_transport(ConsoleTransport())  # Add console output too

  # ===== DEMONSTRATE LOG ROTATION =====

  print("\n\nDemonstrating log rotation...")
  try:
    # Create a simpler rotation logger with larger file size
    rotation_logger = setup_file_logger(
      directory=log_dir,
      filename="rotation-test",
      level=LogLevel.DEBUG,
      retention=RetentionOptions(
        max_files=2,  # Reduced to 2 files
        max_size=2048,  # 2KB per file
        max_age=3600000,  # 1 hour
      ),
    )

    print("Starting log rotation test...")

    # Write just 5 messages with small payloads
    for i in range(5):
      print(f"Writing message {i + 1}/5...")
      try:
        rotation_logger.info(f"Test message {i + 1}")
        rotation_logger.flush()  # Force flush after each message
        print(f"  - Message {i + 1} written successfully")
      except Exception as e:
        print(f"  - Error writing message {i + 1}: {str(e)}")
        raise

    print("\nChecking log files...")
    log_files = [f for f in os.listdir(log_dir) if f.startswith("rotation-test")]
    if not log_files:
      print("No log files found!")
    else:
      print(f"Found {len(log_files)} log files:")
      for file in sorted(log_files):
        file_path = os.path.join(log_dir, file)
        size = os.path.getsize(file_path)
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        print(f"  - {file} ({size:,} bytes, modified: {mtime.strftime('%H:%M:%S')})")

  except Exception as e:
    print(f"\nError during log rotation test: {str(e)}")
    import traceback

    traceback.print_exc()

  # ===== DEMONSTRATE MEMORY TRANSPORT =====

  print("\n\nDemonstrating in-memory logging...")
  memory_transport = MemoryTransport(max_entries=100)

  mem_logger = Logger(level=LogLevel.DEBUG, component="memory-test", transports=[memory_transport, ConsoleTransport()])

  # Log some messages
  mem_logger.info("First message")
  mem_logger.debug("Debug details", {"value": 42})
  mem_logger.warn("Warning message")
  mem_logger.error("Error occurred")

  # Retrieve and display logs from memory
  entries = memory_transport.get_entries(level=LogLevel.WARN)
  print(f"\nRetrieved {len(entries)} warning+ level entries from memory:")
  for entry in entries:
    print(f"  - {LOG_LEVEL_NAMES[entry.level]}: {entry.message}")

  # ===== DEMONSTRATE MULTI-THREADED LOGGING =====

  print("\n\nDemonstrating multi-threaded logging...")
  thread_logger = Logger(level=LogLevel.DEBUG, component="thread-test", transports=[ConsoleTransport()])

  def worker_task(worker_id):
    worker_logger = thread_logger.create_child(f"worker-{worker_id}")
    for i in range(3):
      worker_logger.info(f"Worker {worker_id} progress", {"step": i + 1})
      time.sleep(random.random() * 0.2)

  # Start some worker threads
  threads = []
  for i in range(3):
    thread = Thread(target=worker_task, args=(i + 1,))
    threads.append(thread)
    thread.start()

  # Wait for threads to complete
  for thread in threads:
    thread.join()

  # ===== CLEANUP AND SUMMARY =====

  # Flush all log transports
  agent_logger.flush()
  tool_logger.flush()
  kb_logger.flush()
  workflow_logger.flush()

  print("\n\nLogger demonstration complete!")
  print(f"Log files were created in: {log_dir}")
  print("Log files created:")

  # Count files by type
  file_counts = {}
  total_size = 0
  for file in os.listdir(log_dir):
    file_path = os.path.join(log_dir, file)
    size = os.path.getsize(file_path)
    total_size += size

    # Get the component name from filename
    component = file.split(".")[0]
    if component not in file_counts:
      file_counts[component] = {"count": 0, "size": 0}

    file_counts[component]["count"] += 1
    file_counts[component]["size"] += size

  # Print summary
  for component, stats in file_counts.items():
    print(f"  - {component}: {stats['count']} files ({stats['size']} bytes)")

  print(f"Total log size: {total_size} bytes")
  print("\nIn a real application, you might want to clean up old logs:")
  print(f"  import shutil; shutil.rmtree('{log_dir}')")

  # Example of using the logger in a context manager pattern
  print("\nUsing logger in a 'with' context:")

  class LogContext:
    def __init__(self, logger, context_name):
      self.logger = logger
      self.context_name = context_name

    def __enter__(self):
      self.logger.info(f"Starting context: {self.context_name}")
      self.start_time = time.time()
      return self

    def __exit__(self, exc_type, exc_val, exc_tb):
      duration = time.time() - self.start_time
      if exc_type:
        self.logger.error(f"Context {self.context_name} failed", error=exc_val)
        return False
      else:
        self.logger.info(f"Context {self.context_name} completed", {"duration_ms": round(duration * 1000, 2)})
      return True

  # Use the context manager
  with LogContext(agent_logger, "final-operation"):
    agent_logger.info("Performing final operation")
    # Simulated work
    time.sleep(0.5)

  print("\nLogger demo complete.")
