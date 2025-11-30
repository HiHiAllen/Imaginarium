import logging
import os
import sys

class StreamToLogger:
    """
    将 stdout/stderr 输出重定向到日志文件的类。
    
    智能识别输出类型：
    - tqdm 进度条 -> INFO
    - 普通信息 -> INFO (stdout) 或 WARNING (stderr)
    - 真正的错误 -> ERROR
    """
    def __init__(self, logger_obj, log_level=logging.INFO, original_stream=None, console_handler=None, is_stderr=False):
        self.logger = logger_obj
        self.log_level = log_level
        self.linebuf = ''
        self.original_stream = original_stream
        self.console_handler = console_handler
        self.is_stderr = is_stderr
        
    def _detect_log_level(self, line: str) -> int:
        """
        智能检测日志级别。
        
        Args:
            line: 日志行内容
            
        Returns:
            日志级别
        """
        line_lower = line.lower().strip()
        
        # 1. 识别 tqdm 进度条（包含 % 和 进度条字符）
        if '%|' in line or 'it/s]' in line:
            return logging.INFO
        
        # 2. 如果是 stderr，检查是否是真正的错误
        if self.is_stderr:
            # 常见的错误关键词
            error_keywords = [
                'error:', 'exception:', 'traceback', 'failed:', 
                'failure:', 'critical:', 'fatal:', 'panic:',
                'cannot', 'could not', 'unable to'
            ]
            
            # 如果包含错误关键词，标记为 ERROR
            if any(keyword in line_lower for keyword in error_keywords):
                return logging.ERROR
            
            # 警告关键词
            warning_keywords = ['warning:', 'warn:', 'deprecated:']
            if any(keyword in line_lower for keyword in warning_keywords):
                return logging.WARNING
            
            # 其他 stderr 输出标记为 INFO（很多库只是用 stderr 做普通输出）
            return logging.INFO
        
        # 3. stdout 默认为 INFO
        return self.log_level
        
    def write(self, buf):
        """写入缓冲区"""
        # 直接写入原始流（终端）
        if self.original_stream:
            self.original_stream.write(buf)
            self.original_stream.flush()
        
        # 临时移除 console handler，避免重复输出
        console_removed = False
        if self.console_handler and self.console_handler in self.logger.handlers:
            self.logger.removeHandler(self.console_handler)
            console_removed = True
        
        # 将内容写入日志文件
        for line in buf.rstrip().splitlines():
            if line.strip():  # 只记录非空行
                # 智能检测日志级别
                level = self._detect_log_level(line)
                self.logger.log(level, line.rstrip())
        
        # 恢复 console handler
        if console_removed:
            self.logger.addHandler(self.console_handler)
    
    def flush(self):
        """刷新缓冲区"""
        if self.original_stream:
            self.original_stream.flush()

class Logger:
    """
    Unified Logger class for Imaginarium.
    Imaginarium 统一日志类。
    
    功能：
    1. 记录通过 logger.info() 等方法的日志
    2. 捕获并记录所有 print() 输出
    3. 捕获并记录所有 stderr 输出
    4. 支持分阶段的日志文件
    """
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    def __init__(self, name='Imaginarium', log_file=None, level='INFO', stage_log_dir=None):
        self.logger = logging.getLogger(name)
        
        # Set level
        if isinstance(level, str):
            level = self.LOG_LEVELS.get(level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Clear existing handlers to prevent duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # Formatter - 为了更好地记录 print 输出，使用更简单的格式
        self.detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.simple_formatter = logging.Formatter('%(message)s')  # 用于 print 输出
        
        # Console Handler - 使用无缓冲的stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.simple_formatter)  # 控制台使用简单格式
        console_handler.setLevel(level)
        self.logger.addHandler(console_handler)
        self.console_handler = console_handler
        
        # File Handler (Optional)
        self.main_log_file = log_file
        self.stage_log_dir = stage_log_dir
        self.file_handler = None
        self.stage_file_handler = None
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self.file_handler = logging.FileHandler(log_file, mode='w') # Overwrite mode for new run
            self.file_handler.setFormatter(self.detailed_formatter)
            self.file_handler.setLevel(level)
            self.logger.addHandler(self.file_handler)
        
        # 创建 stage_logs 目录
        if stage_log_dir:
            os.makedirs(stage_log_dir, exist_ok=True)
        
        # 保存handlers引用以便立即flush
        self.handlers = self.logger.handlers
        
        # 当前阶段日志文件
        self.current_stage = None
        
        # 保存原始的 stdout 和 stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # 重定向标志
        self.stdout_redirected = False
        self.stderr_redirected = False
        
    def _flush_all(self):
        """立即刷新所有handler的输出缓冲区"""
        for handler in self.handlers:
            handler.flush()
            
    def info(self, msg):
        self.logger.info(msg)
        self._flush_all()
        
    def debug(self, msg):
        self.logger.debug(msg)
        self._flush_all()
        
    def warning(self, msg):
        self.logger.warning(msg)
        self._flush_all()
        
    def error(self, msg):
        self.logger.error(msg)
        self._flush_all()
        
    def critical(self, msg):
        self.logger.critical(msg)
        self._flush_all()
    
    def get_logger(self):
        """Return self for compatibility with legacy code that expects .get_logger() method."""
        return self
    
    def start_stage(self, stage_name: str):
        """
        开始一个新阶段的日志记录，创建该阶段的独立日志文件。
        同时重定向 stdout 和 stderr 到该阶段的日志文件。
        
        Args:
            stage_name: 阶段名称，如 "S0_geometry", "S1_parsing" 等
        """
        # 如果已有阶段日志，先关闭它
        if self.stage_file_handler:
            self.end_stage()
        
        # 创建新的阶段日志文件
        if self.stage_log_dir:
            stage_log_file = os.path.join(self.stage_log_dir, f"{stage_name}.log")
            self.stage_file_handler = logging.FileHandler(stage_log_file, mode='w')
            self.stage_file_handler.setFormatter(self.detailed_formatter)
            self.stage_file_handler.setLevel(self.logger.level)
            self.logger.addHandler(self.stage_file_handler)
            self.handlers = self.logger.handlers
            self.current_stage = stage_name
            
            self.info(f"=" * 70)
            self.info(f"开始阶段: {stage_name}")
            self.info(f"阶段日志文件: {stage_log_file}")
            self.info(f"=" * 70)
            
            # 重定向 stdout 和 stderr
            self._redirect_streams()
    
    def end_stage(self):
        """
        结束当前阶段的日志记录，关闭并保存阶段日志文件。
        同时恢复 stdout 和 stderr。
        """
        if self.stage_file_handler:
            # 先恢复 stdout 和 stderr
            self._restore_streams()
            
            if self.current_stage:
                self.info(f"=" * 70)
                self.info(f"结束阶段: {self.current_stage}")
                self.info(f"=" * 70 + "\n")
            
            # 刷新并关闭阶段日志handler
            self.stage_file_handler.flush()
            self.stage_file_handler.close()
            self.logger.removeHandler(self.stage_file_handler)
            self.stage_file_handler = None
            self.current_stage = None
            self.handlers = self.logger.handlers
    
    def _redirect_streams(self):
        """重定向 stdout 和 stderr 到日志"""
        if not self.stdout_redirected:
            sys.stdout = StreamToLogger(self.logger, logging.INFO, self.original_stdout, self.console_handler, is_stderr=False)
            self.stdout_redirected = True
        
        if not self.stderr_redirected:
            sys.stderr = StreamToLogger(self.logger, logging.INFO, self.original_stderr, self.console_handler, is_stderr=True)
            self.stderr_redirected = True
    
    def _restore_streams(self):
        """恢复原始的 stdout 和 stderr"""
        if self.stdout_redirected:
            sys.stdout = self.original_stdout
            self.stdout_redirected = False
        
        if self.stderr_redirected:
            sys.stderr = self.original_stderr
            self.stderr_redirected = False
    
    def cleanup(self):
        """清理资源，恢复标准流"""
        # 恢复标准流
        self._restore_streams()
        
        # 关闭所有文件handler
        if self.stage_file_handler:
            self.stage_file_handler.flush()
            self.stage_file_handler.close()
            if self.stage_file_handler in self.logger.handlers:
                self.logger.removeHandler(self.stage_file_handler)
            self.stage_file_handler = None
        
        if self.file_handler:
            self.file_handler.flush()
            self.file_handler.close()
            if self.file_handler in self.logger.handlers:
                self.logger.removeHandler(self.file_handler)
            self.file_handler = None
    
    def __del__(self):
        """析构时确保恢复标准流"""
        try:
            self.cleanup()
        except:
            pass  # 忽略析构时的异常

