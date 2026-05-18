"""
日志模块 — 提供统一的日志配置和工具函数。

功能特性：
  - 统一的日志格式和级别配置
  - 同时输出到终端和文件
  - 支持不同模块的独立日志级别
  - 颜色化终端输出（可选）
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional


# 日志级别映射
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器。"""
    
    COLORS = {
        "DEBUG": "\033[94m",      # 蓝色
        "INFO": "\033[92m",       # 绿色
        "WARNING": "\033[93m",    # 黄色
        "ERROR": "\033[91m",      # 红色
        "CRITICAL": "\033[95m",   # 紫色
        "RESET": "\033[0m",       # 重置
    }
    
    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            record.msg = f"{self.COLORS.get(levelname, '')}{record.msg}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    name: str,
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    file: bool = True,
    colored: bool = True,
) -> logging.Logger:
    """
    设置日志配置。
    
    Args:
        name: 日志器名称
        level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        log_dir: 日志文件目录（默认 logs/）
        log_file: 日志文件名（默认 {name}_{日期}.log）
        console: 是否输出到终端
        file: 是否输出到文件
        colored: 终端输出是否带颜色
    
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level, logging.INFO))
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建日志目录
    if file:
        if not log_dir:
            log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        if not log_file:
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = f"{name}_{date_str}.log"
        log_path = os.path.join(log_dir, log_file)
    
    # 终端处理器
    if console:
        console_handler = logging.StreamHandler()
        console_format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
        if colored:
            console_handler.setFormatter(ColoredFormatter(console_format))
        else:
            console_handler.setFormatter(logging.Formatter(console_format))
        logger.addHandler(console_handler)
    
    # 文件处理器
    if file:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取或创建日志器。"""
    return logging.getLogger(name)


# 预设的日志器
logger = setup_logging("parlantrag")
