#include "ezlog.hpp"

#include <iostream>
#include <csignal>


namespace ezlog
{
    // STATIC INITIALIZATION
    LogLevel Logger::s_LogLevel = LogLevel::INFO;

    std::map<LoggerColor, const char*> Logger::s_LogColors = {
        { LoggerColor::NORMAL, "\e[0m" },
        { LoggerColor::RED,    "\e[31m" },
        { LoggerColor::YELLOW, "\e[33m" },
        { LoggerColor::GREEN,  "\e[32m" },
    };

    std::map<LogLevel, LoggerColor> Logger::s_Colors = {
        { LogLevel::NONE,  LoggerColor::NORMAL },
        { LogLevel::ERROR, LoggerColor::NORMAL },
        { LogLevel::WARN,  LoggerColor::NORMAL },
        { LogLevel::INFO,  LoggerColor::NORMAL },
    };

    // PUBLIC METHODS
    void Logger::SetColor(LogLevel level, LoggerColor color)
    {
        s_Colors[level] = color;
    }

    void Logger::LogERROR(const std::string& message)
    {
        LogMessage(LogLevel::ERROR, "ERROR", message);
        // Signal stop
        std::raise(SIGINT);
    }

    void Logger::LogWARN(const std::string& message)
    {
        LogMessage(LogLevel::WARN, "WARN", message);
    }

    void Logger::LogINFO(const std::string& message)
    {
        LogMessage(LogLevel::INFO, "INFO", message);
    }

    // PRIVATE METHODS
    void Logger::LogMessage(LogLevel level, const std::string& tag, const std::string& message)
    {
        if (level <= s_LogLevel)
            std::cout << "[" << s_LogColors[s_Colors[level]] << tag << s_LogColors[LoggerColor::NORMAL] << "]::" << message << std::endl;
    }
}
