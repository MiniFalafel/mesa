#pragma once

#include <iostream>
#include <csignal>
#include <string>
#include <format>
#include <map>

namespace ezlog
{
    enum class LogLevel
    {
        NONE = 0,
        ERROR,
        WARN,
        INFO
    };

    enum class LoggerColor
    {
        NORMAL = 0,
        RED,
        YELLOW,
        GREEN,
    };

    class Logger
    {
    public:
        // API/public methods
        static void LogERROR(const std::string& message)
        {
            LogMessage(LogLevel::ERROR, "ERROR", message);
            // Signal stop
            std::raise(SIGINT);
        }
        static void LogWARN(const std::string& message)
        {
            LogMessage(LogLevel::WARN, "WARN", message);
        }
        static void LogINFO(const std::string& message)
        {
            LogMessage(LogLevel::INFO, "INFO", message);
        }

        inline static void ASSERT(bool x, const std::string& message) { if(!x) { LogERROR(message); } }

        static void SetColor(LogLevel level, LoggerColor color)
        {
            s_Colors[level] = color;
        }

    private: // methods
        static void LogMessage(LogLevel level, const std::string& tag, const std::string& message)
        {
            if (level <= s_LogLevel)
                std::cout << "[" << s_LogColors[s_Colors[level]] << tag << s_LogColors[LoggerColor::NORMAL] << "]::" << message << std::endl;
        }

    private: // data
        inline static LogLevel s_LogLevel = LogLevel::INFO;
        inline static std::map<LogLevel, LoggerColor> s_Colors = {
            { LogLevel::NONE,  LoggerColor::NORMAL },
            { LogLevel::ERROR, LoggerColor::NORMAL },
            { LogLevel::WARN,  LoggerColor::NORMAL },
            { LogLevel::INFO,  LoggerColor::NORMAL },
        };
        inline static std::map<LoggerColor, const char*> s_LogColors = {
            { LoggerColor::NORMAL, "\e[0m" },
            { LoggerColor::RED,    "\e[31m" },
            { LoggerColor::YELLOW, "\e[33m" },
            { LoggerColor::GREEN,  "\e[32m" },
        };
    };
}
