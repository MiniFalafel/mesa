#pragma once

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
        static void LogERROR(const std::string& message);
        static void LogWARN(const std::string& message);
        static void LogINFO(const std::string& message);

        inline static void ASSERT(bool x, const std::string& message) { if(!x) { LogERROR(message); } }

        static void SetColor(LogLevel level, LoggerColor color);

    private: // methods
        static void LogMessage(LogLevel level, const std::string& tag, const std::string& message);

    private: // data
        static LogLevel s_LogLevel;
        static std::map<LogLevel, LoggerColor> s_Colors;
        static std::map<LoggerColor, const char*> s_LogColors;

    };
}
