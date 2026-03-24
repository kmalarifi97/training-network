"""Windows Service wrapper for GPU Network Agent.

Install:   python service_wrapper.py install
Start:     python service_wrapper.py start
Stop:      python service_wrapper.py stop
Remove:    python service_wrapper.py remove
Debug:     python service_wrapper.py debug  (runs in console)

Requires: pip install pywin32
"""

import sys
import os
import asyncio
import logging

# Add agent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("agent.service")

try:
    import win32serviceutil
    import win32service
    import win32event
    import servicemanager

    class GPUNetworkService(win32serviceutil.ServiceFramework):
        _svc_name_ = "GPUNetworkAgent"
        _svc_display_name_ = "GPU Network Agent"
        _svc_description_ = "Connects this PC's idle GPU to the GPU Network for AI workloads."

        def __init__(self, args):
            win32serviceutil.ServiceFramework.__init__(self, args)
            self.stop_event = win32event.CreateEvent(None, 0, 0, None)
            self.agent = None

        def SvcStop(self):
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.stop_event)
            if self.agent:
                self.agent.stop()

        def SvcDoRun(self):
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, ""),
            )
            self.main()

        def main(self):
            from main import Agent
            self.agent = Agent()
            try:
                asyncio.run(self.agent.run())
            except Exception as e:
                servicemanager.LogErrorMsg(f"GPU Network Agent error: {e}")

    HAS_WIN32 = True

except ImportError:
    HAS_WIN32 = False


def install_service():
    """Install the Windows service."""
    if not HAS_WIN32:
        print("ERROR: pywin32 not installed. Run: pip install pywin32")
        sys.exit(1)

    # Use the frozen exe path if running as PyInstaller bundle
    if getattr(sys, 'frozen', False):
        exe_path = sys.executable
    else:
        exe_path = sys.executable  # python.exe

    print(f"Installing GPU Network Agent service...")
    print(f"  Executable: {exe_path}")
    print(f"  Script: {os.path.abspath(__file__)}")

    sys.argv = ["service_wrapper.py", "install"]
    win32serviceutil.HandleCommandLine(GPUNetworkService)


def main():
    if not HAS_WIN32:
        print("ERROR: pywin32 not installed. Run: pip install pywin32")
        print("")
        print("To run the agent without a service, use: python main.py")
        sys.exit(1)

    if len(sys.argv) == 1:
        # Called without args — running as a service
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(GPUNetworkService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(GPUNetworkService)


if __name__ == "__main__":
    main()
