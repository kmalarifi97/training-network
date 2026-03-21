import ctypes
import ctypes.wintypes
import logging
import time

import psutil

logger = logging.getLogger("agent.idle_detector")


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.wintypes.UINT),
        ("dwTime", ctypes.wintypes.DWORD),
    ]


class IdleDetector:
    GAME_PROCESS_NAMES = {
        "gta5.exe", "gtav.exe", "fivem.exe", "fivem_b2699_gta5.exe",
        "valorant.exe", "csgo.exe", "cs2.exe", "dota2.exe",
        "fortnite.exe", "rocketleague.exe", "pubg.exe",
        "overwatch.exe", "leagueoflegends.exe", "apex_legends.exe",
        "minecraft.exe", "javaw.exe",
        "steam_app_", "epicgameslauncher.exe",
    }

    def __init__(self, idle_threshold_minutes: int = 5):
        self.idle_threshold_seconds = idle_threshold_minutes * 60
        self._last_input_info = LASTINPUTINFO()
        self._last_input_info.cbSize = ctypes.sizeof(LASTINPUTINFO)

    def get_idle_seconds(self) -> float:
        ctypes.windll.user32.GetLastInputInfo(ctypes.byref(self._last_input_info))
        tick_count = ctypes.windll.kernel32.GetTickCount()
        idle_ms = tick_count - self._last_input_info.dwTime
        return idle_ms / 1000.0

    def is_idle(self) -> bool:
        return self.get_idle_seconds() >= self.idle_threshold_seconds

    def is_fullscreen_app_running(self) -> bool:
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            if not hwnd:
                return False

            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))

            # Get screen dimensions
            screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            screen_h = ctypes.windll.user32.GetSystemMetrics(1)

            window_w = rect.right - rect.left
            window_h = rect.bottom - rect.top

            return window_w >= screen_w and window_h >= screen_h
        except Exception:
            return False

    def is_game_running(self) -> bool:
        try:
            for proc in psutil.process_iter(["name"]):
                name = proc.info["name"]
                if name and name.lower() in self.GAME_PROCESS_NAMES:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return False

    def is_user_active(self) -> bool:
        if not self.is_idle():
            return True
        if self.is_game_running():
            return True
        if self.is_fullscreen_app_running():
            return True
        return False

    def get_status(self) -> dict:
        idle_secs = self.get_idle_seconds()
        return {
            "idle_seconds": round(idle_secs, 1),
            "is_idle": idle_secs >= self.idle_threshold_seconds,
            "game_running": self.is_game_running(),
            "fullscreen_app": self.is_fullscreen_app_running(),
            "user_active": self.is_user_active(),
        }
