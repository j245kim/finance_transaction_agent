#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import signal
import json
from pathlib import Path


def handle_shutdown(signum, frame):
    # config.json 파일의 PROCESS_VARIABLE 값을 False로 바꾸고 종료
    json_path = rf'{Path(__file__).parents[0]}/realtime_collect/config.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'PROCESS_VARIABLE': 'False'}, f)
    sys.exit(0)


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    # SIGINT (Ctrl+C) 또는 SIGTERM (시스템 종료) 시그널 감지
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    main()