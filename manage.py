#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import json
from pathlib import Path


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
    main()

    # 매번 실행마다 messages 초기화
    session_path = rf'{Path(__file__).parents[0]}\session\messages.json'
    system_instruction = "You are a very smart AI chatbot. Please answer user questions accurately and kindly. 당신은 아주 똑똑한 AI 챗봇입니다. 사용자의 질문에 정확하고, 친절하게 답변해주세요."
    messages = [
                    {
                        "role": "system",
                        "content": system_instruction
                    }
                ]
    with open(session_path, mode='w', encoding='utf-8', errors='ignore') as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)
