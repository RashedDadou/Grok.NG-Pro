# utils.py
"""
ملف الأدوات المساعدة (Utilities) المشتركة في مشروع Grok.NG Pro
يحتوي على دوال عامة يمكن استخدامها في engine.py و gui.py وغيرها
"""

import os
import re
import time
import logging
from datetime import datetime
from typing import Optional, Tuple


def get_timestamp() -> str:
    """
    إرجاع timestamp موحد لاستخدامه في تسمية الملفات
    صيغة: YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_filename(base_name: str, extension: str = ".png") -> str:
    """
    تنظيف اسم الملف من الحروف الممنوعة في أنظمة التشغيل
    """
    # إزالة أي حرف غير آمن
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', base_name)
    # تقصير لو طويل جدًا
    safe = safe[:100]
    return f"{safe}_{get_timestamp()}{extension}"


def estimate_processing_time(base_time: float, multiplier: float = 1.0, interaction_impact: float = 0.0) -> float:
    """
    تقدير الوقت الكلي بناءً على عوامل مختلفة
    """
    total = base_time * multiplier + interaction_impact
    return round(max(total, 1.0), 1)  # مش أقل من ثانية واحدة


def log_progress(step: str, progress: int, total_steps: int):
    """
    تسجيل تقدم العملية في الـ logging (مفيد للـ debug)
    """
    percentage = int((progress / total_steps) * 100)
    logging.info(f"[{percentage:3d}%] {step}")


def create_directory_if_not_exists(path: str) -> None:
    """
    إنشاء مجلد لو مش موجود
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"تم إنشاء المجلد: {path}")


def format_duration(seconds: float) -> str:
    """
    تحويل الثواني لصيغة سهلة القراءة (مثال: 1m 23s)
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"


# إعداد logging أساسي (يمكن استيراده في الملفات التانية)
def setup_logging(level: int = logging.INFO):
    """
    إعداد logging موحد لكل المشروع
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )


# مثال على كائنات ثابتة (constants)
RESOLUTIONS = {
    "HD": (1280, 720),
    "FHD": (1920, 1080),
    "4K": (3840, 2160)
}

BASE_OUTPUT_DIR = "grokng_outputs"

# إنشاء مجلد الإخراج الافتراضي عند الاستيراد
create_directory_if_not_exists(BASE_OUTPUT_DIR)