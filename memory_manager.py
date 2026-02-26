# memory_manager.py
"""
مدير الذاكرة التوليدية - يتعامل مع الكشف عن المحتوى المرعب، التنظيف، التخزين المؤقت، والاسترجاع الآمن
"""

import re
import json
import hashlib
import logging
import shutil
import json
import time
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from collections import defaultdict

from PIL import Image
from generation_result import GenerationResult

# from Image_generation import CoreImageGenerationEngine   # مؤقتًا معطل
CoreImageGenerationEngine = object  # placeholder عشان ما يرميش NameError

logger = logging.getLogger(__name__)


class GenerativeMemoryManager:
    def __init__(self, context_type: str = None, *args, **kwargs):
        """
        تهيئة مدير الذاكرة التوليدية
        """
        super().__init__(*args, **kwargs)
        self.context_type = context_type or "default"
        self.context_type = context_type
        self.memory_store: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []  # لـ LRU
        self.max_cache_size = 100  # عدد الإدخالات في الذاكرة المؤقتة
        self.default_ttl_seconds = 3600  # مدة صلاحية الذاكرة (ثواني)

        # تهيئة قوائم الكلمات المرعبة
        self.creepy_keywords: Set[str] = set()
        self.creepy_severity: Dict[str, int] = {}

        self._load_creepy_keywords()

        # إحصائيات
        self.stats = {
            "filtered_count": 0,
            "recovered_count": 0,
            "cache_hits": 0,
            "last_cleanup": time.time(),
        }

    def _load_creepy_keywords(self):
        """تحميل كلمات Creepy من ملف خارجي أو قائمة افتراضية"""
        keywords_file = Path(__file__).parent / "creepy_keywords.json"
        if keywords_file.exists():
            try:
                with open(keywords_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.creepy_keywords = set(data.get("keywords", []))
                    self.creepy_severity = data.get("severity", {})
                logger.info(f"تم تحميل {len(self.creepy_keywords)} كلمة مرعبة من الملف")
                return
            except Exception as e:
                logger.error(f"فشل تحميل ملف الكلمات المرعبة: {e}")

        # قائمة افتراضية في حالة الفشل
        logger.warning("استخدام قائمة كلمات مرعبة افتراضية")
        self.creepy_keywords = {
            # إنجليزي
            "creepy", "horror", "scary", "ghost", "monster", "blood", "gore", "kill", "murder",
            "torture", "zombie", "demon", "curse", "slaughter", "haunted", "terrifying",
            # عربي
            "مرعب", "كابوس", "شبح", "وحش", "دموي", "رعب", "قتل", "ذبح", "لعنة", "جثة",
        }
        self.creepy_severity = {
            "ghost": 1, "شبح": 1, "scary": 2, "مرعب": 2,
            "blood": 3, "قتل": 3, "gore": 3, "تعذيب": 3,
        }

    def load_raw_layer(self, prompt_hash: str, layer_name: str) -> Optional[Dict]:
        """قراءة البيانات الخام لطبقة معينة من الأرشيف"""
        archive_dir = Path("layer_archive") / prompt_hash
        if not archive_dir.exists():
            logger.warning(f"لا يوجد أرشيف لـ {prompt_hash}")
            return None

        # نفترض إنك تحفظ كـ json، فنشوف أحدث إصدار للطبقة
        files = list(archive_dir.glob(f"*_{layer_name}_*.json"))
        if not files:
            logger.warning(f"لا يوجد طبقة {layer_name} في {prompt_hash}")
            return None

        # اختيار أحدث ملف (بناءً على timestamp)
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"تم تحميل {layer_name} من {latest_file}")
            return data
        except Exception as e:
            logger.exception(f"فشل تحميل {layer_name} من {latest_file}")
            return None
    
    def load_environment_from_memory(self, prompt_hash: str = None) -> list[dict]:
        from environment_design_engine import environment_design_engine
        engine = environment_design_engine()
        result = engine.design("غابة مع نهر ورياح وإضاءة صباحية")  # أو أي وصف
        layers = []
        
        for elem in result.elements:
            layer = PlaneLayer(
                position=elem.position,
                force=0.2 + np.random.uniform(0, 0.3),
                depth=-elem.position[2],
                label=elem.type,
                color=elem.properties.get("color", "gray"),
                mass=elem.properties.get("size", 1.0) or 1.0
            )
            layer.metadata = elem.properties
            layers.append(layer)
        
        return layers

    def save_raw_layer(self, prompt_hash: str, layer_name: str, layer_result: Any):
        key = f"{layer_name}_{prompt_hash}"
        self._storage[key] = layer_result  # أو أي طريقة حفظ عندك (dict, json, pickle, ...)
    
    def get_prompt_hash(self, prompt: str) -> str:
        """حساب hash فريد مختصر للـ prompt"""
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]

    def check_for_creepy(self, prompt: str, task_data: Dict) -> bool:
        """كشف سريع عن وجود محتوى مرعب"""
        lower = prompt.lower()
        if any(kw in lower for kw in self.creepy_keywords):
            return True

        mood = task_data.get("mood", "").lower()
        if mood in {"dark", "mysterious", "eerie", "horror"}:
            return True

        return False

    def _sanitize_task_data(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """تنظيف البيانات من أي محتوى مرعب"""
        cleaned = task_data.copy()

        # تنظيف entities
        entities = cleaned.get("entities", [])
        cleaned["entities"] = [e for e in entities if e.lower() not in self.creepy_keywords]

        # تنظيف raw_prompt إن وجد
        if "raw_prompt" in cleaned:
            p = cleaned["raw_prompt"]
            for kw in self.creepy_keywords:
                p = re.sub(rf'\b{re.escape(kw)}\b', "[filtered]", p, flags=re.I)
            cleaned["raw_prompt"] = p

        # إعادة تعيين mood إلى آمن إذا كان مرعباً
        if cleaned.get("mood", "").lower() in {"dark", "mysterious", "horror", "eerie"}:
            cleaned["mood"] = "neutral"

        cleaned["sanitized"] = True
        return cleaned

    def refresh_layer_before_and_after_storage(
        self,
        layer_result: "GenerationResult",
        layer_name: str,                    # "environment", "geometric", "traditional"
        stage: str = "before",              # "before" أو "after"
        prompt_hash: str = None,
        extra_checks: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        إنعاش / تدقيق طبقة قبل أو بعد التخزين الأصلي.
        يتم التحقق من: الألوان، الجودة، المقاسات.

        Args:
            layer_result: كائن GenerationResult
            layer_name: اسم الطبقة
            stage: "before" (قبل التخزين) أو "after" (بعد التخزين)
            prompt_hash: للربط بالأرشيف (اختياري)
            extra_checks: تحققات إضافية مخصصة (اختياري)

        Returns:
            تقرير تدقيق (dict) يحتوي على حالة النجاح + الملاحظات + أي اقتراحات تصحيح
        """
        report = {
            "success": True,
            "stage": stage,
            "layer": layer_name,
            "issues": [],
            "warnings": [],
            "suggestions": [],
            "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if not layer_result or not layer_result.success:
            report["success"] = False
            report["issues"].append("النتيجة الأصلية فاشلة أو غير موجودة")
            return report

        # استخراج المسار (إن وجد)
        path = self._extract_path_from_result(layer_result)
        if not path or not os.path.exists(path):
            report["success"] = False
            report["issues"].append("لا يوجد مسار صورة صالح أو الملف غير موجود")
            return report

        try:
            # فتح الصورة للتحقق
            with Image.open(path) as img:
                img.verify()  # التحقق من سلامة الملف
                img = Image.open(path)  # إعادة فتح بعد verify

                # 1. تدقيق المقاسات
                width, height = img.size
                expected_ratio = extra_checks.get("expected_ratio", None)
                if expected_ratio and abs(width / height - expected_ratio) > 0.15:
                    report["warnings"].append(
                        f"نسبة الأبعاد غير متوقعة: {width}x{height} (نسبة {width/height:.2f})"
                    )
                    report["suggestions"].append("إعادة توليد بأبعاد أقرب للنسبة المطلوبة")

                # مقاسات صغيرة جدًا = مشكلة جودة
                if width < 512 or height < 512:
                    report["issues"].append(f"دقة منخفضة جدًا: {width}x{height}")
                    report["success"] = False

                # 2. جودة البيانات / سلامة الملف
                mode = img.mode
                if mode not in ("RGB", "RGBA"):
                    report["warnings"].append(f"وضع لون غير مثالي: {mode} (يفضل RGB/RGBA)")

                file_size_kb = os.path.getsize(path) / 1024
                if file_size_kb < 50:
                    report["warnings"].append(f"حجم الملف صغير جدًا ({file_size_kb:.1f} KB) – قد تكون الجودة منخفضة")

                # 3. تفقد الألوان (بسيط – متوسط السطوع والتباين)
                from PIL import ImageStat
                stat = ImageStat.Stat(img.convert("RGB"))
                brightness = sum(stat.mean) / 3
                contrast = sum(stat.stddev) / 3

                if brightness < 40:
                    report["warnings"].append(f"الصورة مظلمة جدًا (متوسط السطوع: {brightness:.1f})")
                    report["suggestions"].append("زيادة السطوع أو إعادة توليد بإضاءة أفضل")
                if brightness > 220:
                    report["warnings"].append(f"الصورة فاتحة جدًا (متوسط السطوع: {brightness:.1f})")

                if contrast < 30:
                    report["warnings"].append(f"تباين منخفض (stddev: {contrast:.1f}) – قد تبدو الصورة باهتة")

        except Exception as e:
            report["success"] = False
            report["issues"].append(f"فشل فتح/تحليل الصورة: {str(e)}")

        # تسجيل التقرير
        if not report["success"]:
            logger.error(f"[Refresh {stage}] فشل تدقيق {layer_name}: {report['issues']}")
        elif report["issues"] or report["warnings"]:
            logger.warning(f"[Refresh {stage}] {layer_name} – {len(report['warnings'])} تحذيرات")
        else:
            logger.debug(f"[Refresh {stage}] {layer_name} – كل شيء سليم")

        return report
    
    # دالة مساعدة (إذا لم تكن موجودة)
    def _extract_path_from_result(self, result: GenerationResult) -> Optional[Path]:
        if not result.output_data:
            return None
        for key in ["preview_path", "path", "output_path", "file_path"]:
            p = result.output_data.get(key)
            if p and os.path.exists(p):
                return Path(p)
        return None

    def cleanup_temp_references(self):
        # إذا كنت تستخدم ملفات مؤقتة في مكان آخر، انقل التنظيف هنا أو استخدم context manager
        pass  # حالياً فارغة – يمكن توسيعها لاحقاً

    # ... باقي الدوال مثل check_for_creepy, advanced_creepy_filter, should_refresh_stages ...
    def get_prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]

    def _put_in_cache(self, key: str, value: Dict):
        """حفظ في الـ LRU cache"""
        with self.lock:
            self.memory_store[key] = value
            # إذا أردت TTL يدوي، يمكن إضافة timestamp وتنظيف لاحق

    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """استرجاع مع تحديث LRU"""
        with self.lock:
            value = self.memory_store.get(key)
            if value is not None:
                self.stats["cache_hits"] += 1
            return value

    def _quick_heuristic_score(self, prompt: str, task_data: Dict[str, Any]) -> float:
        score = 1.0

        # عقوبات
        if len(prompt) < 40:
            score *= 0.6   # prompt قصير جدًا → جودة منخفضة
        if len(task_data.get("entities", [])) < 2:
            score *= 0.7   # قلة الكيانات → وصف عام
        if "creepy_level" in task_data and task_data["creepy_level"] >= 2:
            score *= 0.5

        # مكافآت
        quality_indicators = {"detailed", "ultra", "cinematic", "masterpiece", "photorealistic", "8k"}
        if any(w in prompt.lower() for w in quality_indicators):
            score *= 1.25

        return max(0.3, min(1.5, score))
    
    def save_and_propagate_layer(
        self,
        layer_result: "GenerationResult",
        layer_name: str,                    # "background", "midground", "foreground"
        prompt_hash: str,                   # من memory_manager.get_prompt_hash(prompt)
        version_id: Optional[str] = None,   # إذا كان إعادة توليد، يُمرر v2, v3...
        extra_metadata: Optional[Dict] = None
    ) -> bool:
        """
        1. حفظ دائم في memory_manager (مع versioning)
        2. تمرير النتيجة الحالية إلى Final_Generation للدمج الفوري
        """
        success = True

        # ─── 1. التحضير للحفظ ────────────────────────────────────────────────
        data_to_save = {
            "layer_name": layer_name,
            "timestamp": time.time(),
            "prompt_hash": prompt_hash,
            "version_id": version_id or f"v{int(time.time())}",
            "success": layer_result.success,
            "total_time": layer_result.total_time,
            "output_data": layer_result.output_data or {},
            "specialization": layer_result.specialization,
            "is_video": layer_result.is_video,
            "extra_metadata": extra_metadata or {}
        }

        # إضافة مسار الملف إذا وُجد
        path = self._extract_layer_path(layer_result)
        if path:
            data_to_save["file_path"] = path
            data_to_save["file_exists"] = os.path.exists(path)

        # ─── 2. الحفظ الدائم في memory_manager ───────────────────────────────
        try:
            stored = self.memory_manager.store_layer_result(
                prompt_hash=prompt_hash,
                layer_data=data_to_save,
                version_id=data_to_save["version_id"]
            )
            if not stored:
                logger.warning(f"فشل حفظ دائم لـ {layer_name}")
                success = False
        except Exception as e:
            logger.exception(f"خطأ في حفظ دائم لـ {layer_name}")
            success = False

        # ─── 3. تمرير النسخة الحالية إلى Final_Generation للدمج ───────────────
        try:
            self.final_generator.receive_layer_result(
                layer_name=layer_name,
                result=layer_result,
                version_id=data_to_save["version_id"]
            )
        except Exception as e:
            logger.exception(f"فشل تمرير {layer_name} إلى Final_Generation")
            success = False

        return success
    
    # ----------------------------------------------------------------------
    # دالة 1: حفظ البيانات الأصلية من المحركات الثلاثة (دائم – original)
    # ----------------------------------------------------------------------
    def store_original_layer_result(
        self,
        prompt_hash: str,
        layer_name: str,                    # "environment" أو "geometric" أو "traditional"
        layer_result: "GenerationResult",     # النتيجة الكاملة من المحرك
        extra_metadata: dict = None
    ) -> bool:
        """
        حفظ نسخة أصلية دائمة من طبقة واحدة (من environment/geometric/traditional)
        تُحفظ مرة واحدة فقط لكل طبقة في كل prompt_hash
        """
        if not layer_result or not layer_result.success:
            logger.warning(f"محاولة حفظ أصلي فاشل لـ {layer_name}")
            return False

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        version_tag = f"original_{layer_name}_{timestamp}"

        payload = {
            "prompt_hash": prompt_hash,
            "layer_name": layer_name,
            "timestamp": timestamp,
            "version": version_tag,
            "success": layer_result.success,
            "total_time": layer_result.total_time,
            "specialization": layer_result.specialization,
            "is_video": layer_result.is_video,
            "output_data": layer_result.output_data or {},
            "metadata": layer_result.metadata or {},
            "extra_metadata": extra_metadata or {},
        }

        # إضافة مسار الملف إذا وُجد
        path = self._extract_path_from_result(layer_result)
        if path:
            payload["file_path"] = str(path)
            payload["file_exists"] = os.path.exists(path)

        # مسار الحفظ
        base_dir = Path("layer_archive") / prompt_hash
        base_dir.mkdir(parents=True, exist_ok=True)
        file_path = base_dir / f"{version_tag}.json"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"[Archive Original] حفظ أصلي لـ {layer_name} → {file_path}")
            return True
        except Exception as e:
            logger.exception(f"فشل حفظ أصلي {layer_name} → {file_path}")
            return False

    # ----------------------------------------------------------------------
    # دالة 2: استقبال وحفظ نتائج الدمج / التعديل من Final_Generation
    # ----------------------------------------------------------------------
    def store_composite_or_modified_result(
        self,
        prompt_hash: str,
        composite_result: GenerationResult,   # نتيجة الدمج أو التعديل
        source: str = "composite",            # "composite" أو "angle_change" أو "opacity_adjust" إلخ
        previous_version: Optional[str] = None,
        extra_metadata: dict = None
    ) -> str:
        """
        حفظ نسخة معدلة / مدمجة من Final_Generation كإصدار متسلسل
        ترجع رقم الإصدار الجديد (مثال: v4)
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # حساب رقم الإصدار الجديد (بسيط: أعلى رقم موجود + 1)
        archive_dir = Path("layer_archive") / prompt_hash
        if not archive_dir.exists():
            archive_dir.mkdir(parents=True, exist_ok=True)
            next_v = 1
        else:
            existing = [f.name for f in archive_dir.glob("v*.json")]
            versions = [int(f.split('_')[0][1:]) for f in existing if f.startswith("v") and f[1].isdigit()]
            next_v = max(versions) + 1 if versions else 1

        version_tag = f"v{next_v}_{source}_{timestamp}"

        payload = {
            "prompt_hash": prompt_hash,
            "version": version_tag,
            "source": source,
            "previous_version": previous_version,
            "timestamp": timestamp,
            "success": composite_result.success,
            "total_time": composite_result.total_time,
            "output_data": composite_result.output_data or {},
            "metadata": composite_result.metadata or {},
            "extra_metadata": extra_metadata or {},
        }

        path = self._extract_path_from_result(composite_result)
        if path:
            payload["file_path"] = str(path)
            payload["file_exists"] = os.path.exists(path)

        file_path = archive_dir / f"{version_tag}.json"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"[Archive Modified] حفظ إصدار {version_tag} → {file_path}")
            return version_tag
        except Exception as e:
            logger.exception(f"فشل حفظ إصدار {version_tag}")
            return None

    def store_environment_elements(
        self,
        prompt_hash: str,
        elements: list,
        analysis: dict,
        extra_metadata: dict = None
    ) -> bool:
        """
        حفظ تصميم البيئة الخام (عناصر + تحليل)
        """
        # دالة تحويل الكائنات المخصصة إلى dict عادي
        def serialize_element(elem):
            if hasattr(elem, '__dict__'):
                d = elem.__dict__.copy()
                # تحويل أي ndarray داخل الخصائص إلى list
                for k, v in d.items():
                    if isinstance(v, np.ndarray):
                        d[k] = v.tolist()
                    elif isinstance(v, dict):
                        d[k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                    elif isinstance(v, list):
                        d[k] = [vv.tolist() if isinstance(vv, np.ndarray) else vv for vv in v]
                return d
            return elem  # لو مش كلاس، رجّعه كما هو

        serialized_elements = [serialize_element(elem) for elem in elements]

        payload = {
            "type": "environment_design",
            "prompt_hash": prompt_hash,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "elements": serialized_elements,
            "analysis": analysis,  # نفترض إنه dict عادي
            "extra_metadata": extra_metadata or {}
        }

        key = f"env_{prompt_hash}"

        if hasattr(self, 'memory_store'):
            self.memory_store[key] = payload

        archive_dir = Path("layer_archive") / prompt_hash
        archive_dir.mkdir(parents=True, exist_ok=True)
        file_path = archive_dir / f"env_{prompt_hash}.json"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"[Memory] حفظ تصميم بيئي → {file_path}")
            return True
        except Exception as e:
            logger.exception(f"فشل حفظ تصميم بيئي → {file_path}")
            return False

    def load_environment_elements(self, prompt_hash: str) -> Optional[dict]:
        """
        استرجاع تصميم البيئة الخام
        """
        key = f"env_{prompt_hash}"

        # أولوية للذاكرة المؤقتة
        if hasattr(self, 'memory_store') and key in self.memory_store:
            logger.debug(f"[Memory] تحميل من الذاكرة المؤقتة: {key}")
            return self.memory_store[key]

        # ثم من الملف
        file_path = Path("layer_archive") / prompt_hash / f"env_{prompt_hash}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"[Memory] تم تحميل تصميم بيئي من {file_path}")
                return data
            except Exception as e:
                logger.exception(f"فشل قراءة {file_path}")
                return None

        logger.warning(f"لا يوجد تصميم بيئي محفوظ لـ {prompt_hash}")
        return None
    
    # ----------------------------------------------------------------------
    # دالة مساعدة صغيرة (يمكن وضعها في مكان مشترك أو هنا)
    # ----------------------------------------------------------------------
    def _extract_path_from_result(self, result: GenerationResult) -> Optional[Path]:
        """استخراج مسار الملف من GenerationResult بأمان"""
        if not result.output_data:
            return None
        for key in ["preview_path", "path", "output_path", "file_path", "layer_path"]:
            p = result.output_data.get(key)
            if p and os.path.exists(p):
                return Path(p)
        return None
