# unified_stage_pipeline.py
"""
نظام إدارة المراحل مع تتبع وصف دقيق أثناء الكتابة + check & refresh ذكي
- المراحل الثلاث (NLP, Integration, Post-processing) تحت إشراف موحد
- Rendering منفصل تمامًا
- تتبع الكتابة الحية + refresh جزئي تلقائي (مع debounce + cache)
"""

import re
import hashlib
import logging
from typing import Dict, Any, Optional, List
from time import perf_counter
from pathlib import Path
from typing import NamedTuple

from Agent_helper import AITab3, AIHelper

# استيراد المحرك هنا – في أعلى الملف

logger = logging.getLogger(__name__)

# ─── صغيرة ومحددة ────────────────────────────────────────────────
class ContextWindowState(NamedTuple):
    text: str
    meaningful_words_count: int
    hash: str
    changed: bool

# ────────────────────────────────────────────────────────────────
#                 Unified Stage Pipeline
# ────────────────────────────────────────────────────────────────
class UnifiedStagePipeline:
    """
    نظام إدارة المراحل الموحدة مع دعم:
    - الكتابة التدريجية (streaming / live typing)
    - debounce و auto-refresh ذكي
    - caching للمراحل
    - حدود آمنة للذاكرة
    """

    def __init__(self, engine: Any = None):
        self.engine = engine
        self.logger = logging.getLogger("UnifiedStagePipeline")

        if self.engine is None:
            self.logger.warning("تم تمرير engine=None → وضع fallback مفعّل")

        # ─── حالة الـ prompt الحالية ───────────────────────────────────────
        self.current_prompt: str = ""
        self.tracked_words: List[str] = []
        self.completed_words: List[str] = []

        # ─── حدود وقائية (مهمة جداً للـ real-time) ──────────────────────────
        self.MAX_CHARS = 2500           # ~400–600 كلمة تقريباً
        self.MAX_WORDS = 450
        self.MIN_WORDS_FOR_CHECK = 3
        self.AUTO_REFRESH_TRIGGER_WORDS = 5

        # ─── توقيت و debounce ────────────────────────────────────────────────
        self.last_input_time: float = 0.0
        self.debounce_delay: float = 1.8          # ثواني
        self.last_processed_hash: str = ""

        # ─── كاش وتتبع المراحل ──────────────────────────────────────────────
        self.stage_cache: Dict[str, Dict[str, Any]] = {}
        self.stage_times: Dict[str, float] = {}
        self.last_refresh_stage: Optional[str] = None
        self.last_refresh_reason: Optional[str] = None

        # ─── المكونات الفرعية ───────────────────────────────────────────────
        self.helper = AIHelper(self)
        self.tab3 = AITab3(self.helper)

        # ─── ربط المحركات الافتراضية (مؤقتًا نربط geometric بس) ──────────────────────
        logger.info("قبل أي attach")
        if self.engine is not None:
            logger.info("قبل geometric")
            self.tab3.attach_engine("geometric", self.engine)
            logger.info("بعد geometric")

            logger.info("قبل traditional – لو وصل هنا يبقى المشكلة لسة موجودة")
            # self.tab3.attach_engine("traditional", self.engine)
            logger.info("بعد traditional")
    
    def on_char(self, char: str) -> Optional[str]:
        if not char:
            return None

        self.append_to_prompt(char)
        self._enforce_input_limits()

        if not self._is_end_of_meaningful_token(char):
            return None

        if not self.should_process_now():
            return None

        window = self.context_tracker.compute(self.current_prompt)
        if window.meaningful_words_count < self.MIN_WORDS_FOR_CHECK:
            return None

        if not window.changed:
            return None

        # هنا نعمل refresh جزئي
        partial_text = window.text
        self.logger.info("[Auto] محاولة معاينة على: %s...", partial_text[:60])

        try:
            # افترض إن generate_composite موجودة في الـ engine أو tab3
            result = self.engine.generate_composite(partial_text, is_video=False)
            preview_path = result.get("preview_path") if result else None
            if preview_path:
                return preview_path
        except Exception as e:
            self.logger.error("فشل auto-composite: %s", e)

        return None

    def _debounce_satisfied(self) -> bool:
        now = perf_counter()
        return (now - self.last_input_time) >= self.debounce_delay

    def _enforce_input_limits(self) -> None:
        if len(self.current_prompt) > self.MAX_CHARS:
            self.current_prompt = self.current_prompt[-self.MAX_CHARS:]
            self.logger.debug("قُصّ النص إلى %d حرف", self.MAX_CHARS)

        words = self.current_prompt.split()
        if len(words) > self.MAX_WORDS:
            self.current_prompt = " ".join(words[-self.MAX_WORDS:])
            self.logger.debug("قُصّ النص إلى %d كلمة", self.MAX_WORDS)

    def _is_end_of_meaningful_token(self, char: str) -> bool:
        """نعالج فقط عند انتهاء كلمة أو علامة ترقيم مهمة"""
        if char.isspace():
            return True
        if char in '.,!?;:\n\r\t"»«–—()[]{}':
            return True
        return False

    def _attempt_auto_composite_refresh(self, partial_text: str) -> Optional[str]:
        if len(partial_text.split()) < 6:
            self.logger.debug("سياق قصير جدًا للـ composite")
            return None

        self.logger.info("[Auto] محاولة توليد معاينة → %s...", partial_text[:60])

        try:
            result = self.engine.generate_composite(
                prompt=partial_text,
                resolution=(768, 768),
                force_refresh=True,
                is_video=False,
                auto_split=True,
                sequential_mode=True
            )

            if not result or not result.get("success", False):
                return None

            preview_path = result.output_data.get("preview_path") or result.output_data.get("output_path")
            if preview_path and Path(preview_path).is_file():
                self.logger.info("→ معاينة جديدة: %s", preview_path)
                return preview_path

        except Exception as exc:
            self.logger.exception("فشل الـ auto composite refresh")

        return None

    def _merge_result_into_task(self, task: dict, stage: str, result: dict) -> None:
        """
        دمج نتيجة مرحلة واحدة داخل الـ task الرئيسي
        
        ملاحظة: يُعدّل الـ task مباشرة (in-place)
        """
        if not result:
            task["warnings"].append(f"{stage}: لا توجد نتيجة للدمج")
            return

        # حفظ النتيجة الخام لكل مرحلة (مفيد للـ debug والـ retry)
        task[f"{stage}_result"] = result.copy()

        if not result.get("success", False):
            error_msg = result.get("error", f"فشل {stage} بدون تفاصيل")
            task["errors"].append(error_msg)
            task["warnings"].append(f"{stage}: تم استخدام fallback جزئي")
            task["success"] = False
            return

        # ─── دمج حسب نوع المرحلة ───────────────────────────────────────
        if stage == "nlp":
            # نأخذ الكيانات والـ mood والـ style إلخ
            task["entities"] = result.get("entities", task.get("entities", []))
            task["main_subject"] = result.get("main_subject", "unknown")
            task["mood"] = result.get("mood", task.get("mood", "neutral"))
            task["style"] = result.get("style", task.get("style", "default"))

        elif stage == "integration":
            # الطبقات والتفاعلات بينها
            task["planes"] = result.get("planes", task.get("planes", []))
            task["plane_interactions"] = result.get(
                "plane_interactions", task.get("plane_interactions", [])
            )

        elif stage == "post_processing":
            # الملخص النهائي + أي تعديلات على الجودة
            post_summary = result.get("post_summary", {})
            task["post_summary"] = post_summary
            if post_summary.get("processed", False):
                task["final_prompt"] = post_summary.get("final_prompt", raw_prompt)

        # إحصائيات إضافية (اختياري)
        if "duration_seconds" in result:
            task.setdefault("stage_durations", {})[stage] = result["duration_seconds"]

        # علامة نجاح جزئي
        task.setdefault("processed_stages", set()).add(stage)
    
    def _safe_call_unit(self, unit: str, data: Any = None) -> Dict[str, Any]:
        if not self.engine or not hasattr(self.engine, '_call_unit'):
            logger.warning("no _call_unit in engine → fallback")
            return {"fallback": True, "unit": unit, "data": data or {}}
        
        if self.engine is None:
            logger.error(f"لا يوجد engine متاح → لا يمكن تنفيذ '{unit}'")
            return {
                "success": False,
                "error": "No engine instance available",
                "fallback": True,
                "stage": unit,
                "returned": None
            }

        # 1. التحقق من وجود الدالة
        if not hasattr(self.engine, "_call_unit"):
            logger.error(f"المحرك لا يحتوي على الدالة '_call_unit' → لا يمكن تنفيذ '{unit}'")
            return {
                "success": False,
                "error": "Engine missing _call_unit method",
                "fallback": True,
                "stage": unit,
                "missing_method": "_call_unit"
            }

        call_method = getattr(self.engine, "_call_unit")

        # 2. التحقق من أنها قابلة للاستدعاء
        if not callable(call_method):
            logger.error(f"'_call_unit' موجود لكنه غير قابل للاستدعاء (نوع: {type(call_method).__name__})")
            return {
                "success": False,
                "error": "_call_unit attribute is not callable",
                "fallback": True,
                "stage": unit,
                "invalid_type": str(type(call_method))
            }

        # 3. محاولة التنفيذ الفعلي
        try:
            result = call_method(unit, data)

            # 4. معالجة أنواع النتائج المختلفة (legacy & modern)
            if result is None:
                logger.warning(f"الوحدة '{unit}' أرجعت None → معاملة كفشل نسبي")
                return {
                    "success": False,
                    "warning": "Unit returned None",
                    "stage": unit,
                    "returned": None
                }

            if isinstance(result, (int, float)):
                # دعم legacy engines اللي كانت ترجع وقت التنفيذ فقط
                logger.info(f"Legacy mode: '{unit}' أرجع عدد (يُعامل كنجاح): {result}")
                return {
                    "success": True,
                    "duration_seconds": float(result),
                    "legacy": True,
                    "stage": unit,
                    "raw_result": result
                }

            if not isinstance(result, dict):
                logger.warning(
                    f"الوحدة '{unit}' أرجعت نوع غير متوقع: {type(result).__name__} → "
                    f"تحويل إلى dict افتراضي"
                )
                return {
                    "success": True,  # نعتبرها نجاح نسبي عشان ما نقطعش السلسلة
                    "stage": unit,
                    "raw_result": result,
                    "warning": f"Unexpected return type: {type(result).__name__}",
                    "normalized": True
                }

            # النتيجة dict → مثالية
            # نضيف بعض metadata إضافي لو مفيد
            result.setdefault("success", True)          # default لو ما حددهاش
            result.setdefault("stage", unit)
            result.setdefault("processed_at", perf_counter())

            return result

        except TypeError as te:
            logger.error(f"خطأ في تمرير الباراميترات لـ '{unit}': {te}")
            return {
                "success": False,
                "error": f"TypeError in call signature: {str(te)}",
                "stage": unit,
                "fallback": True
            }

        except Exception as e:
            logger.exception(f"فشل تنفيذ '{unit}': {e}")
            return {
                "success": False,
                "error": str(e),
                "exc_type": type(e).__name__,
                "stage": unit,
                "fallback": True
            }
        
    def should_process_now(self) -> bool:
        """
        هل حان الوقت لمعالجة الـ prompt الحالي؟ (debounce logic)
        """
        if not self.current_prompt.strip():
            return False

        now = perf_counter()
        elapsed = now - self.last_input_time

        if elapsed < self.debounce_delay:
            self.logger.debug("debounce → ما زال الكتابة مستمرة (%.2f ث)", elapsed)
            return False

        # مقارنة hash لمعرفة إذا تغير شيء فعلياً
        current_hash = hashlib.md5(self.current_prompt.encode('utf-8')).hexdigest()
        if current_hash == self.last_processed_hash:
            self.logger.debug("لا تغيير في الـ prompt → لا حاجة للمعالجة")
            return False

        self.last_processed_hash = current_hash
        return True

    def append_to_prompt(self, char_or_text: str) -> None:
        """
        إضافة حرف أو نص جزئي إلى الـ prompt الحالي مع تطبيق الحدود الآمنة
        """
        if not char_or_text:
            return

        self.current_prompt += char_or_text
        self.last_input_time = perf_counter()

        # 1. تطبيق حد الحروف (الأسرع)
        if len(self.current_prompt) > self.MAX_CHARS:
            self.current_prompt = self.current_prompt[-self.MAX_CHARS:]
            self.logger.debug("تم قص current_prompt إلى %d حرف (حد أقصى)", self.MAX_CHARS)

        # 2. تطبيق حد الكلمات (أثقل قليلاً لكن أكثر دقة)
        words = self.current_prompt.split()
        if len(words) > self.MAX_WORDS:
            self.current_prompt = " ".join(words[-self.MAX_WORDS:])
            self.logger.debug("تم قص current_prompt إلى %d كلمة (حد أقصى)", self.MAX_WORDS)

        # 3. تحديث tracked_words (اختياري – يمكن إزالته إذا لم يُستخدم)
        # self.tracked_words = words

    def _get_refresh_key(self, stage: str, prompt: str) -> str:
        """مفتاح كاش موحد"""
        return f"{stage}:{hashlib.md5(prompt.encode('utf-8')).hexdigest()}"

    # ─── إدارة المراحل (الكاش + التنفيذ) ──────────────────────────────
    
    def _get_cache_key(self, stage: str, payload: dict) -> str:
        """مفتاح كاش مستقر نسبيًا"""
        # يمكن تحسينه لاحقًا (مثلاً: تجاهل بعض الحقول غير المهمة في payload)
        payload_str = str(sorted(payload.items()))   # ترتيب لضمان الثبات
        hash_val = hashlib.md5(payload_str.encode('utf-8')).hexdigest()
        return f"{stage}:{hash_val[:16]}"

    def _execute_stage(self, stage: str, payload: dict) -> dict:
        """
        الطبقة الوحيدة التي تتواصل مباشرة مع الـ engine.
        تقوم بـ:
        - التحقق من وجود engine
        - تنفيذ _call_unit
        - تطبيع النتيجة (legacy float → dict, None → fallback, إلخ)
        - التعامل مع الاستثناءات
        """
        if not self.engine or not hasattr(self.engine, '_call_unit'):
            self.logger.warning(f"لا يوجد engine صالح لتنفيذ {stage}")
            return self._minimal_fallback_for_stage(stage)

        try:
            raw_result = self.engine._call_unit(stage, payload)

            # تطبيع النتيجة حسب النوع
            if raw_result is None:
                self.logger.warning(f"{stage} أرجع None → fallback")
                return {
                    "success": False,
                    "stage": stage,
                    "fallback": True,
                    **self._minimal_fallback_for_stage(stage)
                }

            if isinstance(raw_result, (int, float)):
                # legacy mode: كان يرجع الوقت فقط
                self.logger.info(f"Legacy mode: {stage} أرجع عدد → تحويل")
                return {
                    "success": True,
                    "duration_seconds": float(raw_result),
                    "stage": stage,
                    "legacy": True,
                    "raw": raw_result
                }

            if not isinstance(raw_result, dict):
                self.logger.warning(f"{stage} أرجع نوع غير متوقع: {type(raw_result).__name__}")
                return {
                    "success": True,  # نعاملها كناجحة نسبيًا حتى لا نقطع السلسلة
                    "stage": stage,
                    "raw": raw_result,
                    "warning": f"Unexpected return type: {type(raw_result).__name__}",
                    "normalized": True
                }

            # حالة مثالية: dict جيد
            result = raw_result.copy()
            result.setdefault("success", True)
            result.setdefault("stage", stage)
            result.setdefault("processed_at", perf_counter())
            return result

        except Exception as exc:
            self.logger.exception(f"فشل تنفيذ {stage}")
            return {
                "success": False,
                "stage": stage,
                "error": str(exc),
                "exc_type": type(exc).__name__,
                "fallback": True,
                **self._minimal_fallback_for_stage(stage)
            }

    def _get_or_compute_stage(self, stage: str, payload: dict, *, force: bool = False) -> dict:
        """
        الدالة المركزية الوحيدة لمعالجة أي مرحلة:
        - كاش أولاً (إذا موجود وغير مطلوب force)
        - تنفيذ المرحلة لو مفيش كاش أو force_refresh
        - حفظ في الكاش لو نجحت
        - fallback ذكي لو فشلت
        """
        if not isinstance(payload, dict):
            self.logger.error(f"payload لـ {stage} ليس dict → fallback")
            return self._minimal_fallback_for_stage(stage)

        # 1. مفتاح الكاش (مستقر وسريع)
        cache_key = self._get_cache_key(stage, payload)

        # 2. استرجاع من الكاش (أسرع طريق)
        if not force and cache_key in self.stage_cache:
            self.logger.debug(f"cache hit → {stage} (key: {cache_key[:12]}...)")
            return self.stage_cache[cache_key].copy()

        # 3. تنفيذ فعلي (لو مفيش كاش أو force_refresh)
        self.logger.info(f"تنفيذ جديد لـ {stage} (force={force})")
        
        try:
            # استدعاء المرحلة الحقيقية حسب الاسم
            if stage == "nlp":
                result = self._analyze_prompt(payload.get("prompt", ""))
            elif stage == "integration":
                result = self._integrate(payload)
            elif stage == "post_processing":
                result = self._post_process(payload)
            else:
                result = {"success": False, "error": f"مرحلة غير معروفة: {stage}"}

            # 4. لو نجحت نسبيًا → احفظ في الكاش
            if result.get("success", False) and not result.get("fallback", False):
                self.stage_cache[cache_key] = result.copy()
                self.logger.debug(f"تم حفظ في الكاش → {stage}")
            else:
                self.logger.debug(f"لم يُحفظ في الكاش (فشل/legacy/fallback) → {stage}")

            return result

        except Exception as e:
            self.logger.exception(f"فشل تنفيذ {stage}")
            fallback_result = self._minimal_fallback_for_stage(stage)
            fallback_result["error"] = str(e)
            return fallback_result

    def _minimal_fallback_for_stage(self, stage: str) -> Dict[str, Any]:
        """قيم افتراضية آمنة جدًا لكل مرحلة لو فشلت"""
        fallbacks = {
            "nlp": {
                "entities": [],
                "main_subject": "unknown",
                "mood": "neutral",
                "style": "default",
                "fallback": True
            },
            "integration": {
                "planes": [{"type": "background", "desc": "default"}],
                "plane_interactions": [],
                "fallback": True
            },
            "post_processing": {
                "post_summary": {"processed": False, "note": "fallback"},
                "fallback": True
            }
        }
        return fallbacks.get(stage, {"fallback": True, "note": "no fallback defined"})

    # ─── المعالجة الكاملة (غير الحية) ─────────────────────────────────

    def process(self, prompt: str, force_refresh: bool = False) -> Dict[str, Any]:
        if not prompt or not prompt.strip():
            self.logger.warning("محاولة معالجة prompt فارغ → fallback")
            return self._create_fallback_result("prompt فارغ")

        start_total = perf_counter()
        self.stage_times.clear()

        task_data: Dict[str, Any] = {
            "raw_prompt": prompt.strip(),
            "entities": [],
            "planes": [],
            "style": "default",
            "mood": "neutral",
            "fallback": False,
            "errors": [],
            "warnings": [],
            "success": False,
        }

        stages = ["nlp", "integration", "post_processing"]

        for stage in stages:
            # الدالة المركزية الجديدة
            stage_result = self._get_or_compute_stage(
                stage=stage,
                payload={"prompt": prompt, "task_data": task_data},
                force=force_refresh
            )

            if not stage_result.get("success", False):
                error_msg = stage_result.get("error", f"فشل {stage}")
                task_data["errors"].append(error_msg)
                task_data["fallback"] = True
                task_data["success"] = False
                task_data.update(stage_result.get("data", {}))
                
                if stage == "nlp":  # مثال لمرحلة حرجة
                    self.logger.error(f"فشل nlp → توقف الكل")
                    break
            else:
                task_data.update(stage_result.get("data", {}))

        task_data["success"] = len(task_data["errors"]) == 0
        task_data["unified_time"] = perf_counter() - start_total

        if task_data["success"]:
            self.logger.info("اكتملت بنجاح (وقت: %.2f ث)", task_data["unified_time"])
        else:
            self.logger.warning("اكتملت مع %d أخطاء", len(task_data["errors"]))

        return task_data

    def _prepare_payload(self, stage: str, task: dict) -> dict:
        """
        إعداد البيانات التي ستُمرر لكل مرحلة
        
        Returns:
            dict جاهز للتمرير إلى _get_or_compute_stage
        """
        raw_prompt = task.get("raw_prompt", "").strip()

        if not raw_prompt:
            self.logger.warning("raw_prompt فارغ أثناء تحضير payload")
            return {"raw_prompt": ""}

        base_payload = {
            "raw_prompt": raw_prompt,
            "timestamp": perf_counter(),           # اختياري – للـ debugging
        }

        if stage == "nlp":
            # مرحلة تحليل النص الأولية
            return {
                **base_payload,
                "focus": "entity_extraction",      # يمكن أن تكون ديناميكية لاحقًا
                "max_entities": 8,
                "language": "ar_en_mixed",         # أو اكتشاف تلقائي
            }

        elif stage == "integration":
            # دمج الكيانات مع الطبقات / البيئة
            entities = task.get("entities", [])
            return {
                **base_payload,
                "entities": entities,
                "previous_nlp": task.get("nlp_result", {}),
                "mode": "sequential",              # أو "parallel" حسب الإعداد
                "resolution_hint": (768, 768),
            }

        elif stage == "post_processing":
            # تحسين النهاية (جودة، weighting، إلخ)
            integration_data = task.get("integration_result", {})
            return {
                **base_payload,
                "planes": integration_data.get("planes", []),
                "plane_interactions": integration_data.get("plane_interactions", []),
                "quality_boost": True,
                "remove_redundancy": True,
            }

        else:
            self.logger.warning(f"مرحلة غير معروفة: {stage} → payload افتراضي")
            return base_payload
    
    def _merge_result_into_task(self, task: dict, stage: str, result: dict) -> None:
        """
        دمج نتيجة مرحلة واحدة داخل الـ task الرئيسي
        
        ملاحظة: يُعدّل الـ task مباشرة (in-place)
        """
        if not result:
            task["warnings"].append(f"{stage}: لا توجد نتيجة للدمج")
            return

        # حفظ النتيجة الخام لكل مرحلة (مفيد للـ debug والـ retry)
        task[f"{stage}_result"] = result.copy()

        if not result.get("success", False):
            error_msg = result.get("error", f"فشل {stage} بدون تفاصيل")
            task["errors"].append(error_msg)
            task["warnings"].append(f"{stage}: تم استخدام fallback جزئي")
            task["success"] = False
            return

        # ─── دمج حسب نوع المرحلة ───────────────────────────────────────
        if stage == "nlp":
            # نأخذ الكيانات والـ mood والـ style إلخ
            task["entities"] = result.get("entities", task.get("entities", []))
            task["main_subject"] = result.get("main_subject", "unknown")
            task["mood"] = result.get("mood", task.get("mood", "neutral"))
            task["style"] = result.get("style", task.get("style", "default"))

        elif stage == "integration":
            # الطبقات والتفاعلات بينها
            task["planes"] = result.get("planes", task.get("planes", []))
            task["plane_interactions"] = result.get(
                "plane_interactions", task.get("plane_interactions", [])
            )

        elif stage == "post_processing":
            # الملخص النهائي + أي تعديلات على الجودة
            post_summary = result.get("post_summary", {})
            task["post_summary"] = post_summary
            if post_summary.get("processed", False):
                task["final_prompt"] = post_summary.get("final_prompt", raw_prompt)

        # إحصائيات إضافية (اختياري)
        if "duration_seconds" in result:
            task.setdefault("stage_durations", {})[stage] = result["duration_seconds"]

        # علامة نجاح جزئي
        task.setdefault("processed_stages", set()).add(stage)
    
    # ─── Rendering (منفصل تمامًا تحت إشراف AI مستقل) ────────────────────

    def render(self, task_data: Dict, is_video: bool = False) -> float:
        t = perf_counter()
        try:
            self.engine._create_simple_image(task_data, is_video)
        except Exception as e:
            logger.error(f"خطأ في الـ render: {e}")
        return perf_counter() - t

    # ─── Events / Notifications ─────────────────────────────────────────

    def notify(self, event: str, data: Any = None):
        """
        نقطة دخول مركزية للإشعارات من AIHelper / AITab3 وغيرهم
        """
        if data is None:
            data = {}

        # تسجيل موحد
        if isinstance(data, dict):
            keys_str = f"keys={list(data.keys())}"
        else:
            keys_str = f"type={type(data).__name__}"
        self.logger.info(f"notify ← {event} | {keys_str}")

        # فحص مبكر (اختياري لكن مفيد)
        if not isinstance(data, dict):
            self.logger.warning("data ليس dict → تجاهل المعالجة")
            return

        # جدول المعالجات
        event_handlers = {
            "analyzed_external": self._handle_analyzed_external,
            "preview_ready":     self._handle_preview_ready,
            # أضف لاحقًا: "error_occurred": self._handle_error, إلخ
        }

        handler = event_handlers.get(event)
        if handler:
            try:
                handler(data)
            except Exception as exc:
                self.logger.error(
                    "خطأ داخل معالج الحدث %r: %s",
                    event, exc, exc_info=True
                )
        else:
            self.logger.debug("حدث غير مدعوم: %s", event)

    def _handle_analyzed_external(self, data: dict):
        if data.get("needs_refresh", False):
            try:
                engine_name = data.get("engine", "traditional")  # مرونة إضافية
                self.tab3.refresh_engine(engine_name)
                self.logger.info("تم طلب refresh_engine(%r)", engine_name)
            except AttributeError:
                self.logger.warning("tab3.refresh_engine غير موجود")
            except Exception as e:
                self.logger.error("فشل refresh_engine: %s", e)

        if "suggestions" in data:
            sugs = data["suggestions"]
            if isinstance(sugs, (list, tuple)) and sugs:
                self.logger.info("اقتراحات وصلت (%d): %s", len(sugs), sugs[:3])
            else:
                self.logger.debug("suggestions موجود لكن فارغ أو غير قائمة")

    def _handle_preview_ready(self, data: dict):
        path = data.get("preview_path")
        if path:
            self.last_preview_path = path
            self.logger.info("معاينة جديدة وصلت: %s", path)
            # هنا ممكن تضيف: إشعار UI، حفظ في cache، إلخ
        else:
            self.logger.warning("preview_ready بدون preview_path")
        
    def _minimal_fallback_for_stage(self, stage: str) -> dict:
        """(نفس الدالة القديمة أو نسخة مبسطة)"""
        fallbacks = {
            "nlp": {"entities": [], "main_subject": "unknown", "fallback": True},
            "integration": {"planes": [], "fallback": True},
            "post_processing": {"processed": False, "fallback": True},
        }
        return fallbacks.get(stage, {"fallback": True, "note": "no specific fallback"})

    # ─── تتبع الوصف الدقيقة أثناء الكتابة ──────────────────────────────

    def _should_trigger_auto_refresh(self) -> bool:
        """
        تقرر هل الوقت مناسب لتشغيل الـ auto-refresh أم لا
        (debounce + تغيير في المحتوى + عدد كلمات كافٍ + ...)
        """
        now = perf_counter()

        # 1. Debounce: هل مر وقت كافٍ من آخر معالجة؟
        if now - self.last_input_time < self.debounce_delay:
            return False

        # 2. هل وصلنا للحد الأدنى من الكلمات؟
        if len(self.completed_words) < self.min_words_for_check:
            return False

        # 3. (اختياري) هل تغير المحتوى فعلياً من آخر معالجة؟
        #    لو كنت تحتفظ بنسخة سابقة من النص أو hash، قارن هنا
        #    (في الكود الحالي نعتمد على last_processed_hash في on_char)

        # 4. شروط إضافية ممكنة مستقبلاً:
        #    - هل زاد عدد الكلمات بما يكفي؟
        #    - هل ظهرت كلمة مفتاحية مهمة؟
        #    - هل مر وقت طويل من آخر refresh حتى لو ما تغير شيء؟

        return True

    def _quick_quality_check(self, text: str) -> Dict[str, Any]:
        """فحص سريع نسبي — يركز على التغيير والاكتمال"""
        words = text.split()
        word_count = len(words)

        if word_count < self.min_words_for_check:
            return {
                "needs_refresh": True,
                "recommended_stage": "nlp",
                "reason": f"كلمات قليلة ({word_count} < {self.min_words_for_check})"
            }

        lower_text = text.lower()

        # ─── كيانات أساسية (موضوع رئيسي) ───────────────────────────────
        subject_keywords = [
            "حصان", "فتاة", "تنين", "غابة", "مدينة", "وحش", "قلعة", "بحر",
            "سماء", "جبل", "سيارة", "روبوت", "فضاء", "قمر", "شمس"
        ]
        has_subject = any(kw in lower_text for kw in subject_keywords)

        # ─── تفاصيل / جودة ────────────────────────────────────────────────
        quality_keywords = [
            "detailed", "highly detailed", "intricate", "cinematic", "ultra", "4k", "8k",
            "masterpiece", "best quality", "sharp focus", "beautiful", "epic", "dramatic"
        ]
        quality_count = sum(1 for kw in quality_keywords if kw in lower_text)

        # ─── كلمات هندسية / نمطية (مهمة للـ integration) ─────────────────
        geometry_keywords = ["spiral", "hex", "golden", "pattern", "fractal", "symmetry", "grid"]
        has_geometry = any(kw in lower_text for kw in geometry_keywords)

        # ─── قرارات متدرجة ─────────────────────────────────────────────────
        if not has_subject:
            return {
                "needs_refresh": True,
                "recommended_stage": "nlp",
                "reason": "لا يوجد موضوع/كيان واضح بعد"
            }

        if word_count >= 7 and quality_count == 0:
            return {
                "needs_refresh": True,
                "recommended_stage": "post_processing",
                "reason": f"كلمات كثيرة ({word_count}) بدون أي وصف جودة"
            }

        if has_geometry and word_count >= 5:
            return {
                "needs_refresh": True,
                "recommended_stage": "integration",
                "reason": "ظهرت كلمات هندسية/نمطية → يفضل تحديث integration"
            }

        # إضافة بسيطة: لو النص طويل جدًا بدون تقدم واضح
        if word_count > 15 and quality_count < 2:
            return {
                "needs_refresh": True,
                "recommended_stage": "post_processing",
                "reason": "prompt طويل بدون تفاصيل كافية"
            }

        return {
            "needs_refresh": False,
            "recommended_stage": None,
            "reason": f"مقبول (كيان موجود، جودة: {quality_count} hit(s))"
        }

    def process_delta(self, previous, delta_prompt, full_prompt, force_full=False):
        if force_full or not previous:
            return self.process(full_prompt, force_refresh=force_full)
        # وإلا نسخة بسيطة جدًا
        return previous.copy()

    # ─── المراحل الموحدة (تحت إشراف AI واحد) ──────────────────────────

    def _is_unified_complete(self, task_data: Dict) -> bool:
        return (
            len(task_data.get("entities", [])) >= 2 and
            len(task_data.get("planes", [])) >= 3 and
            task_data.get("post_summary", {}).get("processed", False)
        )
        
    def _get_stage_cache_key(self, stage: str, prompt: str) -> str:
        """إنشاء مفتاح كاش موحد ومحمي"""
        # يمكن تحسينه لاحقاً بإضافة version أو specialization
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        return f"{stage}:{prompt_hash[:16]}"  # نأخذ جزءاً فقط لتقليل الطول

    def _create_fallback_result(self, reason: str) -> Dict[str, Any]:
        """إنشاء نتيجة fallback بسيطة وآمنة عند الفشل المبكر"""
        return {
            "raw_prompt": "",
            "entities": [],
            "planes": [],
            "style": "fallback",
            "mood": "neutral",
            "fallback": True,
            "errors": [reason],
            "unified": False,
            "unified_time": 0.0,
            "total_stages_duration": 0.0
        }

    def _build_current_context_window(self) -> dict:   # ← غيّر النوع هنا
        all_words = [w.strip() for w in self.current_prompt.split() if w.strip()]
        
        trivial = {"the", "and", "or", "in", "on", "at", "to", "of", "a", "an", "is", "it",
                "في", "على", "من", "إلى", "و", "أو", "مع", "ب", "ل", "ك", "هو", "هي"}
        
        meaningful = [w for w in all_words if w.lower().strip(".,!?;:()[]{}") not in trivial]
        window_size = min(12, len(meaningful))
        recent_meaningful = meaningful[-window_size:]
        
        recent_original = all_words[-max(15, window_size * 2):]
        window_text = " ".join(recent_original)
        
        return {
            "words": recent_meaningful,
            "text": window_text
        }

    def _has_significant_change(self, current_text: str) -> bool:
        current_hash = hashlib.md5(current_text.encode('utf-8')).hexdigest()
        if current_hash == self.last_processed_hash:
            return False
        return True

    def _update_last_processed_state(self, text: str) -> None:
        self.last_processed_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        self.last_input_time = perf_counter()
       
    def _analyze_prompt(self, prompt: str) -> dict:
        """
        تحليل نصي بسيط (placeholder لحد ما نضيف LLM أو regex حقيقي)
        """
        self.logger.info(f"[NLP] تحليل prompt: {prompt[:60]}...")
        
        lower = prompt.lower()
        
        # استخراج كيانات بسيطة (أسماء، أماكن، صفات)
        entities = re.findall(r'\b([A-Z][a-z]+|[a-z]{4,})\b', prompt)
        mood_keywords = re.findall(r'\b(happy|sad|dark|mysterious|bright|glowing|peaceful|chaotic)\b', lower)
        
        result = {
            "success": True,
            "entities": list(set(entities)),
            "mood": mood_keywords[0] if mood_keywords else "neutral",
            "style": "realistic" if "real" in lower or "photo" in lower else "artistic",
            "confidence": 0.8,
            "data": {
                "raw_length": len(prompt),
                "entities_count": len(entities)
            }
        }
        
        if not entities:
            result["success"] = False
            result["error"] = "لم يتم استخراج كيانات"
        
        return result
    
    def _integrate(self, task_data: dict) -> dict:
        """دمج بسيط (placeholder)"""
        self.logger.info("[Integration] دمج dummy")
        
        return {
            "success": True,
            "planes": [],  # هنا هتكون الطبقات لاحقًا
            "data": {"integrated": True}
        }

    def _post_process(self, task_data: dict) -> dict:
        """معالجة نهائية بسيطة (placeholder)"""
        self.logger.info("[Post-processing] معالجة dummy")
        
        return {
            "success": True,
            "processed": True,
            "data": {"post_processed": True}
        }
       
# ────────────────────────────────────────────────────────────────
#                   Refresh ومعالجة المراحل
# ────────────────────────────────────────────────────────────────
class ContextWindow:
    def __init__(self):
        self._last_hash = None
        self._trivial = frozenset({
            "the", "and", "or", "in", "on", "at", "to", "of", "a", "an", "is", "it",
            "في", "على", "من", "إلى", "و", "أو", "مع", "ب", "ل", "ك", "هو", "هي"
        })

    def compute(self, full_text: str, max_meaningful=12) -> ContextWindowState:
        all_words = [w.strip() for w in full_text.split() if w.strip()]
        meaningful = [w for w in all_words if w.lower().strip(".,!?;:()[]{}") not in self._trivial]

        window_size = min(max_meaningful, len(meaningful))
        recent_meaningful = meaningful[-window_size:]
        window_text = " ".join(all_words[-max(20, window_size * 2):])

        current_hash = hashlib.md5(window_text.encode('utf-8')).hexdigest()
        changed = current_hash != self._last_hash
        self._last_hash = current_hash

        return ContextWindowState(
            text=window_text,
            meaningful_words_count=len(recent_meaningful),
            hash=current_hash,
            changed=changed
        )
        
# ──────────────────────────────────────────────
#  قوائم الاختبار (تعريف خارجي لتسهيل الصيانة)
# ──────────────────────────────────────────────

BASIC_TEST_PROMPTS = [
    "مراقبة هذا الـ prompt",
    "غابة سحرية هادئة",
    "تنين فقط",
]

EXTRA_TEST_PROMPTS = [
    "تنين أسود يطير فوق جبال مغطاة بالثلج تحت ضوء القمر",
    "فتاة ساحرة مع شعر أزرق طويل في غابة مضيئة بفطريات سحرية",
    "حصان مجنح أبيض يركض في سهل ذهبي عند الغروب",
    "وحش بحري عملاق يخرج من بحيرة ضبابية",
    "شجرة الحياة العملاقة في وسط صحراء مليئة بالنجوم",
    "مشهد غروب شمس برتقالي مع طيور مهاجرة",
    "فقط بحر هادئ بدون أي كائن",
]


# ────────────────────────────────────────────────
#              اختبار سريع للتحميل (اختياري)
# ────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = None
    helper = None
    tab3 = None

    try:
        from Final_Generation import CompositeEngine
        engine = CompositeEngine()
        pipeline = UnifiedStagePipeline(engine)
        print("✓ تم إنشاء UnifiedStagePipeline")

        # محاولة الوصول إلى tab3 بطرق مختلفة
        if hasattr(pipeline, 'helper') and hasattr(pipeline.helper, 'tab3'):
            tab3 = pipeline.helper.tab3
            print("✓ تم العثور على tab3 عبر pipeline.helper.tab3")
        elif hasattr(pipeline, 'tab3'):
            tab3 = pipeline.tab3
            print("✓ تم العثور على tab3 مباشرة في pipeline")
        else:
            print("⚠️ لم يتم العثور على tab3 داخل pipeline")

    except Exception as e:
        print("× فشل إنشاء pipeline:", type(e).__name__, str(e))

    # ──────────────────────── اختبار monitor_engine ────────────────────────
    print("\n" + "─" * 70)
    print("اختبار monitor_engine".center(70))
    print("─" * 70 + "\n")

    prompts_to_test = [
        "تنين أسود يطير فوق جبال مغطاة بالثلج تحت ضوء القمر",
        "فتاة ساحرة مع شعر أزرق طويل في غابة مضيئة بفطريات سحرية",
        "فقط بحر هادئ بدون أي كائن",
    ]

    monitor_func = None

    if tab3 is not None and hasattr(tab3, 'monitor_engine'):
        monitor_func = tab3.monitor_engine
        print("→ سيتم استخدام tab3.monitor_engine")
    elif hasattr(pipeline, 'helper') and hasattr(pipeline.helper, 'monitor_engine'):
        monitor_func = pipeline.helper.monitor_engine
        print("→ سيتم استخدام pipeline.helper.monitor_engine")
    else:
        print("× لا يوجد monitor_engine متاح في هذا السياق")

    if monitor_func:
        for prompt in prompts_to_test:
            print(f"\n→ {prompt}")
            try:
                result = monitor_func("traditional", prompt)
                if isinstance(result, dict):
                    print(f"  ✓ نجح – ثقة: {result.get('confidence', 'غير معروف'):.2f}")
                    if 'suggestions' in result and result['suggestions']:
                        print("  اقتراحات:")
                        for i, s in enumerate(result['suggestions'], 1):
                            print(f"    {i}. {s}")
                else:
                    print("  → رجع نوع غير متوقع:", type(result))
            except Exception as e:
                print("  × خطأ:", type(e).__name__, str(e))
            print("─" * 65)
    else:
        print("لا يمكن تشغيل الاختبار – لا توجد دالة monitor_engine متاحة")

    print("\n" + "═" * 70)
    print("الاختبار انتهى".center(70))
    print("═" * 70)