# Core_Image_Generation_Engine.py

from __future__ import annotations

print("→ بدأ Core_Image_Generation_Engine.py")

"""
النواة المشتركة لجميع محركات توليد الصور
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
from time import perf_counter
import matplotlib.pyplot as plt

# ─── الاستيراد الصحيح ───
from generation_result import GenerationResult

# إعدادات اللوقينج
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')
logger = logging.getLogger(__name__)

# باقي الكود يستمر من هنا (class CoreImageGenerationEngine: ... إلخ)

@dataclass
class PromptState:
    """تجميع كل حالة الـ prompt الحية في كائن واحد"""
    raw_accumulated: str = ""                   # ← بديل current_prompt
    completed_words: list[str] = field(default_factory=list)  # ← بديل tracked_words
    last_stable_hash: str = ""
    last_refresh_timestamp: float = 0.0
    input_chunks: list[str] = field(default_factory=list)      # ← بديل input_port

class CoreImageGenerationEngine(ABC):
    def __init__(self):
        self.specialization = self._get_specialization_config()
        self.tasks: List[Dict[str, Any]] = []
        self.dependencies: Dict[str, List[str]] = {}
        self.reverse_deps: Dict[str, List[str]] = defaultdict(list)
        self.input_port: List[str] = []
        self.render_time: float = 0.0
        self.stage_times: Dict[str, float] = {}
        self.prompt_state = PromptState()
        self.last_task_data: Optional[Dict] = None
        self.memory_manager = None  # ← هيبقى يتعيّن لاحقًا لو عايز
        self.last_render_time = 0.0
        self.last_render_success = False
        self.last_render_message = ""
        self.last_render_metadata = {}
        self.last_render_stage_times = {}
        self.last_render_specialization = self.specialization.get("name", "unknown")
        self.last_render_is_video = False
        self.last_render_mode = "unknown"
        self.render_history: List[GenerationResult] = []
        self.max_history = 20
        spec_name = self.specialization.get('name', 'محرك غير معروف') if isinstance(self.specialization, dict) else str(self.specialization)
        logger.info(f"تم تهيئة {spec_name} بنجاح")

    # ────────────────────────────────────────────────
    #              الدوال الأساسية المجردة
    # ────────────────────────────────────────────────

    @abstractmethod
    def _get_specialization_config(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _integrate(self, task_data: Dict) -> float:
        pass

    @abstractmethod
    def _post_process(self, task_data: Dict) -> Dict[str, Any]:
        """يجب أن ترجع dict يحتوي على نتائج المعالجة النهائية"""
        pass

    @abstractmethod
    def _render(self, task_data: Dict, is_video: bool = False) -> float:
        pass

    @abstractmethod
    def _create_simple_image(self, task_data: Dict, is_video: bool = False) -> Optional[str]:
        pass

    # ────────────────────────────────────────────────
    #              الدوال العامة المشتركة
    # ────────────────────────────────────────────────

    def _ensure_task_data_structure(self, task_data: Optional[Dict] = None) -> Dict:
        """تهيئة task_data بهيكل آمن إذا كان None أو ناقص"""
        if task_data is None:
            task_data = {}
        
        defaults = {
            "entities": [],
            "planes": [],
            "plane_interactions": [],
            "summary": {},
            "warnings": [],
            "metadata": {},
            "success": False,
            "duration_seconds": 0.0,
            "raw_prompt": ""
        }
        
        # نضيف القيم الافتراضية فقط للمفاتيح الغير موجودة
        for key, value in defaults.items():
            task_data.setdefault(key, value)
        
        return task_data

    def receive_input(self, input_data: str) -> bool:
        if not input_data or not isinstance(input_data, str):
            logger.error("الإدخال غير صالح أو فارغ")
            return False

        stripped = input_data.strip()
        if not stripped:
            logger.warning("الإدخال بعد التنظيف فارغ")
            return False

        # بدون أي validate هنا → نقبل كل شيء الآن
        self.input_port.append(stripped)
        logger.info(f"تم إضافة إدخال جديد ({len(self.input_port)} في المنفذ) → {stripped[:60]}...")
        return True

    def _run_generation(
        self,
        prompt: str,
        target_size: Tuple[int, int] = (1024, 1024),
        is_video: bool = False,
        as_layer: bool = True,
        force_refresh: bool = False,
        **kwargs
    ) -> GenerationResult:
        """
        التنفيذ الحقيقي – بدون أي استدعاء لـ generate_image أو receive_input
        """
        logger.info(f"[Core] _run_generation started with direct prompt: {prompt[:70]}...")

        start = perf_counter()
        stage_times = {}

        try:
            # 1. تحليل الـ prompt
            t_analyze = perf_counter()
            task_data = self._analyze_prompt(prompt)
            stage_times["analyze"] = perf_counter() - t_analyze

            # 2. التكامل
            t_integrate = perf_counter()
            task_data = self._integrate(task_data)
            stage_times["integrate"] = perf_counter() - t_integrate

            # 3. الـ rendering
            t_render = perf_counter()
            render_time, preview_path = self._render(task_data, is_video=is_video)
            stage_times["render"] = render_time   # ← لاحظ: هنا نستخدم render_time مباشرة (مش perf_counter - t_render)

            # 4. المعالجة النهائية
            t_post = perf_counter()
            post_result = self._post_process(task_data)
            stage_times["post_process"] = perf_counter() - t_post

            total_time = perf_counter() - start

            return GenerationResult(
                success=True,
                message="تم التوليد بنجاح",
                total_time=total_time,
                stage_times=stage_times,
                specialization=self.specialization.get("name", "unknown"),
                is_video=is_video,
                output_data={
                    "preview_path": preview_path,
                    "task_data": task_data
                }
            )

        except Exception as e:
            logger.exception("[Core] خطأ في _run_generation")
            return GenerationResult(
                success=False,
                message=str(e),
                total_time=perf_counter() - start,
                stage_times=stage_times,
                specialization=self.specialization.get("name", "unknown"),
                is_video=is_video,
                output_data={}
            )
            
    def add_task(self, task_name: str, complexity: float = 1.0, dependencies: Optional[List[str]] = None):
        """إضافة مهمة إلى التخصص الحالي"""
        if not task_name:
            logger.warning("اسم المهمة فارغ")
            return

        task = {"name": task_name, "complexity": max(0.0, float(complexity))}
        self.tasks.append(task)

        deps = dependencies or []
        self.dependencies[task_name] = deps

        for dep in deps:
            self.reverse_deps[dep].append(task_name)

        logger.debug(f"أضيفت مهمة: {task_name} (تعقيد {complexity})")

    def check_dependencies(self) -> bool:
        """التحقق من سلامة الاعتماديات"""
        all_names = {t["name"] for t in self.tasks}
        for task_name, deps in self.dependencies.items():
            missing = [d for d in deps if d not in all_names]
            if missing:
                logger.error(f"اعتماديات مفقودة لـ '{task_name}': {missing}")
                return False
        return True

    def refresh_stage(self, stage: str):
        """وضع علامة إعادة تنشيط على مرحلة معينة"""
        units = self.specialization.get("units", {})
        if stage in units and isinstance(units[stage], dict):
            units[stage]["refreshed"] = True
            logger.info(f"تم تنشيط إعادة التوليد للمرحلة: {stage}")
        else:
            logger.warning(f"المرحلة '{stage}' غير موجودة أو غير قابلة للتنشيط")

    def _topological_sort(self) -> List[str]:
        """ترتيب المهام حسب الاعتماديات (topological order)"""
        indegree = {t["name"]: len(self.dependencies.get(t["name"], [])) for t in self.tasks}
        queue = deque([name for name, deg in indegree.items() if deg == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.reverse_deps[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.tasks):
            logger.warning("يوجد دورة أو اعتماديات مفقودة → ترتيب غير كامل")

        return order

    def get_total_complexity(self) -> float:
        """مجموع تعقيد المهام"""
        return sum(t.get("complexity", 0) for t in self.tasks)

    def _evaluate_accuracy(self, task_data: Dict, prompt: str) -> bool:
        """تقييم تلقائي للدقة بناءً على الـ prompt و الـ task_data"""
        entities_count = len(task_data.get("entities", []))
        prompt_length = len(prompt)
        complexity = sum(t.get("complexity", 0) for t in self.tasks)
        
        # مثال بسيط: إذا الـ prompt طويل أو كثير entities أو تعقيد عالي → يحتاج refresh
        needs_refresh = prompt_length > 100 or entities_count > 3 or complexity > 10
        logger.info(f"تقييم الدقة: {'يحتاج refresh' if needs_refresh else 'دقيق كفاية'} "
                    f"(طول prompt: {prompt_length}, entities: {entities_count}, تعقيد: {complexity})")
        return needs_refresh

    def execute_stage(
        self,
        stage_name: str,
        task_data: Dict[str, Any],
        stage_times_local: Dict[str, float],
        allow_refresh: bool = False,
        max_refresh_attempts: int = 1
    ) -> tuple[float, bool, bool]:  # (duration, refresh_performed, fallback_used)
        """
        تنفيذ مرحلة واحدة مع دعم refresh
        ترجع tuple: (الوقت, هل تم refresh, هل تم fallback)
        """
        if stage_name not in ("integration", "post_processing"):
            logger.warning(f"execute_stage called with unexpected stage: {stage_name}")

        t_start = perf_counter()
        result = self._call_unit(stage_name, task_data)
        duration = perf_counter() - t_start

        fallback_used = False
        refresh_performed = False
        extra_time = 0.0

        if result is None:
            logger.warning(f"مرحلة {stage_name} أرجعت None → fallback")
            fallback_used = True
            duration = max(duration, 0.12)

            if stage_name == "integration":
                task_data.setdefault("planes", [])
                task_data.setdefault("plane_interactions", [])
                summary = task_data.setdefault("integration_summary", {})
                summary.update({"layers": 0, "interactions": 0, "fallback": True})
            elif stage_name == "post_processing":
                summary = task_data.setdefault("post_summary", {})
                summary.update({"processed": False, "fallback": True})

        # محاولة refresh إذا مسموح
        if allow_refresh and max_refresh_attempts > 0:
            logger.info(f"[Refresh] محاولة إعادة {stage_name}")
            t_ref = perf_counter()
            result_retry = self._call_unit(stage_name, task_data)
            extra_time = perf_counter() - t_ref

            if result_retry is not None:
                result = result_retry
                refresh_performed = True
                duration += extra_time
            else:
                logger.warning(f"المحاولة الثانية لـ {stage_name} فشلت أيضاً")

        stage_times_local[stage_name] = duration
        if fallback_used:
            stage_times_local[f"{stage_name}_fallback"] = True
        if refresh_performed:
            stage_times_local[f"{stage_name}_refresh"] = extra_time

        return duration, refresh_performed, fallback_used

    def _process_unified_stages(self, prompt: str, force_refresh: bool = False) -> Dict[str, Any]:
        start = perf_counter()
        stage_times = {}

        # ─── Fallback قاعدي آمن جدًا ───
        def safe_task_data() -> Dict[str, Any]:
            return {
                "entities": [],
                "style": "fallback_safe",
                "mood": "neutral",
                "main_subject": "unknown_scene",
                "raw_prompt": str(prompt)[:500],
                "planes": [],
                "plane_interactions": [],
                "integration_summary": {"layers": 0, "interactions": 0, "fallback": True},
                "post_summary": {"processed": False, "fallback": True},
                "error_occurred": True,
                "last_error": None
            }

        def safe_task_data(prompt: str) -> Dict:
            return {
                "raw_prompt": prompt,
                "entities": [{"text": w, "type": "unknown"} for w in prompt.split()[:5]],
                "planes": [{"type": "background", "description": "default fallback"}],
                "style": "realistic",
                "mood": "neutral",
                "fallback_mode": True,
                "error_occurred": True
            }
        task_data = safe_task_data()

        # ─── NLP ─────────────────────────────────────────────────────────────
        try:
            nlp_out = self._call_unit("nlp", prompt)
            if isinstance(nlp_out, dict):
                task_data.update(nlp_out)
                task_data.pop("error_occurred", None)
            else:
                task_data["last_error"] = "NLP did not return dict"
        except Exception as exc:
            task_data["last_error"] = f"NLP crashed: {type(exc).__name__}"
            logger.exception("NLP stage failed")

        stage_times["nlp"] = perf_counter() - start   # وقت تقريبي


        # ─── Integration ─────────────────────────────────────────────────────
        t_int_start = perf_counter()
        try:
            int_out = self._call_unit("integration", task_data)
            if isinstance(int_out, (int, float)):
                integration_time = float(int_out)
            else:
                integration_time = 0.4
                task_data["integration_fallback"] = True
        except Exception as exc:
            integration_time = 0.4
            task_data["integration_fallback"] = True
            task_data["last_error"] = f"Integration failed: {type(exc).__name__}"
            logger.exception("Integration stage failed")
        stage_times["integration"] = perf_counter() - t_int_start

        # ─── Post-processing ─────────────────────────────────────────────────
        t_post_start = perf_counter()
        post_result = None
        post_time = 0.3  # قيمة افتراضية آمنة

        try:
            raw_post = self._call_unit("post_processing", task_data)
            
            # استخدام الدالة الذكية للتوحيد
            post_result = self.engine._normalize_stage_result(
                stage_name="post_processing",
                raw_result=raw_post,
                default_duration=0.5
            )

            # الآن post_result مضمون أنه dict دائمًا
            task_data["post_summary"]   = post_result.get("summary", "No summary available")
            task_data["post_duration"]  = post_result.get("duration_seconds", post_time)
            task_data["post_success"]   = post_result.get("success", False)

            if post_result.get("warnings"):
                logger.warning(f"Post-processing warnings: {post_result['warnings']}")

            # استخدم الوقت الفعلي من النتيجة (إذا وجد)
            if "duration_seconds" in post_result:
                post_time = post_result["duration_seconds"]

        except Exception as exc:
            post_time = 0.3
            task_data["post_fallback"] = True
            task_data["last_error"] = f"Post-processing failed: {type(exc).__name__}"
            logger.exception("Post-processing stage failed")

        stage_times["post_processing"] = perf_counter() - t_post_start

        # ─── تجميع النتائج النهائية ───────────────────────────────────────
        task_data["unified_time"] = integration_time + post_time
        task_data["stage_times"] = stage_times
        task_data["total_unified_duration"] = perf_counter() - start
        task_data["refresh_decision"] = refresh_dec

        if task_data.get("error_occurred"):
            logger.warning("تم استخدام fallback كامل بسبب أخطاء في المراحل")

        return task_data

    def _detect_input_mode(self, debounce_seconds: float = 2.5) -> str:
        if not self.input_port:
            return "empty"
        
        full_text = " ".join(self.input_port)
        time_since_last = perf_counter() - self.prompt_state.last_refresh_timestamp
        
        if len(full_text) > 180 and time_since_last > debounce_seconds * 1.5:
            return "batch_stable"
        
        if time_since_last < debounce_seconds:
            return "streaming_active"
        
        return "batch"

    def _call_stage(self, spec, stage_name, *args, **kwargs):
        unit = spec["units"][stage_name]
        if unit["refreshed"]:
            logging.info(f"Re-executing {stage_name} (refreshed)")
            # يمكن هنا إعادة تعيين refreshed = False بعد التنفيذ إذا أردت
            # unit["refreshed"] = False
        return unit["function"](*args, **kwargs)

    def _call_unit(self, stage: str, *args, **kwargs):
        unit = self.specialization["units"].get(stage)
        if not unit or "function" not in unit:
            logger.error(f"[CallUnit] المرحلة '{stage}' بدون دالة")
            return None
        
        func = unit["function"]
        if not callable(func):
            logger.error(f"[CallUnit] الدالة لـ '{stage}' مش callable")
            return None
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"[CallUnit] {stage} رجعت نوع: {type(result)}")
            return result
        except Exception as e:
            logger.exception(f"[CallUnit] خطأ في تنفيذ {stage}")
            return None

    def _should_refresh_stage(self, stage: str, force_refresh: bool = False, task_data: Dict = None) -> bool:
        """
        قرار refresh مركزي لمرحلة واحدة — يجمع كل الآليات.
        
        Args:
            stage: اسم المرحلة
            force_refresh: إذا True → دائمًا refresh
            task_data: optional للـ heuristic أو check_improvement
        
        Returns:
            bool: True إذا يجب refresh
        """
        unit = self.specialization.get("units", {}).get(stage, {})
        if not unit:
            return False

        # 1. force_refresh (أولوية عالية)
        if force_refresh:
            return True

        # 2. refreshed flag (من units)
        if unit.get("refreshed", False):
            return True

        # 3. من memory manager (لو موجود)
        if hasattr(self, 'memory_manager') and self.memory_manager.should_refresh_stages(stage):
            return True

        # 4. check_improvement_needed
        if self.check_improvement_needed(stage):
            return True

        # 5. heuristic_score (لو موجود task_data)
        if task_data:
            score = self.heuristic_score(task_data.get("prompt_so_far", ""), task_data)
            if score < 0.65:
                return True

        return False

    def check_improvement_needed(self, stage: str) -> bool:
        """
        تقرر هل يستحق إعادة تنفيذ مرحلة معينة بناءً على حالة المهام الحالية.
        
        Returns:
            bool: True إذا كان من المفيد إعادة التنفيذ (يُفعّل refreshed تلقائياً)
        """
        if stage not in self.specialization["units"]:
            logger.warning(f"المرحلة {stage} غير موجودة في التخصص الحالي")
            return False

        tasks = self.tasks
        if not tasks:
            logger.debug(f"لا توجد مهام → لا حاجة لإعادة تنفيذ {stage}")
            return False

        # معايير بسيطة حالياً (يمكن توسيعها لاحقاً)
        task_count = len(tasks)
        total_complexity = sum(t.get("complexity", 0) for t in tasks)

        # 1. إذا كان عدد المهام كبير نسبياً
        if task_count >= 3:
            logger.debug(f"عدد المهام ({task_count}) كبير → يُفضل إعادة {stage}")
            return True

        # 2. إذا كان التعقيد الكلي مرتفع
        if total_complexity > 7.0:
            logger.debug(f"التعقيد الكلي ({total_complexity:.1f}) مرتفع → يُفضل إعادة {stage}")
            return True

        # 3. إذا كانت المرحلة integration أو post-processing وهناك مهام تعتمد على بعضها
        if stage in ("integration", "post_processing"):
            has_dependencies = any(self.dependencies.get(t["name"], []) for t in tasks)
            if has_dependencies:
                logger.debug(f"يوجد اعتماديات في المهام → يُفضل إعادة {stage}")
                return True

        # 4. حالة خاصة: إذا تم إضافة مهام جديدة مؤخراً (يمكن تتبعها بمتغير)
        # (اختياري – تحتاج إضافة عداد أو flag في الكلاس)
        # if getattr(self, "_tasks_modified_since_last_run", False):
        #     return True

        logger.debug(f"لا حاجة واضحة لإعادة تنفيذ {stage} (مهام: {task_count}, تعقيد: {total_complexity:.1f})")
        return False

    def _normalize_stage_result(
        self,
        stage_name: str,
        raw_result: Any,
        default_duration: float = 0.0
    ) -> Dict[str, Any]:
        """
        توحيد نتيجة أي مرحلة إلى dict آمن
        - يتعامل مع float (legacy)، dict، None، أو أي نوع آخر
        - يضيف معلومات fallback وlogging واضح
        """
        if isinstance(raw_result, dict):
            # الحالة المثالية → نرجعها كما هي مع ضمان وجود keys أساسية
            return {
                "success": raw_result.get("success", True),
                "duration_seconds": raw_result.get("duration_seconds", default_duration),
                "summary": raw_result.get("summary", f"{stage_name} completed"),
                "warnings": raw_result.get("warnings", []),
                "metadata": raw_result.get("metadata", {}),
                "normalized": False  # يعني أصلية dict
            }

        # حالة legacy: float أو int (اللي بيحصل دلوقتي)
        if isinstance(raw_result, (int, float)):
            logger.warning(
                f"Legacy return from {stage_name}: {type(raw_result).__name__} → "
                f"converted to dict (duration: {raw_result:.2f}s)"
            )
            return {
                "success": True,
                "duration_seconds": float(raw_result),
                "summary": f"Legacy {stage_name} result (duration: {raw_result:.2f}s)",
                "warnings": ["Legacy float/int return type - please update _post_process to return dict"],
                "metadata": {},
                "normalized": True,
                "legacy": True
            }

        # حالة None أو نوع غير متوقع
        if raw_result is None:
            msg = f"{stage_name} returned None"
        else:
            msg = f"{stage_name} returned unexpected type: {type(raw_result).__name__}"

        logger.warning(f"{msg} → fallback to empty dict")
        return {
            "success": False,
            "duration_seconds": default_duration,
            "summary": f"Fallback - {msg}",
            "warnings": [msg],
            "metadata": {},
            "normalized": True,
            "fallback": True
        }
    
    def _integrate(self, task_data: Dict) -> Dict:
        task_data.setdefault("planes", [])
        task_data.setdefault("summary", {})
        
        # مثال بسيط: طبقة لكل مهمة
        for task in self.tasks:
            plane = {
                "label": task["name"],
                "z": 0.5,
                "color": "#ffffff",
                "force": 0.6,
                "mass": 1.5
            }
            task_data["planes"].append(plane)
        
        task_data["summary"]["layers_count"] = len(task_data["planes"])
        return task_data

    # ────────────────────────────────────────────────
    #              دوال مساعدة عامة
    # ────────────────────────────────────────────────
    def reset(self):
        self.tasks.clear()
        self.dependencies.clear()
        self.reverse_deps.clear()
        self.prompt_state = PromptState()           # ← إعادة تهيئة كاملة
        self.render_time = 0.0
        self.stage_times.clear()
        logger.info("تم إعادة تعيين المحرك")

    def clear_input(self):
        old_len = len(self.prompt_state.input_chunks)
        self.prompt_state.input_chunks.clear()
        logger.info(f"تم مسح {old_len} وصف سابق")

    def get_status(self) -> Dict[str, Any]:
        """نظرة سريعة على الحالة الحالية"""
        return {
            "specialization": self.specialization.get("name", "غير محدد"),
            "tasks_count": len(self.tasks),
            "input_count": len(self.input_port),
            "last_render_time": self.render_time,
            "dependencies_ok": self.check_dependencies()
        }

    def get_total_complexity(self) -> float:
        """مجموع تعقيد المهام"""
        return sum(t.get("complexity", 0) for t in self.tasks)

    def _create_simple_image(self, task_data: Dict, is_video: bool = False) -> str | None:
        path = "output/latest.png"   # أو من الـ API
        if os.path.exists(path):
            return path
        # fallback إلى matplotlib كما عندك حالياً
        return self._save_simple_preview(task_data["raw_prompt"], 0.0, is_video)
        
    def _save_simple_preview(self, prompt: str, total_time: float, is_video: bool = False) -> str:
        """
        إنشاء صورة preview بسيطة كـ placeholder
        """
        try:
            import matplotlib.pyplot as plt
            from pathlib import Path
            from datetime import datetime

            spec_name = self.specialization.get("name", "unknown")
            suffix = "video" if is_video else "preview"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"{spec_name}_{suffix}_{timestamp}.png")

            fig, ax = plt.subplots(figsize=(10, 6), dpi=144)
            fig.patch.set_facecolor('#0a0a1a')
            fig.patch.set_edgecolor('gray')
            fig.patch.set_linewidth(1)

            ax.set_facecolor('#0a0a1a')
            ax.axis('off')

            # تخصيص حسب الـ specialization name
            color_main = 'white'
            title = f"{spec_name.replace('_', ' ').title()} Design"

            if "traditional" in spec_name.lower():
                title = "Traditional / Organic Design"
                color_main = 'wheat'
                ax.add_patch(plt.Circle((0.5, 0.65), 0.18, color='sienna', alpha=0.85))
                ax.add_patch(plt.Ellipse((0.5, 0.42), 0.8, 0.3, color='forestgreen', alpha=0.65))

            elif "geometric" in spec_name.lower():
                title = "Geometric / Structured Design"
                color_main = '#00eaff'
                ax.add_patch(plt.Circle((0.5, 0.5), 0.38, color='none', ec=color_main, lw=3.5))
                ax.add_patch(plt.Polygon([[0.38,0.68], [0.62,0.68], [0.5,0.32]], color=color_main, alpha=0.45))

            elif "Environment" in spec_name.lower():
                title = "Environment / Tech Design"
                color_main = '#ff00aa'
                ax.add_patch(plt.Rectangle((0.28, 0.28), 0.44, 0.44, color='none', ec=color_main, lw=4, ls='--'))
                ax.plot([0.2, 0.8], [0.72, 0.28], color='#00ffff', lw=2.5, alpha=0.75)

            ax.set_title(title, color=color_main, fontsize=16, pad=20)

            # النصوص
            ax.text(0.5, 0.90, f"Prompt: {prompt[:70]}{'...' if len(prompt)>70 else ''}",
                    ha='center', va='center', fontsize=11, color=color_main, wrap=True)

            ax.text(0.5, 0.15, f"Total time: {total_time:.1f} s   |   {'Video' if is_video else 'Image'}",
                    ha='center', va='center', fontsize=10, color='lightgray')

            ax.text(0.5, 0.06, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    ha='center', va='center', fontsize=9, color='dimgray')

            plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=144)
            plt.close(fig)

            logger.info(f"Preview saved → {output_path}")
            return str(output_path)

        except ImportError:
            logger.warning("matplotlib not installed → skipping preview")
            return "preview_skipped_no_matplotlib"
        except Exception as e:
            logger.error(f"Preview failed: {e}")
            return "preview_failed"

    def retry_layer_generation(
        self,
        generate_func: Callable,                    # self.generate_layer أو دالة مشابهة
        prompt: str,
        layer_name: str,
        max_attempts: int = 3,
        min_quality_score: float = 0.75,
        **generate_kwargs
    ) -> Optional[GenerationResult]:
        """
        إعادة محاولة توليد طبقة مع تدقيق جودة تلقائي
        مشتركة بين كل المحركات المتخصصة
        """
        attempt = 1
        best_result = None
        best_score = 0.0

        while attempt <= max_attempts:
            logger.info(f"[Retry] محاولة {attempt}/{max_attempts} لـ {layer_name}")

            # تعديلات تدريجية في الـ prompt أو الباراميترات
            current_prompt = prompt
            current_kwargs = generate_kwargs.copy()

            if attempt == 2:
                current_prompt = f"{prompt}, highly detailed, ultra sharp, better lighting, masterpiece"
            elif attempt == 3:
                current_kwargs["force_refresh"] = True
                size = current_kwargs.get("target_size", (1024, 1024))
                current_kwargs["target_size"] = (int(size[0] * 1.3), int(size[1] * 1.3))

            # التوليد
            try:
                result = generate_func(prompt=current_prompt, **current_kwargs)
            except Exception as e:
                logger.error(f"[Retry {attempt}] خطأ في التوليد: {e}")
                attempt += 1
                continue

            if not result.success:
                attempt += 1
                continue

            # تدقيق الجودة
            report = self.memory_manager.refresh_layer_before_and_after_storage(
                layer_result=result,
                layer_name=layer_name,
                stage="before",
                prompt_hash=self.memory_manager.get_prompt_hash(prompt)
            )

            # حساب score بسيط (يمكن تطويره)
            score = 1.0
            if report["issues"]:
                score -= len(report["issues"]) * 0.35
            if report["warnings"]:
                score -= len(report["warnings"]) * 0.15
            score = max(0.0, min(1.0, score))

            # حفظ كل محاولة كإصدار فرعي
            self.memory_manager.save_raw_layer(
                prompt_hash=self.memory_manager.get_prompt_hash(prompt),
                layer_name=layer_name,
                layer_result=result,
                extra_metadata={"retry_attempt": attempt, "score": score}
            )

            if score > best_score:
                best_result = result
                best_score = score

            if score >= min_quality_score:
                logger.info(f"[Retry] نجاح في المحاولة {attempt} – score {score:.2f}")
                return result

            attempt += 1

        logger.warning(f"[Retry] فشل بعد {max_attempts} محاولات – أفضل score: {best_score:.2f}")
        return best_result  # أفضل ما لدينا حتى لو تحت العتبة

# ─── subclass بسيط للاختبار فقط ───
class TestEngine(CoreImageGenerationEngine):
    def _get_specialization_config(self) -> Dict[str, Any]:
        return {
            "name": "test_design",
            "units": {
                "nlp":          {"function": self._analyze_prompt, "refreshed": False},
                "integration":  {"function": self._integrate,      "refreshed": False},
                "post_processing": {"function": self._post_process, "refreshed": False},
                "rendering":    {"function": self._render,         "refreshed": False},
            }
        }

    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        logger.info(f"[Test] تحليل: {prompt}")
        return {
            "raw_prompt": prompt,
            "entities": ["pattern", "spiral"],
            "style": "test",
            "warnings": []
        }

    def _integrate(self, task_data: Dict) -> Dict:
        logger.info("[Test] integrate dummy – doing nothing")
        # نرجّع نفس الداتا مع إضافة حاجة بسيطة عشان نعرف إننا عدينا المرحلة
        task_data["integrated"] = True
        task_data["integration_time"] = 0.12
        return task_data

    def _post_process(self, task_data: Dict) -> Dict:
        logger.info("[Test] post-process dummy")
        return {
            "success": True,
            "duration_seconds": 0.18,
            "summary": "Post-processing dummy completed",
            "warnings": [],
            "metadata": {
                "processed_entities": len(task_data.get("entities", []))
            }
        }

    def _render(self, task_data: Dict, is_video: bool = False) -> Tuple[float, Optional[str]]:
        logger.info("[Test] rendering dummy – no real API call")
        start = perf_counter()

        # نعمل مسار وهمي عشان النتيجة تبان كأنها نجحت
        preview_path = f"test_render_{int(start)}.png"

        # وقت وهمي واقعي شوية
        duration = 0.35 + (0.2 if is_video else 0.0)

        logger.info(f"[Test] تم إنشاء معاينة وهمية: {preview_path}")
        return duration, preview_path

    # لو عندك _create_simple_image موجودة، ممكن تخليها كده مؤقتاً
    def _create_simple_image(self, task_data: Dict, is_video: bool = False) -> Optional[str]:
        return "dummy_preview_from_test.png"
    
if __name__ == "__main__":
    print("=== بدء اختبار بسيط للمحرك ===")
    
    try:
        engine = TestEngine()
        print("تم إنشاء TestEngine بنجاح")
        
        engine.receive_input("geometric spiral pattern golden ratio")
        print("تم إضافة prompt")
        
        result = engine._run_generation(
            prompt=" ".join(engine.input_port),
            target_size=(512, 512),
            is_video=False,
            force_refresh=True
        )
        
        print("\nالنتيجة:")
        print("نجاح     :", result.success)
        print("رسالة    :", result.message)
        print("الوقت    :", f"{result.total_time:.2f} ث")
        
        if result.success and "preview_path" in result.output_data:
            print("المعاينة :", result.output_data["preview_path"])
        else:
            print("ما فيش معاينة محفوظة")
            
    except AttributeError as e:
        print("خطأ: دالة ناقصة أو غير معرفة →", e)
    except Exception as e:
        print("خطأ عام أثناء التشغيل:", str(e))
    
    print("\n=== انتهى الاختبار ===")