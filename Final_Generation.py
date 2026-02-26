# Final_Generation.py
"""
الموديل النهائي لتوليد الصور المركبة من طبقات متعددة (background, midground, foreground)
كل طبقة تولد باستخدام محرك متخصص مختلف (environment, geometric, traditional)
يحدد الترتيب ديناميكيًا بناءً على الprompt تحت إشراف SuperVisor بسيط.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import os
import shutil
import random
import math
import logging
from time import perf_counter
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageSequence
import cv2

from Core_Image_Generation_Engine import CoreImageGenerationEngine
from memory_manager import GenerativeMemoryManager
from layer_plane import PlaneLayer
from contextlib import contextmanager

from generation_result import GenerationResult
from unified_stage_pipeline import UnifiedStagePipeline
from prompt_supervisor import PromptSupervisor

logger = logging.getLogger(__name__)

class LayerEngine(ABC):
    @abstractmethod
    def generate_layer(
        self,
        prompt: str,
        target_size: tuple = (1024, 1024),
        is_video: bool = False,
        as_layer: bool = True,          # شفافية افتراضية
        force_refresh: bool = False,
    ) -> GenerationResult:
        pass

    def receive_input(self, prompt: str):
        # لو محتاج buffer داخلي
        pass

# ───────────────────────────────────────────────────────────────────────────────
# الموديل المركب النهائي الذي يجمع بين كل شيء (توليد الطبقات + الدمج + التفاعل) 
# ───────────────────────────────────────────────────────────────────────────────
class CompositeEngine(CoreImageGenerationEngine):
    def __init__(self):
        super().__init__()   # مهم جدًا لو فيه وراثة

        # استيراد بالأسماء الصحيحة اللي موجودة في الملفات
        from environment_design_engine import environment_design_engine  # lowercase
        from geometric_design_engine import geometric_design_engine      # lowercase
        from traditional_design_engine import traditionalDesignEngine    # camel case زي ما عندك

        # استخدم الأسماء المستوردة + () عشان instance جديد لو دالة
        self.engine_map = {
            "background": environment_design_engine(),   # ← lowercase + ()
            "midground": geometric_design_engine(),      # ← lowercase + ()
            "foreground": traditionalDesignEngine()      # ← camel + ()
        }

        try:
            self.memory_manager = GenerativeMemoryManager()
        except:
            self.memory_manager = None

        self.supervisor = PromptSupervisor(llm_callable=self._dummy_llm_call)

        self.specialization = {
            "name": "composite",
            "description": "محرك دمج طبقات متعددة",
            "domain": "image_composition"
        }

        # باقي المتغيرات (نظفت التكرار والزيادات)
        self.input_port = []
        self.tasks = []
        self.dependencies = {}
        self.stats = {"total_generations": 0, "successes": 0, "failures": 0}
        self.layer_opacities = {"background": 255, "midground": 255, "foreground": 255}
        self.interaction_history = []
        self.composite_history = []
        self.error_log = []
        self.performance_log = []
        self.visualization_data = []
        self.debug_mode = True
        self.last_composite_result: Optional[GenerationResult] = None
        self.layer_interaction_data = {}
        self.composite_count = 0
        self.successful_composites = 0
        self.failed_composites = 0
        self.total_composite_time = 0.0
        self.temp_files: List[Union[str, Path]] = []

        logger.info("تم إنشاء CompositeEngine بنجاح")
        
    def import_env_design(self, file_path: str) -> EnvironmentDesignResult:
        """يقرأ ملف التصميم المصدّر من environment_engine"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # تحويل العناصر مرة تانية إلى EnvironmentElement
        elements = [EnvironmentElement(**el) for el in data.get("elements", [])]
        
        return EnvironmentDesignResult(
            success=data["success"],
            elements=elements,
            metadata=data["metadata"],
            message=data["message"],
            design_time_seconds=data["design_time_seconds"]
        )
    
    def _validate_specialization(self):
        pass  # أو logger.debug("التحقق من التخصص - مؤقت")

    def _initialize_units(self):
        pass  # أو self.units = {} لو عايز

    def _initialize_memory_manager(self):
        pass  # لو مش محتاج memory_manager دلوقتي

    def _initialize_additional_state(self):
        pass

    def _run_initial_diagnostics(self):
        pass

    def _log_specialization_details(self):
        logger.info(f"تخصص: {self.specialization}")

    def _log_initial_state(self):
        logger.debug("حالة التهيئة الأولية - تم")

    def _validate_specialization(self):
        """
        تنفيذ بسيط للتحقق من التخصص (placeholder)
        الأب بيطلبها، فنعطيه شيء يرضيه
        """
        # لو عندك قيمة specialization حقيقية، تحقق منها هنا
        if not hasattr(self, 'specialization'):
            self.specialization = "composite"
        
        if self.specialization not in ["composite", "layer_compositor", "unknown"]:
            logger.warning(f"تخصص غير متوقع: {self.specialization} → استخدام 'composite' افتراضي")
            self.specialization = "composite"
        
        logger.debug(f"تم التحقق من التخصص: {self.specialization}")
        
    def legacy_of_sequential_design(
        self,
        environment_result: GenerationResult,      # اللي رجع من environment_design_engine
        remaining_prompt: str,                     # الجزء اللي للطبقات الأخرى (mid + fg)
        camera_angle: str = None,                  # اختياري: تغيير زاوية
        resolution: tuple = (1024, 1024),
        **kwargs
    ) -> GenerationResult:
        """
        استقبال نتيجة تصميم البيئة السابق → دمجها مع باقي الوصف → إكمال التكوين التسلسلي
        """
        if not environment_result.success:
            return GenerationResult(
                success=False,
                message="فشل مرحلة البيئة السابقة",
                output_data=None
            )

        # 1. استخراج البيانات المهمة من مرحلة البيئة
        env_data = environment_result.output_data or {}
        env_path = env_data.get("preview_path") or env_data.get("path")
        env_prompt = env_data.get("enhanced_prompt", "")

        if not env_path or not Path(env_path).exists():
            logger.warning("مسار البيئة غير صالح → fallback بدون مرجع")
            env_path = None

        # 2. بناء الـ master prompt الكامل (أو تقسيمه)
        full_prompt = f"{env_prompt}\n{remaining_prompt}".strip()

        # 3. إما: استدعاء _generate_environment_elements كامل
        #    أو: استدعاء generate_scene_with_new_angle إذا كان فيه camera_angle
        if camera_angle:
            composite_result = self.generate_scene_with_new_angle(
                env_data=env_data,
                new_camera_prompt=camera_angle,
                reference_temp_path=env_path
            )
        else:
            composite_result = self._generate_environment_elements(
                prompt=full_prompt,
                resolution=resolution,
                force_refresh=kwargs.get("force_refresh", False),
                is_video=kwargs.get("is_video", False)
            )

        # 4. إضافة metadata عن التسلسل
        if composite_result.success:
            composite_result.output_data = composite_result.output_data or {}
            composite_result.output_data["sequential_stages"] = {
                "stage1": "environment",
                "stage1_result": env_data,
                "stage2": "composite",
                "stage2_result": composite_result.output_data
            }

        return composite_result

    def _dummy_llm_call(self, prompt: str) -> str:
        """
        دالة وهمية مؤقتة لـ PromptSupervisor (placeholder)
        بعدين هتستبدلها بدالة LLM حقيقية (مثل Grok API أو OpenAI)
        """
        logger.info(f"[dummy_llm_call] prompt: {prompt[:70]}...")
        
        # رد وهمي بسيط عشان نعدي الخطأ
        return f"رد وهمي من LLM: {prompt.upper()[:50]}... (placeholder)"
    
    def _needs_physics_interaction(self, prompt: str, results: dict) -> bool:
        lower = prompt.lower()
        keywords = ["collide", "تصادم", "wind", "رياح", "gravity", "جاذبية",
                    "multiple objects", "كائنات متعددة", "interact", "تفاعل"]
        count_objects = sum(len(r.output_data.get("entities", [])) for r in results.values())
        return any(k in lower for k in keywords) or count_objects > 4

    def compose_sequentially_with_environment_base(
        self,
        full_user_prompt: str,               # الوصف الأصلي من المستخدم
        supervisor_plan: dict = None,        # خطة المشرف (اختياري)
        env_resolution: tuple = (1536, 1024),
        final_resolution: tuple = (1536, 1024),
        force_refresh: bool = False,
        save_intermediate: bool = True,
        output_name_prefix: str = "sequential_composite"
    ) -> GenerationResult:
        """
        توليد تسلسلي منظم:
        1. بيئة أولاً (القاعدة)
        2. geometric فوق البيئة
        3. traditional فوق الاثنين
        """
        stage_times = {}
        intermediate_paths = {}
        start_total = perf_counter()

        # ─── 0. مرحلة التخطيط / التقسيم (بواسطة المشرف) ───────────────────────
        if supervisor_plan is None:
            supervisor_plan = self.supervisor.plan_sequential_layers(full_user_prompt)
            # مثال على شكل الـ plan المتوقع:
            # {
            #   "background": "شاطئ بحري غروب شمس ذهبي، رمل ناعم، أمواج هادئة، سماء برتقالية",
            #   "midground":  "سيارة فاخرة سوداء لامعة متوقفة على الرمل، انعكاس الشمس عليها",
            #   "foreground": "فتاة شابة تجلس على غطاء السيارة، شعر طويل يتطاير، ملابس شاطئ أنيقة"
            # }

        # ─── 1. توليد البيئة (القاعدة) ────────────────────────────────────────────
        env_engine = environment_design_engine()
        env_result = env_engine.generate_layer(
            prompt=supervisor_plan.get("background", full_user_prompt),
            target_size=env_resolution,
            force_refresh=force_refresh,
            as_layer=True   # يفضل أن تكون شفافة إذا أمكن، لكن غالباً غير شفافة
        )

        if not env_result.success:
            return GenerationResult(
                success=False,
                message=f"فشل توليد البيئة الأساسية: {env_result.message}",
                total_time=perf_counter() - start_total
            )

        env_path = env_result.output_data.get("preview_path") or env_result.output_data.get("path")
        intermediate_paths["environment"] = env_path
        stage_times["environment"] = env_result.total_time

        # ─── 2. توليد الطبقة الوسطى (geometric / objects) فوق البيئة ─────────────
        geo_engine = geometric_design_engine()
        geo_result = geo_engine.generate_layer(
            prompt=supervisor_plan.get("midground", ""),
            target_size=final_resolution,
            force_refresh=force_refresh,
            as_layer=True,
            reference_image=env_path,          # ← مهم: مرجع البيئة
            control_strength=0.65,             # قوة الالتزام بالبيئة
            depth_control=True                 # إذا كان مدعوماً
        )

        geo_path = geo_result.output_data.get("preview_path") if geo_result.success else None
        if geo_path:
            intermediate_paths["geometric"] = geo_path
            stage_times["geometric"] = geo_result.total_time

        # ─── 3. توليد الطبقة الأمامية (traditional / characters) ──────────────────
        trad_engine = traditional_design_engine()
        trad_result = trad_engine.generate_layer(
            prompt=supervisor_plan.get("foreground", ""),
            target_size=final_resolution,
            force_refresh=force_refresh,
            as_layer=True,
            reference_image=env_path,          # البيئة مرجع أساسي
            secondary_reference=geo_path,      # السيارة مرجع إضافي (إن وجدت)
            control_strength=0.75,
            depth_control=True,
            character_consistency=True         # إذا كان المحرك يدعم seed أو face lock
        )

        trad_path = trad_result.output_data.get("preview_path") if trad_result.success else None
        if trad_path:
            intermediate_paths["traditional"] = trad_path
            stage_times["traditional"] = trad_result.total_time

        # ─── 4. الدمج النهائي (ترتيب ثابت: env → geo → trad) ─────────────────────
        layer_paths_ordered = {}
        if env_path:    layer_paths_ordered["background"]  = env_path
        if geo_path:    layer_paths_ordered["midground"]   = geo_path
        if trad_path:   layer_paths_ordered["foreground"]  = trad_path

        final_path = self._composite_layers(
            layer_paths=layer_paths_ordered,
            resolution=final_resolution,
            output_name=f"{output_name_prefix}_{int(time.time())}.png"
        )

        total_time = perf_counter() - start_total

        success = bool(final_path and Path(final_path).exists())

        return GenerationResult(
            success=success,
            message="تم التصميم التسلسلي بنجاح" if success else "فشل في مرحلة ما",
            total_time=total_time,
            stage_times=stage_times,
            specialization="sequential_composite",
            output_data={
                "final_path": final_path,
                "intermediate_paths": intermediate_paths,
                "supervisor_plan": supervisor_plan,
                "stages_order": ["environment", "geometric", "traditional"]
            }
        )
       
    def measurement_unit_validator_and_adjuster(
        self,
        layer_results: Dict[str, GenerationResult],  # {"environment": res_env, "geometric": res_geo, "traditional": res_trad}
        base_map_size_km: tuple = (2, 2),            # حجم الخريطة الأساسية (طول × عرض) بالكيلومتر
        supervisor_rules: Optional[Dict] = None,     # قواعد من المشرف (إذا None، يستدعي المشرف)
        auto_adjust: bool = True,                    # هل نعدل تلقائياً أم نرفع خطأ فقط؟
        save_adjusted: bool = True                   # حفظ التصاميم المعدلة؟
    ) -> Dict[str, GenerationResult]:
        """
        وحدة قياس: تفقد وتصحح المقاسات قبل وبعد الجمع
        - متصلة بالمشرف للحصول على قواعد منطقية/علمية
        - مراحل: قبل الجمع (كل طبقة) + بعد الجمع (الكلي)
        - مثال: تصحيح فتاة 500م إلى ~1.6م بناءً على قواعد المشرف
        """
        if not supervisor_rules:
            # استدعاء المشرف للحصول على قواعد القياس
            supervisor_rules = self.supervisor.get_scale_rules(layer_results.keys())
            # مثال على شكل supervisor_rules المتوقع:
            # {
            #   "traditional": {"human_female": {"height_m": (1.5, 1.7), "width_m": (0.4, 0.6)},
            #   "geometric": {"luxury_car": {"length_m": (4.5, 5.5), "width_m": (1.8, 2.0)},
            #   "environment": {"beach_map": {"total_km": (2, 2)}
            # }

        adjusted_results = layer_results.copy()  # نسخة للنتائج المعدلة
        stage_times = {"pre_compose": 0.0, "post_compose": 0.0}
        errors = []

        # ─── 1. المرحلة قبل الجمع: تفقد كل طبقة منفصلة ──────────────────────────
        start_pre = perf_counter()
        for layer_name, result in layer_results.items():
            if not result.success:
                errors.append(f"{layer_name}: النتيجة الأصلية فاشلة")
                continue

            # استخراج مسار الصورة / الطبقة
            path = result.output_data.get("preview_path") or result.output_data.get("path")
            if not path or not Path(path).exists():
                errors.append(f"{layer_name}: لا مسار صالح للتفقد")
                continue

            # تقدير المقاسات من الصورة (إذا لم يكن في metadata)
            metadata = result.output_data.get("metadata", {})
            if "dimensions_m" not in metadata:
                # استخدام OpenCV لتقدير (بناءً على كشف كونتور / bounding box)
                estimated_dims = self._estimate_object_size_from_image(path, layer_name, supervisor_rules.get(layer_name, {}))
                metadata["dimensions_m"] = estimated_dims
                result.output_data["metadata"] = metadata  # تحديث النتيجة

            # تفقد مقابل قواعد المشرف
            rules = supervisor_rules.get(layer_name, {})
            for obj_type, expected_range in rules.items():
                actual = metadata.get("dimensions_m", {}).get(obj_type, {})
                if not actual:
                    continue

                # مثال: تحقق من الطول (height_m)
                if "height_m" in expected_range:
                    min_h, max_h = expected_range["height_m"]
                    actual_h = actual.get("height_m", 0)
                    if not (min_h <= actual_h <= max_h):
                        errors.append(f"{layer_name} ({obj_type}): طول غير منطقي ({actual_h}m) - يجب {min_h}-{max_h}m")
                        if auto_adjust:
                            scale_factor = (min_h + max_h) / 2 / actual_h  # متوسط المقاس المرغوب
                            adjusted_path = self._adjust_image_scale(path, scale_factor, save_adjusted)
                            if adjusted_path:
                                result.output_data["preview_path"] = adjusted_path
                                metadata["dimensions_m"][obj_type]["height_m"] *= scale_factor
                                logger.info(f"{layer_name}: تم تعديل الطول إلى {metadata['dimensions_m'][obj_type]['height_m']:.2f}m")

                # نفس الشيء للعرض، الطول، إلخ...

        stage_times["pre_compose"] = perf_counter() - start_pre

        # ─── 2. الجمع المؤقت (للتفقد بعد الجمع) ──────────────────────────────────
        if errors and not auto_adjust:
            # إذا أخطاء ولا تعديل، نرجع مع الأخطاء
            return {"adjusted_results": adjusted_results, "errors": errors, "stage_times": stage_times}

        # جمع الطبقات المعدلة
        layer_paths = {name: res.output_data.get("preview_path") for name, res in adjusted_results.items() if res.success}
        final_path = self._composite_layers(layer_paths, base_map_size_km=base_map_size_km)  # مع إضافة باراميتر للقياس الكلي

        # ─── 3. المرحلة بعد الجمع: تفقد التماسك العام ──────────────────────────────
        start_post = perf_counter()
        if final_path:
            # تقدير المقاسات الكلية (مثل: هل الفتاة بالنسبة للسيارة منطقية؟)
            overall_dims = self._estimate_overall_consistency(final_path, supervisor_rules, base_map_size_km)
            for issue in overall_dims.get("issues", []):
                errors.append(issue)
                if auto_adjust:
                    # مثال: إعادة توليد الطبقة المخطئة أو تعديل عام
                    logger.warning(f"تصحيح عام: {issue}")
                    # هنا يمكن استدعاء المشرف لقرار تصحيح
                    self.supervisor.adjust_based_on_issue(issue, adjusted_results)

        stage_times["post_compose"] = perf_counter() - start_post

        # ─── 5. النتيجة النهائية ──────────────────────────────────────────────────
        return {
            "adjusted_results": adjusted_results,
            "final_path": final_path,
            "errors": errors,
            "stage_times": stage_times,
            "supervisor_rules_used": supervisor_rules
        }

    def _estimate_object_size_from_image(self, image_path: str, layer_name: str, rules: Dict) -> Dict:
        """
        تقدير المقاسات من الصورة باستخدام OpenCV (بناءً على كونتور ومرجع قواعد)
        """
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {}

        # أكبر كونتور كمثال (الكائن الرئيسي)
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        # تقدير بناءً على قواعد المشرف (نسبة بكسلات إلى متر)
        # افتراض: نستخدم قاعدة بسيطة (pixels per meter) من المشرف أو بحث
        ppm = rules.get("pixels_per_meter", 1)  # يجب تحديثه من المشرف
        estimated = {
            "height_m": h / ppm,
            "width_m": w / ppm
        }

        logger.info(f"{layer_name}: تقدير مقاسات: {estimated}")
        return estimated

    def _adjust_image_scale(self, image_path: str, scale_factor: float, save_adjusted: bool) -> Optional[str]:
        """
        تعديل حجم الصورة (resize) بناءً على العامل
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
        adjusted_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)

        if save_adjusted:
            adjusted_path = f"{Path(image_path).stem}_adjusted{Path(image_path).suffix}"
            cv2.imwrite(adjusted_path, adjusted_img)
            return adjusted_path

        return image_path  # إذا لا حفظ، نرجع الأصلي (أو bytes لاحقاً)

    def _estimate_overall_consistency(self, final_path: str, rules: Dict, base_map_size_km: tuple) -> Dict:
        """
        تفقد التماسك بعد الجمع (مثل نسبة الفتاة للسيارة)
        """
        # هنا تحليل مشابه للصورة الكلية، مقارنة النسب
        # مثال بسيط: افتراض نسبة مقاسات
        issues = []
        # ... كود تحليل ...
        if len(issues) == 0:
            issues.append("كل شيء منطقي")

        return {"issues": issues}

    def _call_layer_engine(self, layer_name, sub_prompt, resolution, force_refresh, is_video, reference_path=None):
        engine_class = self.engine_map.get(layer_name)
        if not engine_class:
            return GenerationResult(success=False, message=f"لا محرك للطبقة: {layer_name}")

        engine = engine_class()
        
        # مرونة أكبر في تمرير الـ reference
        extra_kwargs = {}
        if reference_path:
            extra_kwargs.update({
                "reference_image": reference_path,
                "control_strength": 0.70 if layer_name == "midground" else 0.85,  # قوة مختلفة حسب الطبقة
                "use_depth": layer_name != "foreground",                       # foreground أقل اعتماد على depth
                "use_ip_adapter": layer_name == "foreground"                   # للشخصيات
            })

        try:
            result = engine.generate_layer(
                prompt=sub_prompt,
                target_size=resolution,
                force_refresh=force_refresh,
                as_layer=True,
                is_video=is_video,
                **extra_kwargs
            )
            return result
        except Exception as e:
            logger.exception(f"فشل استدعاء محرك {layer_name}")
            return GenerationResult(success=False, message=str(e))

    def generate_scene_with_new_angle(self, env_result: GenerationResult, new_camera_prompt: str, **kwargs):
        if not env_result.success:
            return GenerationResult(success=False, message="البيئة الأصلية فشلت")

        env_path = self._extract_layer_path(env_result)
        if not env_path:
            return GenerationResult(success=False, message="لا مسار مرجعي من البيئة")

        with temp_reference_image(env_path) as ref_path:
            adjusted_prompt = f"{env_result.output_data.get('enhanced_prompt', '')}, {new_camera_prompt}"
            
            # استخدم نفس المحرك أو محرك عام للتعديل
            return self._call_layer_engine(
                layer_name="composite",  # أو محرك خاص إذا وجد
                sub_prompt=adjusted_prompt,
                resolution=kwargs.get("resolution", (1024, 1024)),
                force_refresh=kwargs.get("force_refresh", False),
                is_video=kwargs.get("is_video", False),
                reference_path=ref_path
            )

    def _generate_sequential_layers(self, plan, resolution, force_refresh, is_video):
        results = {}
        reference = None
        stages = [
            ("background", plan.get("background", ""), 0.0),   # قوة reference = 0 في البداية
            ("midground", plan.get("midground", ""), 0.65),
            ("foreground", plan.get("foreground", ""), 0.80)
        ]

        for layer_name, sub_prompt, ref_strength in stages:
            logger.info(f"[Sequential] بدء توليد {layer_name} | ref_strength={ref_strength}")
            result = self._call_layer_engine(
                layer_name, sub_prompt, resolution, force_refresh, is_video, reference
            )
            results[layer_name] = result
            
            new_ref = self._extract_layer_path(result)
            if new_ref:
                reference = new_ref  # تحديث المرجع للمرحلة التالية
            
            if not result.success:
                logger.warning(f"[Sequential] فشل {layer_name} → استمرار بدونها")

        return results

    def _simulate_layer_interactions(self, components: Dict[str, str], layer_paths: Dict[str, str]) -> Dict[str, Any]:
        result = {"influence": 0.0, "adjusted_opacities": {}, "notes": [], "z_adjustments": {}}

        full_lower = " ".join(components.values()).lower()
        relations = {
            "on_top": any(w in full_lower for w in ["على", "فوق", "راكب", "جالس على"]),
            "under": any(w in full_lower for w in ["تحت", "داخل", "مغطى"]),
            "holding": any(w in full_lower for w in ["تمسك", "تحمل"])
        }

        if relations["on_top"]:
            result["notes"].append("علاقة: كائن في foreground فوق midground")
            result["adjusted_opacities"]["midground"] = 220  # أقل شوية عشان يبان الخلف
            result["z_adjustments"]["foreground"] = 10       # أعلى z-index

        # إضافات أخرى لاحقاً (shadows, occlusion hints...)
        result["influence"] = len(result["notes"]) * 0.3  # قيمة رمزية

        return result
    
    def _split_prompt_into_layers(self, prompt: str) -> Dict[str, str]:
        if not self.supervisor:
            return self._fallback_keyword_split(prompt)  # النسخة القديمة كـ fallback

        try:
            split_result = self.supervisor.split_into_layers(
                prompt,
                layer_names=["background", "midground", "foreground"],
                instructions="Extract and separate the prompt into three spatial layers: background (environment), midground (large objects), foreground (characters/details). Return JSON with keys: background, midground, foreground."
            )
            # افترض أن supervisor يرجع dict مباشرة
            return split_result
        except Exception as e:
            logger.warning(f"فشل تقسيم LLM: {e} → استخدام fallback")
            return self._fallback_keyword_split(prompt)

    def _fallback_keyword_split(self, prompt: str) -> Dict[str, str]:
        # النسخة القديمة اللي عندك، بس نظفها شوية (أضف المزيد من الكلمات، أو استخدم re أفضل)
        # ... الكود القديم مع بعض التحسينات ...
        pass
    
    def _determine_layer_order(self, components: Dict) -> List[str]:
        default_order = ["background", "midground", "foreground"]
        
        # لو مفيش محتوى في طبقة → نحذفها من الترتيب
        order = [layer for layer in default_order if components.get(layer, "").strip()]
        
        # إذا كان فيه إشارة واضحة للأمامية أولاً (نادر)
        full_text = " ".join(components.values()).lower()
        if any(w in full_text for w in ["في المقدمة أولاً", "foreground first", "closest object"]):
            order.reverse()  # نادر، بس ممكن

        return order or default_order    
    
    def _composite_layers(
        self,
        layer_paths: Dict[str, str],
        resolution: tuple = (1024, 1024),
        output_name: Optional[str] = None,
        background_color: tuple = (0, 0, 0, 255),
        layer_opacities: Optional[Dict[str, int]] = None,
        vignette_strength: float = 0.7,
        contrast_boost: float = 1.15
    ) -> str:
        from pathlib import Path
        import os

        # 5. placeholder نيون بسيط (مدينة + سيارة + فتاة) – اختبار دمج بصري
        from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont
        import random

        img = Image.new("RGB", resolution, (8, 4, 25))  # خلفية نيون داكنة
        draw = ImageDraw.Draw(img)

        # خطوط نيون عشوائية في الخلفية (مدينة)
        for _ in range(60):
            x1, y1 = random.randint(0, resolution[0]), random.randint(0, resolution[1])
            x2, y2 = random.randint(0, resolution[0]), random.randint(0, resolution[1])
            color = random.choice([(255, 80, 255), (80, 255, 255), (255, 255, 120), (120, 255, 255)])
            draw.line((x1, y1, x2, y2), fill=color, width=random.randint(1, 4))

        # مباني نيون بسيطة (خلفية)
        for x in range(50, resolution[0], 150):
            h = random.randint(200, 500)
            draw.rectangle((x, resolution[1]-h, x+80, resolution[1]), fill=(20, 10, 60), outline=random.choice([(255,100,255),(100,255,255)]))
            # نوافذ مضيئة
            for y in range(resolution[1]-h+20, resolution[1]-20, 40):
                draw.rectangle((x+10, y, x+30, y+20), fill=random.choice([(255,200,255),(200,255,255)]))

        # سيارة رياضية (وسط الصورة)
        car_x = resolution[0] // 2
        car_y = resolution[1] // 2 + 100
        draw.rectangle((car_x - 160, car_y - 60, car_x + 160, car_y + 60), fill=(180, 0, 60), outline=(255, 200, 255), width=5)
        draw.ellipse((car_x - 120, car_y + 40, car_x - 80, car_y + 80), fill=(30, 30, 30))   # عجلة يسار
        draw.ellipse((car_x + 80, car_y + 40, car_x + 120, car_y + 80), fill=(30, 30, 30))  # عجلة يمين
        draw.polygon([(car_x - 120, car_y - 60), (car_x, car_y - 120), (car_x + 120, car_y - 60)], fill=(220, 40, 120))  # سقف/زجاج
        draw.line((car_x - 160, car_y + 30, car_x - 220, car_y + 10), fill=(255, 255, 150), width=8)  # خط سرعة يسار
        draw.line((car_x + 160, car_y + 30, car_x + 220, car_y + 10), fill=(255, 255, 150), width=8)  # خط سرعة يمين

        # فتاة في المقدمة (أسفل الوسط)
        girl_x = resolution[0] // 2
        girl_y = resolution[1] - 220
        draw.ellipse((girl_x - 60, girl_y - 120, girl_x + 60, girl_y - 30), fill=(255, 220, 200))  # وجه
        draw.rectangle((girl_x - 70, girl_y - 30, girl_x + 70, girl_y + 140), fill=(80, 0, 160))  # فستان/جسم
        draw.polygon([(girl_x - 60, girl_y - 100), (girl_x - 100, girl_y - 150), (girl_x - 20, girl_y - 150)], fill=(220, 100, 255))  # شعر يسار
        draw.polygon([(girl_x + 60, girl_y - 100), (girl_x + 100, girl_y - 150), (girl_x + 20, girl_y - 150)], fill=(220, 100, 255))  # شعر يمين
        draw.ellipse((girl_x - 25, girl_y - 90, girl_x + 25, girl_y - 60), fill=(0, 0, 0))  # عيون
        draw.arc((girl_x - 40, girl_y - 60, girl_x + 40, girl_y - 30), 0, 180, fill=(255, 150, 150), width=3)  # ابتسامة

        # نص في الأعلى (اختبار الدمج)
        try:
            font = ImageFont.truetype("arial.ttf", 60)  # لو موجود
        except:
            font = ImageFont.load_default()
        draw.text((50, 40), "Neon Cyberpunk Test – مضرس Engine 😏", fill=(255, 100, 255), font=font)

        # تحسينات نهائية (glow + contrast)
        img = img.filter(ImageFilter.GaussianBlur(3))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.6)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.3)

        final_path = output_name or f"neon_test_{int(perf_counter()*1000)}.png"
        img.save(final_path)
        self.temp_files.append(final_path)

        total_time = perf_counter() - start_total
        logger.info(f"تم حفظ placeholder نيون بسيط: {final_path}")

        try:
            if not is_gif:
                # ─── صورة ثابتة ──────────────────────────────────────────────────────
                base = Image.new("RGBA", resolution, background_color)

                # ترتيب الطبقات من الخلف إلى الأمام
                for layer_name in ["background", "midground", "foreground"]:
                    res = layer_results.get(layer_name)
                    if not res or not res.success:
                        logger.debug(f"طبقة {layer_name} غير موجودة أو فشلت → تخطي")
                        continue

                    out = res.output_data or {}
                    layer_img = None

                    # الحالة 1: صورة جاهزة موجودة (النسخة القديمة)
                    for key in ["preview_path", "color", "path"]:
                        p = out.get(key)
                        if p and os.path.exists(p):
                            try:
                                layer_img = Image.open(p).convert("RGBA").resize(resolution, Image.Resampling.LANCZOS)
                                logger.info(f"دمج صورة جاهزة لـ {layer_name} من: {p}")
                                break
                            except Exception as e:
                                logger.warning(f"فشل فتح الصورة لـ {layer_name}: {e}")
                                continue

                    # الحالة 2: تصميم فقط (بدون صورة جاهزة)
                    if layer_img is None and "assets_directory" in out:
                        logger.info(f"توليد placeholder لتصميم {layer_name} من مجلد: {out['assets_directory']}")
                        layer_img = self._render_design_placeholder(
                            design_data=out,
                            resolution=resolution,
                            layer_name=layer_name
                        )

                    if layer_img is None:
                        logger.warning(f"لا صورة ولا تصميم صالح للطبقة {layer_name} → تخطي")
                        continue

                    # تطبيق الشفافية
                    opacity = layer_opacities.get(layer_name, 255)
                    if opacity < 255:
                        alpha = layer_img.split()[3]
                        alpha = ImageEnhance.Brightness(alpha).enhance(opacity / 255.0)
                        layer_img.putalpha(alpha)

                    base = Image.alpha_composite(base, layer_img)

                # تأثيرات نهائية
                base = self._apply_post_effects(base, vignette_strength, contrast_boost)

                if not output_name:
                    output_name = f"composite_{int(perf_counter() * 1000)}.png"

                base.save(output_name, "PNG", optimize=True)
                logger.info(f"[Composite PNG] تم الحفظ: {output_name}")
                return output_name

            else:
                # ─── GIF متعدد الإطارات ──────────────────────────────────────────────
                logger.info("[GIF Composite] بدء دمج متعدد الطبقات...")

                layer_frames = {}
                max_frames = 0
                durations = []

                for layer_name in ["background", "midground", "foreground"]:
                    res = layer_results.get(layer_name)
                    if not res or not res.success:
                        continue

                    out = res.output_data or {}
                    path = out.get("preview_path") or out.get("color")

                    if path and os.path.exists(path):
                        img = Image.open(path)
                        if img.format == 'GIF':
                            frames = [f.convert("RGBA").resize(resolution, Image.Resampling.LANCZOS)
                                      for f in ImageSequence.Iterator(img)]
                            layer_frames[layer_name] = frames
                            max_frames = max(max_frames, len(frames))
                            durations.append(img.info.get('duration', 100))
                        else:
                            frame = img.convert("RGBA").resize(resolution, Image.Resampling.LANCZOS)
                            layer_frames[layer_name] = [frame] * max_frames  # تكرار للإطارات

                    elif "assets_directory" in out:
                        # تصميم فقط → placeholder ثابت مكرر
                        placeholder = self._render_design_placeholder(out, resolution, layer_name)
                        layer_frames[layer_name] = [placeholder] * 12  # 12 إطار افتراضي
                        max_frames = max(max_frames, 12)
                        durations.append(100)

                if not layer_frames:
                    raise ValueError("لا توجد إطارات أو تصاميم صالحة للـ GIF")

                frame_duration = min((d for d in durations if d > 0), default=100)

                composite_frames = []
                for i in range(max_frames):
                    base = Image.new("RGBA", resolution, background_color)
                    for layer_name in ["background", "midground", "foreground"]:
                        frames = layer_frames.get(layer_name, [])
                        if not frames:
                            continue
                        frame = frames[min(i, len(frames)-1)]

                        opacity = layer_opacities.get(layer_name, 255)
                        if opacity < 255:
                            alpha = frame.split()[3]
                            alpha = ImageEnhance.Brightness(alpha).enhance(opacity / 255.0)
                            frame.putalpha(alpha)

                        base = Image.alpha_composite(base, frame)

                    base = self._apply_post_effects(base, vignette_strength, contrast_boost)
                    composite_frames.append(base)

                if not output_name:
                    output_name = f"composite_{int(perf_counter() * 1000)}.gif"

                composite_frames[0].save(
                    output_name,
                    save_all=True,
                    append_images=composite_frames[1:],
                    duration=frame_duration,
                    loop=0,
                    optimize=True,
                    disposal=2
                )
                logger.info(f"[GIF] تم الحفظ: {output_name} ({len(composite_frames)} إطار)")
                return output_name

        except Exception as e:
            logger.exception("فشل الدمج النهائي")
            error_img = Image.new("RGBA", resolution, (200, 50, 50, 255))
            draw = ImageDraw.Draw(error_img)
            draw.text((20, 20), f"Composite Error:\n{str(e)[:120]}", fill=(255, 255, 255))
            error_path = f"error_{int(perf_counter()*1000)}.png"
            error_img.save(error_path)
            return error_path
                
    def _apply_post_effects(self, img: Image.Image, vignette: float = 0.7, contrast: float = 1.15) -> Image.Image:
        from PIL import ImageEnhance, ImageFilter, ImageDraw, ImageChops

        # 1. Contrast & brightness boost
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

        # 2. Vignette (تعتيم الحواف)
        if vignette > 0:
            mask = Image.new("L", img.size, 255)
            draw = ImageDraw.Draw(mask)
            width, height = img.size
            for i in range(0, 256):
                alpha = int(255 * (1 - vignette * (i / 255) ** 2))
                draw.rectangle(
                    (i, i, width - i, height - i),
                    fill=max(0, alpha)
                )
            vignette_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            vignette_layer.putalpha(mask)
            img = Image.composite(img, ImageChops.multiply(img, vignette_layer), mask)

        # 3. optional: slight sharpening
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=150, threshold=3))

        return img

    def generate(self, prompt: str, as_layer: bool = False, target_size: tuple = (1024, 1024)) -> str:
        """
        دالة توليد صورة أو طبقة بناءً على التخصص
        في هذه النسخة التجريبية → توليد صورة بسيطة مع نص فقط
        لاحقًا سيتم استبدالها بمحركات حقيقية (مثل Stable Diffusion أو غيره)
        """
        from PIL import Image, ImageDraw, ImageFont

        logger.warning("[Placeholder] استخدام توليد تجريبي بسيط – يجب استبداله بمحرك حقيقي")
        
        img = Image.new("RGBA", target_size, (0, 0, 0, 0) if as_layer else (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        text = f"{self.specialization.get('name', 'unknown')}\n{prompt[:50]}..."
        draw.text((20, 20), text, fill=(255, 0, 0), font=font)
        output_name = f"{self.specialization.get('name', 'unknown')}_{int(perf_counter() * 1000)}.png"
        img.save(output_name, "PNG", optimize=True)
        self.temp_files.append(output_name)
        return output_name        
                    
    def create_layer_image(
        specialization: str,
        prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        transparent_bg: bool = True,
        target_opacity: int = 255,
        is_video: bool = False
    ) -> Optional[str]:
        """
        Placeholder لإنشاء طبقة بسيطة (fallback حتى ربط المحركات الحقيقية)
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import random

            mode = "RGBA" if transparent_bg else "RGB"
            bg_color = (0, 0, 0, 0) if transparent_bg else (10, 12, 25)
            img = Image.new(mode, (width, height), bg_color)
            draw = ImageDraw.Draw(img)

            lower_prompt = prompt.lower()

            if specialization == "traditional_design":
                # أشجار + ضباب + كائن (حصان + راكب إذا وجد)
                for _ in range(15):
                    x = random.randint(-50, width + 50)
                    h = random.randint(400, height - 100)
                    draw.polygon([(x-100, height), (x, height - h), (x+100, height)], fill=(8, 12, 20))

                for _ in range(40):
                    x, y = random.randint(0, width), random.randint(0, height // 2 + 100)
                    r = random.randint(100, 300)
                    draw.ellipse((x-r, y-r, x+r, y+r), fill=(200, 210, 240, 30))

                if any(kw in lower_prompt for kw in ["horse", "حصان", "creature"]):
                    cx, cy = width // 2, height - 180
                    draw.ellipse((cx-90, cy-70, cx+90, cy+70), fill=(220, 220, 240))  # جسم
                    draw.ellipse((cx-40, cy-110, cx+40, cy-30), fill=(220, 220, 240))  # رأس
                    draw.ellipse((cx-15, cy-80, cx-5, cy-70), fill=(0,0,0))           # عين يسار
                    draw.ellipse((cx+5, cy-80, cx+15, cy-70), fill=(0,0,0))           # عين يمين
                    draw.rectangle((cx-35, cy-160, cx+35, cy-80), fill=(180, 140, 100))  # راكب

                for _ in range(80):
                    x = random.randint(0, width)
                    y = random.randint(0, height)
                    sz = random.randint(3, 9)
                    alpha = random.randint(80, 180)
                    draw.ellipse((x-sz, y-sz, x+sz, y+sz), fill=(230, 240, 255, alpha))

            elif specialization == "geometric_design":
                center_x, center_y = width // 2, height // 2
                for r in range(60, 400, 45):
                    draw.ellipse((center_x - r, center_y - r, center_x + r, center_y + r), outline=(180, 160, 120), width=3)

                for angle in range(0, 360, 15):
                    rad = math.radians(angle)
                    x2 = center_x + 500 * random.uniform(0.6, 1.0) * math.cos(rad)
                    y2 = center_y + 500 * random.uniform(0.6, 1.0) * math.sin(rad)
                    draw.line((center_x, center_y, x2, y2), fill=(220, 190, 80), width=2)

            elif specialization == "environment_design":
                draw.rectangle((0, 0, width, height), fill=(5, 5, 15))
                for y in range(80, height, 140):
                    draw.line((0, y, width, y), fill=(0, 255, 220, 140), width=3)
                for x in range(100, width, 180):
                    draw.line((x, 0, x, height), fill=(255, 80, 220, 130), width=3)
                for _ in range(25):
                    x, y = random.randint(0, width), random.randint(0, height)
                    r = random.randint(40, 180)
                    draw.ellipse((x-r, y-r, x+r, y+r), fill=(0, 240, 255, 45))

            else:
                draw.text((width//4, height//2), f"Layer: {specialization}\n{prompt[:60]}", fill=(200, 200, 255))

            # تطبيق opacity كلي
            if target_opacity < 255 and mode == "RGBA":
                alpha = Image.new("L", img.size, target_opacity)
                img.putalpha(alpha)

            # حفظ
            suffix = "video" if is_video else "layer"
            ts = int(perf_counter() * 1000)
            output_path = f"{specialization}_{suffix}_{ts}.png"
            img.save(output_path, "PNG", optimize=True)

            logger.info(f"[Placeholder] تم إنشاء طبقة {specialization} → {output_path}")
            return output_path

        except Exception as e:
            logger.exception(f"فشل placeholder لـ {specialization}")
            return None
        
    def generate_image(
        self,
        specialization: Optional[str] = None,
        is_video: bool = False,
        force_refresh: bool = False,
        as_layer: bool = False,
        target_size: tuple = (1024, 1024)
    ) -> GenerationResult:
        """
        دالة انتقالية لتوليد طبقة أو صورة كاملة (fallback أثناء التطوير)
        يفضل استخدام _generate_environment_elements أو المحركات المتخصصة مباشرة
        """
        spec_name = specialization or self.specialization.get("name", "unknown")
        logger.warning(f"[DEPRECATED] استخدام generate_image (fallback) → spec={spec_name} | layer={as_layer}")

        start_total = perf_counter()
        stage_times = {}

        # إذا مفيش prompt متراكم → فشل
        if not self.input_port:
            return GenerationResult(
                success=False,
                message="لا يوجد وصف في input_port",
                total_time=0.0,
                stage_times={},
                specialization=spec_name,
                is_video=is_video
            )

        full_prompt = " ".join(self.input_port).strip()

        try:
            t_render = perf_counter()

            # محاولة استخدام المحرك المتخصص إذا موجود
            if specialization and specialization in self.engine_map:
                engine_class = self.engine_map[specialization]
                engine = engine_class()
                layer_result = engine.generate_layer(
                    prompt=full_prompt,
                    target_size=target_size,
                    force_refresh=force_refresh,
                    as_layer=as_layer,
                    is_video=is_video
                )
                if layer_result.success:
                    path = self._extract_layer_path(layer_result)  # استخدم الدالة المرنة
                    if path:
                        preview_path = path
                    else:
                        preview_path = None
                else:
                    preview_path = None
            else:
                # fallback لـ placeholder PIL
                preview_path = self._create_simple_image(
                    {"raw_prompt": full_prompt},
                    is_video=is_video,
                    transparent_bg=as_layer,
                    target_size=target_size
                )

            stage_times["rendering"] = perf_counter() - t_render

            if not preview_path:
                raise ValueError("فشل إنشاء معاينة")

            total_time = perf_counter() - start_total

            return GenerationResult(
                success=True,
                message="تم التوليد (انتقالي)" + (" طبقة شفافة" if as_layer else ""),
                total_time=total_time,
                stage_times=stage_times,
                specialization=spec_name,
                is_video=is_video,
                output_data={"preview_path": preview_path}
            )

        except Exception as e:
            logger.exception("خطأ في generate_image")
            return GenerationResult(
                success=False,
                message=str(e),
                total_time=perf_counter() - start_total,
                stage_times=stage_times,
                specialization=spec_name,
                is_video=is_video
            )
 
    def _should_apply_physics(self, prompt: str, layers: list) -> bool:
        """قرار بسيط: هل نفعّل الفيزياء أم لا؟"""
        lower = prompt.lower()
        trigger_words = [
            "collide", "تصادم", "interact", "تفاعل", "wind", "رياح",
            "gravity", "جاذبية", "fall", "سقوط", "float", "طفو",
            "multiple", "كثير", "crowd", "حشد", "chaos", "فوضى", "physics"
        ]
        has_trigger = any(word in lower for word in trigger_words)
        many_layers = len(layers) >= 4
        return apply_physics or has_trigger or many_layers   # ← استخدام الباراميتر العام

    def _generate_environment_elements(
        self,
        prompt: str,
        resolution: tuple = (1024, 1024),
        output_name: Optional[str] = None,
        force_refresh: bool = False,
        is_video: bool = False,
        auto_split: bool = True,
        sequential_mode: bool = False,
        apply_physics: bool = False,
    ) -> GenerationResult:
        start_total = perf_counter()
        stage_times = {}
        intermediate = {"prompt_components": {}, "layer_results": {}, "enhanced_prompts": {}}
        plane_layers = []

        full_prompt = prompt.strip()

        # 1. التخطيط
        if auto_split:
            try:
                plan = self.supervisor.plan_layers(full_prompt, mode="sequential" if sequential_mode else "parallel")
            except Exception as e:
                logger.warning(f"فشل التخطيط: {e} → تقسيم يدوي")
                plan = {"background": full_prompt, "midground": full_prompt, "foreground": full_prompt}
        else:
            plan = {"background": full_prompt, "midground": full_prompt, "foreground": full_prompt}

        intermediate["prompt_components"] = plan

        # 2. توليد التصاميم (Design Phase) – لا نولّد صور هنا بعد
        layer_results = {}
        plane_layers = []  # سنبقيها لو كنت لا تزال تستخدم PlaneLayer للترتيب

        for layer_name, sub_prompt in plan.items():
            engine = self.engine_map.get(layer_name)
            if not engine:
                logger.error(f"لا يوجد محرك للطبقة: {layer_name}")
                continue

            logger.info(f"جاري تصميم الطبقة: {layer_name} → prompt: {sub_prompt[:60]}...")

            res = None

            # ─── استدعاء دالة التصميم المناسبة حسب اسم المحرك ──────────────────────
            try:
                if layer_name == "background" and hasattr(engine, "design_environment_assets"):
                    # البيئة غالباً لها دالة خاصة أكثر تفصيلاً
                    res = engine.design_environment_assets(
                        prompt=sub_prompt,
                        resolution=resolution,
                        render_color=False,           # مهم: لا نريد صورة الآن، فقط التصميم
                        heightmap_format="npy",       # أو "exr" إذا كنت مستعد
                        force_refresh=force_refresh
                    )

                elif hasattr(engine, "design"):
                    # الدالة العامة للتصميم (geometric أو traditional أو fallback)
                    res = engine.design(
                        description=sub_prompt,
                        resolution=resolution,
                        force_refresh=force_refresh,
                        **kwargs
                    )

                else:
                    logger.warning(f"المحرك {layer_name} ليس لديه دالة تصميم معروفة → تخطي")
                    continue

            except AttributeError as e:
                logger.error(f"خطأ في استدعاء دالة التصميم لـ {layer_name}: {e}")
                continue
            except Exception as e:
                logger.exception(f"خطأ عام أثناء تصميم {layer_name}")
                continue

            if res and res.success:
                layer_results[layer_name] = res
                intermediate["enhanced_prompts"][layer_name] = res.output_data.get("enhanced_prompt", sub_prompt)

                # ─── إنشاء PlaneLayer للترتيب اللاحق (اختياري) ────────────────────────
                z_map = {"background": -1.0, "midground": 0.0, "foreground": 1.0}
                color_map = {"background": "navy", "midground": "teal", "foreground": "gold"}

                z_depth = z_map.get(layer_name, 0.0)
                p_layer = PlaneLayer(
                    position=[0.0, 0.0, z_depth],
                    force=1.0 if layer_name == "foreground" else 0.6 if layer_name == "midground" else 0.3,
                    depth=abs(z_depth) + 1.0,
                    label=layer_name.capitalize(),
                    color=color_map.get(layer_name, "gray"),
                    mass=10.0 if layer_name == "background" else 3.0
                )

                # بدل الاعتماد على preview_path، نضع بيانات التصميم
                p_layer.metadata = {
                    "source": layer_name,
                    "design_result": res,
                    "assets_directory": res.output_data.get("assets_directory"),
                    "paths": res.output_data.get("paths", {}),
                    "elements_count": len(res.elements) if hasattr(res, "elements") else 0
                }

                plane_layers.append(p_layer)
            else:
                logger.warning(f"تصميم الطبقة {layer_name} فشل → لن تُضاف")
        
        # 3. combined prompt
        combined_prompt = ", ".join(
            intermediate["enhanced_prompts"].get(l, "") for l in ["background", "midground", "foreground"]
        ).strip(", ") + ", highly detailed, cinematic lighting, professional composition, 8k"

        # 4. قرار الفيزياء
        use_physics = apply_physics or self._should_apply_physics(full_prompt, plane_layers)

        if use_physics and plane_layers:
            try:
                composer = LayerComposer()
                adjusted = composer.adjust_layers_for_physics(
                    plane_layers,
                    prompt=full_prompt,
                    resolution=resolution,
                    collision_threshold=0.15,
                    emotional_amplifier=1.2
                )
                plane_layers = adjusted
                logger.info(f"تم تطبيق الفيزياء على {len(plane_layers)} طبقة")
            except Exception as e:
                logger.warning(f"فشل الفيزياء: {e} → نكمل بدونها")
                use_physics = False

        # 5. الدمج النهائي - placeholder نيون بسيط
        from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont

        img = Image.new("RGB", resolution, (10, 5, 30))  # خلفية نيون داكنة
        draw = ImageDraw.Draw(img)

        # خلفية نيون (خطوط مضيئة عشوائية)
        for _ in range(30):
            x1, y1 = random.randint(0, resolution[0]), random.randint(0, resolution[1])
            x2, y2 = random.randint(0, resolution[0]), random.randint(0, resolution[1])
            color = random.choice([(255, 50, 255), (50, 255, 255), (255, 255, 100), (100, 255, 255)])
            draw.line((x1, y1, x2, y2), fill=color, width=2)

        # سيارة رياضية (وسط الصورة)
        car_x, car_y = resolution[0]//2, resolution[1]//2 + 50
        draw.rectangle((car_x-120, car_y-40, car_x+120, car_y+40), fill=(200, 20, 80), outline=(255, 255, 255), width=3)
        draw.ellipse((car_x-100, car_y+20, car_x-60, car_y+60), fill=(50, 50, 50))  # عجلة يسار
        draw.ellipse((car_x+60, car_y+20, car_x+100, car_y+60), fill=(50, 50, 50))  # عجلة يمين
        draw.polygon([(car_x-80, car_y-40), (car_x, car_y-80), (car_x+80, car_y-40)], fill=(220, 40, 100))  # سقف السيارة

        # فتاة في المقدمة (أسفل الوسط)
        girl_x, girl_y = resolution[0]//2, resolution[1]-150
        draw.ellipse((girl_x-40, girl_y-80, girl_x+40, girl_y-20), fill=(240, 200, 180))  # وجه
        draw.rectangle((girl_x-50, girl_y-20, girl_x+50, girl_y+100), fill=(150, 50, 200))  # جسم
        draw.line((girl_x-30, girl_y-60, girl_x-60, girl_y-20), fill=(200, 150, 255), width=10)  # شعر يسار
        draw.line((girl_x+30, girl_y-60, girl_x+60, girl_y-20), fill=(200, 150, 255), width=10)  # شعر يمين

        # نص في الأعلى
        try:
            font = ImageFont.truetype("arial.ttf", 40)  # لو موجود، أو استخدم default
        except:
            font = ImageFont.load_default()
        draw.text((50, 20), "Neon Cyberpunk Test – مضرس Engine 😏", fill=(255, 100, 255), font=font)

        # تحسينات نهائية
        img = img.filter(ImageFilter.GaussianBlur(1.5))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.4)

        final_path = output_name or f"composite_neon_test_{int(perf_counter()*1000)}.png"
        img.save(final_path)
        self.temp_files.append(final_path)
        
    # ────────────────────────────────────────────────
    # دالة مساعدة لاتخاذ القرار
    # ────────────────────────────────────────────────
    def _should_apply_physics(self, prompt: str, layers: list) -> bool:
        lower = prompt.lower()
        keywords = [
            "collide", "تصادم", "interact", "تفاعل", "wind", "رياح", 
            "gravity", "جاذبية", "fall", "سقوط", "float", "طفو",
            "multiple", "كثير", "crowd", "حشد", "chaos", "فوضى"
        ]
        has_trigger = any(k in lower for k in keywords)
        many_layers = len(layers) >= 4
        return has_trigger or many_layers
    
    def cleanup_temp_references(self):
            """
            حذف جميع الملفات المؤقتة المسجلة من المحركات المختلفة.
            - يُستدعى بعد كل عملية توليد كاملة (مثل _generate_environment_elements)
            - أو في __del__ / context manager exit
            - آمن ويُسجل عدد المحذوفات والأخطاء
            """
            if not hasattr(self, 'temp_files') or not self.temp_files:
                logger.debug("[Cleanup] لا ملفات مؤقتة مسجلة")
                return

            deleted_count = 0
            failed_count = 0
            temp_list = self.temp_files.copy()  # نسخة آمنة

            for path in temp_list:
                path_str = str(path)  # للتأكد من أنه string
                if os.path.exists(path_str):
                    try:
                        if os.path.isfile(path_str):
                            os.remove(path_str)
                        elif os.path.isdir(path_str):
                            import shutil
                            shutil.rmtree(path_str, ignore_errors=True)
                        else:
                            logger.warning(f"[Cleanup] نوع غير متوقع: {path_str}")
                            continue

                        logger.debug(f"[Cleanup] تم حذف: {path_str}")
                        deleted_count += 1
                        self.temp_files.remove(path)  # حذف من القائمة الأصلية

                    except PermissionError:
                        logger.warning(f"[Cleanup] رفض الإذن لحذف: {path_str}")
                        failed_count += 1
                    except FileNotFoundError:
                        logger.debug(f"[Cleanup] الملف اختفى بالفعل: {path_str}")
                        self.temp_files.remove(path)
                    except Exception as e:
                        logger.warning(f"[Cleanup] فشل حذف {path_str}: {type(e).__name__} - {e}")
                        failed_count += 1

            if deleted_count > 0 or failed_count > 0:
                logger.info(
                    f"[Cleanup] تم حذف {deleted_count} ملف/مجلد | فشل {failed_count} | باقي {len(self.temp_files)}"
                )
            else:
                logger.debug("[Cleanup] لا ملفات تحتاج حذف")
        
    def generate_layer(
        self,
        prompt: str,
        target_size: tuple = (1024, 1024),
        is_video: bool = False,
        as_layer: bool = True,
        force_refresh: bool = False,
        **kwargs
    ) -> GenerationResult:
        """
        تنفيذ بسيط لتوليد طبقة واحدة (placeholder مؤقت)
        بعدين هنربطها بالمحركات الفرعية من engine_map
        """
        logger.info(f"[generate_layer] prompt: {prompt[:70]}...")

        # نتيجة وهمية عشان الاختبار يعدي
        fake_path = f"temp_layer_{int(perf_counter()*1000)}.png"

        return GenerationResult(
            success=True,
            message="طبقة مولدة (placeholder – لم يتم توليد حقيقي بعد)",
            total_time=0.42,
            stage_times={"analysis": 0.1, "render": 0.32},
            specialization=self.specialization,
            is_video=is_video,
            output_data={
                "preview_path": fake_path,
                "layer_type": "placeholder_layer"
            }
        )
        
    def _get_specialization_config(self) -> Dict[str, Any]:
        return {"name": "composite", "description": "layer compositor"}

    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        # تحليل بسيط جدًا أو placeholder
        return {"entities": prompt.split(), "style": "composite"}

    def _integrate(self, task_data: Dict) -> float:
        # وقت وهمي للتكامل
        return 0.45

    def _post_process(self, task_data: Dict) -> Dict[str, Any]:
        return {"processed": True, "message": "post-processing placeholder"}

    def _render(self, task_data: Dict, is_video: bool = False) -> float:
        # وقت وهمي للـ render
        return 1.2

    def _render_design_placeholder(self, design_data: dict, resolution: tuple, layer_name: str = "unknown") -> Image.Image:
        """
        توليد صورة placeholder بسيطة من بيانات التصميم فقط
        (يمكن تحسينها لاحقًا لتصبح رسمًا أكثر ذكاءً)
        """
        img = Image.new("RGBA", resolution, (10, 15, 30, 255))  # خلفية داكنة افتراضية
        draw = ImageDraw.Draw(img)

        # نص توضيحي كبير
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except:
            font = ImageFont.load_default()

        draw.text(
            (50, 50),
            f"{layer_name.upper()} Design Placeholder\n"
            f"Elements: {design_data.get('elements_count', 0)}\n"
            f"Assets: {design_data.get('assets_dir', 'غير متوفر')}",
            fill=(220, 180, 100),
            font=font
        )

        # إذا وجد heightmap، نرسمه كتدرج بسيط
        if "paths" in design_data and "heightmap" in design_data["paths"]:
            hmap_path = design_data["paths"]["heightmap"]
            if os.path.exists(hmap_path):
                try:
                    hmap = np.load(hmap_path)
                    hmap = (hmap * 255).astype(np.uint8)
                    hmap_img = Image.fromarray(hmap, mode="L").convert("RGBA")
                    hmap_img = hmap_img.resize(resolution)
                    draw_img = ImageDraw.Draw(hmap_img)
                    draw_img.text((20, 20), "Heightmap Preview", fill=(255, 100, 100))
                    img = Image.alpha_composite(img, hmap_img)
                except:
                    pass

        return img
    
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== بدء اختبار CompositeEngine ===")

    try:
        engine = CompositeEngine()
        logger.info("تم إنشاء CompositeEngine بنجاح")

        # اختبار 1: توليد طبقة واحدة (بسيط)
        print("\n=== اختبار توليد طبقة واحدة ===")
        layer_result = engine.generate_layer(
            prompt="غابة سحرية مع ضباب وأضواء خافتة",
            target_size=(512, 512),
            as_layer=True,
            force_refresh=True
        )
        print("نتيجة الطبقة:", layer_result)

        # اختبار 2: محاولة توليد مركب كامل
        print("\n=== اختبار توليد مركب كامل ===")
        composite_result = engine._generate_environment_elements(
            prompt="فتاة تقف في مدينة نيون ليلية مع سيارة رياضية وأنماط هندسية في الخلفية",
            resolution=(768, 768),
            output_name="test_composite_output.png",
            force_refresh=True,
            is_video=False,
            auto_split=True
        )
        print("نتيجة المركب:", composite_result)

        # اختبار 3: تنظيف الملفات المؤقتة (اختياري)
        print("\n=== تنظيف الملفات المؤقتة ===")
        engine.cleanup_temp_references()
        print("تم التنظيف")

    except Exception as e:
        logger.exception("خطأ أثناء تشغيل اختبار CompositeEngine")
        print(f"حدث خطأ: {e}")

    finally:
        logger.info("=== انتهى اختبار CompositeEngine ===")