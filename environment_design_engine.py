# environment_design_engine.py
print("→ بدأ environment_design_engine.py")

"""
محرك تصميم البيئة: مسؤول عن إنشاء طبقة خلفية / بيئية بناءً على وصف نصي
يأخذ وصفاً نصياً (عربي أو إنجليزي أو مختلط) وينتج طبقة بيئية (PlaneLayer) تحتوي على عناصر مثل الإضاءة، الجو، التضاريس، السماء، إلخ
يستخدم تقنيات مختلفة (مثل استدعاء API خارجي، أو توليد إجرائي، أو حتى LLM خفيف لتحسين الوصف) لتحقيق أفضل نتيجة ممكنة
يرجع نتيجة منظمة (EnvironmentDesignResult) تحتوي على الطبقة + البيانات الوصفية المفيدة للتوليد النهائي أو التوثيق
"""

from typing import List, Dict, Any, Optional  # لو استخدمت List في مكان آخر
import time
from typing import List
import re  # أضف هذا في أعلى الملف لو مش موجود
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from pathlib import Path
from time import perf_counter
from typing import Union
from datetime import datetime
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from collections import defaultdict
from PIL import Image, ImageDraw
import pickle  # اختياري لو عايز binary
import trimesh  # لو عايز OBJ export (pip install trimesh)
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material, PbrMetallicRoughness
# from pygltflib.utils import triangle_indices_to_bytes, triangle_vertices_to_bytes  # تأكد من استيرادها

from memory_manager import GenerativeMemoryManager
from generation_result import GenerationResult
from Core_Image_Generation_Engine import CoreImageGenerationEngine

logger = logging.getLogger(__name__)

@dataclass
class EnvironmentElement:
    type: str  # مثل 'tree', 'river', 'fog', 'wind', 'light', 'shadow'
    position: np.ndarray  # [x, y, z]
    properties: Dict[str, Any]  # خصائص مثل size, intensity, direction, color, إلخ

    def to_dict(self):
        d = self.__dict__.copy()
        d['position'] = self.position.tolist() if hasattr(self.position, 'tolist') else self.position
        return d

@dataclass
class EnvironmentDesignResult:
    success: bool = False
    elements: list[EnvironmentElement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    design_time_seconds: float = 0.0

class environment_design_engine (CoreImageGenerationEngine):
    """
    مسؤول عن **تصميم البيئة فقط** (إنشاء طبقة خلفية / بيئية)
    لا يهتم بالشخصيات أو الأشياء الأمامية أو التجميع النهائي
    يرجع نتيجة منظمة يفهمها Final_Generation أو Layer Compositor
    """

    def __init__(self):
        self.specialization = {
            "domain": "environment",
            "supported_aspects": [
                "lighting", "atmosphere", "weather", "terrain", "sky", "fog", "depth", "color_grading"
            ],
            "typical_output": "background_layer / environment_plane"
        }
        self.memory_manager = GenerativeMemoryManager()  # لو مش موجود، احذفه أو استبدله
        self.memory_manager.context_type = "environment"
        self.temp_files: list[str] = []
        self.logger = logging.getLogger(__name__)  # logging مستقل

        self._log_initial_state()  # إذا تحتاجه
    

        type: str
        position: List[float]  # [x, y, z] بدل np.array
        properties: Dict
        
    def _validate_specialization(self):
        """
        تنفيذ placeholder للتحقق من التخصص (الأب بيطلبها)
        """
        pass  # أو logger.debug("التحقق من التخصص في البيئة - تم")

    def _initialize_units(self):
        pass

    def _initialize_memory_manager(self):
        pass

    def _initialize_additional_state(self):
        pass

    def _log_specialization_details(self):
        logger.debug(f"تفاصيل تخصص البيئة: {self.specialization}")

    def _log_initial_state(self):
        self.logger.debug("حالة تهيئة البيئة تمت")

    def _run_initial_diagnostics(self):
        logger.debug("تشخيص أولي للبيئة - ناجح")
        
    def _get_specialization_config(self) -> Dict[str, Any]:
        return {"name": "environment", "description": "تصميم البيئة"}

    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:  # لو كانت من الأب، هنا نسخة مستقلة
        return {"entities": prompt.split(), "style": "environment"}

    def _integrate(self, task_data: Dict) -> float:
        return 0.6

    def _post_process(self, task_data: Dict) -> Dict[str, Any]:
        return {"processed": True}

    def _render(self, task_data: Dict, is_video: bool = False) -> float:
        return 1.5

    def create_environment(
        self,
        description: str,
        resolution: tuple = (1920, 1080),
        is_looping: bool = False,
        duration_sec: float = 10.0,
        **kwargs
    ):
        """
        الدالة الرئيسية التي تنتج البيئة
        الوصف يأتي جاهزاً
        """
        # مثال على هيكلية منطقية (بدون فرض أي تقنية معينة)
        
        env_type = self._classify_environment(description)
        
        if env_type == "cyberpunk_city":
            return self._generate_cyberpunk_city(description, resolution, **kwargs)
            
        elif env_type == "natural_forest":
            return self._generate_forest_scene(description, resolution, **kwargs)
            
        elif env_type == "space_nebula":
            return self._generate_space_background(description, resolution, **kwargs)
            
        else:
            return self._fallback_generation(description, resolution, **kwargs)

    def _classify_environment(self, desc: str) -> str:
        # تصنيف بسيط جداً (يمكن استبداله بأي طريقة أنت تفضلها)
        lower = desc.lower()
        if "cyber" in lower or "نيون" in lower or "سايبر" in lower:
            return "cyberpunk_city"
        if "غابة" in lower or "forest" in lower:
            return "natural_forest"
        if "فضاء" in lower or "nebula" in lower:
            return "space_nebula"
        return "generic"

    def _generate_cyberpunk_city(self, desc, resolution, **kwargs):
        # هنا تضع الكود الفعلي الذي ينتج البيئة
        # (سواء كان استدعاء API خارجي، أو procedural generation، أو أي شيء)
        raise NotImplementedError("يجب تنفيذ توليد مدينة سايبر")

    def design(
        self,
        description: str,
        resolution: tuple = (1024, 768),
        **kwargs
    ) -> EnvironmentDesignResult:
        from time import perf_counter
        start_time = perf_counter()

        # 1. تحليل الوصف
        analysis = self._analyze_environment_description(description)

        # 2. تصميم العناصر
        elements = self._generate_environment_elements(analysis, resolution)

        # 3. تخزين في الذاكرة بالطريقة الصحيحة
        prompt_hash = self.memory_manager.get_prompt_hash(description)
        
        stored = self.memory_manager.store_environment_elements(
            prompt_hash=prompt_hash,
            elements=elements,
            analysis=analysis,
            extra_metadata={"resolution": resolution, **kwargs}
        )

        total_time = perf_counter() - start_time

        return EnvironmentDesignResult(
            success=stored,
            elements=elements if stored else [],
            metadata=analysis,
            message="تم تصميم وتخزين البيئة" if stored else "فشل التخزين",
            design_time_seconds=total_time
        )
    
    def design_environment_assets(
        self,
        prompt: str,
        resolution: tuple = (1024, 1024),
        is_video: bool = False,               # محتفظ به للتوافق، لكن غير مستخدم حالياً
        force_refresh: bool = False,
        render_color: bool = True,            # جديد: هل ننتج الصورة الملونة أم لا
        heightmap_format: str = "exr",       # "exr", "npy", "png_16bit"
        output_subdir: str | None = None,
        **kwargs
    ) -> GenerationResult:
        """
        تصميم أصول بيئية (heightmap + صورة ملونة اختيارية)
        لا يُفترض أن يُنتج طبقة جاهزة للدمج المباشر، بل مواد خام لمراحل لاحقة.
        """
        start_total = perf_counter()
        stage_times = {}
        output_paths = {}

        try:
            # ─── 1. التحليل ────────────────────────────────────────────────
            t_start = perf_counter()
            analysis = self._analyze_environment_description(prompt)
            if not isinstance(analysis, dict):
                analysis = {}
            stage_times["analysis"] = perf_counter() - t_start

            # ─── 2. توليد heightmap (بيانات ارتفاع عددية) ─────────────────
            t_start = perf_counter()
            heightmap = self._generate_heightmap(resolution, analysis)
            stage_times["heightmap_generation"] = perf_counter() - t_start

            # ─── 3. إعداد مجلد الإخراج المنظم ──────────────────────────────
            base_dir = Path("output/environment")
            if output_subdir:
                base_dir = base_dir / output_subdir
            else:
                base_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
            
            base_dir.mkdir(parents=True, exist_ok=True)
            stem = f"env_{int(perf_counter() * 1000)}"

            # ─── 4. حفظ heightmap ───────────────────────────────────────────
            t_start = perf_counter()
            
            if heightmap_format.lower() == "exr":
                heightmap_path = base_dir / f"{stem}_height.exr"
                # ملاحظة: تحتاج دالة حفظ EXR حقيقية (مثال بسيط بـ imageio أو OpenEXR)
                # هنا مجرد placeholder – يجب استبداله بتنفيذ فعلي
                # imageio.imwrite(heightmap_path, heightmap.astype(np.float32))
                logger.warning("حفظ EXR غير منفذ بعد – يُستخدم .npy مؤقتاً")
                heightmap_path = base_dir / f"{stem}_height.npy"
                np.save(heightmap_path, heightmap)
            
            elif heightmap_format.lower() == "png_16bit":
                heightmap_path = base_dir / f"{stem}_height.png"
                # تحويل إلى 16-bit grayscale (يحتاج تطبيع)
                h_norm = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)
                h_16 = (h_norm * 65535).astype(np.uint16)
                Image.fromarray(h_16).save(heightmap_path)
            
            else:  # default npy
                heightmap_path = base_dir / f"{stem}_height.npy"
                np.save(heightmap_path, heightmap)
            
            output_paths["heightmap"] = str(heightmap_path.resolve())
            stage_times["heightmap_export"] = perf_counter() - t_start

            # ─── 5. إنتاج الصورة الملونة (اختياري) ────────────────────────
            color_path = None
            if render_color:
                t_start = perf_counter()
                img = self._render_real_2d_layer(heightmap, analysis, resolution)
                color_path = base_dir / f"{stem}_color.png"
                img.save(color_path)
                output_paths["color"] = str(color_path.resolve())
                stage_times["color_render_and_export"] = perf_counter() - t_start

            # ─── النتيجة النهائية ──────────────────────────────────────────
            total_time = perf_counter() - start_total

            return GenerationResult(
                success=True,
                message="تم تصميم أصول البيئة بنجاح",
                total_time=total_time,
                stage_times=stage_times,
                specialization=self.specialization.get("name", "environment_design"),
                output_data={
                    "assets_directory": str(base_dir.resolve()),
                    "paths": output_paths,
                    "resolution": resolution,
                    "analysis_summary": analysis,
                    "heightmap_format": heightmap_format,
                    "color_rendered": color_path is not None
                }
            )

        except Exception as e:
            logger.exception("خطأ أثناء تصميم أصول البيئة")
            return GenerationResult(
                success=False,
                message=f"فشل: {str(e)}",
                total_time=perf_counter() - start_total,
                stage_times=stage_times,
                specialization=self.specialization.get("name", "environment_design")
            )

    def _render_real_2d_layer(self, heightmap, analysis, size):
        """رسم بيئة 2D واقعية بـ PIL"""
        w, h = size
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # سماء حسب الإضاءة
        sky_color = (135, 206, 235) if "day" in analysis.get("lighting", "") else (10, 10, 30)
        draw.rectangle((0, 0, w, h//2), fill=sky_color)

        # أرض + تضاريس من heightmap
        for x in range(w):
            height = int(heightmap[x // 10, h // 10] * h // 2) if heightmap is not None else h//2
            ground_color = (34, 139, 34) if height < h//4 else (139, 69, 19)
            draw.line((x, h - height, x, h), fill=ground_color, width=2)

        # أشجار واقعية أكتر
        for _ in range(40):
            x = random.randint(0, w)
            y = random.randint(h//2, h)
            tree_h = random.randint(80, 200)
            draw.rectangle((x-20, y-tree_h, x+20, y), fill=(139, 69, 19))  # جذع
            draw.ellipse((x-60, y-tree_h-80, x+60, y-tree_h+20), fill=(0, 128, 0))  # أوراق

        # ضباب إذا موجود
        if "fog" in analysis.get("weather", ""):
            fog = Image.new("RGBA", size, (200, 200, 200, 120))
            img = Image.alpha_composite(img, fog)

        return img

    def _generate_heightmap(self, resolution: tuple, analysis: dict) -> np.ndarray:
        """توليد heightmap بسيط وسريع باستخدام numpy فقط (بدون scipy)"""
        h, w = resolution
        # توليد ضوضاء أساسية
        heightmap = np.random.rand(h, w).astype(np.float32) * 0.6

        # إضافة بعض التلال والوديان (موجات سينوس)
        x = np.linspace(0, 8, w)
        y = np.linspace(0, 8, h)
        X, Y = np.meshgrid(x, y)
        heightmap += (np.sin(X) + np.sin(Y)) * 0.2

        # تنعيم خفيف
        from scipy.ndimage import gaussian_filter
        heightmap = gaussian_filter(heightmap, sigma=6)

        # تطبيع بين 0 و 1
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)
        return heightmap.astype(np.float32)
    
    def _generate_environment_elements(self, enhanced_prompt: str, analysis: dict, resolution: tuple = (1024, 1024), **kwargs) -> list:
        print(f"[DEBUG] analysis type داخل الدالة: {type(analysis).__name__}")
        analysis = analysis or {}

        if analysis is None:
            analysis = {}
            logger.warning("_generate_environment_elements: analysis كان None → تم تحويله لقاموس فارغ")
            
        elif not isinstance(analysis, dict):
            logger.warning(f"_generate_environment_elements: analysis نوعه {type(analysis).__name__} → تم تحويله لقاموس فارغ")
            analysis = {}

        elements = []
        width, height = resolution

        # استخراج المتغيرات بأمان تام (بدون خطأ حتى لو القيم None)
        terrain = str(analysis.get("terrain", "unknown")).lower()
        
        weather_raw = analysis.get("weather")
        weather = str(weather_raw).lower() if weather_raw is not None else ""
        
        mood_raw = analysis.get("mood")
        mood = str(mood_raw).lower() if mood_raw is not None else "neutral"
        
        lighting_raw = analysis.get("lighting_type")
        lighting = str(lighting_raw).lower() if lighting_raw is not None else "natural"

        # باقي الكود كما هو (أشجار، نهر، ضباب، إلخ)
        if "forest" in terrain or "غابة" in terrain or "woods" in terrain:
            for _ in range(random.randint(8, 18)):
                elements.append(EnvironmentElement(
                    type='tree',
                    position=[random.uniform(0, width), random.uniform(0, height), random.uniform(-8, -1)],
                    properties={
                        'size': random.uniform(4, 14),
                        'color': random.choice(['darkgreen', 'forestgreen', '#1a3c34']),
                        'density': random.uniform(0.6, 1.0)
                    }
                ))

        # نهر أو ماء
        if any(w in weather + terrain for w in ["river", "نهر", "lake", "بحيرة", "sea", "بحر"]):
            elements.append(EnvironmentElement(
                type='water_body',
                position=[width * 0.5, height * 0.7, -4],
                properties={
                    'width': width * random.uniform(0.4, 0.8),
                    'flow': random.uniform(0.1, 0.4),
                    'color': 'deepskyblue' if "sea" in terrain else 'royalblue'
                }
            ))

        # ضباب / mist
        if any(w in weather for w in ["fog", "ضباب", "mist", "haze"]):
            elements.append(EnvironmentElement(
                type='fog_layer',
                position=[0, 0, -1.5],
                properties={
                    'density': random.uniform(0.25, 0.65),
                    'color': (240, 240, 255, int(180 * random.uniform(0.7, 1.0)))
                }
            ))

        # رياح / wind effect (لو موجودة)
        if any(w in weather for w in ["wind", "رياح", "storm", "عاصفة"]):
            elements.append(EnvironmentElement(
                type='wind',
                position=[width * 0.5, height * 0.3, 0],
                properties={
                    'direction': [random.uniform(-1, 1), random.uniform(-0.5, 0.5), 0],
                    'speed': random.uniform(4, 12)
                }
            ))

        # إضاءة رئيسية
        light_color = {
            'daytime': 'gold',
            'night': 'silver',
            'dusk': 'orange',
            'dawn': 'pink'
        }.get(lighting, 'white')

        elements.append(EnvironmentElement(
            type='main_light',
            position=[width * 0.6, height * 0.2, 15],
            properties={
                'intensity': 0.9 if "bright" in lighting else 0.6,
                'color': light_color,
                'type': 'directional' if "sun" in lighting else 'volumetric'
            }
        ))

        # ظلال عامة (تأثر بالمزاج)
        shadow_strength = 0.6 if "dramatic" in mood or "dark" in mood else 0.35
        elements.append(EnvironmentElement(
            type='ambient_shadow',
            position=[0, 0, -0.5],
            properties={'opacity': shadow_strength}
        ))

        return elements

    def _analyze_environment_description(self, text: str) -> dict[str, any]:
        """
        تحليل محسن لوصف البيئة – يدعم العربية والإنجليزية، يعطي وزنًا للكلمات، ويضمن إرجاع dict دائمًا.
        """
        if not text or not isinstance(text, str):
            logger.warning("الوصف فارغ أو غير صالح → إرجاع نتيجة افتراضية")
            return {
                "mood": "neutral",
                "weather": None,
                "time_of_day": None,
                "lighting_type": "natural",
                "key_elements": [],
                "scores": {}
            }

        lower_text = text.lower().strip()
        result = {
            "mood": "neutral",
            "weather": None,
            "time_of_day": None,
            "lighting_type": "natural",
            "key_elements": set(),
            "scores": defaultdict(int)
        }

        # قاموس الكلمات المفتاحية + وزنها (أكثر شمولاً ومنظم)
        keywords = {
            # أجواء / أسلوب
            "cyberpunk": (["cyber", "سايبر", "neon", "نيون", "futuristic", "مستقبلي", "dystopian", "سايبربانك"], 3),
            "forest": (["forest", "غابة", "woods", "woodland", "jungle", "أدغال", "trees", "أشجار", "wood"], 2),
            "space": (["space", "فضاء", "nebula", "سديم", "galaxy", "مجرة", "stars", "نجوم", "cosmic"], 3),
            "desert": (["desert", "صحراء", "sand", "رمال", "dunes", "كثبان", "arid"], 2),
            "urban": (["city", "مدينة", "street", "شارع", "skyscraper", "ناطحات سحاب", "urban", "downtown"], 2),
            "ocean": (["ocean", "محيط", "sea", "بحر", "beach", "شاطئ", "waves", "أمواج"], 2),
            "mountain": (["mountain", "جبل", "peak", "قمة", "hills", "تلال", "alpine", "summit"], 2),
            "magical": (["magic", "سحري", "enchanted", "مسحور", "fantasy", "خيالي", "mystical"], 2),
            "volcanic": (["volcano", "بركان", "lava", "حمم", "eruption", "ماغما"], 3),

            # طقس
            "rainy": (["rain", "مطر", "heavy rain", "مطر غزير", "pouring", "shower"], 3),
            "snowy": (["snow", "ثلج", "winter", "شتاء", "blizzard", "snowfall"], 2),
            "sunny": (["sunny", "مشمس", "bright", "ساطع", "clear sky", "سماء صافية"], 2),
            "foggy": (["fog", "ضباب", "mist", "ضباب خفيف", "haze", "غيم كثيف"], 3),
            "stormy": (["storm", "عاصفة", "thunder", "رعد", "lightning", "برق", "windy", "رياح قوية"], 3),
        }

        # حساب النقاط لكل فئة
        for category, (words, weight) in keywords.items():
            count = sum(lower_text.count(word) for word in words)
            if count > 0:
                result["scores"][category] = count * weight

        # اختيار المود (أعلى نقاط من فئات الأجواء)
        mood_categories = [cat for cat in keywords if cat in ["cyberpunk", "forest", "space", "desert", "urban", "ocean", "mountain", "magical", "volcanic"]]
        if any(cat in result["scores"] for cat in mood_categories):
            result["mood"] = max(
                (cat for cat in mood_categories if cat in result["scores"]),
                key=lambda k: result["scores"][k],
                default="neutral"
            )

        # اختيار الطقس (أعلى نقاط من فئات الطقس)
        weather_categories = ["rainy", "snowy", "sunny", "foggy", "stormy"]
        if any(cat in result["scores"] for cat in weather_categories):
            result["weather"] = max(
                (cat for cat in weather_categories if cat in result["scores"]),
                key=lambda k: result["scores"][k],
                default=None
            )

        # regex للوقت (أكثر دقة ومرونة)
        time_patterns = [
            r"\b(night|ليل|evening|مساء|midnight|منتصف الليل)\b",
            r"\b(morning|صباح|dawn|فجر|sunrise|شروق)\b",
            r"\b(day|نهار|afternoon|ظهر)\b",
            r"\b(dusk|غروب|sunset|غروب الشمس)\b"
        ]
        for pattern in time_patterns:
            match = re.search(pattern, lower_text, re.IGNORECASE)
            if match:
                result["time_of_day"] = match.group(0).lower()
                break

        # تحديد نوع الإضاءة بناءً على النتائج
        if result["mood"] == "cyberpunk":
            result["lighting_type"] = "neon_artificial"
        elif "night" in str(result["time_of_day"] or "") or "ليل" in lower_text:
            result["lighting_type"] = "moonlight" if result["weather"] != "foggy" else "dim_foggy"
        elif result["weather"] in ["rainy", "foggy", "stormy"]:
            result["lighting_type"] = "diffuse_overcast"
        elif result["mood"] == "volcanic":
            result["lighting_type"] = "red_lava_glow"
        elif "day" in str(result["time_of_day"] or "") or "نهار" in lower_text:
            result["lighting_type"] = "bright_natural"
        else:
            result["lighting_type"] = "natural"

        # جمع العناصر الرئيسية بدون تكرار
        result["key_elements"] = list({
            result["mood"],
            result["weather"] or "clear",
            result["time_of_day"] or "daytime",
            result["lighting_type"]
        })

        # تنظيف النتيجة النهائية
        result["key_elements"] = [e for e in result["key_elements"] if e and e != "neutral"]

        logger.info(f"تحليل نهائي للوصف: {result}")
        
        if result is None:
            result = {
                "mood": "neutral",
                "weather": None,
                "time_of_day": None,
                "lighting_type": "natural",
                "key_elements": [],
                "scores": {}
            }
        return result

    def _build_detailed_environment_prompt(
        self,
        prompt: str,
        analysis: dict,
        style_hints: str = "",
        is_video: bool = False
    ) -> str:
        """
        بناء prompt محسن للبيئة بناءً على التحليل + الـ prompt الأصلي
        ترجع string فقط (الـ enhanced prompt)
        """

        # نقطة البداية: نأخذ الـ prompt الأصلي كقاعدة
        base = prompt.strip()
        if not base:
            base = "بيئة طبيعية عالية الجودة، مشهد سينمائي"

        parts = [base]

        # 1. المزاج (mood)
        mood = analysis.get("mood", "neutral")
        if mood and mood.lower() != "neutral":
            parts.append(f"{mood} atmosphere, immersive mood, emotional depth")

        # 2. الطقس (weather)
        weather = analysis.get("weather", "") if isinstance(analysis, dict) else ""
        if weather:
            weather_map = {
                "rainy": "heavy rain, wet reflective surfaces, raindrops, moody atmosphere",
                "foggy": "dense fog, misty layers, soft diffused light, mysterious mood",
                "stormy": "dramatic storm clouds, lightning in distance, wind-swept trees",
                "snowy": "falling snow, frosty details, cold blue tones, winter serenity",
                "sunny": "bright golden sunlight, vivid colors, lens flare, uplifting mood"
            }
            parts.append(weather_map.get(weather.lower(), f"{weather} weather conditions, atmospheric effects"))

        # 3. وقت اليوم + نوع الإضاءة
        time_of_day = analysis.get("time_of_day", "")
        lighting_type = analysis.get("lighting_type", "natural")

        if time_of_day:
            time_map = {
                "night": "night scene, moonlight, deep shadows, starry sky if clear",
                "day": "daytime, bright natural sunlight, crisp details, vibrant colors",
                "dawn": "golden dawn, soft warm light, long gentle shadows",
                "dusk": "golden hour sunset, warm orange and purple sky, cinematic glow"
            }
            parts.append(time_map.get(time_of_day.lower(), f"{time_of_day} lighting"))

        if lighting_type and lighting_type.lower() != "natural":
            parts.append(f"{lighting_type} lighting, dramatic cinematic illumination, volumetric light")

        # 4. إضافة style_hints (إذا وُجدت)
        if style_hints and style_hints.strip():
            # نفترض أن style_hints نص مفصول بفواصل أو جملة واحدة
            parts.append(style_hints.strip())

        # 5. إذا كان فيديو → إضافة حركة خفيفة
        if is_video:
            parts.append("subtle camera movement, gentle parallax, wind-swayed elements, cinematic animation")

        # 6. معززات الجودة العامة (دائمًا في النهاية)
        parts.append("ultra detailed environment, 8k resolution, cinematic composition")
        parts.append("professional landscape photography, sharp focus, depth of field")
        parts.append("masterpiece, best quality, highly atmospheric")

        # دمج الكل بفواصل
        enhanced_prompt = ", ".join(filter(None, parts))

        return enhanced_prompt

    def _execute_environment_design(
        self,
        enhanced_prompt: str,
        resolution: tuple,
        is_video: bool,
        analysis: dict,
        **kwargs
    ) -> dict:
        """
        تصميم البيئة فقط (بدون توليد صورة فعلية)
        ينتج قائمة من العناصر مع مواقعها وخصائصها
        ثم يخزنها في الذاكرة عبر memory_manager
        """
        width, height = resolution
        elements = []  # قائمة العناصر المصممة

        mood = analysis.get("mood", "neutral")
        weather = analysis.get("weather", "") if isinstance(analysis, dict) else ""
        time_of_day = analysis.get("time_of_day")

        # ────────────────────────────────────────────────
        # تحديد الستايل والألوان الأساسية (للاستخدام في الخصائص)
        # ────────────────────────────────────────────────
        style = {
            "bg_color": (135, 206, 235),  # افتراضي
            "accent_color": (255, 255, 255),
            "secondary_color": (200, 200, 200)
        }

        if mood == "cyberpunk":
            style["bg_color"] = (10, 5, 25)
            style["accent_color"] = (0, 255, 255)
            style["secondary_color"] = (255, 0, 200)

        elif "forest" in enhanced_prompt.lower() or "غابة" in enhanced_prompt.lower():
            style["bg_color"] = (20, 50, 20)
            style["accent_color"] = (80, 180, 60)

        elif "space" in enhanced_prompt.lower() or "فضاء" in enhanced_prompt.lower():
            style["bg_color"] = (5, 5, 25)
            style["accent_color"] = (200, 220, 255)

        # ────────────────────────────────────────────────
        # تصميم العناصر (بدون رسم، فقط بيانات)
        # ────────────────────────────────────────────────
        if mood == "cyberpunk":
            # خطوط نيون أفقية (كعناصر خطية)
            for y in range(0, height, 80):
                elements.append({
                    "type": "neon_line",
                    "position": [[0, y], [width, y]],
                    "color": style["accent_color"],
                    "width": 2
                })
                elements.append({
                    "type": "neon_line",
                    "position": [[0, y+3], [width, y+3]],
                    "color": style["secondary_color"],
                    "width": 1
                })

            # دوائر نيون (كعناصر دائرية)
            for _ in range(40):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                r = np.random.randint(5, 35)
                elements.append({
                    "type": "neon_circle",
                    "center": [x, y],
                    "radius": r,
                    "color": style["accent_color"],
                    "width": 2
                })

        elif "forest" in enhanced_prompt.lower() or "غابة" in enhanced_prompt.lower():
            # أشجار (كمثلثات)
            for _ in range(60):
                x = np.random.randint(0, width)
                base_y = np.random.randint(height//2, height)
                h = np.random.randint(80, 220)
                elements.append({
                    "type": "tree",
                    "points": [
                        [x, base_y],
                        [x - h//3, base_y - h],
                        [x + h//3, base_y - h]
                    ],
                    "fill_color": style["accent_color"],
                    "outline_color": (20, 80, 20)
                })

        elif "space" in enhanced_prompt.lower() or "فضاء" in enhanced_prompt.lower():
            # نجوم (نقاط)
            star_count = width * height // 150
            stars = np.random.randint(0, [width, height], size=(star_count, 2))
            for x, y in stars:
                elements.append({
                    "type": "star",
                    "position": [int(x), int(y)],
                    "color": style["accent_color"],
                    "size": 1
                })

            # مجرات صغيرة (دوائر ضبابية)
            for _ in range(6):
                cx = np.random.randint(width//4, width*3//4)
                cy = np.random.randint(height//4, height*3//4)
                r = np.random.randint(40, 140)
                elements.append({
                    "type": "nebula_ring",
                    "center": [cx, cy],
                    "radius": r,
                    "outline_color": (180, 180, 220),
                    "width": 1
                })

        else:
            # شمس/قمر افتراضي
            sun_x = width - 120
            sun_y = 100
            elements.append({
                "type": "sun_or_moon",
                "center": [sun_x, sun_y],
                "radius": 60,
                "fill_color": (255, 240, 150)
            })

        # ────────────────────────────────────────────────
        # تخزين التصميم في الذاكرة
        # ────────────────────────────────────────────────
        prompt_hash = self.memory_manager.get_prompt_hash(enhanced_prompt)
        stored = self.memory_manager.store_environment_elements(
            prompt_hash=prompt_hash,
            elements=elements,
            analysis=analysis,
            extra_metadata={"resolution": resolution, "is_video": is_video}
        )

        # ────────────────────────────────────────────────
        # الإرجاع (بدون صورة)
        # ────────────────────────────────────────────────
        return {
            "success": stored,
            "elements": elements,               # ← أضف هذا
            "elements_count": len(elements),
            "prompt_hash": prompt_hash,
            "message": f"تم تصميم بيئة {mood} بـ {len(elements)} عنصر (محفوظة في الذاكرة)",
            "generated_with": "procedural_metadata_only"
        }
         
    def _prepare_export_path(self, output_dir: str, base_name_prefix: str, extension: str) -> Path:
        """دالة مساعدة مشتركة لتحضير المسار والمجلد"""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name_prefix}_{timestamp}.{extension}"
        full_path = out_dir / filename
        
        return full_path
      
    def export_design_result(
        self,
        result: GenerationResult,
        export_format: str = "json",
        save_to_disk: bool = False,
        output_dir: str = "exported_designs",
        include_preview: bool = False
    ) -> Dict[str, Any]:
        """
        تصدير نتيجة التصميم كبيانات منظمة (dict أو json محفوظ)
        """
        if not result or not result.success:
            return {
                "success": False,
                "error": result.message if result else "No result provided",
                "exported_at": datetime.utcnow().isoformat()
            }

        exported = {
            "success": result.success,
            "specialization": result.specialization,
            "exported_at": datetime.utcnow().isoformat(),
            "total_time_seconds": round(result.total_time, 3) if result.total_time else 0.0,
            "stage_times": result.stage_times or {},
            "is_video": result.is_video,
        }

        # استخراج البيانات بحذر
        out_data = result.output_data or {}
        exported["metadata"] = out_data.get("metadata", {})
        exported["notes"] = out_data.get("notes", [])  # آمن حتى لو غير موجود

        # الـ prompt المحسن إن وجد
        exported["enhanced_prompt"] = (
            out_data.get("enhanced_prompt") or
            out_data.get("prompt") or
            out_data.get("prompt_used", "")
        )

        # المعاينة إذا طُلب وموجودة
        if include_preview:
            for key in ["preview_path", "path", "layer_path", "output_path", "design_path"]:
                p = out_data.get(key)
                if p and Path(p).is_file():
                    exported["preview_path"] = str(Path(p).resolve())
                    break

        exported["assets"] = {
            k: str(Path(v).resolve()) 
            for k, v in out_data.items() 
            if isinstance(v, (str, Path)) and "path" in k.lower() and Path(v).exists()
        }

        result_dict = exported.copy()

        if export_format.lower() == "json":
            json_str = json.dumps(exported, ensure_ascii=False, indent=2)
            
            if save_to_disk:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"design_export_{ts}.json"
                full_path = Path(output_dir) / filename
                full_path.write_text(json_str, encoding="utf-8")
                result_dict["saved_file"] = str(full_path.resolve())
                result_dict["json_string"] = None  # لا حاجة لتكرار النص الكبير
            else:
                result_dict["json_string"] = json_str

        # لاحقاً يمكن إضافة yaml أو msgpack أو غيره

        return result_dict

    def export_design_to_file(
        self,
        result: EnvironmentDesignResult,
        format_type: str = "json",
        output_dir: str = "exported_env_designs",
        include_preview: bool = False
    ) -> str:
        """
        تصدير نتيجة تصميم بيئي كملف (json أو pickle حالياً)
        ترجع مسار الملف المحفوظ
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_name = f"env_design_{ts}"
        
        if format_type.lower() == "json":
            filename = f"{base_name}.json"
            full_path = Path(output_dir) / filename
            
            exported_data = {
                "success": result.success,
                "message": result.message,
                "design_time_seconds": result.design_time_seconds,
                "metadata": result.metadata or {},
                "exported_at": datetime.utcnow().isoformat(),
            }
            
            # تحويل العناصر بحذر
            if hasattr(result, "elements") and result.elements:
                try:
                    exported_data["elements"] = [el.to_dict() if hasattr(el, "to_dict") else vars(el) for el in result.elements]
                except Exception as e:
                    logger.warning(f"فشل تحويل elements: {e}")
                    exported_data["elements"] = [str(el) for el in result.elements]
            
            if include_preview and hasattr(result, "preview_path") and result.preview_path:
                exported_data["preview_path"] = str(Path(result.preview_path).resolve())
            
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(exported_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"تم تصدير التصميم (JSON) → {full_path}")
            return str(full_path.resolve())

        elif format_type.lower() == "pickle":
            filename = f"{base_name}.pkl"
            full_path = Path(output_dir) / filename
            with open(full_path, 'wb') as f:
                pickle.dump(result, f)
            logger.info(f"تم تصدير التصميم (Pickle) → {full_path}")
            return str(full_path.resolve())

        else:
            raise ValueError(f"الصيغة غير مدعومة: {format_type} (استخدم json أو pickle)")
      
    def _export_to_glb(self, task_data: Dict, output_dir: str = "exported_3d_scenes") -> Optional[str]:
        """تصدير بيئة بسيطة كـ GLB باستخدام trimesh (أسهل وأكثر استقراراً)"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ts = int(perf_counter() * 1000)
        path = Path(output_dir) / f"environment_scene_{ts}.glb"

        meshes = []

        for plane in task_data.get("planes", []):
            label = plane["label"].lower()
            z = plane.get("z", 0.0) * 10.0

            if "ground" in label or "sand" in label or "desert" in label:
                verts = np.array([
                    [-100, z, -100],
                    [ 100, z, -100],
                    [ 100, z,  100],
                    [-100, z,  100]
                ])
                faces = np.array([[0,1,2], [0,2,3]])
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                mesh.visual.vertex_colors = [100, 180, 80, 255]   # أخضر/بني
                meshes.append(mesh)

            elif "mountain" in label:
                base_z = z - 2
                peak_z = z + 18
                verts = np.array([
                    [-40, base_z, -40],
                    [ 40, base_z, -40],
                    [  0, peak_z,   0],
                    [-40, base_z,  40],
                    [ 40, base_z,  40]
                ])
                faces = np.array([
                    [0,1,2], [0,2,3], [1,4,2], [3,2,4]
                ])
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                mesh.visual.vertex_colors = [90, 90, 95, 255]   # رمادي صخري
                meshes.append(mesh)

        if not meshes:
            logger.warning("لا توجد planes للتصدير إلى GLB")
            return None

        # دمج كل الميشات في مشهد واحد
        scene = trimesh.Scene()
        for m in meshes:
            scene.add_geometry(m)

        scene.export(str(path), file_type='glb')

        logger.info(f"✅ تم تصدير المشهد ثلاثي الأبعاد بنجاح → {path}")
        return str(path.resolve())
    
@dataclass
class TerrainDesignResult:
    success: bool = False
    heightmap: Optional[np.ndarray] = None          # خريطة الارتفاع (2D array)
    base_texture: Optional[Image.Image] = None      # الصورة الأساسية (ألوان)
    normal_map: Optional[Image.Image] = None        # اختياري – normal map
    metadata: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    design_time_seconds: float = 0.0
    
class terrain_design_engine:
    def __init__(self):
        self.specialization = {
            "domain": "terrain",
            "supported_aspects": [
                "heightmap", "base_texture", "erosion", "rockiness", 
                "slope", "river_paths", "biome_influence"
            ]
        }

    def design(
        self,
        description: str,
        resolution: tuple = (1024, 1024),   # عادة مربع للـ heightmap
        seed: int = 42,
        **kwargs
    ) -> TerrainDesignResult:
        """
        يرجع:
        - heightmap: np.array (float32, normalized 0..1 أو مقياس حقيقي)
        - base_color: PIL.Image أو np.array
        - metadata: slopes, water_mask, rock_areas, إلخ
        """
        # 1. تحليل الوصف (جبال، سهل، صحراء، بركاني، ...)
        analysis = self._analyze_terrain(description)

        # 2. توليد heightmap (perlin/simplex noise + modifiers)
        heightmap = self._generate_heightmap(
            resolution=resolution,
            scale=analysis.get("scale", 100.0),
            octaves=analysis.get("octaves", 6),
            seed=seed
        )

        # 3. إنشاء base texture (ألوان أساسية حسب النوع)
        base_texture = self._generate_base_texture(heightmap, analysis)

        # 4. إضافة تفاصيل إضافية (اختياري: erosion, rivers, ...)
        if "river" in description.lower() or "نهر" in description:
            base_texture = self._add_river_paths(base_texture, heightmap)

        return TerrainDesignResult(
            success=True,
            heightmap=heightmap,
            base_texture=base_texture,
            metadata=analysis,
            message="تم تصميم التضاريس"
        )
        
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    engine = environment_design_engine()

    # اختبار بسيط واحد (تصميم خام)
    print("\n=== اختبار تصميم بيئة واحد ===")
    result = engine.design("غابة مع نهر ورياح وإضاءة صباحية")

    if result.success:
        print("→ نجاح!")
        print(f"  • عدد العناصر المصممة: {len(result.elements)}")
        if result.elements:
            print(f"  • مثال عنصر أول: {result.elements[0]}")
            print(f"  • مثال عنصر ثاني: {result.elements[1] if len(result.elements) > 1 else 'لا يوجد'}")
        else:
            print("  • لا عناصر تم تصميمها")
        print(f"  • الوقت: {result.design_time_seconds:.2f} ثانية")
        print(f"  • مخزن في الذاكرة تحت hash: {engine.memory_manager.get_prompt_hash('غابة مع نهر ورياح وإضاءة صباحية')}")
    else:
        print("→ فشل")
        print(f"  • الرسالة: {result.message}")

    # الاختبارات الأربعة الرئيسية (بدون توقع layer أو image)
    test_cases = [
        "مدينة سايبربانك نيون تحت المطر في الليل، أضواء ملونة، جو مستقبلي",
        "غابة مطيرة خضراء كثيفة في الصباح الباكر، ضباب خفيف، أشعة شمس تخترق الأشجار",
        "فضاء عميق مع سديم أرجواني ونجوم، مجرة بعيدة، أسلوب سينمائي",
        "شاطئ استوائي عند الغروب، ماء فيروزي، نخيل، سماء برتقالية"
    ]

    for i, desc in enumerate(test_cases, 1):
        print(f"\n┌─────────────── اختبار {i} ───────────────┐")
        print(f"الوصف: {desc}")

        result = engine.design(
            description=desc,
            resolution=(1280, 720),  # غيّر target_resolution إلى resolution لو الدالة تستخدمه كده
            style_hints=["cinematic", "ultra detailed"],
            is_video=False
        )

        if result.success:
            print("→ نجاح!")
            print(f"  • عدد العناصر: {len(result.elements)}")
            print(f"  • الوقت: {result.design_time_seconds:.2f} ثانية")
            print(f"  • مخزن تحت hash: {engine.memory_manager.get_prompt_hash(desc)}")
            if result.elements:
                print(f"  • أول 3 عناصر (مثال):")
                for el in result.elements[:3]:
                    print(f"    - {el}")
        else:
            print("→ فشل")
            print(f"  • الرسالة: {result.message}")

        print("└──────────────────────────────────────────┘")