# traditional_design_engine.py

import numpy as np
from dataclasses import dataclass, field   # ← أضف dataclass لو مش موجود
from typing import Dict, Any, Optional, List
from datetime import datetime
from PIL import Image, ImageDraw
import random
import logging
import math
from pathlib import Path
from time import perf_counter
from PIL import Image, ImageDraw, ImageFont

# الاستيراد الوحيد المهم (من الكور)
from Core_Image_Generation_Engine import CoreImageGenerationEngine
from memory_manager import GenerativeMemoryManager
from Image_generation import CoreImageGenerationEngine  # ← استيراد النواة
from generation_result import GenerationResult

print("تم تحميل traditional_design_engine.py")

logger = logging.getLogger(__name__)

from typing import List, Dict, Any, Optional
from copy import deepcopy

@dataclass
class TraditionalDesignResult:
    success: bool
    enhanced_prompt: str
    metadata: Dict[str, Any]
    layer_type: str = "foreground"
    message: str = ""
    design_time_seconds: float = 0.0
    
class traditionalDesignEngine(CoreImageGenerationEngine):
    """
    محرك متخصص في تحليل وتحسين وصف الكائنات التقليدية / العضوية فقط.
    الإخراج الرئيسي: dict نصي محسن + metadata
    لا يتدخل في توليد الصور النهائية (دور Final_Generation)
    """

    def __init__(self):
        super().__init__()
        self.specialization = {
            "name": "traditional_design",
            "type": "foreground",
            "description": "تصميم شخصيات ومخلوقات عضوية، تفاصيل تشريحية، تعبيرات عاطفية"
        }
        self.memory_manager = GenerativeMemoryManager()
        self.memory_manager.context_type = "traditional"
        logger.info("[TraditionalEngine] تم التهيئة – تصميم نصي فقط (foreground)")
        
    def _create_simple_image(self, prompt: str, target_size: tuple) -> Image.Image:
        """
        تنفيذ placeholder بسيط للدالة المطلوبة من الأب
        (هتنشئ صورة فارغة مع نص مؤقت عشان الكلاس يصير concrete)
        """
        img = Image.new("RGB", target_size, color=(220, 220, 220))  # رمادي فاتح
        draw = ImageDraw.Draw(img)
        draw.text((50, target_size[1]//2), f"Traditional Placeholder\n{prompt[:50]}...", fill=(0, 0, 0))
        return img
    
    def _get_specialization_config(self) -> Dict[str, Any]:
        return {"name": "traditional", "description": "تصميم عضوي"}

    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        return {"entities": prompt.split(), "style": "organic"}

    def _enhance_traditional_prompt(self, prompt: str) -> Dict[str, Any]:
        lower = prompt.lower()

        metadata = {
            "entities": self._extract_creatures(lower),
            "mood": self._detect_mood(lower),
            "lighting": self._suggest_lighting(lower),
            "anatomy": "highly detailed anatomy, realistic proportions, emotional expression",
            "details": "intricate skin texture, individual hair strands, fabric folds, subtle imperfections",
            "timestamp": datetime.now().isoformat()
        }

        enhanced = (
            f"{prompt}, {metadata['anatomy']}, {metadata['details']}, "
            f"{metadata['mood']} atmosphere, {metadata['lighting']}, "
            "masterpiece, best quality, ultra detailed, cinematic, 8k"
        )

        return {"prompt": enhanced, "metadata": metadata}

    def receive_input(self, prompt: str) -> bool:
        """
        تلقي prompt جديد مع التحقق الأساسي
        """
        if not isinstance(prompt, str) or not prompt.strip():
            logger.warning("Prompt غير صالح أو فارغ")
            return False

        stripped = prompt.strip()
        self.append_prompt_chunk(stripped)
        logger.info(f"[{self.specialization.get('name', 'unknown')}] received: {stripped[:60]}...")
        return True
    
    def _cleanup_temp_files(self):
        """
        حذف أي ملفات مؤقتة تم إنشاؤها (اختياري، حسب الحاجة)
        """
        for file_path in self.temp_files:
            try:
                Path(file_path).unlink()
                logger.info(f"تم حذف الملف المؤقت: {file_path}")
            except Exception as e:
                logger.warning(f"فشل حذف الملف المؤقت {file_path}: {e}")
        self.temp_files.clear()

    def append_prompt_chunk(self, chunk: str):
        """
        إضافة جزء من الـ prompt إلى المنفذ (بسيط ونظيف)
        """
        if not hasattr(self, "input_port"):
            self.input_port = []
        self.input_port.append(chunk.strip())
        logger.debug(f"Input chunk appended: {chunk[:50]}...")

    def add_task(self, task_name: str, complexity: float = 1.0, dependencies: Optional[List[str]] = None):
        """
        إضافة مهمة وصفية بسيطة (اختياري – للتتبع فقط، مش بصري)
        """
        if not hasattr(self, "tasks"):
            self.tasks = []
        self.tasks.append({
            "name": task_name,
            "complexity": complexity,
            "dependencies": dependencies or []
        })
        logger.debug(f"Added task: {task_name} (complexity: {complexity})")

    def _validate_specialization(self):
        pass

    def _initialize_units(self):
        pass

    def _initialize_memory_manager(self):
        self.memory_manager = GenerativeMemoryManager()
        self.memory_manager.context_type = "traditional"

    def _initialize_additional_state(self):
        self.temp_files = []

    def _log_specialization_details(self):
        logger.debug(f"تفاصيل تخصص: {self.specialization}")

    def _log_initial_state(self):
        logger.debug("حالة تهيئة تمت")

    def _run_initial_diagnostics(self):
        logger.debug("تشخيص أولي - ناجح")

    def _integrate(self, task_data: Dict) -> Dict:
        """
        دالة تكامل خفيفة جدًا (نصية فقط)
        """
        task_data.setdefault("entities", [])
        task_data.setdefault("summary", {})
        task_data.setdefault("warnings", [])
        task_data.setdefault("metadata", {})

        task_data["metadata"].update({
            "entities_count": len(task_data["entities"]),
            "mood": task_data.get("mood", "neutral"),
            "integration_done": True
        })

        task_data["summary"] = {
            "note": "تم التكامل النصي بنجاح"
        }

        return task_data

    def _post_process(self, task_data: Dict) -> Dict[str, Any]:
        return {"processed": True, "message": "post-processing placeholder"}

    def _render(self, task_data: Dict, is_video: bool = False) -> float:
        return 1.2

    def generate_layer(
        self,
        prompt: str,
        target_size: tuple = (1024, 1024),
        is_video: bool = False,
        force_refresh: bool = False,
        **kwargs
    ) -> GenerationResult:
        """
        توليد طبقة تقليدية حقيقية (صورة فعلية مع رسم بسيط للشخصية/الكائن)
        """
        start_time = perf_counter()
        stage_times = {}

        try:
            # 1. تحليل وتحسين الـ prompt (نحتفظ به كما هو)
            t_start = perf_counter()
            analysis = self._analyze_prompt(prompt) or {}
            enhanced_dict = self._enhance_traditional_prompt(prompt)
            enhanced_prompt = enhanced_dict.get("prompt", prompt)
            metadata = enhanced_dict.get("metadata", {})
            stage_times["analysis"] = perf_counter() - t_start

            # 2. إنشاء صورة شفافة كقاعدة
            t_start = perf_counter()
            img = Image.new("RGBA", target_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            stage_times["init_canvas"] = perf_counter() - t_start

            # 3. توليد عناصر قابلة للرسم
            t_start = perf_counter()
            elements = self._generate_traditional_elements(enhanced_prompt, analysis, target_size)
            stage_times["generate_elements"] = perf_counter() - t_start

            # 4. رسم العناصر على الصورة
            t_start = perf_counter()
            w, h = target_size
            for elem in elements:
                if elem["type"] == "main_subject":
                    # رسم جسم بسيط (مستطيل + رأس دائري + شعر)
                    x, y = w // 2, h // 2
                    # جسم
                    draw.rectangle((x-120, y-20, x+120, y+280), fill=(180, 140, 100), outline=(60, 40, 20), width=3)
                    # رأس
                    draw.ellipse((x-80, y-180, x+80, y-20), fill=(240, 220, 200), outline=(80, 60, 40), width=3)
                    # شعر بسيط (خطوط عشوائية)
                    for _ in range(20):
                        sx = x + random.randint(-70, 70)
                        sy = y - 180 + random.randint(-40, 0)
                        ex = sx + random.randint(-40, 40)
                        ey = sy - random.randint(60, 140)
                        draw.line((sx, sy, ex, ey), fill=(100, 60, 40), width=4)

                elif elem["type"] == "detail":
                    x, y = elem["center"]
                    r = elem["radius"]
                    draw.ellipse((x-r, y-r, x+r, y+r), fill=elem["color"])

            stage_times["draw_elements"] = perf_counter() - t_start

            # 5. حفظ الصورة
            t_start = perf_counter()
            output_dir = Path("output/traditional")
            output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            preview_path = output_dir / f"traditional_layer_{ts}.png"
            img.save(preview_path, "PNG")
            stage_times["save_image"] = perf_counter() - t_start

            total_time = perf_counter() - start_time

            return GenerationResult(
                success=True,
                message=f"تم توليد طبقة تقليدية حقيقية ({len(elements)} عنصر مرسوم)",
                total_time=total_time,
                stage_times=stage_times,
                specialization=self.specialization,
                is_video=is_video,
                output_data={
                    "original_prompt": prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "preview_path": str(preview_path.resolve()),
                    "image_size": target_size,
                    "elements_drawn": len(elements),
                    "metadata": metadata,
                    "file_created": str(preview_path)
                }
            )

        except Exception as e:
            logger.exception("فشل في generate_layer")
            return GenerationResult(
                success=False,
                message=f"خطأ: {str(e)}",
                total_time=perf_counter() - start_time,
                specialization=self.specialization
            )
            
    # ────────────────────────────────────────────────
    #     توليد عناصر قابلة للرسم (الدالة المعدلة)
    # ────────────────────────────────────────────────

    def _generate_traditional_elements(
        self,
        enhanced_prompt: str,
        analysis: dict,
        resolution: tuple
    ) -> list:
        """
        إرجاع قائمة عناصر مرسومة فعليًا (موقع + خصائص رسم)
        """
        w, h = resolution
        elements = []

        # مثال: شخصية رئيسية في المنتصف
        elements.append({
            "type": "main_subject",
            "bbox": (w//2 - 180, h//2 - 300, w//2 + 180, h//2 + 300),
            "color": (240, 220, 200),  # لون بشرة تقريبي
            "outline": (80, 60, 40),
            "details": ["eyes", "hair", "clothing"]
        })

        # إضافة تفاصيل عشوائية حول الشخصية
        for _ in range(random.randint(4, 9)):
            x = random.randint(w//3, w*2//3)
            y = random.randint(h//4, h*3//4)
            size = random.randint(20, 80)
            elements.append({
                "type": "detail",
                "center": (x, y),
                "radius": size,
                "color": random.choice([(200,180,160), (160,140,120), (100,80,60)]),
                "label": random.choice(["hair strand", "fabric fold", "skin texture", "jewelry"])
            })

        # إضافة إكسسوارات أو تأثيرات
        if "dragon" in enhanced_prompt.lower() or "تنين" in enhanced_prompt:
            elements.append({
                "type": "aura",
                "center": (w//2, h//2),
                "radius": 220,
                "color": (80, 220, 255, 120),
                "glow": True
            })

        return elements

    # ────────────────────────────────────────────────
    #          رسم عنصر واحد على الصورة
    # ────────────────────────────────────────────────

    def _draw_element(self, draw: ImageDraw, elem: dict, resolution: tuple):
        w, h = resolution

        if elem["type"] == "main_subject":
            # رسم جسم بسيط (مستطيل + رأس)
            x1, y1, x2, y2 = elem["bbox"]
            draw.rectangle((x1, y1, x2, y2), fill=elem["color"], outline=elem["outline"], width=4)
            # رأس
            draw.ellipse((x1+60, y1-60, x2-60, y1+60), fill=(240,220,200), outline=(80,60,40))

        elif elem["type"] == "detail":
            x, y = elem["center"]
            r = elem["radius"]
            draw.ellipse((x-r, y-r, x+r, y+r), fill=elem["color"])

        elif elem["type"] == "aura":
            x, y = elem["center"]
            r = elem["radius"]
            draw.ellipse((x-r, y-r, x+r, y+r), outline=(80,220,255), width=12)
            # تأثير glow بسيط
            for i in range(3):
                alpha = 60 - i*15
                draw.ellipse((x-r-i*20, y-r-i*20, x+r+i*20, y+r+i*20),
                             outline=(80,220,255,alpha), width=8)

    def _extract_creatures(self, text: str) -> List[str]:
        # قائمة كلمات مفتاحية للكائنات
        creature_keywords = {
            "dragon", "elf", "orc", "goblin",
            "girl", "boy", "woman", "man", "child", "baby", "horse", "dragon", "lion", "wolf",
            "cat", "dog", "bird", "fish"
        }

        words = text.split()
        entities = set()
        for word in words:
            clean = word.strip(".,!?;:()[]{}'\"")
            if clean in creature_keywords:
                entities.add(clean)

        return list(entities)

    def _detect_mood(self, text: str) -> str:
        mood_indicators = {
            "غامض": "mysterious", "ظلام": "dark", "ضباب": "foggy", "رعب": "horror",
            "هادئ": "calm", "جميل": "beautiful", "رقيق": "delicate", "ساحر": "enchanted",
            "ملحمي": "epic", "قوي": "powerful", "مهيب": "majestic"
        }

        words = text.split()
        for word in words:
            if word in mood_indicators:
                return mood_indicators[word]
        return "neutral"

    def _suggest_lighting(self, text: str) -> str:
        if "ليل" in text or "night" in text:
            return "moonlight, dramatic shadows"
        elif "صباح" in text or "morning" in text:
            return "soft golden light, warm tones"
        return "natural daylight"

# ────────────────────────────────────────────────
#              اختبار سريع (اختياري)
# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("traditional_design_engine.py تم تحميله كـ __main__")
    engine = traditionalDesignEngine()
    print("تم إنشاء الكائن بنجاح")
    print("generate_layer موجود؟", hasattr(engine, "generate_layer"))
    print("═" * 70)
    print("اختبار Traditional Design Engine")
    print("═" * 70)

    engine = traditionalDesignEngine()

    # prompt مناسب للـ traditional
    engine.receive_input("a majestic dragon creature in enchanted misty forest with glowing aura and organic flow")

    # أضف مهام بسيطة
    engine.add_task("main_creature", complexity=4.8)
    engine.add_task("forest_background", complexity=3.5)
    engine.add_task("aura_effect", complexity=2.7)

    result = engine.generate_layer("test prompt")

    print("\nنتيجة التوليد:")
    print(f"نجاح: {result.success}")
    print(f"رسالة: {result.message}")
    print(f"الوقت الكلي: {result.total_time:.2f} ث")
    if result.output_data and "preview_path" in result.output_data:
        print(f"مسار المعاينة: {result.output_data['preview_path']}")