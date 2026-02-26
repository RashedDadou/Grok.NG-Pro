# geometric_design_engine.py
"""
محرك التوليد الخاص بالتصاميم الهندسية
(أشكال، أنماط، تناظر، هياكل رياضية، تصاميم دقيقة ومتكررة...)
"""

import random
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from time import perf_counter

from memory_manager import GenerativeMemoryManager
from generation_result import GenerationResult

logger = logging.getLogger(__name__)

class geometric_design_engine:
    """
    محرك متخصص في تحليل وتحسين وصف الأشكال الهندسية / الهيكلية فقط.
    الإخراج الرئيسي: dict نصي محسن + metadata
    لا يتدخل في توليد الصور النهائية (دور Final_Generation)
    """

    def __init__(self):
        self.specialization = {
            "name": "geometric_design",
            "type": "midground",
            "description": "تصاميم هندسية، أشكال، تناظر، أنماط متكررة، هياكل رياضية، فركتل، شبكات",
            "keywords": [
                # إنجليزي
                "shape", "structure", "design", "pattern", "form", "symmetry", "polygon", "grid", "fractal",
                "hex", "honeycomb", "hexagonal", "spiral", "golden", "geometry", "abstract",
                # عربي
                "شكل", "هيكل", "تصميم", "نمط", "تناظر", "مضلع", "شبكة", "فركتل", "سداسي", "دوامة", "ذهبي", "هندسي"
            ]
        }
        self.memory_manager = GenerativeMemoryManager()
        self.memory_manager.context_type = "geometric"
        self.input_port: List[str] = []  # لتجميع أجزاء الـ prompt
        self.temp_files: List[str] = []  # لتتبع الملفات المؤقتة (إذا لزم)
        logger.info("[GeometricEngine] تم تهيئة المحرك – تصميم نصي فقط (midground)")

    def receive_input(self, prompt: str) -> bool:
        """
        تلقي prompt جديد مع التحقق الأساسي
        """
        if not isinstance(prompt, str) or not prompt.strip():
            logger.warning("Prompt غير صالح أو فارغ")
            return False

        stripped = prompt.strip()
        self.input_port.append(stripped)
        logger.info(f"[{self.specialization['name']}] received: {stripped[:60]}...")
        return True

    def generate_layer(
        self,
        prompt: str,
        target_size: tuple = (1024, 1024),
        is_video: bool = False,
        force_refresh: bool = False,
        **kwargs
    ) -> GenerationResult:
        """
        توليد تصميم طبقة هندسية (metadata + elements نصية + enhanced prompt)
        مشابه لـ environment لكن متخصص في الأشكال والأنماط
        """
        start_time = perf_counter()
        stage_times = {}

        try:
            # 1. تحليل الوصف – حماية قصوى
            t_start = perf_counter()
            analysis_raw = self._analyze_prompt(prompt)

            if analysis_raw is None or not isinstance(analysis_raw, dict):
                logger.warning(f"[Geometric] تحليل رجع {type(analysis_raw).__name__ if analysis_raw is not None else 'None'} → قاموس فارغ")
                analysis = {}
            else:
                analysis = analysis_raw

            stage_times["analysis"] = perf_counter() - t_start

            # 2. بناء enhanced prompt هندسي
            t_start = perf_counter()
            enhanced_dict = self._enhance_geometric_prompt(prompt)
            enhanced_prompt = enhanced_dict.get("prompt", prompt)
            metadata = enhanced_dict.get("metadata", {})

            # ضمان أن enhanced_prompt ليس فارغ
            if not enhanced_prompt.strip():
                enhanced_prompt = prompt + ", precise geometric pattern, ultra detailed"

            stage_times["prompt_enhance"] = perf_counter() - t_start

            # 3. توليد عناصر نصية بسيطة (بدل EnvironmentElement لأنها هندسية)
            t_start = perf_counter()
            elements = self._generate_geometric_elements(enhanced_prompt, analysis, target_size)
            stage_times["design_elements"] = perf_counter() - t_start

            # 4. الهاش
            prompt_hash = (
                self.memory_manager.get_prompt_hash(prompt)
                if hasattr(self, "memory_manager") and self.memory_manager
                else f"no_memory_{int(perf_counter()*1000)}"
            )

            total_time = perf_counter() - start_time

            logger.info(f"[Geometric Layer] نجاح – {len(elements)} عنصر | hash: {prompt_hash[:12]}...")

            return GenerationResult(
                success=True,
                message=f"تصميم هندسي ناجح ({len(elements)} عنصر)",
                total_time=total_time,
                stage_times=stage_times,
                specialization=self.specialization,
                is_video=is_video,
                output_data={
                    "original_prompt": prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "analysis": analysis,
                    "elements": elements,
                    "metadata": {
                        **metadata,
                        "element_count": len(elements),
                        "resolution_suggested": target_size,
                        "is_video": is_video,
                        "force_refresh": force_refresh
                    },
                    "prompt_hash": prompt_hash,
                    "layer_type": "geometric",
                    # لتوافق مع Final_Generation (حتى لو placeholder)
                    "preview_path": None,
                    "path": None
                }
            )

        except Exception as e:
            logger.exception("فشل توليد طبقة هندسية")
            total_time = perf_counter() - start_time
            return GenerationResult(
                success=False,
                message=f"خطأ في تصميم هندسي: {str(e)}",
                total_time=total_time,
                stage_times=stage_times,
                specialization=self.specialization
            )
            
    def _generate_geometric_elements(self, enhanced_prompt: str, analysis: dict, resolution: tuple) -> list:
        """
        توليد عناصر هندسية نصية بسيطة (للتوافق مع الهيكل)
        """
        elements = []
        main_subject = analysis.get("main_subject", "geometric pattern")
        symmetry = analysis.get("symmetry", "medium")

        # مثال: 3–5 عناصر نصية تمثل الأنماط
        for i in range(random.randint(3, 6)):
            elem = {
                "type": "geometric_element",
                "index": i,
                "pattern": random.choice(["spiral", "grid", "polygon", "fractal", "symmetric motif"]),
                "symmetry_level": symmetry,
                "description": f"{main_subject} part {i+1}"
            }
            elements.append(elem)

        return elements
    
    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        تحليل بسيط: استخراج كيانات + مستوى تناظر + صفات أخرى
        """
        lower = prompt.lower()
        entities = self._extract_entities(lower)
        symmetry = "high" if any(k in lower for k in ["symmetry", "mirror", "تناظر", "مرآة"]) else \
                   "medium" if any(k in lower for k in ["pattern", "grid", "نمط", "شبكة"]) else "low"
        
        main_subject = entities[0] if entities else "geometric shape"
        
        return {
            "entities": entities,
            "main_subject": main_subject,
            "symmetry": symmetry,
            "style": "geometric"
        }

    def _enhance_geometric_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        تحسين الوصف النصي بإضافة تفاصيل هندسية
        """
        lower = prompt.lower()

        metadata = {
            "entities": self._extract_entities(lower),
            "symmetry": self._detect_symmetry(lower),
            "patterns": self._suggest_patterns(lower),
            "details": "precise lines, mathematical proportions, repeating motifs, fractal elements, golden ratio",
            "timestamp": datetime.now().isoformat()
        }

        enhanced = (
            f"{prompt}, {metadata['details']}, "
            f"{metadata['symmetry']} symmetry, {metadata['patterns']}, "
            "ultra precise, vector style, high resolution, mathematical beauty"
        )

        return {"prompt": enhanced, "metadata": metadata}

    def _extract_entities(self, text: str) -> List[str]:
        """
        استخراج كيانات هندسية بناءً على الكلمات المفتاحية
        """
        words = text.split()
        entities = [w.strip(".,!?;:()[]{}'\"") for w in words if w in self.specialization["keywords"]]
        return list(set(entities))  # إزالة التكرار

    def _detect_symmetry(self, text: str) -> str:
        """
        كشف مستوى التناظر من الوصف
        """
        if "high symmetry" in text or "مرآة كاملة" in text:
            return "high"
        elif "symmetry" in text or "تناظر" in text:
            return "medium"
        return "low"

    def _suggest_patterns(self, text: str) -> str:
        """
        اقتراح أنماط بناءً على الوصف
        """
        if "spiral" in text or "دوامة" in text:
            return "golden spiral, logarithmic curves"
        elif "grid" in text or "شبكة" in text:
            return "hexagonal grid, honeycomb pattern"
        return "repeating polygons, fractal geometry"

# ────────────────────────────────────────────────
#              اختبار سريع (اختياري)
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    engine = geometric_design_engine()

    # اختبار بسيط
    test_prompt = "golden spiral with hexagonal grid and high symmetry"
    result = engine.generate_layer(test_prompt)

    if result.success:
        print("→ نجاح!")
        print(f"  • الوصف المحسن: {result.output_data['enhanced_prompt'][:100]}...")
        print(f"  • الكيانات: {result.output_data['entities']}")
        print(f"  • مستوى التناظر: {result.output_data['metadata']['symmetry']}")
        print(f"  • الوقت: {result.total_time:.2f} ثانية")
    else:
        print("→ فشل")
        print(f"  • الرسالة: {result.message}")