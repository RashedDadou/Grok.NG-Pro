# Agent_helper.py

import layer_plane

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s'
)

logger = logging.getLogger(__name__)


class AIHelper:
    """
    مساعد للـ Grok.Supervisor — مخصص لتحليل ومعالجة الـ prompts الخارجية
    والتفاعل مع المحركات المختلفة (traditional, geometric, ...)
    """

    def __init__(self, supervisor: Any):
        self.supervisor = supervisor
        self.external_prompts: list = []
        self.logger = logging.getLogger("AIHelper")
        
    def analyze_external_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        تحليل prompt خارجي وإرجاع اقتراحات / تصنيف / توصيات

        Returns:
            dict يحتوي على:
            - type: نوع المحرك المقترح
            - suggestions: نص توجيهي
            - code: مثال كود (إن وجد)
            - image_gen: اقتراح توليد صورة (إن وجد)
            - needs_refresh: هل يحتاج إعادة معالجة
        """
        if not prompt or not isinstance(prompt, str):
            self.logger.warning("تم تمرير prompt فارغ أو غير صالح")
            return {"success": False, "error": "prompt فارغ أو غير صالح"}

        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            return {"success": False, "error": "prompt فارغ بعد التنظيف"}

        self.external_prompts.append(cleaned_prompt)
        self.logger.info("تحليل prompt خارجي (طول: %d حرف)", len(cleaned_prompt))

        lower = cleaned_prompt.lower()

        # ─── تصنيف سريع بناءً على كلمات مفتاحية ───────────────────────────────
        result: Dict[str, Any] = {
            "success": True,
            "type": "unknown",
            "needs_refresh": len(cleaned_prompt.split()) > 8,  # قيمة أكثر تساهلاً
        }

        if any(kw in lower for kw in ["هندسي", "cad", "هندسة", "تصميم ميكانيكي", "هيكل"]):
            result.update({
                "type": "geometric",
                "suggestions": (
                    "يبدو أن الوصف هندسي → ركّز على المقاييس، التحليل الهيكلي، "
                    "اختيار المواد (صلب، ألمنيوم، كربون فايبر)، وصف دقيق للأبعاد"
                ),
                "code_hint": "استخدم Fusion 360 / SolidWorks: sketch → extrude → revolve → assembly"
            })

        elif any(kw in lower for kw in ["معماري", "تخيلي", "تصميم داخلي", "مبنى", "مدينة", "فيلا"]):
            result.update({
                "type": "traditional",
                "suggestions": (
                    "وصف معماري/تخيلي → يناسب محرك traditional. "
                    "أضف تفاصيل الإضاءة، المواد، الجو، العناصر الـ sci-fi إذا أردت"
                ),
                "image_gen": "generate architectural / futuristic city image"
            })

        else:
            # حالة عامة
            result["suggestions"] = (
                "وصف عام → حاول تحديد نوع المشهد (كائن، بيئة، معماري، هندسي) "
                "وأضف كلمات جودة (highly detailed, cinematic, 8k, masterpiece)"
            )
            result["code_hint"] = "# مثال بسيط لتوليد صورة أو معالجة prompt\npass"

        # إشعار المشرف (Supervisor)
        try:
            self.supervisor.notify("analyzed_external", result)
        except Exception as exc:
            self.logger.warning("فشل إرسال notify للمشرف: %s", exc)

        return result

    def modify_code(self, code: str, instruction: str) -> str:
        """
        تعديل كود خارجي بناءً على تعليمة نصية

        Args:
            code: الكود الأصلي (نص)
            instruction: التعليمة (ما الذي يجب تغييره/إضافته)

        Returns:
            الكود بعد التعديل (حالياً تعديل بسيط – يمكن تطويره لاحقاً)
        """
        if not code or not isinstance(code, str):
            self.logger.warning("كود غير صالح تم تمريره لـ modify_code")
            return "# كود غير صالح\npass"

        if not instruction or not isinstance(instruction, str):
            self.logger.warning("تعليمة فارغة أو غير صالحة")
            return code  # نرجع الأصلي بدون تغيير

        self.logger.info("تعديل كود بناءً على: %r (طول الكود الأصلي: %d)", 
                         instruction[:60], len(code))

        # ─── تعديل بسيط حالياً (يمكن استبداله بـ LLM أو AST لاحقاً) ───────
        instruction_clean = instruction.strip()
        comment = f"\n\n# تعديل بناءً على: {instruction_clean}\n"

        if len(code.strip()) == 0:
            return comment + "# كود فارغ → أضف المحتوى هنا"

        # إضافة تعليق + سطر توضيحي بسيط
        modified = code.rstrip() + comment

        # أمثلة على تعديلات ذكية بسيطة (يمكن توسيعها)
        lower_inst = instruction_clean.lower()
        if "print" in lower_inst or "اظهر" in lower_inst or "debug" in lower_inst:
            modified += "print('تم التعديل بنجاح!')\n"
        elif "function" in lower_inst or "دالة" in lower_inst:
            modified += "\ndef example_function():\n    print('مثال دالة بعد التعديل')\n"

        return modified
    
class AITab3:
    """
    مساعد فرعي لـ AIHelper — مشرف على محركي traditional و geometric
    يقوم بمراقبة الـ prompt أثناء الكتابة، تحليلها، اقتراح تحسينات،
    ومحاولة توليد معاينات مبكرة عندما تكون الثقة كافية.
    """

    def __init__(self, helper: AIHelper):
        self.helper = helper
        self.engines: Dict[str, Any] = {
            "traditional": None,
            "geometric": None,
            # يمكن إضافة المزيد من المحركات هنا لاحقاً
        }
        self._last_monitor_time = 0.0
        self._monitor_cooldown = 0.5  # نصف ثانية بين كل تحليل أثناء الكتابة
        self.logger = logging.getLogger("AITab3")

    def _get_pipeline_class(self):
        if self._pipeline_class is None:
            from unified_stage_pipeline import UnifiedStagePipeline
            self._pipeline_class = UnifiedStagePipeline
        return self._pipeline_class
    
    def attach_engine(self, name: str, engine: Any) -> bool:
        """ربط محرك معين بالمساعد"""
        if name not in self.engines:
            self.logger.warning("اسم المحرك غير مدعوم: %s", name)
            return False

        self.engines[name] = engine
        self.logger.info("تم ربط محرك %s بنجاح", name)
        return True

    def track_prompt(self, char: str):
        if hasattr(self.helper.supervisor, 'on_char'):
            response = self.helper.supervisor.on_char(char)
            if response:
                self.logger.debug("رد من pipeline على on_char: %s", response)
        else:
            self.logger.debug("الـ supervisor لسه ما عندوش on_char – تجاهل")

    def monitor_engine(self, engine_name: str, prompt: str) -> Dict[str, Any]:
        """
        مراقبة وتحليل الـ prompt في الوقت الحقيقي أثناء الكتابة

        Returns:
            قاموس يحتوي على التحليل + الاقتراحات + حالة المعاينة
        """
        if not prompt or not isinstance(prompt, str):
            self.logger.warning("prompt غير صالح تم تمريره لـ monitor_engine")
            return {"success": False, "error": "prompt غير صالح"}

        # cooldown بسيط لتجنب المعالجة المتكررة جداً أثناء الكتابة السريعة
        from time import perf_counter
        now = perf_counter()
        if now - self._last_monitor_time < self._monitor_cooldown:
            self.logger.debug("تم تجاهل monitor_engine بسبب cooldown")
            return {"success": False, "skipped": True, "reason": "cooldown"}

        self._last_monitor_time = now

        self.logger.info("مراقبة prompt لمحرك %s (طول: %d حرف)", 
                         engine_name, len(prompt))

        result: Dict[str, Any] = {
            "success": True,
            "engine": engine_name,
            "preview_path": None,
            "skipped": False
        }

        # 1. استخراج الكيانات والتحليل الأساسي
        analysis = self.extract_entities_multilingual(prompt)
        result.update(analysis)

        entities = analysis.get("entities", [])
        negated = analysis.get("negated", [])
        lang = analysis.get("lang_hint", "unknown")

        # 2. تقدير المزاج
        mood = self._estimate_mood(prompt)
        result["mood"] = mood

        # 3. حساب درجة الثقة
        confidence = self._calculate_confidence(entities, negated, mood, lang, prompt)
        result["confidence"] = confidence

        # 4. اقتراحات تحسين
        suggestions = self._generate_suggestions(entities, negated, mood, lang, prompt)
        result["suggestions"] = suggestions

        # تسجيل النتيجة الرئيسية
        self.logger.info("تحليل monitor_engine → ثقة: %.2f | كيانات: %d | مزاج: %s",
                         confidence, len(entities), mood)

        if suggestions:
            self.logger.info("اقتراحات تحسين (%d): %s", len(suggestions), "; ".join(suggestions[:2]))

        # 5. محاولة إنشاء معاينة إذا كانت الثقة جيدة
        if confidence >= 0.75 and len(entities) >= 1:
            preview_path = self._try_generate_preview(engine_name, result)
            result["preview_path"] = preview_path
            if preview_path:
                self.logger.info("تم إنشاء معاينة بنجاح: %s", preview_path)
            else:
                self.logger.warning("فشل إنشاء المعاينة رغم الثقة الكافية")

        return result

    # ─── دوال مساعدة داخلية (private) ───────────────────────────────────────

    def _estimate_mood(self, prompt: str) -> str:
        """تقدير المزاج بناءً على كلمات مفتاحية"""
        lower = prompt.lower()
        mood_keywords = {
            "mysterious": ["غامض", "ضباب", "ظلال", "سري", "مظلم", "mysterious", "fog", "shadow", "dark"],
            "epic": ["ملحمي", "عظيم", "epic", "grand", "majestic"],
            "calm": ["هادئ", "سكينة", "calm", "peaceful", "serene"],
            "dramatic": ["درامي", "مثير", "dramatic", "intense"]
        }

        indicators = []
        for mood, words in mood_keywords.items():
            if any(w in lower for w in words):
                indicators.append(mood)

        return ", ".join(set(indicators)) or "neutral"

    def _calculate_confidence(self, entities: list, negated: list, mood: str,
                              lang: str, prompt: str) -> float:
        """حساب درجة الثقة بشكل منظم"""
        base = 0.30
        entity_bonus = 0.18 * len(entities)
        negated_penalty = -0.12 * len(negated)
        mood_bonus = 0.15 if mood != "neutral" else 0.0
        length_bonus = 0.12 if len(prompt.split()) >= 6 else 0.0
        lang_bonus = 0.08 if lang in ("arabic", "english") else 0.0

        # يفترض أن analysis يحتوي على score (من extract_entities_multilingual)
        score_bonus = 0.4 * (getattr(self, '_last_analysis_score', 0.0))

        confidence = min(0.96, base + entity_bonus + negated_penalty +
                         mood_bonus + length_bonus + lang_bonus + score_bonus)

        if len(prompt.split()) < 4:
            confidence *= 0.65

        return round(confidence, 3)

    def _generate_suggestions(self, entities: list, negated: list, mood: str,
                              lang: str, prompt: str) -> list[str]:
        """إنشاء اقتراحات تحسين ذكية"""
        lower = prompt.lower()
        sugs = []

        if not entities:
            sugs.append("الوصف لا يحتوي كيان رئيسي واضح → أضف كائن أو عنصر بيئي بارز")
        elif len(entities) == 1:
            sugs.append("وصف قصير نسبياً → أضف كيان ثانوي أو تفصيل بصري")

        if "creature" in entities and "effect" not in entities:
            sugs.append("الكائن يحتاج تأثير بصري → جرب: هالة ذهبية، دخان سحري، وهج أزرق")

        if any(e in entities for e in ["nature_element", "environment"]) and "effect" not in entities:
            if not any(w in lower for w in ["ضوء", "إضاءة", "لون", "غروب", "ليل", "نهار", "light", "glow", "sunset"]):
                sugs.append("أضف وصفاً للإضاءة أو الجو → مثال: ضباب الصباح، ضوء قمري فضي")

        if "mysterious" in mood.lower() and "ضباب" not in lower and "fog" not in lower:
            sugs.append("للمزاج الغامض: جرب ضباب كثيف، أشعة تخترق الغيوم، ظلال طويلة")

        if lang == "mixed" and len(entities) < 2:
            sugs.append("الوصف مختلط (عربي+إنجليزي) → حاول توحيد اللغة أو إضافة تفاصيل أكثر")

        return sugs

    def _try_generate_preview(self, engine_name: str, analysis_data: Dict) -> Optional[str]:
        """محاولة إنشاء معاينة باستخدام المحرك المناسب"""
        engine = self.engines.get(engine_name) or self.engines.get("traditional")
        if not engine or not hasattr(engine, "_create_simple_image"):
            self.logger.warning("لا يوجد محرك صالح لتوليد معاينة: %s", engine_name)
            return None

        try:
            task_data = {
                "entities": analysis_data.get("entities", []),
                "negated": analysis_data.get("negated", []),
                "mood": analysis_data.get("mood", "neutral"),
                "confidence": analysis_data.get("confidence", 0.0),
                "lang": analysis_data.get("lang_hint", "unknown"),
                "raw_prompt": analysis_data.get("raw_prompt", ""),
            }

            path = engine._create_simple_image(task_data, is_video=False)
            if path and Path(path).exists():
                return str(path)

            self.logger.warning("تم استدعاء _create_simple_image لكن المسار غير صالح")
            return None

        except Exception as e:
            self.logger.error("فشل إنشاء معاينة: %s", e, exc_info=False)
            return None

    # ─── extract_entities_multilingual ما زالت كما هي (يمكن تحسينها لاحقاً) ──
    # ... (يمكن نسخها كما هي أو تحسينها بنفس الأسلوب)