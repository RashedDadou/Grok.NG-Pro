# prompt_supervisor.py
"""
كلاس SuperVisor / Critic لمراجعة وتحسين الـ master prompt
قبل إرساله لنموذج التوليد النهائي.
يدعم حلقات تحسين متعددة مع تقييم ذاتي.
"""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptSupervisor:
    """
    مشرف ذكي لمراجعة وتحسين الـ master prompt عبر جولات متعددة.
    يستخدم LLM كـ critic لتقييم الجودة واقتراح التحسينات.
    """

    def __init__(
        self,
        llm_callable=None,              # دالة تستدعي LLM (مثل openai أو groq أو grok api)
        default_max_rounds: int = 3,
        satisfaction_threshold: float = 8.5,
        model_name: str = "gpt-4-turbo",  # أو أي نموذج تدعمه
    ):
        self.llm_callable = llm_callable
        self.max_rounds = default_max_rounds
        self.threshold = satisfaction_threshold
        self.model_name = model_name

        if not self.llm_callable:
            raise ValueError("يجب تمرير دالة استدعاء LLM عند إنشاء PromptSupervisor")


    def supervise_and_refine(
        self,
        collected_prompts: Dict[str, str],
        max_rounds: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> str:
        """
        الدالة الرئيسية: تبني الـ master prompt ثم تقيّمه وتحسّنه عبر جولات
        
        Returns:
            الـ master prompt النهائي بعد التحسين
        """
        max_rounds = max_rounds if max_rounds is not None else self.max_rounds
        threshold = threshold if threshold is not None else self.threshold

        # 1. بناء النسخة الأولية
        current_master = self._build_initial_master_prompt(collected_prompts)
        logger.info("[Supervisor] تم بناء الـ master prompt الأولي")

        best_prompt = current_master
        best_score = 0.0

        for round_num in range(1, max_rounds + 1):
            logger.info(f"[Supervisor] بدء جولة التقييم رقم {round_num}/{max_rounds}")

            critique = self._get_critique(current_master)
            if not critique:
                logger.warning("لم يتم الحصول على تقييم صالح ← نستخدم النسخة الحالية")
                break

            score = critique.get("score", 0.0)
            suggestions = critique.get("suggestions", [])
            revised_prompt = critique.get("revised_prompt", current_master)

            logger.info(f"[Supervisor] جولة {round_num} → Score: {score:.2f}")

            if score > best_score:
                best_score = score
                best_prompt = revised_prompt

            if score >= threshold:
                logger.info(f"[Supervisor] تم الوصول لعتبة الرضا ({threshold}) في الجولة {round_num}")
                return revised_prompt

            # تحسين للجولة التالية
            current_master = self._apply_suggestions(revised_prompt, suggestions, collected_prompts)

        logger.warning(
            f"[Supervisor] انتهت الجولات ({max_rounds}) دون الوصول للعتبة "
            f"(أفضل score كان {best_score:.2f})"
        )
        return best_prompt


    def _build_initial_master_prompt(self, collected: Dict[str, str]) -> str:
        """بناء الـ master prompt الأولي من مخرجات المحركات المتخصصة"""
        meta = collected.get("metadata", {})

        parts = []

        if "background" in collected and collected["background"].strip():
            parts.append(f"Background / Environment:\n{collected['background']}")

        if "midground" in collected and collected["midground"].strip():
            parts.append(f"Midground / Structural elements:\n{collected['midground']}")

        if "foreground" in collected and collected["foreground"].strip():
            parts.append(f"Foreground / Main subjects:\n{collected['foreground']}")

        overall = "\n\n".join(parts)

        master = f"""A cinematic, ultra-detailed scene in 8K resolution:

{overall}

Overall style: {meta.get('style', 'cinematic, photorealistic, highly detailed')}
Mood & atmosphere: {meta.get('mood', 'dramatic, immersive')}
Lighting: {meta.get('lighting', 'dynamic, volumetric, cinematic')}
Colors: {meta.get('color_palette', 'vibrant high-contrast')}

Composition: rule of thirds, depth of field, balanced layers
--ar 16:9 --v 6 --q 2"""
        
        return master.strip()


    def _get_critique(self, prompt_text: str) -> Optional[Dict]:
        """استدعاء الـ LLM لتقييم الـ prompt"""
        if not self.llm_callable:
            logger.error("لا يوجد دالة LLM محددة")
            return None

        critique_prompt = f"""أنت ناقد فني محترف متخصص في كتابة وتقييم prompts لتوليد الصور بالذكاء الاصطناعي.

قم بتقييم الـ prompt التالي من 1.0 إلى 10.0 بناءً على:
• الوضوح والدقة
• التماسك بين الطبقات (background/mid/foreground)
• التوازن والتناسق البصري
• قوة الوصف الإبداعي والجمالي
• مناسبة لنموذج توليد صور حديث (مثل Flux أو Aurora)

الـ prompt:

{prompt_text}

أرجع الرد بتنسيق JSON صالح فقط، بدون أي نص إضافي خارج الكائن:

{{
  "score": عدد عشري من 1.0 إلى 10.0,
  "strengths": ["نقطة قوة 1", "نقطة قوة 2", ...],
  "weaknesses": ["نقطة ضعف 1", "نقطة ضعف 2", ...],
  "suggestions": ["اقتراح تحسين 1", "اقتراح تحسين 2", ...],
  "revised_prompt": "نسخة محسنة كاملة من الـ prompt (إذا كان يحتاج تعديل واضح)"
}}
"""

        try:
            raw_response = self.llm_callable(critique_prompt)
            # تنظيف الرد في حال وجود markdown أو نص إضافي
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            data = json.loads(cleaned)
            return data
        except Exception as e:
            logger.exception("فشل تحليل رد الـ critic")
            return None


    def _apply_suggestions(
        self,
        current: str,
        suggestions: List[str],
        original_parts: Dict[str, str]
    ) -> str:
        """تطبيق اقتراحات التحسين على الـ prompt (نسخة بسيطة)"""
        revised = current

        for sug in suggestions:
            sug_lower = sug.lower()
            if "balance" in sug_lower or "توازن" in sug_lower:
                revised += "\nBetter balance between layers, avoid overcrowding"
            elif "lighting" in sug_lower or "إضاءة" in sug_lower:
                revised += "\nImproved lighting: more dramatic volumetric lighting, god rays, realistic shadows"
            elif "detail" in sug_lower or "تفاصيل" in sug_lower:
                revised += "\nUltra-detailed textures, intricate details, 8k quality"
            elif "coherence" in sug_lower or "تماسك" in sug_lower:
                revised += "\nStrong visual coherence and style consistency across all layers"

        return revised.strip()


    def plan_layers(self, full_prompt: str, mode: str = "parallel") -> dict:
        """
        تقسيم أولي بسيط للـ prompt إلى طبقات (fallback حتى يتم ربط LLM)
        """
        lower = full_prompt.lower()
        result = {
            "background": full_prompt,
            "midground": full_prompt,
            "foreground": full_prompt
        }

        # قواعد بدائية – يمكن تحسينها لاحقًا
        if any(x in lower for x in ["city", "مدينة", "night", "ليل", "forest", "غابة", "beach", "شاطئ"]):
            result["background"] = f"wide cinematic {full_prompt}, environment, atmosphere, distant view"

        if any(x in lower for x in ["car", "سيارة", "vehicle", "building", "building", "bridge", "جسر"]):
            result["midground"] = f"structural elements: {full_prompt}, mid layer focus"

        if any(x in lower for x in ["girl", "فتاة", "woman", "person", "شخص", "character", "إلف", "creature"]):
            result["foreground"] = f"detailed main subject: {full_prompt}, sharp focus, emotional"

        logger.info(f"[plan_layers fallback] {result}")
        return result
    
# ────────────────────────────────────────────────
# مثال استخدام (اختباري)
# ────────────────────────────────────────────────

if __name__ == "__main__":
    # مثال على دالة وهمية لاستدعاء LLM
    def dummy_llm(prompt):
        # هنا تضع استدعاء حقيقي لـ Grok / OpenAI / Groq / ...
        return '''{
            "score": 7.2,
            "strengths": ["تفاصيل جيدة في الخلفية", "أسلوب سينمائي"],
            "weaknesses": ["الإضاءة ضعيفة", "عدم توازن بين الطبقات"],
            "suggestions": ["أضف إضاءة volumetric أقوى", "قلل من كثافة العناصر في المقدمة"],
            "revised_prompt": "نسخة محسنة من الـ prompt..."
        }'''

    supervisor = PromptSupervisor(llm_callable=dummy_llm)

    test_data = {
        "background": "cyberpunk city rainy night neon lights",
        "midground": "flying cars and bridges",
        "foreground": "girl with umbrella",
        "metadata": {"style": "cyberpunk noir", "mood": "melancholic"}
    }

    final_prompt = supervisor.supervise_and_refine(test_data)
    print("الـ prompt النهائي:")
    print(final_prompt)