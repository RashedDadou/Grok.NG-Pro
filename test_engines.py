# test_engines.py
import time
from pathlib import Path
from PIL import Image

from traditional_design_engine import traditionalDesignEngine
from geometric_design_engine import geometric_design_engine
from environment_design_engine import environment_design_engine

from Image_generation import GenerationResult, print_generation_result

HAS_PIL = 'Image' in dir(Image)

def simple_overlay(base_path: str, overlay_path: str, output_path: str, opacity=0.4) -> str | None:
    if not HAS_PIL:
        print("Pillow غير موجود → مفيش دمج")
        return None
    try:
        base = Image.open(base_path).convert("RGBA")
        overlay = Image.open(overlay_path).convert("RGBA").resize(base.size, Image.Resampling.LANCZOS)
        overlay.putalpha(int(255 * opacity))
        Image.alpha_composite(base, overlay).save(output_path, quality=90)
        print(f"تم الدمج → {output_path}")
        return output_path
    except Exception as e:
        print(f"فشل الدمج: {e}")
        return None

def run_one_engine(engine_class, name, prompt, tasks=None):
    print(f"\n===== اختبار {name} =====")
    try:
        engine = engine_class()

        if tasks:
            for task, comp, deps in tasks:
                engine.add_task(task, complexity=comp, dependencies=deps)

        if not engine.receive_input(prompt):
            print("فشل استقبال الـ prompt")
            return None

        # ← استخدم الدالة الجديدة
        result = engine.generate_layer(prompt=prompt, force_refresh=True)

        print(f"نجاح: {result.success} | {result.message} | وقت: {result.total_time:.2f}s")

        if result.success and result.output_data:
            print("  Enhanced prompt:")
            enhanced = result.output_data.get("enhanced_prompt", "غير موجود")
            print("   ", enhanced[:200] + "..." if len(enhanced) > 200 else enhanced)
            
            print("  Metadata:")
            for k, v in result.output_data.get("metadata", {}).items():
                print(f"    • {k}: {v}")

        return result

    except Exception as e:
        print(f"خطأ في {name}: {type(e).__name__}: {e}")
        return None

# ─── الاختبارات الفعلية ─────────────────────────────────────
test_cases = [
    {
        "cls": traditionalDesignEngine,
        "name": "Traditional",
        "prompt": "فتاة تركب حصان أبيض في غابة سحرية ضبابية مرعبة، إضاءة سينمائية",
        "tasks": [("main_subject", 4.8, []), ("environment", 3.7, []), ("atmosphere", 2.9, [])],
        "spec": "traditional_design"
    },
    {
        "cls": geometric_design_engine,
        "name": "Geometric",
        "prompt": "golden ratio fibonacci spiral، sacred geometry، لون ذهبي لامع",
        "tasks": [("base_pattern", 3.2, []), ("spiral", 3.5, ["base_pattern"])],
        "spec": "geometric_design"
    },
    {
        "cls": environment_design_engine,
        "name": "Cyber Environment",
        "prompt": "مدينة سايبربانك ليلية، نيون، سيارات طائرة، مطر، انعكاسات",
        "tasks": [("cityscape", 5.0, []), ("neon", 3.5, []), ("vehicles", 4.0, ["cityscape"])],
        "spec": "environment_design"
    }
]

# محاولة دمج ذكي (طبقات متتالية)
generated = []
for case in test_cases:
    path = run_one_engine(
        engine_class=case["cls"],
        name=case["name"],
        prompt=case["prompt"],
        tasks=case["tasks"],
        specialization=case["spec"]
    )
    if path:
        generated.append(path)
        
if len(generated) >= 2 and HAS_PIL:
    current = generated[0]
    print(f"  بداية الدمج من: {Path(current).name}")

    for i, overlay in enumerate(generated[1:], 1):
        out = f"merged_layer{i}_{int(time.time()*1000)}.png"
        opacity = 0.35 if i == 1 else 0.25
        current = simple_overlay(current, overlay, out, opacity=opacity)
        if current:
            print(f"  → تم دمج الطبقة {i} بنجاح: {out}")
        else:
            print(f"  × فشل دمج الطبقة {i}")
            break

    if current:
        print("\n" + "═" * 80)
        print("🎉 النتيجة النهائية المدمجة:")
        print(f"  → المسار: {current}")
        if Path(current).is_file():
            print("    (الملف موجود فعليًا)")
            print(f"    الحجم: {Path(current).stat().st_size:,} بايت")
        else:
            print("    تحذير: الملف غير موجود!")
        print("═" * 80)

        # محاولة فتح الصورة تلقائيًا (ويندوز)
        try:
            import os
            os.startfile(current)
            print("تم فتح الصورة النهائية تلقائيًا")
        except Exception as e:
            print(f"ما قدرناش نفتح الصورة تلقائيًا: {e}")
    else:
        print("فشل الدمج الكلي – ما فيش صورة نهائية")
else:
    print("مش كفاية صور صالحة للدمج أو Pillow مش موجود")