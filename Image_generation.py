# Image_generation.py
"""
النواة الأساسية + المحركات المتخصصة (Pipeline لتوليد الصور)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from time import perf_counter
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from pathlib import Path
from PIL import Image

from generation_result import GenerationResult
from Core_Image_Generation_Engine import CoreImageGenerationEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s'
)
logger = logging.getLogger(__name__)

def print_generation_result(name: str, result_or_list) -> Optional[str]:
    """
    طباعة ملخص نتيجة التوليد – تدعم كائن واحد أو قائمة من الكائنات
    """
    preview_paths = []

    if isinstance(result_or_list, list):
        print(f"\nنتيجة {name} (قائمة من {len(result_or_list)} نتائج):")
        for i, res in enumerate(result_or_list, 1):
            print(f"  نتيجة رقم {i}:")
            print(f"    نجاح          : {res.success}")
            print(f"    رسالة         : {res.message}")
            print(f"    الوقت الكلي   : {res.total_time:.2f} ث")

            # ────────────── الإضافة الجديدة ──────────────
            if res.output_data and "enhanced_prompt" in res.output_data:
                enhanced = res.output_data["enhanced_prompt"]
                print("    Enhanced prompt:")
                print("     ", enhanced[:120] + "..." if len(enhanced) > 120 else enhanced)

            if res.output_data and "preview_path" in res.output_data:
                path = res.output_data["preview_path"]
                preview_paths.append(path)
                print(f"    → معاينة       : {path}")
                if Path(path).is_file():
                    print("      (الملف موجود فعليًا)")
                else:
                    print("      تحذير: المسار موجود لكن الملف غير موجود!")
            print("    ────────────────")
        return preview_paths

    # حالة كائن واحد
    print(f"\nنتيجة {name}:")
    print(f"  نجاح          : {result_or_list.success}")
    print(f"  رسالة         : {result_or_list.message}")
    print(f"  الوقت الكلي   : {result_or_list.total_time:.2f} ث")

    # ────────────── الإضافة الجديدة ──────────────
    if result_or_list.output_data and "enhanced_prompt" in result_or_list.output_data:
        enhanced = result_or_list.output_data["enhanced_prompt"]
        print("  Enhanced prompt (من traditional أو غيره):")
        print("   ", enhanced[:120] + "..." if len(enhanced) > 120 else enhanced)

    if result_or_list.stage_times:
        print("  أوقات المراحل:")
        for k, v in result_or_list.stage_times.items():
            print(f"    • {k:14} : {v:.3f} ث")

    preview_path = None
    if result_or_list.output_data and "preview_path" in result_or_list.output_data:
        preview_path = result_or_list.output_data["preview_path"]
        print(f"  → معاينة       : {preview_path}")
        if Path(preview_path).is_file():
            print("    (الملف موجود فعليًا)")
        else:
            print("    تحذير: المسار موجود لكن الملف غير موجود!")
    else:
        print("  → لم يتم إرجاع مسار معاينة")

    return preview_path

class CoreImageGenerationEngine(ABC):
    def __init__(self):
        super().__init__()  # لو كان فيه super من قبل
        self.specialization = self._get_specialization_config()
        self.tasks: List[Dict[str, Any]] = []
        self.dependencies: Dict[str, List[str]] = {}
        self.reverse_deps: Dict[str, List[str]] = defaultdict(list)
        self.input_port: List[str] = []
        self.render_time: float = 0.0
        self.current_prompt = ""               # النص المتراكم حالياً
        self.tracked_words = []                # الكلمات المكتملة حتى الآن
        self.stage_times = {}  # ← هنا نحفظ أوقات كل مرحلة
        self.last_refresh_time = 0.0
        self.min_refresh_interval = 1.8      # ثواني
        self.min_words_for_auto = 4
        self.last_prompt_hash = ""
        
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
    def _post_process(self, task_data: Dict) -> float:
        pass

    @abstractmethod
    def _render(self, task_data: Dict, is_video: bool = False) -> float:
        pass

    @abstractmethod
    def _create_simple_image(self, task_data: Dict, is_video: bool = False) -> Optional[str]:
        pass

class GeometricDesignEngine(CoreImageGenerationEngine):
    """
    النواة المشتركة - لا يتم إنشاء كائن منها مباشرة
    كل محرك متخصص يرث منها
    """

    def receive_input(self, input_data: str) -> bool:
        if not input_data:
            logger.warning("إدخال فارغ")
            return False
        stripped = input_data.strip()
        self.input_port.append(stripped)           # ← غيّر هنا بدل append_prompt_chunk
        logger.info(f"تم إضافة وصف جديد: {stripped[:60]}...")
        return True

    def add_task(self, name: str, complexity: float = 1.0, dependencies: Optional[List[str]] = None):
        task = {"name": name, "complexity": max(0.0, complexity)}
        self.tasks.append(task)
        self.dependencies[name] = dependencies or []
        for dep in self.dependencies[name]:
            self.reverse_deps[dep].append(name)
        logger.debug(f"أضيفت مهمة: {name} (تعقيد {complexity})")

    def check_dependencies(self) -> bool:
        all_names = {t["name"] for t in self.tasks}
        for task_name, deps in self.dependencies.items():
            missing = [d for d in deps if d not in all_names]
            if missing:
                logger.error(f"اعتماديات مفقودة لـ '{task_name}': {missing}")
                return False
        return True

    def refresh_stage(self, stage: str):
        units = self.specialization.get("units", {})
        if stage in units and isinstance(units[stage], dict):
            units[stage]["refreshed"] = True
            logger.info(f"تم تنشيط إعادة التوليد للمرحلة: {stage}")
        else:
            logger.warning(f"المرحلة '{stage}' غير موجودة أو غير قابلة للتنشيط")

    def _topological_sort(self) -> List[str]:
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

    def auto_refresh_check(self, stage: str) -> bool:  # غيرتها إلى دالة داخل الكلاس لتسهيل الوصول إلى self
        """
        تقييم ما إذا كان من المفيد إعادة تنفيذ مرحلة معينة،
        وتفعيل refreshed تلقائياً إذا كان ذلك ضرورياً.
        """
        units = self.specialization.get("units", {})
        if stage not in units:
            logger.warning(f"المرحلة '{stage}' غير موجودة")
            return False

        # إذا كانت refreshed مفعلة بالفعل → ننفذ ونرجع
        if units[stage].get("refreshed", False):
            return True

        tasks = self.tasks
        if not tasks:
            return False

        task_count = len(tasks)
        total_complexity = sum(t.get("complexity", 0) for t in tasks)

        need_refresh = False

        if task_count >= 3:
            need_refresh = True
            logger.debug(f"عدد المهام ({task_count}) كبير → تفعيل refreshed لـ {stage}")

        elif total_complexity > 7.0:
            need_refresh = True
            logger.debug(f"تعقيد كلي مرتفع ({total_complexity:.1f}) → تفعيل refreshed لـ {stage}")

        elif stage in ("integration", "post_processing"):
            has_deps = any(self.dependencies.get(t["name"], []) for t in tasks)
            if has_deps:
                need_refresh = True
                logger.debug(f"يوجد اعتماديات → تفعيل refreshed لـ {stage}")

        if need_refresh:
            units[stage]["refreshed"] = True
            logger.info(f"تم تفعيل refreshed تلقائياً للمرحلة: {stage}")

        return need_refresh
  
    def _get_specialization_config(self) -> Dict[str, Any]:
        return {
            "name": "geometric_design",
            "rendering_style": "geometric",
            "units": {
                "nlp": {"function": self._analyze_prompt},
                "integration": {"function": self._integrate},
                "post_processing": {"function": self._post_process},
                "rendering": {"function": self._render},
            }
        }

    def get_current_prompt_hash(self):
        full = " ".join(self.input_port).strip()
        if not full:
            return "empty_prompt"
        return hashlib.sha256(full.encode('utf-8')).hexdigest()[:16]

    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        logger.info("[Geometric] Analyzing prompt...")
        entities = []
        if "spiral" in prompt.lower():
            entities.append("spiral")
        if "hex" in prompt.lower():
            entities.append("hexagon")
        if "fractal" in prompt.lower():
            entities.append("fractal")
        
        result = {
            "entities": entities,
            "symmetry_level": "high" if "symmetrical" in prompt.lower() else "medium",
            "planes": []  # مهم جدًا عشان ما يحصلش KeyError لاحقًا
        }
        return result   # ← تأكد إن ده موجود
    
    def generate_image(self, is_video: bool = False, force_refresh: bool = False) -> GenerationResult:
        """
        الدالة الرئيسية لتوليد الصورة أو الفيديو
        """
        spec_name = self.specialization.get("name", "unknown")
        logger.info(
            f"بدء توليد | spec: {spec_name} | فيديو: {is_video} | force_refresh: {force_refresh}"
        )

        start_total = perf_counter()
        stage_times: Dict[str, float] = {}
        task_data = None
        preview_path = None

        if not self.input_port:
            return GenerationResult(
                success=False,
                message="لا يوجد وصف (input_port فارغ)",
                total_time=0.0,
                stage_times={},
                specialization=spec_name,
                is_video=is_video
            )

        if not self.check_dependencies():
            return GenerationResult(
                success=False,
                message="فشل التحقق من الاعتماديات",
                total_time=0.0,
                stage_times={},
                specialization=spec_name,
                is_video=is_video
            )

        try:
            full_prompt = " ".join(self.input_port).strip()
            if not full_prompt:
                raise ValueError("الـ prompt بعد التنظيف فارغ")

            logger.info(f"الـ prompt النهائي: {full_prompt[:120]}{'...' if len(full_prompt)>120 else ''}")

            # ────────────────────────────────────────────────
            # lazy import + إنشاء pipeline بدون parameters إضافية
            # ────────────────────────────────────────────────
            from unified_stage_pipeline import UnifiedStagePipeline
            pipeline = UnifiedStagePipeline(self)  # ← بس كده (self بس)

            # استدعاء process (غيّر الاسم لو الدالة الحقيقية مختلفة)
            task_data = pipeline.process(
                prompt=full_prompt,
                force_refresh=force_refresh
            )

            # حماية + إضافة raw_prompt
            if not isinstance(task_data, dict):
                task_data = {}
            task_data["raw_prompt"] = full_prompt

            # أخذ الأوقات (لو موجودة)
            stage_times["unified_stages"] = task_data.get("unified_time", 0.0)
            stage_times["analyze"] = task_data.get("stage_times", {}).get("nlp", 0.0)
            stage_times["integrate"] = task_data.get("stage_times", {}).get("integration", 0.0)
            stage_times["post_process"] = task_data.get("stage_times", {}).get("post_processing", 0.0)

            # الـ Rendering
            t_render = perf_counter()
            render_duration, preview_path = self._render(task_data, is_video=is_video)
            stage_times["render"] = render_duration

            total_time = perf_counter() - start_total

            result = GenerationResult(
                success=True,
                message="تم التوليد بنجاح (باستخدام UnifiedStagePipeline)",
                total_time=total_time,
                stage_times=stage_times,
                specialization=spec_name,
                is_video=is_video,
                output_data={
                    "preview_path": preview_path,
                    "task_data": task_data
                }
            )

            # حفظ في الذاكرة (اختياري)
            try:
                if hasattr(self, 'memory_manager') and self.memory_manager:
                    prompt_hash = self.memory_manager.get_prompt_hash(full_prompt)
                    self.memory_manager.store_original_layer_result(
                        prompt_hash=prompt_hash,
                        layer_name=spec_name,
                        layer_result=result
                    )
                    logger.info(f"[Memory] تم الحفظ → hash: {prompt_hash[:8]}...")
            except Exception as mem_err:
                logger.warning(f"فشل الحفظ في الذاكرة: {mem_err}")

            if reset_input_after:
                logger.debug("[Input] مسح input_port بعد التوليد")
                self.input_port.clear()

            return result

        except Exception as e:
            logger.exception("[Generate Image] خطأ عام")
            print("الخطأ الكامل:", str(e))
            import traceback
            traceback.print_exc()

            total_time = perf_counter() - start_total
            return GenerationResult(
                success=False,
                message=f"خطأ أثناء التوليد: {str(e)}",
                total_time=total_time,
                stage_times=stage_times,
                specialization=spec_name,
                is_video=is_video,
                output_data={}
            )
            
    def _render(self, task_data: Dict, is_video: bool = False) -> tuple[float, str | None]:
        try:
            logger.info("[Geometric] Rendering...")
            print("داخل _render – قبل create_simple_image")
            
            path = self._create_simple_image(task_data, is_video=is_video)
            
            print("داخل _render – بعد create_simple_image", path)
            duration = 0.6  # أو قيس الوقت الحقيقي
            return duration, path
        except Exception as e:
            print("خطأ داخل _render:", str(e))
            import traceback
            traceback.print_exc()
            return 0.1, None

    def _integrate(self, task_data):
        logger.info("[Geometric] Integrating...")
        print("وصلت بداية _integrate")   # ← طباعة للديباج
        
        try:
            # الكود الأصلي بتاعك هنا
            print("قبل أي عملية في integrate")
            # ... أي سطر في الكود ...
            print("بعد كل عملية مهمة")
            
            print("نهاية _integrate – راجع task_data:", task_data)
            return task_data  # ← تأكد إن في return
        except Exception as e:
            print("خطأ داخل _integrate:", str(e))
            raise  # عشان نطبع الخطأ في اللوج

    def _post_process(self, task_data: Dict) -> Dict[str, Any]:
        entities_count = len(task_data.get("entities", []))
        symmetry_bonus = 0.3 if task_data.get('symmetry_level') == 'high' else 0.0
        duration = 0.4 + entities_count * 0.15 + symmetry_bonus

        logger.info(
            f"[Geometric Post-process] وقت تقريبي: {duration:.1f}s - "
            f"تناظر عالي: {task_data.get('symmetry_level') == 'high'}"
        )

        return {
            "success": True,
            "duration_seconds": duration,
            "summary": f"Post-processing completed | symmetry: {task_data.get('symmetry_level', 'unknown')}",
            "entities_count": entities_count,
            "warnings": [],
            "metadata": {
                "symmetry_level": task_data.get("symmetry_level", "medium"),
                "processed_at": perf_counter()
            }
        }

    def _create_simple_image(self, task_data: Dict, is_video: bool = False) -> Optional[str]:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.animation import FuncAnimation, PillowWriter
        from pathlib import Path
        import matplotlib

        specialization_name = self.specialization.get("name", "geometric")
        base_name = f"{specialization_name}_{'animation' if is_video else 'preview'}"
        output_path = Path(f"{base_name}.{'gif' if is_video else 'png'}")
        
        prompt = task_data.get("raw_prompt") or task_data.get("prompt", "geometric test pattern")
      
        try:
            print("داخل _create_simple_image – قبل إنشاء fig")
            fig, ax = plt.subplots(figsize=(10, 10))
            print("تم إنشاء fig و ax")

            ax.set_aspect('equal')
            ax.set_facecolor('#0a0a14')
            print("تم ضبط الـ ax")

            ax.set_xlim(-5.5, 5.5)
            ax.set_ylim(-5.5, 5.5)
            ax.set_xticks([])
            ax.set_yticks([])

            entities = task_data.get("entities", [])
            symmetry_level = task_data.get("symmetry_level", "medium")
            has_pattern = any(word in entities for word in ["pattern", "grid", "spiral", "fractal"])

            # ─── Koch snowflake ───────────────────────────────────────────────────────
            def koch_curve(start, end, iterations=3):
                if iterations == 0:
                    return [start, end]
                dx = end - start
                p1 = start + dx * 1/3
                p3 = start + dx * 2/3
                perp = np.array([-dx[1], dx[0]]) * (np.sqrt(3)/3)
                p2 = p1 + perp
                return (koch_curve(start, p1, iterations-1) +
                        koch_curve(p1, p2, iterations-1) +
                        koch_curve(p2, p3, iterations-1) +
                        koch_curve(p3, end, iterations-1))

            def draw_koch_snowflake(center, size, color='#00ffcc', alpha=0.72, iterations=4):
                angles = np.linspace(0, 2*np.pi, 4)[:-1]
                points = [center + size * np.array([np.cos(a), np.sin(a)]) for a in angles]
                points.append(points[0])
                curve_points = []
                for i in range(len(points)-1):
                    curve_points.extend(koch_curve(points[i], points[i+1], iterations))
                curve_points = np.array(curve_points)
                ax.plot(curve_points[:,0], curve_points[:,1], color=color, lw=1.6, alpha=alpha, zorder=8)

            if "fractal" in entities or has_pattern:
                draw_koch_snowflake(np.array([0,0]), 3.4, iterations=4)

            # ─── Golden spiral ────────────────────────────────────────────────────────
            golden_ratio = (1 + np.sqrt(5)) / 2
            spiral_data = None
            if has_pattern:
                n_turns = 7 if symmetry_level == "high" else 5
                theta = np.linspace(0, n_turns * 2 * np.pi, 500)
                r = 0.12 * np.exp(theta / (golden_ratio * 2.3))
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                ax.plot(x, y, color='#ffd700', lw=2.3, alpha=0.9, zorder=7)

                # نقاط لامعة
                for i in range(0, len(theta), 38):
                    ax.scatter(x[i], y[i], s=90 + 30*np.sin(i/12), color='#ffeb3b',
                            alpha=0.75, edgecolor='white', zorder=9, linewidth=0.8)

                spiral_data = {'r': r.copy(), 'theta': theta.copy()}

            # ─── Planes + trails ──────────────────────────────────────────────────────
            planes_data = task_data.get("planes", [])
            planes_copy = []
            for p in planes_data:
                pos = np.array(p.get("position", [0, 0])[:2])
                color = p.get("color", "silver")
                scale = 0.25 + p.get("force", 1.0) * 0.18
                planes_copy.append({'position': pos, 'color': color, 'scale': scale})

                if "trail" in p and p["trail"]:
                    trail = np.array(p["trail"])
                    ax.plot(trail[:,0], trail[:,1], color=color, alpha=0.48, lw=1.4,
                            ls='--', zorder=6)

                ax.add_patch(plt.Circle(pos, scale*1.15, color=color, alpha=0.7,
                                    ec='white', lw=2.0, zorder=10))
                ax.text(pos[0] + scale*1.4, pos[1] + scale*1.4,
                        p.get("label", "Plane"), fontsize=9.5, color="white",
                        fontweight="bold", zorder=11)

            # ─── الـ Rendering حسب is_video ──────────────────────────────────────────
            if is_video:
                # التحقق: هل فيه محتوى يستحق الدوران؟
                if spiral_data is None and not planes_copy:
                    logger.info("[GIF] لا يوجد spiral ولا planes متحركة → حفظ صورة ثابتة")
                    output_path = output_path.with_suffix('.png')
                else:
                    # إعدادات الـ animation
                    frames_count = 140
                    rotation_speed = 0.75     # درجة لكل فريم
                    fps = 30

                    def update(frame):
                        ax.clear()
                        ax.set_xlim(-5.5, 5.5)
                        ax.set_ylim(-5.5, 5.5)
                        ax.set_facecolor('#0a0a14')
                        ax.set_xticks([])
                        ax.set_yticks([])

                        rot_rad = np.radians(frame * rotation_speed)

                        # دوران الـ spiral
                        if spiral_data is not None:
                            xr = spiral_data['r'] * np.cos(spiral_data['theta'] + rot_rad)
                            yr = spiral_data['r'] * np.sin(spiral_data['theta'] + rot_rad)
                            ax.plot(xr, yr, color='#ffd700', lw=2.4, alpha=0.92)

                            # نقاط لامعة متحركة
                            for i in range(0, len(xr), 42):
                                alpha_p = 0.65 + 0.35 * np.sin(frame * 0.28 + i * 0.12)
                                ax.scatter(xr[i], yr[i], s=110, color='#ffeb3b',
                                        alpha=alpha_p, edgecolor='white', zorder=9)

                        # دوران الـ planes
                        for plane in planes_copy:
                            pos = plane['position']
                            dx = pos[0] * np.cos(rot_rad) - pos[1] * np.sin(rot_rad)
                            dy = pos[0] * np.sin(rot_rad) + pos[1] * np.cos(rot_rad)
                            ax.add_patch(plt.Circle((dx, dy), plane['scale']*1.2,
                                                color=plane['color'], alpha=0.72,
                                                ec='white', lw=2.2, zorder=10))

                        ax.set_title(f"Geometric Harmony • {frame:03d}  |  {rotation_speed*frame:04.1f}°",
                                    color="white", fontsize=11, pad=12)

                    ani = FuncAnimation(fig, update, frames=frames_count,
                                        interval=1000/fps, blit=False)

                    writer = PillowWriter(fps=fps)
                    ani.save(output_path, writer=writer, dpi=180,
                            savefig_kwargs={'facecolor': '#05050f'})

                    logger.info(f"[Geometric Video] تم حفظ GIF دوراني "
                                f"({frames_count} فريم، {fps} fps): {output_path}")
                    plt.close(fig)
                    return str(output_path)

            # حفظ الصورة الثابتة
            fig.patch.set_facecolor('#05050f')
            
            print("قبل الحفظ (savefig)")
            try:
                plt.savefig(output_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
                print("تم حفظ الصورة بنجاح في:", output_path)
            except Exception as save_err:
                print("خطأ في savefig:", str(save_err))
                import traceback
                traceback.print_exc()

            try:
                plt.close(fig)
                print("تم إغلاق fig بنجاح")
            except Exception as close_err:
                print("خطأ في close(fig):", str(close_err))
            
            logger.info(f"[Geometric Preview] تم حفظ الصورة الثابتة: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.exception(f"[Geometric Preview] فشل إنشاء المعاينة: {e}")
            plt.close('all')
            return None
    
    def _call_unit(self, unit_name: str, data: Any = None) -> Dict[str, Any]:
        """
        الدالة اللي بيستدعيها UnifiedStagePipeline
        """
        if data is None:
            data = {}

        start = perf_counter()

        try:
            if unit_name == "nlp":
                prompt = data.get("raw_prompt") or data.get("prompt", "")
                return self._analyze_prompt(prompt)

            elif unit_name == "integration":
                return {
                    "success": True,
                    "planes": self._integrate(data),
                    "stage": "integration"
                }

            elif unit_name == "post_processing":
                return self._post_process(data)

            else:
                logger.warning(f"unit غير معروف: {unit_name}")
                return {"success": False, "error": f"Unknown unit: {unit_name}"}

        except Exception as e:
            logger.exception(f"خطأ في _call_unit({unit_name})")
            return {
                "success": False,
                "error": str(e),
                "stage": unit_name
            }
            
# ────────────────────────────────────────────────
# الاختبار الرئيسي
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== اختبار GeometricDesignEngine - رسم حقيقي ===")
    print("الوقت الحالي:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        engine = GeometricDesignEngine()
        print("تم إنشاء GeometricDesignEngine بنجاح")
        
        prompt = "symmetrical geometric pattern with golden spiral and fractal elements"
        engine.receive_input(prompt)
        print(f"الـ prompt: {prompt}")
        
        # توليد صورة ثابتة أولاً (أسرع وأضمن)
        print("\nجاري توليد صورة ثابتة...")
        result = engine.generate_image(
            is_video=False,
            force_refresh=True
        )
        
        print_generation_result("الصورة الثابتة", result)
        
        # لو عايز تجرب الـ GIF (ممكن ياخد وقت أطول)
        # print("\nجاري توليد GIF دوراني...")
        # video_result = engine.generate_image(
        #     is_video=True,
        #     force_refresh=True
        # )
        # print_generation_result("الـ GIF", video_result)
        
    except Exception as e:
        print("خطأ أثناء التشغيل:")
        import traceback
        traceback.print_exc()
    
    print("\n=== انتهى الاختبار ===")
