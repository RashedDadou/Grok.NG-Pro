# engine.py
import os
import logging
import time
import math
import re
import requests
import threading
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

# استيراد PlaneLayer من ملف layers.py (هنعمله بعدين)
from layers import PlaneLayer
from draw import *

class GrokNGEngine:
    def __init__(self, prefer_api: bool = True, fallback_always_vis: bool = True):
        self.prefer_api = prefer_api
        self.fallback_always_vis = fallback_always_vis
        self.ai_swp = {"elements": ["neon_lights", "robotic_limbs", "glowing_circuits"], "intensity": 0.5}

        # 1. إعداد الـ logging (ممتاز، بس نضيف handler للـ console لو مش موجود)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S',
            handlers=[logging.StreamHandler()]  # تأكيد عرض في console
        )

        # 2. API Key مع fallback واضح
        self.api_key = os.getenv("XAI_API_KEY")
        if not self.api_key:
            logging.warning("⚠️  XAI_API_KEY غير موجود → سيتم استخدام Ultimate Fallback فقط")

        # 3. المدخلات والقواعد
        self.global_input_port = []  # لتوزيع الـ prompts تلقائيًا
        self.integration_rules = {}  # tuple keys → priority

        # 4. التخصصات مع هيكل موحد وقابل للتوسع
        default_structure = {
            "input_port": [],
            "tasks": [],
            "dependencies": {},
            "units": {"refreshed": False}
        }

        self.specializations = {
            "traditional_design": default_structure.copy(),
            "geometric_design": default_structure.copy(),
            "futuristic_design": default_structure.copy(),
        }

        # 5. نظام الأوزان الذكي لاختيار التخصص (نقلناه هنا عشان يكون جزء من الكلاس)
        self.keyword_weights = {
            "traditional_design": {
                "creature": 4, "animal": 4, "nature": 3, "organic": 3,
                "environment": 3, "forest": 2, "tree": 2, "mountain": 2, "river": 2, "plant": 2
            },
            "geometric_design": {
                "bridge": 5, "structure": 5, "building": 4, "beam": 4,
                "engine": 4, "aircraft": 4, "plane": 4, "vehicle": 3,
                "car": 3, "truck": 3, "tower": 3, "pillar": 3, "mechanical": 4, "architecture": 4
            },
            "futuristic_design": {
                "spaceship": 6, "cybercity": 5, "cyberpunk": 5, "neon": 4,
                "tech": 4, "holographic": 4, "sci-fi": 5, "future": 3,
                "superhero": 5, "superman": 7, "robot": 4, "drone": 3, "ai": 3, "gadget": 3
            }
        }

        # 6. Unified Pipeline (اللي هنطوره لكل التخصصات)
        self.UNIFIED_PIPELINE = {
            "geometric_design": [
                ("analyze", self._analyze_prompt_geometric),
                ("enhance_tasks", self.enhance_tasks_with_relations),
                ("simulate_physics", self.simulate_physics_for_tasks),
                ("fallback_render", self._render_with_ultimate_fallback),  # ← هنا التعديل الوحيد
                ("visualize", self.visualize_interaction_path),
            ],
            # باقي التخصصات لو موجودة
        }

        self.ai_effects_library = {
            "neon": {"color": "cyan", "glow": True, "pulse": True},
            "robotic": {"material": "metal", "joints": True, "servos": True},
            "glowing_circuits": {"color": "lime", "pattern": "circuit_board"},
            "holographic": {"transparency": 0.6, "flicker": True},
            "plasma": {"color": ["purple", "pink"], "energy": True}
        }
        
        logging.info("تم تهيئة GrokNGEngine بنجاح 🚀 | التخصصات المتاحة: 3 | Fallback جاهز")
                
# ==================== الدالة الموحدة هنا ====================
    def _create_mirrored_drawer(self, original_draw_func, mirror_axis: str = "vertical", width: int = 1920, height: int = 1080, variation: float = 0.0):
        """
        مصنع دوال (Factory Function) يرجع دالة رسم مرآة ديناميكية بناءً على المحور والتغيير المطلوب.
        
        هذه الدالة تحول أي دالة رسم عادية (مثل رسم جناح أيسر أو محرك) إلى نسخة مرآة تلقائية،
        مما يضمن تماثل مثالي أو شبه مثالي (مع variation للواقعية).
        
        Parameters:
        -----------
        original_draw_func : callable
            الدالة الأصلية للرسم، يجب أن تأخذ (frame: np.ndarray, pos: tuple)
            مثال: _draw_left_wing أو _draw_engine_glow
        
        mirror_axis : str, optional (default: "vertical")
            نوع التماثل:
                - "vertical"   : مرآة عمودية (يسار ←→ يمين) → مثالي للأجنحة، المحركات، العيون
                - "horizontal" : مرآة أفقية (فوق ←→ تحت) → مثالي للزعانف أو الطبقات
                - "both"       : مرآة كاملة (عمودي + أفقي) → للتماثل الرباعي أو الـ radial البسيط
        
        width : int, optional (default: 1920)
            عرض الإطار (frame) بالبكسل – ضروري لحساب المرآة بدقة
        
        height : int, optional (default: 1080)
            ارتفاع الإطار (frame) بالبكسل
        
        variation : float, optional (default: 0.0)
            نسبة التغيير العشوائي في الموقع المرآة (0.0 = تماثل مثالي، 0.1-0.3 = واقعي زي الطبيعة)
            مثال: variation=0.15 → إزاحة عشوائية تصل لـ ±15% من المركز
        
        Returns:
        --------
        callable
            دالة جديدة تأخذ (frame, pos) وترسم الجزء الأصلي في الموقع المرآة (مع variation إن وجد)
        
        Example Usage:
        --------------
        mirrored_wing = self._create_mirrored_drawer(self._draw_left_wing, "vertical", width, height, variation=0.1)
        mirrored_wing(frame, left_wing_pos)  # هيرسم الجناح الأيمن تلقائيًا مع شوية تغيير طبيعي
        """
        import numpy as np  # عشان np.random لو استخدمنا variation

        def mirrored_drawer(frame: np.ndarray, pos: tuple):
            x, y = pos
            
            # حساب الموقع المرآة حسب المحور
            if mirror_axis == "vertical":
                mirrored_x = width - x
                mirrored_y = y
            elif mirror_axis == "horizontal":
                mirrored_x = x
                mirrored_y = height - y
            elif mirror_axis == "both":
                mirrored_x = width - x
                mirrored_y = height - y
            else:
                # fallback آمن لو المحور غلط
                mirrored_x, mirrored_y = x, y
            
            mirrored_pos = (mirrored_x, mirrored_y)
            
            # إضافة تغيير عشوائي لو مطلوب (للواقعية – زي الطيور أو الكائنات الحية)
            if variation > 0:
                max_offset_x = int(variation * width * 0.15)  # حد أقصى 15% من العرض
                max_offset_y = int(variation * height * 0.10)  # حد أقصى 10% من الارتفاع
                offset_x = int(np.random.uniform(-max_offset_x, max_offset_x))
                offset_y = int(np.random.uniform(-max_offset_y, max_offset_y))
                mirrored_pos = (mirrored_pos[0] + offset_x, mirrored_pos[1] + offset_y)
            
            # استدعاء الدالة الأصلية بالموقع المرآة
            original_draw_func(frame, mirrored_pos)
        
        return mirrored_drawer

    def _process_unified(self, specialization: str, user_prompt: str, is_video: bool = False, progress_callback=None):
        """
        عملية موحدة للتخصص (حاليًا geometric فقط، قابلة للتوسع)
        """
        if specialization not in self.UNIFIED_PIPELINE:
            logging.warning(f"لا يوجد pipeline موحد لـ {specialization} → استخدام التدفق العادي")
            return self.generate("auto", user_prompt, is_video, progress_callback)

        pipeline = self.UNIFIED_PIPELINE[specialization]
        total_time = 0.0
        context = {
            "prompt": user_prompt,
            "specialization": specialization,
            "is_video": is_video,
            "engine": self,  # عشان الدوال اللي محتاجة self
            "progress_callback": progress_callback
        }

        for step_idx, (step_name, step_func) in enumerate(pipeline):
            if progress_callback:
                progress = int((step_idx / len(pipeline)) * 80)  # 80% للخطوات
                progress_callback(progress, f"خطوة موحدة {step_idx + 1}/{len(pipeline)}: {step_name} جاري...")

            start = time.time()

            try:
                # لو الدالة تحتاج self (method)
                if hasattr(step_func, '__self__'):
                    result = step_func(context)
                else:
                    # دالة خارجية زي generate_ultimate_fallback
                    result = step_func(**context)
                
                if isinstance(result, dict):
                    context.update(result)
                
                step_time = time.time() - start
                total_time += step_time
                logging.info(f"خطوة موحدة '{step_name}' انتهت في {step_time:.2f}s")
            
            except Exception as e:
                logging.error(f"خطأ في خطوة {step_name}: {e}")
                break

        if progress_callback:
            progress_callback(100, f"انتهت العملية الموحدة في {total_time:.1f} ثانية يا قمري! 💜")

        # النتيجة النهائية من context
        return context.get("image"), context.get("video")
            
    def _analyze_prompt_geometric(self, context: dict) -> dict:
        """
        تحليل خاص بـ geometric_design ضمن الـ Unified Pipeline
        """
        prompt = context["prompt"]
        specialization = context["specialization"]
        
        parsed = self.parse_prompt(prompt, specialization=specialization)
        
        # إضافات خاصة بالـ geometric
        parsed.update({
            "style": "geometric",
            "detail_level": "highly detailed, technical drawing, precise lines, blueprint style",
            "recommended_aspect": "landscape"
        })
        
        logging.info("تحليل geometric انتهى مع إضافات خاصة 🚀")
        return parsed
                                    
    def set_integration_rule(self, group: list[str], priority: int = 10):
        """
        تحديد قاعدة تكامل لمجموعة مهام (يدوي أو تلقائي)
        الأولوية العالية = تُرسم أولاً
        """
        if not group or len(group) < 2:
            logging.warning(f"مجموعة صغيرة جدًا للتكامل: {group}")
            return

        # تنظيف وترتيب
        cleaned_group = sorted(set(str(g).strip() for g in group if g))
        if len(cleaned_group) < 2:
            return

        key = tuple(cleaned_group)
        old_priority = self.integration_rules.get(key)

        self.integration_rules[key] = priority
        logging.info(
            f"قاعدة تكامل {'محدثة' if old_priority is not None else 'جديدة'}: "
            f"{cleaned_group} → أولوية {priority}"
            f"{' (كانت ' + str(old_priority) + ')' if old_priority is not None else ''}"
        )
        
    def get_best_specialization(self, input_data: str) -> str:
        """
        اختيار أفضل تخصص تلقائيًا بناءً على نظام الأوزان
        """
        if not input_data.strip():
            return "futuristic_design"

        lower_input = input_data.lower()
        words = lower_input.split()

        scores = {spec: 0 for spec in self.specializations}

        for word in words:
            for spec, weights in self.keyword_weights.items():
                if word in weights:
                    scores[spec] += weights[word]

        # إضافة وزن إضافي لو كلمة كاملة (مش جزء)
        for spec, weights in self.keyword_weights.items():
            for keyword in weights:
                if keyword in lower_input and f" {keyword} " in f" {lower_input} ":
                    scores[spec] += weights[keyword] * 0.5  # بونص صغير

        best_spec = max(scores, key=scores.get)
        best_score = scores[best_spec]

        if best_score == 0:
            best_spec = "futuristic_design"

        logging.info(f"اختيار التخصص: {best_spec} (نتيجة: {best_score}) - من: {input_data[:50]}...")
        return best_spec
    
    def auto_specialize_and_generate_tasks(self, user_prompt: str, spec_from_gui: str = None) -> dict:
        """
        دالة موحدة تقوم بـ:
        1. التحقق من التخصص المناسب للـ prompt
        2. توزيع الـ prompt على التخصصات المتطابقة
        3. توليد مهام ديناميكية ذكية للتخصص المختار
        4. دعم المرايا والتماثل التلقائي
        
        Returns:
        dict مع:
            - "best_specialization": التخصص الأمثل
            - "tasks_generated": عدد المهام اللي اتولدت
            - "is_symmetric": لو التصميم يطلب تماثل
        """
        if not user_prompt.strip():
            logging.warning("الـ prompt فارغ → لا توليد مهام")
            return {"best_specialization": "futuristic_design", "tasks_generated": 0, "is_symmetric": False}

        lower_prompt = user_prompt.lower()
        logging.info(f"بدء التوزيع والتوليد التلقائي لـ: '{user_prompt}'")

        # 1. كلمات مفتاحية محسّنة ومتوسّعة لكل تخصص
        keywords = {
            "traditional_design": ["creature", "nature", "environment", "organic", "animal", "tree", "forest", "mountain"],
            "geometric_design": ["bridge", "aircraft", "plane", "structure", "building", "engine", "beam", "vehicle", "car", "truck", "mechanical"],
            "futuristic_design": ["spaceship", "cybercity", "cyberpunk", "tech", "neon", "holographic", "superhero", "superman", "sci-fi", "spaceship", "futuristic"]
        }

        # حساب النتيجة لكل تخصص
        scores = {}
        for spec, spec_keywords in keywords.items():
            score = sum(1 for kw in spec_keywords if kw in lower_prompt)
            scores[spec] = score

        # التخصص الأفضل (أو اللي من GUI)
        best_spec = max(scores, key=scores.get) if max(scores.values()) > 0 else "futuristic_design"
        if spec_from_gui and spec_from_gui in self.specializations:
            best_spec = spec_from_gui

        logging.info(f"التخصص الأفضل: {best_spec} (نتيجة: {scores[best_spec]})")

        # 2. توزيع الـ prompt على التخصصات المناسبة
        matched_specs = [spec for spec, score in scores.items() if score > 0]
        if not matched_specs:
            matched_specs = [best_spec]

        for spec in matched_specs:
            self.specializations[spec]["input_port"].append(user_prompt)
            logging.info(f"توزيع الـ prompt على {spec}")

        # 3. تنظيف global_input_port
        if user_prompt in self.global_input_port:
            self.global_input_port.remove(user_prompt)

        # 4. توليد مهام ذكية للتخصص المختار
        self.specializations[best_spec]["tasks"].clear()  # تنظيف قبل التوليد

        symmetry_keywords = ["symmetric", "mirrored", "balanced", "twin", "bilateral", "symmetrical"]
        is_symmetric = any(word in lower_prompt for word in symmetry_keywords)

        tasks_generated = 0

        # توليد مهام أساسية حسب التخصص + الـ prompt
        if best_spec == "geometric_design":
            # Geometric: هياكل معقدة
            if any(word in lower_prompt for word in ["bridge", "building"]):
                self.add_task(best_spec, "main_beam", 7, position="center")
                self.add_task(best_spec, "left_support", 5, position="left")
                self.add_task(best_spec, "right_support", 5, position="right")
                tasks_generated += 3
            else:
                self.add_task(best_spec, "main_structure", 6, position="center")
                self.add_task(best_spec, "secondary_beam", 4, position="front")
                tasks_generated += 2

        elif best_spec == "futuristic_design":
            # Futuristic: فضائي/سايبر
            if "spaceship" in lower_prompt:
                self.add_task(best_spec, "main_hull", 6, position="center")
                self.add_task(best_spec, "left_wing", 5, position="left")
                self.add_task(best_spec, "right_wing", 5, position="right")
                self.add_task(best_spec, "engine_core", 7, position="rear")
                tasks_generated += 4
            elif any(word in lower_prompt for word in ["cybercity", "city"]):
                self.add_task(best_spec, "main_tower", 6, position="center")
                self.add_task(best_spec, "neon_building_1", 4, position="left")
                self.add_task(best_spec, "neon_building_2", 4, position="right")
                tasks_generated += 3
            else:
                self.add_task(best_spec, "main_body", 5, position="center")
                tasks_generated += 1

        else:  # traditional_design
            self.add_task(best_spec, "main_body", 5, position="center")
            if "creature" in lower_prompt or "animal" in lower_prompt:
                self.add_task(best_spec, "head", 4, position="front")
                self.add_task(best_spec, "wings", 4, position="top")
                tasks_generated += 3
            else:
                tasks_generated += 1

        # 5. إضافة مرايا تلقائية لو مطلوب
        if is_symmetric and tasks_generated > 0:
            current_tasks = self.specializations[best_spec]["tasks"][:]
            for task in current_tasks:
                if task.get("position") in ["left", "right"]:
                    mirrored_pos = "right" if task.get("position") == "left" else "left"
                    mirrored_name = f"mirrored_{task['name']}"
                    mirrored_complexity = int(task["complexity"] * 0.8)
                    self.add_task(best_spec, mirrored_name, complexity=mirrored_complexity, position=mirrored_pos)
                    tasks_generated += 1
                    logging.info(f"مرايا تلقائية: {mirrored_name} في {mirrored_pos}")

        result = {
            "best_specialization": best_spec,
            "tasks_generated": tasks_generated,
            "is_symmetric": is_symmetric,
            "matched_specs": matched_specs
        }

        logging.info(f"انتهى التوليد التلقائي: {tasks_generated} مهمة لـ {best_spec}")
        return result
          
    def apply_advanced_customizations(self, specialization: str, user_prompt: str):
        """
        تطبيق التخصيصات المتقدمة (rust, wear, إلخ) مع معاملة خاصة للأجزاء المرآة
        """
        tasks = self.specializations[specialization]["tasks"]
        if not tasks:
            return

        lower_prompt = user_prompt.lower()

        customization_effects = {
            "rust": {"increase": 1.5, "mirrored": "faded"},
            "wear": {"increase": 1.2, "mirrored": "partial"},
            "scratches": {"increase": 1.0, "mirrored": "none"},
            "weathering": {"increase": 2.0, "mirrored": "symmetric"},
            "damage": {"increase": 1.3, "mirrored": "partial"},
            "aged": {"increase": 1.4, "mirrored": "faded"}
        }

        applied = [key for key, val in customization_effects.items() if key in lower_prompt]
        if not applied:
            return

        logging.info(f"تطبيق تخصيصات: {applied}")

        for task in tasks:
            name_lower = task["name"].lower()
            is_mirrored = "mirrored" in name_lower or name_lower.startswith("mirrored_")

            base_complexity = task.get("complexity", 5)
            new_complexity = float(base_complexity)

            for cust in applied:
                effect = customization_effects[cust]
                new_complexity += effect["increase"]

                if is_mirrored:
                    mode = effect["mirrored"]
                    if mode == "faded":
                        new_complexity *= 0.7
                    elif mode == "partial":
                        new_complexity *= 0.5
                    elif mode == "none":
                        new_complexity -= effect["increase"]

            task["complexity"] = max(1, round(new_complexity))

        logging.info(f"تم تطبيق التخصيصات على {len(tasks)} مهمة")
        
    def parse_positions_and_relations(self, user_prompt: str) -> dict:
        """
        تحليل متقدم لاستخراج المواقع والعلاقات من الوصف
        """
        import re

        lower_prompt = user_prompt.lower()
        positions = {}
        relations = {}

        # مواقع أكثر دقة
        pos_pattern = r'\b(\w+(?:\s+\w+)?)\s+(?:on|at|in|above|below|behind|in front of|front of|near)\s+(top|bottom|left|right|center|front|rear|middle|above|below|back)\b'
        for match in re.finditer(pos_pattern, lower_prompt):
            part = match.group(1).strip()
            position = match.group(2).strip()
            positions[part] = position
            logging.info(f"موقع مكتشف: {part} → {position}")

        # علاقات أوسع
        rel_pattern = r'\b(\w+(?:\s+\w+)?)\s+(?:is |are |of|on|attached to|connected to|part of|mounted on|linked to)\s+(\w+(?:\s+\w+)?)\b'
        for match in re.finditer(rel_pattern, lower_prompt):
            child = match.group(1).strip()
            parent = match.group(2).strip()
            relations[child] = {"attached_to": parent}
            logging.info(f"علاقة مكتشفة: {child} → {parent}")

        return {"positions": positions, "relations": relations}
    
    def calculate_video_multiplier(self, specialization: str, task_count: int, interaction_impact: float, selected_duration: int = 6) -> float:
        """
        حساب مضاعف الفيديو ديناميكيًا
        """
        base = {
            "traditional_design": 2.2,
            "geometric_design": 2.8,
            "futuristic_design": 3.6
        }.get(specialization, 3.0)

        task_penalty = max(0, (task_count - 5) / 10.0) * 0.5
        interaction_penalty = interaction_impact * 0.08

        duration_factor = {3: 0.5, 6: 1.0, 10: 1.7, 15: 2.4}.get(selected_duration, 1.0)

        multiplier = (base + task_penalty + interaction_penalty) * duration_factor
        multiplier = round(max(1.0, multiplier), 2)

        logging.info(f"مضاعف الفيديو: {multiplier}x (قاعدة: {base}, مهام: +{task_penalty:.2f}, تفاعلات: +{interaction_penalty:.2f}, مدة: ×{duration_factor})")
        return multiplier
    
    def visualize_interaction_path(self, plane_layers: list, save_path: str = None):
        """
        رسم بياني 3D متقدم لمسار التفاعلات الفيزيائية مع دلع بصري
        """
        if not plane_layers or len(plane_layers) < 2:
            logging.info("عدد الطبقات قليل جدًا للرسم البياني")
            return

        # ضمان تام إننا في وضع headless مهما حصل
        import matplotlib
        if matplotlib.get_backend() != 'Agg':
            matplotlib.use('Agg', force=True)  # force=True عشان يغير حتى لو مستورد قبل كده

        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime

        fig = plt.figure(figsize=(12, 8), facecolor='#0f0020')
        ax = fig.add_subplot(111, projection='3d')

        positions = np.array([layer.position for layer in plane_layers])
        forces = np.array([layer.force for layer in plane_layers])

        # النقاط الملونة حسب القوة
        scatter = ax.scatter(positions[:,0], positions[:,1], positions[:,2],
                            c=forces, cmap='plasma', s=forces*40, alpha=0.9,
                            edgecolors='w', linewidth=0.5)

        # الأسهم للتفاعلات القوية
        for i in range(len(plane_layers)):
            for j in range(i + 1, len(plane_layers)):
                interaction = plane_layers[i].interact(plane_layers[j])
                if abs(interaction) > 1.5:
                    start = positions[i]
                    end = positions[j]
                    vec = end - start
                    color = 'limegreen' if interaction > 0 else 'crimson'
                    ax.quiver(start[0], start[1], start[2],
                            vec[0], vec[1], vec[2],
                            length=np.linalg.norm(vec)*0.8, normalize=True,
                            color=color, alpha=0.7, arrow_length_ratio=0.15)

        # التسميات
        for idx, layer in enumerate(plane_layers):
            ax.text(layer.position[0], layer.position[1], layer.position[2] + 0.2,
                    f"{layer.type}_{idx}", color='white', fontsize=9, weight='bold')

        # العنوان والمحاور
        ax.set_title('مسار التفاعلات الفيزيائية في التصميم', fontsize=16, color='#ff99ff')
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)

        # الخلفية الدلوعة
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        fig.patch.set_facecolor('#0f0020')
        ax.set_facecolor('#0f0020')

        # شريط الألوان
        cbar = plt.colorbar(scatter, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('قوة الطبقة (Force)', color='white', fontsize=12)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0f0020')
            logging.info(f"تم حفظ الرسم البياني في: {save_path}")
        else:
            fallback_path = f"interaction_vis_fallback_{datetime.now().strftime('%H%M%S')}.png"
            plt.savefig(fallback_path, dpi=300, bbox_inches='tight', facecolor='#0f0020')
            logging.info(f"تم حفظ تلقائي بديل: {fallback_path}")

        plt.close(fig)  # مهم جدًا: نغلق الشكل عشان ما يتراكمش في الذاكرة
            
    def generate_with_grok_api(self, prompt: str):
        """
        محاولة توليد الصورة عبر Grok Imagine API مع Retry Logic قوي
        """
        if not self.api_key:
            logging.warning("XAI_API_KEY غير موجود → استخدام Fallback مباشرة")
            return None, None

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "prompt": prompt,
            "model": "grok-imagine-aurora",
            "n": 1,
            "size": "1792x1024"
        }

        max_retries = 4
        base_delay = 2  # ثواني

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.x.ai/v1/images/generations",
                    json=payload,
                    headers=headers,
                    timeout=120
                )
                response.raise_for_status()

                data = response.json()
                if "data" not in data or not data["data"]:
                    logging.error("رد API بدون بيانات صورة")
                    return None, None

                url = data["data"][0]["url"]
                img_response = requests.get(url, timeout=60)
                img_response.raise_for_status()

                from utils import safe_filename
                filename = safe_filename("grokng_api", ".png")

                with open(filename, "wb") as f:
                    f.write(img_response.content)

                logging.info(f"تم التوليد عبر Grok API (محاولة {attempt + 1}): {filename}")
                return filename, None

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "unknown"
                if status in [429, 500, 502, 503, 504]:  # Rate limit أو server errors → retry
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logging.warning(f"خطأ HTTP {status} → إعادة المحاولة بعد {delay}s (محاولة {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    logging.error(f"خطأ HTTP غير قابل للإعادة: {status}")
                    break

            except requests.exceptions.Timeout:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"Timeout → إعادة المحاولة بعد {delay}s (محاولة {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue

            except requests.exceptions.ConnectionError:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"مشكلة اتصال → إعادة المحاولة بعد {delay}s (محاولة {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue

            except Exception as e:
                logging.error(f"خطأ غير متوقع في API (محاولة {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    break
                time.sleep(base_delay * (2 ** attempt))

        logging.info("فشل الـ API بعد كل المحاولات → الرجوع للـ Ultimate Fallback")
        return None, None
       
    def add_task(self, specialization: str, name: str, complexity: int = 5, position: str = None, deps: list = None):
        """
        إضافة مهمة جديدة لتخصص معين مع دعم الموقع والتبعيات
        """
        if specialization not in self.specializations:
            logging.warning(f"تخصص غير موجود: {specialization} → تخطي المهمة {name}")
            return

        task_dict = {
            "name": name,
            "complexity": max(1, int(complexity))  # على الأقل 1
        }
        if position:
            task_dict["position"] = position.lower()

        tasks_list = self.specializations[specialization]["tasks"]
        # تجنب التكرار
        if any(t["name"] == name for t in tasks_list):
            logging.info(f"المهمة {name} موجودة بالفعل → تخطي")
            return

        tasks_list.append(task_dict)

        if deps:
            deps_list = [d for d in deps if isinstance(d, str)]  # تنظيف
            if deps_list:
                dependencies_dict = self.specializations[specialization]["dependencies"]
                dependencies_dict[name] = deps_list
                logging.info(f"أضيفت تبعيات لـ {name}: {deps_list}")

        logging.info(f"أضيفت مهمة جديدة: {name} (تعقيد: {task_dict['complexity']})")
        
    def check_improvement_needed(self, specialization: str) -> bool:
        """تحقق لو التخصص يحتاج تحسين متقدم (علاقات، فيزياء، إلخ)"""
        tasks = self.specializations[specialization]["tasks"]
        if not tasks:
            return False
        total_complexity = sum(t.get("complexity", 5) for t in tasks)
        return len(tasks) > 2 or total_complexity > 10
    
    def calculate_render_time(self, specialization: str, is_video: bool = False, interaction_impact: float = 0.0, video_multiplier: float = 1.0) -> float:
        """
        حساب الوقت المتوقع للرندر بناءً على التعقيد والعوامل
        """
        tasks = self.specializations[specialization]["tasks"]
        if not tasks:
            return 2.0  # وقت افتراضي بسيط

        total_complexity = sum(t.get("complexity", 5) for t in tasks)

        # وزن حسب التخصص
        base_factor = {
            "traditional_design": 1.3,
            "geometric_design": 1.0,
            "futuristic_design": 1.6
        }.get(specialization, 1.2)

        base_time = total_complexity * base_factor

        # فيديو → ضعف الوقت + مضاعف
        if is_video:
            base_time *= 3.2 * video_multiplier

        # تأثير التفاعلات الفيزيائية
        base_time += interaction_impact

        # تحديث الوحدات (refreshed)
        if self.specializations[specialization]["units"]["refreshed"]:
            base_time *= 1.05

        final_time = max(2.0, round(base_time, 1))
        logging.info(f"الوقت المتوقع لـ {specialization}: {final_time}s")
        return final_time

# ==================== Run Unified Pipeline ====================
    def run_unified_pipeline(self, **kwargs) -> dict:
        specialization = kwargs.get('specialization')
        user_prompt = kwargs.get('user_prompt', '')
        is_video = kwargs.get('is_video', False)
        duration = kwargs.get('duration') or kwargs.get('selected_duration', 6)
        progress_callback = kwargs.get('progress_callback')
    
        logging.info(f"بدء Unified Pipeline لـ {specialization} (مدة: {duration}s)")

        lower_prompt = user_prompt.lower()
        full_prompt = user_prompt

        # 1. توليد المهام حسب التخصص
        full_prompt = self._generate_tasks_for_specialization(specialization, user_prompt, lower_prompt, full_prompt)

        # 2. تحسين المهام (مرايا، تخصيصات، علاقات)
        self.enhance_tasks_with_relations(specialization)

        # 3. محاكاة الفيزياء
        physics = self.simulate_physics_for_tasks(specialization)
        plane_layers = physics["layers"]
        interaction_impact = physics["interaction_impact"]

        # 4. ترتيب التسلسل الأمثل
        sequence = self.optimize_sequence(specialization)

        # 5. حساب مضاعف الفيديو
        task_count = len(self.specializations[specialization]["tasks"])
        video_multiplier = self.calculate_video_multiplier(
            specialization, task_count, interaction_impact, duration
        )
        logging.info(f"مضاعف الفيديو المحسوب: {video_multiplier:.2f}x")

        # 6. الرندر النهائي عبر Ultimate Fallback
        img_path, vid_path = self._render_with_ultimate_fallback(
            specialization=specialization,
            tasks=self.specializations[specialization]["tasks"],
            prompt=full_prompt,
            is_video=is_video,
            video_multiplier=video_multiplier
        )

        # 7. رسم بياني للتفاعلات
        vis_path = self._save_interaction_visualization(plane_layers, specialization)

        # 8. حساب الوقت الكلي
        total_time = self.calculate_render_time(specialization, is_video, interaction_impact) + interaction_impact

        result = {
            "image": img_path,
            "video": vid_path if is_video else None,
            "interaction_vis": vis_path,
            "time": total_time,
            "tasks_count": task_count,
            "interaction_impact": interaction_impact,
            "sequence": sequence
        }

        logging.info(f"انتهى Unified Pipeline لـ {specialization} بنجاح 🚀")
        return result
   
    def _generate_tasks_for_specialization(self, specialization: str, user_prompt: str, lower_prompt: str, full_prompt: str) -> str:
        """
        توليد مهام ذكية ديناميكية حسب التخصص والـ prompt باستخدام parse_prompt
        وإضافة وصف مناسب للـ full_prompt النهائي
        """
        # 1. تحليل الـ prompt مرة واحدة باستخدام الدالة المركزية
        parsed = self.parse_prompt(user_prompt, specialization=specialization)

        part_counts = parsed["part_counts"]
        positions = parsed["positions"]
        complexity_weights = parsed["complexity_weights"]
        additional_parts = parsed["additional_parts"]
        customizations = parsed["customizations"]
        is_symmetric = parsed["is_symmetric"]

        # تنظيف المهام القديمة قبل البداية
        self.specializations[specialization]["tasks"].clear()

        if specialization == "geometric_design":
            logging.info("توليد مهام ذكية لـ geometric_design بناءً على التحليل")

            # قاعدية تعقيد حسب الحجم
            base_complexity = {"large": 7, "medium": 5, "small": 3}

            # توليد مهام من الأجزاء المستخرجة
            for size, count in part_counts.items():
                if count > 0:
                    for i in range(count):
                        part_name = f"{size}_structure_{i+1}"
                        complexity = base_complexity.get(size, 5)
                        # زيادة التعقيد لو مذكور
                        if part_name in complexity_weights:
                            level = complexity_weights[part_name]
                            complexity += {"high": 4, "medium": 2, "low": 0}.get(level, 2)
                        pos = positions.get(part_name, "center")
                        self.add_task(specialization, part_name, complexity=complexity, position=pos)

            # إضافة الأجزاء الإضافية (engines, beams...)
            for part in additional_parts:
                comp = {"high": 8, "medium": 6, "low": 4}.get(part.get("complexity", "medium"), 6)
                pos = part.get("position", "center")
                self.add_task(specialization, part["name"], complexity=comp, position=pos)

            # fallback لو مفيش أجزاء محددة
            if not self.specializations[specialization]["tasks"]:
                self.add_task(specialization, "main_structure", complexity=7, position="center")
                self.add_task(specialization, "support_beam_left", complexity=5, position="left")
                self.add_task(specialization, "support_beam_right", complexity=5, position="right")

            full_prompt += ", geometric design, blueprint aesthetic, highly detailed technical drawing, precise clean lines, engineering style, symmetrical composition"

        elif specialization == "futuristic_design":
            logging.info("توليد مهام ذكية لـ futuristic_design")

            # كلمات مفتاحية لتخصيص أفضل
            if "spaceship" in lower_prompt or "ship" in lower_prompt:
                tasks = [
                    ("main_hull", 8, "center"),
                    ("left_wing", 6, "left"),
                    ("right_wing", 6, "right"),
                    ("engine_core", 9, "rear"),
                    ("cockpit", 7, "front"),
                    ("neon_thrusters", 5, "rear")
                ]
            elif "cybercity" in lower_prompt or "city" in lower_prompt:
                tasks = [
                    ("central_tower", 9, "center"),
                    ("neon_building_left", 6, "left"),
                    ("neon_building_right", 6, "right"),
                    ("holographic_billboard", 5, "top"),
                    ("flying_vehicle_1", 4, "above")
                ]
            else:
                tasks = [
                    ("main_body", 7, "center"),
                    ("energy_core", 8, "center"),
                    ("holographic_wing_left", 5, "left"),
                    ("holographic_wing_right", 5, "right")
                ]

            for name, comp, pos in tasks:
                self.add_task(specialization, name, complexity=comp, position=pos)

            full_prompt += ", futuristic sci-fi design, cyberpunk aesthetic, neon glow, holographic elements, ultra detailed, cinematic lighting, high tech"

        elif specialization == "traditional_design":
            logging.info("توليد مهام لـ traditional_design")

            if "creature" in lower_prompt or "animal" in lower_prompt or "dragon" in lower_prompt:
                tasks = [
                    ("main_body", 7, "center"),
                    ("head", 6, "front"),
                    ("wings_left", 5, "left"),
                    ("wings_right", 5, "right"),
                    ("tail", 5, "rear"),
                    ("claws", 4, "bottom")
                ]
            elif "forest" in lower_prompt or "nature" in lower_prompt:
                tasks = [
                    ("ancient_tree_center", 8, "center"),
                    ("surrounding_plants", 5, "bottom"),
                    ("mountain_background", 6, "back"),
                    ("river_flow", 4, "bottom")
                ]
            else:
                tasks = [("organic_form", 7, "center")]

            for name, comp, pos in tasks:
                self.add_task(specialization, name, complexity=comp, position=pos)

            full_prompt += ", organic natural design, highly realistic, detailed textures, natural lighting, traditional art style, beautiful environment"

        # إضافة كلمات التخصيصات لو موجودة
        if customizations:
            full_prompt += ", " + ", ".join(customizations)

        # إضافة تماثل لو مطلوب
        if is_symmetric:
            full_prompt += ", perfectly symmetrical, mirrored design"

        logging.info(f"تم توليد {len(self.specializations[specialization]['tasks'])} مهمة لـ {specialization}")
        return full_prompt

    def _render_with_ultimate_fallback(self, **kwargs) -> tuple[str, str | None]:
        """
        دالة وسيطة مرنة جدًا - تقبل أي باراميتر مهما كان اسمه
        """
        # استخراج القيم بأي اسم ممكن
        specialization = kwargs.get('specialization')
        tasks = kwargs.get('tasks', [])
        prompt = kwargs.get('prompt') or kwargs.get('user_prompt', '') or kwargs.get('full_prompt', '')
        is_video = kwargs.get('is_video', False)
        video_multiplier = kwargs.get('video_multiplier', 1.0)
        resolution = kwargs.get('resolution', (1920, 1080))
        context = kwargs.get('context', {})

        try:
            # لو عندك كلاس Draw في draw.py
            from draw import Draw
            drawer = Draw()
            img_path, video_path = drawer.generate_ultimate_fallback(
                spec=specialization,
                tasks=tasks,
                prompt=prompt,
                resolution=resolution,
                is_video=is_video,
                video_multiplier=video_multiplier,
                context=context
            )

            # لو الـ renderer functions عادية
            # from draw import generate_ultimate_fallback
            # img_path, video_path = generate_ultimate_fallback(**kwargs)

            logging.info("تم التوليد بنجاح عبر Ultimate Fallback Renderer 🚀")
            return img_path, video_path

        except Exception as e:
            logging.error(f"خطأ في Ultimate Fallback Renderer: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None, None
                                
    def _save_interaction_visualization(self, plane_layers: list, specialization: str) -> str:
        """حفظ الرسم البياني للتفاعلات الفيزيائية"""
        from datetime import datetime
        vis_path = f"interaction_vis_{specialization}_{datetime.now().strftime('%H%M%S')}.png"
        self.visualize_interaction_path(plane_layers, save_path=vis_path)
        return vis_path
   
    def generate(self, spec_from_gui: str, user_prompt: str, is_video: bool = False, progress_callback=None):
        logging.info("=== بدء Generate الذكي (API + Fallback) ===")
        if not user_prompt.strip():
            if progress_callback:
                progress_callback(0, "اكتب وصف أول يا قمري 🥺")
            return None, None, None

        # اختيار التخصص
        auto_spec = self.get_best_specialization(user_prompt)
        final_spec = spec_from_gui if spec_from_gui in self.specializations else auto_spec
        logging.info(f"التخصص النهائي: {final_spec}")

        if progress_callback:
            progress_callback(10, f"التخصص: {final_spec.replace('_', ' ').title()} 🚀")

        # ===================================
        # 1. محاولة Grok API أولاً (الأقوى والأسرع)
        # ===================================
        if progress_callback:
            progress_callback(30, "جاري المحاولة عبر Grok Imagine API السحري... ⚡")

        api_image, _ = self.generate_with_grok_api(user_prompt)  # بنعدلها تحت

        if api_image and os.path.exists(api_image):
            logging.info("نجح Grok API! هنستخدم النتيجة الخارقة دي 🌟")
            
            # حتى لو جاء من API، نعمل simulation محلي عشان visualization الدلع
            self.auto_specialize_and_generate_tasks(user_prompt, final_spec)
            physics = self.simulate_physics_for_tasks(final_spec)
            vis_path = self._save_interaction_visualization(physics["layers"], final_spec)

            if progress_callback:
                progress_callback(100, f"تم يا قمري عبر Grok API! 💜 الصورة: {os.path.basename(api_image)}")

            return api_image, None, vis_path  # صورة API + visualization محلي

        # ===================================
        # 2. لو فشل → Ultimate Fallback بكل قوته
        # ===================================
        logging.info("Grok API مش متاح دلوقتي → نروح للـ Ultimate Fallback اللي مفيش زيه 🛡️")
        if progress_callback:
            progress_callback(50, "Grok API مشغول... نروح للـ Fallback الدلوع 💜")

        # تشغيل Unified Pipeline كامل زي الأول
        try:
            result = self.run_unified_pipeline(final_spec, user_prompt, is_video)

            if progress_callback:
                progress_callback(100, f"تم يا قمري عبر Ultimate Fallback! 💜 الصورة: {os.path.basename(result['image'])}")

            return result["image"], result["video"], result["interaction_vis"]

        except Exception as e:
            logging.error(f"حتى Fallback فشل: {e}")
            if progress_callback:
                progress_callback(0, "حصل خطأ كبير 😢")
            return None, None, None
            
    def parse_prompt(self, prompt: str, specialization: str = "geometric_design") -> dict:
        """
        دالة مركزية لتحليل الـ prompt كاملاً واستخراج كل المعلومات اللي بنحتاجها.
        Returns dict موحد مع:
        - part_counts: dict لعدد الأجزاء حسب الحجم
        - positions: dict (part_name: position)
        - complexity_weights: dict (part_name: complexity level)
        - customizations: list من التخصيصات
        - additional_parts: list من dicts للأجزاء الإضافية
        - relations: dict (child: {"attached_to": parent})
        - is_symmetric: bool لو مطلوب تماثل
        """
        import re
        
        logging.info(f"بدء التحليل المركزي للـ prompt: '{prompt[:50]}...' (تخصص: {specialization})")
        
        lower_prompt = prompt.lower()
        
        # افتراضيات
        part_counts = {"large": 0, "medium": 0, "small": 0}
        positions = {}
        complexity_weights = {}
        customizations = []
        additional_parts = []
        relations = {}
        is_symmetric = False
        
        # 1. كلمات التماثل (للـ symmetry)
        symmetry_keywords = ["symmetric", "mirrored", "balanced", "twin", "bilateral", "symmetrical"]
        is_symmetric = any(word in lower_prompt for word in symmetry_keywords)
        
        # 2. استخراج التخصيصات (rust, wear, etc.)
        customization_keywords = ["rust", "wear", "scratches", "color variation", "texture variation", 
                                "material type", "weathering", "pattern variation", "damage", "aged"]
        customizations = [cust for cust in customization_keywords if cust in lower_prompt]
        
        # 3. استخراج عدد الأجزاء حسب الحجم (من size_pattern)
        size_pattern = r'(\d+)\s*(large|medium|small)\s*(parts|structures|components|beams|engines)?'
        for num, size, _ in re.findall(size_pattern, lower_prompt):
            part_counts[size] = max(part_counts[size], int(num))
        
        # 4. استخراج التفاصيل (أجزاء، مواقع، تعقيدات) من detail_pattern
        detail_pattern = r'(\d*)\s*(large|medium|small)?\s*(structure|part|beam|engine|component)\s*(?:on\s+(top|bottom|left|right|center|front|rear|above|below|middle|back))?\s*(?:with\s+(high|medium|low)\s+complexity)?'
        for num_str, size, part_type, position, complexity in re.findall(detail_pattern, lower_prompt):
            num = int(num_str) if num_str else 0
            size = size or "medium"
            part_name = f"{size}_{part_type}_{num}"
            
            if position:
                positions[part_name] = position
            if complexity:
                complexity_weights[part_name] = complexity
            
            if part_type in ["engine", "beam"]:
                additional_parts.append({
                    "name": part_name,
                    "position": position or "center",
                    "complexity": complexity or "medium"
                })
        
        # 5. استخراج المواقع العامة (من pos_pattern في fallback)
        pos_pattern = r'\b(\w+(?:\s+\w+)?)\s+(?:on|at|in|above|below|behind|in front of|front of|near)\s+(top|bottom|left|right|center|front|rear|middle|above|below|back)\b'
        for match in re.finditer(pos_pattern, lower_prompt):
            part = match.group(1).strip()
            position = match.group(2).strip()
            positions[part] = position  # هيضيف لو جديد أو يحدث
        
        # 6. استخراج العلاقات (من rel_pattern)
        rel_pattern = r'\b(\w+(?:\s+\w+)?)\s+(?:is |are |of|on|attached to|connected to|part of|mounted on|linked to)\s+(\w+(?:\s+\w+)?)\b'
        for match in re.finditer(rel_pattern, lower_prompt):
            child = match.group(1).strip()
            parent = match.group(2).strip()
            relations[child] = {"attached_to": parent}
        
        # 7. إضافات خاصة بالتخصص (لو حابب توسع بعدين)
        if specialization == "geometric_design":
            # أي إضافات خاصة هنا
            pass
        
        result = {
            "part_counts": part_counts,
            "positions": positions,
            "complexity_weights": complexity_weights,
            "customizations": customizations,
            "additional_parts": additional_parts,
            "relations": relations,
            "is_symmetric": is_symmetric
        }
        
        logging.info(f"انتهى التحليل المركزي: {len(positions)} موقع، {len(customizations)} تخصيص، {len(relations)} علاقة")
        return result
                                  
    def simulate_physics_for_tasks(self, specialization: str) -> dict:
        """
        دالة موحدة تقوم بمحاكاة الفيزياء للمهام:
        1. توليد PlaneLayer من كل مهمة
        2. حساب التفاعلات بين الطبقات
        3. إرجاع الطبقات + التأثير على الوقت
        
        Returns
        -------
        dict مع:
            - "layers": list من PlaneLayer
            - "interaction_impact": float (زيادة الوقت بالثواني)
        """
        tasks = self.specializations[specialization]["tasks"]
        if not tasks:
            logging.info(f"لا توجد مهام في {specialization} → لا محاكاة فيزيائية")
            return {"layers": [], "interaction_impact": 0.0}

        logging.info("بدء محاكاة الفيزياء...")

        # خريطة مواقع محسنة (مسافات أكبر عشان التفاعل يبقى واقعي)
        position_map = {
            "center": [0, 0, 0], "middle": [0, 0, 0],
            "left": [-3, 0, 0], "right": [3, 0, 0],
            "top": [0, 3, 0], "above": [0, 3, 0],
            "bottom": [0, -3, 0], "below": [0, -3, 0],
            "front": [0, 0, 3], "nose": [0, 0, 3],
            "rear": [0, 0, -3], "back": [0, 0, -3], "tail": [0, 0, -3]
        }

        layers = []
        for task in tasks:
            # الموقع: من المهمة لو موجود، وإلا افتراضي
            pos_key = task.get("position", "center").lower()
            position = position_map.get(pos_key, [0, 0, 0])

            # القوة بناءً على التعقيد
            force = max(1.0, task.get("complexity", 5) * 1.3)

            # نوع الطبقة حسب الاسم
            name_lower = task["name"].lower()
            if any(k in name_lower for k in ["structure", "beam", "hull", "body", "pillar"]):
                layer_type = "structural"
            elif any(k in name_lower for k in ["engine", "weapon", "shield", "cockpit"]):
                layer_type = "functional"
            else:
                layer_type = "decorative"

            layer = PlaneLayer(
                position=position,
                force=force,
                depth=1.0,
                layer_type=layer_type
            )
            layers.append(layer)

        logging.info(f"تم توليد {len(layers)} طبقة فيزيائية")

        # حساب التفاعلات
        if len(layers) < 2:
            logging.info("عدد الطبقات أقل من 2 → لا تفاعلات")
            return {"layers": layers, "interaction_impact": 0.0}

        interaction_sum = 0.0
        valid_pairs = 0

        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                inter = layers[i].interact(layers[j])
                interaction_sum += abs(inter)  # القيمة المطلقة
                valid_pairs += 1

        avg_interaction = interaction_sum / valid_pairs if valid_pairs > 0 else 0

        # تأثير أسي واقعي
        base_impact = avg_interaction * 0.08
        exponential_penalty = 0.0
        if avg_interaction > 6:
            exponential_penalty = math.pow(avg_interaction - 6, 1.6) * 0.06

        total_impact = base_impact + exponential_penalty
        total_impact = round(min(total_impact, 30.0), 2)  # حد أقصى عشان ما يتجننش

        logging.info(
            f"محاكاة الفيزياء انتهت: "
            f"متوسط التفاعل = {avg_interaction:.2f} → "
            f"زيادة وقت = {total_impact:.2f} ثانية"
        )

        return {
            "layers": layers,
            "interaction_impact": total_impact
        }
        
    def enhance_tasks_with_relations(self, specialization: str):
        tasks = self.specializations[specialization]["tasks"]
        if not tasks:
            logging.info(f"لا توجد مهام في {specialization} → لا تحسين علاقات")
            return

        task_by_name = {t["name"]: t for t in tasks}
        lower_names = {t["name"].lower(): t["name"] for t in tasks}

        added_deps = 0
        generated_rules = 0
        seen_groups = set()

        # تبعيات
        dependency_rules = {
            "engine": r"(main_hull|fuselage|hull|body|main_structure)",
            "wing": r"(main_hull|fuselage|hull|body)",
            "cockpit": r"(front|nose|main_hull|body)",
            "tail": r"(rear|main_hull|body)",
            "support": r"(main_beam|pillar|main_structure)",
            "weapon": r"(wing|hull|turret|body)",
            "shield": r"(hull|generator|body)"
        }

        for task_name, task in task_by_name.items():
            task_lower = task_name.lower()
            for child_key, parent_pattern in dependency_rules.items():
                if child_key in task_lower:
                    matches = [orig_name for l_name, orig_name in lower_names.items() if re.search(parent_pattern, l_name)]
                    if matches:
                        if "dependencies" not in task:
                            task["dependencies"] = []
                        for parent in matches:
                            if parent not in task["dependencies"]:
                                task["dependencies"].append(parent)
                                logging.info(f"تبعية تلقائية: {task_name} → {parent}")
                                added_deps += 1

        # تكامل
        keywords = ["hull", "body", "engine", "wing", "beam", "structure", "neon", "light"]
        for keyword in keywords:
            related = [orig_name for l_name, orig_name in lower_names.items() if keyword in l_name]
            if len(related) > 1:
                group_tuple = tuple(sorted(related))
                if group_tuple not in seen_groups:
                    seen_groups.add(group_tuple)
                    priority = 20 + len(related) * 8
                    self.set_integration_rule(list(group_tuple), priority=priority)
                    generated_rules += 1

        logging.info(f"انتهى تحسين العلاقات: +{added_deps} تبعية، +{generated_rules} قاعدة تكامل")
      
    def optimize_sequence(self, specialization: str) -> list:
        tasks = self.specializations[specialization]["tasks"]
        if not tasks:
            return []

        task_by_name = {t["name"]: t for t in tasks}
        sequence = []
        processed = set()

        if self.integration_rules:
            for group_tuple, priority in sorted(self.integration_rules.items(), key=lambda x: -x[1]):
                group = list(group_tuple)
                if all(g in task_by_name for g in group):
                    group_tasks = [task_by_name[n] for n in group]
                    group_tasks.sort(key=lambda t: -t.get("complexity", 5))
                    sequence.append([t["name"] for t in group_tasks])
                    processed.update(group)

        remaining = [t for t in tasks if t["name"] not in processed]
        remaining.sort(key=lambda t: -t.get("complexity", 5))
        sequence.extend([[t["name"]] for t in remaining])

        logging.info(f"التسلسل المحسن: {sequence}")
        return sequence      
               
if __name__ == "__main__":
    # مثال اختبار بسيط (اختياري، ممكن تمسحه بعدين)
    engine = GrokNGEngine()
    logging.info("GrokNGEngine تم تهيئته بنجاح!")
    logging.info(f"التخصصات المتاحة: {list(engine.specializations.keys())}")