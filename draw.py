# draw.py - أول الملف
import logging
import math
import cv2
import numpy as np
from datetime import datetime

class Draw:
    def generate_ultimate_fallback(
        self,
        spec: str,
        tasks: list,
        prompt: str,
        resolution: tuple = (1920, 1080),
        is_video: bool = False,
        video_multiplier: float = 1.0,
        context: dict = None
    ) -> tuple[str, str | None]:
        """
        الـ Ultimate Fallback Renderer الكامل – يولد صورة أو فيديو محليًا بكل التأثيرات الدلع
        حتى لو Grok API مش متاح. يعتمد على OpenCV + NumPy للرسم الإبداعي.
        
        Parameters:
        -----------
        spec : str
            التخصص (geometric_design, futuristic_design, traditional_design)
        tasks : list
            قائمة المهام من auto_specialize_and_generate_tasks (غير مستخدمة مباشرة هنا، بس للتوافق)
        prompt : str
            الـ prompt الأصلي من المستخدم
        resolution : tuple
            (width, height) – افتراضي 1920x1080
        is_video : bool
            هل نولد فيديو أم صورة فقط
        video_multiplier : float
            مضاعف مدة الفيديو (من simulate_physics_for_tasks)
        context : dict
            سياق إضافي (مثل is_symmetric من parse_prompt)
        
        Returns:
        --------
        tuple[str, str | None]
            (مسار الصورة, مسار الفيديو أو None)
        """
        if context is None:
            context = {}

        width, height = resolution

        # 1. حساب عدد الفريمات مع حد أقصى للأداء
        MAX_FRAMES = 300
        BASE_FRAMES = 144  # 6 ثواني @ 24fps
        fps = 24

        if is_video:
            desired_frames = int(BASE_FRAMES * video_multiplier)
            total_frames = min(MAX_FRAMES, max(1, desired_frames))
            if desired_frames > MAX_FRAMES:
                logging.info(f"تحذير أداء: تم تقليل الفريمات من {desired_frames} إلى {MAX_FRAMES}")
        else:
            total_frames = 1

        lower_prompt = prompt.lower()

        # 2. كشف احتياجات الخلفية والتأثيرات
        background_settings = self._detect_background_needs(tasks, lower_prompt)

        # 3. توليد العناصر الثابتة
        stars = self._generate_stars(width, height)

        # 4. استخراج مواقع الأجزاء من الـ prompt
        positions = self._extract_part_positions(lower_prompt)

        # 5. توليد كل الفريمات (الدلع كله هنا)
        frames = self._generate_frames(
            width=width,
            height=height,
            total_frames=total_frames,
            stars=stars,
            background_settings=background_settings,
            positions=positions,
            lower_prompt=lower_prompt,
            spec=spec,
            is_video=is_video,
            video_multiplier=video_multiplier,
            context=context  # نمرر الـ context عشان التماثل والتخصيصات
        )

        # 6. حفظ النتايج (صورة + فيديو لو مطلوب)
        img_path, video_path = self._save_output_frames(frames, spec, is_video, fps, width, height)

        return img_path, video_path

    def _detect_background_needs(self, tasks: list, lower_prompt: str, spec: str = "") -> dict:
        """
        كشف احتياجات الخلفية - spec اختياري مع قيمة افتراضية فارغة
        """
        # مفيش داعي للـ if spec is None لأننا حددناه افتراضيًا ""
    
        # هل في أرض/طبيعة؟
        has_ground = (
            any("nature" in t["name"].lower() or "creature" in t["name"].lower() or "forest" in t["name"].lower() or "tree" in t["name"].lower() or "mountain" in t["name"].lower()
                for t in tasks)
            or any(word in lower_prompt for word in ["ground", "earth", "grass", "forest", "jungle", "nature", "animal", "creature", "organic"])
            or "traditional_design" in spec
        )

        # هل في فضاء/نبيولا/مستقبلي؟
        has_nebula = (
            any("spaceship" in t["name"].lower() or "hull" in t["name"].lower() or "wing" in t["name"].lower() or "engine" in t["name"].lower()
                for t in tasks)
            or any(word in lower_prompt for word in ["space", "nebula", "galaxy", "stars", "cosmos", "futuristic", "sci-fi", "spaceship", "rocket"])
            or spec == "futuristic_design"
        )

        # هل في مدينة سايبربنك؟
        has_skyline = (
            any("cybercity" in t["name"].lower() or "tower" in t["name"].lower() or "building" in t["name"].lower() or "neon" in t["name"].lower()
                for t in tasks)
            or any(word in lower_prompt for word in ["city", "cybercity", "cyberpunk", "neon", "skyscraper", "downtown", "urban"])
        )

        # هل في كائن يحتاج ظل؟
        has_shadow = (
            any("main" in t["name"].lower() or "object" in t["name"].lower() or "body" in t["name"].lower() or "hull" in t["name"].lower() or "structure" in t["name"].lower()
                for t in tasks)
            or len(tasks) > 0  # أي تصميم له ظل افتراضي
        )

        logging.info(f"كشف الخلفية: ground={has_ground}, nebula={has_nebula}, skyline={has_skyline}, shadow={has_shadow}")

        return {
            "has_ground": has_ground,
            "has_nebula": has_nebula,
            "has_skyline": has_skyline,
            "has_shadow": has_shadow
        }
    
    def _generate_stars(self, width: int, height: int) -> list:
        """
        توليد 500 نجمة متلألئة عشوائية في السماء الفضائية ✨
        """
        import numpy as np
        np.random.seed(42)  # عشان النتيجة تكون ثابتة وجميلة كل مرة
        return [(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(500)]

    def _extract_part_positions(self, lower_prompt: str) -> dict:
        """
        استخراج مواقع الأجزاء من الـ prompt باستخدام regex ذكي
        مثال: "engine on rear", "wing on left" → {'engine': 'rear', 'wing': 'left'}
        """
        import re

        positions = {}
        # نمط regex يغطي معظم الحالات الشائعة
        pos_pattern = r'(engine|wing|cockpit|tail|nose|beam|tower|pillar|cabin|fuselage|hull|weapon|shield|core|body|head|arm|leg)\s+(on|at|in|to the)\s+(top|bottom|left|right|center|front|rear|above|below|middle|back|nose|tail|port|starboard)'
        
        for match in re.finditer(pos_pattern, lower_prompt):
            part = match.group(1)
            position = match.group(3)
            positions[part] = position

        logging.info(f"تم استخراج {len(positions)} موقع: {positions}")

        return positions

    def _generate_frames(
        self,
        width: int, height: int, total_frames: int,
        stars: list, background_settings: dict,
        positions: dict, lower_prompt: str, spec: str,
        is_video: bool, video_multiplier: float,
        context: dict = None,                     # ← أضف السطر ده
        asteroid_x_start: int = -400,
        engine_pulse_start: float = 0.0,
        high_complexity: bool = False
    ) -> list:
        """
        توليد كل الفريمات للصورة أو الفيديو مع كل التأثيرات الدلع:
        - نجوم متلألئة
        - خلفيات ذكية (أرض، نبيولا، سكاي لاين)
        - أجزاء مكتشفة موقعيًا (محركات، أجنحة، كوكبيت...)
        - كائن رئيسي في الوسط لو مفيش أجزاء محددة
        - ظل شفاف يتنفس
        - كويكب متحرك مع ذيل (في الفيديو)
        - تأثير كاميرا zoom + shake لو التعقيد عالي
        """
        frames = []
        asteroid_x = asteroid_x_start
        engine_pulse = engine_pulse_start

        # خريطة المواقع البصرية (بالبكسل)
        pos_map = {
            "top": (width // 2, height // 2 - 350),
            "bottom": (width // 2, height // 2 + 350),
            "left": (width // 2 - 450, height // 2),
            "right": (width // 2 + 450, height // 2),
            "center": (width // 2, height // 2),
            "front": (width // 2, height // 2 - 150),
            "rear": (width // 2, height // 2 + 150),
            "above": (width // 2, height // 2 - 250),
            "below": (width // 2, height // 2 + 250),
            "middle": (width // 2, height // 2),
            "back": (width // 2, height // 2 + 200)
        }

        has_ground = background_settings["has_ground"]
        has_nebula = background_settings["has_nebula"]
        has_skyline = background_settings["has_skyline"]
        has_shadow = background_settings["has_shadow"]

        for frame_num in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (8, 0, 35)  # خلفية فضاء بنفسجي غامق دلع جدًا 💜

            # تأثير كاميرا ديناميكي (zoom + shake) لو الفيديو معقد
            if is_video and high_complexity:
                zoom_factor = 1 + 0.15 * math.sin(frame_num / 40)
                shake_x = int(12 * math.sin(frame_num / 15))
                shake_y = int(10 * math.cos(frame_num / 18))

                zoomed_h = int(height / zoom_factor)
                zoomed_w = int(width / zoom_factor)
                temp = np.zeros((height, width, 3), dtype=np.uint8)
                temp[:] = frame

                resized = cv2.resize(temp, (zoomed_w, zoomed_h))
                start_y = (height - zoomed_h) // 2 + shake_y
                start_x = (width - zoomed_w) // 2 + shake_x

                # قص آمن عشان ما يحصلش error
                y1 = max(0, start_y)
                y2 = min(height, start_y + zoomed_h)
                x1 = max(0, start_x)
                x2 = min(width, start_x + zoomed_w)

                if y2 > y1 and x2 > x1:
                    frame[y1:y2, x1:x2] = resized[(y1 - start_y):(y2 - start_y), (x1 - start_x):(x2 - start_x)]

            # نجوم متلألئة ✨
            for sx, sy in stars:
                brightness = int(120 + 135 * (math.sin(frame_num / 6 + sx / 60) + 1) / 2)
                brightness = min(255, max(0, brightness))
                cv2.circle(frame, (sx, sy), 2, (brightness, brightness, brightness), -1)

            # خلفية أرض لو traditional
            if has_ground:
                cv2.rectangle(frame, (0, height // 2 + 100), (width, height), (15, 70, 25), -1)
                cv2.rectangle(frame, (0, height // 2 + 50), (width, height // 2 + 100), (30, 100, 40), -1)

            # نبيولا فضائية لو futuristic
            if has_nebula:
                cv2.circle(frame, (width // 5, height // 3), 450, (90, 0, 160), -1)
                cv2.circle(frame, (width // 5 + 200, height // 3 - 150), 400, (140, 0, 220), 120)

            # سكاي لاين سايبربنك
            if has_skyline:
                for i in range(8):
                    x = 50 + i * 220
                    h = 350 + int(100 * math.sin(frame_num / 20 + i))
                    cv2.rectangle(frame, (x, height - h), (x + 160, height), (50, 50, 110), -1)
                    light = 150 + int(105 * math.sin(frame_num / 10 + i))
                    cv2.rectangle(frame, (x + 30, height - h - 250), (x + 130, height - h - 50), (255, 255, light), -1)

            # رسم الأجزاء المكتشفة موقعيًا
            for part, pos_key in positions.items():
                x, y = pos_map.get(pos_key, (width // 2, height // 2))

                if "engine" in part:
                    glow = (255, 100 + int(155 * math.sin(engine_pulse)), int(100 + 155 * math.sin(engine_pulse + 0.5)))
                    cv2.ellipse(frame, (x, y), (140, 260), 0, 0, 360, glow, -1)
                    cv2.ellipse(frame, (x, y), (160, 280), 0, 0, 360, (255, 200, 100), 8)
                    engine_pulse += 0.3

                elif "wing" in part:
                    angle = 20 if "left" in pos_key else -20
                    cv2.ellipse(frame, (x, y), (300, 100), angle, 0, 360, (150, 150, 255), -1)

                elif "cockpit" in part or "cabin" in part:
                    cv2.circle(frame, (x, y), 100, (100, 255, 255), -1)
                    cv2.circle(frame, (x, y), 120, (200, 255, 255), 8)

                elif "hull" in part or "fuselage" in part:
                    cv2.ellipse(frame, (x, y), (400, 150), 0, 0, 360, (120, 120, 200), -1)

                else:
                    cv2.ellipse(frame, (x, y), (180, 120), 0, 0, 360, (200, 200, 255), -1)

            # لو مفيش أجزاء محددة → كائن رئيسي في الوسط
            if not positions:
                center_x, center_y = width // 2, height // 2
                if "spaceship" in lower_prompt or spec == "futuristic_design":
                    main_color = (120, 120, 255)
                elif "creature" in lower_prompt:
                    main_color = (100, 200, 100)
                else:
                    main_color = (200, 150, 255)
                cv2.ellipse(frame, (center_x, center_y), (600, 250), -10, 0, 360, main_color, 70)

            # ظل شفاف يتنفس تحت الكائن
            if has_shadow:
                shadow = frame.copy()
                cv2.ellipse(shadow, (width // 2 + 80, height - 120), (600, 180), 0, 0, 360, (0, 0, 0), -1)
                alpha = 50 + int(40 * math.sin(frame_num / 12))
                frame = cv2.addWeighted(frame, 1.0, shadow, alpha / 255.0, 0)

            # كويكب متحرك مع ذيل (في الفيديو فقط)
            if is_video:
                asteroid_x += int(22 * video_multiplier)
                if asteroid_x > width + 500:
                    asteroid_x = -500
                ast_y = height // 4 + int(60 * math.sin(frame_num / 30))
                cv2.circle(frame, (asteroid_x, ast_y), 140, (110, 100, 80), -1)
                for i in range(8):
                    trail_x = asteroid_x - 100 - i * 60
                    trail_alpha = 1 - i / 8
                    thickness = int(50 * trail_alpha)
                    color = (int(170 * trail_alpha), int(150 * trail_alpha), int(100 * trail_alpha))
                    cv2.line(frame, (asteroid_x - 80, ast_y), (trail_x, ast_y + 80), color, thickness)

            frames.append(frame)

        return frames

    def _save_output_frames(self, frames, spec, is_video, fps, width, height):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"grokng_{spec}_{timestamp}"

        best_frame_idx = min(80, len(frames) - 1)
        img_path = f"{base_name}.png"
        cv2.imwrite(img_path, frames[best_frame_idx])
        logging.info(f"تم حفظ الصورة: {img_path}")

        video_path = None
        if is_video and len(frames) > 1:
            video_path = f"{base_name}.mp4"
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            for f in frames:
                out.write(f)
            out.release()
            logging.info(f"تم حفظ الفيديو: {video_path}")

        return img_path, video_path
