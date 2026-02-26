# layer_plane.py
print("→ بدأ layer_plane.py")

"""
موديول PlaneLayer – طبقات مرجعية مع تفاعلات فيزيائية/عاطفية مبسطة
...
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Any, Optional

from memory_manager import GenerativeMemoryManager
from environment_design_engine import environment_design_engine
from geometric_design_engine import geometric_design_engine
from traditional_design_engine import traditionalDesignEngine
        
import logging
logger = logging.getLogger(__name__)

class PlaneLayer:
    def __init__(
        self,
        position: list | tuple | np.ndarray,
        force: float = 1.0,
        depth: float = 1.0,
        label: str = "",
        color: str = "gray",
        mass: float = 1.0,
    ):
        """
        Args:
            position: إحداثيات [x, y, z] أو [x, y]
            force: قوة التأثير / الجاذبية النسبية
            depth: عمق الطبقة – يؤثر على الترتيب والوزن
            label: اسم وصفي (hand, cup, tree, spiral, ...)
            color: لون تمثيلي للـ visualization والـ debug
            mass: كتلة لاستخدامها في حساب التسارع والتفاعلات
        """
        self.position = np.array(position, dtype=float)
        if self.position.shape not in ((2,), (3,)):
            raise ValueError("position يجب أن يكون 2 أو 3 أبعاد")

        self.force    = float(force)
        self.depth    = float(depth)
        self.label    = str(label)
        self.color    = str(color)
        self.mass     = float(mass)
        self.image_ref: Optional[Any] = None   # PIL.Image أو path أو texture

        # لتتبع الحركة في المحاكاة (اختياري)
        self.velocity = np.zeros_like(self.position)
        self.trail: list = [self.position.copy()[:2]]  # فقط x,y للرسم 2D

        # ← هذا السطر اللي كان ناقص
        self.metadata: dict = {}   # قاموس حر لتخزين أي بيانات إضافية (source, type, parent, ...)

    def distance_to(self, other: 'PlaneLayer') -> float:
        """المسافة الإقليدية إلى طبقة أخرى"""
        return np.linalg.norm(self.position - other.position)

    def raw_interaction(self, other: 'PlaneLayer') -> float:
        """
        القوة التفاعلية الأساسية (مشابهة لقانون الجاذبية / كولوم المبسط)
        """
        dist = self.distance_to(other)
        return (self.force + other.force) / (dist + 1e-5)

    def x2_effect(self, other: 'PlaneLayer') -> float:
        dist = self.distance_to(other)
        if dist < 0.01:
            dist = 0.01
        influence = (self.force * other.force) / (dist ** 1.5)   # 1.5 → سقوط أبطأ من 2
        return min(influence, 10.0)   # حد أعلى لمنع انفجار الأرقام

    def extract_entities_simple(prompt: str) -> List[str]:
        """تقسيم بسيط جدًا – يمكن تحسينه بـ LLM أو spacy لاحقًا"""
        words = prompt.replace(",", " ").replace(" و ", " ").split()
        entities = [w for w in words if len(w) > 3 and w.lower() not in {"with", "and", "in", "on", "at"}]
        return entities or ["element"]

    def apply_force(self, force_vector: np.ndarray, dt: float = 0.05, damping: float = 0.92):
        acceleration = force_vector / self.mass
        self.velocity += acceleration * dt
        self.velocity *= damping          # ← يقلل السرعة تدريجيًا
        self.position += self.velocity * dt
        self.trail.append(self.position.copy()[:2])
        if len(self.trail) > 80:
            self.trail.pop(0)

    def total_influence_from(self, others: list['PlaneLayer']) -> float:
        return sum(self.x2_effect(other) for other in others)

    def __repr__(self):
        return (f"PlaneLayer({self.label!r} | pos={self.position} | "
                f"force={self.force:.2f} | depth={self.depth:.2f} | {self.color})")

    def __str__(self) -> str:
        return f"{self.label} @ {self.position.round(2)} (force={self.force:.2f})"

class LayerComposer:
    """مسؤول عن تجميع طبقات من المحركات المختلفة + محاكاة التفاعلات الأولية"""

    def __init__(self, engine_registry: dict = None):
        self.engine_registry = engine_registry or {
            "environment": lambda: environment_design_engine(),
            "traditional": lambda: traditionalDesignEngine(),
            "geometric":   lambda: geometric_design_engine(),
        }
        self.z_levels = {
            "environment": -1.0,
            "geometric":    0.0,
            "traditional":  1.0,
        }

    def compose_from_prompts(
        self,
        prompts: dict[str, str],
        global_seed: int = 42,
        collision_threshold: float = 0.8,
        emotional_amplifier: float = 2.0,
        default_resolution: Tuple[int, int] = (1280, 720),
    ) -> List[PlaneLayer]:
        layers: List[PlaneLayer] = []
        np.random.seed(global_seed)

        for engine_name, prompt in prompts.items():
            if engine_name not in self.engine_registry:
                continue

            engine = self.engine_registry[engine_name]()

            try:
                if engine_name == "environment":
                    layer = self._create_environment_layer(engine, prompt, default_resolution)
                    if layer:
                        layers.append(layer)

                elif engine_name == "traditional":
                    layer_group = self._create_traditional_layers(engine, prompt, default_resolution)
                    if layer_group:
                        layers.extend(layer_group)

                elif engine_name == "geometric":
                    result = engine.generate_layer(
                        prompt=prompt,
                        target_size=default_resolution,
                        is_video=False
                    )

                    if not result.success:
                        logger.warning(f"فشل geometric: {result.message}")
                        continue

                    data = result.output_data or {}
                    enhanced = data.get("enhanced_prompt", prompt)
                    entities = data.get("entities", [])
                    symmetry = data.get("symmetry", "medium")
                    metadata = data.get("metadata", {})

                    # ────────────── قرارات مخصصة بناءً على geometric ───────────────
                    layer_group = self._create_geometric_layers(
                        entities=entities,
                        symmetry=symmetry,
                        metadata=metadata,
                        base_z=self.z_levels.get("geometric", 0.0),
                        seed=global_seed + abs(hash("geometric")) % (2**32)
                    )

                    if layer_group:
                        layers.extend(layer_group)

                else:
                    # باقي المحركات العامة
                    layer_group = self._create_generic_layers(engine_name, prompt, global_seed)
                    if layer_group:
                        layers.extend(layer_group)

            except Exception as e:
                logger.error(f"خطأ في محرك {engine_name}: {e}", exc_info=True)
                continue

        layers.sort(key=lambda p: p.position[2] if len(p.position) >= 3 else 0)
        self._apply_initial_collisions(layers, collision_threshold, emotional_amplifier)

        # طباعة تفصيلية للطبقات النهائية
        print("\n┌─────── الطبقات النهائية (مرتبة حسب z) ───────┐")
        for i, layer in enumerate(layers, 1):
            pos = layer.position.round(2)
            pos_str = f"[{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]" if len(pos) == 3 else f"[{pos[0]:6.2f}, {pos[1]:6.2f}]"
            print(f"  {i:3}. {layer.label:30} | z = {pos[2]:6.2f} | force = {layer.force:5.2f} | color = {layer.color:10} | mass = {layer.mass:5.2f}")
        print(f"  • إجمالي الطبقات: {len(layers)}")
        print("└───────────────────────────────────────────────────────┘")

        logger.info(f"تم إنشاء {len(layers)} طبقة من {len(prompts)} محرك")
        return layers

    def _create_environment_layer(self, engine, prompt, resolution):
        result = engine.design( ... )
        if not result.success:
            return None

        layer = PlaneLayer(
            position=[0.0, 0.0, self.z_levels.get("environment", -1.0)],
            force=0.35,
            depth=15.0,
            label="Environment Background",
            color="midnightblue",
            mass=200.0
        )
        layer.image_ref = getattr(result.layer, "image", None)
        layer.metadata = {
            "prompt": prompt,
            "enhanced": result.enhanced_description if hasattr(result, "enhanced_description") else "",
            "resolution": resolution
        }
        return layer

    def _create_traditional_layers(self, engine, prompt, resolution):
        result = engine.generate_layer(prompt=prompt, target_size=resolution, is_video=False)
        if not result.success:
            return []

        data = result.output_data or {}
        enhanced = data.get("enhanced_prompt", prompt)
        subject = data.get("main_subject", "character")
        mood = data.get("mood", "neutral")
        entities = data.get("entities", [])

        layers = []

        # الطبقة الرئيسية
        pos = np.array([
            np.random.uniform(-0.12, 0.12),
            np.random.uniform(0.05, 0.30),
            self.z_levels.get("traditional", 1.2)
        ])

        main = PlaneLayer(
            position=pos,
            force=1.3 + len(enhanced) * 0.003,
            depth=1.8,
            label=subject,
            color=self._suggest_color_from_mood(mood),
            mass=3.2 + np.random.uniform(-0.8, 1.5)
        )
        main.metadata = {
            "source": "traditional_engine",
            "enhanced_prompt": enhanced,
            "mood": mood,
            "main_subject": subject,
            "entities": entities,
            "is_foreground": True,
            **data.get("metadata", {})
        }
        layers.append(main)

        # العناصر الثانوية
        for entity in entities[1:4]:
            offset = np.random.uniform(-0.35, 0.35, 2)
            extra_pos = np.append(pos[:2] + offset, pos[2] + np.random.uniform(0.08, 0.25))
            extra = PlaneLayer(
                position=extra_pos,
                force=0.7,
                depth=extra_pos[2],
                label=entity[:24],
                color="gray",
                mass=0.9
            )
            extra.metadata["parent_subject"] = subject
            layers.append(extra)

        return layers

    def _create_geometric_layers(
        self,
        entities: List[str],
        symmetry: str,
        metadata: dict,
        base_z: float = 0.0,
        seed: int = 42
    ) -> List[PlaneLayer]:
        np.random.seed(seed)
        layers = []

        # استخدام metadata بشكل حقيقي
        preferred_color = metadata.get("preferred_color", "silver")
        main_pattern = metadata.get("main_pattern", "geometric core")
        particle_density = metadata.get("particle_density", 12)

        # ─── 1. تحديد الأنماط الرئيسية ───────────────────────
        has_spiral   = any(w in entities for w in ["spiral", "golden", "دوامة"])
        has_grid     = any(w in entities for w in ["grid", "شبكة", "hex", "hexagonal"])
        has_pattern  = any(w in entities for w in ["pattern", "نمط", "repeat"])
        is_high_sym  = symmetry == "high"

        # ─── 2. طبقة رئيسية مركزية ─────────────
        if has_spiral or len(entities) > 0:
            main_entity = "golden spiral" if has_spiral else (entities[0] if entities else main_pattern)
            
            main_layer = PlaneLayer(
                position=[0.0, 0.05, base_z + 0.4],
                force=1.1,
                depth=1.2,
                label=main_entity,                     # ← استخدمنا main_pattern لو مفيش entity
                color=preferred_color if has_spiral else "gold",  # ← استخدمنا preferred_color
                mass=2.8
            )
            main_layer.metadata = {
                "source": "geometric",
                "type": "primary_element",
                "symmetry_influence": symmetry,
                "is_central": True
            }
            layers.append(main_layer)

        # ─── 3. طبقات خلفية / شبكة ────────────────────────────────────
        if has_grid or has_pattern:
            grid_count = 5 if is_high_sym else 3
            
            for i in range(grid_count):
                x = (i % 3 - 1) * 0.4 + np.random.uniform(-0.08, 0.08)
                y = (i // 3 - 1) * 0.3 + np.random.uniform(-0.06, 0.06)
                
                grid_layer = PlaneLayer(
                    position=[x, y, base_z - 0.3 + np.random.uniform(-0.15, 0.15)],
                    force=0.45,
                    depth=0.6,
                    label=f"grid background {i+1}",
                    color="slategray",
                    mass=15.0
                )
                grid_layer.metadata["type"] = "background_pattern"
                layers.append(grid_layer)

        # ─── 4. نقاط / جسيمات لامعة ────────────────────
        particle_count = particle_density if is_high_sym else 6   # ← استخدمنا particle_density
        for _ in range(particle_count):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.15, 0.65)
            px = np.cos(angle) * dist
            py = np.sin(angle) * dist
            
            p = PlaneLayer(
                position=[px, py, base_z + np.random.uniform(0.1, 0.6)],
                force=0.25,
                depth=0.9 + np.random.uniform(-0.2, 0.2),
                label="particle",
                color="white",
                mass=0.4
            )
            p.metadata["type"] = "highlight_particle"
            layers.append(p)

        return layers

    def _create_geometric_specific_layers(self, entities: list[str], seed: int) -> list[PlaneLayer]:
        """
        إنشاء طبقات مخصصة للمحرك الهندسي بناءً على الكيانات والأنماط
        """
        np.random.seed(seed)
        layers = []
        base_z = self.z_levels.get("geometric", 0.0)

        # ─── كشف الأنماط الرئيسية ───────────────────────────────────────────────
        has_spiral    = any(w in entities for w in ["spiral", "golden", "logarithmic", "دوامة"])
        has_grid      = any(w in entities for w in ["grid", "شبكة", "hex", "hexagonal", "honeycomb"])
        has_fractal   = any(w in entities for w in ["fractal", "مفرق", "self-similar"])
        has_polygon   = any(w in entities for w in ["polygon", "مضلع", "triangle", "pentagon"])
        is_high_sym   = "high" in entities or "symmetry" in entities or "تناظر" in entities

        # ─── 1. العنصر المركزي / الرئيسي (غالباً spiral أو الشكل الأكبر) ────────
        if has_spiral or len(entities) > 0:
            main_label = "golden spiral" if has_spiral else (entities[0] if entities else "geometric core")

            main = PlaneLayer(
                position=[0.0, 0.08, base_z + 0.45],  # قريب من المنتصف، عمق متوسط-أمامي
                force=1.15,
                depth=1.3,
                label=main_label,
                color="gold" if has_spiral else "silver",
                mass=3.2
            )
            main.metadata.update({
                "type": "primary_geometric",
                "is_central": True,
                "symmetry_influence": "high" if is_high_sym else "medium"
            })
            layers.append(main)

        # ─── 2. طبقة خلفية شبكية (إذا وجد grid أو pattern) ────────────────────────
        if has_grid or "pattern" in entities:
            grid_layers = 4 if is_high_sym else 2

            for i in range(grid_layers):
                # توزيع شبكي بسيط
                col = i % 2 - 0.5
                row = i // 2 - 0.5
                x = col * 0.55
                y = row * 0.4

                grid_l = PlaneLayer(
                    position=[x, y, base_z - 0.4 + np.random.uniform(-0.1, 0.1)],
                    force=0.38,
                    depth=0.5,
                    label=f"grid layer {i+1}",
                    color="dimgray",
                    mass=18.0   # كتلة عالية = تأثير خفيف جدًا على الأمام
                )
                grid_l.metadata["type"] = "background_grid"
                layers.append(grid_l)

        # ─── 3. عناصر فرعية (fractal branches, polygon instances, particles) ──────
        secondary_count = 8 if is_high_sym else 4

        for i in range(secondary_count):
            angle = i * (2 * np.pi / secondary_count) + np.random.uniform(-0.3, 0.3)
            dist = np.random.uniform(0.22, 0.68)

            x = np.cos(angle) * dist
            y = np.sin(angle) * dist
            z_var = np.random.uniform(-0.15, 0.35)

            sec = PlaneLayer(
                position=[x, y, base_z + z_var],
                force=0.45 + (0.08 if has_fractal else 0),
                depth=0.9 + z_var,
                label="secondary element",
                color="lightsteelblue" if has_polygon else "white",
                mass=0.9
            )

            sec.metadata.update({
                "type": "secondary_detail",
                "parent_pattern": "fractal" if has_fractal else "polygon" if has_polygon else "particle"
            })
            layers.append(sec)

        return layers

    def _suggest_color_from_mood(self, mood: str) -> str:
        """اقتراح لون بناءً على المزاج/الجو"""
        mood_colors = {
            "happy": "gold",
            "sad": "lightblue",
            "angry": "crimson",
            "mysterious": "purple",
            "calm": "teal",
            "epic": "darkorange",
            "neutral": "wheat",
            "horror": "darkred",
            "enchanted": "violet",
            "majestic": "royalblue",
        }
        return mood_colors.get(mood.lower(), "rosybrown")  # fallback

    def adjust_layers_for_physics(self, layers, prompt, resolution, collision_threshold=0.15, emotional_amplifier=1.0):
        # حاليًا مجرد إرجاع الطبقات كما هي + log
        logger.info(f"[Physics stub] تعديل {len(layers)} طبقة – حاليًا بدون تغيير فعلي")
        return layers

    def load_environment_from_memory(self, prompt_hash: str) -> List[PlaneLayer]:
        # إنشاء instance مؤقت لو مش موجود
        memory = GenerativeMemoryManager()  # أو استخدم self.memory_manager لو موجود
        data = memory.load_environment_elements(prompt_hash)
        
        if not data or not data.get("elements"):
            logger.warning(f"لا بيانات بيئية لـ {prompt_hash}")
            return []

        layers = []
        for elem in data["elements"]:
            pos = elem.get("position", [0, 0, 0])
            if isinstance(pos, list):
                pos = np.array(pos)

            layer = PlaneLayer(
                position=pos,
                force=elem.get("properties", {}).get("force", 1.0),
                depth=-pos[2] if len(pos) > 2 else 0.0,
                label=elem.get("type", "unknown"),
                color=elem.get("properties", {}).get("color", "gray"),
                mass=elem.get("properties", {}).get("size", 1.0) or 1.0
            )
            layer.metadata = elem.get("properties", {})
            layers.append(layer)

        logger.info(f"تم تحويل {len(layers)} عنصر بيئي إلى PlaneLayer من {prompt_hash}")
        return layers
    
    def _extract_entities(self, prompt: str) -> list[str]:
        import re
        text = re.sub(r'[^\w\s]', ' ', prompt.lower())  # إزالة علامات
        words = text.split()
        stopwords = {
            'في', 'على', 'مع', 'من', 'إلى', 'و', 'أو', 'ف', 'ب', 'ل', 'ك', 
            'in', 'on', 'with', 'and', 'or', 'at', 'by', 'for', 'the', 'a', 'an'
        }
        entities = []
        for w in words:
            if len(w) > 3 and w not in stopwords:
                entities.append(w)
        return list(dict.fromkeys(entities))  # إزالة التكرار مع الحفاظ على الترتيب

    def _apply_initial_collisions(self, layers: List[PlaneLayer], threshold: float, amp: float):
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                p1, p2 = layers[i], layers[j]
                dist = p1.distance_to(p2)
                if dist < threshold:
                    # نفس منطق التصادم السابق ...
                    pass

# ────────────────────────────────────────────────
# مثال استخدام سريع (للاختبار عند تشغيل الملف مباشرة)
# ────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from typing import List, Dict, Any, Tuple, Optional

    # ضبط backend مناسب للرسم (جرب غيّر إذا ما اشتغل)
    matplotlib.use('TkAgg')   # أو جرب 'Qt5Agg' أو 'WXAgg' حسب نظامك

    # إعداد logging مرتب
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    print("→ بدأ layer_plane.py (اختبار مفصّل ومنظّم)")

    composer = LayerComposer()

    # ────────────── مرحلة 1: تحميل بيئة من الذاكرة ──────────────
    print("\n" + "═" * 70)
    print("مرحلة 1: تحميل بيئة سابقة من الذاكرة")
    print("═" * 70)

    env_prompt_hash = "b8716032fa7781ce"
    env_layers: List[PlaneLayer] = composer.load_environment_from_memory(env_prompt_hash)

    print(f"  • تم تحميل {len(env_layers)} طبقة بيئية من الهاش: {env_prompt_hash}")
    if env_layers:
        print("  • أول 5 طبقات (مثال):")
        for i, layer in enumerate(env_layers[:5], 1):
            pos = layer.position.round(2)
            pos_str = f"[{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]" if len(pos) == 3 else f"[{pos[0]:6.2f}, {pos[1]:6.2f}]"
            print(f"    {i:2}. {layer.label:25} | pos = {pos_str} | force = {layer.force:.2f} | color = {layer.color}")
    else:
        print("  • لا طبقات بيئية محملة")
    print("═" * 70 + "\n")

    # ────────────── مرحلة 2: إنشاء طبقات يدوية واختبار التفاعل ──────────────
    print("مرحلة 2: إنشاء طبقات يدوية بسيطة + حساب التفاعلات")
    print("═" * 70)

    hand = PlaneLayer([0.0, 0.0, 0.0], force=0.35, label="Hand", color="sienna")
    cup  = PlaneLayer([1.2, 0.3, 0.0], force=0.50, label="Cup",  color="cyan")
    face = PlaneLayer([3.5, 0.0, 0.5], force=0.10, label="Face", color="lightpink")

    layers_manual = [hand, cup, face]

    print("  • الطبقات اليدوية المُنشأة:")
    for layer in layers_manual:
        pos = layer.position.round(2)
        print(f"    - {layer.label:6} @ {pos} | force={layer.force:.2f} | color={layer.color}")

    hc = hand.x2_effect(cup)
    cf = cup.x2_effect(face)
    total = hc + cf

    print("\n  • التفاعلات (x² effect):")
    print(f"      Hand → Cup   : {hc:.3f}")
    print(f"      Cup  → Face  : {cf:.3f}")
    print(f"      التأثير الكلي: {total:.3f}")
    print("═" * 70 + "\n")

    # ────────────── مرحلة 3: رسم بسيط للمواقع (اختبار الـ visualization) ──────────────
    print("مرحلة 3: رسم المواقع الأولية + حفظ الصورة")
    print("═" * 70)

    try:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        for layer in layers_manual:
            ax.scatter(*layer.position[:2], s=400, c=layer.color, label=layer.label, zorder=10, edgecolor='black', linewidth=1.5)
            ax.text(layer.position[0] + 0.15, layer.position[1] + 0.15, layer.label, fontsize=11, fontweight='bold')

        ax.plot([hand.position[0], cup.position[0]], [hand.position[1], cup.position[1]], '--', c='gray', alpha=0.7, linewidth=1.5)
        ax.plot([cup.position[0], face.position[0]], [cup.position[1], face.position[1]], '--', c='gray', alpha=0.7, linewidth=1.5)

        ax.set_title("مواقع الطبقات اليدوية + التأثيرات (x²)", fontsize=14, fontweight='bold')
        ax.set_xlabel("X المحور", fontsize=12)
        ax.set_ylabel("Y المحور", fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.set_aspect('equal')

        # حفظ الصورة
        save_path = "layer_positions_manual_test.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  • تم حفظ الرسم البياني بنجاح في: {save_path}")
        print("  • افتح الصورة لترى التوزيع المكاني + الروابط بين الطبقات")

        # محاولة عرض (لو الشاشة متاحة)
        plt.show(block=False)
        print("  • تم عرض النافذة (إذا كانت الشاشة متاحة)")
    except ImportError:
        print("  • matplotlib غير مثبت → قم بـ pip install matplotlib")
    except Exception as e:
        print(f"  • فشل في الرسم أو الحفظ: {str(e)}")
    print("═" * 70 + "\n")

    print("→ انتهى الاختبار المفصّل بنجاح!")
    print("  • جرب عدّل الطبقات أو أضف prompts جديدة في الكود أعلاه")
    
        # ────────────── مرحلة 4: اختبار compose_from_prompts الكامل (التجميع الحقيقي) ──────────────
    print("\n" + "═" * 80)
    print("مرحلة 4: تجميع طبقات كامل من prompts متنوعة (environment + traditional + geometric)")
    print("═" * 80)

    test_prompts = {
        "environment": "غابة سحرية مضيئة في الليل مع ضباب خفيف وأضواء زرقاء وأشجار طويلة",
        "traditional": "فتاة إلف شابة بفستان فضي طويل، تعبير حزن عميق، شعر أشقر يتطاير مع الريح",
        "geometric": "golden spiral with hexagonal grid and high symmetry, sacred geometry style, intricate patterns"
    }

    print("الـ prompts المُرسلة:")
    for engine, p in test_prompts.items():
        print(f"  • {engine.upper():12}: {p[:80]}...")

    print("\nجاري التجميع...")
    final_layers = composer.compose_from_prompts(
        prompts=test_prompts,
        global_seed=123,
        default_resolution=(1920, 1080)
    )

    print("\n→ انتهى التجميع الكامل!")
    print(f"إجمالي الطبقات الناتجة من التجميع: {len(final_layers)}")
    print("  • الجدول التفصيلي موجود أعلاه (داخل الدالة نفسها)")
    print("═" * 80 + "\n")