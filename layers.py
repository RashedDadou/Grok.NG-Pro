# layers.py
import numpy as np
from typing import List

class PlaneLayer:
    """
    طبقة فيزيائية تمثل جزء من التصميم (مثل جناح، محرك، هيكل رئيسي...)
    بتستخدم في محاكاة التفاعلات الفيزيائية والترتيب ثلاثي الأبعاد
    """
    def __init__(
        self,
        position: List[float],
        force: float = 5.0,
        depth: float = 1.0,
        layer_type: str = "structural",  # structural, functional, decorative
        name: str = "unnamed_layer"
    ):
        self.position = np.array(position, dtype=float)
        self.force = float(force)        # قوة الجذب/الدفع (تعقيد × عامل)
        self.depth = float(depth)        # عمق في المشهد (للـ depth sorting)
        self.type = layer_type
        self.name = name

    def interact(self, other: 'PlaneLayer') -> float:
        """
        حساب التفاعل الفيزيائي بين طبقتين (جذب أو تنافر)
        بناءً على القوة والمسافة، مع تهدئة قوية عشان ما يبقاش فلكي
        """
        if other is self:
            return 0.0

        vec = other.position - self.position
        distance = np.linalg.norm(vec)

        if distance < 0.5:  # تجنب division by zero أو انفجار
            distance = 0.5

        # قانون مشابه للجاذبية، بس مع تهدئة قوية جدًا
        raw_interaction = (self.force * other.force) / (distance ** 2)

        # تهدئة أسية + fixed cap
        damped = raw_interaction / (1 + raw_interaction / 1000)  # soft cap
        interaction = np.clip(damped * 0.00005, -50, 50)  # حد أقصى منطقي

        # إشارة حسب النوع (structural مع structural = جذب، functional مع structural = دفع خفيف)
        if self.type == "structural" and other.type == "structural":
            return -abs(interaction)  # جذب (سالب)
        elif "functional" in (self.type, other.type):
            return abs(interaction) * 0.3  # دفع خفيف
        else:
            return interaction * 0.1

    def get_stability_score(self) -> float:
        """نقاط الاستقرار (كل ما أعلى كل ما أحسن في الترتيب)"""
        return self.force / (self.depth + 1)

    def __repr__(self):
        return f"<PlaneLayer '{self.name}' | pos={self.position.tolist()} | force={self.force:.1f} | type={self.type}>"