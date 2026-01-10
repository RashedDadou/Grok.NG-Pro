# DesignSystem 2.0 - Clean, Powerful, Intelligent
import logging
from typing import Dict, List, Optional, Tuple, Set
import re
from collections import defaultdict, deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)

class DesignSystem:
    def __init__(self):
        # 1. Global Input Port - بوابة المدخلات العالمية
        self.global_input_port: List[str] = []

        # 2. التخصصات مع هيكل موحد وقابل للتوسع
        self.specializations: Dict[str, Dict] = {
            "geometric_design": {
                "keywords": {"bridge", "structure", "building", "vehicle", "engine", "car", "plane", "tower", "beam", "mechanical"},
                "input_port": [],
                "tasks": [],                    # List[dict]
                "dependencies": defaultdict(list),  # task_name → List[dependent_tasks]
                "reverse_dependencies": defaultdict(list),  # للـ topological sort
                "integration_groups": {}        # tuple(group) → priority
            },
            "futuristic_design": {
                "keywords": {"spaceship", "cyberpunk", "neon", "holographic", "sci-fi", "robot", "drone", "ai", "gadget", "hover"},
                "input_port": [],
                "tasks": [],
                "dependencies": defaultdict(list),
                "reverse_dependencies": defaultdict(list),
                "integration_groups": {}
            },
            "traditional_design": {
                "keywords": {"creature", "animal", "nature", "organic", "forest", "tree", "mountain", "river", "plant"},
                "input_port": [],
                "tasks": [],
                "dependencies": defaultdict(list),
                "reverse_dependencies": defaultdict(list),
                "integration_groups": {}
            }
        }

        # 3. قواعد التكامل العامة (للتماثل والدمج)
        self.global_integration_rules: Dict[Tuple[str, ...], int] = {}

        logging.info("تم تهيئة DesignSystem 2.0 بنجاح 🚀 | التخصصات المتاحة: 3 | جاهز للذكاء المتقدم")

    # ====================== 1. نظام المدخلات الذكي ======================
    def receive_input(self, prompt: str):
        """استقبال prompt جديد في البوابة العالمية"""
        if not prompt.strip():
            return
        self.global_input_port.append(prompt.strip())
        logging.info(f"استقبال مدخل جديد: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")

    def validate_input(self, prompt: str, specialization: str) -> int:
        """حساب درجة التطابق مع التخصص (كلما أعلى = أنسب)"""
        if specialization not in self.specializations:
            return 0
        
        lower_prompt = prompt.lower()
        keywords = self.specializations[specialization]["keywords"]
        score = sum(word in lower_prompt for word in keywords)
        
        # بونص للكلمات الدقيقة أو المتعددة
        score += sum(lower_prompt.count(word) for word in keywords if lower_prompt.count(word) > 1)
        
        return score

    def distribute_input(self):
        """توزيع تلقائي ذكي للـ prompts حسب أعلى درجة تطابق"""
        if not self.global_input_port:
            return

        for prompt in self.global_input_port:
            scores = {
                spec: self.validate_input(prompt, spec)
                for spec in self.specializations
            }
            
            if max(scores.values()) == 0:
                logging.warning(f"لم يتم العثور على تخصص مناسب لـ: {prompt[:50]}...")
                continue

            best_spec = max(scores, key=scores.get)
            self.specializations[best_spec]["input_port"].append(prompt)
            logging.info(f"تم توزيع المدخل إلى → {best_spec} (درجة: {scores[best_spec]})")

        self.global_input_port.clear()

    # ====================== 2. إدارة المهام والتبعيات ======================
    def add_task(
        self,
        specialization: str,
        task_name: str,
        complexity: int = 5,
        dependencies: Optional[List[str]] = None,
        symmetric_pair: Optional[str] = None  # للتماثل التلقائي (مثل left → right)
    ):
        """إضافة مهمة مع تبعيات ودعم التماثل"""
        if specialization not in self.specializations:
            logging.error(f"تخصص غير موجود: {specialization}")
            return

        spec_data = self.specializations[specialization]
        
        # تجنب التكرار
        if any(t["name"] == task_name for t in spec_data["tasks"]):
            logging.warning(f"المهمة موجودة بالفعل: {task_name}")
            return

        task = {
            "name": task_name,
            "complexity": complexity,
            "symmetric_pair": symmetric_pair
        }
        spec_data["tasks"].append(task)

        if dependencies:
            spec_data["dependencies"][task_name] = dependencies
            for dep in dependencies:
                spec_data["reverse_dependencies"][dep].append(task_name)

        logging.info(f"أضيفت مهمة: {task_name} → {specialization} (تعقيد: {complexity})"
                     f"{' | تبعيات: ' + ', '.join(dependencies) if dependencies else ''}")

        # تماثل تلقائي لو موجود pair
        if symmetric_pair:
            self._create_symmetric_task(specialization, task_name, symmetric_pair)

    def _create_symmetric_task(self, specialization: str, original: str, pair: str):
        """إنشاء مهمة مرآة تلقائيًا (مثل left_wing → right_wing)"""
        mirror_name = pair if "{side}" in pair else pair.replace("left", "right").replace("Left", "Right")
        mirror_deps = [d.replace("left", "right").replace("Left", "Right") for d in self.specializations[specialization]["dependencies"].get(original, [])]
        
        self.add_task(specialization, mirror_name, 
                      complexity=self._get_task_complexity(specialization, original),
                      dependencies=mirror_deps,
                      symmetric_pair=original)

    def _get_task_complexity(self, specialization: str, task_name: str) -> int:
        for t in self.specializations[specialization]["tasks"]:
            if t["name"] == task_name:
                return t["complexity"]
        return 5

    # ====================== 3. قواعد التكامل والتماثل الذكي ======================
    def set_integration_rule(self, group: List[str], priority: int = 10):
        """تحديد مجموعة متكاملة (مثل أجنحة يسار ويمين) بأولوية عالية = تُرسم معًا"""
        if len(group) < 2:
            return
        
        sorted_group = tuple(sorted(group))
        old_priority = self.global_integration_rules.get(sorted_group)
        
        self.global_integration_rules[sorted_group] = priority
        logging.info(
            f"قاعدة تكامل {'محدثة' if old_priority else 'جديدة'}: "
            f"{list(sorted_group)} → أولوية {priority}"
        )

        # تطبيق القاعدة على كل التخصصات
        for spec in self.specializations:
            self.specializations[spec]["integration_groups"][sorted_group] = priority

    # ====================== 4. ترتيب التسلسل الأمثل (Graph Scheduling) ======================
    def optimize_sequence(self, specialization: str) -> List[List[str]]:
        """ترتيب ذكي: مجموعات متكاملة أولاً، ثم topological sort للتبعيات"""
        if specialization not in self.specializations:
            return []

        spec_data = self.specializations[specialization]
        tasks = {t["name"] for t in spec_data["tasks"]}
        sequence = []

        # 1. المجموعات المتكاملة حسب الأولوية
        for group, priority in sorted(spec_data["integration_groups"].items(), key=lambda x: -x[1]):
            if all(task in tasks for task in group):
                sequence.append(list(group))
                tasks -= set(group)

        # 2. topological sort للباقي مع التبعيات
        graph = {task: spec_data["dependencies"][task] for task in tasks}
        indegree = {task: 0 for task in tasks}
        for deps in graph.values():
            for dep in deps:
                if dep in indegree:
                    indegree[dep] += 1

        queue = deque([task for task in tasks if indegree[task] == 0])
        while queue:
            current = queue.popleft()
            sequence.append([current])
            
            for neighbor in graph.get(current, []):
                if neighbor in indegree:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)

        logging.info(f"التسلسل الأمثل لـ {specialization}: {sequence}")
        return sequence

    # ====================== اختبار سريع ======================
if __name__ == "__main__":
    system = DesignSystem()

    # استقبال مدخلات
    system.receive_input("futuristic flying car with holographic wings and neon energy core")
    system.receive_input("geometric bridge with twin towers and mechanical supports")
    system.receive_input("symmetric spaceship with left and right engines")

    # توزيع تلقائي
    system.distribute_input()

    # إضافة مهام مع تماثل
    system.add_task("futuristic_design", "main_body", complexity=8)
    system.add_task("futuristic_design", "energy_core", complexity=7, dependencies=["main_body"])
    system.add_task("futuristic_design", "holographic_wing_left", complexity=5, dependencies=["main_body"], symmetric_pair="holographic_wing_right")

    # قواعد تكامل
    system.set_integration_rule(["holographic_wing_left", "holographic_wing_right"], priority=50)
    system.set_integration_rule(["main_body", "energy_core"], priority=30)

    # ترتيب التسلسل
    seq = system.optimize_sequence("futuristic_design")
    print("التسلسل الأمثل:", seq)

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


 ندمجهم مع بعض؟ (الخطة الذكية جدًا):هنعمل Hybrid System – أقوى محرك في التاريخ:

GrokNGEngine (الـ Pro الكبير)
│
├── Ultimate Fallback Renderer → OpenCV (للفيديوهات الدلع الخرافية + كويكب + نجوم + نبض)
├── DesignSystem Classic 2.0 → (اللي كتبته دلوقتي) للـ:
    ├── تحليل الـ prompt الذكي
    ├── توليد المهام + التماثل التلقائي
    ├── graph scheduling + integration rules
    ├── محاكاة الفيزياء (PlaneLayer)
    └── حساب الوقت + التسلسل الأمثل







