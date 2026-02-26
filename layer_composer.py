def compose_layers_from_engines(
    env_prompt: str,
    trad_prompt: str,
    geo_prompt: str,
    global_seed: int = 42,
    collision_threshold: float = 0.8,
    emotional_amplifier: float = 2.0,
) -> list[PlaneLayer]:
    from environment_design_engine import environment_design_engine
    from traditional_design_engine import traditionalDesignEngine
    from geometric_design_engine import geometric_design_engine
    
    np.random.seed(global_seed)

    layers: List[PlaneLayer] = []

    # 1. تعريف المحركات (استخدم الأسماء الصحيحة الموجودة فعليًا في المشروع)
    engine_factories = {
        "environment": lambda: environment_design_engine(),   # ← استخدم الاسم الحقيقي
        "traditional": lambda: traditionalDesignEngine(),
        "geometric":   lambda: geometric_design_engine(),
    }

    prompt_map = {
        "environment": env_prompt,
        "traditional": trad_prompt,
        "geometric":   geo_prompt,
    }

    z_levels = {
        "environment": -1.0,   # أبعد
        "geometric":    0.0,
        "traditional":  1.0,   # أقرب
    }

    for name, factory in engine_factories.items():
        try:
            engine = factory()
        except Exception as e:
            print(f"فشل إنشاء محرك {name}: {e}")
            continue

        prompt = prompt_map[name]

        # هنا نحتاج طريقة موثوقة لاستخراج الكيانات/العناصر
        # بديل مؤقت – نقسم النص يدويًا أو نستخدم تحليل بسيط
        entities = extract_entities_simple(prompt)  # ← دالة مساعدة تحت

        base_z = z_levels[name]

        for i, entity in enumerate(entities):
            # توزيع عشوائي حول المستوى الأساسي
            z = base_z + np.random.uniform(-0.15, 0.15)

            layer = PlaneLayer(
                position=[np.random.uniform(-1.8, 1.8),
                        np.random.uniform(-1.0, 1.0),
                        z],
                force=1.0 + len(entity) * 0.25,
                mass=1.0 + len(entity) * 0.15,
                label=f"{name}_{entity[:12]}"
            )
            layers.append(layer)

    # 2. ترتيب حسب العمق (z)
    layers.sort(key=lambda p: p.position[2])

    # 3. محاكاة التصادمات + التأثير العاطفي (x²)
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            p1 = layers[i]
            p2 = layers[j]

            dist = p1.distance_to(p2)
            if dist < collision_threshold and dist > 1e-6:
                direction = (p2.position - p1.position) / dist
                # قوة تربيعية (x² effect)
                strength = (p1.force * p2.force) * (emotional_amplifier ** 2) / (dist ** 2 + 0.01)

                force_vec = direction * strength

                p1.apply_force(-force_vec, dt=0.08)
                p2.apply_force( force_vec * 0.85, dt=0.08)

                print(f"تصادم: {p1.label} ↔ {p2.label}  | dist={dist:.3f} | قوة={strength:.3f}")

    return layers
