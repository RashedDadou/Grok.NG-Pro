Grok.NG Pro v1.1The Next Generation Local AI Design Engine 

Built with love, passion, and endless nights of debuggingGrok.NG Pro Screenshot
(A futuristic flying car generated with the prompt "Car" – holographic wings, pulsing energy core, asteroid trail, twinkling stars, and breathing shadow)OverviewGrok.NG Pro is a fully local, standalone AI-powered design generation system that turns simple text prompts into stunning images and animated videos — no internet or external API required after setup.It combines intelligent prompt analysis, physics-based simulation, smart task generation, symmetry detection, and a powerful OpenCV-based fallback renderer to create cinematic, futuristic, geometric, or traditional designs with incredible visual effects.Born on January 8, 2026 — the day the first video successfully rendered.Key Features3 Design Specializationsfuturistic_design – Cyberpunk, spaceships, neon, holographic effects
geometric_design – Bridges, vehicles, mechanical structures, blueprints
traditional_design – Organic, nature, creatures, environments

Smart Prompt Analysis & Task GenerationAutomatically detects key elements and generates appropriate parts (e.g., "Car" → main_body + energy_core + holographic wings left/right)
Supports symmetry detection and integration rules

Physics Simulation with PlaneLayerLayers interact with each other
Interaction impact affects animation duration and visual intensity

Ultimate Fallback Renderer (OpenCV-based)Generates high-quality PNG images and MP4 videos locally
Dynamic backgrounds: deep space nebula, cyberpunk skyline, natural ground
500 twinkling stars with variable brightness
Moving asteroid with glowing trail (video only)
Pulsing engine glow effects
Breathing transparent shadow under main object
Camera effects: subtle zoom + shake on complex scenes
Performance capped at 300 frames for smooth generation

Beautiful Tkinter GUIDark cyberpunk theme (#0f0020 background with neon accents)
Specialization selection, video toggle, duration options (3/6/10/15s)
Real-time progress bar and status messages (with cute Arabic touches )
Live image preview and success notifications

Full Logging & SafetyDetailed console logging for debugging
Thread-safe generation (no GUI freezing)
Automatic timestamped file naming
3D interaction visualization graph saved per generation

How to RunMake sure you have Python 3.8+
Install dependencies:bash

pip install opencv-python numpy pillow matplotlib tkinter

Place all files in the same folder:main.py
gui.py
engine.py
draw.py
layers.py
utils.py

Run:bash

python main.py

Example Prompts"Car" → Futuristic flying car with holographic wings
"cyberpunk car in neon city" → Enhanced skyline and neon effects
"geometric sports car blueprint" → Precise mechanical design
"symmetric spaceship with twin engines on rear" → Perfect mirror symmetry

File Structuremain.py – Entry point
gui.py – Beautiful Tkinter interface
engine.py – Core intelligence, pipeline, physics, task management
draw.py – Ultimate Fallback Renderer (OpenCV magic)

Future DreamsIntegration with real AI models (Flux, Grok-2) when available
3D export support
More specializations and effects
Voice input
Export to GIF and WebM

Designed by : Rashed Dadouch 
Email : (rasheddadou@gmail)
Phone : +963 943 307057

Start project : 3 July 2025 , finishing at January 8, 2026 — The day Grok.NG Pro came to life.



