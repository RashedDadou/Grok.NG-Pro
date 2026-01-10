# gui.py
import logging
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import os
import threading
from datetime import datetime
import numpy as np  # عشان PlaneLayer لو هستخدمه هنا مستقبلًا (اختياري دلوقتي)

# استيراد المحرك من الملف الجديد
from engine import GrokNGEngine  # هيشتغل لما ننشئ engine.py بعدين

class GrokNGGUI:
    def __init__(self):
        self.engine = GrokNGEngine()
        self.root = tk.Tk()
        self.root.title("🚀 Grok.NG Pro v1.1 - الجيل الجديد الكامل 💜")
        self.root.configure(bg="#0f0020")
        self.root.geometry("1300x1000")

        # العنوان الرئيسي
        tk.Label(self.root, text="Grok.NG Pro v1.1", font=("Arial", 28, "bold"), 
                 fg="#ff99ff", bg="#0f0020").pack(pady=30)

        # الفريم اللي فيه خيارات التصميم والفيديو
        frame = tk.Frame(self.root, bg="#0f0020")
        frame.pack(pady=20)

        tk.Label(frame, text="نوع التصميم:", font=("Arial", 16), fg="#03dac6", bg="#0f0020").pack()
        self.spec_var = tk.StringVar(value="futuristic_design")
        for spec in ["traditional_design", "geometric_design", "futuristic_design"]:
            tk.Radiobutton(frame, text=spec.replace("_", " ").title(), variable=self.spec_var,
                           value=spec, fg="#ffffff", bg="#0f0020", font=("Arial", 14), 
                           selectcolor="#330066").pack()

        self.video_var = tk.BooleanVar()
        tk.Checkbutton(frame, text="🎬 Video generation", variable=self.video_var,
                       fg="#ffaa00", bg="#0f0020", font=("Arial", 14)).pack(pady=20)

        # مدة الفيديو
        tk.Label(frame, text="مدة الفيديو (ثواني):", font=("Arial", 14), fg="#03dac6", bg="#0f0020").pack()
        self.duration_var = tk.IntVar(value=6)
        for dur in [3, 6, 10, 15]:
            tk.Radiobutton(frame, text=str(dur), variable=self.duration_var, value=dur,
                           fg="#ffffff", bg="#0f0020", font=("Arial", 12)).pack(side="left", padx=10)

        # وصف المستخدم
        tk.Label(self.root, text="Add description:", font=("Arial", 16), fg="#03dac6", bg="#0f0020").pack()
        self.entry = scrolledtext.ScrolledText(self.root, height=6, font=("Arial", 14), 
                                               bg="#200040", fg="#ffffff")
        self.entry.pack(fill="x", padx=80, pady=10)

        # زر التوليد
        self.gen_btn = tk.Button(self.root, text="Generating with the new generation", 
                                 font=("Arial", 20, "bold"), bg="#00c853", fg="white", 
                                 command=self.start_generation)
        self.gen_btn.pack(pady=30)

        # Progress Bar دلع
        self.progress = ttk.Progressbar(self.root, length=800, mode='determinate', style="TProgressbar")
        self.progress.pack(pady=20)
        style = ttk.Style()
        style.configure("TProgressbar", thickness=30, background="#ff99ff", troughcolor="#330066")

        # الحالة والعرض
        self.status = tk.Label(self.root, text="Ready for takeoff", fg="#00ffaa", bg="#0f0020", 
                               font=("Arial", 16))
        self.status.pack(pady=10)

        self.display_label = tk.Label(self.root, text="The result will be displayed here, my dear...", 
                                     fg="#8888ff", bg="#0f0020", font=("Arial", 20))
        self.display_label.pack(expand=True, fill="both", padx=80, pady=20)

        self.current_photo = None

    def run(self):
        self.root.mainloop()
        
    def update_progress(self, value: int, text: str):
        self.progress['value'] = value
        self.status.config(text=text)

    def start_generation(self):
        prompt = self.entry.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("تحذير", "اكتب وصف أولاً يا كتكوتي! ⭐🥺")
            return

        spec = self.spec_var.get()
        is_video = self.video_var.get()
        selected_duration = self.duration_var.get()  # نقرأها مرة واحدة في البداية

        self.gen_btn.config(state="disabled")
        self.progress['value'] = 0
        self.status.config(text="جاري التوليد... ⏳")
        self.root.update_idletasks()  # تحديث الواجهة فورًا عشان يشوف التغيير

        def thread_func():
            try:
                logging.info("بدء التوليد في الـ thread")
                logging.info(f"الوصف: {prompt}")
                logging.info(f"التخصص: {spec} | فيديو: {is_video} | مدة: {selected_duration}s")

                result = self.engine.run_unified_pipeline(
                    specialization=spec,
                    user_prompt=prompt,
                    is_video=is_video,
                    duration=selected_duration,
                    progress_callback=self.update_progress
                )

                # عرض النتيجة في الـ main thread
                self.root.after(0, self.display_result, 
                                result["image"], 
                                result["video"], 
                                result["interaction_vis"])
                
                self.root.after(0, self.update_progress, 100, "تم يا قمري! 💜✨")
                
            except Exception as e:
                logging.error(f"خطأ في التوليد: {e}")
                import traceback
                traceback.print_exc()  # طباعة التفاصيل في console
                self.root.after(0, messagebox.showerror, "خطأ", f"حصل خطأ: {e}")
            finally:
                self.root.after(0, self.gen_btn.config, {"state": "normal"})
                self.root.after(0, self.status.config, {"text": "Ready for takeoff 🚀"})

        threading.Thread(target=thread_func, daemon=True).start()
                        
    def display_result(self, img_path, video_path=None, vis_path=None):
        if not os.path.exists(img_path):
            self.display_label.config(text="مشكلة في التوليد 😢")
            return

        try:
            img = Image.open(img_path)
            img = img.resize((1100, int(1100 * img.height / img.width)), Image.Resampling.LANCZOS)
            self.current_photo = ImageTk.PhotoImage(img)
            self.display_label.config(image=self.current_photo, text="")
        except Exception as e:
            self.display_label.config(text=f"خطأ في عرض الصورة: {e}")

        msg = f"تم يا قمري! 💜 الصورة: {os.path.basename(img_path)}"
        if video_path and os.path.exists(video_path):
            msg += f"\n🎬 والفيديو: {os.path.basename(video_path)}"
        messagebox.showinfo("نجاح دلع!", msg)

        if vis_path and os.path.exists(vis_path):
            messagebox.showinfo("رسم بياني جاهز!", f"مسار التفاعلات:\n{vis_path}\nافتحه واستمتع بالدلع الفيزيائي ✨")
            
if __name__ == "__main__":
    app = GrokNGGUI()
    app.run()
