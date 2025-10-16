import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# ===========================
#  CARGAR MODELO DE IA
# ===========================
try:
    model = load_model("modelo_yoga.h5")
    etiquetas = ["perro", "arbol"]  # cambia según tus clases entrenadas
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    model = None
    etiquetas = []
    print("⚠️ No se pudo cargar el modelo:", e)

# ===========================
#  CONFIGURAR MEDIAPIPE
# ===========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ===========================
#  RECOMENDACIONES POR POSE
# ===========================
recomendaciones = {
    "perro": [
        "Asegúrate de mantener la espalda recta.",
        "Empuja el suelo con las manos.",
        "Relaja el cuello y deja que la cabeza caiga naturalmente."
    ],
    "arbol": [
        "Mantén el equilibrio con el pie firmemente apoyado.",
        "Lleva las manos al pecho o por encima de la cabeza.",
        "Evita inclinar el tronco hacia un lado."
    ]
}


# ===========================
#  CLASE PRINCIPAL DE LA APP
# ===========================
class YogaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Asistente de Yoga con IA 🧘‍♀️")
        self.root.geometry("950x700")
        self.root.resizable(False, False)

        self.pose_type = tk.StringVar(value="Reconocimiento libre")
        self.video_source = None
        self.cap = None
        self.running = False
        self.use_camera = False
        self.pose = None

        # ===================== CONTROLES =====================
        frame_controls = tk.Frame(root, bg="#1f1f1f")
        frame_controls.pack(fill=tk.X, pady=5)

        ttk.Label(frame_controls, text="Modo:").pack(side=tk.LEFT, padx=10)
        opciones = ["Reconocimiento libre"] + etiquetas
        ttk.OptionMenu(frame_controls, self.pose_type, *opciones).pack(side=tk.LEFT, padx=10)

        ttk.Button(frame_controls, text="Cargar video", command=self.load_video).pack(side=tk.LEFT, padx=10)
        ttk.Button(frame_controls, text="Usar cámara en vivo", command=self.use_webcam).pack(side=tk.LEFT, padx=10)
        ttk.Button(frame_controls, text="Iniciar análisis", command=self.start_video).pack(side=tk.LEFT, padx=10)
        ttk.Button(frame_controls, text="Detener", command=self.stop_video).pack(side=tk.LEFT, padx=10)

        # ===================== ETIQUETAS =====================
        self.lbl_feedback = tk.Label(root, text="Esperando inicio...", fg="white", bg="#333", font=("Arial", 12))
        self.lbl_feedback.pack(fill=tk.X, pady=5)

        self.lbl_video = tk.Label(root, bg="black")
        self.lbl_video.pack(padx=10, pady=10)

    # ===================== FUNCIONES =====================
    def load_video(self):
        filename = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi")])
        if filename:
            self.video_source = filename
            self.use_camera = False
            self.lbl_feedback.config(text=f"Video cargado: {filename}")

    def use_webcam(self):
        self.video_source = 2  # cambia a 1 o 2 si tu cámara no es la principal
        self.use_camera = True
        self.lbl_feedback.config(text="Cámara lista para usar.")

    def start_video(self):
        if self.video_source is None:
            messagebox.showinfo("Fuente no seleccionada", "Selecciona un video o activa la cámara en vivo.")
            return

        self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara o el video.")
            return

        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.running = True
        self.lbl_feedback.config(text="Analizando postura...")
        self.run_frame()

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()
        self.lbl_feedback.config(text="Análisis detenido.")
        self.lbl_video.config(image="")

    # ===================== PROCESAR FRAMES =====================
    def run_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            if not self.use_camera:
                self.lbl_feedback.config(text="Fin del video.")
            self.stop_video()
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        texto = ""
        color = (255, 255, 255)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if model is not None and len(landmarks) == 99:
                X_input = np.array(landmarks).reshape(1, -1)
                pred = model.predict(X_input)
                clase_predicha = etiquetas[np.argmax(pred)]
                confianza = np.max(pred)
                seleccion = self.pose_type.get()

                if seleccion == "Reconocimiento libre":
                    texto = f"Pose detectada: {clase_predicha} ({confianza*100:.1f}%)"
                    color = (0, 255, 0) if confianza > 0.7 else (0, 0, 255)
                else:
                    if clase_predicha == seleccion:
                        texto = f"✅ Excelente, estas haciendo la pose {seleccion} ({confianza*100:.1f}%)"
                        color = (0, 255, 0)
                    else:
                        texto = f"⚠️ Parece que no estas en la pose {seleccion}. El modelo detecta: {clase_predicha}."
                        color = (0, 0, 255)
                        if seleccion in recomendaciones:
                            texto += "\n" + "\n".join(recomendaciones[seleccion])
            else:
                texto = "Modelo no cargado o puntos insuficientes."
        else:
            texto = "No se detecta cuerpo."

        # Mostrar texto en cámara
        y_offset = 40
        for i, line in enumerate(texto.split("\n")):
            cv2.putText(frame, line, (20, y_offset + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mostrar en ventana de Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        self.lbl_video.imgtk = img
        self.lbl_video.config(image=img)
        self.lbl_feedback.config(text=texto)

        self.root.after(20, self.run_frame)


# ===========================
#  EJECUCIÓN PRINCIPAL
# ===========================
if __name__ == "__main__":
    root = tk.Tk()
    app = YogaApp(root)
    root.mainloop()
