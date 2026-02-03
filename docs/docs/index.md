
## **device**

Variable principale d'allocation des performances.

### **Apple Silicon (macOS)**
- Si le système d'exploitation est macOS (nommé `darwin` dans `platform.system()`), la fonction vérifie si l'accélérateur **Metal Performance Shaders** (MPS) est disponible sur l'appareil.
  - Si MPS est disponible (`torch.backends.mps.is_available()`), l'appareil cible sera défini sur `'mps'` (c'est un équivalent de CUDA pour les appareils Apple Silicon).
  
### **Windows**
- Si le système d'exploitation est Windows, la fonction vérifie d'abord si **CUDA** (NVIDIA) est disponible avec `torch.cuda.is_available()`. Si c'est le cas, le périphérique sera défini sur **CUDA**.
  
### **Linux**
- Si le système d'exploitation est Linux, plusieurs vérifications sont effectuées :
  1. **CUDA** (NVIDIA) : Si `torch.cuda.is_available()` renvoie `True`, le périphérique sera défini sur `'cuda'`.
  2. **ROCm** (AMD) : Si le système supporte **ROCm** via `torch.backends.hip.is_available()`, l'appareil sera défini sur `'cuda'` (ROCm est utilisé pour les cartes AMD dans le cadre de l'API CUDA).
  3. **Intel oneAPI / XPU** : Si le système prend en charge **Intel oneAPI** ou **XPU** via `torch.xpu.is_available()`, le périphérique sera défini sur **XPU**.
  
### **Système non reconnu**
- Si aucune des conditions ci-dessus n'est remplie, la fonction retourne `'cpu'` comme périphérique par défaut.

---