@echo off
echo ============================================
echo   3D Generator - Instalacion
echo ============================================
echo.

echo [1/4] Instalando dependencias Python base...
cd backend
pip install fastapi uvicorn[standard] python-multipart Pillow trimesh numpy
echo.

echo [2/4] Instalando PyTorch con soporte CUDA (RTX 3070 Ti)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.

echo [3/4] Instalando modelos de IA...
if not exist TripoSR (
    git clone https://github.com/VAST-AI-Research/TripoSR.git TripoSR
)
pip install -r TripoSR/requirements.txt
pip install git+https://github.com/openai/shap-e.git
cd ..
echo.

echo [4/4] Instalando dependencias Node.js...
cd frontend
npm install
cd ..
echo.

echo ============================================
echo   Instalacion completada!
echo   Ejecuta start.bat para iniciar la app.
echo ============================================
pause
