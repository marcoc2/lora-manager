import OpenGL.GL as gl
import OpenGL.GLUT as glut

try:
    glut.glutInit()
    print("✅ OpenGL está funcional no Python.")
except Exception as e:
    print(f"❌ Erro ao inicializar OpenGL: {e}")
