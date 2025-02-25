import math
import random
import numpy as np
import moderngl
import moderngl_window as mglw

# --------------------------------------------------
# Funções utilitárias de matrizes
# --------------------------------------------------
def perspective(fovy, aspect, near, far):
    """Retorna a matriz de projeção perspectiva."""
    f = 1.0 / math.tan(fovy / 2)
    return np.array([
        [f / aspect, 0,   0,                           0],
        [0,          f,   0,                           0],
        [0,          0,   (far + near)/(near - far),   (2 * far * near)/(near - far)],
        [0,          0,   -1,                          0]
    ], dtype=np.float32)

def look_at(eye, target, up):
    """Retorna a matriz view posicionando a câmera."""
    f = target - eye
    f /= np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[:3, 0] = s
    M[:3, 1] = u
    M[:3, 2] = -f
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    return M @ T

# --------------------------------------------------
# Função para criar uma geometria de cubo (exemplo)
# Cada vértice tem: posição (3f), normal (3f) e uv (2f)
# --------------------------------------------------
def create_cube_geometry(size=1.0):
    half = size / 2
    vertices = [
        # Frente (z = half) normal (0,0,1)
        -half, -half,  half,  0, 0, 1,  0, 0,
         half, -half,  half,  0, 0, 1,  1, 0,
         half,  half,  half,  0, 0, 1,  1, 1,
        -half,  half,  half,  0, 0, 1,  0, 1,
        # Trás (z = -half) normal (0,0,-1)
         half, -half, -half,  0, 0, -1, 0, 0,
        -half, -half, -half,  0, 0, -1, 1, 0,
        -half,  half, -half,  0, 0, -1, 1, 1,
         half,  half, -half,  0, 0, -1, 0, 1,
        # Esquerda (x = -half) normal (-1,0,0)
        -half, -half, -half, -1, 0, 0,  0, 0,
        -half, -half,  half, -1, 0, 0,  1, 0,
        -half,  half,  half, -1, 0, 0,  1, 1,
        -half,  half, -half, -1, 0, 0,  0, 1,
        # Direita (x = half) normal (1,0,0)
         half, -half,  half,  1, 0, 0,  0, 0,
         half, -half, -half,  1, 0, 0,  1, 0,
         half,  half, -half,  1, 0, 0,  1, 1,
         half,  half,  half,  1, 0, 0,  0, 1,
        # Topo (y = half) normal (0,1,0)
        -half,  half,  half,  0, 1, 0,  0, 0,
         half,  half,  half,  0, 1, 0,  1, 0,
         half,  half, -half,  0, 1, 0,  1, 1,
        -half,  half, -half,  0, 1, 0,  0, 1,
        # Base (y = -half) normal (0,-1,0)
        -half, -half, -half,  0, -1, 0,  0, 0,
         half, -half, -half,  0, -1, 0,  1, 0,
         half, -half,  half,  0, -1, 0,  1, 1,
        -half, -half,  half,  0, -1, 0,  0, 1,
    ]
    vertices = np.array(vertices, dtype='f4')
    indices = [
         0,  1,  2,  0,  2,  3,        # frente
         4,  5,  6,  4,  6,  7,        # trás
         8,  9, 10,  8, 10, 11,        # esquerda
        12, 13, 14, 12, 14, 15,        # direita
        16, 17, 18, 16, 18, 19,        # topo
        20, 21, 22, 20, 22, 23         # base
    ]
    indices = np.array(indices, dtype='i4')
    return vertices, indices

# --------------------------------------------------
# Shaders GLSL
# --------------------------------------------------
vertex_shader_source = """
#version 330

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

uniform mat4 MVP;
uniform mat4 model;

out vec3 v_normal;

void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    v_normal = mat3(model) * in_normal;
}
"""

fragment_shader_source = """
#version 330

uniform vec3 baseColor;
uniform vec3 lightDir;

in vec3 v_normal;

out vec4 fragColor;

void main() {
    vec3 norm = normalize(v_normal);
    float diff = max(dot(norm, normalize(lightDir)), 0.0);
    vec3 ambient = 0.3 * baseColor;
    vec3 litColor = ambient + diff * baseColor;
    fragColor = vec4(litColor, 1.0);
}
"""

# --------------------------------------------------
# Classe interativa com moderngl-window
# --------------------------------------------------
class InteractiveCamera(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Interactive Camera Demo"
    window_size = (1024, 1024)
    resource_dir = '.'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Setup Camera
        self.fov = 75.0  # em graus
        self.camera_distance = 2.5
        self.camera_angle_x = 0.0  # ângulo de elevação
        self.camera_angle_y = 0.0  # ângulo azimutal

        # Cria a projeção
        self.aspect = self.wnd.size[0] / self.wnd.size[1]
        self.proj = perspective(math.radians(self.fov), self.aspect, 0.1, 100.0)

        # Matriz modelo (para rotação, etc.)
        self.model = np.eye(4, dtype=np.float32)

        # Cria o shader e geometria
        self.program = self.ctx.program(
            vertex_shader=vertex_shader_source,
            fragment_shader=fragment_shader_source
        )

        # Cria a geometria
        vertices, indices = create_cube_geometry(size=1.0)
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        
        # Criação do VAO usando apenas posição e normal
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (
                    self.vbo, 
                    '3f 3f 2x',  # 3 floats para posição, 3 para normal, 2 floats ignorados (x)
                    0, 1  # Apenas posição (0) e normal (1)
                ),
            ],
            self.ibo
        )

    def on_render(self, time, frame_time):
        # Controle de câmera via setas
        keys = self.wnd.keys
        if self.wnd.is_key_pressed(keys.LEFT):
            self.camera_angle_y -= 0.02
        if self.wnd.is_key_pressed(keys.RIGHT):
            self.camera_angle_y += 0.02
        if self.wnd.is_key_pressed(keys.UP):
            self.camera_angle_x = max(self.camera_angle_x - 0.02, -math.pi/2 + 0.1)
        if self.wnd.is_key_pressed(keys.DOWN):
            self.camera_angle_x = min(self.camera_angle_x + 0.02, math.pi/2 - 0.1)

        # Calcula a posição da câmera a partir dos ângulos e distância
        eye_x = self.camera_distance * math.cos(self.camera_angle_x) * math.cos(self.camera_angle_y)
        eye_y = self.camera_distance * math.cos(self.camera_angle_x) * math.sin(self.camera_angle_y)
        eye_z = self.camera_distance * math.sin(self.camera_angle_x)
        eye = np.array([eye_x, eye_y, eye_z], dtype=np.float32)
        view = look_at(eye, np.array([0, 0, 0], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))
        MVP = self.proj @ view @ self.model

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.2, 0.2, 0.2, 1.0)
        self.program['MVP'].write(MVP.tobytes())
        self.program['model'].write(self.model.tobytes())
        self.program['baseColor'].value = (1.0, 1.0, 1.0)
        self.program['lightDir'].value = (0.5, 0.5, 1.0)
        self.vao.render()

if __name__ == '__main__':
    mglw.run_window_config(InteractiveCamera)