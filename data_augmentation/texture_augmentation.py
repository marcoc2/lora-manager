import os
import sys
import random
import math
import numpy as np
from PIL import Image
import moderngl

# ===============================
# Funções utilitárias de matrizes
# ===============================
def perspective(fovy, aspect, near, far):
    """Retorna a matriz de projeção perspectiva corrigida."""
    f = 1.0 / math.tan(fovy / 2)
    return np.array([
        [f / aspect, 0,           0,                           0],
        [0,          f,           0,                           0],
        [0,          0,  (far + near) / (near - far),             -1],
        [0,          0,  (2 * far * near) / (near - far),           0]
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
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    return M @ T

def random_camera_pose(distance=2.5):
    """
    Gera uma matriz view com a câmera posicionada aleatoriamente.
    --> Para aproximar ou afastar a câmera, altere o parâmetro 'distance'.
    """
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, math.pi / 2)  # câmera um pouco acima do horizonte
    x = distance * math.sin(phi) * math.cos(theta)
    y = distance * math.sin(phi) * math.sin(theta)
    z = distance * math.cos(phi)
    eye = np.array([x, y, z], dtype=np.float32)
    target = np.array([0, 0, 0], dtype=np.float32)
    # Calcula o vetor up de forma dinâmica, para evitar achatamento
    forward = target - eye
    right = np.cross(forward, np.array([0, 0, 1], dtype=np.float32))
    if np.linalg.norm(right) < 1e-6:  # em caso de alinhamento, usa outro vetor
        right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return look_at(eye, target, up)

def random_model_matrix():
    """Gera uma matriz de rotação aleatória para o modelo."""
    angle = random.uniform(0, 2 * math.pi)
    axis = np.random.rand(3) - 0.5
    axis /= np.linalg.norm(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c
    x, y, z = axis
    rot = np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y, 0],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x, 0],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c,   0],
        [0,            0,            0,           1]
    ], dtype=np.float32)
    return rot

# ===============================
# Geometrias com mapeamento UV
# ===============================

def create_plane_geometry(scale=1.0, tex_aspect=1.0):
    """
    Cria um plano com razão de aspecto ajustada ao logo.
    tex_aspect = largura/altura do logo.
    """
    width = scale
    height = scale / tex_aspect
    positions = np.array([
        [-width/2, -height/2, 0],
        [ width/2, -height/2, 0],
        [ width/2,  height/2, 0],
        [-width/2,  height/2, 0]
    ], dtype=np.float32)
    uvs = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ], dtype=np.float32)
    vertex_data = np.hstack([positions, uvs])
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return vertex_data, indices

def create_cylinder_geometry(height=1.0, radius=0.5, sections=32):
    """
    Gera a superfície lateral de um cilindro (sem tampas) com mapeamento UV.
    U varia conforme o ângulo; V de 0 (base inferior) a 1 (base superior).
    """
    vertices = []
    indices = []
    for i in range(sections + 1):
        angle = 2 * math.pi * i / sections
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        u = i / sections
        vertices.append([x, y, -height/2, u, 0.0])
        vertices.append([x, y,  height/2, u, 1.0])
    vertices = np.array(vertices, dtype=np.float32)
    for i in range(sections):
        i0 = 2 * i
        i1 = i0 + 1
        i2 = i0 + 2
        i3 = i0 + 3
        indices.extend([i0, i1, i3, i0, i3, i2])
    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices

def create_sphere_geometry(radius=0.5, sectors=32, stacks=16):
    """
    Gera uma esfera com mapeamento UV usando coordenadas esféricas.
    U varia de 0 a 1 (azimutal) e V de 0 a 1 (polar).
    """
    vertices = []
    indices = []
    for i in range(stacks + 1):
        v = i / stacks
        theta = v * math.pi
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for j in range(sectors + 1):
            u = j / sectors
            phi = u * 2 * math.pi
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            x = radius * sin_theta * cos_phi
            y = radius * sin_theta * sin_phi
            z = radius * cos_theta
            vertices.append([x, y, z, u, v])
    vertices = np.array(vertices, dtype=np.float32)
    for i in range(stacks):
        for j in range(sectors):
            first = i * (sectors + 1) + j
            second = first + sectors + 1
            indices.extend([first, second, first + 1, second, second + 1, first + 1])
    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices

def create_cube_geometry(size=1.0):
    """
    Gera um cubo com 6 faces. Cada face tem 4 vértices exclusivos para mapeamento UV.
    """
    half = size / 2
    faces = [
        # Frente (z = half)
        ( [ [-half, -half, half], [0, 0] ],
          [ [ half, -half, half], [1, 0] ],
          [ [ half,  half, half], [1, 1] ],
          [ [-half,  half, half], [0, 1] ] ),
        # Trás (z = -half)
        ( [ [ half, -half, -half], [0, 0] ],
          [ [-half, -half, -half], [1, 0] ],
          [ [-half,  half, -half], [1, 1] ],
          [ [ half,  half, -half], [0, 1] ] ),
        # Esquerda (x = -half)
        ( [ [-half, -half, -half], [0, 0] ],
          [ [-half, -half,  half], [1, 0] ],
          [ [-half,  half,  half], [1, 1] ],
          [ [-half,  half, -half], [0, 1] ] ),
        # Direita (x = half)
        ( [ [ half, -half,  half], [0, 0] ],
          [ [ half, -half, -half], [1, 0] ],
          [ [ half,  half, -half], [1, 1] ],
          [ [ half,  half,  half], [0, 1] ] ),
        # Topo (y = half)
        ( [ [-half, half,  half], [0, 0] ],
          [ [ half, half,  half], [1, 0] ],
          [ [ half, half, -half], [1, 1] ],
          [ [-half, half, -half], [0, 1] ] ),
        # Base (y = -half)
        ( [ [-half, -half, -half], [0, 0] ],
          [ [ half, -half, -half], [1, 0] ],
          [ [ half, -half,  half], [1, 1] ],
          [ [-half, -half,  half], [0, 1] ] ),
    ]
    vertices = []
    indices = []
    vertex_count = 0
    for face in faces:
        for vert in face:
            pos, uv = vert
            vertices.append(pos + uv)
        indices.extend([vertex_count, vertex_count+1, vertex_count+2,
                        vertex_count, vertex_count+2, vertex_count+3])
        vertex_count += 4
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices

def create_pyramid_geometry(size=1.0, height=1.0):
    """
    Gera uma pirâmide com base quadrada.
    A base fica no plano z = 0 e o ápice em (0, 0, height).
    Para aplicar a textura mais próxima da base, os vértices do ápice nas faces laterais
    têm coordenadas UV com valor V reduzido (0.3 em vez de 1.0).
    """
    half = size / 2
    vertices = []
    indices = []
    # Base (dois triângulos)
    base_vertices = [
        [-half, -half, 0, 0, 0],
        [ half, -half, 0, 1, 0],
        [ half,  half, 0, 1, 1],
        [-half,  half, 0, 0, 1]
    ]
    base_indices = [0, 1, 2, 0, 2, 3]
    vertices.extend(base_vertices)
    indices.extend(base_indices)
    current_offset = len(base_vertices)
    # Faces laterais: cada face recebe o ápice com V = 0.3 para aproximar a textura da base
    lateral_faces = []
    # Face 1: entre vértices 0 e 1
    lateral_faces.append([
        [-half, -half, 0, 0, 0],
        [ half, -half, 0, 1, 0],
        [0, 0, height, 0.5, 0.3]
    ])
    # Face 2: entre vértices 1 e 2
    lateral_faces.append([
        [ half, -half, 0, 0, 0],
        [ half,  half, 0, 1, 0],
        [0, 0, height, 0.5, 0.3]
    ])
    # Face 3: entre vértices 2 e 3
    lateral_faces.append([
        [ half,  half, 0, 0, 0],
        [-half,  half, 0, 1, 0],
        [0, 0, height, 0.5, 0.3]
    ])
    # Face 4: entre vértices 3 e 0
    lateral_faces.append([
        [-half,  half, 0, 0, 0],
        [-half, -half, 0, 1, 0],
        [0, 0, height, 0.5, 0.3]
    ])
    for face in lateral_faces:
        vertices.extend(face)
        indices.extend([current_offset, current_offset+1, current_offset+2])
        current_offset += 3
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices

# ===============================
# Renderização via moderngl
# ===============================
def render_geometry(ctx, program, vertex_data, indices, MVP, texture, base_color, image_size=(1024, 1024)):
    """
    Renderiza a geometria no framebuffer offscreen e retorna a imagem renderizada.
    """
    fbo = ctx.simple_framebuffer(image_size)
    fbo.use()
    ctx.clear(1.0, 1.0, 1.0, 1.0)  # Fundo branco opaco

    vbo = ctx.buffer(vertex_data.astype('f4').tobytes())
    ibo = ctx.buffer(indices.astype('i4').tobytes())

    vao = ctx.simple_vertex_array(
        program, vbo,
        'in_position', 'in_uv',
        index_buffer=ibo
    )
    program['MVP'].write(MVP.tobytes())
    program['baseColor'].value = tuple(base_color)
    texture.use(location=0)
    program['Texture'].value = 0

    vao.render()
    data = fbo.read(components=4, alignment=1)
    img = Image.frombytes('RGBA', image_size, data)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    vbo.release()
    ibo.release()
    vao.release()
    fbo.release()
    return img

# ===============================
# Shaders GLSL
# ===============================
vertex_shader_source = """
#version 330
uniform mat4 MVP;
in vec3 in_position;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    v_uv = in_uv;
}
"""

fragment_shader_source = """
#version 330
uniform sampler2D Texture;
uniform vec3 baseColor;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec4 texColor = texture(Texture, v_uv);
    fragColor = vec4(mix(baseColor, texColor.rgb, texColor.a), 1.0);
}
"""

# ===============================
# Função principal
# ===============================
def main():
    if len(sys.argv) < 2:
        print("Uso: python texture_augmentation_mgl.py <caminho_para_logo.png> [quantidade_imagens]")
        sys.exit(1)

    logo_path = sys.argv[1]

    # Obtém a quantidade de imagens a gerar (se especificada)
    num_images = 10  # Valor padrão
    if len(sys.argv) > 2:
        try:
            num_images = int(sys.argv[2])
            if num_images <= 0:
                raise ValueError
        except ValueError:
            print("Erro: O número de imagens deve ser um inteiro positivo.")
            sys.exit(1)

    try:
        logo_img = Image.open(logo_path).convert("RGBA")
    except Exception as e:
        print("Erro ao carregar a imagem:", e)
        sys.exit(1)

    logo_data = logo_img.tobytes()
    tex_width, tex_height = logo_img.size
    logo_aspect = tex_width / tex_height

    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    ctx = moderngl.create_standalone_context()
    program = ctx.program(
        vertex_shader=vertex_shader_source,
        fragment_shader=fragment_shader_source,
    )
    texture = ctx.texture((tex_width, tex_height), 4, logo_data)
    texture.build_mipmaps()
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

    image_size = (1024, 1024)
    aspect_proj = image_size[0] / image_size[1]
    proj = perspective(math.radians(75), aspect_proj, 0.1, 100.0)

    geometries = [
        lambda: create_plane_geometry(scale=1.0, tex_aspect=logo_aspect),
        lambda: create_cylinder_geometry(height=1.0, radius=0.5, sections=32),
        lambda: create_sphere_geometry(radius=0.5, sectors=32, stacks=16),
        lambda: create_cube_geometry(size=1.0),
        lambda: create_pyramid_geometry(size=1.0, height=1.0)
    ]

    for i in range(num_images):
        geom_func = random.choice(geometries)
        vertex_data, indices = geom_func()

        model = random_model_matrix()
        view = random_camera_pose(distance=2.5)
        MVP = proj @ view @ model

        base_color = (random.random(), random.random(), random.random())

        rendered_img = render_geometry(ctx, program, vertex_data, indices, MVP, texture, base_color, image_size=image_size)
        output_path = os.path.join(output_dir, f"aug_{i:03d}.png")
        rendered_img.save(output_path)
        print(f"Imagem salva em: {output_path}")

    texture.release()
    program.release()
    ctx.release()

if __name__ == "__main__":
    main()
