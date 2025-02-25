import os
import sys
import random
import math
import numpy as np
from PIL import Image
import moderngl

from moderngl import DEPTH_TEST

# ===============================
# Funções utilitárias de matrizes
# ===============================
def perspective(fovy, aspect, near, far):
    """Matriz de projeção perspectiva corrigida."""
    f = 1.0 / math.tan(fovy / 2)
    return np.array([
        [f / aspect, 0,  0,                           0],
        [0,          f,  0,                           0],
        [0,          0,  (far + near)/(near - far),  (2 * far * near)/(near - far)],
        [0,          0,  -1,                          0]
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

def random_camera_pose(distance=2.5):
    """Gera poses mais variadas incluindo ângulos abaixo do horizonte."""
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.pi/4, 3*math.pi/4)  # entre 45° e 135°
    x = distance * math.sin(phi) * math.cos(theta)
    y = distance * math.sin(phi) * math.sin(theta)
    z = distance * math.cos(phi)
    eye = np.array([x, y, z], dtype=np.float32)
    target = np.array([0, 0, 0], dtype=np.float32)
    # Calcula o vetor up de forma dinâmica
    forward = target - eye
    right = np.cross(forward, np.array([0, 0, 1], dtype=np.float32))
    if np.linalg.norm(right) < 1e-6:
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
# Geometrias com normais e UV
# Cada vértice: [x, y, z, nx, ny, nz, u, v]
# ===============================

def create_plane_geometry(scale=1.0, tex_aspect=1.0):
    """Plano com razão de aspecto ajustada ao logo."""
    width = scale
    height = scale / tex_aspect
    # Posições
    positions = [
        [-width/2, -height/2, 0],
        [ width/2, -height/2, 0],
        [ width/2,  height/2, 0],
        [-width/2,  height/2, 0]
    ]
    # Normais: para um plano no XY, normal = (0, 0, 1)
    normals = [[0, 0, 1]] * 4
    uvs = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ]
    vertex_data = np.hstack([np.array(positions), np.array(normals), np.array(uvs)]).astype(np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return vertex_data, indices

def create_cylinder_geometry(height=1.0, radius=0.5, sections=32):
    """Superfície lateral de um cilindro sem tampas."""
    vertices = []
    indices = []
    for i in range(sections + 1):
        angle = 2 * math.pi * i / sections
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        # Normal para a lateral: (cos(angle), sin(angle), 0)
        normal = [math.cos(angle), math.sin(angle), 0]
        u = i / sections
        vertices.append([x, y, -height/2] + normal + [u, 0.0])
        vertices.append([x, y,  height/2] + normal + [u, 1.0])
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
    """Esfera com mapeamento UV e normais iguais à posição normalizada."""
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
            normal = [x/radius, y/radius, z/radius]
            vertices.append([x, y, z] + normal + [u, v])
    vertices = np.array(vertices, dtype=np.float32)
    for i in range(stacks):
        for j in range(sectors):
            first = i * (sectors + 1) + j
            second = first + sectors + 1
            indices.extend([first, second, first + 1, second, second + 1, first + 1])
    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices

def create_cube_geometry(size=1.0):
    """Cubo com 6 faces e normais por face."""
    half = size / 2
    # Cada face: 4 vértices com a mesma normal.
    faces = [
        # Frente (z = half), normal (0,0,1)
        ( [[-half, -half, half], [0,0,1], [0,0]],
          [[ half, -half, half], [0,0,1], [1,0]],
          [[ half,  half, half], [0,0,1], [1,1]],
          [[-half,  half, half], [0,0,1], [0,1]] ),
        # Trás (z = -half), normal (0,0,-1)
        ( [[ half, -half, -half], [0,0,-1], [0,0]],
          [[-half, -half, -half], [0,0,-1], [1,0]],
          [[-half,  half, -half], [0,0,-1], [1,1]],
          [[ half,  half, -half], [0,0,-1], [0,1]] ),
        # Esquerda (x = -half), normal (-1,0,0)
        ( [[-half, -half, -half], [-1,0,0], [0,0]],
          [[-half, -half,  half], [-1,0,0], [1,0]],
          [[-half,  half,  half], [-1,0,0], [1,1]],
          [[-half,  half, -half], [-1,0,0], [0,1]] ),
        # Direita (x = half), normal (1,0,0)
        ( [[ half, -half,  half], [1,0,0], [0,0]],
          [[ half, -half, -half], [1,0,0], [1,0]],
          [[ half,  half, -half], [1,0,0], [1,1]],
          [[ half,  half,  half], [1,0,0], [0,1]] ),
        # Topo (y = half), normal (0,1,0)
        ( [[-half, half,  half], [0,1,0], [0,0]],
          [[ half, half,  half], [0,1,0], [1,0]],
          [[ half, half, -half], [0,1,0], [1,1]],
          [[-half, half, -half], [0,1,0], [0,1]] ),
        # Base (y = -half), normal (0,-1,0)
        ( [[-half, -half, -half], [0,-1,0], [0,0]],
          [[ half, -half, -half], [0,-1,0], [1,0]],
          [[ half, -half,  half], [0,-1,0], [1,1]],
          [[-half, -half,  half], [0,-1,0], [0,1]] )
    ]
    vertices = []
    indices = []
    vertex_count = 0
    for face in faces:
        for vert in face:
            pos, norm, uv = vert
            vertices.append(pos + norm + uv)
        indices.extend([vertex_count, vertex_count+1, vertex_count+2,
                        vertex_count, vertex_count+2, vertex_count+3])
        vertex_count += 4
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices

def create_pyramid_geometry(size=1.0, height=1.0):
    """Pirâmide com base quadrada. Na lateral, o ápice tem V = 0.3 para aproximar a textura da base."""
    half = size / 2
    vertices = []
    indices = []
    # Base: normal para baixo (0,0,-1)
    base_norm = [0, 0, -1]
    base = [
        [-half, -half, 0,] + base_norm + [0, 0],
        [ half, -half, 0,] + base_norm + [1, 0],
        [ half,  half, 0,] + base_norm + [1, 1],
        [-half,  half, 0,] + base_norm + [0, 1]
    ]
    base_indices = [0, 1, 2, 0, 2, 3]
    vertices.extend(base)
    indices.extend(base_indices)
    current_offset = len(base)
    # Lateral faces: cada face (3 vértices)
    # O ápice recebe UV com V = 0.3
    lateral_faces = []
    apex = [0, 0, height]
    for i in range(4):
        # Índices da base (quadrado)
        i0 = i
        i1 = (i+1) % 4
        # Para cada face, calcule a normal (usando cross product)
        p0 = np.array(base[i0][:3])
        p1 = np.array(base[i1][:3])
        p2 = np.array(apex)
        edge1 = p1 - p0
        edge2 = p2 - p0
        norm = np.cross(edge1, edge2)
        norm = (norm / np.linalg.norm(norm)).tolist()
        # Face com vértices: base[i0], base[i1] e o ápice
        lateral_faces.append([
            base[i0][:3] + norm + [0, 0],
            base[i1][:3] + norm + [1, 0],
            apex + norm + [0.5, 0.3]
        ])
    for face in lateral_faces:
        vertices.extend(face)
        indices.extend([current_offset, current_offset+1, current_offset+2])
        current_offset += 3
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices

# ===============================
# Shaders GLSL com iluminação simples
# ===============================
vertex_shader_source = """
#version 330
uniform mat4 MVP;
uniform mat4 model;
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
out vec2 v_uv;
out vec3 v_normal;
void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    v_uv = in_uv;
    // Transforma a normal usando a parte 3x3 da matriz model
    v_normal = mat3(model) * in_normal;
}
"""

fragment_shader_source = """
#version 330
uniform sampler2D Texture;
uniform vec3 baseColor;
uniform vec3 lightDir;  // direção da luz (em espaço de visão)
in vec2 v_uv;
in vec3 v_normal;
out vec4 fragColor;
void main() {
    // Cálculo difuso simples
    vec3 norm = normalize(v_normal);
    float diff = max(dot(norm, normalize(lightDir)), 0.0);
    vec3 ambient = 0.3 * baseColor;
    vec3 litColor = ambient + diff * baseColor;
    vec4 texColor = texture(Texture, v_uv);
    // Mistura a cor base com a textura e aplica iluminação
    fragColor = vec4(litColor * mix(baseColor, texColor.rgb, texColor.a), 1.0);
}
"""

# ===============================
# Renderização via moderngl com depth buffer
# ===============================
def render_geometry(ctx, program, vertex_data, indices, MVP, model, texture, base_color, image_size=(1024, 1024)):
    """Renderiza a geometria com depth buffer."""
    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture(image_size, 4)],
        depth_attachment=ctx.depth_texture(image_size)
    )
    fbo.use()
    ctx.enable(DEPTH_TEST)
    ctx.clear(1.0, 1.0, 1.0, 1.0, depth=1.0)

    vbo = ctx.buffer(vertex_data.astype('f4').tobytes())
    ibo = ctx.buffer(indices.astype('i4').tobytes())

    vao = ctx.vertex_array(program, [
        (vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')
    ], index_buffer=ibo)
    
    program['MVP'].write(MVP.tobytes())
    program['model'].write(model.tobytes())
    program['baseColor'].value = tuple(base_color)
    # Define uma direção de luz (por exemplo, vindo de cima e à direita)
    program['lightDir'].value = (0.5, 0.5, 1.0)
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
# Função principal
# ===============================
def main():
    if len(sys.argv) < 2:
        print("Uso: python texture_augmentation_mgl.py <caminho_para_logo.png> [quantidade_imagens]")
        sys.exit(1)

    logo_path = sys.argv[1]
    num_images = 10
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
    proj = perspective(math.radians(45), aspect_proj, 0.1, 100.0)

    # Lista de funções para gerar diferentes geometrias
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

        rendered_img = render_geometry(ctx, program, vertex_data, indices, MVP, model, texture, base_color, image_size=image_size)
        output_path = os.path.join(output_dir, f"aug_{i:03d}.png")
        rendered_img.save(output_path)
        print(f"Imagem salva em: {output_path}")

    texture.release()
    program.release()
    ctx.release()

if __name__ == "__main__":
    main()
