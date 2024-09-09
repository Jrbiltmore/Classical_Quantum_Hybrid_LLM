
# rendering_extension.py
# Plugin for adding additional rendering features, such as advanced shaders or custom visual effects.

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

class RenderingExtension:
    """Class responsible for adding advanced rendering techniques such as shaders or custom effects."""

    def __init__(self):
        self.custom_shaders = {}  # Store custom shaders for rendering

    def load_shader(self, shader_name: str, vertex_code: str, fragment_code: str):
        """Loads and compiles a custom shader for rendering."""
        shader_program = glCreateProgram()
        vertex_shader = self._compile_shader(vertex_code, GL_VERTEX_SHADER)
        fragment_shader = self._compile_shader(fragment_code, GL_FRAGMENT_SHADER)

        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, fragment_shader)
        glLinkProgram(shader_program)

        if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError("Shader linking failed.")

        self.custom_shaders[shader_name] = shader_program
        print(f"Shader '{shader_name}' loaded successfully.")

    def use_shader(self, shader_name: str):
        """Activates a custom shader for rendering."""
        if shader_name in self.custom_shaders:
            glUseProgram(self.custom_shaders[shader_name])
        else:
            raise ValueError(f"Shader '{shader_name}' not found.")

    def _compile_shader(self, source_code: str, shader_type: int) -> int:
        """Compiles shader code and returns the shader ID."""
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source_code)
        glCompileShader(shader)

        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(shader))

        return shader
