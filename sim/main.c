#include "../network/network.h"
#include "world.h"
#include "agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define GLFW_INCLUDE_GLCOREARB 1
#include <GLFW/glfw3.h>

static int WIN_W = 900, WIN_H = 540;
static int CELL_PX = 30;

static GLuint vao, vbo;
static GLuint prog;

static const char *VS = "#version 330\n"
    "const vec2 verts[4] = vec2[4](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,1));"
    "out vec2 uv;"
    "uniform vec2 cell_pos;"
    "uniform float cell_size;"
    "uniform vec2 win_size;"
    "void main(){"
    "  vec2 p = cell_pos + verts[gl_VertexID]*cell_size;"
    "  vec2 ndc = (p/win_size)*2.0 - 1.0;"
    "  gl_Position = vec4(ndc.x, -ndc.y, 0.0, 1.0);"
    "  uv = verts[gl_VertexID];"
    "}";

static const char *FS = "#version 330\n"
    "in vec2 uv;"
    "out vec4 frag;"
    "uniform vec3 color;"
    "void main(){"
    "  frag = vec4(color, 1.0);"
    "}";

static GLuint compile_shader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), NULL, log);
        fprintf(stderr, "shader error: %s\n", log);
        exit(1);
    }
    return s;
}

static void draw_cell(int x, int y, float r, float g, float b)
{
    glUseProgram(prog);
    glUniform2f(glGetUniformLocation(prog, "cell_pos"), (float)x*CELL_PX, (float)y*CELL_PX);
    glUniform1f(glGetUniformLocation(prog, "cell_size"), (float)CELL_PX);
    glUniform2f(glGetUniformLocation(prog, "win_size"), (float)WIN_W, (float)WIN_H);
    glUniform3f(glGetUniformLocation(prog, "color"), r, g, b);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}

static void key_callback(GLFWwindow *win, int key, int scancode, int action, int mods)
{
    (void)scancode; (void)mods;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(win, GLFW_TRUE);
}

int main(void)
{
    srand(time(NULL));

    int width = 30, height = 18;
    WIN_W = width * CELL_PX;
    WIN_H = height * CELL_PX;

    World world;
    world_init(&world, width, height);
    world_populate(&world, 40, 20);

    char *attr_names[] = { "Vita", "Fame" };
    int neurons[] = { 16 };
    Agent agent;
    agent_init(&agent, width / 2, height / 2, 2, 1, neurons, 2, attr_names);

    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

    GLFWwindow *win = glfwCreateWindow(WIN_W, WIN_H, "Simple Creatures", NULL, NULL);
    if (!win) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSetKeyCallback(win, key_callback);
    glfwSwapInterval(1);

    GLuint vs = compile_shader(GL_VERTEX_SHADER, VS);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FS);
    prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    (void)vbo;

    int steps = 0;
    while (!glfwWindowShouldClose(win) && agent.alive && steps < 2000) {
        glViewport(0, 0, WIN_W, WIN_H);
        glClear(GL_COLOR_BUFFER_BIT);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (x == agent.x && y == agent.y) continue;
                CellType c = world_get(&world, x, y);
                if (c == CELL_FOOD)          draw_cell(x, y, 0.2f, 0.9f, 0.2f);
                else if (c == CELL_POISON)   draw_cell(x, y, 0.9f, 0.2f, 0.2f);
                else                          draw_cell(x, y, 0.1f, 0.1f, 0.12f);
            }
        }

        draw_cell(agent.x, agent.y, 1.0f, 0.85f, 0.0f);

        char title[128];
        snprintf(title, sizeof(title), "Step %d | Vita=%.0f Fame=%.0f",
                 steps, agent.attr_values[0], agent.attr_values[1]);
        glfwSetWindowTitle(win, title);

        Action act = agent_decide(&agent, &world);
        agent_move(&agent, &world, act);
        steps++;

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    printf("Simulation ended at step %d, alive=%d\n", steps, agent.alive);

    agent_free(&agent);
    world_free(&world);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}