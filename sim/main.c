#include "../network/network.h"
#include "world.h"
#include "agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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
    "uniform vec2 cell_size;"
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

static void draw_rect(float px, float py, float pw, float ph, float r, float g, float b)
{
    glUseProgram(prog);
    glUniform2f(glGetUniformLocation(prog, "cell_pos"), px, py);
    glUniform2f(glGetUniformLocation(prog, "cell_size"), pw, ph);
    glUniform2f(glGetUniformLocation(prog, "win_size"), (float)WIN_W, (float)WIN_H);
    glUniform3f(glGetUniformLocation(prog, "color"), r, g, b);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}

static void draw_cell(int x, int y, float r, float g, float b)
{
    draw_rect((float)x*CELL_PX, (float)y*CELL_PX, (float)CELL_PX, (float)CELL_PX, r, g, b);
}

static const char font3x5[10][5][3] = {
    {{1,1,1},{1,0,1},{1,0,1},{1,0,1},{1,1,1}},
    {{0,1,0},{1,1,0},{0,1,0},{0,1,0},{1,1,1}},
    {{1,1,1},{0,0,1},{1,1,1},{1,0,0},{1,1,1}},
    {{1,1,1},{0,0,1},{1,1,1},{0,0,1},{1,1,1}},
    {{1,0,1},{1,0,1},{1,1,1},{0,0,1},{0,0,1}},
    {{1,1,1},{1,0,0},{1,1,1},{0,0,1},{1,1,1}},
    {{1,1,1},{1,0,0},{1,1,1},{1,0,1},{1,1,1}},
    {{1,1,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1}},
    {{1,1,1},{1,0,1},{1,1,1},{1,0,1},{1,1,1}},
    {{1,1,1},{1,0,1},{1,1,1},{0,0,1},{1,1,1}},
};

static void draw_number(int cell_x, int cell_y, int num, float r, float g, float b)
{
    int pix = 3;
    int dw = 3 * pix;
    int dh = 5 * pix;
    char buf[8];
    int len = snprintf(buf, sizeof(buf), "%d", num);
    int total_w = len * dw + (len - 1) * pix;
    int ox = cell_x * CELL_PX + (CELL_PX - total_w) / 2;
    int oy = cell_y * CELL_PX + (CELL_PX - dh) / 2;

    for (int d = 0; d < len; d++) {
        int digit = buf[d] - '0';
        for (int row = 0; row < 5; row++) {
            for (int col = 0; col < 3; col++) {
                if (font3x5[digit][row][col]) {
                    draw_rect(
                        (float)(ox + d * (dw + pix) + col * pix),
                        (float)(oy + row * pix),
                        (float)pix, (float)pix, r, g, b
                    );
                }
            }
        }
    }
}

static void key_callback(GLFWwindow *win, int key, int scancode, int action, int mods)
{
    (void)scancode; (void)mods;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(win, GLFW_TRUE);
}

#define MUTATION_RATE 0.1f
#define MUTATION_MAG 0.2f

static void mutate(Agent *dst, Agent *src)
{
    for (int l = 0; l < src->total_layers; l++) {
        int postSize = src->post_sizes[l];
        int preSize = src->pre_sizes[l];
        for (int j = 0; j < postSize; j++) {
            for (int i = 0; i < preSize; i++) {
                float w = src->brain.weights[l][j][i];
                if ((float)rand() / RAND_MAX < MUTATION_RATE)
                    w += (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * MUTATION_MAG;
                dst->brain.weights[l][j][i] = w;
            }
        }
    }
}

int main(void)
{
    srand(time(NULL));

    int width = 30, height = 18;
    WIN_W = width * CELL_PX;
    WIN_H = height * CELL_PX;

    int num_agents = 15;
    int max_generations = 20;

    World world;
    world_init(&world, width, height);
    world_populate(&world, 20, 30);

    char *attr_names[] = { "Vita", "Fame" };
    int neurons[] = { 16 };
    Agent agents[15];
    for (int i = 0; i < num_agents; i++) {
        int ax = 2 + rand() % (width - 4);
        int ay = 2 + rand() % (height - 4);
        agent_init(&agents[i], ax, ay, 2, 1, neurons, 2, attr_names);
    }

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

    int gen = 0;
    while (!glfwWindowShouldClose(win) && gen < max_generations) {
        int steps = 0;
        int alive_count = num_agents;

        while (!glfwWindowShouldClose(win) && alive_count > 0 && steps < 2000) {
            glViewport(0, 0, WIN_W, WIN_H);
            glClear(GL_COLOR_BUFFER_BIT);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    CellType c = world_get(&world, x, y);
                    if (c == CELL_FOOD)          draw_cell(x, y, 0.2f, 0.9f, 0.2f);
                    else if (c == CELL_POISON)   draw_cell(x, y, 0.6f, 0.1f, 0.8f);
                    else                          draw_cell(x, y, 0.1f, 0.1f, 0.12f);
                }
            }

            alive_count = 0;
            for (int i = 0; i < num_agents; i++) {
                if (!agents[i].alive) continue;
                alive_count++;
                draw_cell(agents[i].x, agents[i].y, 1.0f, 0.85f, 0.0f);
                draw_number(agents[i].x, agents[i].y, i + 1, 0.0f, 0.0f, 0.0f);
            }

            char title[128];
            snprintf(title, sizeof(title), "Gen %d | Step %d | Alive %d/%d",
                     gen, steps, alive_count, num_agents);
            glfwSetWindowTitle(win, title);

            for (int i = 0; i < num_agents; i++) {
                if (!agents[i].alive) continue;
                Action act = agent_decide(&agents[i], &world);
                agent_move(&agents[i], &world, act);
            }
            steps++;

            glfwSwapBuffers(win);
            glfwPollEvents();
        }

        printf("=== Gen %d ended: step %d, alive=%d/%d ===\n", gen, steps, alive_count, num_agents);
        for (int i = 0; i < num_agents; i++) {
            printf("Agent %2d: food=%.0f poison=%.0f score=%g learn=%d\n",
                   i, agents[i].food_eaten, agents[i].poison_eaten,
                   agents[i].food_eaten - agents[i].poison_eaten,
                   agents[i].learn_events);
        }

        int rank[15];
        for (int i = 0; i < num_agents; i++) rank[i] = i;
        for (int i = 0; i < num_agents - 1; i++) {
            for (int j = i + 1; j < num_agents; j++) {
                float si = agents[rank[i]].food_eaten - agents[rank[i]].poison_eaten;
                float sj = agents[rank[j]].food_eaten - agents[rank[j]].poison_eaten;
                if (sj > si) {
                    int tmp = rank[i]; rank[i] = rank[j]; rank[j] = tmp;
                }
            }
        }

        int keep = num_agents / 2;
        for (int k = 0; k < keep; k++) {
            Agent *src = &agents[rank[k]];
            int ax = 2 + rand() % (width - 4);
            int ay = 2 + rand() % (height - 4);
            agent_reset(src, ax, ay);
        }
        for (int k = keep; k < num_agents; k++) {
            Agent *src = &agents[rank[k % keep]];
            Agent *dst = &agents[rank[k]];
            mutate(dst, src);
            int ax = 2 + rand() % (width - 4);
            int ay = 2 + rand() % (height - 4);
            agent_reset(dst, ax, ay);
        }

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                world_set(&world, x, y, CELL_EMPTY);
        world_populate(&world, 20, 30);

        gen++;
    }

    for (int i = 0; i < num_agents; i++)
        agent_free(&agents[i]);
    world_free(&world);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}