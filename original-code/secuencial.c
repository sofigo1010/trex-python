#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

typedef struct {
    float x;
    float v;
} Obstacle;

typedef struct {
    float x;
    float y;
} Player;

#define FPS 60
#define DT  (1.0f / FPS)

static inline float randf01(void) { return (float)rand() / (float)RAND_MAX; }
static inline int overlapX(float ax, float bx) { return (ax - bx < 0 ? bx - ax : ax - bx) <= 1.0f; }
static inline double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

// Update obstacle positions (parallelizable)
void actualizar_obstaculos(Obstacle* obs, int n, float dt) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        obs[i].x += obs[i].v * dt;
    }
}

// Count collisions (parallelizable reduction)
int verificar_colisiones(const Obstacle* obs, int n, const Player* p) {
    if (p->y >= 1.0f) return 0;
    int count = 0;
    #pragma omp parallel for reduction(+:count)
    for (int i = 0; i < n; ++i) {
        if (overlapX(p->x, obs[i].x)) count++;
    }
    return count;
}

void renderizar_escena(int frame, const Player* p, const Obstacle* obs, int n) {
    printf("frame=%4d | y=%.2f | obs0_x=% .2f\n",
           frame, p->y, (n>0? obs[0].x: -999.0f));
}

int main(int argc, char** argv) {
    int N_FRAMES = 300, N_OBS = 5;
    float JUMP_H = 1.2f, PROB_SALTO = 0.25f;
    Obstacle* obs = malloc(sizeof(Obstacle) * N_OBS);
    for (int i = 0; i < N_OBS; ++i) { obs[i].x = i*10; obs[i].v = -5.0f - i; }

    Player p = {5.0f, 0.0f};
    int colisiones_totales = 0;
    double t0 = now_ms();

    for (int frame = 0; frame < N_FRAMES; ++frame) {
        p.y = (randf01() < PROB_SALTO) ? JUMP_H : 0.0f;

        actualizar_obstaculos(obs, N_OBS, DT);   // parallel
        colisiones_totales += verificar_colisiones(obs, N_OBS, &p); // parallel
        renderizar_escena(frame, &p, obs, N_OBS);
    }

    double t1 = now_ms();
    printf("\nTotal collisions: %d\nTime elapsed: %.3f ms\n", colisiones_totales, t1 - t0);
    free(obs);
    return 0;
}
