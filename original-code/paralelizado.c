#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define FPS 60
#define DT  (1.0f / FPS)

typedef struct {
    float x;
    float v;
} Obstacle;

typedef struct {
    float x;
    float y;
} Player;

// ----------------- UTILIDADES -----------------
static inline float randf01(void) {
    return (float)rand() / (float)RAND_MAX;
}

static inline int overlapX(float ax, float bx) {
    float dx = ax - bx;
    if (dx < 0) dx = -dx;
    return (dx <= 1.0f);
}

static inline double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

// ----------------- FUNCIONES DEL SIMULADOR -----------------
void actualizar_obstaculos(Obstacle* obs, int n, float dt) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        obs[i].x += obs[i].v * dt;
    }
}

int verificar_colisiones(const Obstacle* obs, int n, const Player* p) {
    if (p->y >= 1.0f) return 0;
    int c = 0;
    #pragma omp parallel for reduction(+:c)
    for (int i = 0; i < n; ++i) {
        if (overlapX(p->x, obs[i].x)) {
            c++;
        }
    }
    return c;
}

void renderizar_escena(int frame, const Player* p, const Obstacle* obs, int n) {
    printf("frame=%d  player.y=%.2f  obs0.x=% .2f\n",
           frame, p->y, (n > 0 ? obs[0].x : -999.0f));
}

// ----------------- SIMULACION -----------------
void simular_paralelo(Obstacle* obs, int nObs,
                      int N_FRAMES, float JUMP_H, float PROB_SALTO) {
    Player player = {5.0f, 0.0f};
    int colisiones_totales = 0;
    srand((unsigned)time(NULL));

    for (int frame = 0; frame < N_FRAMES; ++frame) {
        player.y = (randf01() < PROB_SALTO) ? JUMP_H : 0.0f;

        int colisiones_frame = 0;

        #pragma omp parallel sections firstprivate(frame)
        {
            #pragma omp section
            actualizar_obstaculos(obs, nObs, DT);

            #pragma omp section
            colisiones_frame = verificar_colisiones(obs, nObs, &player);

            #pragma omp section
            renderizar_escena(frame, &player, obs, nObs);
        }

        colisiones_totales += colisiones_frame;
    }

    printf("Colisiones totales: %d\n", colisiones_totales);
    if (colisiones_totales < 5) printf("RESULTADO: SOBREVIVE\n");
    else if (colisiones_totales > 5) printf("RESULTADO: MUERE\n");
    else printf("RESULTADO: LIMITE (5 colisiones)\n");
}

// ----------------- MAIN -----------------
int main(int argc, char** argv) {
    int N_FRAMES = 300, N_OBS = 5;
    float JUMP_H = 1.2f, PROB_SALTO = 0.25f;

    Obstacle* obs = malloc(sizeof(Obstacle) * N_OBS);
    for (int i = 0; i < N_OBS; ++i) {
        obs[i].x = i * 10;
        obs[i].v = -5.0f - i;
    }

    double t0 = now_ms();
    simular_paralelo(obs, N_OBS, N_FRAMES, JUMP_H, PROB_SALTO);
    double t1 = now_ms();

    printf("Tiempo total simulacion: %.3f ms\n", t1 - t0);
    free(obs);
    return 0;
}
