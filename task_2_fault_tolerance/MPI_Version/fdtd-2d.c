#include "fdtd-2d.h"
#include "mpi.h"

double bench_t_start, bench_t_end;
MPI_Status status;

static
double rtclock()
{
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, NULL);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
  bench_t_start = rtclock();
}

void bench_timer_stop()
{
  bench_t_end = rtclock();
}

void bench_timer_print()
{
  printf("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
  fflush(stdout);
}

static
void init_array(int tmax, int nx, int ny, float ex[nx][ny], float ey[nx][ny], float hz[nx][ny])
{
  int i, j;
  for(i = 0; i < nx; i++)
    for(j = 0; j < ny; j++) {
      ex[i][j] = ((float) i * (j + 1)) / nx;
      ey[i][j] = ((float) i * (j + 2)) / ny;
      hz[i][j] = ((float) i * (j + 3)) / nx;
    }
}

static
void print_array(int nx, int ny, float ex[nx][ny], float ey[nx][ny], float hz[nx][ny])
{
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "ex");
  for(i = 0; i < nx; ++i)
    for(j = 0; j < ny; ++j) {
      if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2f ", ex[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "ex");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");

  fprintf(stderr, "begin dump: %s", "ey");
  for(i = 0; i < nx; ++i)
    for(j = 0; j < ny; ++j) {
      if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2f ", ey[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "ey");

  fprintf(stderr, "begin dump: %s", "hz");
  for(i = 0; i < nx; ++i)
    for(j = 0; j < ny; ++j) {
      if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2f ", hz[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "hz");
}

static
void kernel_fdtd_2d(int tmax, int nx, int ny, int rows, float ex[rows][ny],
                    float ey[rows][ny], float hz[rows][ny], int ProcNum, int ProcRank)
{
  int t, i, j;
  float *ex_p = (float *) malloc(ny * sizeof(float));
  float *ey_p = (float *) malloc(ny * sizeof(float));
  float *hz_p = (float *) malloc(ny * sizeof(float));
  for(t = 0; t < tmax; ++t) {

    if (ProcRank == 0) {
      MPI_Request masRequest;
      MPI_Status masStat;
      MPI_Isend(&hz[rows - 1][0], ny, MPI_FLOAT, ProcRank + 1, 1, MPI_COMM_WORLD, &masRequest);
      for (i = 1; i < rows; ++i) {
        for (j = 0; j < ny; ++j) {
          ey[i][j] -= 0.5f * (hz[i][j] - hz[i - 1][j]);
        }
      }
      MPI_Waitall(1, &masRequest, &masStat);
    } else if (ProcRank != ProcNum - 1) {
      MPI_Request masRequest[2];
      MPI_Status masStat[2];
      MPI_Irecv(hz_p, ny, MPI_FLOAT, ProcRank - 1, 1, MPI_COMM_WORLD, &masRequest[0]);
      MPI_Isend(&hz[rows - 1][0], ny, MPI_FLOAT, ProcRank + 1, 1, MPI_COMM_WORLD, &masRequest[1]);
      MPI_Waitall(2, masRequest, masStat);
      for (i = 0; i < rows; ++i) {
        for (j = 0; j < ny; ++j) {
          if (i == 0) {
            ey[i][j] -= 0.5f * (hz[i][j] - hz_p[j]);
          } else {
            ey[i][j] -= 0.5f * (hz[i][j] - hz[i - 1][j]);
          }
        }
      }
    } else {
      MPI_Request masRequest;
      MPI_Status masStat;
      MPI_Irecv(hz_p, ny, MPI_FLOAT, ProcRank - 1, 1, MPI_COMM_WORLD, &masRequest);
      MPI_Waitall(1, &masRequest, &masStat);
      for (i = 0; i < rows; ++i) {
        for (j = 0; j < ny; ++j) {
          if (i == 0) {
            ey[i][j] -= 0.5f * (hz[i][j] - hz_p[j]);
          } else {
            ey[i][j] -= 0.5f * (hz[i][j] - hz[i - 1][j]);
          }
        }
      }
    }

    for (i = 0; i < rows; ++i) {
      for (j = 1; j < ny; ++j) {
        ex[i][j] -= 0.5f * (hz[i][j] - hz[i][j - 1]);
      }
    }

    if (ProcRank == 0) {
      MPI_Request masRequest;
      MPI_Status masStat;
      MPI_Irecv(ey_p, ny, MPI_FLOAT, ProcRank + 1, 1, MPI_COMM_WORLD, &masRequest);
      MPI_Waitall(1, &masRequest, &masStat);
    } else if (ProcRank != ProcNum - 1) {
      MPI_Request masRequest[2];
      MPI_Status masStat[2];
      MPI_Irecv(ey_p, ny, MPI_FLOAT, ProcRank + 1, 1, MPI_COMM_WORLD, &masRequest[0]);
      MPI_Isend((float *) ey, ny, MPI_FLOAT, ProcRank - 1, 1, MPI_COMM_WORLD, &masRequest[1]);
      MPI_Waitall(2, masRequest, masStat);
    } else {
      MPI_Request masRequest;
      MPI_Status masStat;
      MPI_Isend((float *) ey, ny, MPI_FLOAT, ProcRank - 1, 1, MPI_COMM_WORLD, &masRequest);
      MPI_Waitall(1, &masRequest, &masStat);
    }
    if (ProcRank == 0) {
      for (j = 0; j < ny - 1; ++j) {
        if (rows == 1) {
          hz[0][j] -= 0.7f * (ex[0][j + 1] - ex[0][j] + ey_p[j] - t);
        }
        else {
          hz[0][j] -= 0.7f * (ex[0][j + 1] - ex[0][j] + ey[1][j] - t);
        }
      }
    }

    if (ProcRank == 0) {

      for(i = 1; i < rows && i < nx - 1; ++i) {
        for (j = 0; j < ny - 1; ++j) {
          if (i + 1 >= rows) {
            hz[i][j] -= 0.7f * (ex[i][j + 1] - ex[i][j] + ey_p[j] - ey[i][j]);
          } else {
            hz[i][j] -= 0.7f * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
          }
        }
      }
    } else if (ProcRank != ProcNum - 1) {
      for(i = 0; i < rows; ++i) {
        for (j = 0; j < ny - 1; ++j) {
          if (i + 1 >= rows) {
            hz[i][j] -= 0.7f * (ex[i][j + 1] - ex[i][j] + ey_p[j] - ey[i][j]);
          } else {
            hz[i][j] -= 0.7f * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
          }
        }
      }
    } else {
      for(i = 0; i < rows - 1; ++i) {
        for (j = 0; j < ny - 1; ++j) {
          if (i + 1 >= rows) {
            hz[i][j] -= 0.7f * (ex[i][j + 1] - ex[i][j] + ey_p[j] - ey[i][j]);
          } else {
            hz[i][j] -= 0.7f * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
          }
        }
      }
    }
    //MPI_Barrier(MPI_COMM_WORLD);
  }
  if (ProcRank == 0) {
    for (j = 0; j < ny; j++)
      ey[0][j] = tmax - 1;
  }
  free((void *) ex_p);
  free((void *) ey_p);
  free((void *) hz_p);
}

int main(int argc, char **argv)
{

  int success = MPI_Init(&argc, &argv);
  if (success)
  {
    fprintf(stderr, "\nend   dump: %s\n", "Ошибка запуска, выполнение остановлено ");
    MPI_Abort(MPI_COMM_WORLD, success);
  }
  int ProcNum;
  int ProcRank;
  MPI_Comm_size (MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank (MPI_COMM_WORLD, &ProcRank);

  int tmax = TMAX;
  int nx = NX;
  int ny = NY;
  float (*ex)[nx][ny];
  float (*ey)[nx][ny];
  float (*hz)[nx][ny];

  float *ex_p = (float *) malloc((nx / ProcNum + (nx % ProcNum > ProcRank)) * ny * sizeof(float));
  float *ey_p = (float *) malloc((nx / ProcNum + (nx % ProcNum > ProcRank)) * ny * sizeof(float));
  float *hz_p = (float *) malloc((nx / ProcNum + (nx % ProcNum > ProcRank)) * ny * sizeof(float));

  if (ProcRank == 0) {
    ex = (float (*)[nx][ny]) malloc((nx) * (ny) * sizeof(float));

    ey = (float (*)[nx][ny]) malloc((nx) * (ny) * sizeof(float));

    hz = (float (*)[nx][ny]) malloc((nx) * (ny) * sizeof(float));
    init_array(tmax, nx, ny,
               *ex,
               *ey,
               *hz);
    free((void *) ex_p);
    free((void *) ey_p);
    free((void *) hz_p);
    ex_p = (float *) ex;
    ey_p = (float *) ey;
    hz_p = (float *) hz;

    bench_timer_start();
    int next = (nx / ProcNum + (nx % ProcNum > 0)) * ny;
    for (int i = 1; i < ProcNum; ++i) {
      MPI_Request s;
      MPI_Isend((float *) ex + next, (nx / ProcNum + (nx % ProcNum > i)) * ny, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &s);
      MPI_Isend((float *) ey + next, (nx / ProcNum + (nx % ProcNum > i)) * ny, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &s);
      MPI_Isend((float *) hz + next, (nx / ProcNum + (nx % ProcNum > i)) * ny, MPI_FLOAT, i, 3, MPI_COMM_WORLD, &s);
      next += (nx / ProcNum + (nx % ProcNum > i)) * ny;
    }
  } else {
    MPI_Recv(ex_p, (nx / ProcNum + (nx % ProcNum > ProcRank)) * ny,MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(ey_p, (nx / ProcNum + (nx % ProcNum > ProcRank)) * ny, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(hz_p, (nx / ProcNum + (nx % ProcNum > ProcRank)) * ny, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  int rows = nx / ProcNum + (nx % ProcNum > ProcRank);
  float (*xx)[rows][ny] = (float (*)[rows][ny]) ex_p;
  float (*yy)[rows][ny] = (float (*)[rows][ny]) ey_p;
  float (*hh)[rows][ny] = (float (*)[rows][ny]) hz_p;
  kernel_fdtd_2d(tmax, nx, ny, rows, *xx, *yy, *hh, ProcNum, ProcRank);

  if (ProcRank == 0) {
    int next = (nx / ProcNum + (nx % ProcNum > 0)) * ny;
    MPI_Request masRequest_ex[ProcNum-1];
    MPI_Request masRequest_ey[ProcNum-1];
    MPI_Request masRequest_hz[ProcNum-1];
    MPI_Status masStat_x[ProcNum-1];
    MPI_Status masStat_y[ProcNum-1];
    MPI_Status masStat_h[ProcNum-1];
    for (int i = 1; i < ProcNum; ++i) {
      MPI_Irecv((float *) ex + next, (nx / ProcNum + (nx % ProcNum > i)) * ny, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &masRequest_ex[i - 1]);
      MPI_Irecv((float *) ey + next, (nx / ProcNum + (nx % ProcNum > i)) * ny, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &masRequest_ey[i - 1]);
      MPI_Irecv((float *) hz + next, (nx / ProcNum + (nx % ProcNum > i)) * ny, MPI_FLOAT, i, 3, MPI_COMM_WORLD, &masRequest_hz[i - 1]);
      next += (nx / ProcNum + (nx % ProcNum > i)) * ny;
    }
    MPI_Waitall(ProcNum - 1, masRequest_ex, masStat_x);
    MPI_Waitall(ProcNum - 1, masRequest_ey, masStat_y);
    MPI_Waitall(ProcNum - 1, masRequest_hz, masStat_h);
  } else {
    MPI_Request masRequest[3];
    MPI_Status masStat[3];
    MPI_Isend(ex_p, (nx / ProcNum + (nx % ProcNum > ProcRank)) * ny, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &masRequest[0]);
    MPI_Isend(ey_p, (nx / ProcNum + (nx % ProcNum > ProcRank)) * ny, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &masRequest[1]);
    MPI_Isend(hz_p, (nx / ProcNum + (nx % ProcNum > ProcRank)) * ny, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &masRequest[2]);
    MPI_Waitall(3, masRequest, masStat);
  }
  if (ProcRank == 0) {
    bench_timer_stop();
    bench_timer_print();
    print_array(nx, ny, *ex, *ey, *hz);
    fflush(stdout);

    free((void *) ex);
    free((void *) ey);
    free((void *) hz);
  } else {
    free((void *) ex_p);
    free((void *) ey_p);
    free((void *) hz_p);
  }
  MPI_Finalize();
  return 0;
}
