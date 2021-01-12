#include "fdtd-2d.h"
#include "mpi.h"

double bench_t_start, bench_t_end;
int ReservProcNum;
int ReservProcBoss;
int StartNumberReservProc;
int ProcNum;
int ProcRank;
MPI_Status status;
int tmax = TMAX;
int nx = NX;
int ny = NY;

int is_proc_bad = 0;
int number_block_proc_erase;
int number_iterate_in_block_proc_erase;
int* numbers_of_correct_proc;

int id_iterate_failll = -1;

int skip_inizialize = 0;

// Сохранение данных в файл в контрольной точке
static void save_checkpoint(int block_id, int iterate, int rows, float ex[rows][ny],
                     float ey[rows][ny], float hz[rows][ny]) {
  char name_file[50];
  snprintf(name_file, 50, "save_point_%d_%d_%d.txt", ProcRank, block_id, iterate);
  FILE *file = fopen(name_file, "w");

  // Пишем в файл ex
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < ny; ++j) {
      fprintf(file, "%0.20f ", ex[i][j]);
    }
    fprintf(file, "\n");
  }

  // Пишем в файл ey
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < ny; ++j) {
      fprintf(file, "%0.20f ", ey[i][j]);
    }
    fprintf(file, "\n");
  }

  // Пишем в файл hz
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < ny; ++j) {
      fprintf(file, "%0.20f ", hz[i][j]);
    }
    fprintf(file, "\n");
  }
  fclose(file);
}

// Восстановление последней сохранненой контрольной точки
static void load_checkpoint(int block_id, int iterate, int rows, float ex[rows][ny],
                            float ey[rows][ny], float hz[rows][ny]) {
  char name_file[50];
  snprintf(name_file, 50, "save_point_%d_%d_%d.txt", ProcRank, block_id, iterate);
  FILE *file = fopen(name_file, "r");
  if (file == NULL) {
    return;
  }
  // Читаем из файла ex
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < ny; ++j) {
      fscanf(file, "%f ", &ex[i][j]);
    }
  }

  // Читаем из файла ey
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < ny; ++j) {
      fscanf(file, "%f ", &ey[i][j]);
    }
  }

  // Читаем из файла hz
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < ny; ++j) {
      fscanf(file, "%f ", &hz[i][j]);
    }
  }
  fclose(file);

  fprintf(stderr, "Checkpoint for proc with Number: %d succesed\n", ProcNum);
  fflush(stderr);
}

static void try_suicide(int block_id, int iterate) {
  if (is_proc_bad && block_id == number_block_proc_erase && iterate == number_iterate_in_block_proc_erase) {
    fprintf(stderr, "Процес с номером: %d умирает в блоке: %d на итерации: %d\n",
            ProcRank, block_id, iterate);
    fflush(stderr);
    raise(SIGTERM);
  }
}

static double rtclock() {
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, NULL);
  if (stat != 0) {
    fprintf(stdout, "Error return from gettimeofday: %d", stat);
    fflush(stdout);
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start() {
  bench_t_start = rtclock();
}

void bench_timer_stop() {
  bench_t_end = rtclock();
}

void bench_timer_print() {
  fprintf(stdout, "Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
  fflush(stdout);
}

static void init_array(float ex[nx][ny], float ey[nx][ny], float hz[nx][ny]) {
  int i, j;
  for(i = 0; i < nx; i++)
    for(j = 0; j < ny; j++) {
      ex[i][j] = ((float) i * (j + 1)) / nx;
      ey[i][j] = ((float) i * (j + 2)) / ny;
      hz[i][j] = ((float) i * (j + 3)) / nx;
    }
}

static void print_array(float ex[nx][ny], float ey[nx][ny], float hz[nx][ny]) {
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

static void kernel_fdtd_2d(int rows, float ex[rows][ny],
                    float ey[rows][ny], float hz[rows][ny]) {
  try_suicide(1, 0);
  int t = 0, i, j;
  float *ex_p = (float *) malloc(ny * sizeof(float));
  float *ey_p = (float *) malloc(ny * sizeof(float));
  float *hz_p = (float *) malloc(ny * sizeof(float));
  if (id_iterate_failll != -1) {
    t = id_iterate_failll / 3;
    if (t == tmax) {
      goto iter_last;
    }
    int mod = id_iterate_failll % 3;
    if (mod == 0) {
      goto iter_0;
    } else if (mod == 1) {
      goto iter_1;
    } else {
      goto iter_2;
    }
  }
  int warn;
  MPI_Status masStat;
  MPI_Request mpi_request_;
  int check_;
  for(; t < tmax; ++t) {
    try_suicide(1, t * 3);
iter_0:
    if (ProcRank == 0) {
      MPI_Isend(&hz[rows - 1][0], ny, MPI_FLOAT, numbers_of_correct_proc[ProcRank + 1], 1, MPI_COMM_WORLD, &mpi_request_);
      warn = MPI_Recv(&check_, 1, MPI_INT, numbers_of_correct_proc[ProcRank + 1], 1, MPI_COMM_WORLD, &masStat);
      if (warn) {
        int test[3];
        test[0] = numbers_of_correct_proc[ProcRank + 1];
        test[1] = 1;
        test[2] = t * 3;
        MPI_Ssend(&test, 3, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD);
        MPI_Recv(&numbers_of_correct_proc[ProcRank + 1], 1, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD, &masStat);
        goto iter_0;
      }
      for (i = 1; i < rows; ++i) {
        for (j = 0; j < ny; ++j) {
          ey[i][j] -= 0.5f * (hz[i][j] - hz[i - 1][j]);
        }
      }
    } else if (ProcRank != ProcNum - 1) {
      MPI_Request masRequest;
      MPI_Status masStatSend;
      MPI_Status masStatEnd;
block_1_back_recv_0_0_0:
      MPI_Isend(&hz[rows - 1][0], ny, MPI_FLOAT, numbers_of_correct_proc[ProcRank + 1], 1, MPI_COMM_WORLD, &mpi_request_);
      warn = MPI_Recv(&check_, 1, MPI_INT, numbers_of_correct_proc[ProcRank + 1], 1, MPI_COMM_WORLD, &masStat);
      if (warn) {
        int test[3];
        test[0] = numbers_of_correct_proc[ProcRank + 1];
        test[1] = 1;
        test[2] = t * 3;
        MPI_Ssend(&test, 3, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD);
        MPI_Recv(&numbers_of_correct_proc[ProcRank + 1], 1, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD, &masStat);
        goto block_1_back_recv_0_0_0;
      }
block_1_back_recv_0:
      warn = MPI_Recv(hz_p, ny, MPI_FLOAT, numbers_of_correct_proc[ProcRank - 1], 1, MPI_COMM_WORLD, &masStatEnd);
      MPI_Isend(&check_, 1, MPI_INT, numbers_of_correct_proc[ProcRank - 1], 1, MPI_COMM_WORLD, &mpi_request_);
      if (warn) {
        int test[3];
        test[0] = numbers_of_correct_proc[ProcRank - 1];
        test[1] = 1;
        test[2] = t * 3;
        MPI_Ssend(&test, 3, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD);
        MPI_Recv(&numbers_of_correct_proc[ProcRank - 1], 1, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD, &masStat);
        goto block_1_back_recv_0;
      }
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
block_1_back_recv_1:
      warn = MPI_Recv(hz_p, ny, MPI_FLOAT, numbers_of_correct_proc[ProcRank - 1], 1, MPI_COMM_WORLD, &masStat);
      MPI_Isend(&check_, 1, MPI_INT, numbers_of_correct_proc[ProcRank - 1], 1, MPI_COMM_WORLD, &mpi_request_);
      if (warn) {
        int test[3];
        test[0] = numbers_of_correct_proc[ProcRank - 1];
        test[1] = 1;
        test[2] = t * 3;
        MPI_Ssend(&test, 3, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD);
        MPI_Recv(&numbers_of_correct_proc[ProcRank - 1], 1, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD, &masStat);
        goto block_1_back_recv_1;
      }
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
    if (ProcRank != 0) {
      save_checkpoint(1, t * 3, rows, ex, ey, hz);
    }
    try_suicide(1, t * 3 + 1);
iter_1:
    for (i = 0; i < rows; ++i) {
      for (j = 1; j < ny; ++j) {
        ex[i][j] -= 0.5f * (hz[i][j] - hz[i][j - 1]);
      }
    }

    if (ProcRank == 0) {
block_0_back_recv_0_1:
      warn = MPI_Recv(ey_p, ny, MPI_FLOAT, numbers_of_correct_proc[ProcRank + 1], 1, MPI_COMM_WORLD, &masStat);
      MPI_Isend(&check_, 1, MPI_INT, numbers_of_correct_proc[ProcRank + 1], 1, MPI_COMM_WORLD, &mpi_request_);
      if (warn) {
        int test[3];
        test[0] = numbers_of_correct_proc[ProcRank + 1];
        test[1] = 1;
        test[2] = t * 3 + 1;
        MPI_Ssend(&test, 3, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD);
        MPI_Recv(&numbers_of_correct_proc[ProcRank + 1], 1, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD, &masStat);
        goto block_0_back_recv_0_1;
      }
    } else if (ProcRank != ProcNum - 1) {
      MPI_Request masRequest;
      MPI_Status masStatSend;
      MPI_Status masStatEnd;
block_0_back_send_1_1:
      MPI_Isend((float *) ey, ny, MPI_FLOAT, numbers_of_correct_proc[ProcRank - 1], 1, MPI_COMM_WORLD, &mpi_request_);
      warn = MPI_Recv(&check_, 1, MPI_INT, numbers_of_correct_proc[ProcRank - 1], 1, MPI_COMM_WORLD, &masStat);
      if (warn) {
        int test[3];
        test[0] = numbers_of_correct_proc[ProcRank - 1];
        test[1] = 1;
        test[2] = t * 3 + 1;
        MPI_Ssend(&test, 3, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD);
        MPI_Recv(&numbers_of_correct_proc[ProcRank - 1], 1, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD, &masStat);
        goto block_0_back_send_1_1;
      }
block_0_back_recv_1_1:
      warn = MPI_Recv(ey_p, ny, MPI_FLOAT, numbers_of_correct_proc[ProcRank + 1], 1, MPI_COMM_WORLD, &masStatEnd);
      MPI_Isend(&check_, 1, MPI_INT, numbers_of_correct_proc[ProcRank + 1], 1, MPI_COMM_WORLD, &mpi_request_);
      if (warn) {
        int test[3];
        test[0] = numbers_of_correct_proc[ProcRank + 1];
        test[1] = 1;
        test[2] = t * 3 + 1;
        MPI_Ssend(&test, 3, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD);
        MPI_Recv(&numbers_of_correct_proc[ProcRank + 1], 1, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD, &masStat);
        goto block_0_back_recv_1_1;
      }
    } else {
block_1_go_back_a:
      MPI_Isend((float *) ey, ny, MPI_FLOAT, numbers_of_correct_proc[ProcRank - 1], 1, MPI_COMM_WORLD, &mpi_request_);
      warn = MPI_Recv(&check_, 1, MPI_INT, numbers_of_correct_proc[ProcRank - 1], 1, MPI_COMM_WORLD, &masStat);
      if (warn) {
        int test[3];
        test[0] = numbers_of_correct_proc[ProcRank - 1];
        test[1] = 1;
        test[2] = t * 3 + 1;
        MPI_Ssend(&test, 3, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD);
        MPI_Recv(&numbers_of_correct_proc[ProcRank - 1], 1, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD, &masStat);
        goto block_1_go_back_a;
      }
    }
    if (ProcRank != 0) {
      save_checkpoint(1, t * 3 + 1, rows, ex, ey, hz);
    }
    try_suicide(1, t * 3 + 2);
iter_2:
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
    if (ProcRank != 0) {
      save_checkpoint(1, t * 3 + 2, rows, ex, ey, hz);
    }
  }
  try_suicide(1, tmax * 3);
iter_last:
  if (ProcRank == 0) {
    for (j = 0; j < ny; j++)
      ey[0][j] = tmax - 1;
  }
  free((void *) ex_p);
  free((void *) ey_p);
  free((void *) hz_p);
}

void recieve_results(float (*ex)[nx][ny], float (*ey)[nx][ny], float (*hz)[nx][ny]) {
  int next = (nx / ProcNum + (nx % ProcNum > 0)) * ny;
  MPI_Status masStat_x[ProcNum - 1];
  MPI_Status masStat_y[ProcNum - 1];
  MPI_Status masStat_h[ProcNum - 1];
  int warn;
  for (int i = 1; i < ProcNum; ++i) {
back_1:
    warn = MPI_Recv((float *) ex + next, (nx / ProcNum + (nx % ProcNum > i)) * ny, MPI_FLOAT, numbers_of_correct_proc[i], 1, MPI_COMM_WORLD, &masStat_x[i - 1]);
    if (warn) {
      int test[3];
      test[0] = numbers_of_correct_proc[i];
      test[1] = 2;
      test[2] = 0;
      MPI_Ssend(&test, 3, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD);
      MPI_Recv(&numbers_of_correct_proc[i], 1, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD, &masStat_x[i - 1]);
      goto back_1;
    }
back_2:
    warn = MPI_Recv((float *) ey + next, (nx / ProcNum + (nx % ProcNum > i)) * ny, MPI_FLOAT, numbers_of_correct_proc[i], 2, MPI_COMM_WORLD, &masStat_y[i - 1]);
    if (warn) {
      int test[3];
      test[0] = numbers_of_correct_proc[i];
      test[1] = 2;
      test[2] = 0;
      MPI_Ssend(&test, 3, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD);
      MPI_Recv(&numbers_of_correct_proc[i], 1, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD, &masStat_x[i - 1]);
      goto back_2;
    }
back_3:
    warn = MPI_Recv((float *) hz + next, (nx / ProcNum + (nx % ProcNum > i)) * ny, MPI_FLOAT, numbers_of_correct_proc[i], 3, MPI_COMM_WORLD, &masStat_h[i - 1]);
    if (warn) {
      int test[3];
      test[0] = numbers_of_correct_proc[i];
      test[1] = 2;
      test[2] = 0;
      MPI_Ssend(&test, 3, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD);
      MPI_Recv(&numbers_of_correct_proc[i], 1, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD, &masStat_x[i - 1]);
      goto back_3;
    }
    next += (nx / ProcNum + (nx % ProcNum > i)) * ny;
  }
}

void send_complete_result_to_main_proc(int rows, float (*ex_p)[rows][ny],
                                       float (*ey_p)[rows][ny], float (*hz_p)[rows][ny]) {
  MPI_Request masRequest[3];
  MPI_Status masStat[3];
  MPI_Isend(ex_p, rows * ny, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &masRequest[0]);
  MPI_Isend(ey_p, rows * ny, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &masRequest[1]);
  MPI_Isend(hz_p, rows * ny, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &masRequest[2]);
  MPI_Waitall(3, masRequest, masStat);
}

// Выбираем cnt_bad_proces рандомных процессов, которые упадут
void creat_configuration(int cnt_bad_proces) {
  if (!cnt_bad_proces) {
    return;
  }
  srand(RANDOM_SEED);
  is_proc_bad = 0;  // Нулевой процесс в этой программе не падает
  for (int i = 0; i < cnt_bad_proces; ++i) {
    int num_bad_proc = rand() % (ProcNum - 1) + 1;
    if (num_bad_proc == ProcRank) {
      is_proc_bad = 1;
      break;
    }
  }
}

int main(int argc, char **argv) {
  int success = MPI_Init(&argc, &argv);
  if (success) {
    fprintf(stderr, "\nend   dump: %s\n", "Ошибка запуска, выполнение остановлено ");
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, success);
  }
  MPI_Comm_size (MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank (MPI_COMM_WORLD, &ProcRank);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  int cnt_bad_proces;
  if (argc == 2) {
    cnt_bad_proces = atoi(argv[1]); // Количество процессов, которые будут зафейлены
  } else {
    cnt_bad_proces = 0;
  }
  ReservProcBoss = ProcNum - 1;
  ReservProcNum = cnt_bad_proces;
  StartNumberReservProc = ReservProcBoss - cnt_bad_proces;
  ProcNum = StartNumberReservProc;

  if (ProcNum < 2) {
    mpi_printf(stderr, "Для эффективной работы программы требуется более одного процесса.\n"
    "В распоряжении данной программы всего: %d процессов.\n"
    "Увеличьте количество процессов.\n", ProcNum);
    fflush(stderr);
    MPI_Finalize();
    return 0;
  }
  if (cnt_bad_proces > ProcNum - 1) {
    mpi_printf(stderr, "Количество процессов, которые погибнут меньше количества доступных процессов"
                       " для этой операции.\n Увеличьте количество процессов или уменьшите количество"
                       " процессов, которые погибнут.\n");
    fflush(stderr);
    MPI_Finalize();
    return 0;
  }
  if (ProcNum > nx) {
    mpi_printf(stderr, "Количество процессов: %d, которые будут вычислять результат больше доступного: %d.\n"
                       " Уменьшите количество процессов или увеличьте датасет.\n", ProcNum, nx);
    fflush(stderr);
    MPI_Finalize();
    return 0;
  }
  if (ProcRank < ProcNum) {
    creat_configuration(cnt_bad_proces);
  }

  numbers_of_correct_proc = (int *) malloc(ProcNum * sizeof(int));
  for (int i = 0; i < ProcNum; ++i) {
    numbers_of_correct_proc[i] = i;
  }

  if (is_proc_bad && ProcRank < ProcNum) {
    // Есть три основных блока в программе:
    //        - отправка стартовых данных из 0-го процесса всем остальным
    //        - отправка вычисление результата
    //        - отправка вычисленных результатов 0-му процессу
    // Определим с помощью рандома блок, в котором упадет данный процесс
    srand(ProcRank);
    number_block_proc_erase = rand() % 3;
    if (number_block_proc_erase == 0) {
      number_iterate_in_block_proc_erase = 0;
    } else if (number_block_proc_erase == 1) {
      number_iterate_in_block_proc_erase = rand() % (tmax * 3 + 1);
    } else {
      number_iterate_in_block_proc_erase = 0;
    }
    fprintf(stderr, "Процес с номером: %d умрет в блоке: %d на итерации: %d \n",
               ProcRank, number_block_proc_erase, number_iterate_in_block_proc_erase);
    fflush(stderr);
  }
  float (*ex)[nx][ny];
  float (*ey)[nx][ny];
  float (*hz)[nx][ny];
  float *ex_p, *ey_p, *hz_p;
  int rows = nx / ProcNum + (nx % ProcNum > ProcRank);

  if (ProcRank == ReservProcBoss) {
    int test[ReservProcBoss * 3];
    int end = -1;
    for (int i = 0; i < ReservProcBoss * 3; ++i) {
      test[i] = -1;
    }
    int uselles[ReservProcBoss];
    for (int i = 0; i < ReservProcBoss; ++i) {
      uselles[i] = 0;
    }
    MPI_Status status_test;
    MPI_Request request_test[ReservProcBoss + 1];
    for (int i = 0; i < ReservProcBoss; i++) {
      MPI_Irecv(&test[i * 3], 3, MPI_INT, i, 666, MPI_COMM_WORLD, &request_test[i]);
    }
    MPI_Irecv(&end, 1, MPI_INT, 0, 555, MPI_COMM_WORLD, &request_test[ReservProcBoss]);
    while(1) {
      if (end == 1) {
        for (int i = StartNumberReservProc; i < ReservProcBoss; ++i) {
          int s = 1;
          MPI_Ssend(&s, 1, MPI_INT, i, 999, MPI_COMM_WORLD);
        }
        MPI_Finalize();
        return 0;
      }
      int id_proc = -1;
      for (int i = 0; i <= ReservProcBoss; i++) {
        if (!uselles[i]) {
          int flag;
          MPI_Test(&request_test[i], &flag, &status_test);
          if (flag) {
            MPI_Wait(&request_test[i], &status_test);
            if (end == 1) {
              for (int j = StartNumberReservProc; j < ReservProcBoss; ++j) {
                int s = 1;
                MPI_Ssend(&s, 1, MPI_INT, j, 999, MPI_COMM_WORLD);
              }
              MPI_Finalize();
              return 0;
            }
            if (test[i * 3] != -1 && test[i * 3 + 1] != -1 && test[i * 3 + 2] != -1) {
              int id_fail_proc = test[i * 3];
              int id_block_fail = test[i * 3 + 1];
              int id_iterate_fail = test[i * 3 + 2];
              if (numbers_of_correct_proc[id_fail_proc] >= ProcNum) {
                MPI_Ssend(&numbers_of_correct_proc[id_fail_proc], 1, MPI_INT, i, 666, MPI_COMM_WORLD);
              } else {
                numbers_of_correct_proc[id_fail_proc] = StartNumberReservProc++;
                MPI_Ssend(&test[i * 3], 3, MPI_INT, numbers_of_correct_proc[id_fail_proc], 777, MPI_COMM_WORLD);
                MPI_Request_free(&request_test[id_fail_proc]);
                MPI_Ssend(&numbers_of_correct_proc[id_fail_proc], 1, MPI_INT, i, 666, MPI_COMM_WORLD);
              }
              test[i * 3] = -1;
              test[i * 3 + 1] = -1;
              test[i * 3 + 2] = -1;
              MPI_Irecv(&test[i * 3], 3, MPI_INT, i, 666, MPI_COMM_WORLD, &request_test[i]);
            } else {
              int id_fail_proc = test[i * 3];
              int id_block_fail = test[i * 3 + 1];
              int id_iterate_fail = test[i * 3 + 2];
              uselles[i] = 1;
              MPI_Request_free(&request_test[i]);
            }
          }
        }
      }
    }
  } else if (ProcRank >= StartNumberReservProc) {
    MPI_Request request_test[2];
    int s;
    int test[3];
    MPI_Irecv(&s, 1, MPI_INT, ReservProcBoss, 999, MPI_COMM_WORLD, &request_test[0]);
    MPI_Irecv(&test, 3, MPI_INT, ReservProcBoss, 777, MPI_COMM_WORLD, &request_test[1]);
    int id_proc = -1;
    MPI_Status status_test;
    MPI_Waitany(2, request_test, &id_proc, &status_test);
    if (status_test.MPI_TAG == 999) {
      MPI_Finalize();
      return 0;
    } else {
      int id_fail_proc = test[0];
      int id_block_fail = test[1];
      id_iterate_failll = test[2];
      ProcRank = id_fail_proc;
      rows = nx / ProcNum + (nx % ProcNum > ProcRank);
      ex_p = (float *) malloc(rows * ny * sizeof(float));
      ey_p = (float *) malloc(rows * ny * sizeof(float));
      hz_p = (float *) malloc(rows * ny * sizeof(float));
      if (id_block_fail == 1 && id_iterate_failll == 0) {
        load_checkpoint(0, 0, rows, *(float (*)[rows][ny])ex_p, *(float (*)[rows][ny])ey_p, *(float (*)[rows][ny])hz_p);
      } else if (id_block_fail == 1) {
        load_checkpoint(1, id_iterate_failll - 1, rows, *(float (*)[rows][ny])ex_p, *(float (*)[rows][ny])ey_p, *(float (*)[rows][ny])hz_p);
      } else if (id_block_fail == 2) {
        load_checkpoint(1, tmax * 3, rows, *(float (*)[rows][ny])ex_p, *(float (*)[rows][ny])ey_p, *(float (*)[rows][ny])hz_p);
      }

      if (id_block_fail == 0) {
        goto block_0;
      } else if (id_block_fail == 1) {
        skip_inizialize = 1;
        goto inizialize;
      } else {
        skip_inizialize = 2;
        goto inizialize;
      }
    }
  }
block_0:
  // Блок 0
  try_suicide(0, 0);
  if (ProcRank < ProcNum) {

    if (ProcRank != 0) {
      ex_p = (float *) malloc(rows * ny * sizeof(float));
      ey_p = (float *) malloc(rows * ny * sizeof(float));
      hz_p = (float *) malloc(rows * ny * sizeof(float));
    }
  }
  if (ProcRank == 0) {
    ex = (float (*)[nx][ny]) malloc((nx) * (ny) * sizeof(float));
    ey = (float (*)[nx][ny]) malloc((nx) * (ny) * sizeof(float));
    hz = (float (*)[nx][ny]) malloc((nx) * (ny) * sizeof(float));

    init_array(*ex, *ey, *hz);

    ex_p = (float *) ex;
    ey_p = (float *) ey;
    hz_p = (float *) hz;

    bench_timer_start();
    int next = rows * ny;
    int warn;
    MPI_Status masStat;
    int check_;
    MPI_Request mpi_request_;
    for (int i = 1; i < ProcNum; ++i) {
block_0_back_1:
      MPI_Isend((float *) ex + next, (nx / ProcNum + (nx % ProcNum > i)) * ny, MPI_FLOAT, numbers_of_correct_proc[i], 1, MPI_COMM_WORLD, &mpi_request_);
      MPI_Isend((float *) ey + next, (nx / ProcNum + (nx % ProcNum > i)) * ny, MPI_FLOAT, numbers_of_correct_proc[i], 2, MPI_COMM_WORLD, &mpi_request_);
      MPI_Isend((float *) hz + next, (nx / ProcNum + (nx % ProcNum > i)) * ny, MPI_FLOAT, numbers_of_correct_proc[i], 3, MPI_COMM_WORLD, &mpi_request_);
      warn = MPI_Recv(&check_, 1, MPI_INT, numbers_of_correct_proc[i], 1, MPI_COMM_WORLD, &masStat);
      if (warn) {
        int test[3];
        test[0] = numbers_of_correct_proc[i];
        test[1] = 0;
        test[2] = 0;
        MPI_Ssend(&test, 3, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD);
        MPI_Recv(&numbers_of_correct_proc[i], 1, MPI_INT, ReservProcBoss, 666, MPI_COMM_WORLD, &masStat);
        goto block_0_back_1;
      }
      next += (nx / ProcNum + (nx % ProcNum > i)) * ny;
    }
  } else {
    int check_;
    MPI_Request mpi_request_;
    MPI_Status masStat[3];
    MPI_Recv(ex_p, rows * ny, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &masStat[0]);
    MPI_Recv(ey_p, rows * ny, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &masStat[1]);
    MPI_Recv(hz_p, rows * ny, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &masStat[2]);
    MPI_Isend(&check_, 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &mpi_request_);
  }
inizialize:
  if (skip_inizialize) {}
  float (*xx)[rows][ny] = (float (*)[rows][ny]) ex_p;
  float (*yy)[rows][ny] = (float (*)[rows][ny]) ey_p;
  float (*hh)[rows][ny] = (float (*)[rows][ny]) hz_p;

  if (skip_inizialize == 2) {
    goto block_2;
  } else if (skip_inizialize == 1) {
    goto block_1;
  }

  if (ProcRank != 0) {
    save_checkpoint(0, 0, rows, *xx, *yy, *hh);
  }

  // Блок 1
block_1:
  kernel_fdtd_2d(rows, *xx, *yy, *hh);

  if (ProcRank != 0) {
    save_checkpoint(1, tmax * 3, rows, *xx, *yy, *hh);
  }

  // Блок 2
block_2:
  if (ProcRank == 0) {
    recieve_results(ex, ey, hz);
  } else {
    try_suicide(2, 0);
    send_complete_result_to_main_proc(rows, xx, yy, hh);
  }

  if (ProcRank == 0) {
    bench_timer_stop();
    bench_timer_print();
    print_array(*ex, *ey, *hz);
    fflush(stderr);

    free((void *) ex);
    free((void *) ey);
    free((void *) hz);
    int s = 1;
    MPI_Ssend(&s, 1, MPI_INT, ReservProcBoss, 555, MPI_COMM_WORLD);
  } else if (ProcRank < ProcNum){
    free((void *) ex_p);
    free((void *) ey_p);
    free((void *) hz_p);
  }
  MPI_Finalize();
  return 0;
}
